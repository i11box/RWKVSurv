import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from .block import Block, RWKV_Init, RMSNorm

logger = logging.getLogger(__name__)

class AKIConfig:
    def __init__(self, 
                 static_dim, 
                 dynamic_dim, 
                 embed_dim=128, 
                 n_layer=3, 
                 n_head=4, 
                 n_attn=128, 
                 n_ffn=256, 
                 ctx_len=5, 
                 dropout=0.1, 
                 **kwargs):
        self.static_dim = static_dim      # 静态特征维度
        self.dynamic_dim = dynamic_dim    # 每个时间步的动态特征维度
        self.embed_dim = embed_dim        # 嵌入维度
        self.n_layer = n_layer            # RWKV块的数量
        self.n_head = n_head              # 注意力头数
        self.n_attn = n_attn              # 注意力维度
        self.n_ffn = n_ffn                # 前馈网络维度
        self.ctx_len = ctx_len            # 上下文长度（时间步数）
        self.dropout = dropout            # Dropout率
        self.n_embd = embed_dim           # 兼容RWKV块的参数
        self.model_type = 'RWKV'          # 模型类型
        
        # 添加其他参数
        self.vocab_size = 1               # 在AKI预测任务中使用的词汇表大小
        self.rwkv_emb_scale = 1.0         # RWKV_Init函数的参数
        
        for k, v in kwargs.items():
            setattr(self, k, v)

class AKIPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 静态特征嵌入层
        self.static_embed = nn.Linear(config.static_dim, config.embed_dim)
        
        # 动态特征嵌入层
        self.dynamic_embed = nn.Linear(config.dynamic_dim, config.embed_dim)
        
        # 批量归一化层 - 提高数值稳定性
        self.bn_static = nn.BatchNorm1d(config.embed_dim)
        self.bn_dynamic = nn.BatchNorm1d(config.embed_dim)
        self.bn_combined = nn.BatchNorm1d(config.embed_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(config.dropout)
        
        # RWKV块
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        
        # 层归一化
        self.ln_f = nn.LayerNorm(config.embed_dim)
        
        # 风险评分层
        self.risk_score = nn.Linear(config.embed_dim, 1)
        
        # 时间步预测层
        self.time_predictor = nn.Linear(config.embed_dim, 1)
        
        # 初始化参数
        if self.config.model_type == 'RWKV':
            RWKV_Init(self, config)
        else:
            self.apply(self._init_weights)
            
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 将嵌入矩阵初始化为很小的值
            if 'embed' in module._get_name().lower():
                module.weight.data.normal_(mean=0.0, std=1e-4)
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
            
            if module.bias is not None:
                module.bias.data.zero_()
    
    def configure_optimizers(self, train_config):
        # 分离参数为需要权重衰减和不需要权重衰减的组
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, 
                                  torch.nn.BatchNorm3d, torch.nn.Embedding)
        
        # 记录未分类的参数，用于调试
        unclassified = set()
        
        # 遍历所有参数，确保每个参数只被分类一次
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # 先创建一个完整的参数到模块的映射
        param_to_module = {}
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名
                if fpn in param_dict:  # 确保这是一个直接参数，不是子模块的参数
                    param_to_module[fpn] = m
        
        # 现在对每个参数进行分类
        for pn, p in param_dict.items():
            # 首先检查是否包含'time'关键字
            if 'time' in pn:
                no_decay.add(pn)
                continue  # 跳过这个参数，已经分类完成
            
            # 检查是否以'bias'结尾
            if pn.endswith('bias'):
                no_decay.add(pn)
                continue  # 跳过这个参数，已经分类完成
            
            # 检查是否是权重参数
            if pn.endswith('weight'):
                # 检查模块类型
                if pn in param_to_module:
                    module = param_to_module[pn]
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(pn)
                        continue  # 跳过这个参数，已经分类完成
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(pn)
                        continue  # 跳过这个参数，已经分类完成
                
                # 检查参数名中是否包含'ln'或'norm'
                if 'ln' in pn or 'norm' in pn:
                    no_decay.add(pn)
                    continue  # 跳过这个参数，已经分类完成
            
            # 如果还没有分类，则添加到unclassified
            unclassified.add(pn)
        
        # 将未分类的参数添加到no_decay集合
        if unclassified:
            logger.warning(f"Found {len(unclassified)} unclassified parameters, adding to no_decay set: {unclassified}")
            no_decay.update(unclassified)
        
        # 验证所有参数都被考虑到，并且没有重复
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # 记录参数分布
        logger.info(f"Parameters with weight decay: {len(decay)}")
        logger.info(f"Parameters without weight decay: {len(no_decay)}")
        
        # 检查参数是否重复出现在两个集合中
        if len(inter_params) > 0:
            error_msg = f"Parameters in both decay and no_decay sets: {inter_params}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 检查是否有参数未被分类
        missing_params = param_dict.keys() - union_params
        if len(missing_params) > 0:
            error_msg = f"Parameters not in any set: {missing_params}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 创建参数组
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=train_config.learning_rate, 
            betas=train_config.betas, 
            eps=train_config.eps,
            weight_decay=train_config.weight_decay
        )
        
        # 注意：直接返回优化器对象，而不是字典
        # 学习率调度器和梯度裁剪需要在trainer.py中单独处理
        return optimizer
        
        # 以下代码适用于PyTorch Lightning等框架，但与当前trainer.py不兼容
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.5, 
        #     patience=3, 
        #     verbose=True
        # )
        # 
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss',  # 监控验证集损失
        #         'interval': 'epoch',    # 按epoch更新学习率
        #         'frequency': 1,         # 每个epoch后更新
        #         'reduce_on_plateau': True,
        #     },
        #     'gradient_clip_val': 1.0,   # 梯度裁剪
        # }
    
    def _check_nan_inf(self, tensor, name, raise_error=True):
        """检查张量中是否存在NaN或Inf值，并打印相关信息"""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            stats = f"{name} stats - shape: {tensor.shape}, " \
                   f"mean: {tensor.mean().item() if not torch.isnan(tensor.mean()) else 'NaN'}, " \
                   f"std: {tensor.std().item() if not torch.isnan(tensor.std()) else 'NaN'}, " \
                   f"min: {tensor.min().item() if not torch.isnan(tensor.min()) else 'NaN'}, " \
                   f"max: {tensor.max().item() if not torch.isnan(tensor.max()) else 'NaN'}"
            
            if has_nan and has_inf:
                error_msg = f"检测到NaN和Inf值在 {name}"
            elif has_nan:
                error_msg = f"检测到NaN值在 {name}"
            else:
                error_msg = f"检测到Inf值在 {name}"
                
            error_msg += f"\n{stats}"
            
            if raise_error:
                raise ValueError(error_msg)
            else:
                print(f"警告: {error_msg}")
                return True
        return False

    def forward(self, static_features, dynamic_features, targets=None, durations=None):
        """
        前向传播 - 使用RWKV序列建模能力预测未来AKI风险
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        - durations: AKI发生的时间点 [batch_size]
        
        返回:
        - risk_scores: 每个时间步对未来的AKI风险预测 [batch_size, time_steps, prediction_horizon]
        - time_preds: 每个时间步对AKI发生时间的预测 [batch_size, time_steps, 1]
        - loss: 损失值
        """
        # 检查输入
        self._check_nan_inf(static_features, "static_features")
        self._check_nan_inf(dynamic_features, "dynamic_features")
        
        batch_size, time_steps, _ = dynamic_features.shape
        
        # 预测视野 - 我们希望预测未来多少个时间步内的AKI风险
        prediction_horizon = min(12, time_steps)  # 最多预测未来12个时间步
        
        # 嵌入静态特征
        static_embedded = self.static_embed(static_features)  # [batch_size, embed_dim]
        self._check_nan_inf(static_embedded, "static_embedded (after embedding)")
        
        # 应用批量归一化到静态特征
        static_embedded = self.bn_static(static_embedded)  # [batch_size, embed_dim]
        self._check_nan_inf(static_embedded, "static_embedded (after batch norm)")
        
        # 扩展静态特征到所有时间步
        static_expanded = static_embedded.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, embed_dim]
        self._check_nan_inf(static_expanded, "static_expanded")
        
        # 嵌入动态特征 - 需要重新整形以适应BatchNorm1d
        # 将[batch_size, time_steps, dynamic_dim]重新整形为[batch_size * time_steps, dynamic_dim]
        dynamic_reshaped = dynamic_features.reshape(-1, dynamic_features.size(-1))
        dynamic_embedded = self.dynamic_embed(dynamic_reshaped)  # [batch_size * time_steps, embed_dim]
        self._check_nan_inf(dynamic_embedded, "dynamic_embedded (after embedding)")
        
        # 应用批量归一化
        dynamic_embedded = self.bn_dynamic(dynamic_embedded)  # [batch_size * time_steps, embed_dim]
        self._check_nan_inf(dynamic_embedded, "dynamic_embedded (after batch norm)")
        
        # 重新整形回[batch_size, time_steps, embed_dim]
        dynamic_embedded = dynamic_embedded.reshape(batch_size, time_steps, -1)
        self._check_nan_inf(dynamic_embedded, "dynamic_embedded (reshaped)")
        
        # 合并静态和动态特征
        combined_features = static_expanded + dynamic_embedded  # [batch_size, time_steps, embed_dim]
        self._check_nan_inf(combined_features, "combined_features (after addition)")
        
        # 应用批量归一化到合并特征
        # 需要重新整形以适应BatchNorm1d
        combined_reshaped = combined_features.reshape(-1, combined_features.size(-1))
        combined_normalized = self.bn_combined(combined_reshaped)
        combined_features = combined_normalized.reshape(batch_size, time_steps, -1)
        self._check_nan_inf(combined_features, "combined_features (after batch norm)")
        
        # 应用dropout
        x = self.dropout(combined_features)
        self._check_nan_inf(x, "x (after dropout)")
        
        # 通过RWKV块处理序列
        x = self.blocks(x)  # [batch_size, time_steps, embed_dim]
        self._check_nan_inf(x, "x (after RWKV blocks)")
        
        # 层归一化
        x = self.ln_f(x)  # [batch_size, time_steps, embed_dim]
        self._check_nan_inf(x, "x (after layer norm)")
        
        # 初始化风险预测矩阵 [batch_size, time_steps, prediction_horizon]
        risk_scores = torch.zeros(batch_size, time_steps, prediction_horizon, device=x.device)
        
        # 对每个时间步t，预测未来prediction_horizon个时间步的AKI风险
        for t in range(time_steps):
            # 当前时间步的特征表示
            current_repr = x[:, t, :]  # [batch_size, embed_dim]
            
            # 预测未来时间步的AKI风险
            for h in range(min(prediction_horizon, time_steps - t)):
                future_risk = self.risk_score(current_repr)  # [batch_size, 1]
                risk_scores[:, t, h] = torch.sigmoid(future_risk).squeeze(-1)  # 使用sigmoid确保输出在[0,1]范围内
        
        self._check_nan_inf(risk_scores, "risk_scores (after prediction)")
        
        # 预测AKI发生的时间
        time_logits = self.time_predictor(x)  # [batch_size, time_steps, 1]
        time_preds = torch.sigmoid(time_logits) * time_steps  # 归一化到[0, time_steps]范围
        self._check_nan_inf(time_preds, "time_preds")
        
        # 计算损失
        loss = None
        if targets is not None and durations is not None:
            # 创建目标矩阵 [batch_size, time_steps, prediction_horizon]
            # 对于每个时间步t，如果在未来h个时间步内发生AKI，则target_matrix[b, t, h] = 1
            target_matrix = torch.zeros_like(risk_scores)
            
            for b in range(batch_size):
                if targets[b] == 1:  # 如果该样本发生了AKI
                    aki_time = int(durations[b].item())  # AKI发生的时间步
                    
                    # 对于每个时间步t
                    for t in range(time_steps):
                        # 如果t < aki_time，计算未来多少个时间步内会发生AKI
                        if t < aki_time:
                            # 对于未来的每个预测时间步h
                            for h in range(min(prediction_horizon, time_steps - t)):
                                # 如果t+h >= aki_time，说明在t时刻预测未来h个时间步会发生AKI
                                if t + h >= aki_time:
                                    target_matrix[b, t, h] = 1
            
            # 检查并处理NaN/Inf值
            if self._check_nan_inf(risk_scores, "risk_scores (before loss)", raise_error=False):
                print("警告: 风险评分中存在NaN或Inf值，尝试修复...")
                risk_scores = torch.nan_to_num(risk_scores, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 计算二元交叉熵损失
            bce_loss = F.binary_cross_entropy(risk_scores.view(-1), target_matrix.view(-1))
            self._check_nan_inf(bce_loss, "bce_loss")
            
            # 计算时间预测损失（仅对发生AKI的样本）
            time_loss = 0.0
            aki_mask = targets == 1  # 只考虑实际发生AKI的样本
            if aki_mask.sum() > 0:  # 如果有实际发生AKI的样本
                # 取每个样本最后一个有效时间步的预测
                pred_times = time_preds[aki_mask, -1, 0]
                actual_times = durations[aki_mask].float()
                
                # 计算MSE损失
                time_loss = F.mse_loss(pred_times, actual_times)
                self._check_nan_inf(time_loss, "time_loss")
            
            # 组合损失
            loss = bce_loss + 0.1 * time_loss
            self._check_nan_inf(loss, "total_loss")
        
        # 返回完整的风险评分矩阵和时间预测
        return risk_scores, time_preds, loss

def prepare_data(data, time_steps=None):
    """
    准备模型输入数据 - 简化版本，适用于预处理后的数据
    
    参数:
    - data: 包含预处理后的静态和动态特征的DataFrame
    - time_steps: 使用的时间步数（如果为None，则使用数据中的所有时间步）
    
    返回:
    - static_features: 静态特征 [batch_size, static_dim]
    - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
    - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
    - durations: AKI发生的时间点 [batch_size]
    """
    # 1. 提取静态特征（所有不包含_t的列，除了subject_id, aki_time和aki_status）
    static_cols = [col for col in data.columns if ('_t' not in col) and (col not in ['subject_id', 'aki_time', 'aki_status'])]
    static_features = torch.tensor(data[static_cols].values, dtype=torch.float32)
    
    # 2. 提取动态特征（所有包含_t的列，但排除aki_time）
    dynamic_cols = [col for col in data.columns if ('_t' in col) and (col != 'aki_time')]
    
    # 确保所有动态特征列都符合预期的格式
    valid_dynamic_cols = []
    for col in dynamic_cols:
        try:
            # 尝试提取时间步编号
            t = col.split('_t')[1].split('_')[0] if '_' in col.split('_t')[1] else col.split('_t')[1]
            int(t)  # 验证能否转换为整数
            valid_dynamic_cols.append(col)
        except (IndexError, ValueError):
            # 跳过不符合格式的列
            continue
    
    dynamic_cols = valid_dynamic_cols
    
    # 按时间步排序动态特征列
    dynamic_cols.sort(key=lambda x: int(x.split('_t')[1].split('_')[0]) if '_' in x.split('_t')[1] else int(x.split('_t')[1]))
    
    # 确定时间步数和每个时间步的特征数
    if not dynamic_cols:
        raise ValueError("未找到动态特征列（包含_t的列）")
    
    # 确定时间步数
    time_points = set()
    for col in dynamic_cols:
        # 所有列都已经验证过格式，可以安全地提取时间步
        t = col.split('_t')[1].split('_')[0] if '_' in col.split('_t')[1] else col.split('_t')[1]
        time_points.add(int(t))
    
    max_time_step = max(time_points)
    if time_steps is None:
        time_steps = max_time_step
    else:
        time_steps = min(time_steps, max_time_step)
    
    # 按时间步重组动态特征
    dynamic_features_list = []
    for t in range(1, time_steps + 1):
        # 获取当前时间步的所有特征列
        # 根据列名格式，应该是以t{t}_开头或以_t{t}结尾
        t_cols = [col for col in dynamic_cols if col.startswith(f"t{t}_") or col.endswith(f"_t{t}")]
        
        if not t_cols:
            print(f"警告：时间步 {t} 的特征不存在，请检查数据格式")
            # 如果没有找到列，尝试更宽松的匹配
            t_cols = [col for col in dynamic_cols if f"_t{t}" in col]
            if not t_cols:
                print(f"错误：无法找到时间步 {t} 的任何特征列，跳过此时间步")
                continue
            else:
                print(f"找到 {len(t_cols)} 个包含 t{t} 的列：{t_cols}")
        
        # 提取当前时间步的特征值
        t_features = data[t_cols].values
        
        # 检查数据是否包含NaN或Inf
        if np.isnan(t_features).any() or np.isinf(t_features).any():
            print(f"警告：时间步 {t} 的特征包含NaN或Inf值，尝试修复")
            # 替换NaN和Inf值
            t_features = np.nan_to_num(t_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        dynamic_features_list.append(t_features)
    
    # 确保至少有一个时间步的特征
    if not dynamic_features_list:
        raise ValueError("没有找到任何有效的时间步特征，请检查数据格式")
    
    # 转换为张量
    try:
        dynamic_features = torch.tensor(np.stack(dynamic_features_list, axis=1), dtype=torch.float32)
        # 验证数据是否有效
        if torch.isnan(dynamic_features).any() or torch.isinf(dynamic_features).any():
            print("警告：动态特征张量包含NaN或Inf值，尝试修复")
            dynamic_features = torch.nan_to_num(dynamic_features, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"错误：创建动态特征张量失败 - {str(e)}")
        print(f"dynamic_features_list长度：{len(dynamic_features_list)}")
        if dynamic_features_list:
            print(f"第一个元素形状：{dynamic_features_list[0].shape}")
        raise
    
    # 3. 提取目标值
    aki_times = data['aki_time'].values
    targets = np.where(aki_times == -1, 0, 1)  # -1=未发生，其他值=发生
    
    # 4. 处理持续时间：未发生的设为最大时间步，其他值保持不变
    durations = np.where(aki_times == -1, time_steps, aki_times)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    durations = torch.tensor(durations, dtype=torch.float32)
    
    return static_features, dynamic_features, targets, durations
