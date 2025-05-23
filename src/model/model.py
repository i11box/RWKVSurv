import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from .block import Block, RWKV_Init, RMSNorm
from .lstm import LSTMBlock
from .gru import GRUBlock
from .transformer import TransformerBlock
import math

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
                 h=6,                    # 提前预测步数，默认为6
                 model_type='RWKV',       # 模型类型: 'RWKV', 'LSTM', 'GRU', 'Transformer'
                 
                 # LSTM特定参数
                 lstm_layers=1,           # LSTM层数
                 lstm_bidirectional=False, # 是否使用双向LSTM
                 
                 # GRU特定参数
                 gru_layers=1,            # GRU层数
                 gru_bidirectional=False,  # 是否使用双向GRU
                 
                 # Transformer特定参数
                 attn_dropout=0.1,        # 注意力机制的dropout率
                 ff_activation='gelu',    # 前馈网络的激活函数类型
                 
                 **kwargs):
        # 基本参数
        self.static_dim = static_dim      # 静态特征维度
        self.dynamic_dim = dynamic_dim    # 每个时间步的动态特征维度
        self.embed_dim = embed_dim        # 嵌入维度
        self.n_layer = n_layer            # 网络块的数量
        self.n_head = n_head              # 注意力头数
        self.n_attn = n_attn              # 注意力维度
        self.n_ffn = n_ffn                # 前馈网络维度
        self.ctx_len = ctx_len            # 上下文长度（时间步数）
        self.dropout = dropout            # Dropout率
        self.n_embd = embed_dim           # 兼容各类网络块的参数
        self.model_type = model_type      # 模型类型: 'RWKV', 'LSTM', 'GRU', 'Transformer'
        self.h = h                        # 提前预测步数，小于这个时间步发生的数据先筛除
        
        # LSTM特定参数
        self.lstm_layers = lstm_layers
        self.lstm_bidirectional = lstm_bidirectional
        
        # GRU特定参数
        self.gru_layers = gru_layers
        self.gru_bidirectional = gru_bidirectional
        
        # Transformer特定参数
        self.attn_dropout = attn_dropout
        self.ff_activation = ff_activation
        
        # RWKV特定参数
        self.vocab_size = 1               # 在AKI预测任务中使用的词汇表大小
        self.rwkv_emb_scale = 1.0         # RWKV_Init函数的参数
        
        # 处理其他参数
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # 根据模型类型设置日志
        logger.info(f"初始化AKI配置: 模型类型={self.model_type}, 层数={self.n_layer}, 嵌入维度={self.embed_dim}")

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
        
        # 位置编码层（正余弦编码）
        self.register_buffer(
            'position_encoding',
            self._create_position_encoding(config.ctx_len, config.embed_dim)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(config.dropout)
        
        # 根据模型类型选择不同的网络块
        if config.model_type == 'RWKV':
            logger.info(f"使用RWKV块作为网络块")
            self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        elif config.model_type == 'LSTM':
            logger.info(f"使用LSTM块作为网络块: 层数={config.lstm_layers}, 双向={config.lstm_bidirectional}")
            self.blocks = nn.Sequential(*[LSTMBlock(config, i) for i in range(config.n_layer)])
        elif config.model_type == 'GRU':
            logger.info(f"使用GRU块作为网络块: 层数={config.gru_layers}, 双向={config.gru_bidirectional}")
            self.blocks = nn.Sequential(*[GRUBlock(config, i) for i in range(config.n_layer)])
        elif config.model_type == 'Transformer':
            logger.info(f"使用Transformer块作为网络块: 注意力头数={config.n_head}, 激活函数={config.ff_activation}")
            self.blocks = nn.Sequential(*[TransformerBlock(config, i) for i in range(config.n_layer)])
        else:
            raise ValueError(f"不支持的模型类型: {config.model_type}")
        
        # 层归一化
        self.ln_f = nn.LayerNorm(config.embed_dim)
        
        # AKI预测层 - 直接输出是否会在几步内发生AKI
        self.aki_predictor = nn.Linear(config.embed_dim, 1)
        
        # 初始化参数
        if self.config.model_type == 'RWKV':
            RWKV_Init(self, config)
        else:
            self.apply(self._init_weights)
            
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    
    def _create_position_encoding(self, max_len, d_model):
        """创建正余弦位置编码
        
        参数:
        - max_len: 最大序列长度
        - d_model: 模型维度
        
        返回:
        - 位置编码矩阵 [max_len, d_model]
        """
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
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

    def forward(self, static_features, dynamic_features, targets=None, durations=None, is_training=True):
        """
        前向传播 - 使用RWKV序列建模能力预测未来AKI风险
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        - durations: AKI发生的时间点 [batch_size]
        - is_training: 是否处于训练模式
        
        返回:
        - aki_probs: 每个样本在提前预测h步内发生AKI的概率 [batch_size, 1]
        - loss: 损失值
        """
        # 检查输入
        self._check_nan_inf(static_features, "static_features")
        self._check_nan_inf(dynamic_features, "dynamic_features")
        
        batch_size, time_steps, _ = dynamic_features.shape
        h = self.config.h  # 提前预测步数
        
        # 处理训练数据，根据是否发生AKI采用不同的特征选择策略
        if is_training and targets is not None and durations is not None:
            # 创建新的特征张量和目标
            selected_features = []
            selected_targets = []
            
            for b in range(batch_size):
                if targets[b] == 1:  # 如果该样本发生了AKI
                    aki_time = int(durations[b].item())  # AKI发生的时间步
                    
                    # 如果AKI发生时间小于h，则跳过该样本
                    if aki_time < h:
                        continue
                    
                    # 使用t-h步前的特征预测
                    t = aki_time - h
                    # 选择静态特征和动态特征的t时刻之前的数据
                    selected_static = static_features[b].unsqueeze(0)  # [1, static_dim]
                    selected_dynamic = dynamic_features[b, :t+1].unsqueeze(0)  # [1, t+1, dynamic_dim]
                    
                    # 添加到选定特征列表
                    selected_features.append((selected_static, selected_dynamic))
                    selected_targets.append(1)  # 该样本在h步内发生AKI
                else:  # 如果该样本未发生AKI
                    # 随机选择(t,T)之间的时间步
                    max_t = time_steps - h  # 确保有足够的后续时间步
                    if max_t <= 0:
                        continue  # 如果没有足够的时间步，跳过该样本
                    
                    t = torch.randint(0, max_t, (1,)).item()
                    # 选择静态特征和动态特征的t时刻之前的数据
                    selected_static = static_features[b].unsqueeze(0)  # [1, static_dim]
                    selected_dynamic = dynamic_features[b, :t+1].unsqueeze(0)  # [1, t+1, dynamic_dim]
                    
                    # 添加到选定特征列表
                    selected_features.append((selected_static, selected_dynamic))
                    selected_targets.append(0)  # 该样本在h步内不会发生AKI
            
            # 如果没有有效样本，返回默认值
            if len(selected_features) == 0:
                return torch.zeros(batch_size, 1, device=static_features.device), None
            
            # 处理选定的特征
            new_batch_size = len(selected_features)
            max_seq_len = max([feat[1].shape[1] for feat in selected_features])
            
            # 创建新的静态和动态特征张量
            new_static_features = torch.cat([feat[0] for feat in selected_features], dim=0)
            new_dynamic_features = torch.zeros(new_batch_size, max_seq_len, dynamic_features.shape[2], device=dynamic_features.device)
            
            # 填充动态特征
            for i, (_, dyn_feat) in enumerate(selected_features):
                seq_len = dyn_feat.shape[1]
                new_dynamic_features[i, :seq_len] = dyn_feat
            
            # 更新批量大小和时间步
            batch_size = new_batch_size
            time_steps = max_seq_len
            static_features = new_static_features
            dynamic_features = new_dynamic_features
            targets = torch.tensor(selected_targets, dtype=torch.float32, device=static_features.device)
        
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
        
        # 添加位置编码
        position_encoding = self.position_encoding[:time_steps].unsqueeze(0).expand(batch_size, -1, -1)
        
        # 合并静态和动态特征，并添加位置编码
        combined_features = static_expanded + dynamic_embedded + position_encoding  # [batch_size, time_steps, embed_dim]
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
        
        # 使用序列的最后一个时间步的表示进行预测
        final_repr = x[:, -1, :]  # [batch_size, embed_dim]
        
        # 预测未来h步内是否会发生AKI
        aki_logits = self.aki_predictor(final_repr)  # [batch_size, 1]
        aki_probs = torch.sigmoid(aki_logits)  # 使用sigmoid确保输出在[0,1]范围内
        self._check_nan_inf(aki_probs, "aki_probs")
        
        # 计算损失
        loss = None
        if targets is not None:
            # 计算二元交叉熏损失
            loss = F.binary_cross_entropy(aki_probs.squeeze(-1), targets)
            self._check_nan_inf(loss, "loss")
        
        # 返回AKI发生概率和损失
        return aki_probs, loss

def prepare_data(data, time_steps=None, h=6):
    """
    准备模型输入数据 - 简化版本，适用于预处理后的数据
    
    参数:
    - data: 包含预处理后的静态和动态特征的DataFrame
    - time_steps: 使用的时间步数（如果为None，则使用数据中的所有时间步）
    - h: 提前预测步数，小于这个时间步发生的数据先筛除
    
    返回:
    - static_features: 静态特征 [batch_size, static_dim]
    - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
    - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
    - durations: AKI发生的时间点 [batch_size]
    """
    # 1. 提取静态特征（所有不包含_t的列，除了subject_id, aki_time和aki_status）
    static_cols = [col for col in data.columns if ('_t' not in col) and (col not in ['subject_id', 'aki_time', 'aki_status'])]
    
    # 打印静态特征列名，以便于对应static_1, static_2, ...
    print("静态特征列名（对应static_1, static_2, ...）:")
    for i, col in enumerate(static_cols):
        print(f"static_{i+1}: {col}")
    
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
    
    # 5. 筛选小于h时间步发生的AKI数据
    if h > 0:
        # 创建掩码，标识需要保留的样本
        # 保留条件：1) 未发生AKI的样本 或 2) AKI发生时间 >= h的样本
        valid_mask = (aki_times == -1) | (aki_times >= h)
        print(f"筛选前样本数: {len(targets)}, 筛选后样本数: {np.sum(valid_mask)}")
        print(f"筛除了 {len(targets) - np.sum(valid_mask)} 个小于{h}时间步发生AKI的样本")
        
        # 应用掩码筛选数据
        if np.sum(valid_mask) < len(targets):
            static_features = static_features[valid_mask]
            dynamic_features = dynamic_features[valid_mask]
            targets = np.array(targets)[valid_mask]
            durations = np.array(durations)[valid_mask]
    
    targets = torch.tensor(targets, dtype=torch.float32)
    durations = torch.tensor(durations, dtype=torch.float32)
    
    return static_features, dynamic_features, targets, durations
