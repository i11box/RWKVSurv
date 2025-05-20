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
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def configure_optimizers(self, train_config):
        # 将参数分为需要和不需要权重衰减的两组
        decay = set()
        no_decay = set()
        
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (RMSNorm, nn.LayerNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # 完整参数名
                
                if pn.endswith('bias') or ('time' in fpn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # 验证所有参数都被考虑到
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer
    
    def forward(self, static_features, dynamic_features, targets=None, durations=None):
        """
        前向传播
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        - durations: AKI发生的时间点 [batch_size]
        
        返回:
        - risk_scores: 风险评分 [batch_size, 1]
        - loss: 损失值
        """
        batch_size, time_steps, _ = dynamic_features.shape
        
        # 嵌入静态特征
        static_embedded = self.static_embed(static_features)  # [batch_size, embed_dim]
        
        # 扩展静态特征到所有时间步
        static_expanded = static_embedded.unsqueeze(1).expand(-1, time_steps, -1)  # [batch_size, time_steps, embed_dim]
        
        # 嵌入动态特征
        dynamic_embedded = self.dynamic_embed(dynamic_features)  # [batch_size, time_steps, embed_dim]
        
        # t0步是纯静态特征，后续时间步是静态特征+动态特征
        x = torch.zeros_like(static_expanded)
        x[:, 0, :] = static_embedded  # t0步是纯静态特征
        x[:, 1:, :] = static_expanded[:, 1:, :] + dynamic_embedded[:, :-1, :]  # 后续时间步是静态特征+动态特征
        
        # 应用dropout
        x = self.dropout(x)
        
        # 通过RWKV块
        x = self.blocks(x)
        
        # 层归一化
        x = self.ln_f(x)
        
        # 使用最后一个时间步的输出计算风险评分
        risk_scores = self.risk_score(x[:, -1, :])
        
        # 预测AKI发生的时间步
        # 使用sigmoid函数将输出映射到[0,1]区间，然后乘以时间步数
        time_pred_normalized = torch.sigmoid(self.time_predictor(x[:, -1, :]))
        time_pred = time_pred_normalized * time_steps  # 乘以最大时间步
        
        # 对于风险评分低于阈值的样本，将时间步设为-1（表示不发生AKI）
        threshold = 0.5  # 可以设为超参数
        time_pred = torch.where(risk_scores.squeeze() < threshold, 
                           torch.tensor(-1.0, device=time_pred.device), 
                           time_pred)
        
        # 计算损失
        loss = None
        if targets is not None and durations is not None:
            # 创建CoxPH损失函数
            cox_loss = CoxPHLoss()
            # 计算风险评分的损失
            risk_loss = cox_loss(risk_scores.squeeze(), targets, durations)
            
            # 暂时禁用时间步预测损失，只使用风险评分损失
            # 这是一个临时解决方案，直到我们完全理解形状不匹配的原因
            loss = risk_loss
            
            # 下面是注释掉的原始代码，以便将来可能的恢复
            # mask = targets == 1  # 只考虑实际发生AKI的样本
            # if mask.sum() > 0:  # 如果有实际发生AKI的样本
            #     # 确保两个张量具有相同的形状
            #     pred_times = time_pred[mask].view(-1)  # 展平为一维张量
            #     actual_times = durations[mask].float().view(-1)  # 展平为一维张量
            #     
            #     # 检查形状是否匹配
            #     if pred_times.shape == actual_times.shape:
            #         time_loss = F.mse_loss(pred_times, actual_times)
            #         # 组合两种损失，可以加权
            #         loss = risk_loss + time_loss
            #     else:
            #         print(f"警告：形状不匹配 - pred_times: {pred_times.shape}, actual_times: {actual_times.shape}")
            #         # 如果形状不匹配，只使用风险评分损失
            #         loss = risk_loss
            # else:
            #     loss = risk_loss
        
        return risk_scores, time_pred, loss

def prepare_data(data, time_steps=48):
    """
    准备模型输入数据
    
    参数:
    - data: 包含静态和动态特征的DataFrame
    - time_steps: 使用的时间步数（默认为48，可以指定更小的值使用前N个时间步）
    
    返回:
    - static_features: 静态特征 [batch_size, static_dim]
    - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
    - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
    - durations: AKI发生的时间点 [batch_size]
    """
    # 提取静态特征（gender和age）
    # gender通常需要进行独热编码，这里简单处理为数值
    static_cols = ['gender', 'age']
    data['gender'] = data['gender'].replace({'M': 1, 'F': 0})
    
    if 'age' in data.columns:
        age_mean = data['age'].mean()
        age_std = data['age'].std()
        data['age'] = (data['age'] - age_mean) / (age_std + 1e-8)
    
    static_features = torch.tensor(data[static_cols].values, dtype=torch.float32)
    
    # 提取动态特征（肌酐和尿量）
    dynamic_features = []
    for t in range(1, time_steps + 1):
        # 对于每个时间步，提取肌酐和尿量特征
        creatinine_col = f'creatinine_t{t}'
        urine_output_col = f'urine_output_t{t}'
        
        # 确保列存在
        if creatinine_col in data.columns and urine_output_col in data.columns:
            # 对特征进行预处理，处理缺失值和归一化
            creatinine = data[creatinine_col].fillna(0).values
            urine_output = data[urine_output_col].fillna(0).values
            
            # 避免除零错误
            creatinine_mean = creatinine.mean()
            creatinine_std = creatinine.std() + 1e-8
            urine_mean = urine_output.mean()
            urine_std = urine_output.std() + 1e-8
            
            # 归一化
            creatinine = (creatinine - creatinine_mean) / creatinine_std
            urine_output = (urine_output - urine_mean) / urine_std
            
            dynamic_t = np.column_stack([creatinine, urine_output])
            dynamic_features.append(dynamic_t)
        else:
            # 如果列不存在，可能是因为time_steps设置过大
            print(f"警告：时间步 {t} 的特征不存在，请检查数据或减小time_steps参数")
            break
    
    # 实际使用的时间步数可能小于指定的time_steps
    actual_time_steps = len(dynamic_features)
    if actual_time_steps < time_steps:
        print(f"警告：只找到了 {actual_time_steps} 个时间步的数据，而不是指定的 {time_steps} 个")
        time_steps = actual_time_steps
    
    dynamic_features = torch.tensor(np.stack(dynamic_features, axis=1), dtype=torch.float32)
    
    # 提取目标值
    aki_times = data['aki_time'].values
    targets = np.where(aki_times == -1, 0, 1)  # -1=未发生，其他值=发生
    
    # 处理持续时间：未发生的设为最大时间步，其他值保持不变
    durations = np.where(aki_times == -1, time_steps, aki_times)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    durations = torch.tensor(durations, dtype=torch.float32)
    
    return static_features, dynamic_features, targets, durations
