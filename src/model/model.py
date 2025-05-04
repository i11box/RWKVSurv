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
        
        # 输出层 - 用于Cox回归的风险评分
        self.risk_score = nn.Linear(config.embed_dim, 1)
        
        # 初始化权重
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
        
        # 计算损失
        loss = None
        if targets is not None and durations is not None:
            # 创建CoxPH损失函数
            cox_loss = CoxPHLoss()
            # 计算损失
            loss = cox_loss(risk_scores.squeeze(), targets, durations)
        
        return risk_scores, loss

def prepare_data(data, time_steps=4):
    """
    准备模型输入数据
    
    参数:
    - data: 包含静态和动态特征的DataFrame
    - time_steps: 时间步数
    
    返回:
    - static_features: 静态特征 [batch_size, static_dim]
    - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
    - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
    - durations: AKI发生的时间点 [batch_size]
    """
    # 提取静态特征
    static_cols = [col for col in data.columns if '静态特征' in col]
    static_features = torch.tensor(data[static_cols].values, dtype=torch.float32)
    
    # 提取动态特征
    dynamic_features = []
    for t in range(1, time_steps + 1):
        dynamic_cols = [col for col in data.columns if f'_t{t}' in col]
        dynamic_t = data[dynamic_cols].values
        dynamic_features.append(dynamic_t)
    
    dynamic_features = torch.tensor(np.stack(dynamic_features, axis=1), dtype=torch.float32)
    
    # 提取目标值
    aki_times = data['AKI发生时间点'].values
    targets = np.where(aki_times == '未发生', 0, 1)  # 0=未发生, 1=发生
    durations = np.where(aki_times == '未发生', time_steps, aki_times.astype(int))  # 未发生的设为最大时间步
    
    targets = torch.tensor(targets, dtype=torch.float32)
    durations = torch.tensor(durations, dtype=torch.float32)
    
    return static_features, dynamic_features, targets, durations
