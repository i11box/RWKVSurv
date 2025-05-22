import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # 层归一化，与原Block类保持一致
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # GRU层，替代RWKV_TimeMix
        self.gru = nn.GRU(
            input_size=config.n_embd,
            hidden_size=config.n_embd,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=config.gru_bidirectional
        )
        
        # 如果是双向GRU，需要将输出维度调整为n_embd
        if config.gru_bidirectional:
            self.direction_proj = nn.Linear(config.n_embd * 2, config.n_embd)
        
        # 前馈网络，替代RWKV_ChannelMix
        hidden_size = 4 * config.n_embd  # 类似于原始实现中的hidden_sz
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(hidden_size, config.n_embd)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
    
    def forward(self, x):
        # 保存原始输入用于残差连接
        identity = x
        
        # 第一个子层：GRU + 残差连接
        # 应用层归一化
        normed_x = self.ln1(x)
        
        # 应用GRU层
        gru_out, _ = self.gru(normed_x)
        
        # 如果是双向GRU，需要将输出维度调整为n_embd
        if self.config.gru_bidirectional:
            gru_out = self.direction_proj(gru_out)
        
        # 应用dropout
        gru_out = self.dropout(gru_out)
        
        # 残差连接
        x = identity + gru_out
        
        # 第二个子层：前馈网络 + 残差连接
        # 保存当前状态用于残差连接
        identity = x
        
        # 应用层归一化
        normed_x = self.ln2(x)
        
        # 应用前馈网络
        ffn_out = self.ffn(normed_x)
        
        # 应用dropout
        ffn_out = self.dropout(ffn_out)
        
        # 残差连接
        x = identity + ffn_out
        
        return x