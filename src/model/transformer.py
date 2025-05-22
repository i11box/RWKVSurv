import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        # 确保嵌入维度可以被头数整除
        assert self.n_embd % self.n_head == 0, "嵌入维度必须能被头数整除"
        
        # 查询、键、值的线性投影
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # 输出投影
        self.output = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影并分离头
        # 形状变换: [batch_size, seq_len, n_embd] -> [batch_size, seq_len, n_head, head_size]
        q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_size)
        k = self.key(x).view(batch_size, seq_len, self.n_head, self.head_size)
        v = self.value(x).view(batch_size, seq_len, self.n_head, self.head_size)
        
        # 转置以便进行批量矩阵乘法
        # 形状变换: [batch_size, seq_len, n_head, head_size] -> [batch_size, n_head, seq_len, head_size]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        # [batch_size, n_head, seq_len, head_size] @ [batch_size, n_head, head_size, seq_len]
        # -> [batch_size, n_head, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        # [batch_size, n_head, seq_len, seq_len] @ [batch_size, n_head, seq_len, head_size]
        # -> [batch_size, n_head, seq_len, head_size]
        context = torch.matmul(attn_weights, v)
        
        # 转置回原始形状
        # [batch_size, n_head, seq_len, head_size] -> [batch_size, seq_len, n_head, head_size]
        context = context.transpose(1, 2)
        
        # 合并头
        # [batch_size, seq_len, n_head, head_size] -> [batch_size, seq_len, n_embd]
        context = context.reshape(batch_size, seq_len, self.n_embd)
        
        # 最终线性投影
        output = self.output(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # 层归一化，与原Block类保持一致
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # 多头自注意力，替代RWKV_TimeMix
        self.attn = MultiHeadAttention(config)
        
        # 前馈网络，替代RWKV_ChannelMix
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        # 第一个子层：多头自注意力 + 残差连接
        # 应用层归一化（Pre-LN架构）
        normed_x = self.ln1(x)
        
        # 应用多头自注意力
        attn_output = self.attn(normed_x)
        
        # 残差连接
        x = x + attn_output
        
        # 第二个子层：前馈网络 + 残差连接
        # 应用层归一化（Pre-LN架构）
        normed_x = self.ln2(x)
        
        # 应用前馈网络
        ffn_output = self.ffn(normed_x)
        
        # 残差连接
        x = x + ffn_output
        
        return x