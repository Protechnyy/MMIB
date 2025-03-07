import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import pdb
import math


def clones(module, N):
    """创建1个模块的N个副本"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """计算缩放点积注意力计算"""
    d_k = query.size(-1) # 缩放因子
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # 计算分数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # 掩码操作，将被屏蔽位置的注意力分数设置为极小的负数
    p_attn = F.softmax(scores, dim=-1) # 得到注意力权重
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, in_features, d_model, dropout=0.1):
        """多头注意力机制"""
        super(MultiHeadedAttention, self).__init__()
        print("heads: ", h, "d_model: ", d_model)
        assert d_model % h == 0
        # 假定d_v总是等于d_k
        self.d_k = d_model // h # 每个头的维度
        self.h = h # 头的数量
        # 创建三个线性变换层，用于转换query、key和value
        self.linears = clones(nn.Linear(in_features, d_model, bias=None), 3)
        self.last_linear = nn.Linear(d_model, d_model) # 最后一个线性层用于合并多个头的输出
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        nbatches = query.size(0)

        # 1) 对query, key, value进行线性变换
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) 对每个头分别计算注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # pdb.set_trace()
        # 3) 使用view合并多头结果并应用最后一个线性层
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.last_linear(x)


# 跨注意力机制，用于融合文本和图像模态
class CrossAttention(nn.Module):
    def __init__(self, heads, in_size, out_size, dropout):
        super(CrossAttention, self).__init__()

        self.heads = heads
        self.hidden_size = out_size

        self.cross_attn = MultiHeadedAttention(self.heads,  in_size, out_size, dropout) # 由前面定义的MultiHeadedAttention类来实现跨模态的多头注意力计算
        self.ln1 = nn.LayerNorm(self.hidden_size) # 层归一化模块，应用于第一个残差连接后
        self.ln2 = nn.LayerNorm(self.hidden_size) # 层归一化模块，应用于第2个残差连接后
        self.dense = nn.Linear(in_size, self.hidden_size) # 线性变换层，用于将输入转换为隐藏层维度
       

    def forward(self, query, key, value, key_mask):
        
        attn_output = self.cross_attn(query, key, value, key_mask)##
        # P = self.ln1(self.dense(query) + attn_output)
        PP = self.ln1(query + attn_output) # 残差连接+层归一化
        P = self.ln2(PP + self.dense(PP)) # 线性变化+残差连接+归一化
        return P