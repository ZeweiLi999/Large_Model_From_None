import torch
import math
from typing import Tuple, Optional
from input_model import device
from RMSNorm import ModelArgs, x_norm
from torch import nn
from torch.nn import functional as F
from RoPE import precompute_freqs_cis, apply_rotary_emb, xk

## 注意力模块 [步骤2c: KV缓存; 步骤2d: 分组查询注意力]
## 如前所述，命名约定遵循原始Meta LLama3 GitHub

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # 嵌入维度
        self.dim = args.dim
        # 分配给查询的头数
        self.n_heads = args.n_heads
        # 分配给键和值的头数。如果为"None"，则数量与查询相同。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 每个头相对于模型维度的维度
        self.head_dim = args.dim // args.n_heads
        # 重复次数，以使键、值头数与查询头数匹配
        self.n_rep = args.n_heads // args.n_kv_heads

        # 初始化键、查询、值和输出的权重。注意q和kv的权重out_feature值基于其头数
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=device)

        # 初始化缓存以在开始时存储键、值 (KV缓存实现)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos, inference):
        # 输入嵌入的形状: [bsz,seq_len,dim]
        bsz, seq_len, _ = x.shape
        # 掩码将在"训练"期间使用，由于使用KV缓存，"推理"不需要掩码。
        mask = None

        xq = self.wq(x)  # x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
        xk = self.wk(x)  # x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
        xv = self.wv(x)  # x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

        # 根据头数重塑查询、键和值 (分组查询注意力实现)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)  # xq[bsz,seq_len,n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # xk[bsz,seq_len,n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # xv[bsz,seq_len,n_kv_heads, head_dim]

        # 模型 - 推理模式: kv-cache仅在推理模式下启用
        if inference:
            # 计算序列中每个位置的旋转矩阵
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2)
            # 在推理过程中,我们应该只取从当前标记位置开始的旋转矩阵范围
            freqs_cis = freqs_cis[start_pos: start_pos + seq_len]
            # 将RoPE应用于查询和键嵌入
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # 将键和值标记嵌入存储到它们各自的缓存中 [KV缓存实现]
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

            # 为注意力计算分配所有直到当前标记位置的先前标记嵌入给键和值变量
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]

            # 此时,键和值的形状与查询嵌入不同,但为了计算注意力分数,它们必须相同
            # 使用repeat_kv函数使键、值的形状与查询形状相同
            keys = repeat_kv(keys, self.n_rep)  # keys[bsz,seq_len,n_heads,head_dim]
            values = repeat_kv(values, self.n_rep)  # values[bsz,seq_len,n_heads,head_dim]

            # 模式 - 训练模式: 未实现KV-Cache
        else:
            # 计算旋转矩阵并将RoPE应用于训练的查询和键
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len)

            # xq[bsz,seq_len,n_heads, head_dim], xk[bsz,seq_len,n_heads, head_dim]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # 使用repeat_kv函数使键、值的形状与查询形状相同
            # keys[bsz,seq_len,n_heads,head_dim], #values[bsz,seq_len,n_heads,head_dim]
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)

            # 对于训练模式,我们将计算掩码并稍后应用于注意力分数
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.args.device)
            mask = torch.triu(mask, diagonal=1).to(self.args.device)

            # 为了计算注意力,我们需要执行转置操作来重塑所有查询、键和值,将头部放在维度1,序列放在维度2
            xq = xq.transpose(1, 2)  # xq[bsz,n_heads,seq_len,head_dim]
            keys = keys.transpose(1, 2)  # keys[bsz,n_heads,seq_len,head_dim]
            values = values.transpose(1, 2)  # values[bsz,n_heads,seq_len,head_dim]

            # 计算注意力分数
            scores = torch.matmul(xq, keys.transpose(2, 3)).to(self.args.device) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask

            # 对注意力分数应用softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 注意力分数与值的矩阵乘法
            output = torch.matmul(scores, values).to(self.args.device)

            # 我们得到了每个头部的上下文嵌入
            # 所有头部需要重塑回来并组合,以给出单个上下文注意力输出
            # 形状变化: output[bsz,n_heads,seq_len,head_dim] -> output[bsz,seq_len, n_heads,head_dim] -> output[bsz,seq_len, n_heads * head_dim]
            output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

            # 形状: output [bsz,seq_len,dim]
            return self.wo(output)

# 如果键/值头的数量少于查询头,此函数使用所需的重复次数扩展键/值嵌入
def repeat_kv(x: torch.Tensor, n_rep: int)->torch.Tensor:
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :].expand(bsz, seq_len, n_kv_heads, n_rep, head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim))

### 测试: Repeat_kv函数 ###
# 注: xk, x_norm已在RoPE, RMSNorm测试中计算,这里用于测试
# 取消下面的三重引号来执行测试

n_rep = ModelArgs.n_heads // ModelArgs.n_kv_heads  
keys = repeat_kv(xk, n_rep)  
print(f"xk.shape: {xk.shape}")  
print(f"keys.shape: {keys.shape}")  

## 测试: Attention函数  
# 取消下面的三重引号来执行测试  

attention = Attention(ModelArgs)
x_out = attention(x_norm,start_pos=0, inference=False)  
print(f"x_out.shape: {x_out.shape}")  

### 测试结果: ###
"""  
xk.shape: torch.Size([10, 256, 4, 64])  
keys.shape: torch.Size([10, 256, 8, 64])  
x_out.shape: torch.Size([10, 256, 512])  
"""
