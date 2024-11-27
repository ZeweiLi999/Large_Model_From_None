# 旋转位置编码RoPE
import torch
import math
from typing import Tuple, Optional
from input_model import device
from RMSNorm import ModelArgs, x_norm
from torch import nn

## 步骤2b: RoPE实现
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算每对维度的Theta值，即dim/2
    device = ModelArgs.device
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)].float() / dim))

    # 计算序列中位置(m)的范围
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # freqs给出序列中所有标记位置的Theta值范围
    freqs = torch.outer(t, freqs).to(device)

    # 这是需要转换为极坐标形式的旋转矩阵，以便对嵌入执行旋转
    freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "freqs_cis的最后两个维度必须与x匹配"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    device = ModelArgs.device
    # 同时对查询和键嵌入应用旋转位置编码
    # 首先：xq和xk嵌入的最后一个维度需要重塑为一对。因为旋转矩阵应用于每对维度。
    # 其次：将xq和xk转换为复数，因为旋转矩阵只适用于复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)  # xq_:[bsz, seq_len, n_heads, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)  # xk_:[bsz, seq_len, n_heads, head_dim/2]

    # 旋转矩阵(freqs_cis)在seq_len(dim=1)和head_dim(dim=3)维度上应与嵌入匹配
    # 此外，freqs_cis的形状应与xq和xk相同，因此将freqs_cis的形状从[seq_len,head_dim]改变为[1,seq_len,1,head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 最后，通过与freqs_cis相乘执行旋转操作。
    # 旋转完成后，将xq_out和xk_out转换回实数并返回
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device)  # xq_out:[bsz, seq_len, n_heads, head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device)  # xk_out:[bsz, seq_len, n_heads, head_dim]
    return xq_out.type_as(xq), xk_out.type_as(xk)

### RoPE代码测试 ###
# 注：x_norm在RMSNorm测试中计算，这里用于测试。
# 取消下面的三重引号来执行测试

head_dim = ModelArgs.dim//ModelArgs.n_heads
wq = nn.Linear(ModelArgs.dim, ModelArgs.n_heads * head_dim, bias=False, device=device)
wk = nn.Linear(ModelArgs.dim, ModelArgs.n_kv_heads * head_dim, bias=False, device=device)
xq = wq(x_norm)
xk = wk(x_norm)
print(f"xq.shape: {xq.shape}")
print(f"xk.shape: {xk.shape}")

xq = xq.view(xq.shape[0],xq.shape[1],ModelArgs.n_heads, head_dim)
xk = xk.view(xk.shape[0],xk.shape[1],ModelArgs.n_kv_heads, head_dim)
print(f"xq.re-shape: {xq.shape}")
print(f"xk.re-shape: {xk.shape}")

freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=ModelArgs.max_seq_len)  
print(f"freqs_cis.shape: {freqs_cis.shape}")

xq_rotate, xk_rotate = apply_rotary_emb(xq, xk, freqs_cis)
print(f"xq_rotate.shape: {xq_rotate.shape}")
print(f"xk_rotate.shape: {xk_rotate.shape}")

### 测试结果: ###
"""  
xq.shape: torch.Size([10, 256, 512])  
xk.shape: torch.Size([10, 256, 256])  
xq.re-shape: torch.Size([10, 256, 8, 64])  
xk.re-shape: torch.Size([10, 256, 4, 64])  
freqs_cis.shape: torch.Size([256, 32])  
xq_rotate.shape: torch.Size([10, 256, 8, 64])  
xk_rotate.shape: torch.Size([10, 256, 4, 64])  
"""
