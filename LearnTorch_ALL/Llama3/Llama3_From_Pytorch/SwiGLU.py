from torch import nn
from torch.nn import functional as F
from input_model import device
from RMSNorm import ModelArgs, rms_norm
from KV_Attention import x_out
from typing import Optional

## 步骤2e: 前馈网络 (SwiGLU激活)
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        # 模型嵌入维度
        self.dim = dim

        # 我们必须使用Meta提供的隐藏维度计算方法,这是该模型的理想设置
        # 隐藏维度的计算方式使其是256的倍数
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义隐藏层权重
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

    def forward(self, x):
        # 形状: [bsz,seq_len,dim]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

### 测试: 前馈模块 ###
# 注: x_out已在Attention测试中计算,这里用于测试
# 取消下面的三重引号来执行测试

feed_forward = FeedForward(ModelArgs.dim, 4 * ModelArgs.dim, ModelArgs.multiple_of, ModelArgs.ffn_dim_multiplier)  
x_out = rms_norm(x_out)  
x_out = feed_forward(x_out)  
print(f"前馈输出: x_out.shape: {x_out.shape}")  


### 测试结果: ###
"""  
前馈输出: x_out.shape: torch.Size([10, 256, 512])  
"""
