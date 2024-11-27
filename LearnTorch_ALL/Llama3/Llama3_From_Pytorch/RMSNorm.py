from input_model import vocab, device
import torch
from torch import nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
# 步骤2: 解码器模块
# 注：由于Llama 3模型由Meta开发，为了与他们的代码库保持一致并考虑未来兼容性，
# 我将使用Meta GitHub上的大部分代码，并进行必要的修改以实现我们的目标。

# 定义参数数据类：我们将在模型构建、训练和推理过程中使用这些参数。
# 注：为了更快地看到训练和推理结果，而不是专注于高准确性，我们对大多数参数采用较低的值，
# 这些值在Llama 3模型中设置得更高。


@dataclass
class ModelArgs:
    dim: int = 512  # 嵌入维度
    n_layers: int = 8  # 模型解码器块的数量
    n_heads: int = 8  # 查询嵌入的头数
    n_kv_heads: int = 4  # 键和值嵌入的头数
    vocab_size: int = len(vocab)  # 词汇表长度
    multiple_of: int = 256  # 用于计算前馈网络维度
    ffn_dim_multiplier: Optional[float] = None  # 用于计算前馈网络维度
    norm_eps: float = 1e-5  # RMSNorm计算的默认Epsilon值
    rope_theta: float = 10000.0  # RePE计算的默认theta值

    max_batch_size: int = 10  # 最大批量大小
    max_seq_len: int = 256  # 最大序列长度

    epochs: int = 2500  # 总训练迭代次数
    log_interval: int = 10  # 打印日志和损失值的间隔数
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据可用性分配设备为cuda或cpu


## 步骤2a: RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        device = ModelArgs.device
        self.eps = eps
        # 缩放参数gamma，初始化为1，参数数量等于dim的大小
        self.weight = nn.Parameter(torch.ones(dim).to(device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(device)

    def forward(self, x):
        # 形状: x[bs,seq,dim]
        output = self._norm(x.float()).type_as(x)

        # 形状: x[bs,seq,dim] -> x_norm[bs,seq,dim]
        return output * self.weight

### RMSNorm代码测试 ###
# 取消下面的三重引号来执行测试

x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)  
rms_norm = RMSNorm(dim=ModelArgs.dim)  
x_norm = rms_norm(x)  

print(f"x的形状: {x.shape}")  
print(f"x_norm的形状: {x_norm.shape}")  

### 测试结果: ###
"""  
x的形状: torch.Size([10, 256, 512])  
x_norm的形状: torch.Size([10, 256, 512])  
"""
