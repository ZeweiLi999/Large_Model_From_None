import torch
from torch import nn
from RMSNorm import ModelArgs, RMSNorm
from KV_Attention import Attention
from SwiGLU import FeedForward
from input_model import device

## 步骤2f: 解码器块。类名为TransformerBlock,以匹配Meta Llama 3代码库

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # 初始化注意力的RMSNorm
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # 初始化注意力类
        self.attention = Attention(args)
        # 初始化前馈网络的RMSNorm
        self.ff_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        # 初始化前馈网络类
        self.feedforward = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)

    def forward(self, x, start_pos, inference):
        # start_pos: 推理模式下的标记位置, inference: True表示推理模式,False表示训练模式
        # 1) 将输入嵌入传递给attention_norm,然后传递给注意力模块
        # 2) 注意力的输出与原始输入(归一化前)相加
        h = x + self.attention(self.attention_norm(x), start_pos, inference)

        # 1) 将注意力输出传递给ff_norm，然后传递给前馈网络
        # 2) 前馈网络的输出与注意力输出(ff_norm前)相加
        out = h + self.feedforward(self.ff_norm(h))
        # 形状: [bsz,seq_len,dim]
        return out

### 测试: TransformerBlock ###
# 取消下面的三重引号来执行测试

# x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)
# transformer_block = TransformerBlock(ModelArgs())
# transformer_block_out = transformer_block(x, start_pos=0, inference=False)
# print(f"transformer_block_out.shape: {transformer_block_out.shape}")


### 测试结果: ###
"""  
transformer_block_out.shape: torch.Size([10, 64, 128])  
"""
