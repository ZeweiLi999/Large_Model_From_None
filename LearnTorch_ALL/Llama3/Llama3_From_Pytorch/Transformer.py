import torch
from torch import nn
from RMSNorm import ModelArgs, RMSNorm
from torch.nn import functional as F
from TransformerBlock import TransformerBlock

## 步骤3: 输出模块
# 这是Llama 3模型。类名保持为Transformer以匹配Meta Llama 3模型

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        # 设置params变量中的所有ModelArgs
        self.params = params
        # 从输入模块初始化嵌入类
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # 初始化解码器块并将其存储在ModuleList中
        # 这是因为我们的Llama 3模型中有4个解码器块 (官方Llama 3有32个块)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(args=params))

        # 为输出模块初始化RMSNorm
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # 在输出模块初始化线性层
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x, start_pos=0, targets=None):

        # start_pos: 推理模式的标记位置, inference: True表示推理模式, False表示训练模式
        # x是使用分词器从文本或提示生成的标记ID批次
        # x[bsz, seq_len] -> h[bsz, seq_len, dim]
        h = self.tok_embeddings(x)

        # 如果目标为None，则激活推理模式并设置为"True"，否则为训练模式"False"
        inference = targets is None

        # 嵌入(h)然后将通过所有解码器块
        for layer in self.layers:
            h = layer(h, start_pos, inference)

            # 最后解码器块的输出将馈入RMSNorm
        h = self.norm(h)

        # 归一化后，嵌入h将馈入线性层
        # 线性层的主要任务是生成将嵌入映射到词汇表大小的logits
        # h[bsz, seq_len, dim] -> logits[bsz, seq_len, vocab_size]
        logits = self.output(h).float()
        loss = None

        # 如果目标不可用，则为推理模式
        if targets is None:
            loss = None
        # 如果目标可用，则为训练模式。计算损失以进行进一步的模型训练
        else:
            loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

        return logits, loss

### 测试: Transformer (Llama模型) ###
# 取消下面的三重引号来执行测试

# model = Transformer(ModelArgs).to(ModelArgs.device)
# print(model)

