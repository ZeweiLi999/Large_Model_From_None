import torch
from RMSNorm import ModelArgs
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from input_model import encode, data, token_bos, token_eos
from Transformer import Transformer

## 步骤4: 训练Llama 3模型:

# 使用我们在输入模块部分构建的分词器的encode函数，通过对整个tiny_shakespeare数据进行编码来创建数据集
dataset = torch.tensor(encode(data), dtype=torch.int).to(ModelArgs.device)
print(f"dataset-shape: {dataset.shape}")


# 定义函数从给定数据集生成批次
def get_dataset_batch(data, split, args: ModelArgs):
    seq_len = args.max_seq_len
    batch_size = args.max_batch_size
    device = args.device

    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    batch_data = train
    if split == "val":
        batch_data = val
    elif split == "test":
        batch_data = test

    # 从数据集中选择随机起点，为训练、验证和测试提供随机样本
    ix = torch.randint(0, len(batch_data) - seq_len - 3, (batch_size,)).to(device)
    x = torch.stack([torch.cat([token_bos, batch_data[i:i + seq_len - 1]]) for i in ix]).long().to(device)
    y = torch.stack([torch.cat([batch_data[i + 1:i + seq_len], token_eos]) for i in ix]).long().to(device)

    return x, y


# 定义evaluate_loss函数来计算和存储训练和验证损失，用于日志记录和绘图
@torch.no_grad()
def evaluate_loss(model, args: ModelArgs):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_dataset_batch(dataset, split, args)
            _, loss = model(x=xb, targets=yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)

    model.train()
    return out


# 定义训练函数来执行模型训练
def train(model, optimizer, args: ModelArgs):
    epochs = args.epochs
    log_interval = args.log_interval
    device = args.device
    losses = []  # 用于记录每个epoch的训练和验证损失
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        xs, ys = get_dataset_batch(dataset, 'train', args)
        xs = xs.to(device)
        ys = ys.to(device)
        logits, loss = model(x=xs, targets=ys)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, args)
            losses.append(x)
            print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f}")
            start_time = time.time()

    # 打印最终验证损失
    print("验证损失: ", losses[-1]['val'])

    # # 将损失数据转化为DataFrame，确保列名正确
    # losses_df = pd.DataFrame(losses)
    # print("最终的损失 DataFrame:")s
    # print(losses_df)
    #
    # # 绘制损失图像
    # plt.figure(figsize=(10, 6))  # 设置图像大小
    # plt.plot(losses_df['train'], label='Train Loss', color='blue')
    # plt.plot(losses_df['val'], label='Validation Loss', color='red')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.grid(True)
    #
    # # 显示图像
    # plt.show()

    return losses_df  # 返回 DataFrame，供后续使用


## 开始训练我们的Llama 3模型

model = Transformer(ModelArgs()).to(ModelArgs.device)
optimizer = torch.optim.Adam(model.parameters())

# 训练模型并返回损失的 DataFrame
losses_df = train(model, optimizer, ModelArgs())
