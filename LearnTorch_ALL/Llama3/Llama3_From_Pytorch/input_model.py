# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
from matplotlib import pyplot as plt

### 步骤1: 输入模块 ###

# 使用Tiny Shakespeare数据集实现字符级分词器。部分字符级分词器代码参考自Andrej Karpathy的GitHub仓库
# (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py)
# 加载tiny_shakespeare数据文件 (https://github.com/tamangmilan/llama3/blob/main/tiny_shakespeare.txt)

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据可用性分配设备为cuda或cpu

# 加载tiny_shakespeare数据文件
with open('tiny_shakespeare.txt', 'r') as f:
    data = f.read()

# 通过提取tiny_shakespeare数据中的所有唯一字符准备词汇表
vocab = sorted(list(set(data)))

# 训练Llama 3模型需要额外的标记，如<|begin_of_text|>、<|end_of_text|>和<|pad_id|>，将它们添加到词汇表中
vocab.extend(['<|begin_of_text|>', '<|end_of_text|>', '<|pad_id|>'])
vocab_size = len(vocab)

# 创建字符与词汇表中对应整数索引之间的映射。
# 这对于构建分词器的编码和解码函数至关重要。
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

# 分词器编码函数：输入字符串，输出整数列表
def encode(s):
    return [stoi[ch] for ch in s]

# 分词器解码函数：输入整数列表，输出字符串
def decode(l):
    return ''.join(itos[i] for i in l)

# 定义稍后在模型训练中使用的张量标记变量
token_bos = torch.tensor([stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
token_eos = torch.tensor([stoi['<|end_of_text|>']], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi['<|pad_id|>']], dtype=torch.int, device=device)

prompts = "Hello World"
encoded_tokens = encode(prompts)
decoded_text = decode(encoded_tokens)

### 输入模块代码测试 ###
# 取消下面的三重引号来执行测试
"""  
print(f"Shakespeare文本字符长度: {len(data)}")
print(f"词汇表内容: {''.join(vocab)}\n")
print(f"词汇表大小: {vocab_size}")
print(f"编码后的标记: {encoded_tokens}")
print(f"解码后的文本: {decoded_text}")
"""
### 测试结果: ###
"""  
Shakespeare文本字符长度: 1115394  
词汇表内容:   
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<|begin_of_text|><|end_of_text|><|pad_id|>  

词汇表大小: 68  
编码后的标记: [20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42]  
解码后的文本: Hello World  
"""
