import torch
from RMSNorm import ModelArgs
from input_model import token_bos, token_eos, token_pad, encode, decode
from Llama3_train import model

## 步骤5: Llama 3模型推理
# 这个函数使用我们构建和训练的Llama 3模型，基于提供的提示生成文本序列

def generate(model, prompts: str, params: ModelArgs, max_gen_len: int = 500, temperature: float = 0.6, top_p: float = 0.9):

    # prompt_tokens: 用户输入文本或提示列表
    # max_gen_len: 生成文本序列的最大长度
    # temperature: 用于控制采样随机性的温度值。默认为0.6
    # top_p: 从logits采样prob输出的top-p概率阈值。默认为0.9
    bsz = 1  # 对于推理，通常用户只输入一个提示，我们将其作为1个批次
    prompt_tokens = token_bos.tolist() + encode(prompts)
    assert len(prompt_tokens) <= params.max_seq_len, "提示标记长度应小于max_seq_len"
    total_len = min(len(prompt_tokens) + max_gen_len, params.max_seq_len)

    # 这个tokens矩阵用于存储输入提示和模型生成的所有输出
    # 稍后我们将使用分词器的decode函数来解码这个token，以文本格式查看结果
    tokens = torch.full((bsz, total_len), fill_value=token_pad.item(), dtype=torch.long, device=params.device)

    # 将提示tokens填入token矩阵
    tokens[:, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=params.device)

    # 创建一个prompt_mask_token，用于稍后识别token是提示token还是填充token
    # 如果是提示token则为True，如果是填充token则为False
    input_text_mask = tokens != token_pad.item()

    # 现在我们可以从第一个位置开始，一次使用一个token从prompt_tokens列表开始推理
    prev_pos = 0
    for cur_pos in range(1, total_len):
        with torch.no_grad():
            logits, _ = model(x=tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)

        # 只有在是填充token时才替换token
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        prev_pos = cur_pos
        if tokens[:, cur_pos] == token_pad.item() and next_token == token_eos.item():
            break

    output_tokens, output_texts = [], []

    for i, toks in enumerate(tokens.tolist()):
        if token_eos.item() in toks:
            eos_idx = toks.index(token_eos.item())
            toks = toks[:eos_idx]

        output_tokens.append(toks)
        output_texts.append(decode(toks))
    return output_tokens, output_texts

# 对概率分布执行top-p (nucleus) 采样
# probs (torch.Tensor): 由logits导出的概率分布张量
# p: top-p采样的概率阈值
# 根据相关研究，Top-p采样选择累积概率质量超过阈值p的最小标记集
# 基于选定的标记重新归一化分布
def sample_top_p(probs, p):
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(prob_idx, -1, next_token)
    # 返回从词汇表中采样的标记索引
    return next_token

## 对用户输入的提示执行推理
prompts = "Consider you what services he has done"
output_tokens, output_texts = generate(model, prompts, ModelArgs())
output_texts = output_texts[0].replace("<|begin_of_text|>", "")
print(output_texts)

## 输出 ##
"""  
Consider you what services he has done o eretrane  
adetranytnn i eey i ade hs rcuh i eey,ad hsatsTns rpae,T  
eon o i hseflns o i eee ee hs ote i ocal ersl,Bnnlnface  
o i hmr a il nwye ademto nt i a ere  
h i ees.  
Frm oe o etrane o oregae,alh,t orede i oeral  
"""
