import torch
from transformers import pipeline, TextStreamer

def Chat_reponse(input, prompt='你现在扮演的是猫娘!' , model_dir = './LLM/Qwen2.5-Coder-3B-Instruct',max_tokens = 512):

    pipe = pipeline(
        "text-generation",
        model = model_dir,
        torch_dtype = torch.bfloat16,
        device_map = "cuda",
    )
    messages = [
        {"role": "system", "content": "{}".format(prompt)},
        {"role": "user", "content": "{}".format(input)},
    ]

    streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

    outputs = pipe(messages, max_new_tokens = max_tokens, streamer=streamer)

    return outputs[0]["generated_text"][-1]['content']