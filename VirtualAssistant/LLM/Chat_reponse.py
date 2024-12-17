import torch
from transformers import pipeline, TextStreamer

def Chat_reponse(input, prompt='You are a private code assistant!' , model_dir = 'D:\pythonProject\Qwen\Qwen\Qwen2.5-Coder-3B-Instruct'):

    pipe = pipeline(
        "text-generation",
        model = model_dir,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
    )
    messages = [
        {"role": "system", "content": "{}".format(prompt)},
        {"role": "user", "content": "{}".format(input)},
    ]

    streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

    outputs = pipe(messages, max_new_tokens = 1024, streamer=streamer)

    return outputs[0]["generated_text"][-1]['content']