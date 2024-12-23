if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import streamlit as st
from LearnTorch_ALL.TeachImage.ShowLearningRate import gradient_descent_show

st.set_page_config(page_title="低参数大模型测评", page_icon="🖥️")



base = '''
Base模型通常是指**只经过了预训练**，未经特定任务微调的基础模型，在训练过程中最初被开发和优化的，它旨在平衡性能和资源消耗。

用途：这些模型通常用于进一步的微调，以适应特定任务或应用场景。如：智能对话、文本内容生成等

特点：它们包含了大量通用知识，但没有针对特定任务进行优化。'''

instruct = '''
Instruct模型是指**经过了指令微调**为遵循指令或完成特定任务而设计和优化的模型。

用途：用于执行具体指令，如回答问题、生成文本、翻译等任务。

特点：经过指令数据集微调，能够更好地理解和执行用户提供的指令。
'''
chat = '''
上面两步完成后，**偏好训练**中专门为对话系统（聊天机器人）设计和优化。

用途：用于生成自然语言对话，能够理解上下文并生成连贯且有意义的回复。如：聊天机器人、智能助力

特点：通常经过大量对话数据微调，具备更好的上下文理解能力和对话生成能力。'''

intro = '''你已经学过了深度学习的重要概念，现在可以配好环境，挑点最热门最新的模型玩玩上手试一试

模型介绍：
- Qwen2.5-3B-instruct
- Qwen2.5-3B-coder
- LLama3.2B-instruct


LearnTorch提供了低参数大模型测评，帮助你快速了解业界最新的模型！'''

st.markdown("# 低参数大模型测评🖥️")
st.markdown(intro)
st.divider()

st.markdown("## 1.模型种类介绍")
container1 = st.container(border=True)
with container1:
    st.markdown('''就像进了餐厅要点餐，我们先要看懂菜名。
                挑选大模型也是如此，要挑模型我们先要**看懂模型名称**''')
    st.image("./imgs/8_ShowLLM_imgs/show_model_name.PNG")

    st.header("1.1参数名称", divider=True)
    st.image("./imgs/8_ShowLLM_imgs/name.PNG")
    st.write("代表这是哪一款模型，这里Qwen2.5 是 Qwen 大型语言模型系列的最新成果，2024/9/25推出。")

    st.header("1.2模型参数", divider=True)
    st.image("./imgs/8_ShowLLM_imgs/3B.PNG")
    st.markdown("代表这个模型有多少参数数量，参数数量决定了机器能否运行这个模型，参数数量和显存占用有具体的公式，我们目前只需要大致知道"
             "不进行量化，原始模型**1B参数≈1G显存占用**就可以")

    st.header("1.3模型类别", divider=True)
    st.image("./imgs/8_ShowLLM_imgs/model_class.PNG")
    st.markdown("代表这个模型是什么类别，下面是一些例子")
    st.write("大模型训练一般有三个步骤：**模型预训练、指令微调、对齐**")
    st.subheader("Base")
    st.write(base)
    st.subheader("Instruct")
    st.write(instruct)
    st.subheader("Chat")
    st.write(chat)

    st.header("1.4量化类型", divider=True)
    st.image("./imgs/8_ShowLLM_imgs/quantify.PNG")
    st.markdown("代表这个模型把权重量化，一般可以减少计算量和空间占用，但是会有性能损失，有些也不能微调，一般是最后一步。")



st.markdown("## 2.模型测评")
container2 = st.container(border=True)
with container2:
    st.header("Qwen2.5-3B-Instruct模型", divider=True)
    st.markdown("安全限制强，中文能力强，语言优美，讲故事能力强")
    st.image("./imgs/8_ShowLLM_imgs/Qwen2.5-3B-Instruct.png")
    st.header("Qwen2.5-3B-Code模型", divider=True)
    st.markdown("专门为了写代码和修正代码设计的模型，"
                "具体可参照技术报告 https://arxiv.org/pdf/2409.12186 ，"
                "基本上每个回答都会有可复制的代码块")
    st.image("./imgs/8_ShowLLM_imgs/Qwen2.5-3B-Code.png")
    st.header("LLama3.2-3B-Instruct模型", divider=True)
    st.markdown("安全限制弱，中文能力一般，有时候讲中文会胡言乱语，需要中文加强微调，讲故事能力强")
    st.image("./imgs/8_ShowLLM_imgs/LLama3.2-3B-Instruct.png")




