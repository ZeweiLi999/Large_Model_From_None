import streamlit as st

st.title("CUDA pytorch环境安装教程🪁")
st.divider()
st.header("1.安装cuda和pytorch🚁")
container1 = st.container(border=True)
with container1:
    st.subheader("1.1首先查看自己电脑驱动的最高CUDA版本",divider=True)
    st.code("在终端输入nvidia-smi，查看自己能安装的最大CUDA版本")
    st.image("./imgs/10_Environment/nvidia-smi.png")
    st.write("选择的Pytorch的CUDA版本要小于等于你实际安装的CUDA版本")

    st.subheader("1.2通过阿里云镜像来下载所需要的pytorch版本",divider=True)
    st.write("这是因为清华源没有CUDA版本的pytorch，会下载为cpu版本的pytorch")
    st.image("./imgs/10_Environment/qwen.png")
    st.code("pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118")
    st.image("./imgs/10_Environment/pytorch.png")

st.header("2.安装modelscope💾")
container2 = st.container(border=True,key=2)
with container2:
    st.write("modelscope类似于国内版的huggingface，下载模型速度快")
    st.code("pip install modelscope")
    st.write("下载完modelscope后通过以下命令来下载所需要的模型")
    st.code("modelscope download --model Qwen/Qwen2.5-3B-Instruct")
    st.image("./imgs/10_Environment/model.png")

st.header("3.安装transformers🔋")
container3 = st.container(border=True, key=3)
with container3:
    st.write("需要安装模型指定最低版本以上的transformers，否则模型会产生error")
    st.code("pip install transformers -U")
    st.image("./imgs/10_Environment/transformers.png")
