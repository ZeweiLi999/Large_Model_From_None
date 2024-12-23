import  streamlit as st

st.title("CUDA pytorch环境安装教程🪁")

st.header("安装cuda 11.8🖥︎")
st.subheader("1.首先查看自己电脑的CUDA版本，选择的Pytorch的CUDA版本要小于你实际安装的CUDA版本🖨")
st.image("./Configure_Environment/img.png")
st.subheader("2.通过阿里云镜像来下载所需要的pytorch版本🚁")
st.image("./Configure_Environment/qwen.png")
st.subheader("pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118")
st.image("./Configure_Environment/pytorch.png")

st.header("安装modelscope💾")
st.subheader("pip install modelscope")
st.subheader("下载完modelscope后通过以下命令来下载所需要的模型")
st.subheader("modelscope download --model Qwen/Qwen2.5-3B-Instruct")
st.image("./Configure_Environment/model.png")

st.header("安装transformers🔋")
st.subheader("pip install transformers -U")
st.subheader("需要安装一定版本以上的transformers，否则模型会产生error")
st.image("./Configure_Environment/transformers.png")
