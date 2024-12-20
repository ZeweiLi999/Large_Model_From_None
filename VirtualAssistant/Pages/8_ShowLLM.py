if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import streamlit as st
from LearnTorch_ALL.TeachImage.ShowLearningRate import gradient_descent_show

st.set_page_config(page_title="低参数大模型测评", page_icon="🖥️")

intro = '''你已经学过了深度学习的重要概念，现在可以挑点最热门最新的模型玩玩上手试一试，同时还能配好环境

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
    st.header("1.1参数", divider=True)
    st.header("1.2base模型", divider=True)
    st.header("1.3instruct模型", divider=True)




st.markdown("## 2.Qwen2.5-3B-instruct")
container2 = st.container(border=True)
with container2:
    st.header("Qwen2.5模型", divider=True)



st.markdown("## 2.Qwen2.5-3B-code")
container2 = st.container(border=True)
with container2:
    st.header("Qwen2.5-3B-code模型", divider=True)


st.markdown("## 4.LLama3.2-3B-instruct")
container2 = st.container(border=True)
with container2:
    st.header("LLama3.2-3B-instruct", divider=True)

