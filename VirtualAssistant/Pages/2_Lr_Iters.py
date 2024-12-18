if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import streamlit as st
from LearnTorch_ALL.TeachImage.ShowLearningRate import gradient_descent_show

print("Current working directory:", os.getcwd())
st.set_page_config(page_title="学习率和迭代次数", page_icon="📊")

intro = '''学习率和迭代次数是深度学习训练的重要**超参数**

学习率和迭代次数影响的重要概念：
- 欠拟合
- 拟合
- 过拟合


LearnTorch提供了可视化学习率和迭代功能，帮助你快速了解深度学习框架有关拟合的概念！'''

st.markdown("# 学习率和迭代次数📊")
st.markdown(intro)
st.divider()

st.markdown("## 1.学习率")
container1 = st.container(border=True)
with container1:
    st.header("1.1学习率过低", divider=True)
    st.write("lr=0.001, iters=200 学习率过低，拟合慢，无法逃出局部最优")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Underfit_lr0.001_iters200_FPS10.gif")
    st.header("1.2学习率适中", divider=True)
    st.write("lr=0.085, iters=200，拟合速度较快")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Wellfit_lr0.085_iters200_FPS10.gif")
    st.header("1.3学习率过大", divider=True)
    st.write("lr=0.2, iters=200 ，无法接近全局最优")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Overfit_lr0.2_iters200_FPS10.gif")


st.markdown("## 2.迭代次数")
container1 = st.container(border=True)
with container1:
    st.header("2.1迭代次数过低", divider=True)
    st.write("lr=0.085, iters=5 迭代次数少，无法拟合到最优")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterlow_lr0.085_iters5_FPS10.gif")
    st.header("2.2迭代次数适中", divider=True)
    st.write("lr=0.085, iters=50，迭代次数适中，接近全局最优")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_itermain_lr0.085_iters50_FPS10.gif")
    st.header("2.3迭代次数过多", divider=True)
    st.write("lr=0.085, iters=100 ，浪费训练资源")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterhigh_lr0.085_iters100_FPS10.gif")

st.markdown("## 3.动手试一试")
container2 = st.container(border=True)
with container2:
    st.markdown("调节学习率和迭代次数试一下吧！")
    lr = st.slider("学习率：", 0.0, 1.0, 0.1)
    iters = st.slider("迭代次数：", 0, 100, 1)
    file_path = st.text_input("文件保存路径", "../imgs")
    if st.button("启动训练", type="secondary",use_container_width=True):
        st.image(gradient_descent_show(lr = lr, iters = iters,fps=10,file_path = file_path))