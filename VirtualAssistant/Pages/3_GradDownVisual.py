# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
# import streamlit as st
# from LearnTorch_ALL.TeachImage.ShowFit import arg_show
# from LearnTorch_ALL.TeachImage.ShowGrad3D import visualize_rosenbrock_show
#
# st.set_page_config(page_title="梯度下降可视化", page_icon="📈")
#
# intro = '''梯度下降是深度学习训练的**主要过程**
#
# 梯度下降是利用一阶导数来优化模型
#
# LearnTorch提供了可视化学习率和迭代功能，帮助你快速了解梯度下降框架有关的概念！'''
#
# intro2 ='''我们使用一个两层的神经网络来可视化模型参数对拟合的影响:
#
# **I**是输入层，固定为1个节点
#
# **H1**是第一个隐藏层，节点数量是可调整的参数**H1**
#
# **H2**是第二个隐藏层，节点数量是可调整的参数**H2**
#
# **O**是输出层，固定为1个节点'''
#
# st.markdown("# 梯度下降可视化📈")
# st.markdown(intro)
# st.divider()
#
# st.markdown("## 1.梯度优化来拟合函数")
# container1 = st.container(border=True)
# with container1:
#     st.header("1.1线性回归梯度下降优化可视化", divider=True)
#     st.write("lr=0.001, iters=200 学习率过低，拟合慢，无法逃出局部最优")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/linear_regression_small_iter_200_lr_0.1.gif")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/linear_regression_sin_small_iter_10000_lr_0.2.gif")
#     st.header("1.2梯度下降优化对比牛顿法优化", divider=True)
#     st.write("牛顿法和海森矩阵都很好，但是计算量太大，而且不是所有问题都有解。\n所以，现在深度学习框架大多采用一阶导数梯度下降优化模型")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/GradV.S.Newton_small_iter_200_10_FPS10.gif")
#
#
# st.markdown("## 2.神经网络模型参数对拟合的影响")
# container1 = st.container(border=True)
# with container1:
#     st.markdown(intro2)
#     st.header("2.1模型参数过少", divider=True)
#     st.write("lr=0.5, iter=10000, H1=5, H2=5 无法很好地拟合复杂模型，欠拟合。")
#
#     st.image("./imgs/nn55.png")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/underfitting_iter_10000_lr_0.5_H1_5_H2_5.gif")
#
#     st.header("2.2模型参数适中", divider=True)
#     st.write("lr=0.5, iter=10000, H1=10, H2=5，适中的模型参数，良好地拟合。")
#
#     st.image("./imgs/nn105.png")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/wellfitting_iter_10000_lr_0.5_H1_10_H2_5.gif")
#
#     st.header("2.3模型参数过多", divider=True)
#     st.write("lr=0.5, iter=10000, H1=15, H2=10 ，过多的模型参数过度拟合模型参数，过拟合。")
#
#     st.image("./imgs/nn1510.png")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/overfitting_iter_10000_lr_0.5_H1_15_H2_10.gif")
#
# st.markdown("## 3.动手试一试")
# container2 = st.container(border=True)
# with container2:
#     st.markdown("调节**模型参数、学习率、迭代次数**试一下吧！")
#     lr = st.slider("学习率：", 0.0, 2.0, 0.1)
#     iters = st.slider("迭代次数：", 0, 15000, 1)
#     h1 = st.slider("第一层隐藏层参数数量：", 0, 20, 1)
#     h2 = st.slider("第二层隐藏层参数数量：", 0, 20, 1)
#     file_path = st.text_input("文件保存路径", "./imgs")
#     if st.button("启动训练", type="secondary",use_container_width=True):
#         st.image(arg_show(lr = lr, iters = iters,fps=10,file_path = file_path,hidden_units1=h1, hidden_units2=h2))
#     st.divider()
#
#     st.markdown("调节**模型参数、学习率、迭代次数、初始点**试一下吧！")
#     lr2 = st.slider("学习率：", 0.000, 1.000, 0.001 , step=0.001,key=5)
#     iters2 = st.slider("迭代次数：", 0, 1000, 1,key=6)
#     x = st.slider("X初始点：", -5, 5, -1,key=7)
#     y = st.slider("Y初始点：", -5, 5, -1,key=8)
#     file_path = st.text_input("文件保存路径", "./imgs",key=2)
#     if st.button("启动训练", type="secondary",use_container_width=True,key=3):
#         st.image(visualize_rosenbrock_show(lr = lr2, iters = iters2 ,file_path = file_path,starting_point=[x,y]))

import streamlit as st
from LearnTorch_ALL.TeachImage.ShowFit import arg_show
from LearnTorch_ALL.TeachImage.ShowGrad3D import visualize_rosenbrock_show

st.set_page_config(page_title="梯度下降可视化", page_icon="📈")

# 页面标题
st.title("📈 梯度下降可视化：优化之旅！")

# 介绍部分
st.write(
    """
    ### 🌟 什么是梯度下降？
    梯度下降是深度学习模型训练的核心方法之一。

    - 它通过计算一阶导数，逐步优化模型参数。
    - 在深度学习中，几乎所有的框架都采用这种高效的优化方法。

    🚀 本页面将通过动态可视化，帮助你直观地理解梯度下降的过程和效果！
    """
)

st.divider()

# 梯度优化部分
st.header("🧠 1. 梯度优化：从理论到实践")

st.write(
    """
    让我们从简单的线性回归开始，看看梯度下降是如何优化模型的：
    """
)

# 线性回归部分
st.subheader("📊 1.1 线性回归中的梯度下降")

st.markdown("#### 🖼️ 动态演示")
st.image("../LearnTorch_ALL/TeachImage/Grad/linear_regression_small_iter_200_lr_0.1.gif", caption="学习率较低的线性回归")
st.image("../LearnTorch_ALL/TeachImage/Grad/linear_regression_sin_small_iter_10000_lr_0.2.gif", caption="较高迭代次数下的线性回归")

st.subheader("⚖️ 1.2 梯度下降与牛顿法对比")

st.write(
    """
    - 牛顿法利用二阶导数（海森矩阵）优化，收敛速度快，但计算量较大。
    - 梯度下降只需一阶导数，简单高效，适用于深度学习。
    """
)

st.image("../LearnTorch_ALL/TeachImage/Grad/GradV.S.Newton_small_iter_200_10_FPS10.gif", caption="梯度下降与牛顿法对比")

st.divider()

# 神经网络部分
st.header("🤖 2. 神经网络的参数与拟合")

st.write(
    """
    通过调整神经网络的隐藏层参数，我们可以看到模型的拟合效果如何变化：
    - **H1**：第一隐藏层节点数。
    - **H2**：第二隐藏层节点数。
    """
)

# 参数调节
st.subheader("🔧 参数调整对拟合的影响")

# 参数过少
st.markdown("#### 2.1 模型参数过少：欠拟合 😢")
st.image("./imgs/nn55.png", caption="H1=5, H2=5 的网络结构")
st.image("../LearnTorch_ALL/TeachImage/Grad/underfitting_iter_10000_lr_0.5_H1_5_H2_5.gif", caption="欠拟合演示")

# 参数适中
st.markdown("#### 2.2 模型参数适中：良好拟合 😊")
st.image("./imgs/nn105.png", caption="H1=10, H2=5 的网络结构")
st.image("../LearnTorch_ALL/TeachImage/Grad/wellfitting_iter_10000_lr_0.5_H1_10_H2_5.gif", caption="良好拟合演示")

# 参数过多
st.markdown("#### 2.3 模型参数过多：过拟合 🤯")
st.image("./imgs/nn1510.png", caption="H1=15, H2=10 的网络结构")
st.image("../LearnTorch_ALL/TeachImage/Grad/overfitting_iter_10000_lr_0.5_H1_15_H2_10.gif", caption="过拟合演示")

st.divider()

# 互动部分
st.header("🎮 3. 动手实验！")

st.write(
    """
    ### 💡 试一试
    - 调节学习率、迭代次数和隐藏层参数，观察模型的变化。
    - 还能通过调节初始点，体验优化路径！
    """
)

# 动手部分 1
st.markdown("#### 📊 实验 1：调节学习率和参数数量")
lr = st.slider("学习率：", 0.0, 2.0, 0.1)
iters = st.slider("迭代次数：", 0, 15000, 1000)
h1 = st.slider("第一隐藏层节点数：", 0, 20, 5)
h2 = st.slider("第二隐藏层节点数：", 0, 20, 5)
file_path = st.text_input("文件保存路径 (可选)：", "./imgs")

if st.button("启动训练 🚀"):
    st.write(f"🔍 正在训练：学习率={lr}, 迭代次数={iters}, H1={h1}, H2={h2}...")
    result_image = arg_show(lr=lr, iters=iters, fps=10, file_path=file_path, hidden_units1=h1, hidden_units2=h2)
    st.image(result_image, caption="训练结果")

# 动手部分 2
st.markdown("#### 🌀 实验 2：优化路径与初始点")
lr2 = st.slider("学习率：", 0.000, 1.000, 0.001, step=0.001, key=5)
iters2 = st.slider("迭代次数：", 0, 1000, 100, key=6)
x = st.slider("X 初始点：", -5, 5, -1, key=7)
y = st.slider("Y 初始点：", -5, 5, -1, key=8)
file_path2 = st.text_input("文件保存路径 (可选)：", "./imgs", key=2)

if st.button("启动路径优化 🚀", key=3):
    st.write(f"🔍 正在优化路径：学习率={lr2}, 迭代次数={iters2}, 初始点=({x}, {y})...")
    result_image2 = visualize_rosenbrock_show(lr=lr2, iters=iters2, file_path=file_path2, starting_point=[x, y])
    st.image(result_image2, caption="优化路径结果")

st.success("🎉 实验完成！快来查看你的结果吧！")
