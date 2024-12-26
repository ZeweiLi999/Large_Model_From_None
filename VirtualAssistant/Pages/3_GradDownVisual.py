if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import streamlit as st
from LearnTorch_ALL.TeachImage.ShowFit import arg_show
from LearnTorch_ALL.TeachImage.ShowGrad3D import visualize_rosenbrock_show

st.set_page_config(page_title="梯度下降可视化", page_icon="📈")

intro0 = '''梯度下降是利用一阶导数来优化模型，是深度学习**训练**的**主要过程**。
'''

intro1='''
### 📈 1. 梯度下降基本概念
梯度下降（Gradient Descent）是一种最优化算法，用于通过反向传播不断更新模型的参数，以最小化损失函数（Loss Function）。它的目标是找到使损失函数最小的参数。我们通过计算损失函数对每个参数的**梯度**（即导数）来决定如何更新这些参数。

公式：

给定一个损失函数 $ L(\theta) $ 和参数向量 $ \theta $，梯度下降的更新规则是：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L(\theta)
$$

其中：
- $ \theta_{\text{old}} $ 是当前的参数值。
- $ \eta $ 是学习率，控制步长的大小。
- $ \nabla L(\theta) $ 是损失函数相对于参数的梯度（即导数），表示损失函数对每个参数的变化率。

在训练过程中，梯度下降通过迭代调整 $ \theta $，不断降低损失值，直到找到最优解。
'''

intro2='''
### 🚀 2. 学习率的影响
学习率（Learning Rate）是梯度下降中的一个超参数，它决定了每次更新参数时步伐的大小。如果学习率过大，可能会跳过最优解；如果学习率过小，收敛速度可能会非常慢，甚至陷入局部最优。

- **学习率过大**：会导致模型在优化过程中震荡，无法找到最优解。
- **学习率适中**：可以帮助模型高效地找到最优解。
- **学习率过小**：会导致优化过程过慢，可能需要更多的迭代才能收敛。

 **数学公式：**

更新公式中，学习率 $ \eta $ 控制每次迭代的步长：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L(\theta)
$$
'''

intro3='''
### 🌍 3. 迭代次数的影响
迭代次数决定了我们进行梯度下降优化的步数。在实际应用中，迭代次数过少可能导致模型欠拟合，而迭代次数过多可能导致过拟合。

- **迭代次数过少**：模型训练不足，不能充分学习数据的特征。
- **迭代次数适中**：训练足够充分，模型能够较好地拟合数据。
- **迭代次数过多**：可能导致过拟合，模型学习到过多的噪音。

**数学公式：**

每一次迭代，我们更新模型的参数：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L(\theta)
$$

其中，$\theta$ 是我们要优化的参数，$\nabla L(\theta)$ 是损失函数的梯度。'''

intro4='''
### 🏆 4. 梯度下降的最终目标
梯度下降的目标是通过反复迭代，找到一组使得损失函数最小的参数 $ \theta $，从而使得模型能够最优地拟合训练数据。


LearnTorch提供了可视化学习率和迭代功能，帮助你快速了解梯度下降框架有关的概念！'''

intro5 ='''我们使用一个两层的神经网络来可视化模型参数对拟合的影响:

**I**是输入层，固定为1个节点

**H1**是第一个隐藏层，节点数量是可调整的参数**H1**

**H2**是第二个隐藏层，节点数量是可调整的参数**H2**

**O**是输出层，固定为1个节点'''

st.markdown("# 梯度下降可视化📈")
st.markdown("## :star:1.理论学习")
st.markdown(intro0)
st.divider()
container1 = st.container(border=True,key=1)
with container1:
    st.markdown(intro1)
    st.divider()
    st.markdown(intro2)
    st.divider()
    st.markdown(intro3)
    st.divider()
    st.markdown(intro4)

st.markdown("## :star:2.可视化加深理解")
container2 = st.container(border=True,key=2)
with container2:
    st.header("2.1梯度优化来拟合函数")
    st.subheader("2.1.1线性回归梯度下降优化可视化", divider=True)
    st.write("lr=0.001, iters=200 学习率过低，拟合慢，无法逃出局部最优")
    st.image("../LearnTorch_ALL/TeachImage/Grad/linear_regression_small_iter_200_lr_0.1.gif")
    st.image("../LearnTorch_ALL/TeachImage/Grad/linear_regression_sin_small_iter_10000_lr_0.2.gif")

    st.header("2.2神经网络模型参数对拟合的影响")
    st.markdown(intro5)
    st.subheader("2.2.1模型参数过少", divider=True)
    st.write("lr=0.5, iter=10000, H1=5, H2=5 无法很好地拟合复杂模型，欠拟合。")

    st.image("./imgs/3_GradDownVisual/nn55.png")
    st.image("../LearnTorch_ALL/TeachImage/Grad/underfitting_iter_10000_lr_0.5_H1_5_H2_5.gif")

    st.subheader("2.2.2模型参数适中", divider=True)
    st.write("lr=0.5, iter=10000, H1=10, H2=5，适中的模型参数，良好地拟合。")

    st.image("./imgs/3_GradDownVisual/nn105.png")
    st.image("../LearnTorch_ALL/TeachImage/Grad/wellfitting_iter_10000_lr_0.5_H1_10_H2_5.gif")

    st.subheader("2.2.3模型参数过多", divider=True)
    st.write("lr=0.5, iter=10000, H1=15, H2=10 ，过多的模型参数过度拟合模型参数，过拟合。")
    st.image("./imgs/3_GradDownVisual/nn1510.png")
    st.image("../LearnTorch_ALL/TeachImage/Grad/overfitting_iter_10000_lr_0.5_H1_15_H2_10.gif")

st.markdown("## :star:3.动手试一试")
container3 = st.container(border=True,key=3)
with container3:
    st.markdown("调节**模型参数、学习率、迭代次数**试一下吧！")
    lr = st.slider("学习率：", 0.0, 2.0, 0.1)
    iters = st.slider("迭代次数：", 0, 15000, 1)
    h1 = st.slider("第一层隐藏层参数数量：", 0, 20, 1)
    h2 = st.slider("第二层隐藏层参数数量：", 0, 20, 1)
    file_path = st.text_input("文件保存路径", "./imgs/3_GradDownVisual")
    if st.button("启动训练", type="secondary",use_container_width=True):
        st.image(arg_show(lr = lr, iters = iters,fps=10,file_path = file_path,hidden_units1=h1, hidden_units2=h2))
    st.divider()

    st.markdown("调节**模型参数、学习率、迭代次数、初始点**试一下吧！")
    lr2 = st.slider("学习率：", 0.000, 1.000, 0.001 , step=0.001,key=5)
    iters2 = st.slider("迭代次数：", 0, 1000, 1,key=6)
    x = st.slider("X初始点：", -5, 5, -1,key=7)
    y = st.slider("Y初始点：", -5, 5, -1,key=8)
    file_path = st.text_input("文件保存路径", "./imgs/3_GradDownVisual",key=9)
    if st.button("启动训练", type="secondary",use_container_width=True,key=10):
        st.image(visualize_rosenbrock_show(lr = lr2, iters = iters2 ,file_path = file_path,starting_point=[x,y]))

st.markdown("## :star:4.拓展")
container4 = st.container(border=True,key=4)
with container4:
    st.header("4.1梯度下降优化对比牛顿法优化", divider=True)
    st.write(
        "牛顿法和海森矩阵都很好，但是计算量太大，而且不是所有问题都有解。\n所以，现在深度学习框架大多采用一阶导数梯度下降优化模型")
    st.image("../LearnTorch_ALL/TeachImage/Grad/GradV.S.Newton_small_iter_200_10_FPS10.gif")