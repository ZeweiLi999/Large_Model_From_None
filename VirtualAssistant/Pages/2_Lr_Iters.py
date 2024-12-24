if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import streamlit as st
from LearnTorch_ALL.TeachImage.ShowLearningRate import gradient_descent_show

st.set_page_config(page_title="学习率和迭代次数可视化", page_icon="📊")

intro1 = '''
讲讲两个非常重要的概念——**学习率（Learning Rate）**和**迭代次数（Iterations）**。


### 1.什么是学习率？🤔

首先，我们从一个简单的例子说起。

##### 假设你在爬山⛰️：

你站在一个山顶，想要下到山脚。为了找到最快的路径，你决定根据山坡的坡度来一步一步走下去。

- **学习率**就好比你每一步走的步伐大小。
- 如果你每一步走得太大（学习率太大），可能会走得过远，错过了最短路径❌。
- 如果你每一步走得太小（学习率太小），可能需要走很长时间才能到达目标，甚至可能会停留在某个局部最小值，无法达到最终的目标⏳。

所以，学习率的选择直接影响我们能否高效、准确地找到目标。

##### 数学公式表示：

我们可以通过一个简单的梯度下降公式来表示学习率的作用。

假设我们要最小化一个损失函数 $L(\theta)$，其中 \theta 是模型的参数。梯度下降的公式如下：

$$
\theta_{new} = \theta_{old} - \eta \cdot \nabla L(\theta)
$$

这里：

- $ \theta_{new} $ 和 $ \theta_{old} $ 分别是当前和更新后的模型参数。
- $ \eta $（即学习率）是控制步长的系数。
- $ \nabla L(\theta) $ 是损失函数的梯度，表示当前点的坡度。

通过这个公式，我们可以看到，学习率 $ \eta $ 决定了每次参数更新的步伐大小。

#### 小结：

- **学习率太大**：可能导致参数更新过大，跳过了最优解，甚至导致不收敛❌。
- **学习率太小**：可能导致收敛速度很慢，训练时间变长⏳。

**正确的学习率**应该能够帮助你平稳而快速地下降到损失函数的最低点。

'''
intro2 = '''
## 2.什么是迭代次数？🤔

迭代次数，顾名思义，就是你进行学习的次数。

#### 假设你在游泳：🏊‍♂️

想象一下你正在学习游泳。如果你一次只游十秒钟，你是不是需要多次重复练习才能学会游泳？当然是的。你每游一次（每次迭代），就会提升一些技能，慢慢地你就能游得更好。

在机器学习中，我们也是通过多次更新模型参数来不断优化模型。这些更新的过程就叫做**迭代**。每次迭代，模型会根据梯度信息调整参数，直到找到最优解。

#### 数学公式表示：

在每次迭代中，模型参数会根据梯度下降法进行更新：

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \nabla L(\theta^{(t)})
$$

这里的 $ t $ 表示当前的迭代次数，$ \theta^{(t)} $ 表示在第 $ t $ 次迭代时的模型参数。

#### 迭代次数与学习过程：

- **迭代次数太少**：训练不足，模型可能没有学到足够的知识，效果不好。
- **迭代次数太多**：虽然模型可能会进一步优化，但是也可能会因为过拟合而导致性能下降，或者训练时间过长。

**理想的迭代次数**是能够在合适的时间内找到一个足够好的解，避免过拟合。

'''

intro3 = '''

## 3.学习率与迭代次数的关系🔗

学习率和迭代次数是互相影响的。一个合理的学习率和合适的迭代次数才能够帮助你得到一个既高效又准确的模型。

- **学习率过大**，即使你进行很多次迭代，也可能因为步伐过大导致无法收敛。
- **学习率过小**，即使进行很多次迭代，也可能会因为收敛太慢而浪费时间。

我们需要通过实验来找到合适的学习率和迭代次数。
'''

intro4 = '''
## 4.如何选择学习率和迭代次数？🔍

1. **学习率的选择**：
   - 可以从一个小的学习率开始，例如 0.001，然后逐步增加或减小，观察模型的收敛情况。
   - 可以使用 **学习率衰减**（learning rate decay）来逐渐减小学习率，这样可以在训练后期更精细地调整模型参数。

2. **迭代次数的选择**：
   - 通过 **早停（early stopping）** 技术来避免过拟合，即如果模型在一定的迭代次数内，验证集的误差没有降低，就停止训练。
'''

intro5 ='''

## 5.可视化学习率与迭代次数的影响📊

我们可以通过一个简单的图示来看看学习率和迭代次数对训练效果的影响：

- **图1：不同的学习率**

    ![Learning Rate](https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-05-23-%E5%AD%A6%E4%B9%A0%E7%8E%87Learning%20rate/pic1.jpg)

    在这个图中，步伐太大导致了更新过程的剧烈波动，甚至跳出了最优解的区域。
'''

intro6 = '''
## 6.结论🔑

在机器学习中，**学习率**决定了每一步的步伐大小，**迭代次数**决定了我们进行多少次学习和优化。选择合适的学习率和迭代次数是至关重要的，它们直接影响到模型的训练效率和效果。
'''

intro7= '''你可能发现学习率适中的情况也有点小问题，因为他学习率不够低，一直在最低点周围震荡。
    
其实你已经接触到学习率进阶应用：

**学习率衰减算法**

https://zh-v2.d2l.ai/chapter_optimization/lr-scheduler.html
'''


st.markdown("# 学习率和迭代次数📊")
st.write(
    """
    ### 为什么学习率和迭代次数这么重要？
    在深度学习中，学习率（learning rate, lr）和迭代次数（iterations, iters）是一对非常关键的超参数！

    - 学习率太低？模型就像蜗牛一样慢慢走，永远到不了终点 🐌。
    - 学习率太高？模型就像坐上了过山车，忽上忽下，总是错过最优解 🎢。
    - 迭代次数不够？模型还没学会走路就被停下了 🙁。
    - 迭代次数太多？模型反而过度训练，学到了奇怪的东西 🤯。

    来吧！**LearnTorch**提供了可视化学习率和迭代功能，帮助你成为深度学习超参数调节的高手！✨
    """
)
st.divider()

st.markdown("## :star:1.理论学习")
container1 = st.container(border=True,key=1)
with container1:
    st.markdown(intro1)
    st.divider()
    st.markdown(intro2)
    st.divider()
    st.markdown(intro3)
    st.divider()
    st.markdown(intro4)
    st.divider()
    st.markdown(intro5)
    st.divider()
    st.markdown(intro6)


st.markdown("## :star:2.可视化加深理解")
container2 = st.container(border=True,key=2)
with container2:
    st.header("1.学习率的影响")
    st.write("学习率决定了模型每次更新时的步伐大小。接下来，让我们看看三种典型情况：")
    st.subheader("1.1学习率过低：走得太慢 🐢", divider=True)
    st.write("当 lr=0.001，iters=200 时，模型拟合速度非常慢，可能卡在局部最优。")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Underfit_lr0.001_iters200_FPS10.gif")
    st.subheader("1.2学习率适中：刚刚好 😊", divider=True)
    st.write("当 lr=0.085，iters=200 时，模型拟合速度适中，快速找到全局最优。")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Wellfit_lr0.085_iters200_FPS10.gif")
    st.subheader("1.3学习率过大：太激进 🚀", divider=True)
    st.write("当 lr=0.2，iters=200 时，模型无法稳定下来，总是在震荡中错过全局最优。")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Overfit_lr0.2_iters200_FPS10.gif")
    st.divider()

    st.header("2.迭代次数的影响")
    st.write("迭代次数决定了模型训练的总步数。让我们看看迭代次数不同的三种情况：")
    st.subheader("2.1迭代次数过低：还没学会呢 🤷", divider=True)
    st.write("当 lr=0.085，iters=5 时，模型的迭代次数太少，训练不足，无法接近全局最优。")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterlow_lr0.085_iters5_FPS10.gif")
    st.subheader("2.2迭代次数适中：完美拟合 🏆", divider=True)
    st.write("当 lr=0.085，iters=50 时，模型刚好接近全局最优，资源利用高效。")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_itermain_lr0.085_iters50_FPS10.gif")
    st.subheader("2.3迭代次数过多：浪费资源 💸", divider=True)
    st.write("当 lr=0.085，iters=100 时，模型过度训练，浪费了时间和计算资源。")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterhigh_lr0.085_iters100_FPS10.gif")

st.markdown("## :star:3.动手试一试")
container3 = st.container(border=True,key=3)
with container3:
    st.markdown("调节学习率和迭代次数试一下吧！")
    lr = st.slider("学习率：", 0.0, 1.0, 0.1)
    iters = st.slider("迭代次数：", 0, 100, 1)
    file_path = st.text_input("文件保存路径", "./imgs/2_Lr_Iters/")
    if st.button("启动训练 🚀", type="secondary",use_container_width=True):
        st.write(f"🔍 正在训练模型：学习率 = {lr}, 迭代次数 = {iters} ...")
        st.image(gradient_descent_show(lr = lr, iters = iters,fps=10,file_path = file_path))

st.markdown("## :star:4.拓展")
container4 = st.container(border=True,key=4)
with container4:
    st.markdown(intro7)
    st.image("./imgs/2_Lr_Iters/lr.PNG")
