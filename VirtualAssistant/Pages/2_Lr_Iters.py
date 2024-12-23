# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
# import streamlit as st
# from LearnTorch_ALL.TeachImage.ShowLearningRate import gradient_descent_show
#
# st.set_page_config(page_title="学习率和迭代次数可视化", page_icon="📊")
#
# intro = '''学习率和迭代次数是深度学习训练的重要**超参数**
#
# 学习率和迭代次数影响的重要概念：
# - 欠拟合
# - 拟合
# - 过拟合
#
#
# LearnTorch提供了可视化学习率和迭代功能，帮助你快速了解深度学习框架有关拟合的概念！'''
#
# st.markdown("# 学习率和迭代次数📊")
# st.markdown(intro)
# st.divider()
#
# st.markdown("## 1.学习率")
# container1 = st.container(border=True)
# with container1:
#     st.header("1.1学习率过低", divider=True)
#     st.write("lr=0.001, iters=200 学习率过低，拟合慢，无法逃出局部最优")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Underfit_lr0.001_iters200_FPS10.gif")
#     st.header("1.2学习率适中", divider=True)
#     st.write("lr=0.085, iters=200，拟合速度较快")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Wellfit_lr0.085_iters200_FPS10.gif")
#     st.header("1.3学习率过大", divider=True)
#     st.write("lr=0.2, iters=200 ，无法接近全局最优")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Overfit_lr0.2_iters200_FPS10.gif")
#
#
# st.markdown("## 2.迭代次数")
# container1 = st.container(border=True)
# with container1:
#     st.header("2.1迭代次数过低", divider=True)
#     st.write("lr=0.085, iters=5 迭代次数少，无法拟合到最优")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterlow_lr0.085_iters5_FPS10.gif")
#     st.header("2.2迭代次数适中", divider=True)
#     st.write("lr=0.085, iters=50，迭代次数适中，接近全局最优")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_itermain_lr0.085_iters50_FPS10.gif")
#     st.header("2.3迭代次数过多", divider=True)
#     st.write("lr=0.085, iters=100 ，浪费训练资源")
#     st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterhigh_lr0.085_iters100_FPS10.gif")
#
# st.markdown("## 3.动手试一试")
# container2 = st.container(border=True)
# with container2:
#     st.markdown("调节学习率和迭代次数试一下吧！")
#     lr = st.slider("学习率：", 0.0, 1.0, 0.1)
#     iters = st.slider("迭代次数：", 0, 100, 1)
#     file_path = st.text_input("文件保存路径", "./imgs")
#     if st.button("启动训练", type="secondary",use_container_width=True):
#         st.image(gradient_descent_show(lr = lr, iters = iters,fps=10,file_path = file_path))

import streamlit as st
from LearnTorch_ALL.TeachImage.ShowLearningRate import gradient_descent_show

st.set_page_config(page_title="学习率和迭代次数可视化", page_icon="📊")

# 页面标题
st.title("📊 学习率和迭代次数：一场拟合的冒险！")

# 介绍部分
st.write(
    """
    ### 🤔 为什么学习率和迭代次数这么重要？
    在深度学习中，学习率（learning rate, lr）和迭代次数（iterations, iters）是一对非常关键的超参数！

    - 学习率太低？模型就像蜗牛一样慢慢走，永远到不了终点 🐌。
    - 学习率太高？模型就像坐上了过山车，忽上忽下，总是错过最优解 🎢。
    - 迭代次数不够？模型还没学会走路就被停下了 🙁。
    - 迭代次数太多？模型反而过度训练，学到了奇怪的东西 🤯。

    来吧！通过可视化探索学习率和迭代次数对模型的影响，成为深度学习超参数调节的高手！✨
    """
)

st.divider()

# 学习率部分
st.header("🌟 1. 学习率大揭秘！")

st.write("学习率决定了模型每次更新时的步伐大小。接下来，让我们看看三种典型情况：")

# 学习率过低
st.subheader("🔻 1.1 学习率过低：走得太慢 🐢")
st.write("当 lr=0.001，iters=200 时，模型拟合速度非常慢，可能卡在局部最优。")
st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Underfit_lr0.001_iters200_FPS10.gif")

# 学习率适中
st.subheader("✅ 1.2 学习率适中：刚刚好 😊")
st.write("当 lr=0.085，iters=200 时，模型拟合速度适中，快速找到全局最优。")
st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Wellfit_lr0.085_iters200_FPS10.gif")

# 学习率过高
st.subheader("🔺 1.3 学习率过高：太激进 🚀")
st.write("当 lr=0.2，iters=200 时，模型无法稳定下来，总是在震荡中错过全局最优。")
st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Overfit_lr0.2_iters200_FPS10.gif")

st.divider()

# 迭代次数部分
st.header("🌀 2. 迭代次数的影响")

st.write("迭代次数决定了模型训练的总步数。让我们看看不同迭代次数的影响：")

# 迭代次数过低
st.subheader("🔻 2.1 迭代次数过低：还没学会呢 🤷")
st.write("当 lr=0.085，iters=5 时，模型的迭代次数太少，训练不足，无法接近全局最优。")
st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterlow_lr0.085_iters5_FPS10.gif")

# 迭代次数适中
st.subheader("✅ 2.2 迭代次数适中：完美拟合 🏆")
st.write("当 lr=0.085，iters=50 时，模型刚好接近全局最优，资源利用高效。")
st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_itermain_lr0.085_iters50_FPS10.gif")

# 迭代次数过多
st.subheader("🔺 2.3 迭代次数过多：浪费资源 💸")
st.write("当 lr=0.085，iters=100 时，模型过度训练，浪费了时间和计算资源。")
st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_iterhigh_lr0.085_iters100_FPS10.gif")

st.divider()

# 交互部分
st.header("🛠️ 3. 动手试试吧！")

st.write(
    """
    ### 🎮 让我们来试试调节学习率和迭代次数，看看对模型的影响吧！
    - 拖动滑块选择学习率和迭代次数。
    - 点击 **启动训练** 按钮，观察训练结果的动态变化。
    """
)

lr = st.slider("学习率 (lr):", 0.0, 1.0, 0.1, step=0.01)
iters = st.slider("迭代次数 (iters):", 0, 100, 10, step=5)
file_path = st.text_input("保存训练结果路径 (可选):", "./imgs")

if st.button("启动训练 🚀"):
    st.write(f"🔍 正在训练模型：学习率 = {lr}, 迭代次数 = {iters} ...")
    result_image = gradient_descent_show(lr=lr, iters=iters, fps=10, file_path=file_path)
    st.image(result_image, caption="训练结果可视化")

st.success("🎉 试验完成???!!! 看看结果是否符合你的预期？")
