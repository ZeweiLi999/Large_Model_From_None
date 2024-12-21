# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# import streamlit as st
#
# st.set_page_config(page_title="计算图可视化", page_icon="🔠")
#
# intro = '''自动微分是深度学习的基础。深度学习的自动微分又通过**计算图**来实现
#
# 计算图分为：
# - 前向计算图(Forward Propagation)
# - 后向计算图(Backward Propagation)
#
# LearnTorch提供了可视化计算图功能，帮助你快速了解深度学习框架自动微分的概念！'''
#
# code_sphere = '''def sphere(x, y):
#     z = x ** 2 + y ** 2
#     return z'''
#
# code_sphere_backward ='''x = Variable(np.array(1.0)) # Variable接收ndarray类型
# y = Variable(np.array(1.0))
# z = sphere(x, y)             # 计算函数
# z.backward(retain_grad=True) # 反向传播retain_grad=True表示保存中间变量导数'''
#
# code_matyas = '''def matyas(x, y):
#     z = 0.26 * (x ** 2 + y ** 2 ) - 0.48 * x * y
#     return z'''
#
# code_matyas_backward ='''x = Variable(np.array(1.0))
# y = Variable(np.array(1.0))
# z = matyas(x, y)
# z.backward(retain_grad=True)'''
#
# st.markdown("# 计算图可视化🔠")
# st.markdown(intro)
# st.divider()
#
# st.markdown("## 1.学习")
# container1 = st.container(border=True)
# with container1:
#     st.markdown("球体公式计算图可视化")
#     st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
#     st.markdown("球体计算公式")
#     st.code(code_sphere, language="python")
#     st.markdown("球体公式反向传播")
#     st.code(code_sphere_backward, language="python")
#     st.markdown("球体公式计算图可视化")
#     st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
#     st.divider()
#     st.markdown("matyas函数计算公式")
#     st.code(code_matyas, language="python")
#     st.markdown("matyas函数反向传播")
#     st.code(code_matyas_backward, language="python")
#     st.markdown("matyas函数计算图可视化")
#     st.image("../LearnTorch_ALL/TeachImage/CGMap/matyas_All.png")

import streamlit as st

st.set_page_config(page_title="计算图可视化", page_icon="🔠")

# 页面标题
st.title("🔠 计算图可视化：理解深度学习的核心！")

# 介绍部分
st.write(
    """
    ### 🌟 为什么计算图如此重要？
    在深度学习中，自动微分是核心，而计算图则是实现自动微分的关键。

    - **前向计算图 (Forward Propagation)**：描述数据从输入到输出的流动路径。
    - **后向计算图 (Backward Propagation)**：计算梯度，优化模型参数。

    🚀 LearnTorch 提供了计算图的可视化功能，帮助你从直观的角度快速理解深度学习框架的原理！
    """
)

st.divider()

# 学习部分
st.header("🧠 1. 理解计算图的结构")

st.write(
    """
    通过以下两个公式，我们来探索计算图的奥秘：

    1️⃣ **球体公式 (Sphere Function)**：$z = x^2 + y^2$。
    2️⃣ **Matyas公式**：$z = 0.26(x^2 + y^2) - 0.48xy$。

    让我们通过计算公式和图形展示它们的前向和后向传播！
    """
)

# Sphere Function 部分
st.subheader("⚽ 球体公式")

st.markdown("#### 🟢 计算图展示")
st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png", caption="球体公式的计算图")

st.markdown("#### 🖋️ 计算公式")
st.code(
    """python
def sphere(x, y):
    z = x ** 2 + y ** 2
    return z
    """,
    language="python"
)

st.markdown("#### 🔄 反向传播代码")
st.code(
    """python
x = Variable(np.array(1.0)) # Variable接收ndarray类型
y = Variable(np.array(1.0))
z = sphere(x, y)             # 计算函数
z.backward(retain_grad=True) # 反向传播
    """,
    language="python"
)

st.divider()

# Matyas Function 部分
st.subheader("📐 Matyas公式")

st.markdown("#### 🟢 计算图展示")
st.image("../LearnTorch_ALL/TeachImage/CGMap/matyas_All.png", caption="Matyas公式的计算图")

st.markdown("#### 🖋️ 计算公式")
st.code(
    """python
def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2 ) - 0.48 * x * y
    return z
    """,
    language="python"
)

st.markdown("#### 🔄 反向传播代码")
st.code(
    """python
x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y) 
z.backward(retain_grad=True)
    """,
    language="python"
)

st.divider()

# 互动部分
st.header("🎮 动手试试看！")

st.write(
    """
    ### 💻 让我们尝试自定义计算：
    - 输入 $x$ 和 $y$ 值。
    - 选择公式，看看计算图和反向传播的结果吧！
    """
)

formula = st.selectbox("选择计算公式：", ["球体公式 (Sphere)", "Matyas公式"])
x = st.number_input("输入 x 的值：", value=1.0)
y = st.number_input("输入 y 的值：", value=1.0)

if st.button("开始计算 🚀"):
    if formula == "球体公式 (Sphere)":
        st.write(f"🌟 使用球体公式计算结果：z = {x**2 + y**2}")
        st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png", caption="球体公式的计算图")
    elif formula == "Matyas公式":
        z = 0.26 * (x**2 + y**2) - 0.48 * x * y
        st.write(f"🌟 使用Matyas公式计算结果：z = {z}")
        st.image("../LearnTorch_ALL/TeachImage/CGMap/matyas_All.png", caption="Matyas公式的计算图")

st.success("🎉 计算完成！快看看结果是否符合你的预期！")
