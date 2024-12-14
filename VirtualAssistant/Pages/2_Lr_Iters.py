if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import streamlit as st

st.set_page_config(page_title="学习率和迭代次数", page_icon="📊")

intro = '''学习率和迭代次数是深度学习训练的重要**超参数**

学习率和迭代次数影响的重要概念：
- 欠拟合
- 拟合
- 过拟合


LearnTorch提供了可视化学习率和迭代功能，帮助你快速了解深度学习框架有关拟合的概念！'''

code_sphere = '''def sphere(x, y):
    z = x ** 2 + y ** 2
    return z'''

code_sphere_backward ='''x = Variable(np.array(1.0)) # Variable接收ndarray类型
y = Variable(np.array(1.0)) 
z = sphere(x, y)             # 计算函数
z.backward(retain_grad=True) # 反向传播retain_grad=True表示保存中间变量导数'''

code_matyas = '''def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2 ) - 0.48 * x * y
    return z'''

code_matyas_backward ='''x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y) 
z.backward(retain_grad=True)'''

st.markdown("# 学习率和迭代次数📊")
st.markdown(intro)
st.divider()

st.markdown("## 1.学习")
container1 = st.container(border=True)
with container1:
    st.markdown("### 1.1欠拟合")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Underfit_lr0.001_iters200_FPS10.gif")
    st.markdown("### 1.2拟合")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Wellfit_lr0.085_iters200_FPS10.gif")
    st.markdown("### 1.3过拟合")
    st.image("../LearnTorch_ALL/TeachImage/Grad/Gradient_Overfit_lr0.2_iters200_FPS10.gif")

st.markdown("## 2.动手试一试")
container2 = st.container(border=True)
with container2:
    st.markdown("调节学习率和迭代次数试一下吧！")
