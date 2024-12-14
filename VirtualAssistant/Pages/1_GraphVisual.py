if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import streamlit as st

st.set_page_config(page_title="计算图可视化", page_icon="🔠")

intro = '''自动微分是深度学习的基础。深度学习的自动微分又通过**计算图**来实现

计算图分为：
- 前向计算图(Forward Propagation)
- 后向计算图(Backward Propagation)

LearnTorch提供了可视化计算图功能，帮助你快速了解深度学习框架自动微分的概念！'''

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

st.markdown("# 计算图可视化🔠")
st.markdown(intro)
st.divider()

st.markdown("## 1.学习")
container1 = st.container(border=True)
with container1:
    st.markdown("球体公式计算图可视化")
    st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
    st.markdown("球体计算公式")
    st.code(code_sphere, language="python")
    st.markdown("球体公式反向传播")
    st.code(code_sphere_backward, language="python")
    st.markdown("球体公式计算图可视化")
    st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
    st.divider()
    st.markdown("matyas函数计算公式")
    st.code(code_matyas, language="python")
    st.markdown("matyas函数反向传播")
    st.code(code_matyas_backward, language="python")
    st.markdown("matyas函数计算图可视化")
    st.image("../LearnTorch_ALL/TeachImage/CGMap/matyas_All.png")

