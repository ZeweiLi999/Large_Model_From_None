import streamlit as st

st.set_page_config(page_title="计算图可视化", page_icon="🔠")

intro = '''自动微分是深度学习的基础。深度学习的自动微分又通过**计算图**来实现

计算图分为：
- 前向计算图(Forward Propagation)
- 后向计算图(Backward Propagation)

LearnTorch提供了可视化计算图功能，试一试吧！'''

code_sphere = '''def sphere(x, y):
    z = x ** 2 + y ** 2
    return z'''
code_backward ='''x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y) 
z.backward(retain_grad=True)'''


st.markdown("# 计算图可视化")
st.markdown(intro)
st.divider()

st.markdown("## 1.学习")
st.markdown("球体公式计算图可视化")
st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
st.markdown("球体计算公式")
st.code(code_sphere, language="python")
st.markdown("球体公式反向传播")
st.code(code_backward, language="python")
st.divider()

st.markdown("## 2.模仿改造")
container2 = st.container(border=True)
with container2:
    st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
    left, right = st.columns(2)
    agree1 = left.checkbox("是否保存文件", key="Grad31")
    if agree1:
        file_name1 = left.text_input("文件名")
        option1 = left.selectbox("文件类型",("PNG", "PDF"),)
        st.write("选择文件类型:", option1)
    verbose1 = right.checkbox("是否详细显示", key="Grad32")
    if container2.button("运行",icon="😃", use_container_width=True,key=2):
        container2.markdown("开始运行！")

st.markdown("## 3.创新")
st.markdown("构建你的函数")
st.code(code_sphere, language="python")
st.markdown("进行反向传播吧！")
st.code(code_backward, language="python")
st.divider()
container3 = st.container(border=True)
with container3:
    st.image("../LearnTorch_ALL/TeachImage/CGMap/sphere_All.png")
    left, right = st.columns(2)
    agree1 = left.checkbox("是否保存文件", key="Grad33")
    if agree1:
        file_name1 = left.text_input("文件名")
        option1 = left.selectbox("文件类型",("PNG", "PDF"),)
        st.write("选择文件类型:", option1)
    verbose1 = right.checkbox("是否详细显示", key="Grad34")
    if container3.button("运行",icon="😃", use_container_width=True,key=3):
        container3.markdown("开始运行！")