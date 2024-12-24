if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import streamlit as st

st.set_page_config(page_title="计算图可视化", page_icon="🔠")

intro_0 ='''
### :star:1.理论学习
'''
intro_1 = '''
导数广泛应用在现代科学技术的各个领域，尤其在包括深度学习在内的
机器学习的各个领域，导数起着核心作用。 从某种意义上来说， 深度学习框
架就是计算导数的工具。 因此，求导，即**自动微分**是深度学习框架中的重要部分。 这里所说的自动微分指的是由计算机(而不是人)来计算导数。
具体来说， 就是指在对某个计算(函数)编码后， 由计算机自动求出该计算的导数的机制。
###### 反向传播
在求导方式的选择上，我们通常可以选择数值微分。但是，数值微分在计算成本和精度方面存在问题。 **反向传播**可以解决这两个问题。

理解反向传播的关键是链式法则(连锁律)。链式法意为连接起来的多
个函数(复合函数)的导数可以分解为各组成的数的导数的乘积。换言之，复合函数的导数可以分解为各组成函数导数的乘积，这就是链式法则。

复合函数的导数可以分解为各函数导数的乘积。但是，它并没有规定各导数相乘的顺序,所以这一点我们可以自由决定。

举一个简单的例子，假设有一个函数$ y=F(x)$,这个函数F由三个函数组成$a=A(x), b=B(a)和y=C(b)$，该函数的计算图如图所示
'''
intro_2 = '''它的求导过程如下图'''
intro_3 = '''如果按照从输出$y$到输入$x$的方向依次相乘计算得出导数，相应的计算图如下'''
intro_4 = '''将上图的导函数和乘号合并表示为一个函数节点。这样导数计算的流程就明确了。'''
intro_5 = '''从图可以看出，"$y$对各变量的导数"从右向左传播。传播的数据都是$y$的导数。这就是反向传播。下面我们将正向传播与反向传播的计算图上下排列展现出来。'''
intro_6 = '''从图可以看出，正向传播和反向传播之间存在明确的对应关系。正向传播时的变量a对应于反向传播时的导数 $\\frac{dy}{da}$ , 
这样一来，我们可以认为变量有普通值和导数值，函数有普通计算(正向传播)和求导计算(反向传播)。 于是，反向传播设计好了。


最后来关注一下图中$ C'(b)$ 的函数节点。 
它是$y= C(b)$的导数，但要注意的是，计算$C'(b)$需要用到 $b$ 的值。同理，要计算$B'(a)$就得输入 $a$ 的值，
这意味着进行反向传播时需要用到正向传播中使用的数据。 因此，在实现反
向传播时，需要先进行正向传播，并且存储各函数输入的变量值，也就是前
面例子中的$x、 a和b$， 之后就能对每个函数迸行反向传播的计算了。'''
intro_7 = '''

##### 复杂的计算图可视化

前面我们处理的都是如下图一样的笔直计算图

'''
intro_8 = '''
然而。随着函数的不断复杂，已经不局限于这种简单的连接方式。
现在我们可以创建更为复杂的计算图了'''

intro_9 = '''
上图所示的计算重复使用了同一个变量，也使用了支持多个变量的函数。
通过这样的方式，可以建立更复杂的“连接”。



'''
intro_12 = '''
### :star:2.可视化加深理解

下面是一些简单的例子，对计算图可视化的具体实现。'''

intro_13_1='''

### :star:3.动手试一试

'''

intro_13 = '''

LearnTouch作为自制深度学习框架，提供了ShowGradGraph文件，文件可支持将计算图转化为DOT语言格式，高效快速的实现对计算图的可视化。

'''

intro_14 = '''
### :star:4.拓展
'''

intro_15 = '''
了解主流深度学习框架如何运用计算图：
- 1.pytorch如何构建计算图
https://pytorch.ac.cn/blog/computational-graphs-constructed-in-pytorch/
'''

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
st.markdown(intro_0)
container0 = st.container(border=True,key=0)
with container0:
    st.markdown(intro_1)
    st.markdown("球体公式计算图可视化")
    st.image("./imgs/1_GraphVisual/functions_computation_graph.png")
    st.markdown(intro_2)
    st.image("./imgs/1_GraphVisual/output_to_input.png")
    st.markdown(intro_3)
    st.image("./imgs/1_GraphVisual/o_t_i_graph.png")
    st.markdown(intro_4)
    st.image("./imgs/1_GraphVisual/o_t_i_computation_graph.png")
    st.markdown(intro_5)
    st.image("./imgs/1_GraphVisual/zhengfan.png")
    st.markdown(intro_6)
    st.markdown(intro_7)
    st.image("./imgs/1_GraphVisual/bizhide_jisuantu.png")
    st.markdown(intro_8)
    st.image("./imgs/1_GraphVisual/fuzadejisuantu1.png")
    st.markdown(intro_9)

st.divider()
st.markdown(intro_12)
container1 = st.container(border=True,key=1)
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


st.divider()
st.markdown(intro_13_1)
container2 = st.container(border=True,key=2)
with container2:
     st.markdown(intro_13)
     st.image("./imgs/1_GraphVisual/showgraph.png")


st.divider()
st.markdown(intro_14)
container3 = st.container(border=True,key=3)
with container3:
    st.markdown(intro_15)
    st.image("./imgs/1_GraphVisual/pytorch_graph.png")
