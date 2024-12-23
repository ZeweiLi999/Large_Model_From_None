if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import streamlit as st

st.set_page_config(page_title="计算图可视化", page_icon="🔠")

intro_1 = '''
#### :star: 知识学习

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
### :star: 可视化的加深理解

下面是一些简单的例子，对计算图可视化的具体实现。'''

intro_13 = '''
### :star: 动手试一试

LearnTouch作为自制深度学习框架，提供了ShowGradGraph文件，文件可支持将计算图转化为DOT语言格式，高效快速的实现对计算图的可视化。'''

# intro_10 = '''图中的变量a，它是在计算过程中出现的变量。通过上一个步骤可知，对于重复使用的同一变量，我们需要在反向传播时加上从输出端传来的导数。因此，要想求出$a$的导数，就要使用从$a$的输出端传来的两个导数。这两个导数传播出去之后，导数就可以从$a$向$x$传播了。
# 因此，反向传播的流程就如图所示'''
# intro_11 = '''上图是由变量$y$向$x$传播导数的流程。在向变量a传播两个导数之后，从a向$x$传播导数，也就是反向传播按照D、B、C、A或D、C、B、A的顺序进行。
# 在进行函数A的反向传播之前，要先完成函数B和函数C的反向传播。在这个基础上，通过优化算法，我们可以实现更多复杂的计算图。
#
# 我们通过Graphviz实现的计算图的可视化，帮助你更好的理解与学习深度学习的内容！'''

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
st.markdown(intro_1)
st.image("./imgs/GraphVisual/functions_computation_graph.png")
st.markdown(intro_2)
st.image("./imgs/GraphVisual/output_to_input.png")
st.markdown(intro_3)
st.image("./imgs/GraphVisual/o_t_i_graph.png")
st.markdown(intro_4)
st.image("./imgs/GraphVisual/o_t_i_computation_graph.png")
st.markdown(intro_5)
st.image("./imgs/GraphVisual/zhengfan.png")
st.markdown(intro_6)
st.markdown(intro_7)
st.image("./imgs/GraphVisual/bizhide_jisuantu.png")
st.markdown(intro_8)
st.image("./imgs/GraphVisual/fuzadejisuantu1.png")
st.markdown(intro_9)
# st.image("./imgs/GraphVisual/cuwudejisuantu.png")
# st.markdown(intro_10)
# st.image("./imgs/GraphVisual/fanxiangchuanbodeshunxu.png")
# st.markdown(intro_11)


st.markdown(intro_12)

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
st.markdown(intro_13)
st.image("./imgs/GraphVisual/微信图片_20241223201713.png")

