import math
import numpy as np
from LearnTorch import Function,utils,as_variable

# Sin函数
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs # 加逗号，是为了解包操作
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

# 用泰勒展开近似的sin函数
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i +1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs # 加逗号，是为了解包操作
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


# Exp函数，就是调用np.exp,但是封装成Variable
class Exp(Function):
    def forward(self,x):
        return np.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

#通过继承Function实现了平方函数
class Square(Function):
    def forward(self,x):
        return x**2

    def backward(self,gy):
        #平方函数只有一个ndarray，但是参数改成了inputs，取出第一个ndarray
        x = self.inputs[0].data
        #2*x是x的平方的导数
        gx = 2 * x * gy
        return gx
def square(x):
    return Square()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # 弱引用
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape # 保留输入变量的shape，后面的backward要用
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x) # 前向传播调用

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes    # 转置的轴顺序
    def forward(self, x):
        y = x.transpose(self.axes) # 指定轴转置，参照Function的__call__，会取出data来运算，所以x是ndarray，这里调用的是numpy的transpose
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy) # 二阶张量转置再转置回去，就是逆转置了

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        # 取余数是为了处理负轴索引的情况 ax = -1 → -1 % 3 = 2
        # np.argsort中返回数组元素排序后对应索引
        # np.argsort用于计算逆轴顺序，即反向传播时所需的逆转置。
        # 例如，如果self.axes = (2, 0, 1)，逆转置应该是(1, 2, 0)。
        return transpose(gy, inv_axes)

def transpose(x, axes=None): # 调用前向传播
    return Transpose(axes)(x)

class Sum(Function):    # 张量求和函数，输出是一个标量
    def __init__(self, axis, keepdims):
        # axix沿哪个维度求和，keepdims选择是否保留维度s
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape # 反向传播要将输出梯度形状变为输入变量的形状
        y = x.sum(axis=self.axis, keepdims=self.keepdims) # 这里调用的是numpy.sum()，因为是取出data
        return y

    def backward(self, gy):
        # 调整gy梯度形状的与输入变量一致，特殊情况排除，只调整维度，不改动值
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        # 广播，改动值，复制gy的元素
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    # 默认全部轴求和，不保留维度，返回标量
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape: # shape一样就不用广播了
        return as_variable(x)
    return BroadcastTo(shape)(x)

# 深度学习版的sumto，带有反向传播
# sum_to和broad_cast反向传播相互依赖，但是正向传播是独立的
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        # 一维是向量内积，高维是矩阵乘法 (a x b) x (b x c) = (a x c)
        y = x.dot(W) # x.dot，numpy类型实例也能用
        return y

    def backward(self, gy):
        # 将梯度形状变为输入变量的形状
        x, W = self.inputs
        gx = matmul(gy, W.T)  # (a x c)x(c x b) = (a x b)
        gW = matmul(x.T, gy)  # (b x a)x(a x c) = (b x c)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)



class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b = None):
    return Linear()(x, W, b)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)