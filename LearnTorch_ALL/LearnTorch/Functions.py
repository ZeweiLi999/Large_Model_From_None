import math
import numpy as np
from LearnTorch import Function
from LearnTorch.VariableFunction import as_variable

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

    def backward(self,gy):
        x = self.inputs[0].data
        #np.exp(x)是导数
        gx = np.exp(x) * gy
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