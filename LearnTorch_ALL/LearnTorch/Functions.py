import math
import numpy as np
from LearnTorch import Function

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

