import numpy as np
from LearnTorch import Function


#就是调用np.exp,但是封装成Variable
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

