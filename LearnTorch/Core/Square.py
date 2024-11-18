import numpy as np
from .VariableFunction import Variable,Function

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

if __name__ == '__main__':
    x0 = Variable(np.array([2.0,1.0]))
    y = square(x0)
    y.backward()
    print(x0.grad)
