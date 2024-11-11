import numpy as np
from Variable import Variable
from Function import Function

#通过继承Function实现了平方函数

class Square(Function):
    def forward(self,x):
        return x**2

if __name__ == '__main__':
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    #打印y的类型和y的数值
    print(type(y))
    print(y.data)
