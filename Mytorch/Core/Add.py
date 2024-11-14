import numpy as np
from .Variable import Variable
from .Function import Function

class Add(Function):
    def forward(self,x0,x1):
        y = x0 + x1
        return y    #返回的是Variable

    def backward(self,gy):  #返回的两个偏导数都是导数是（1*输入的导数）
        return gy,gy


def add(x0, x1):
    return Add()(x0, x1)

if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)