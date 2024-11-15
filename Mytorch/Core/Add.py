import numpy as np
from .Variable import Variable
from .Function import Function

class Add(Function):
    def forward(self,x0,x1):
        y = x0 + x1
        return y    #返回的是Variable

    def backward(self,gy):  #返回的两个偏导数都是导数是（1*输入的导数）
        gx = 1 * gy
        return gx, gx


def add(x0, x1):
    return Add()(x0, x1)