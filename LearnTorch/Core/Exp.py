import numpy as np
from .VariableFunction import Variable,Function

#就是调用np.exp,但是封装成Variable
class Exp(Function):
    def forward(self,x):
        return np.exp(x)

    def backward(self,gy):
        x = self.input.data
        #np.exp(x)是导数
        gx = np.exp(x) * gy
        return gx

def exp():
    return Exp()(x)

if __name__ == '__main__':
    x = Variable(np.array(20))
    f = Exp()
    y = f(x)
    #打印y的类型和数值
    print(type(y))
    print(y.data)