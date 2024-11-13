import numpy as np
from .Variable import Variable

def as_array(x):
    #写这个函数是因为numpy特性不一定返回ndarray，也可能返回一个标量，要通过检测
    if np.isscalar(x):#检测是否是标量类型
        return np.array(x) #是标量，就转化成np.array再返回
    return x

#这是基类Function函数，不写明具体用法，继承使用
class Function:
    #接收一个Variable类型的变量作为输入
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)   #让输出变量保留创造者信息
                                    #这句是关键，是把Function对象传入记录了
        self.input = input #保存输入的变量，可在反向传播中调用
        self.output = output #保留输出变量
        return output

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,gy):
        #输入：反向传播链条中上一步传播而来的导数乘积
        #输出：反向传播链条中进一步传播的导数乘积
        raise NotImplementedError()


