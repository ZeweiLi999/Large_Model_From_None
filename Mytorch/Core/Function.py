import numpy as np
from .Variable import Variable

def as_array(x):
    #写这个函数是因为numpy特性不一定返回ndarray，也可能返回一个标量，要通过检测
    if np.isscalar(x):#检测是否是标量类型
        return np.array(x) #是标量，就转化成np.array再返回
    return x

#这是基类Function函数，不写明具体用法，继承使用
class Function:
    #接收多个Variable类型的变量作为输入
    def __call__(self,*inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) #使用*号解包成单独的参数，传递给函数
        if not isinstance(ys,tuple):    #对于非元组情况的额外处理
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)#让输出变量保留创造者信息
                                    #这句是关键，是把Function对象传入记录了
        self.inputs = inputs        #保存输入的变量，可在反向传播中调用
        self.outputs = outputs      #保留输出变量
        return outputs if len(outputs) > 1 else outputs[0]
        #如果列表中只有一个元素，则返回第一个元素，而非列表

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,gy):
        #输入：反向传播链条中上一步传播而来的导数乘积
        #输出：反向传播链条中进一步传播的导数乘积
        raise NotImplementedError()




