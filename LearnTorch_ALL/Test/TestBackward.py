#用于测试各函数的文件
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch import Variable
from LearnTorch.Functions import Square,Exp


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    #手动反向传播的方法，第一个链条
    y.grad = np.array(1.0)
    C = y.creator   #1.获取y的函数
    b = C.inputs[0]   #2.获取C函数的输入
    b.grad = C.backward(y.grad) #3.调用函数的backward方法，获取链条上进一步的输入

    #第二个链条
    B = b.creator   #1.获取b的函数
    a = B.inputs[0]     #2.获取B函数的输入
    a.grad = B.backward(b.grad)#3.反向传播，获取链条进一步输入

    #第三个链条
    A = a.creator   #1.获取a的函数
    x = A.inputs[0]     #2.获取A函数的输入
    x.grad = A.backward(a.grad)#3.最终获取到了y对x的梯度！
                                #同时也得出了反向传播链条上y对各中间变量的梯度
    print(x.grad)
    #完成了所有的反向传播，现在要将这个流程加入到Variable中去。

    #自动反向传播
    x.grad = 0
    y.backward()
    print(x.grad)
