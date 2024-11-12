import numpy as np

class Variable:#定义深度学习的变量类
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None #变量和函数的连接，创建自己的函数

    def set_creator(self,func):#获取创建自己的函数
        self.creator = func

    def backward(self):
        f = self.creator #1.获取变量的创造函数
        if f is not None:
            x = f.input  #2.获取函数的输入
            x.grad = f.backward(self.grad)  #3.调用函数的backward，将链条推进一步
            x.backward()                    #递归调用前一步变量的backward，将链条不断推进
        #如果没有creator，反向传播到此结束

if __name__ == '__main__':
#通过numpy.array测试
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(2.0)
    print(x.data)

