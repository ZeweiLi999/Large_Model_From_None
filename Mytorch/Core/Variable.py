import numpy as np

class Variable:#定义深度学习的变量类
    def __init__(self, data):
        #鲁棒性检测，检测输入的是否是ndarray
        if data is not None:
            if not isinstance(data , np.ndarray):
                raise TypeError("{} is not supported\nOnly support ndarray".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None #变量和函数的连接，创建自己的函数

    def set_creator(self,func):#获取创建自己的函数
        self.creator = func

    def backward(self):
        if self.grad is None:       #如果没有梯度，就不用在外部创建梯度了
            self.grad = np.ones_like(self.data)

        funcs = [self.creator] #1.创造循环的函数列表
        while funcs:
            f = funcs.pop()     #2.获取变量的创造函数
            x,y = f.input,f.output  #3.获取函数的输入和输出
            x.grad = f.backward(y.grad)  #4.将反向传播的链条推进一步
            if x.creator is not None:
                funcs.append(x.creator)
        #如果没有creator，反向传播到此结束

if __name__ == '__main__':
#通过numpy.array测试
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(2.0)
    print(x.data)

