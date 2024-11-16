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
        self.generation = 0 #用于决定反向传播路径的顺序,初始化为0

    def cleargrad(self):
        self.grad = None    #清除导数，使得可利用同一个变量求出不同计算的导数
                            #还可以用来解决优化问题
    def set_creator(self,func):#获取创建自己的函数
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:       #如果没有导数，就不用在外部创建梯度了
            self.grad = np.ones_like(self.data)

        funcs = [] #1.创造循环的函数列表
        seen_set = set() #用去去重的集合

        def add_func(f):
            if f not in seen_set: #不在去重集合里，则可以加入函数列表
                                    #为什么不用if not in列表查重，还是空间换时间，因为集合是用哈希表实现的，所以速度会比列表查重更快
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x:x.generation) #函数从小到大排列

        add_func(self.creator)
        while funcs:
            f = funcs.pop()     #2.获取变量的创造函数
            gys = [output().grad for output in f.outputs]#3.获取函数的输出的所有导数,() 是 weakref.ref 对象的调用方法，用于返回被弱引用的对象。
            gxs = f.backward(*gys) #3.将反向传播的链条推进一步，获取所有输入的导数
            if not isinstance(gxs,tuple):
                #改为元组类型便于后面的zip循环
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  #4.更新变量的梯度
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx #这样会新创建内存空间，+=会原地操作

                #记住要放在循环内，不断反向传播
                if x.creator is not None:
                    add_func(x.creator)
                #如果没有creator，反向传播到此结束


