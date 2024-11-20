import numpy as np
import weakref
import contextlib


class Config:   #用于决定是否启用反向传播的类
    enable_backprop = True

@contextlib.contextmanager #调用上下文的库
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad(): #封装函数
    return using_config('enable_backprop', False)

class Variable:#定义深度学习的变量类
    __array_priority__ = 200 # 大于0.0，运算符优先级就会大于ndarray
    def __init__(self, data, name = None):
        #鲁棒性检测，检测输入的是否是ndarray
        if data is not None:
            if not isinstance(data , np.ndarray):
                raise TypeError("{} is not supported\nOnly support ndarray".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None #变量和函数的连接，创建自己的函数
        self.generation = 0 #用于决定反向传播路径的顺序,初始化为0

    # numpy的实例变量
    @property   #@property装饰器它将一个方法转换为属性的getter方法，使得我们可以像访问属性一样访问方法。
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    # numpy的实例变量

    # 重载函数
    def __len__(self):
        # 重载len函数，返回第一维度的元素数量
        return len(self.data)

    def __repr__(self):
        # 重载输出函数，分行输出数字会对齐
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'
    # 重载函数

    def cleargrad(self):
        self.grad = None    #清除导数，使得可利用同一个变量求出不同计算的导数
                            #还可以用来解决优化问题
    def set_creator(self,func):#获取创建自己的函数
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:       #如果没有导数，就不用在外部创建了
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

            if not retain_grad:
                for y in f.outputs:
                    #不要保留各函数输出变量的导数
                    y().grad = None #因为y是weakref,必须要用y()访问

#把传来的参数转化为Variable实例
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    #写这个函数是因为numpy特性不一定返回ndarray，也可能返回一个标量，要通过检测
    if np.isscalar(x):#检测是否是标量类型
        return np.array(x) #是标量，就转化成np.array再返回
    return x

#这是基类Function函数，不写明具体用法，继承使用
class Function:
    #接收多个Variable类型的变量作为输入
    def __call__(self,*inputs):
        inputs = [as_variable(x) for x in inputs] #过一遍循环，把参数都转化为Variable
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) #使用*号解包成单独的参数，传递给函数
        if not isinstance(ys,tuple):    #对于非元组情况的额外处理
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop: #如果启用反向传播，才会保留辈分和函数链接
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)#让输出变量保留创造者信息
                                        #这句是关键，是把Function对象传入记录了
            self.inputs = inputs        #保存输入的变量，可在反向传播中调用
            self.outputs = [weakref.ref(output) for output in outputs]      #使用弱引用保留输出变量
        return outputs if len(outputs) > 1 else outputs[0]
        #如果列表中只有一个元素，则返回第一个元素，而非列表

    def forward(self,xs):
        raise NotImplementedError()

    def backward(self,gys):
        #输入：反向传播链条中上一步传播而来的导数乘积
        #输出：反向传播链条中进一步传播的导数乘积
        raise NotImplementedError()


class Add(Function):
    def forward(self,x0,x1):
        y = x0 + x1
        return y    #返回的是Variable

    def backward(self,gy):  #返回的两个偏导数都是导数是（1*输入的导数）
        gx = 1 * gy
        return gx, gx

def add(x0, x1):
    x1 = as_array(x1) #因为self本身肯定是Variable，所以只要针对x1是标量的情况就可以了，是ndarray会在function变为Variable
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        #前向传播
        y = x0 * x1
        return y

    def backward(self, gy):
        gx0, gx1 = self.inputs[0].data * gy, self.inputs[1].data * gy
        return gx0, gx1

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0,x1)

class Neg(Function):
    #取负函数
    def forward(self,x):
        return -x

    def backward(self,gy):
        gx = -1 * gy
        return gx

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self,gy):
        gx0 = gy
        gx1 = -1 * gy
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0,x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1,x0) # 交换x1和x0，因为self是x0，是减数

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self,gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) # 交换x1和x0，因为self是x0，是除数

class Pow(Function):
    def __init__(self, c):
        # 前向传播和后向传播都需要用到指数，所以列为属性
        self.c = c

    def forward(self,x):
        y = x ** self.c
        return y

    def backward(self,gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

#重载Variable的函数
Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
#重载Variable的函数
