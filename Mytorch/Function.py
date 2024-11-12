from Variable import Variable
#这是基类Function函数，不写明具体用法，继承使用
class Function:
    #接收一个Variable类型的变量作为输入
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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


