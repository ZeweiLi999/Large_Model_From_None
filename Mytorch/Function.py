from Variable import Variable
#这是基类Function函数，不写明具体用法，继承使用
class Function:
    #接收一个Variable类型的变量作为输入
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self,x):
        raise NotImplementedError()