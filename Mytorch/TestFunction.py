#用于测试各函数的文件
import numpy as np
from Variable import Variable
from Square import Square
from Exp import Exp

if __name__ == "__main__":
    A = Exp()
    B = Square()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    #assert用法：如果后面的语句不为ture，就会抛出异常
    assert y.creator == C #y的创造函数是C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x



