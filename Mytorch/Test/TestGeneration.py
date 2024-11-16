import unittest
import numpy as np
from Core.Variable import Variable
from Core.Function import Function
from Core.Square import square
from Core.Add import add

#测试类，测试复杂计算图的正向传播和反向传播是否正确
class SquareTest(unittest.TestCase):
    # 正向传播的测试
    def test_forward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        expected = np.array(32.0)
        self.assertEqual(y.data, expected)
    #反向传播的手动测试
    def test_backward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        expected = np.array(64.0)
        self.assertEqual(x.grad,expected)