import unittest
import numpy as np
from LearnTorch.VariableFunction import Variable,Function,add
from LearnTorch.Functions import square


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
        gx = x.grad
        expected = np.array(64.0)
        self.assertEqual(gx.data,expected)