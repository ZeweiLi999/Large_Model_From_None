import unittest
import numpy as np
from LearnTorch import Variable
from LearnTorch.Functions import Square,square
from Numberical_diff import numerical_diff



#测试类，测试Square的正向传播和反向传播是否正确
class SquareTest(unittest.TestCase):
    #正向传播的测试
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data,expected)

    #反向传播的手动测试
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        gx = x.grad
        expected = np.array(6.0) #平方的导数是2 * x
        self.assertEqual(gx.grad,expected)

    #利用数值微分的反向传播自动测试
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) #np.random.rand() 函数用于生成指定形状的数组，数组中的元素是[0, 1)区间内的均匀分布的随机数。
        y = square(x)
        y.backward()
        gx = x.grad
        num_grad = numerical_diff(square,x)
        flg = np.allclose(gx.grad,num_grad) #判断数值微分的梯度和x的梯度是否足够接近
        self.assertTrue(flg)
