import unittest
import numpy as np
from Core.Variable import Variable
from Core.Square import Square,square

#数值微分，就是用很小值近似导数，结果准备用于反向传播的测试
def numerical_diff(f,x,eps = 1e-5):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

#测试类，测试Square的正向传播和反向传播是否正确
class SquareTest(unittest.TestCase):
    #正向传播的测试
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEquals(y.data,expected)

    #反向传播的手动测试
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0) #平方的导数是2 * x
        self.assertEquals(x.grad,expected)

    #利用数值微分的反向传播自动测试
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) #np.random.rand() 函数用于生成指定形状的数组，数组中的元素是[0, 1)区间内的均匀分布的随机数。
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square,x)
        flg = np.allclose(x.grad,num_grad) #判断数值微分的梯度和x的梯度是否足够接近
        self.assertTrue(flg)
