import math
import unittest
import numpy as np
from LearnTorch import Variable



#测试类，测试高阶导数的计算结果是否正确
def f(x):
    y = x ** 20
    return y
class HigherGradTest(unittest.TestCase):
    #利用math库的反向传播自动测试
    def test_high_grad(self):
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        for i in range(20):
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
        result = gx.data
        expected = math.factorial(20)
        flg = np.allclose(expected, result) #判断自动求导和高阶导数的数值计算结果是否足够接近
        self.assertTrue(flg)
