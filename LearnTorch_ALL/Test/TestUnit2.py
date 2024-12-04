import unittest
import numpy as np
from LearnTorch import Variable
from Numberical_diff import numerical_diff_twoinput

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2 ) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


class ComplexFunctionTest(unittest.TestCase):
    #利用数值微分的结果 测试复杂计算反向传播的导数是否正确
    def test_gradient_check(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z_sphere = sphere(x, y)
        z_sphere.backward()
        gx = x.grad
        gy = y.grad
        sphere_num_x_grad, sphere_num_y_grad = numerical_diff_twoinput(sphere, x, y)
        flg_sphere_x = np.allclose(gx.data, sphere_num_x_grad) #判断数值微分的导数和LearnTorch反向传播x的导数是否足够接近
        flg_sphere_y = np.allclose(gy.data, sphere_num_y_grad)
        self.assertTrue(flg_sphere_x)
        self.assertTrue(flg_sphere_y)

        #清理梯度，开始matyas函数的验证
        x.cleargrad()
        y.cleargrad()
        z_matyas = matyas(x, y)
        z_matyas.backward()
        gx = x.grad
        gy = y.grad
        matyas_num_x_grad, matyas_num_y_grad = numerical_diff_twoinput(matyas, x, y)
        flg_matyas_x = np.allclose(gx.data, matyas_num_x_grad)  # 判断数值微分的导数和LearnTorch反向传播x的导数是否足够接近
        flg_matyas_y = np.allclose(gy.data, matyas_num_y_grad)
        self.assertTrue(flg_matyas_x)
        self.assertTrue(flg_matyas_y)

        #清理梯度，开始goldstein函数的验证
        x.cleargrad()
        y.cleargrad()
        z_goldstein = goldstein(x, y)
        z_goldstein.backward()
        gx = x.grad
        gy = y.grad
        goldstein_num_x_grad, goldstein_num_y_grad = numerical_diff_twoinput(goldstein, x, y)
        flg_goldstein_x = np.allclose(gx.data, goldstein_num_x_grad)  # 判断数值微分的导数和LearnTorch反向传播x的导数是否足够接近
        flg_goldstein_y = np.allclose(gy.data, goldstein_num_y_grad)
        self.assertTrue(flg_goldstein_x)
        self.assertTrue(flg_goldstein_y)