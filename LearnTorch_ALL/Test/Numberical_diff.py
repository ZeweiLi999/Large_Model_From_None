from LearnTorch import Variable
from LearnTorch import as_array

#数值微分，就是用很小值近似导数，结果准备用于反向传播的测试

#输入都要加上as_array，是因为numpy零维度不会是ndarray，只是数字，所以要给它封装成as_array
def numerical_diff(f,x,eps = 1e-5):
    #函数f,输入x
    #eps是极小值参数，默认为1e-5
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def numerical_diff_twoinput(f,x,y,eps = 1e-5):
    #eps是极小值参数，默认为1e-5
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = Variable(as_array(y.data - eps))
    y1 = Variable(as_array(y.data + eps))

    #计算y的差分，把其他变量视作常量
    zysub = f(x, y0)
    zyadd = f(x, y1)
    zy = (zyadd.data - zysub.data) / (2 * eps)

    zxsub = f(x0, y)
    zxadd = f(x1, y)
    zx = (zxadd.data - zxsub.data) / (2 * eps)
    return zx, zy