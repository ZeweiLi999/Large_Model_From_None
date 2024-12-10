# __file__ 是一个特殊变量，指向当前脚本文件的路径。
# 如果当前代码是从一个文件中运行的，IDE中运行也会，__file__ 会存在。
# 如果代码是在交互式环境（如 Python REPL 或 Jupyter Notebook）中运行，__file__ 不存在。因此，这个检查是为了确保代码只在脚本文件中执行。
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch import Variable
import LearnTorch.Functions as F

if __name__ == "__main__":
    # 一维求和
    x = Variable(np.array([1,2,3,4,5,6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    # 二维求和
    x2 = Variable(np.array([[1,2,3], [4,5,6]]))
    y2 = F.sum(x2)
    y2.backward()
    print(y2)
    print(x2.grad)

    # 指定维度求和
    x3 = Variable(np.array([[1,2,3], [4,5,6]]))
    y3 = F.sum(x3, axis=0)
    y3.backward(retain_grad = True)
    print("axis = 0")
    print("y3:", y3)
    print("y3.grad.shape:{} -> x3.grad.shape:{}".format(y3.grad.shape, x3.grad.shape))

    # 保留维度求和，输出不再是标量，而是和输入维度一样
    x4 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y4 = F.sum(x4, keepdims=True)
    y4.backward(retain_grad = True)
    print("keepdims = True")
    print("y4:", y4)
    print("y4.grad.shape:{} -> x4.grad.shape:{}".format(y4.grad.shape, x4.grad.shape))

    # 测试Variable的sum方法
    x5 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y5 = x5.sum()
    y5.backward(retain_grad = True)
    print("y5:", y5)
    print("y5.grad.shape:{} -> x5.grad.shape:{}".format(y5.grad.shape, x5.grad.shape))