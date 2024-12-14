if '__file__' in globals():
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
from LearnTorch import Variable


if __name__ == "__main__":
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W) # 矩阵乘法
    y.backward()
    print("x.shape{} x W.shape{}".format(x.shape, W.shape))
    print("y.shape", y.shape)
    print("x.grad.shape", x.grad.shape)
    print("W.grad.shape", W.grad.shape)
