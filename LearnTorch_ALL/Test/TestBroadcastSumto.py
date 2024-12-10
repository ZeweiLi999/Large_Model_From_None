if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch.utils import sum_to
from LearnTorch import Variable

if __name__ == "__main__":
    x = np.array([[1,2,3], [4,5,6]])
    y = sum_to(x, (1, 3))
    print(y)

    y = sum_to(x, (2, 1))
    print(y)

    # 测试正向传播的广播功能
    x0 = Variable(np.array([1,2,3]))
    x1 = Variable(np.array([10]))
    y1 = x0 + x1
    print(y1)

    # 测试反向传播的广播功能
    print("before x1.grad.shape:", x1.grad.shape)
    y1.backward()
    print("after x1.grad.shape:",x1.grad.shape)