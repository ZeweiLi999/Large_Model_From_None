if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 +(x0 - 1) ** 2
    return y

if __name__ == "__main__":
    # (x0.grad, x1.grad)就是梯度，是y的值增加最快的方向
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    lr = 0.001 # 学习率
    iters = 50000 # 迭代次数

    for i in range(iters):
        print(x0, x1)
        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad # 向y减少的最快的方向（因为是-=）移动lr * x0.grad
        x1.data -= lr * x1.grad # 向y减少的最快的方向（因为是-=）移动lr * x1.grad