if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
from LearnTorch import Variable


if __name__ == "__main__":
    x = Variable(np.array([[1,2,3], [4,5,6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad = True)
    print("y.data: ", y)
    print("y.grad: ", y.grad)
    print("x.data: ", x)
    print("x.grad: ", x.grad)

    x = x.reshape((2,3))
    print("x.reshape(2,3): \n", x)

    x = x.reshape([3,2])
    print("x.reshape([3,2]): \n", x)

    x = x.reshape(6,1)
    print("x.reshape(6,1): \n", x)