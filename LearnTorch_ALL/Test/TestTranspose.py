if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
from LearnTorch import Variable

if __name__ == "__main__":
    A, B, C, D = 1, 2, 3, 4
    x = np.random.randint(low = 0, high =10 , size = (A, B, C, D))
    print("before transpose x.shape: \n", x.shape)
    print("before transpose x: \n", x)
    y = x.transpose(1, 0, 3, 2)
    print("after transpose x.shape: \n", y.shape)
    print("after transpose x: \n", y)


