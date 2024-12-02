if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch import Variable

def f(x):
    return x ** 4 - 2 * x ** 2

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)