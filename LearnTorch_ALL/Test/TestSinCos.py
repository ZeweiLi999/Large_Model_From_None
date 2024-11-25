if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch import Variable
from LearnTorch.Functions import sin,my_sin

if __name__ == "__main__":
    x = Variable(np.array(np.pi/4))
    y = sin(x)
    y.backward()
    print("y.data", y.data)
    print("x.grad", x.grad)

    x1 = Variable(np.array(np.pi / 4))
    y1 = my_sin(x1)
    y1.backward()
    print("y1.data", y1.data)
    print("x1.grad", x1.grad)