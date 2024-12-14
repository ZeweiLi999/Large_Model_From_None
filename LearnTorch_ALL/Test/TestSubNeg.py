if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch.VariableFunction import Variable,sub,neg


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    y0 = neg(x0)
    print("x0", x0)
    y0.backward()
    print("y0.data", y0.data)
    print("x0.grad", x0.grad)

