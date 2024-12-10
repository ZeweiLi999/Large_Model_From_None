if '__file__' in globals():
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Layers as L
from LearnTorch import Variable, Parameter


if __name__ == "__main__":
    x = Variable(np.array(1.0))
    p = Parameter(np.array(2.0))
    y = x * p

    # 可以用isinstance来分别Variable和Parameter
    print((p, Parameter))
    print((x, Parameter))
    print((y, Parameter))

    layer = L.Layer()