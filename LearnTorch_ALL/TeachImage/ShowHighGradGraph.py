if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
from LearnTorch import Variable
from LearnTorch.utils import plot_dot_graph

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph = True)

    iters = 3

    for i in range(iters):
        gx = x.grad
        gx.name = 'gx' + str(i + 1)
        plot_dot_graph(gx, save_file=True, verbose=False, to_file='ShowTanh_{}.png'.format(i + 1), file_path='CGMap')
        x.cleargrad()
        gx.backward(create_graph = True)
