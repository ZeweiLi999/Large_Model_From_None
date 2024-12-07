if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch.Functions import my_sin
from LearnTorch import Variable
from LearnTorch.utils import plot_dot_graph

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2 ) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z



if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y) 
    z.backward(retain_grad=True)
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, save_file=True, verbose=False, to_file='sphere.png', file_path="./CGMap")

    z.cleargrad()
    z = goldstein(x, y) 
    z.backward(retain_grad=True)
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, save_file=True, verbose=False, to_file='goldstein.png', file_path="./CGMap")

    z.cleargrad()
    z = matyas(x, y) 
    z.backward(retain_grad=True)
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, save_file=True, verbose=False, to_file='matyas.png', file_path="./CGMap")

    z.cleargrad()
    x1 = Variable(np.array(np.pi / 4))
    z = my_sin(x1)
    z.backward()
    x1.name = 'x1'
    z.name = 'z'
    plot_dot_graph(z, save_file=True, verbose=False, to_file='taylorsin.png', file_path="./CGMap")

