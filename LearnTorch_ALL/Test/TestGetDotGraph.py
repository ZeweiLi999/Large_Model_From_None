if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from TestUnit2 import sphere,matyas,goldstein
from LearnTorch.Functions import my_sin
from LearnTorch import Variable
from LearnTorch.utils import plot_dot_graph

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y) 
    z.backward(retain_grad=True)
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, savefile=True, verbose=False, to_file='sphere.png', file_path="../TeachImage/CGMap")

    z.cleargrad()
    z = goldstein(x, y) 
    z.backward(retain_grad=True)
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, savefile=True, verbose=False, to_file='goldstein.png', file_path="../TeachImage/CGMap")

    z.cleargrad()
    z = matyas(x, y) 
    z.backward(retain_grad=True)
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, savefile=True, verbose=False, to_file='matyas.png', file_path="../TeachImage/CGMap")

    z.cleargrad()
    x1 = Variable(np.array(np.pi / 4))
    z = my_sin(x1)
    z.backward()
    x1.name = 'x1'
    z.name = 'z'
    plot_dot_graph(z, savefile=True, verbose=False, to_file='taylorsin.png', file_path="../TeachImage/CGMap")

