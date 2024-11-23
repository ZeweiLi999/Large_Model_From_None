if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from TestUnit2 import sphere,matyas,goldstein
from LearnTorch import Variable
from LearnTorch.utils import plot_dot_graph

if __name__ == "__main__":
    x = Variable(np.array([[1.0]]))
    y = Variable(np.array([[1.0]]))
    z = sphere(x, y) 
    z.backward() 
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, verbose=True, to_file='sphere.png', file_path="../TeachImage/CGMap")

    z.cleargrad()
    z = goldstein(x, y) 
    z.backward() 
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, verbose=True, to_file='goldstein.png', file_path="../TeachImage/CGMap")

    z.cleargrad()
    z = matyas(x, y) 
    z.backward() 
    x.name = 'x' 
    y.name = 'y' 
    z.name = 'z' 
    plot_dot_graph(z, verbose=True, to_file='matyas.png', file_path="../TeachImage/CGMap")

