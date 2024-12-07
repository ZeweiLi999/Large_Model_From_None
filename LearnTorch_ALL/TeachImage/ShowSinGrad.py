if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
import matplotlib.pyplot as plt
from LearnTorch import Variable

if __name__ == "__main__":
    x2 = Variable(np.linspace(0, 2*np.pi, 200))
    y2 = F.sin(x2)
    y2.backward(create_graph = True)
    logs = [y2.data]

    for i in range(3):
        logs.append(x2.grad.data) # !!!重要，图记录的是导数的值
        x_grad = x2.grad
        x2.cleargrad()
        x_grad.backward(create_graph = True) # create_graph = True代表允许高阶导数

    labels = ["y=sin(x)", "y'=cos(x)", "y''=-sin(x)", "y'''=-cos(x)"]
    for i , v in enumerate(logs):
        plt.plot(x2.data, logs[i], label=labels[i])
    plt.legend(loc = "upper center")
    plt.title("[0,2π] Sin(x) Grad")
    plt.savefig("./Grad/0_2pi_singrad.png")
    plt.show()