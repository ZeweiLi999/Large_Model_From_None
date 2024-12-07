if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
import matplotlib.pyplot as plt
from LearnTorch import Variable

if __name__ == "__main__":
    x2 = Variable(np.linspace(0, 10, 200))
    y2 = x2 ** 4
    y2.backward(create_graph = True)
    logs = [y2.data]

    for i in range(3):
        logs.append(x2.grad.data) # !!!重要，图记录的是导数的值
        x_grad = x2.grad
        x2.cleargrad()
        x_grad.backward(create_graph = True) # create_graph = True代表允许高阶导数

    labels = ["y=x**4", "y'=4x**3", "y''=12x**2", "y'''=24x"]
    for i , v in enumerate(logs):
        plt.plot(x2.data, logs[i], label=labels[i])
    plt.legend(loc = "upper center")
    plt.title("[0,10] y=x**4  Grad")
    plt.savefig("./Grad/0_10_yx4.png")
    plt.show()