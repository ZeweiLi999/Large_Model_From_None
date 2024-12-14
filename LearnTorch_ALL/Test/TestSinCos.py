if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import LearnTorch.Functions as F
from LearnTorch import Variable

if __name__ == "__main__":
    x = Variable(np.array([np.pi/4, np.pi/4]))
    y = F.sin(x)
    y.backward()
    print("y.data", y.data)
    print("x.grad", x.grad)

    x1 = Variable(np.array(np.pi / 4))
    y1 = F.my_sin(x1)
    y1.backward()
    print("y1.data", y1.data)
    print("x1.grad", x1.grad)

    x2 = Variable(np.array(np.pi / 6))
    y2 = F.sin(x2)
    y2.backward(create_graph = True)
    print("x2.grad -> sin(pi/6) 的1阶导数:{}".format(x2.grad))
    print("x2.grad.data -> sin(pi/6) 的1阶导数:{}".format(x2.grad.data))
    for i in range(2,10):
        x_grad = x2.grad # x2.grad.data 就是导数
        x2.cleargrad()
        x_grad.backward(create_graph = True) # create_graph = True代表允许高阶导数
        print("x2.grad -> sin(pi/6) 的{}阶导数:{}".format(i, x2.grad)) # Variable格式
        print("x2.grad.data -> sin(pi/6) 的{}阶导数:{}".format(i, x2.grad.data)) # data是np标量

