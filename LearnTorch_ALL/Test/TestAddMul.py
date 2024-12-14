if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from LearnTorch.VariableFunction import Variable,add,mul


if __name__ == "__main__":
    print("不保留中间变量导数")
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print("y.grad, t.grad", y.grad, t.grad)
    print("x0.grad, x1.grad", x0.grad, x1.grad)

    print("保留中间变量导数")
    x0.cleargrad()
    x1.cleargrad()
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward(retain_grad = True)
    print("y.grad, t.grad", y.grad, t.grad)
    print("x0.grad, x1.grad", x0.grad, x1.grad)

    print("乘法测试")
    x0.cleargrad()
    x1 = add(x0, x0)
    x2 = add(x0, x0)
    y1 = mul(x1, x2)
    y1.backward()
    print("x1.grad, x2.grad", x1.grad, x2.grad)
    print("x0.grad", x0.grad)

