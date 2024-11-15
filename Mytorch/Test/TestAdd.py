import numpy as np
from Core.Variable import Variable
from Core.Add import add


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    y = add(add(x0, x0), x0)
    y.backward()
    print("y.data", y.data)
    print("x0.grad", x0.grad)

    x0.cleargrad()
    y = add(x0,x0)
    y.backward()
    print("y.data", y.data)
    print("x0.grad", x0.grad)