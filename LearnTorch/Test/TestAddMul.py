import numpy as np
from Core.VariableFunction import Variable,add,mul


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    y0 = add(add(x0, x0), x0)
    y0.backward()
    print("y0.data", y0.data)
    print("x0.grad", x0.grad)

    x0.cleargrad()
    y1 = x0 + x0 +x0
    y1.backward()
    print("y1.data", y1.data)
    print("x0.grad", x0.grad)

    x0.cleargrad()
    y0 = mul(mul(x0, x0), x0)
    y0.backward()
    print("y0.data", y0.data)
    print("x0.grad", x0.grad)

    x0.cleargrad()
    y1 = x0 * x0 * x0
    y1.backward()
    print("y1.data", y1.data)
    print("x0.grad", x0.grad)