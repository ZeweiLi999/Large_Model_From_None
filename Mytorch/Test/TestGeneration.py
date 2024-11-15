import numpy as np
from Core.Variable import Variable
from Core.Function import Function
from Core.Square import square
from Core.Add import add

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a),square(a))
    y.backward()

    print(y.data)
    print(x.grad)
    #y = 8 * x**3