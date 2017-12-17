import numpy as np

# simple activation function
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

  

if __name__ == '__main__':
    pass
