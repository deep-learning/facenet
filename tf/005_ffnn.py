import numpy as np
from .comm import sigmoid

# 5 samples each with 3 features
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1],
              [1, 1, 1]])
print(x.shape)

y = np.array([[0],
              [1],
              [1],
              [0],
              [0]])
print(y.shape)

# re-produceable random values
np.random.seed(1)

# L0 -(W0)-> L1 -(W1)-> L2

W0 = 2 * np.random.random((3, 4)) - 1  # (-1, 1)
W1 = 2 * np.random.random((4, 1)) - 1  # (-1, 1)

print(W0)

for j in range(60000):
    L0 = x
    L1 = sigmoid(np.dot(L0, W0))
    L2 = sigmoid(np.dot(L1, W1))

    # Loss/goal function: 1/(2 * (y - y^)^2 ' = y - y^

    L2_error = L2 - y
    # L2_error.shape -> (5,1)
    if j % 10000 == 0:
        print("Error={}".format(np.mean(np.abs(L2_error))))

    L2_delta = L2_error * sigmoid(L2, deriv=True)
    # L2_delta.shape -> (5,1)
    L1_error = L2_delta.dot(W1.T)
    print(L1_error.shape)
    L1_delta = L1_error * sigmoid(L1, deriv=True)

    # update
    W1 -= L1.T.dot(L2_delta)
    W0 -= L0.T.dot(L1_delta)
