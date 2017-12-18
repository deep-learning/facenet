import numpy as np
import tensorflow as tf

# simple activation function
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def l2_pooling(previous, name=None):
    return tf.sqrt(tf.nn.avg_pool(tf.square(previous)), name=name)

if __name__ == '__main__':
    pass
