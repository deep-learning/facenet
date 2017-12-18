import tensorflow as tf

# One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0
# gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive
# initial bias to avoid "dead neurons".
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)

def conv2d(x, W, name=None):
    return tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        data_format='NHWC',
                        name=name)

def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape=shape)
    b = bias_variable(shape[3])
    return tf.nn.relu(conv2d(input, W) + b)

sess = tf.InteractiveSession()
w = weight_variable([12, 100])
tf.global_variables_initializer().run()
print(w.eval())
