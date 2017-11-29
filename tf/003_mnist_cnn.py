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


sess = tf.InteractiveSession()
w = weight_variable([12, 100])
tf.global_variables_initializer().run()
print(w.eval())
