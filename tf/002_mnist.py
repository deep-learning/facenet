import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

DATA_PATH = os.path.expanduser('~/data/mnist')
mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

# A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we
#  convert that evidence into probabilities.

# None means that a dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# unstable??
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
# Before Variables can be used within a session, they must be initialized using that session.
tf.global_variables_initializer().run()
# Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent.
# Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what
#  we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap
# and has much of the same benefit.
for i in range(900):
    batch_xs, batch_ys = mnist.train.next_batch(10000)
    # you can replace any tensor in your computation graph using feed_dict -- it's not restricted to just placeholders.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy={}, cross_entropy={}'.format(
            sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}),
            sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
        ))
        # 1 2 3
        # 4 5 6
