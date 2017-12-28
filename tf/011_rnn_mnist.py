import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# todo http://colah.github.io/posts/2015-08-Understanding-LSTMs/

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets(os.path.expanduser('~/data/mnist'),
                                  one_hot=True)

learning_rate = 0.001
batch_size = 128

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def RNN(x, n_steps, n_input, n_hidden, n_classes):
    # parameters
    # input gate: input, previous input, and bias
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], mean=-0.1, stddev=0.1))
    ib = tf.Variable(tf.zeros([1, n_hidden]))

    # forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, n_hidden]))

    # memory cell, input, state, and bias
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, n_hidden]))

    # output gate: input, previous output and bias
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, n_hidden]))

    # classisifer weights and bias
    w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # definition of the cell computation
    def lstm_cell(i, prev_output, prev_state):
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(prev_output, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(prev_output, fm) + fb)
        update = tf.tanh(tf.matmul(i, cx) + tf.matmul(prev_output, cm) + cb)
        state = forget_gate * prev_state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(prev_output, om) + ob)
        return output_gate * tf.tanh(state), state

    # unrolled ltsm loop
    outputs = []
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    output = tf.Variable(tf.zeros([batch_size, n_hidden]))

    # x shape: (batch_size, n_steps, n_input)
    # -> list of n_steps with element shape ( batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps, axis=0)
    for i in x:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    logits = tf.matmul(outputs[-1], w) + b
    return logits

pred = RNN(x, n_steps, n_input, n_hidden, n_classes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

sess.run(init)
for step in range(20000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    if step % 50 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print('iter:{},batch loss={:.6f},train acc={:.6f}'.format(step, loss, acc))

# test
test_len = batch_size
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
print('test acc={:.6f}'.format(sess.run(accuracy, feed_dict={x: test_data, y: test_label})))
