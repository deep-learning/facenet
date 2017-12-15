import sys
import tensorflow as tf

def my_network(input):
    W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1), name="W_1")
    b_1 = tf.Variable(tf.zeros([100]), name="biases_1")
    output_1 = tf.matmul(input, W_1) + b_1
    W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1),
        name="W_2")
    b_2 = tf.Variable(tf.zeros([50]), name="biases_2")
    output_2 = tf.matmul(output_1, W_2) + b_2
    W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1), name="W_3")
    b_3 = tf.Variable(tf.zeros([10]), name="biases_3")
    output_3 = tf.matmul(output_2, W_3) + b_3
    # printing names
    print("Printing names of weight parameters")
    print(W_1.name, W_2.name, W_3.name)
    print("Printing names of bias parameters")
    print(b_1.name, b_2.name, b_3.name)
    return output_3

input_1 = tf.placeholder(dtype=tf.float32, shape=[1000, 784], name='input_1')
my_network(input_1)
input_2 = tf.placeholder(dtype=tf.float32, shape=[10000, 784], name='input_2')
my_network(input_2)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

sys.exit(0)

weights = tf.Variable(
    tf.random_normal([300, 200], stddev=0.5),
    name='weights',
    trainable=False
)

V1 = tf.constant([1., 2.])
V2 = tf.constant([3., 4.])
M = tf.constant([[1., 2.]])  # Matrix, 2d
N = tf.constant([[1., 2.], [3., 4.]])  # Matrix, 2d
K = tf.constant([[[1., 2.], [3., 4.]]])  # Tensor, 3d

x = tf.placeholder(tf.float32, name='x', shape=[None, 784])
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
f = tf.matmul(x, W) + b

## broadcast
a1 = tf.Variable(tf.ones([4, 10], name='3x10_1'))
a2 = tf.Variable(tf.ones([10], name='10_1'))
a3 = tf.Variable(tf.ones([10, 1], name='10x1_1'))

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(a1 + a2))
    print(sess.run(a2))
    print(sess.run(a3))
    print(sess.run(a2 + a3))  # broadcast
    print(sess.run(weights))
    k = sess.run(K)
    print(sess.run(M * M))
    print(sess.run(tf.multiply(M, M)))
    print(sess.run(tf.matmul(M, N)))

sys.exit(0)

print('*' * 30)
sess = tf.InteractiveSession()
print(M.eval())
print((M * M).eval())

W = tf.Variable(0, name="weight")
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(W.eval())  # 0

a = tf.constant(1)
W += a
print(W.eval())

print(sess.run([W, a]))

c = tf.constant(12)
d = tf.Variable(2)
b = tf.Variable(3)
E = d + b + c
sess.run(tf.global_variables_initializer())
print(E.eval())

# We can use feed_dict to supply a custom value to a node anywhere in our
# computation when returning a value.
print("E with customize intermediate var d as 4: ", sess.run(E, feed_dict={d: 4.}))
print("E with customize intermediate const var as 4: ", sess.run(E, feed_dict={c: 100.}))

with tf.Session() as sess:
    fetches = [c, d, b, E]
    outs = sess.run(fetches)
    print('outs={}'.format(outs))
    print(type(outs))
    print(type(outs[0]))
