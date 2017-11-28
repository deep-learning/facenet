import tensorflow as tf

V1 = tf.constant([1., 2.])
V2 = tf.constant([3., 4.])
M = tf.constant([[1., 2.]])  # Matrix, 2d
N = tf.constant([[1., 2.], [3., 4.]])  # Matrix, 2d
K = tf.constant([[[1., 2.], [3., 4.]]])  # Tensor, 3d

with tf.Session() as sess:
    k = sess.run(K)
    print(sess.run(M * M))
    print(sess.run(tf.matmul(M, N)))

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
