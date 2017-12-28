import tensorflow as tf

# todo http://www.cnblogs.com/welhzh/p/6607581.html

sess = tf.InteractiveSession()

input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))

# 1x3x3x1
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

sess.run(tf.global_variables_initializer())
print(sess.run(op).shape)

input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

sess.run(tf.global_variables_initializer())
print(sess.run(op).shape)
print(sess.run(op))
