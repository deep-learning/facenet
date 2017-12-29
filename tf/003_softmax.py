import tensorflow as tf

sess = tf.InteractiveSession()

# NN output
logits = tf.constant([[1., 2, 3.],
                      [1., 2, 3.],
                      [1., 2., 3.]])

y = tf.nn.softmax(logits=logits)

print(sess.run(logits))
print(sess.run(y))

# true label
y_ = tf.constant([[0., 0., 1.],
                  [0., 0., 1.],
                  [0., 0., 1.]])

# calculate cross entroyp in one pass
# logits instead of y!!!
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
print(sess.run(cross_entropy))

# calculate cross entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
print(sess.run(cross_entropy))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
