import matplotlib.pyplot as plt;
import tensorflow as tf;

learning_rate = 0.1
decay_rate = 0.96
global_steps = 1000
decay_steps = 100

# global_step refer to the number of batches seen by the graph. Everytime a batch is provided, the weights are updated in the direction that minimizes the loss. global_step just keeps track of the number of batches seen so far. When it is passed in the minimize() argument list, the variable is increased by one. Have a look at optimizer.minimize().
#
# You can get the global_step value using tf.train.global_step().
# The 0 is the initial value of the global step in this context.
global_ = tf.Variable(tf.constant(0))
# staircase : 如果为 True global_step/decay_step 向下取整
c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_steps):
        T_c = sess.run(c, feed_dict={global_: i})
        T_C.append(T_c)
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)

plt.figure(1)
plt.plot(range(global_steps), F_D, 'r-')
plt.plot(range(global_steps), T_C, 'b-')

plt.show()
