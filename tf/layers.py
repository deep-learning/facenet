import tensorflow as tf

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)

    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    return tf.matmul(input, W) + b

def simple_network(input):
    with tf.variable_scope('layer_1'):
        output_1 = layer(input, [784, 100], [100])
        print(output_1.name)

    with tf.variable_scope('layer_2'):
        output_2 = layer(output_1, [100, 50], [50])
        print(output_2.name)

    with tf.variable_scope('layer_3'):
        output_3 = layer(output_2, [50, 10], [10])
        print(output_3.name)

    return output_3

# name_scopesymotion-prefix)w 是给op_name加前缀, variable_scope是给get_variable()创建的变量的名字加前缀。
# todo ???
if __name__ == '__main__':
    with tf.variable_scope("shared_variables") as scope:
        i_1 = tf.placeholder(tf.float32, [1000, 784], name='i_1')
        simple_network(i_1)
        scope.reuse_variables()
        i_2 = tf.placeholder(tf.float32, [1000, 784], name='i_2')
        simple_network(i_2)

    c = []
    for d in ['/gpu:0', '/gpu:1']:
        with tf.device(d):
            a = tf.constant([1., 2., 3., 4.], shape=[2, 2], name='a')
            b = tf.constant([1., 2.], shape=[2, 1], name='b')
            c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        print(sess.run(sum))
