import tensorflow as tf


def tf_info():
    print(tf.__version__)


def tf_hello():
    h = tf.constant('hello')
    w = tf.constant('World')
    hw = h + w
    with tf.Session() as sess:
        ans = sess.run(hw)
        print(str(ans, encoding='utf-8'))


def tf_random_crop():
    v1 = tf.random_uniform([5, 5], 0, 10, dtype=tf.int32, seed=0)
    v2 = tf.random_crop(v1, [5, 5])
    with tf.Session() as sess:
        a1, a2 = sess.run([v1, v2])
        print(a1)
        print(a2)


if __name__ == '__main__':
    # tf_info()
    # tf_hello()
    tf_random_crop()
