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


if __name__ == '__main__':
    tf_info()
    tf_hello()
