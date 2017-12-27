import numpy as np
import tensorflow as tf

def conv_layer(x, kHeight, kWidth, strideX, strideY, feature_num, name, padding='SAME'):
    # assuming NHWC data format
    channel_num = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[kHeight, kWidth, channel_num, feature_num])
        b = tf.get_variable('b', shape=[feature_num])
        feature_map = tf.nn.conv2d(x,
                                   w,
                                   strides=[1, strideX, strideY, 1],
                                   padding=padding)
        out = tf.nn.bias_add(feature_map, b)
        return tf.nn.relu(tf.reshape(out, feature_map.get_shape().as_list(), name=scope.name))

def max_pooling_layer(x, kHeight, kWidth, stride, name, padding):
    return tf.nn.max_pool(x,
                          ksize=[1, kHeight, kWidth, 1],  # [batch, height, width, channels]
                          strides=[1, stride, stride, 1],  # no pooling on batch & channel dimension
                          padding=padding,
                          name=name)

def dropout(x, keep_prop, name=None):
    return tf.nn.dropout(x, keep_prob=keep_prop, name=name)

def fc_layer(x, inputD, outputD, name, relu=True):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[inputD, outputD], dtype=tf.float32)
        b = tf.get_variable('b', [outputD], dtype=tf.float32)
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if relu:
            return tf.nn.relu(out)
        else:
            return out

_VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    """
    The network configuration:
    - RGB 224x224x3
    - conv3-64
    - conv3-64
    - maxpool

    - (112x112x128)

    - conv3-128
    - conv3-128
    - maxpool

    - (56x56x256)

    - conv3-256
    - conv3-256
    - conv3-256
    - conv3-256 (vs. vgg16)
    - maxpool

    - (28x28x512)

    - conv3-512
    - conv3-512
    - conv3-512
    - conv3-512 (vs. vgg16)
    - maxpool

    - (14x14x512)

    - conv3-512
    - conv3-512
    - conv3-512
    - conv3-512 (vs.vgg16)
    - maxpool

    - (7x7x512x4096)

    - fc-4096
    - (4096x4096)
    - fc-4096
    - (4096x1000)
    - fc-1000
    - softmax
    """
    WIDTH = 224
    HEIGHT = 224
    CHANNELS = 3
    LABELS = 1000

    model = {}
    model_save_path = None
    model_save_iter_freq = 0

    learning_rate = 0.05

    _inputRGB = None
    _inputBGR = None
    _inputNormalizedBGR = None

    _conv1_1 = None
    _conv1_2 = None
    _pool = None

    _conv2_1 = None
    _conv2_2 = None
    _pool2 = None

    _conv3_1 = None
    _conv3_2 = None
    _conv3_3 = None
    _conv3_4 = None
    _pool3 = None

    _conv4_1 = None
    _conv4_2 = None
    _conv4_3 = None
    _conv4_4 = None
    _pool4 = None

    _conv5_1 = None
    _conv5_2 = None
    _conv5_3 = None
    _conv5_4 = None
    _pool5 = None

    _fc6 = None
    _relu6 = None

    _fc7 = None
    _relu7 = None

    _fc8 = None

    _preds = None
    # in [? 1000] shape

    _loss = None
    _optimizer = None
    _train_labels = None

    def __init__(self,
                 model=None,
                 model_save_path=None,
                 model_save_iter_freq=0):
        self.model = self.__init__empty_model() if not model else model
        self.model_save_path = model_save_path
        self.model_save_iter_freq = model_save_iter_freq

        # define train labels
        self.__train_labels = tf.placeholder(tf.float32,
                                             shape=[None, Vgg19.LABELS])
        self._inputRGB = tf.placeholder(tf.float32,
                                        shape=[None, Vgg19.HEIGHT, Vgg19.WIDTH, Vgg19.CHANNELS])

        r, g, b = tf.split(self._inputRGB, 3, 3)
        self._inputBGR = tf.concat([b, g, r], 3)

        self._inputNormalizedBGR = tf.concat([
            b - _VGG_MEAN[0],
            g - _VGG_MEAN[1],
            r - _VGG_MEAN[2]
        ], 3)

        # setup vgg-net graph
        self._conv1_1 = self.conv

    @property
    def inputRGB(self):
        """of shape [?, 224, 224, 3] in RGB order"""
        return self._inputRGB

    @property
    def inputBGR(self):
        return self._inputBGR

    @property
    def preds(self):
        return self._preds

    @property
    def train_labels(self):
        return self._train_labels

    def _avg_pool(self, value, name):
        return tf.nn.avg_pool(value,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def _max_pool(self, value, name):
        return tf.nn.max_pool(value,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def _conv_layer(self, value, name):
        with tf.variable_scope(name):
            shape = value.get_shape().as_list()
            dim = 1


    def _get_conv_filter(self, name):
        return tf.Variable(self.model[name][0], name='filter')

    def _get_biases(self, name):
        return tf.Variable(self.model[name][1], name='biases')

    def _get_fc_weights(self, name):
        return tf.Variable(self.model[name][0], name='weights')

    def __init__empty_model(self):
        self.model = {
            # [wights, biases]
            'conv1_1': [np.ndarray([3, 3, 3, 64]),
                        np.ndarray([64])],

            'conv1_2': [np.ndarray([3, 3, 64, 64]),
                        np.ndarray([64])],

            'conv2_1': [np.ndarray([3, 3, 64, 128]),
                        np.ndarray([128])],

            'conv2_2': [np.ndarray([3, 3, 128, 128]),
                        np.ndarray([128])],

            'conv3_1': [np.ndarray([3, 3, 128, 256]),
                        np.ndarray([256])],
            'conv3_2': [np.ndarray([3, 3, 256, 256]),
                        np.ndarray([256])],
            'conv3_3': [np.ndarray([3, 3, 256, 256]),
                        np.ndarray([256])],
            'conv3_4': [np.ndarray([3, 3, 256, 256]),
                        np.ndarray([256])],

            'conv4_1': [np.ndarray([3, 3, 256, 512]),
                        np.ndarray([512])],
            'conv4_2': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            'conv4_3': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            'conv4_4': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],

            'conv5_1': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            'conv5_2': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            'conv5_3': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],
            'conv5_4': [np.ndarray([3, 3, 512, 512]),
                        np.ndarray([512])],

            'fc6': [np.ndarray([7 * 7 * 512 * 4096, 4096]),
                    np.ndarray([4096])],
            'fc7': [np.ndarray([4096, 4096]),
                    np.ndarray([4096])],
            'fc8': [np.ndarray([4096, 1000]),
                    np.ndarray([1000])]
        }

    def _max_pool(self, value, name):
        return tf.nn.max_pool(value,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def _conv_layer(self, value, name):
        pass
