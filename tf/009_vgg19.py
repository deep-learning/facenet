import argparse
import os
import urllib.request

import caffe_classes
import cv2
import numpy as np
import sys
import tensorflow as tf

# todo https://gist.github.com/zhenglaizhang/f672f72d854d89d20c4dd1f1cece662e
# todo conv2d http://www.cnblogs.com/welhzh/p/6607581.html

def max_pooling_layer(x, kHeight, kWidth, strideX, strideY, name, padding='SAME'):
    return tf.nn.max_pool(x,
                          ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1],
                          padding=padding,
                          name=name)

def dropout_layer(x, keepProp, name=None):
    return tf.nn.dropout(x, keepProp, name)

def fc_layer(x, inputD, outputD, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[inputD, outputD], dtype=tf.float32)
        b = tf.get_variable('b', [outputD], dtype=tf.float32)
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def conv_layer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding='SAME'):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[kHeight, kWidth, channel, featureNum])
        b = tf.get_variable('b', shape=[featureNum])
        featureMap = tf.nn.conv2d(input=x,
                                  filter=w,
                                  strides=[1, strideX, strideY, 1],
                                  padding=padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()),
                          name=scope.name)

class Vgg19(object):
    def __init__(self, x, keep_prop, class_num, skip, model_path='~/models/vgg/vgg19.npy'):
        self.x = x
        self.KEEPPRO = keep_prop
        self.CLASSNUM = class_num
        self.SKIP = skip
        self.MODELPATH = os.path.expanduser(model_path)
        self.fc8 = None
        self.build_cnn()

    def build_cnn(self):
        conv1_1 = conv_layer(self.x, 3, 3, 1, 1, 64, 'conv1_1')
        conv1_2 = conv_layer(conv1_1, 3, 3, 1, 1, 64, 'conv1_2')
        pool1 = max_pooling_layer(conv1_2, 2, 2, 2, 2, 'pool1')

        conv2_1 = conv_layer(pool1, 3, 3, 1, 1, 128, 'conv2_1')
        conv2_2 = conv_layer(conv2_1, 3, 3, 1, 1, 128, 'conv2_2')
        pool2 = max_pooling_layer(conv2_2, 2, 2, 2, 2, 'pool2')

        conv3_1 = conv_layer(pool2, 3, 3, 1, 1, 256, 'conv3_1')
        conv3_2 = conv_layer(conv3_1, 3, 3, 1, 1, 256, 'conv3_2')
        conv3_3 = conv_layer(conv3_2, 3, 3, 1, 1, 256, 'conv3_3')
        conv3_4 = conv_layer(conv3_3, 3, 3, 1, 1, 256, 'conv3_4')
        pool3 = max_pooling_layer(conv3_4, 2, 2, 2, 2, 'pool3')

        conv4_1 = conv_layer(pool3, 3, 3, 1, 1, 512, 'conv4_1')
        conv4_2 = conv_layer(conv4_1, 3, 3, 1, 1, 512, 'conv4_2')
        conv4_3 = conv_layer(conv4_2, 3, 3, 1, 1, 512, 'conv4_3')
        conv4_4 = conv_layer(conv4_3, 3, 3, 1, 1, 512, 'conv4_4')
        pool4 = max_pooling_layer(conv4_4, 2, 2, 2, 2, 'pool4')

        conv5_1 = conv_layer(pool4, 3, 3, 1, 1, 512, 'conv5_1')
        conv5_2 = conv_layer(conv5_1, 3, 3, 1, 1, 512, 'conv5_2')
        conv5_3 = conv_layer(conv5_2, 3, 3, 1, 1, 512, 'conv5_3')
        conv5_4 = conv_layer(conv5_3, 3, 3, 1, 1, 512, 'conv5_4')
        pool5 = max_pooling_layer(conv5_4, 2, 2, 2, 2, 'pool5')

        fc_in = tf.reshape(pool5, [-1, 7 * 7 * 512])
        fc6 = fc_layer(fc_in, 7 * 7 * 512, 4096, True, 'fc6')
        dropout1 = dropout_layer(fc6, self.KEEPPRO)

        fc7 = fc_layer(dropout1, 4096, 4096, True, 'fc7')
        dropout2 = dropout_layer(fc7, self.KEEPPRO)
        self.fc8 = fc_layer(dropout2, 4096, self.CLASSNUM, True, 'fc8')

    def loadModel(self, sess):
        wDict = np.load(self.MODELPATH, encoding='bytes').item()
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse=True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            # bias
                            sess.run(tf.get_variable('b', trainable=False).assign(p))
                        else:
                            # wights
                            sess.run(tf.get_variable('w', trainable=False).assign(p))

def main(args):
    parser = argparse.ArgumentParser(description='VGG19 image classifier')
    parser.add_argument('mode', choices=['folder', 'url'], default='folder')
    parser.add_argument('path', default='vgg_test_data')
    args = parser.parse_args(args)

    if args.mode == 'folder':
        withPath = lambda f: '{}/{}'.format(args.path, f)
        testImg = dict((f, cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
    elif args.mode == 'url':
        def url2img(url):
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype='uint8')
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image

        testImg = {args.path: url2img(args.path)}

    if testImg.values():
        dropoutPro = 1
        classNum = 1000
        skip = []

        # image mean of ImageNet
        imgMean = np.array([104, 117, 124], np.float)
        x = tf.placeholder(tf.float32, [1, 224, 224, 3])
        model = Vgg19(x, dropoutPro, classNum, skip)
        score = model.fc8
        softmax = tf.nn.softmax(score)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.loadModel(sess)
            for key, img in testImg.items():
                resized = cv2.resize(img.astype(np.float), (224, 224)) - imgMean
                maxx = np.argmax(sess.run(softmax, feed_dict={
                    x: resized.reshape((1, 224, 224, 3))
                }))
                res = caffe_classes.class_names[maxx]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 255, 0), 2)
                print('{}:{}\n---'.format(key, res))
                cv2.imshow('demo', img)
                cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])
