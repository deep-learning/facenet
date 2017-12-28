# todo http://blog.csdn.net/u011974639/article/details/76460849#googlenet在tensorflow的实现
# todo http://blog.csdn.net/s_sunnyy/article/details/70808474

import tensorflow as tf
import time
from datetime import datetime
import math
import argparse
import sys

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0, 0, stddev)

def inception_v1_arg_scope(weight_decay = 0.00004,
                           stddev = 0.1,
                           batch_norm_var_collection = 'moving_vars'):
    pass
