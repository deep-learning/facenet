import numpy as np
import os

DATA_PATH = '/home/zhenglai/data/cifar10/cifar-10-batches-py'

class CifarLoader:
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack(d['data'] for d in data)
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d['labels']]))

    def unpickle(fpath):
        with open(os.path.join(DATA_PATH, fpath), 'rb') as fo:
            dic = cPickle.load(fo)
        return dic
