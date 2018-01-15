from __future__ import division
import tensorflow as tf
import os
import h5py
import random
import numpy as np
import math

class Dataset(object):
    def __init__(self, batch_size=64, scale=3):
        root = os.path.join('preprocessed_data', 'train',
                str(scale) + 'x', 'dataset.h5')
        f = h5py.File(root)
        self.data = np.array(f.get('data'), dtype='float32')
        self.label = np.array(f.get('label'), dtype='float32')
        self.len = self.data.shape[0]

        self.num_batches = int(math.ceil(self.len/batch_size))
        self.batch_size = batch_size
        self.perm = np.random.permutation(self.len)
        self.cur_batch_idx = 0

    def get_next_batch(self):
        mask = self.perm[self.cur_batch_idx*self.batch_size: (self.cur_batch_idx+1)*self.batch_size].tolist()
        data = self.data[mask]
        label = self.label[mask]
        
        assert data.shape[0] > 0
        assert label.shape[0] >0

        data -= 0.5
        label -= 0.5

        data = data.transpose(0, 2, 3, 1)
        label= label.transpose(0, 2, 3, 1)

        self.cur_batch_idx += 1
        if self.cur_batch_idx >= self.num_batches:
            self.cur_batch_idx = 0
            self.perm = np.random.permutation(self.len)

        return (data, label)

class Dataset1(object):
    def __init__(self, path, batch_size=64):
        f = h5py.File(path)
        self.inputs = f.get('data')
        self.labels = f.get('label')

        #self.inputs = self.inputs[0: 200]
        #self.labels = self.labels[0: 200]

        self.inputs_ph = tf.placeholder(self.inputs.dtype, self.inputs.shape)
        self.labels_ph = tf.placeholder(self.labels.dtype, self.labels.shape)
        
        dataset = tf.data.Dataset.from_tensor_slices((self.inputs_ph, self.labels_ph))
        dataset = dataset.map(self._map_fn)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)

        # use initializable iterator to feed big dataset
        self.iterator = dataset.make_initializable_iterator()
        self.num_batches = int(math.ceil(self.inputs.shape[0]/batch_size))

    def _map_fn(self, inp, lbl):
        """ transpose inp and lbl to get channels-last dataset """
        inp_map = tf.transpose(inp, [1, 2, 0])
        lbl_map = tf.transpose(lbl, [1, 2, 0])
        return inp_map, lbl_map

    def get_next_batch(self):
        inputs_batch, labels_batch = self.iterator.get_next()
        return inputs_batch, labels_batch




        
