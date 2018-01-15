from __future__ import division
import tensorflow as tf
import os
import h5py
import random
import numpy as np
import math

class Dataset(object):
    def __init__(self, path, batch_size=64):
        f = h5py.File(path)
        self.inputs = f.get('data')
        self.labels = f.get('label')

        #self.inputs = self.inputs[0: 1000]
        #self.labels = self.labels[0: 1000]
         
        with tf.name_scope('Dataset'):
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
        with tf.name_scope('NextBatch'):
            inputs_batch, labels_batch = self.iterator.get_next()
            return inputs_batch, labels_batch




        
