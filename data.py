from __future__ import division
import tensorflow as tf
import os
import h5py
import glob
import math

class Trainset(object):
    def __init__(self, path, batch_size=64):
        f = h5py.File(path)
        self.inputs = f.get('data')
        self.labels = f.get('label')

        # test on small dataset
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
        inp_map_norm = inp_map - 0.5
        lbl_map_norm = lbl_map - 0.5
        return inp_map_norm, lbl_map_norm

    def get_next_batch(self):
        with tf.name_scope('NextBatch'):
            inputs_batch, labels_batch = self.iterator.get_next()
            return inputs_batch, labels_batch

class Testset(object):
    def __init__(self, path):
        self.inputs_path = glob.glob(os.path.join(path, 'low_res', '*.bmp'))
        self.labels_path = glob.glob(os.path.join(path, 'high_res', '*.bmp'))
        self.inputs_path.sort()
        self.labels_path.sort()
        print(self.inputs_path)
        self.inputs_path = tf.constant(self.inputs_path, dtype=tf.string)
        self.labels_path = tf.constant(self.labels_path, dtype=tf.string)
        print(self.inputs_path)
        dataset = tf.data.Dataset.from_tensor_slices((self.inputs_path, self.labels_path))
        dataset = dataset.map(self._map_fn)

        self.iterator = dataset.make_one_shot_iterator()

    def _map_fn(self, inp_path, lbl_path):
        inp_string = tf.read_file(inp_path)
        lbl_string = tf.read_file(lbl_path)

        inp_image = tf.image.decode_bmp(inp_string, 0)
        lbl_image = tf.image.decode_bmp(lbl_string, 0)
        return inp_image, lbl_image

    def get_next_image(self):
        inp, lbl = self.iterator.get_next()
        return inp, lbl

        





        
