import tensorflow as tf
import numpy as np

class SRCNN(object):
    def __init__(self):
        self.name = SRCNN
    
    def __call__(self, input_tensor):
        conv1 = tf.layers.conv2d(
                inputs=input_tensor,
                filters=64,
                kernel_size=[9, 9],
                padding='same',
                activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=[1, 1],
                padding='same',
                activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=1,
                kernel_size=[5, 5],
                padding='same')
        return conv3


class SRCNNv1(object):
    def __init__(self):
        pass

    def inference(self, input_tensor):
        conv1 = tf.layers.conv2d(
                inputs=input_tensor,
                filters=64,
                kernel_size=[9, 9],
                padding='same',
                activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=[1, 1],
                padding='same',
                activation=tf.nn.relu)

        outputs = tf.layers.conv2d(
                inputs=conv2,
                filters=1,
                kernel_size=[5, 5],
                padding='same')
        return outputs

    def loss(self, labels, outputs):
        return tf.losses.mean_squared_error(labels, outputs)
        
    def optimize(self, loss, learning_rate=1e-4):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step


        


