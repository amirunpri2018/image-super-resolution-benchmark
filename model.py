import tensorflow as tf
import numpy as np

class SRCNN(object):
    def __init__(self):
        pass

    def inference(self, input_tensor):
        conv1 = tf.layers.conv2d(
                inputs=input_tensor,
                filters=64,
                kernel_size=[9, 9],
                padding='same',
                activation=tf.nn.relu,
                name='conv1')

        conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=[1, 1],
                padding='same',
                activation=tf.nn.relu,
                name='conv2')

        outputs = tf.layers.conv2d(
                inputs=conv2,
                filters=1,
                kernel_size=[5, 5],
                padding='same',
                name='conv3')
        return outputs

    def loss(self, labels, outputs):
        with tf.name_scope('MSELoss'):
            return tf.losses.mean_squared_error(labels, outputs)
        
    def optimize(self, loss, global_step, initial_lr, decay_step, decay_rate):
        with tf.name_scope('Optimize'):
            lr = tf.train.exponential_decay(initial_lr,
                                            global_step,
                                            decay_step,
                                            decay_rate,
                                            staircase=True)
            tf.summary.scalar('learning_rate', lr)
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
            return train_step


        


