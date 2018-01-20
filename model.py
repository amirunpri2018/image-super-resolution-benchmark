import tensorflow as tf
import numpy as np


def get_model(model):
    if model == 'SRCNN':
        return SRCNN()
    elif model == 'VDSR':
        return VDSR()
    else:
        raise Exception('Unknown model %s' %model)

class BaseModel(object):
    def __init__(self):
        self.initializer=tf.contrib.layers.xavier_initializer_conv2d()
        pass
    
    def inference(self):
        raise Exception('Unimplemented method')

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


class SRCNN(BaseModel):
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
                name='conv1')

        outputs = tf.layers.conv2d(
                inputs=conv2,
                filters=1,
                kernel_size=[5, 5],
                padding='same',
                name='conv3')
        return outputs

    
class VDSR(BaseModel):
    def inference(self, input_tensor):
        conv_next = tf.layers.conv2d(inputs=input_tensor,
                                     filters=64,
                                     kernel_size=3,
                                     kernel_initializer=self.initializer,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     name='conv_first')
        for i in range(1,17):
            conv_next = tf.layers.conv2d(inputs=conv_next,
                                        filters=64,
                                        kernel_size=3,
                                        kernel_initializer=self.initializer,
                                        padding='same',
                                        activation=tf.nn.relu,
                                        name='conv_next_{}'.format(i))

        conv_last = tf.layers.conv2d(inputs=conv_next,
                                   filters=1,
                                   kernel_size=3,
                                   kernel_initializer=self.initializer,
                                   padding='same',
                                   name='conv_last')
        outputs = tf.add(conv_last, input_tensor, name='output')
        return outputs


        


