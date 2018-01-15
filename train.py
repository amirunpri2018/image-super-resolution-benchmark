from data import *
from model import *
from solver import *
from loss import *
from utils import *
import progressbar

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'SRCNN',
                           """ Model name (SRCNN, ) """)
tf.app.flags.DEFINE_integer('num_epochs', 3,
                            """ Number of training epochs """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """ batch size""")
tf.app.flags.DEFINE_float('initial_lr', 1e-4,
                          """ initial learning rate """)
tf.app.flags.DEFINE_string('log_dir', 'log',
                           """ log dir to visual tensorboard """)
tf.app.flags.DEFINE_bool('verbose', False,
                         """ print training info """)



def main(argv=None):
    global_step = tf.train.get_or_create_global_step()

    model = SRCNN()

    with tf.device('/cpu:0'):
        dataset = Dataset('data/preprocessed_data/train/3x/dataset.h5')
        inputs, labels = dataset.get_next_batch()
    
    # build graph
    outputs = model.inference(inputs)
    loss = model.loss(labels, outputs)

    DECAY_STEP = dataset.num_batches*1
    DECAY_RATE = 0.5
    train_step = model.optimize(loss, global_step, FLAGS.initial_lr, DECAY_STEP, DECAY_RATE)

    # tensorboard
    tf.summary.scalar('loss', loss)
    merge = tf.summary.merge_all()
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.num_epochs):
            sess.run(dataset.iterator.initializer, feed_dict={dataset.inputs_ph: dataset.inputs, 
                                                               dataset.labels_ph: dataset.labels})

            bar = progressbar.ProgressBar(max_value=dataset.num_batches)
            
            running_loss = 0
            for batch in range(dataset.num_batches):
                loss_val, result, _ = sess.run([loss, merge, train_step])

                running_loss += loss_val
                bar.update(batch+1, force=True)
                writer.add_summary(result, epoch*dataset.num_batches + batch)
            
            average_loss = running_loss/dataset.num_batches
            print('Epoch  %5d, loss %.5f' \
                    %(epoch, average_loss))
        writer.close()

if __name__ == '__main__':
    tf.app.run()
