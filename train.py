from data import *
from model import *
from solver import *
from loss import *
from utils import *
import progressbar

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'SRCNN',
                           """Model name (SRCNN, ) """)
tf.app.flags.DEFINE_string('scale', '3x',
                            """Super resolution scale (2x, 3x, 4x)""")
tf.app.flags.DEFINE_integer('num_epochs', 100,
                            """Number of training epochs """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size""")
tf.app.flags.DEFINE_float('initial_lr', 1e-3,
                          """Initial learning rate """)
tf.app.flags.DEFINE_integer('decay_epoch', 20,
                            """Number of epochs to decay learning rate""")
tf.app.flags.DEFINE_float('decay_rate', 0.5,
                          """Learning rate decay factor""")
tf.app.flags.DEFINE_bool('verbose', False,
                         """Print training info """)

def main(argv=None):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        model = get_model(FLAGS.model)

        with tf.device('/cpu:0'):
            dataset_path = os.path.join('data/preprocessed_data/train', FLAGS.scale, 'dataset.h5')
            if not tf.gfile.Exists(dataset_path):
                raise Exception('Cannot find %s' %dataset)
            dataset = Trainset('data/preprocessed_data/train/3x/dataset.h5', FLAGS.batch_size)
            inputs, labels = dataset.get_next_batch()
        
        # build graph
        outputs = model.inference(inputs)
        loss = model.loss(labels, outputs)
        train_step = model.optimize(loss, global_step, FLAGS.initial_lr, 
                                    FLAGS.decay_epoch*dataset.num_batches, FLAGS.decay_rate)

        # tensorboard
        tf.summary.scalar('loss', loss)
        merge = tf.summary.merge_all()
        log_dir = os.path.join('log', FLAGS.model, FLAGS.scale)
        clean_and_create_dir(log_dir)
        
        # model saver
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            for epoch in range(FLAGS.num_epochs):
                # observe training progress
                if FLAGS.verbose:
                    bar = progressbar.ProgressBar(max_value=dataset.num_batches)
                
                sess.run(dataset.iterator.initializer, feed_dict={dataset.inputs_ph: dataset.inputs, 
                                                                   dataset.labels_ph: dataset.labels})
                running_loss = 0
                for batch in range(dataset.num_batches):
                    loss_val, result, _ = sess.run([loss, merge, train_step])
                    running_loss += loss_val
                    
                    writer.add_summary(result, epoch*dataset.num_batches + batch)
                    
                    if FLAGS.verbose:
                        bar.update(batch+1, force=True)
                
                average_loss = running_loss/dataset.num_batches
                if FLAGS.verbose:
                    print('Epoch  %5d, loss %.5f' %(epoch, average_loss))

            # save model
            checkpoint_path = os.path.join('checkpoint', FLAGS.model, FLAGS.scale)
            checkpoint_file = os.path.join(checkpoint_path, 'model.ckpt')
            clean_and_create_dir(checkpoint_path)
            saver.save(sess, checkpoint_file)
            if FLAGS.verbose:
                print('Check point file is saved in %s' %checkpoint_path)
            
            writer.close()

if __name__ == '__main__':
    tf.app.run()
