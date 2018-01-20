from __future__ import division
import tensorflow as tf
import scipy
import glob
import numpy as np
from model import *
from utils import *
import time
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'SRCNN',
                           """Model name (SRCNN, ) """)
tf.app.flags.DEFINE_string('scale', '3x',
                           """Super resolution scale (2x, 3x, 4x)""")
tf.app.flags.DEFINE_string('test_set', 'set5',
                           """Test set (set5, set14, )""")

def compute_PSNR(out, lbl):
    """ compute PSNR of outpus and labels
    args: 
    - outputs: np array of shape (1, H, W, 1)
    - labels: np array of shape (1, H, W, 1)
    """
    out = out[0, :, :, 0]
    lbl = lbl[0, :, :, 0]
    diff = out - lbl
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr

def main(argv=None):
    test_path = os.path.join('data/preprocessed_data/test', FLAGS.test_set, FLAGS.scale)
    save_path = os.path.join('results', FLAGS.model, FLAGS.test_set, FLAGS.scale)
    clean_and_create_dir(save_path)
    
    lr_paths = glob.glob(os.path.join(test_path, 'low_res', '*.bmp'))
    hr_paths = glob.glob(os.path.join(test_path, 'high_res', '*.bmp'))
    lr_paths.sort()
    hr_paths.sort()
    
    inp_ph = tf.placeholder(tf.float32, [1, None, None, 1])
    lbl_ph = tf.placeholder(tf.float32, [1, None, None, 1])

    model = get_model(FLAGS.model)
    out_graph = model.inference(inp_ph)
    
    sess = tf.Session()
    check_point = os.path.join('checkpoint', FLAGS.model, FLAGS.scale, 'model.ckpt')
    if tf.gfile.Exists(check_point):
        raise Exception('Cannot find %s' %check_point)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, check_point)

    for lr_path, hr_path in zip(lr_paths, hr_paths):
        # inp and lbl is in [0: 255] range
        inp = scipy.misc.imread(lr_path)
        inp = inp/255 - 0.5
        inp = inp[np.newaxis, :, :, np.newaxis]

        since = time.time()
        out = sess.run(out_graph, feed_dict={inp_ph: inp})    
        print(time.time() - since)
        
        out = (out + 0.5)* 255
        lbl = scipy.misc.imread(hr_path)
        lbl = lbl[np.newaxis, :, :, np.newaxis]
        print('%20s: %.3fdB' %(os.path.basename(lr_path), compute_PSNR(out, lbl)))
        scipy.misc.imsave(os.path.join(save_path, os.path.basename(lr_path)), out[0,:,:,0])

if __name__ == '__main__':
    tf.app.run()
    


