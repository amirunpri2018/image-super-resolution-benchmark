from data import *
from model import *
from solver import *
from loss import *
import progressbar
model=SRCNN()
num_epochs = 1
batch_size = 64

loss_fn = MSE_Loss()
learning_rate = 1e-4
verbose=True
print_every=10


#solver = Solver(model, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, 
#        loss_fn=loss_fn, verbose=verbose, print_every=print_every)
#
#dataset = Dataset()
#solver.train(dataset)

model = SRCNNv1()
dataset = Dataset1('data/preprocessed_data/train/3x/dataset.h5')

# build graph
inputs, labels = dataset.get_next_batch()
outputs = model.inference(inputs)
loss = model.loss(labels, outputs)
train_step = model.optimize(loss, learning_rate=learning_rate)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        sess.run(dataset.iterator.initializer, feed_dict={dataset.inputs_ph: dataset.inputs, 
                                                           dataset.labels_ph: dataset.labels})

        bar = progressbar.ProgressBar(max_value=dataset.num_batches)
        
        running_loss = 0
        for batch in range(dataset.num_batches):
            loss_val, _ = sess.run([loss, train_step])
            running_loss += loss_val
            bar.update(batch+1, force=True)

        average_loss = running_loss/dataset.num_batches
        print('Epoch  %5d, loss %.5f' \
                %(epoch, average_loss))


