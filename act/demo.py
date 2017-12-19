#!/usr/bin/env python

from __future__ import print_function

from act_cell import ACTCell

import tensorflow as tf

import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
float_batch_size = float(batch_size)
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
original_timesteps = 28 # timesteps
timesteps = 28 # timesteps
num_hidden = 120 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
epsilon = 0.01


def loss(brother, x, y):
    y_ = brother.model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=y_, labels=y)) + brother.model.act_cell.calculate_ponder_cost(time_penalty=0.1)
    return loss


def load_data():
    """Returns training and test tf.data.Dataset objects."""
    data = input_data.read_data_sets('/tmp/data', one_hot=True)
    train_ds = tf.data.Dataset.from_tensor_slices((data.train.images,
                                                   data.train.labels))
    test_ds = tf.data.Dataset.from_tensors((data.test.images, data.test.labels))
    return (train_ds, test_ds)


class Model(tfe.Network):
    def __init__(self):
        super(Model, self).__init__(name='')
        # Define weights
        with tf.device("/gpu:0"):
            with tf.variable_scope("w_iga"):
                self.w = tfe.Variable(tf.random_normal([num_hidden, num_classes]))
            with tf.variable_scope("b_iga"):
                self.b = tfe.Variable(tf.random_normal([num_classes]))
            with tf.variable_scope("lstm"):
                #self.cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
                self.cell = rnn.GRUCell(num_hidden)
                self.act_cell = ACTCell(num_hidden, self.cell, epsilon,
                                        max_computation=50,
                                        batch_size=batch_size)


    def call(self, x):
        self.act_cell.reset_stats()
        x = tf.unstack(x, timesteps, 1)
        with tf.variable_scope("rnn_iga"):
            outputs, states = rnn.static_rnn(self.act_cell, x, dtype=tf.float32)
        print(" ".join(["%2.1f" % (x/float_batch_size) for x in self.act_cell.stats]))
        return tf.matmul(outputs[-1], self.w) + self.b


class BigBrother(object):
    def __init__(self):
        self.model = Model()
        self.step = 0

    def train(self):
        (train_ds, test_ds) = load_data()
        train_ds = train_ds.shuffle(60000).batch(batch_size)
        batches_per_epoch = 60000 / batch_size
        
        #optimizer = tf.train.GradientDescentOptimizer(
        #   learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer()  # learning_rate=learning_rate)
        with tf.device("/gpu:0"):
            for step in range(0, 100):
                for (batch, (batch_x, batch_y)) in enumerate(tfe.Iterator(train_ds)):
                    if batch_x.shape[0] < batch_size:
                        continue
                    s = step * batches_per_epoch + batch
                    #print("Step: " + str(s))
                    # Reshape data to get 28 seq of 28 elements
                    batch_x = tf.reshape(batch_x, [batch_size, original_timesteps, num_input])
                    # batch_x = tf.concat([batch_x, tf.zeros([batch_size, 4, num_input])], axis = 1)
                    #batch_x = tf.concat([tf.zeros([batch_size, 4, num_input]), batch_x], axis = 1)
                    grads = tfe.implicit_gradients(loss)(self, batch_x, batch_y)
                    #import pdb
                    #pdb.set_trace()
                    optimizer.apply_gradients(grads)
                    #optimizer.minimize(lambda: loss(self, batch_x, batch_y))
                    
                    if s % 10 == 0:
                        loss_val = loss(self, batch_x, batch_y)

                        logits = self.model(batch_x)
                        prediction = tf.nn.softmax(logits)
                        # Evaluate model (with test logits, for dropout to be disabled)
                        correct_pred = tf.equal(tf.argmax(prediction, 1),
                                                tf.argmax(batch_y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                        print("Step: %d  Loss: %.2f  Accuracy: %.2f" % (s,
                                                                        loss_val.numpy(),
                                                                        accuracy.numpy()))

    def test(self):
        #prediction = tf.nn.softmax(logits)
        
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer.minimize(loss_op)
        
        # Evaluate model (with test logits, for dropout to be disabled)
        #correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #return accuracy


def main():
    brother = BigBrother()
    brother.train()
    


if False:
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
    
        for step in range(1, training_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    
        print("Optimization Finished!")
    
        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))


main()
