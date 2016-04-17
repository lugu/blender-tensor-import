#!/usr/bin/env python

import sys
directory = sys.argv[1]
print("saving results to " + directory)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])

# TIPS: use a name scope to organize nodes in the graph visualizer
# other scopes are declared to simplify the graph structure.
with tf.name_scope("network") as scope:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # TIPS: display one sample image in order to make sure we are
    # dealing with the correct input.
    input_image = tf.image_summary("input", x_image, 1)

    # TIPS: to display the 32 convolution filters, re-arrange the
    # weigths to look like 32 images with a transposition.
    conv1_25_32 = tf.reshape(W_conv1, [25, 32])
    conv1_32_25 = tf.transpose(conv1_25_32)
    conv1_32_5_5_1 = tf.reshape(conv1_32_25, [32, 5, 5, 1])
    W_conv1_image = tf.image_summary("conv1 kernels", conv1_32_5_5_1, 32)

    # TIPS: by looking at the weights histogram, we can see the the
    # weigths are explosing or vanishing.
    W_conv1_hist = tf.histogram_summary("conv1 weights", W_conv1)
    b_conv1_hist = tf.histogram_summary("conv1 biases", b_conv1)

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("xent") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    cross_entropy_sum = tf.scalar_summary('cross entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_sum = tf.scalar_summary('accuracy', accuracy)

# TIPS: use two different files for trainning and testing: this allow
# tensorboard to compare the two on the same graph.
train_writer = tf.train.SummaryWriter(directory + "/train", sess.graph_def)
test_writer = tf.train.SummaryWriter(directory + "/test", sess.graph_def)

# during trainning, collect all the summaries
summary_op = tf.merge_all_summaries()

# TIPS: testing only need a few metrics: do not collect all summaries.
# during testing purpose, just run the accuracy and the cross entropy
test_summary_op = tf.merge_summary([accuracy_sum, cross_entropy_sum])

sess.run(tf.initialize_all_variables())

for i in range(1000):
    print("batch " + str(i))

    # TIPS: train by batch. Adjust the batch size as an hyper
    # parameter (should be a placeholder).
    batch = mnist.train.next_batch(50)
    # TIPS: use pooling to make the network more robust: disactivate
    # 50% of the nodes during testing. Do not use pooling when
    # measuring accuracy.
    feed = {x: batch[0], y_: batch[1], keep_prob: 0.5}
    sess.run(train_step, feed_dict=feed)
    if i%10 == 0:

        feed = { x:batch[0], y_: batch[1], keep_prob: 1.0}
        summary = sess.run(summary_op, feed_dict=feed)
        train_writer.add_summary(summary, i)

        test_batch = mnist.test.next_batch(50)
        test_feed = { x:test_batch[0], y_: test_batch[1], keep_prob: 1.0}
        test_summary = sess.run(test_summary_op, feed_dict=test_feed)
        test_writer.add_summary(test_summary, i)

        ## TIPS: to display metrics in respected to cpu time, simply
        ## replace the index i with the time.
        ##
        # import time
        # cpu_tensor = tf.convert_to_tensor(time.process_time())
        # writer.add_summary(summary, time.process_time())

# feed = { x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
# print("test accuracy %g"%accuracy.eval(feed_dict=feed))
