#!/usr/bin/env python

import os.path

import tensorflow as tf
sess = tf.InteractiveSession()

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('directory', 'dataset',
                    'Directory with the datasets.')
flags.DEFINE_string('summary', 'summary', 'Directory to output the result summaries')

TRAIN_FILE = 'train.tfrecords'

filename = os.path.join(FLAGS.directory, TRAIN_FILE)
filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

reader = tf.TFRecordReader()
_, dataset = reader.read(filename_queue)

features={
        'height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'depth': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
}
result = tf.parse_single_example(dataset, features)

height = result['height'].eval()
width = result['width'].eval()
depth = result['depth'].eval()

features={
        'image_left': tf.FixedLenFeature([], tf.string),
        'zimage_left': tf.FixedLenFeature([], tf.string),
}

samples = tf.parse_example(dataset, features)

image_left_input = tf.decode_raw(samples['image_left'], tf.uint8)
zimage_left_input = tf.decode_raw(samples['zimage_left'], tf.uint8)

# Now setup the network
image_shape = width * height * depth

data_left = tf.placeholder(tf.uint8, shape=[None, image_shape])
image_left = tf.reshape(data_left, [-1,width,height,depth])

zdata_left = tf.placeholder(tf.uint8, shape=[None, image_shape])
zimage_left = tf.reshape(zdata_left, [-1,width,height,depth])

def weight_variable(shape):
    # One should generally initialize weights with a small amount of
    # noise for symmetry breaking, and to prevent 0 gradients.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # Since we're using ReLU neurons, it is also good practice to
    # initialize them with a slightly positive initial bias to avoid
    # "dead neurons."
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def conv_max_pool_2x2(x, conv_width, conv_height, in_depth, out_depth, name="conv"):

    with tf.name_scope(name) as scope:
        W_conv = weight_variable([conv_width, conv_height, in_depth, out_depth])
        b_conv = bias_variable([out_depth])
        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)

    with tf.name_scope("summaries") as scope:

        # TIPS: to display the 32 convolution filters, re-arrange the
        # weigths to look like 32 images with a transposition.
        a = tf.reshape(W_conv, [conv_width * conv_height * in_depth, out_depth])
        b = tf.transpose(a)
        c = tf.reshape(b, [out_depth, conv_width, conv_height * in_depth, 1])
        conv_image = tf.image_summary(name + " filter", c, out_depth)

        # TIPS: by looking at the weights histogram, we can see the the
        # weigths are explosing or vanishing.
        W_conv_hist = tf.histogram_summary(name + " weights", W_conv)
        b_conv_hist = tf.histogram_summary(name + " biases", b_conv)
    
    return h_pool

def up_conv(x, conv_width, conv_height, in_depth, out_depth, name="conv"):
    # TODO

left_pool1 = conv_max_pool_2x2(image_left, 5, 5, 1, 32, "layer1")
left_pool2 = conv_max_pool_2x2(left_pool1, 5, 5, 32, 64, "layer2")
