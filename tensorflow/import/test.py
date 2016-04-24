#!/usr/bin/env python

import tensorflow as tf

def decode_feature(samples, width, height, depth):
    image = tf.decode_raw(samples, tf.uint8)
    print(str(image.dtype))
    # FIXME: set shape
    image.set_shape([width * height * depth])
    # FIXME: Convert from [0, 255] -> [-0.5, 0.5] floats.
    # return tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image

def read_and_decode(filename_queue):

    # FIXME: read those values from the features
    width = 960
    height = 540
    depth = 4

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    samples = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
              'image_left': tf.FixedLenFeature([], tf.string),
              'image_right': tf.FixedLenFeature([], tf.string),
              'zimage_left': tf.FixedLenFeature([], tf.string),
              'zimage_right': tf.FixedLenFeature([], tf.string),
        })

    left = decode_feature(samples['image_left'], width, height, depth)
    right = decode_feature(samples['image_right'], width, height, depth)
    zleft = decode_feature(samples['zimage_left'], width, height, 1)
    zright = decode_feature(samples['zimage_right'], width, height, 1)
    return left, right, zleft, zright


def inputs(filename, batch_size):
    # num_epochs tells how many times we can read the input data
    filename_queue = tf.train.string_input_producer([filename])
    left, right, zleft, zright = read_and_decode(filename_queue)

    left, right, zleft, zright = tf.train.shuffle_batch(
        [left, right, zleft, zright], batch_size=batch_size, num_threads=4,
        capacity=10, min_after_dequeue=1) # FIXME

    return left, right, zleft, zright


with tf.Session() as sess:

    batch_size = 1 # FIXME
    input_file = "dataset/train.tfrecords"

    left, right, zleft, zright = inputs(input_file, batch_size)

    tf.image_summary("left", left)
    tf.image_summary("right", right)
    tf.image_summary("zleft", zleft)
    tf.image_summary("zright", zright)

    sum_writer = tf.train.SummaryWriter("sum", sess.graph_def)

    init_op = tf.initialize_all_variables()
    summary_op = tf.merge_all_summaries()

    _, summary = sess.run([init_op, summary_op])

    train_writer.add_summary(summary, 1)
