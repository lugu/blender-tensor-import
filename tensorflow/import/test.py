#!/usr/bin/env python

import tensorflow as tf

def decode_image(samples, width, height, depth):
    image = tf.decode_raw(samples, tf.uint8)
    image.set_shape([width * height * depth])
    image = tf.reshape(image, [-1, height, width, depth])
    return image

def read_and_decode(filename_queue):

    # FIXME: read those values from the features
    width = 960
    height = 540
    in_depth = 4
    out_depth = 1

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

    left = decode_image(samples['image_left'], width, height, in_depth)
    right = decode_image(samples['image_right'], width, height, in_depth)
    zleft = decode_image(samples['zimage_left'], width, height, out_depth)
    zright = decode_image(samples['zimage_right'], width, height, out_depth)

    return left, right, zleft, zright


def inputs(filename):
    # num_epochs tells how many times we can read the input data
    filename_queue = tf.train.string_input_producer([filename])
    left, right, zleft, zright = read_and_decode(filename_queue)

    return left, right, zleft, zright


with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    sum_writer = tf.train.SummaryWriter("sum", sess.graph_def)

    input_file = "dataset/train.tfrecords"
    features = inputs(input_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        print(str(i))
        left, right, zleft, zright = sess.run(features)

        left_sum = tf.image_summary("left", left)
        right_sum = tf.image_summary("right", right)
        zleft_sum = tf.image_summary("zleft", zleft)
        zright_sum = tf.image_summary("zright", zright)

        sum_ops = [ left_sum , right_sum, zleft_sum, zright_sum ]
        summary = sess.run(sum_ops)

        sum_writer.add_summary(summary[0], i)
        sum_writer.add_summary(summary[1], i)
        sum_writer.add_summary(summary[2], i)
        sum_writer.add_summary(summary[3], i)

    coord.request_stop()
    coord.join(threads)
