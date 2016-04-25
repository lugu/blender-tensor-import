#!/usr/bin/env python

import tensorflow as tf

def decode_image(samples, label, width, height, depth):
    image = tf.decode_raw(samples[label], tf.uint8)
    image.set_shape([width * height * depth])
    image = tf.reshape(image, [-1, height, width, depth])
    tf.image_summary(label, image, 10)
    return image

def read_and_decode(filename_queue):

    # FIXME: read those values from the features
    width = 960
    height = 540
    in_depth = 4
    out_depth = 4

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

    left = decode_image(samples, 'image_left', width, height, in_depth)
    right = decode_image(samples, 'image_right', width, height, in_depth)
    zleft = decode_image(samples, 'zimage_left', width, height, out_depth)
    zright = decode_image(samples, 'zimage_right', width, height, out_depth)

    return left, right, zleft, zright


def inputs(filename):
    # num_epochs tells how many times we can read the input data
    filename_queue = tf.train.string_input_producer([filename])
    left, right, zleft, zright = read_and_decode(filename_queue)

    return left, right, zleft, zright


# display an numpy array representing an image
def show(image):
    import matplotlib.pyplot as plt
    import numpy as np
    print(str(image.shape))
    plt.imshow(np.squeeze(image))
    plt.show()


with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    sum_writer = tf.train.SummaryWriter("sum", sess.graph_def)

    input_file = "dataset/train.tfrecords"

    # tensors for each images
    left, right, zleft, zright = inputs(input_file)

    # a summary is attached to each image
    sum_ops = tf.merge_all_summaries()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        print(str(i))

        summary = sess.run(sum_ops)
        sum_writer.add_summary(summary, i)

        # display the numpy arrays representing the images
        features = [ left, right, zleft, zright ]
        for data in sess.run(features):
            show(data)

    coord.request_stop()
    coord.join(threads)
