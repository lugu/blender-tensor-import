#!/usr/bin/env python

import tensorflow as tf

def decode_image(samples, label, width, height, depth):
    image = tf.decode_raw(samples[label], tf.uint8)
    image.set_shape([width * height * depth])
    image = tf.reshape(image, [-1, height, width, depth])
    tf.image_summary(label, image, 1)
    return image

def open_dataset(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    return example

def extract_features(samples, features):
    return tf.parse_single_example(samples, features)

def read_and_decode(samples):

    features={
        'name': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'depth': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image_left': tf.FixedLenFeature([], tf.string),
        'image_right': tf.FixedLenFeature([], tf.string),
        'zimage_left': tf.FixedLenFeature([], tf.string),
        'zimage_right': tf.FixedLenFeature([], tf.string),
    }
    example = extract_features(samples, features)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    print(" ===== ")
    print(type(example))
    height = example['height'].eval()
    width = example['width'].eval()
    depth = example['depth'].eval()

    # width, height, depth = read_images_size(filename)
    # width, height, depth = [960, 540, 4]
    out_depth = 4

    left = decode_image(example, 'image_left', width, height, depth)
    right = decode_image(example, 'image_right', width, height, depth)
    zleft = decode_image(example, 'zimage_left', width, height, out_depth)
    zright = decode_image(example, 'zimage_right', width, height, out_depth)

    return left, right, zleft, zright


# display an numpy array representing an image
def show(image):
    import matplotlib.pyplot as plt
    import numpy as np
    print(str(image.shape))
    plt.imshow(np.squeeze(image))
    plt.show()


tf.app.flags.DEFINE_string('dataset', 'dataset/train.tfrecords',
                           'Directory to download data files and write the converted result')
tf.app.flags.DEFINE_string('summary', 'sum', 'Directory to output the result summaries')
FLAGS = tf.app.flags.FLAGS

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    sum_writer = tf.train.SummaryWriter(FLAGS.summary, sess.graph_def)

    samples = open_dataset(FLAGS.dataset)
    # tensors for each images
    left, right, zleft, zright = read_and_decode(samples)

    # a summary is attached to each image
    sum_ops = tf.merge_all_summaries()

    for i in range(100):
        print(str(i))

        summary = sess.run(sum_ops)
        sum_writer.add_summary(summary, i)

        # display the numpy arrays representing the images
        features = [ left, right, zleft, zright ]
        # for data in sess.run(features):
        #   show(data)

