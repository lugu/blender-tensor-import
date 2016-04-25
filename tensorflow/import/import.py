#!/usr/bin/env python

import sys
import os
import random
import tensorflow as tf

# shuffle order a list according to the indexes of indices.
def shuffle(indices, files):
    return [files[i] for i in indices]

# select files if the end with  'pattern'.
def filter_files(files, pattern):
    l = []
    for file in files:
        if file.endswith(pattern):
            l.append(file)
    return l


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_feature(image):
    return bytes_feature(image.tostring())

def create_example(left, right, zleft, zright):

    rows = left.shape[0]  # height (540)
    cols = left.shape[1]  # width (960)
    depth = left.shape[2] # sRGB (4)

    feature_left = image_feature(left)
    feature_right = image_feature(right)
    feature_zleft = image_feature(zleft)
    feature_zright = image_feature(zright)

    example = tf.train.Example(features=tf.train.Features(feature={
                'height': int64_feature(rows),
                'width': int64_feature(cols),
                'depth': int64_feature(depth),
                'image_left': feature_left,
                'image_right': feature_right,
                'zimage_left': feature_zleft,
                'zimage_right': feature_zright}))

    return example


def open_writer(name):
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    # writer.close()
    return writer

def write_example(writer, example):
    writer.write(example.SerializeToString())

def read_image(files):
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    return image

def process_inputs(dataset_name, files_left, files_right, files_left_z, files_right_z) : 

    print("dataset: " + dataset_name)
    writer = open_writer(dataset_name)
    nb_examples = len(files_right)

    left_op = read_image(files_right)
    right_op = read_image(files_right)
    zleft_op = read_image(files_right)
    zright_op = read_image(files_right)

    images_ops = [ left_op, right_op, zleft_op, zright_op ]

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(nb_examples):
            print("processing image " + str(i))
            results = sess.run(images_ops)
            example = create_example(results[0], results[1], results[2], results[3])
            write_example(writer, example)


        coord.request_stop()
        coord.join(threads)

    writer.close()


tf.app.flags.DEFINE_string('directory', 'dataset',
                           'Directory to download data files and write the converted result')
FLAGS = tf.app.flags.FLAGS

# Ordered list of files (so that left, right, zleft and zright follow
# each other)
files = sorted(sys.argv[1:])

# the total number of sample is the number of images divided by 4.
nb_samples = len(files) // 4

# to randomize the samples create a random list of indices
indices = random.sample(range(nb_samples), nb_samples)

# number of files in the training set
nb_training_sample = int(nb_samples * 0.70)

# indices for the training set.
train_indices = indices[:nb_training_sample]

# trainning samples images
train_files_left = shuffle(train_indices, filter_files(files, ".Left.png"))
train_files_right = shuffle(train_indices, filter_files(files, ".Right.png"))
train_files_left_z = shuffle(train_indices, filter_files(files, ".Left-depth.png"))
train_files_right_z = shuffle(train_indices, filter_files(files, ".Right-depth.png"))

process_inputs("train", train_files_left, train_files_right, train_files_left_z, train_files_right_z)

# indices for the test set.
test_indices = indices[nb_training_sample:]

# test samples images
test_files_left = shuffle(test_indices, filter_files(files, ".Left.png"))
test_files_right = shuffle(test_indices, filter_files(files, ".Right.png"))
test_files_left_z = shuffle(test_indices, filter_files(files, ".Left-depth.png"))
test_files_right_z = shuffle(test_indices, filter_files(files, ".Right-depth.png"))

process_inputs("test", test_files_left, test_files_right, test_files_left_z, test_files_right_z)

