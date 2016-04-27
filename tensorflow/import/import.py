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

def create_example(name, left, right, zleft, zright):

    rows = left.shape[0]  # height (540)
    cols = left.shape[1]  # width (960)
    depth = left.shape[2] # sRGB (4)

    feature_left = image_feature(left)
    feature_right = image_feature(right)
    feature_zleft = image_feature(zleft)
    feature_zright = image_feature(zright)

    example = tf.train.Example(features=tf.train.Features(feature={
                'name': bytes_feature(bytes(name, 'utf_8')),
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

    left_op = read_image(files_left)
    right_op = read_image(files_right)
    zleft_op = read_image(files_left_z)
    zright_op = read_image(files_right_z)

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
            name = files_left[i]
            example = create_example(name,
                    results[0], results[1], results[2], results[3])
            write_example(writer, example)


        coord.request_stop()
        coord.join(threads)

    writer.close()

def make_dataset(name, indices, files):
    files_left = shuffle(indices, filter_files(files, ".Left.png"))
    files_right = shuffle(indices, filter_files(files, ".Right.png"))
    files_left_z = shuffle(indices, filter_files(files, ".Left-depth.png"))
    files_right_z = shuffle(indices, filter_files(files, ".Right-depth.png"))
    process_inputs(name , files_left, files_right, files_left_z, files_right_z)


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

# indices for the training and test set.
train_indices = indices[:nb_training_sample]
test_indices = indices[nb_training_sample:]

make_dataset("train", train_indices, files)
make_dataset("test", test_indices, files)

