#!/usr/bin/env python

import sys
import os
import tensorflow as tf

# list of files to read
files = sorted(sys.argv[1:])
# print("importing: " + str(files))

def split_files(files, expected_size, pattern):
    l = []
    for file in files:
        if file.endswith(pattern):
            l.append(file)
    if len(l) != expected_size:
        print("invalid number of files")
        sys.exit(-1)
    return l


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_bytes(image):
    return _bytes_feature(image.tostring())

def create_example(left, right, zleft, zright):

    rows = left.shape[0] # FIXME: or 1, 2, 3 ?
    cols = left.shape[1]
    depth = left.shape[2]

    image_left = image_bytes(left)
    image_right = image_bytes(right)
    zimage_left = image_bytes(zleft)
    zimage_right = image_bytes(zright)

    example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'image_left': image_left,
                'iamge_right': image_right,
                'zimage_left': image_left,
                'zimage_right': image_left}))

    return example


def open_writer(name):
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    # writer.close()
    return writer

def write_example(writer, example):
    print('Writing...')
    writer.write(example.SerializeToString())

# splits files in 4 categories (left, right, depth_left, depth_right)
nb_samples = len(files) // 4

files_right = split_files(files, nb_samples, ".Right.png")
files_left = split_files(files, nb_samples, ".Left.png")
files_right_z = split_files(files, nb_samples, ".Right-depth.png")
files_left_z = split_files(files, nb_samples, ".Left-depth.png")
 
tf.app.flags.DEFINE_string('directory', 'dataset',
                           'Directory to download data files and write the converted result')
FLAGS = tf.app.flags.FLAGS


def read_image(files):
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    return image


init_op = tf.initialize_all_variables()
writer = open_writer("train")

left_op = read_image(files_right)
right_op = read_image(files_right)
zleft_op = read_image(files_right)
zright_op = read_image(files_right)

images_ops = [ left_op, right_op, zleft_op, zright_op ]

with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(nb_samples):

        results = sess.run(images_ops)
        example = create_example(results[0], results[1], results[2], results[3])
        write_example(writer, example)


    coord.request_stop()
    coord.join(threads)

writer.close()

