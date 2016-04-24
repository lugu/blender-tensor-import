#!/usr/bin/env python

import sys

# list of files to read
files = sorted(sys.argv[1:])
print("importing: " + str(files))

def split_files(files, expected_size, pattern):
    l = []
    for file in files:
        if file.endswith(pattern):
            l.append(file)
    if len(l) != expected_size:
        print("invalid number of files")
        sys.exit(-1)
    return l

# splits files in 4 categories (left, right, depth_left, depth_right)
nb_samples = len(files) // 4

files_right = split_files(files, nb_samples, ".Right.png")
files_left = split_files(files, nb_samples, ".Left.png")
files_right_depth = split_files(files, nb_samples, ".Right-depth.png")
files_left_depth = split_files(files, nb_samples, ".Left-depth.png")
 
import tensorflow as tf

def import_files(files, nb_samples):

    filename_queue = tf.train.string_input_producer(files)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_png(value)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(nb_samples): #length of your filename list
            image_data = image.eval() #here is your image Tensor :) 

            print(image_data.shape)

        coord.request_stop()
        coord.join(threads)

import_files(files_right, nb_samples)
