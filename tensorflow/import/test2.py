#!/usr/bin/env python

import os.path

import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('directory', 'dataset',
                    'Directory with the datasets.')
flags.DEFINE_string('summary', 'summary', 'Directory to output the result summaries')

TRAIN_FILE = 'train.tfrecords'


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_left': tf.FixedLenFeature([], tf.string),
          'image_right': tf.FixedLenFeature([], tf.string),
      })

  image_left = tf.decode_raw(features['image_left'], tf.uint8)
  image_right = tf.decode_raw(features['image_right'], tf.uint8)
  width = 960
  height = 540
  depth = 4
  image_left.set_shape([width*height*depth])
  image_right.set_shape([width*height*depth])

  return image_left, image_right


def inputs(batch_size, num_epochs):
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.directory, TRAIN_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    left_images, right_images = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    left, right= tf.train.shuffle_batch(
        [left_images, right_images], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return left, right


def run_training():
  """Train MNIST for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    left, right = inputs(batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    width = 960
    height = 540
    depth = 4
    first_left = tf.reshape(left, [-1, height, width, depth])
    first_right = tf.reshape(right, [-1, height, width, depth])

    tf.image_summary("left", first_left, 10)
    tf.image_summary("right", first_right, 10)

    # The op for initializing the variables.
    init_op = tf.initialize_all_variables()

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # a summary is attached to each image
    sum_ops = tf.merge_all_summaries()
    sum_writer = tf.train.SummaryWriter(FLAGS.summary, sess.graph_def)

    step = 0
    try:
      while not coord.should_stop():
        step += 1
        summary = sess.run(sum_ops)
        sum_writer.add_summary(summary, step)

    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
