#!/usr/bin/env python

import tensorflow as tf

with tf.Session() as sess:
    c = tf.constant("Hello, world!")
    print(str(sess.run(c)))
