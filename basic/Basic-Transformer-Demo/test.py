#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


# [batch, in_height, in_width, in_channels]
inputs = np.arange(12, dtype=np.float32).reshape([1,3,4,1])
print inputs
# [filter_height, filter_width, in_channels, out_channels], 第四个参数相当于卷积核的个数
filters = np.array([1,2,3,4,5,6,7,8], dtype=np.float32).reshape([2,2,1,2])
print filters[0,:,:,:]
strides = [1,1,1,1]
padding = "VALID"


#
outputs = tf.nn.conv2d(inputs, filters, strides, padding)
# conv1d_inputs = np.squeeze(inputs, axis=-1)
# conv1d_filters = np.squeeze(filters, axis=-1)
# conv1d_outputs = tf.nn.conv1d(conv1d_inputs, conv1d_filters, stride=1, padding="VALID")

with tf.get_default_graph().as_default():
  sess = tf.Session()
  result = sess.run(outputs)
  print np.shape(result)
  print result
  #
  # conv1d_result = sess.run(conv1d_outputs)
  # print np.shape(conv1d_result)
  # print conv1d_result
