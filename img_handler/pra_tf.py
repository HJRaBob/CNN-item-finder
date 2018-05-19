import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib

import cv2 as cv
sess = tf.Session()

batch_size = 4
data_dir = 'img/background/'
output_every = 50
generation = 625
eval_every = 500
image_height = 354
image_width = 623
crop_height = 256
crop_width = 256
num_channels = 3
num_targets = 10

learning_rate = 0.1
lr_decay = 0.9
num_gens_to_wait = 250

image_vec_length = image_height*image_width*num_channels
record_length = 1+image_vec_length

def read_image_files(filename_queue, distort_images = True):
    reader = tf.FixedLengthRecordReader(record_bytes = image_vec_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string,tf.uint8)
    print("hello")
    print(record_bytes.shape)

    image_uint8image = tf.transpose(filename_queue,[1,2,0]) #change order of Tensor [image_height, image_width, num_channels]
    reshaped_image = tf.cast(image_uint8image, tf.float32) #convert to float
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,crop_width,crop_height) #crop image

    if distort_images:
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image,max_delta = 63)
        final_image = tf.image.random_contrast(final_image, lower = 0.2, upper = 1.8)

    final_image = tf.image.per_image_standardization(final_image)
    print(final_image.shape)
    return (final_image)

def input_pipeline(batch_size,train_logical = True):
    image_queue = [cv.imread(data_dir+'back_{}.png'.format(i)) for i in range(0,10)]
    image = read_image_files(image_queue)
    label = [1,0,0,0,0,0,0]
    # print(image.shape)

    min_after_dequeue = 5000 #important to memory share recommended (number of thread+range of error)*batch size
    capacity = min_after_dequeue + 3*batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)

    return (example_batch, label_batch)

images, targets = input_pipeline(batch_size,train_logical=True)
