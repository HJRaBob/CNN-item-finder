import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
sess = tf.Session()

batch_size = 4
data_dir = 'img/background'
output_every = 50
generation = 625
eval_every = 500
image_height = 354
image_width = 623
crop_height = 256
crop_width = 256
num_channels = 3
num_targets = 10
extract_folder = 'cifar-10-batches-bin'

learning_rate = 0.1
lr_decay = 0.9
num_gens_to_wait = 250

image_vec_length = image_height*image_width*num_channels
record_length = 1+image_vec_length

def read_cifar_files(filename_queue, distort_images = True):
    reader = tf.FixedLengthRecordReader(record_bytes = record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string,tf.uint8)
    image_label = tf.cast(tf.slice(record_bytes,[0],[1]),tf.int32)

    # load image from string and restore to image's format
    imgae_extracted = tf.reshape(tf.slice(record_bytes, [1],[image_vec_length]),[num_channels,image_height,image_width])


    image_uint8image = tf.transpose(imgae_extracted,[1,2,0]) #change order of Tensor [image_height, image_width, num_channels]
    reshaped_image = tf.cast(image_uint8image, tf.float32) #convert to float
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,crop_width,crop_height) #crop image

    if distort_images:
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image,max_delta = 63)
        final_image = tf.image.random_contrast(final_image, lower = 0.2, upper = 1.8)

    final_image = tf.image.per_image_standardization(final_image)
    return (final_image,image_label)

def input_pipeline(batch_size,train_logical = True):
    if train_logical:
        files = ['img/background/back_{}.png'.format(i) for i in range(0,1000)]

    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)
    print('first image: '+str(image.shape))

    min_after_dequeue = 5000 #important to memory share recommended (number of thread+range of error)*batch size
    capacity = min_after_dequeue + 3*batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)

    return (example_batch, label_batch)


def cifar_cnn_model(input_images, batch_size, train_logical = True):
    def truncated_nomal_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype,initializer=tf.truncated_normal_initializer(stddev=0.05)))

    def zero_var(name,shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype,initializer=tf.constant_initializer(0.0)))

    with tf.variable_scope('conv1') as scope:
        conv1_kernel = truncated_nomal_var(name='conv_kernel1',shape=[5,5,3,64],dtype=tf.float32)
        conv1 = tf.nn.conv2d(input_images,conv1_kernel,[1,1,1,1],padding='SAME')
        conv1_bias = zero_var(name='conv_bias1',shape=[64],dtype=tf.float32) #?????
        conv1_add_bias = tf.nn.bias_add(conv1,conv1_bias)
        relu_conv1 = tf.nn.relu(conv1_add_bias)
        pool1 = tf.nn.max_pool(relu_conv1, ksize=[1,3,3,1],strides = [1,2,2,1],padding='SAME',name='pool_layer1')

    with tf.variable_scope('conv2') as scope:
        conv2_kernel = truncated_nomal_var(name='conv_kernel2',shape=[5,5,3,64],dtype=tf.float32)
        conv2 = tf.nn.conv2d(input_images,conv2_kernel,[1,1,1,1],padding='SAME')
        conv2_bias = zero_var(name='conv_bias2',shape=[64],dtype=tf.float32) #?????
        conv2_add_bias = tf.nn.bias_add(conv2,conv2_bias)
        relu_conv2 = tf.nn.relu(conv2_add_bias)
        pool2 = tf.nn.max_pool(relu_conv2, ksize=[1,3,3,1],strides = [1,2,2,1],padding='SAME',name='pool_layer2')
    
    norm2 = tf.nn.lrn(pool2,depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75,name='norm2')
    
    reshaped_output = tf.reshape(norm2, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    with tf.variable_scope('full1') as scope:
        full_weight1 = truncated_nomal_var(name='full_mult1',shape=[reshaped_dim,384],dtype=tf.float32)
        full_bias1 = zero_var(name = 'full_bias1',shape=[384],dtype = tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight1),full_bias1))

    
    with tf.variable_scope('full2') as scope:
        full_weight2 = truncated_nomal_var(name='full_mult2',shape=[384,192],dtype=tf.float32)
        full_bias2 = zero_var(name = 'full_bias2',shape=[192],dtype = tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1,full_weight2),full_bias2))

    with tf.variable_scope('full3') as scope:
        full_weight3 = truncated_nomal_var(name='full_mult3',shape=[192,num_targets],dtype=tf.float32)
        full_bias3 = zero_var(name = 'full_bias3',shape=[num_targets],dtype = tf.float32)
        final_output = tf.nn.relu(tf.add(tf.matmul(full_layer2,full_weight3),full_bias3))

    return final_output

def cifar_loss(logits, targets):
    targets = tf.squeeze(tf.cast(targets,tf.int32))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = targets)
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    return(cross_entropy_mean)

def train_step(loss_value,generation_num):
    model_learning_rate = tf.train.exponential_decay(learning_rate,generation_num,num_gens_to_wait,lr_decay,staircase=True)
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    train_step = my_optimizer.minimize(loss_value)
    return(train_step)

def accuracy_of_batch(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits,1),tf.int32)
    predicted_correctly = tf.equal(batch_predictions,targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly,tf.float32))
    return accuracy

images, targets = input_pipeline(batch_size,train_logical=True)
test_images, test_targets = input_pipeline(batch_size,train_logical=False)

with tf.variable_scope('model_definition') as scope:
    model_output = cifar_cnn_model(images,batch_size)
    scope.reuse_variables()
    test_output=cifar_cnn_model(test_images,batch_size)

loss = cifar_loss(model_output,targets)
accuracy = accuracy_of_batch(test_output,test_targets)
generation_num = tf.Variable(0,trainable=False)
train_op = train_step(loss,generation_num)

init=tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

train_loss = []
test_accuracy = []

for i in range(generation):
    _,loss_value = sess.run([train_op,loss])

    if(i+1)%output_every ==0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {:.5f}'.format((i+1),loss_value)
        print(output)

    if(i+1)%eval_every ==0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = '-----Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
        print(acc_output)


        