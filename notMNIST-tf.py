""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import sys
import os
import numpy as np
import tensorflow as tf
import pickle

print('OS: ', sys.platform)
print('Python: ', sys.version)
print('NumPy: ', np.__version__)
print('TensorFlow: ', tf.__version__)

# Checking tensorflow processing devices
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print([x for x in local_device_protos if x.device_type == 'GPU'])

dir_path = os.path.dirname(os.path.realpath(__file__))
pickle_file = 'notMNIST.pickle'

with open(dir_path+'\\'+pickle_file, 'rb') as f:
    save = pickle.load(f, encoding='iso-8859-1')
    X_train = save['train_dataset']
    y_train = save['train_labels']
    X_validation = save['valid_dataset']
    y_validation = save['valid_labels']
    X_test = save['test_dataset']
    y_test = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', X_train.shape, y_train.shape)
    print('Validation set', X_validation.shape, y_validation.shape)
    print('Test set', X_test.shape, y_test.shape)

image_size = 28
num_labels = 10
num_channels = 1  # grayscale

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 1001
batch_size = 128
display_step = 500

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def batch_norm(x):
    """
    Convenience function for batch normalization
    """
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, fused=True,)


# Conv(5,5) -> Conv(5,5) -> MaxPooling -> Conv(3,3) -> Conv(3,3) -> MaxPooling -> FC1024 -> FC1024 -> SoftMax
def model(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Conv(5, 5)
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    bnorm1 = batch_norm(conv1)

    # Conv(5, 5) -> Max Pooling
    conv2 = conv2d(bnorm1, weights['wc2'], biases['bc2'])
    bnorm2 = batch_norm(conv2)
    pool1 = maxpool2d(bnorm2, k=2)  # 14x14
    drop1 = tf.nn.dropout(pool1, keep_prob=0.5)

    # Conv(3, 3)
    conv3 = conv2d(drop1, weights['wc3'], biases['bc3'])
    bnorm3 = batch_norm(conv3)

    # Conv(3, 3) -> Max Pooling
    conv4 = conv2d(bnorm3, weights['wc4'], biases['bc4'])
    bnorm4 = batch_norm(conv4)
    pool2 = maxpool2d(bnorm4, k=2)  # 7x7
    drop2 = tf.nn.dropout(pool2, keep_prob=0.5)

    # FC1024
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(drop2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)  # Activation
    fc1 = tf.nn.dropout(fc1, dropout)  # Dropout

    # FC1024
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)  # Activation

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # fully connected, 1024, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = model(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Start training
with tf.Session(config=config) as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))