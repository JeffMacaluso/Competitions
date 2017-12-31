from __future__ import division, print_function, absolute_import

import sys
import os
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import time

print('OS: ', sys.platform)
print('Python: ', sys.version)
print('NumPy: ', np.__version__)
print('TensorFlow: ', tf.__version__)

# Checking tensorflow processing devices
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print([x for x in local_device_protos if x.device_type == 'GPU'])

# GPU memory management settings
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Importing the data
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
    print('\nNative data shapes:')
    print('Training set', X_train.shape, y_train.shape)
    print('Validation set', X_validation.shape, y_validation.shape)
    print('Test set', X_test.shape, y_test.shape, '\n')

image_size = 28
num_labels = 10
num_channels = 1  # grayscale

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

X_train, y_train = reformat(X_train, y_train)
X_validation, y_validation = reformat(X_validation, y_validation)
X_test, y_test = reformat(X_test, y_test)

print('Reformatted data shapes:')
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_validation.shape, y_validation.shape)
print('Test set', X_test.shape, y_test.shape, '\n')

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# Training Parameters
learning_rate = 0.001
num_steps = y_train.shape[0] + 1  # 200,000 per epoch
batch_size = 128
epochs = 3
display_step = 250  # To print progress

# Network Parameters
num_input = 784  # Data input (image shape: 28x28)
num_classes = 10  # Total classes (10 characters)

graph = tf.Graph()

with graph.as_default():
    # Input data
    tf_X_train = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_X_validation = tf.constant(X_validation)
    tf_X_test = tf.constant(X_test)

    # Create some wrappers for simplicity
    def maxpool2d(x, k=2):
        """
        Max Pooling wrapper
        """
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def batch_norm(x):
        """
        Batch Normalization wrapper
        """
        return tf.contrib.layers.batch_norm(x, center=True, scale=True, fused=True,)

    def conv2d(data, outputs=32, kernel_size=(5, 5), stride=1, regularization=0.00005):
        """
        Conv2D wrapper, with bias and relu activation
        """
        layer = tf.contrib.layers.conv2d(inputs=data, 
                                         num_outputs=outputs,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding='SAME',
                                         weights_regularizer=tf.contrib.layers.l2_regularizer(scale=regularization),
                                         biases_regularizer=tf.contrib.layers.l2_regularizer(scale=regularization))
        return layer

    # Conv(5,5) -> Conv(5,5) -> MaxPooling -> Conv(3,3) -> Conv(3,3) -> MaxPooling -> FC1024 -> FC1024 -> SoftMax
    def model(x):
        # notMNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Conv(5, 5)
        conv1 = conv2d(x)
        bnorm1 = batch_norm(conv1)

        # Conv(5, 5) -> Max Pooling
        # conv2 = conv2d(bnorm1, weights['wc2'], biases['bc2'])
        conv2 = conv2d(bnorm1, outputs=64)
        bnorm2 = batch_norm(conv2)
        pool1 = maxpool2d(bnorm2, k=2)  # 14x14
        drop1 = tf.nn.dropout(pool1, keep_prob=0.5)

        # Conv(3, 3)
        # conv3 = conv2d(drop1, weights['wc3'], biases['bc3'])
        conv3 = conv2d(drop1, outputs=64, kernel_size=(3, 3))
        bnorm3 = batch_norm(conv3)

        # Conv(3, 3) -> Max Pooling
        # conv4 = conv2d(bnorm3, weights['wc4'], biases['bc4'])
        conv4 = conv2d(bnorm3, outputs=64, kernel_size=(3, 3))
        bnorm4 = batch_norm(conv4)
        pool2 = maxpool2d(bnorm4, k=2)  # 7x7
        drop2 = tf.nn.dropout(pool2, keep_prob=0.5)

        # FC1024
        # Reshape conv2 output to fit fully connected layer input
        # fc1 = tf.reshape(drop2, [-1, weights['wd1'].get_shape().as_list()[0]])
        flatten = tf.contrib.layers.flatten(drop2)
        fc1 = tf.contrib.layers.fully_connected(
            flatten,
            1024,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005),
            biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005),
        )
        drop3 = tf.nn.dropout(fc1, keep_prob=0.5)

        # FC1024
        fc2 = tf.contrib.layers.fully_connected(
            fc1,
            1024,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005),
            biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005),
        )

        # Output, class prediction
        # out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
        out = tf.contrib.layers.fully_connected(
            fc2,
            10,
            activation_fn=None,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005),
            biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00005),
        )
        return out

    # Construct model
    logits = model(tf_X_train)
    
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_train, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    # valid_prediction = tf.nn.softmax(logits.eval(), tf_y_validation)
    # valid_prediction = tf.nn.softmax(model(tf_X_validation, weights, biases))
    # test_prediction = tf.nn.softmax(model(tf_X_test, weights, biases))




# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session(config=config, graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    start_time = time.time()

    for epoch in range(1, epochs+1):
        print('Beginning Epoch {0} -'.format(epoch))

        def next_batch(num, data, labels):
            """
            Return a total of `num` random samples and labels. 
            Mimicks the mnist.train.next_batch() function
            """
            idx = np.arange(0 , len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            data_shuffle = [data[i] for i in idx]
            labels_shuffle = [labels[i] for i in idx]

            return np.asarray(data_shuffle), np.asarray(labels_shuffle)


        for step in range(num_steps):
            batch_data, batch_labels = next_batch(batch_size, X_train, y_train)
            # offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # batch_data = X_train[offset:(offset + batch_size), :, :, :]
            # batch_labels = y_train[offset:(offset + batch_size), :]

            feed_dict = {tf_X_train : batch_data, tf_y_train : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 250 == 0) or (step == num_steps):
                print('Epoch %d Step %d (%.2f%%)' % (epoch, step, ((step/float(num_steps)*100))))
                print('------------------------------------')
                print('Minibatch loss: %f' % l)
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                # print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), y_validation))
                # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), y_test))
                print(datetime.now())
                print('Total execution time: %.2f minutes' % ((time.time() - start_time)/60.))
                print()
    
    # Saver object
    saver = tf.train.Saver()
    saver.save(session, dir_path+'\\'+'tfTestModel')

# To-Do:
# Fix validation/test accuracy
# Add model saving
# Add augmented data