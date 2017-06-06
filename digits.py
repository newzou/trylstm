import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def weight_variable(shape):
    print("weight", shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    print("bias", shape)
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")

def conv_layer(input, shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, w)+b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    w = weight_variable([in_size, size])
    print(in_size, size)
    b = bias_variable([size])
    return tf.matmul(input, w) + b

x = tf.placeholder("float32", shape=[None, 784])
y = tf.placeholder("float32", shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
print("x_image: ", x_image.get_shape())
conv1 = conv_layer(x_image, shape=[5,5,1,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1, shape=[5,5,32,64])
conv2_pool = max_pool_2x2(conv2)

#conv2_flat = tf.reshape(conv2, shape=[-1, 7*7*64])
conv2_flat = tf.contrib.layers.flatten(conv2)
print("conv2_flat: ", conv2_flat.get_shape())

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
print("full1: ", full_1.get_shape())

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
print("full1_drop: ", full1_drop.get_shape())

y_conv = full_layer(full1_drop, 10)
print("y_conv: ", y_conv.get_shape())
    
DATA_DIR = './'
mnist = read_data_sets(DATA_DIR, one_hot=True)

#pred = tf.nn.softmax(y_conv)

#print("y_pred: ", pred.get_shape())

#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#print("cost: ", cost.get_shape())





cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)

correct_prediction=tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

STEPS = 1000 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = mnist.train.next_batch(50)
        #print("batch: ", batch[0].shape, batch[1].shape)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict = {x: batch[0],
                          y: batch[1], keep_prob: 1.0})

            print "step {}, training accuracy {}".format(i, train_accuracy)
 
        sess.run(train_step,feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)

    test_accuracy = np.mean([sess.run(accuracy,
                              feed_dict={x: X[i], y: Y[i], keep_prob:1.0})
                              for i in range(10)])
     
print "test accuracy: {}".format(test_accuracy)

