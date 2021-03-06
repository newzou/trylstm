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

LOG_DIR='/home/zpan/logdir'

#This helper function taken from official TensorFlow documentation,
# simply adds some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


element_size = 5 
time_steps = 28
num_classes = 2 
batch_size = 128
hidden_layer_size = 128

_inputs = tf.placeholder("float32", 
                         shape=[None, time_steps, element_size], 
                         name="input")

y = tf.placeholder("float32", shape=[None, 2], name="inputs")

with tf.name_scope("rnn_weights"):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.truncated_normal([element_size, hidden_layer_size],mean=0,stddev=1))
        
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.truncated_normal([hidden_layer_size,hidden_layer_size], mean=0, stddev=1))

    with tf.name_scope("bias"):
        b = tf.Variable(tf.truncated_normal([hidden_layer_size], mean=0, stddev=1))

 
def run_rnn(pre_state, x1):
    return tf.tanh(tf.matmul(pre_state, Wh)+tf.matmul(x1, Wx)+b)    

processed_input = tf.transpose(_inputs, perm=[1,0,2]) 

initial_hidden = tf.truncated_normal([batch_size, hidden_layer_size], mean=0, stddev=1)

all_hidden_layers = tf.scan(run_rnn, 
                            processed_input,
                            initializer=initial_hidden,
                            name="state")

#rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1)

outputs, states = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=1))

bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=1))

def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

last_hidden_state = outputs[:,-1,:]

output = get_linear_layer(last_hidden_state)

cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))

tf.summary.scalar('cross_entropy', cross_entropy)

#train_step = tf.train.RMSPropOptimizer(0.0000001,0.9).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(output,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

DATA_DIR = './'

from numpy import genfromtxt

daily_price = genfromtxt('IBM.csv', delimiter=',', skip_header=1)

from sklearn import preprocessing

dt = daily_price[:,0:1]

price = np.column_stack((daily_price[:,1:5], daily_price[:,6:]))


# prepare y [1,0] buy  [0,1] not buy
y_data = np.zeros((dt.shape[0], 2))

k = dt.shape[0]
NUM_SAMPLES = y_data.shape[0]
for i in range(k):
     if i+5 < k and price[i,3] < price[i+5,3]:
         y_data[i][0] = 1
     else:
         y_data[i][1] = 1

scaler = preprocessing.StandardScaler().fit(price)

price = scaler.transform(price)

#price1 = tf.nn.l2_normalize(price, 0)

processed_x = np.zeros((price.shape[0]-27-5, time_steps, price.shape[1]))

for i in range(0, processed_x.shape[0]):
    processed_x[i] = price[i:i+28,:] 

y_data = y_data[27:y_data.shape[0]-5]

import math
STEPS = int(y_data.shape[0] / batch_size) 

test_x = processed_x[(STEPS-2)*batch_size:(STEPS-1)*batch_size]

test_y =  y_data[(STEPS-2)*batch_size:(STEPS-1)*batch_size]

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())


    for i in range(STEPS-1):
        batch_x = processed_x[i*batch_size:(i+1)*batch_size]
       
        batch_y = y_data[i*batch_size:(i+1)*batch_size]
        #print(batch_x[-2:-1,-2:-1,:], batch_y[-2:-1,:])

        train_accuracy = sess.run(accuracy, feed_dict = {_inputs: test_x,
                       y: test_y})


        print "step {0}, training accuracy {1}".format(i, train_accuracy)
 
        m1 = np.mean(batch_x, axis=0)
        m2 = np.mean(m1, axis=0)
        #print(m2)
        _,t = sess.run([train_step, cross_entropy],feed_dict={_inputs: batch_x, y: batch_y})
        tf.summary.scalar('cross_entropy', t)
        print t
        #train_writer.add_summary(summary,i)

    #X = mnist.test.images.reshape(10, 1000, 784)
    #Y = mnist.test.labels.reshape(10, 1000, 10)

    #test_accuracy = np.mean([sess.run(accuracy,
#                              feed_dict={x: X[i], y: Y[i], keep_prob:1.0})
#                              for i in range(10)])
     
#print "test accuracy: {}".format(test_accuracy)


