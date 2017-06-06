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

element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

_inputs = tf.placeholder("float32", shape=[None, time_steps, element_size], 
                    name="input")

y = tf.placeholder("float32", shape=[None, 10], name="inputs")

with tf.name_scope("rnn_weights"):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size,hidden_layer_size]))

    with tf.name_scope("bias"):
        b = tf.Variable(tf.zeros([hidden_layer_size]))

 
def run_rnn(pre_state, x):
    return tf.tanh(tf.matmul(pre_state, Wh)+tf.matmul(x, Wx)+b)    


processed_input = tf.transpose(_inputs, perm=[1,0,2]) 

initial_hidden = tf.zeros([batch_size, hidden_layer_size])

all_hidden_layers = tf.scan(run_rnn, 
                            processed_input,
                            initializer=initial_hidden,
                            name="state")


Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=0.01))

bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01))

def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

all_outputs = tf.map_fn(get_linear_layer, all_hidden_layers)
output = all_outputs[-1]

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y))

train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)


correct_prediction=tf.equal(tf.argmax(output,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100


DATA_DIR = './'
mnist = read_data_sets(DATA_DIR, one_hot=True)

STEPS = 5000 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        print type(batch_x)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict = {_inputs: batch_x,
                          y: batch_y})

            print "step {}, training accuracy {}".format(i, train_accuracy)
 
        sess.run(train_step,feed_dict={_inputs: batch_x, y: batch_y})

    #X = mnist.test.images.reshape(10, 1000, 784)
    #Y = mnist.test.labels.reshape(10, 1000, 10)

    #test_accuracy = np.mean([sess.run(accuracy,
#                              feed_dict={x: X[i], y: Y[i], keep_prob:1.0})
#                              for i in range(10)])
     
#print "test accuracy: {}".format(test_accuracy)

