import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.layers import fully_connected
from numpy import genfromtxt
from sklearn import preprocessing
import matplotlib.pyplot as plt 

LOG_DIR='/home/zpan/logdir'

n_steps = 20
n_inputs = 1 
n_neurons = 100
n_outputs = 1 

batch_size = 1 

learning_rate = 0.0001

X = tf.placeholder("float32", shape=[None, n_steps, n_inputs], name="input")
y = tf.placeholder("float32", shape=[None, n_steps, n_outputs], name="inputs")

# axis 0 should be the time steps 
#processed_X = tf.transpose(X, perm=[1,0,2]) 

#cell = tf.contrib.rnn.BasicLSTMCell(n_neurons, forget_bias=1)
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op=optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 1000

# get data
DATA_DIR = './'
daily_price = genfromtxt('IBM.csv', delimiter=',', skip_header=1)
dt = daily_price[:,0:1]
price = daily_price[:,5] # adjusted last price

target = price[1:] 
price = price[:-1]

n_prices = price.shape[0]
n_train_instances = n_prices/n_steps
n_prices = n_train_instances*n_steps 

price = price[0:n_prices].reshape(n_train_instances,n_steps,n_inputs)
target = target[0:n_prices].reshape(n_train_instances,n_steps,n_inputs)

n_batches = n_train_instances/batch_size

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_batches):
        X_batch = price[i*batch_size:i*batch_size+batch_size]
        y_batch = target[i*batch_size:i*batch_size+batch_size]

        sess.run([train_op, loss], feed_dict={X:X_batch, y:y_batch})
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
    

        if i % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print(i, "\tmse: ", mse)
            plt.plot(X_batch[-1])
            plt.plot(y_pred[-1])
            plt.show()
        
    
        


