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

total = 1000000.0
cash = 1000000.0
shares = 0
n_horizon=4
close_date = None

n_prices = price.shape[0]

X_batch = np.zeros((1,n_steps, n_inputs))
y_batch = np.zeros((1,n_steps, n_inputs))

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_prices-500, n_prices):
        X_batch[0,:,0] = price[i+1-n_steps-n_horizon: i+1-n_horizon]
        y_batch[0,:,0] = price[i+1-n_steps: i+1]
      
        sess.run([train_op, loss], feed_dict={X:X_batch, y:y_batch})

        y_pred = sess.run(outputs, feed_dict={X: y_batch})

        target = y_pred[0,-1,0]
        if target > price[i]*1.03:
            if shares == 0:
                shares = int(cash/price[i])
                cash = cash - price[i]*shares
                print("{} buy {} shares + cash {} = total {}".format(dt[i],
                     shares, cash, cash+shares*price[i]))

            close_date = i + n_horizon

        if (close_date == i or target < price[i]*0.97) and shares > 0:
            print("{} sell {} shares + cash {} = total {}".format(dt[i],
                     shares, cash, cash+shares*price[i]))
            cash += price[i]*shares
            total = cash
            shares = 0
            
        
    
           # mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
           # print(i, "\tmse: ", mse)
           # plt.plot(X_batch[-1])
           # plt.plot(y_pred[-1])
           # plt.show()
        

print("return: {}".format((total/1000000.0-1)*100))
    
        


