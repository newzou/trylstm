import tensorflow as tf
import numpy as np

x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = 0.2

noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real, x_data.T)+b_real+noise

NUM_STEPS = 30

w = tf.Variable([[0.1, 0.1, 0.1]], dtype="float32")
b = tf.Variable(0.1, dtype="float32")

x = tf.placeholder(shape=[None, 3], dtype="float32")
y = tf.placeholder(shape=None, dtype="float32")

y_pred = tf.matmul(w,tf.transpose(x)) + b

loss = tf.reduce_mean(tf.square(y-y_pred))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(0, NUM_STEPS):
        sess.run(train, {x: x_data, y: y_data})

        if step % 5 == 0:
            print step, sess.run([w, b])
  






