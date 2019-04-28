# encoding = utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def add_layers(value,inputs,outputs,activation_function = None):
	Weights = tf.Variable(tf.random_normal([inputs, outputs], stddev=0.35),name="weights")
	biases = tf.Variable(tf.zeros([1,outputs]))
	# Weights = tf.get_variable(shape = [inputs,outputs],name = 'Weights', tf.random_normal_initializer(mean = 0, stddev = 1))
	# biases = tf.get_variable(shape = [1,outputs],name = 'Weights', tf.zeros_initializer)
	# Out1 = tf.matmul(value,Weights) + biases
	if activation_function == None:
		return tf.matmul(value,Weights) + biases
	else:
		return activation_function(tf.matmul(value,Weights) + biases)


xs = tf.placeholder(dtype = tf.float32,shape = [None, 1], name = 'xs')
ys = tf.placeholder(dtype = tf.float32,shape = [None, 1], name = 'ys')

x_data = np.linspace(-1, 2, num=300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.power(x_data,3) + np.square(x_data) - 0.5 + noise


l1 = add_layers(xs,1,10,activation_function = tf.nn.relu)
prediction = add_layers(l1,10,1,activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init_op = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x_data, y_data)
plt.ion()
plt.show()
lines = None
time.sleep(4)

with tf.Session() as sess:
	sess.run(init_op)
	for i in range(1000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 50 == 0:
		# to see the step improvement
			print('loss = ',sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			prediction_value = sess.run(prediction, feed_dict={xs: x_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
			plt.pause(0.1)
plt.ioff()
plt.show()

