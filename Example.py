# encoding = utf-8
# http://www.omegaxyz.com/2018/04/19/2nn_tensorflow_py/
def add_layer(inputs,in_size,out_size,activation_function=None):
	weigths = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size] + 0.1))
	Wx_plus_b = tf.matmul(inputs,weigths)+biases
	if activation_function = None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplot.pyplot as plt
import time

# ���ݼ������ɣ�����np������-1~2��300����
x_data = np.linspace(-1, 2, num=300)[:, np.newaxis]
# �������
noise = np.random.normal(0, 0.05, x_data.shape)
# ����Y���ֵ
y_data = np.square(x_data) - 0.5 + noise

# ���ռλ��
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# ������ز�
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
# ��������
prediction = add_layer(l1,10,1,activation_function = None)
# �ɴ�����������������
# �����������ݶ��½�ʹ�������С
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# import step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# ����ԭʼֵ
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
lines = None
time.sleep(4)
# ��ʾ���
for i in range(1000):
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
	try:
		ax.lines.remove(lines[0])
	except Exception:
		pass
	prediction_value = sess.run(prediction, feed_dict={xs: x_data})
	lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
	plt.pause(0.1)
# writer = tf.train.SummaryWriter("logs/", sess.graph)




