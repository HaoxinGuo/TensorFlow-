# encoding = utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# read Data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
# test print Data
x = tf.placeholder(tf.float32,shape = [None,784])
y_ = tf.placeholder(shape = [None,10],dtype = tf.float32)
x_image = tf.reshape(x,[-1,28,28,1])
def Weights_shape(shape):
	initial = tf.truncated_normal(shape,stddev = 1)
	return tf.Variable(initial)

def biases_shape(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides =[1,1,1,1], padding = 'SAME')

def max_pool_2X2(x):
	tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME')

# first conv2d
W_conv1 = Weights_shape([5,5,1,32])
b_conv1 = biases_shape([32])
out_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
out_pool1 = max_pool_2X2(out_conv1)

# second conv2d
W_conv2 = Weights_shape([5,5,32,64])
b_conv2 = biases_shape([64])
out_conv2 = tf.nn.relu(conv2d(out_pool1,W_conv2)+b_conv2)
out_pool2 = max_pool_2X2(out_conv2)
# Full 
W_fc1 = Weights_shape([7*7*64,1024])
b_fcl = biases_shape([1024])
out_pool2_flat = tf.reshape(out_pool2,[-1,7*7*64])
out_fcl = tf.nn.relu(tf.matmul(out_pool2_flat,W_fc1)+b_fcl))

# dropout
keep_prop = tf.placeholder(tf.float32)
h_fcl_prop = tf.nn.dropout(out_fcl,keep_prop)

W_fc2 = Weights_shape([1024,10])
b_fc2 = biases_shape([10])

y_conv = tf.matmul(h_fcl_prop,W_fc2) + b_fc2

## tf.nn.softmax_cross_entropy_with_logits

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv)

train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuray = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess
	sess.run(init_op)
	for _ in range(2000):
		batch= mnist.train.next_batch(50)
		sess.run(train_step,feed_dict = {x:batch[0],y_:batch[1],keep_prop = 0.5})
		if i % 50 == 0:
			train_accuray = sess.run(accuray, feed_dict = {x:batch[0],y:batch[1],keep_prop=1.0})
			print('step %d, train_accuray %g ' % (i,train_accuray)
	if _ == 2000:
		print("test_accuray = %g" % sess.run(accuray,feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prop = 1})






