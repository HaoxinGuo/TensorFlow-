# encoding = utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# read Data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
# test print Data
x = tf.placeholder(tf.float32,shape = [None,784])
W = tf.Variable(tf.random.normal([784,10],stddev = 1),name = 'W',dtype = tf.float32)
b = tf.Variable(tf.zeros([1,10])+0.1,dtype = tf.float32)
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(shape = [None,10],dtype = tf.float32)
loss = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for _ in range(1000):
    batch_xs ,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict = {x:batch_xs,y_:batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuray = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuray,feed_dict = {x:mnist.test.images,y_:mnist.test.labels}))

