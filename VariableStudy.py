# encoding = utf-8
import tensorflow as tf
a = tf.constant(value=[1,2],dtype = tf.float32, shape = (1,2),name = 'a')
b = tf.Variable(initial_value = tf.random_normal([2, 3],stddev = 1),name = 'b',dtype = tf.float32)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
