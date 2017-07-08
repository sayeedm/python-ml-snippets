'''
    adding output of ReLUs
    ReLU - Rectified Linear Unit
    Computes Linear Function of inputs and gives the output result
    if positive, else 0


    Note: its not complete yet
    Author:SayeedM
    Date: 08/07/2017
'''

import tensorflow as tf

def relu(X):
    with tf.variable_scope("relu", reuse = True):
        threshold = tf.get_variable("threshold", shape=(), initializer = tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name = "weights")
        b = tf.Variable(0.0, name = "bias")
        z = tf.add(tf.matmul(X, w), b, name = "z")
        return tf.maximum(z, 0, name = "relu")


n_features = 3
X = tf.placeholder(tf.float32, shape = (None, n_features), name = "X")

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))

relus = [relu(X) for i in range(5)]

summary_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())
output = tf.add_n(relus, name = "output")
