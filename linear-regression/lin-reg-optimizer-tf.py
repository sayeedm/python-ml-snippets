'''
    linear regression batch gradient descent as before
    on TensorFlow
    And this time we are going to use TensorFlow's built-in Optimizers
    to calculate the gradient for us

    Author: SayeedM
    Date: 07-07-2014
'''

import numpy as np
import tensorflow as tf

# again lets say we train a naive system to calculate y = 4x - 1
# this time using batch gradient descent
# we will generate a bunch of randoms as training data

X = np.random.rand(1000, 1)
y = -1 + 4 * X + np.random.randn(1000, 1) # giving some random noise

# adding a bias (1)
X_b = np.c_[np.ones((1000, 1)), X]

# hyper parameters
eta = 0.1
n_iterations = 1000
m = 1000

X = tf.constant(X_b, dtype = tf.float32, name = "X")
y = tf.constant(y, dtype = tf.float32, name = "y")

# initial theta (random)
theta = tf.Variable( tf.random_uniform([2, 1], -1, 1, seed = 42), name = "theta" )

# define prediction function y = X.thetaT
y_pred = tf.matmul(X, theta, name = "prediction")

# define error function (error = y_predict - y_ideal)
error = y_pred - y

# mse function
mse = tf.reduce_mean(tf.square(error), name = "mse")


# use either of the two but MomentumOptimizer seems better
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = eta)
optimizer = tf.train.MomentumOptimizer(learning_rate = eta, momentum = 0.9)

training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for i in range(n_iterations):
        if (i % 100 == 0):
            print("iteration # ", i, "  :  MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

    save_path = saver.save(sess, "lin-reg-optimize.ckpt")

    for i in range(10):
        X_new = np.array([i], dtype = np.float32)
        X_new_b = np.c_[np.ones((1, 1), dtype = np.float32), X_new]
        y_predict = X_new_b.dot(best_theta)

        print("prediction for x = ", i, " is : ", y_predict)
