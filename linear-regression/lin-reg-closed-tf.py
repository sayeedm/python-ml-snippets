'''
    linear regression closed form on TensorFlow
    using the equation theta = (XT.X)^-1.XT.y

    Author: SayeedM
    Date: 03-07-2014
'''

import tensorflow as tf
import numpy as np

# lets say we train a naive system to calculate y = 4x - 1
# we will generate a bunch of randoms as training data

X = np.random.rand(1000, 1)
y = -1 + 4 * X + np.random.rand(1000, 1) # giving some random noise

# adding a bias (1)
X_b = np.c_[np.ones((1000, 1)), X]

X = tf.constant(X_b, dtype = tf.float32, name = "X")
y = tf.constant(y, dtype = tf.float32, name = "y")
XT = tf.transpose(X)

theta_func = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    best_theta = theta_func.eval()
    X_new = np.array([5])
    X_new_b = np.c_[np.ones((1, 1)), X_new]
    prediction = X_new_b.dot(best_theta)
    print("Prediction for ", 5, " is : ", prediction)
