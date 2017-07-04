'''
    linear regression batch gradient descent

    Author: SayeedM
    Date: 04-07-2014
'''

import numpy as np
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

# initial theta (random)
theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

for i in range(10):
    X_new = np.array([i])
    X_new_b = np.c_[np.ones((1, 1)), X_new]
    y_predict = X_new_b.dot(theta)
    print("prediction for x = ", i, " is : ", y_predict)
