'''
    linear regression closed form
    using the equation theta = (XT.X)^-1.XT.y

    Author: SayeedM
    Date: 03-07-2014
'''

import numpy as np
# lets say we train a naive system to calculate y = 4x - 1
# we will generate a bunch of randoms as training data

X = np.random.rand(1000, 1)
y = -1 + 4 * X + np.random.rand(1000, 1) # giving some random noise

# adding a bias (1)
X_b = np.c_[np.ones((1000, 1)), X]

best_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# lets predict for X = 5 should be 19
X_new = np.array([5])
X_new_b = np.c_[np.ones((1, 1)), X_new]
y_predict = X_new_b.dot(best_theta)
print("Prediction for ", 5, " is : ", y_predict)

# now lets predict for a bunch of numbers
for i in range(10):
    X_new = np.array([i])
    X_new_b = np.c_[np.ones((1, 1)), X_new]
    y_predict = X_new_b.dot(best_theta)
    print("Prediction for ", i, " is : ", y_predict)
