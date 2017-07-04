'''
    Linear Regression Closed Form using Scikit Learn
    using the equation theta = (XT.X)^-1.XT.y

    Author: SayeedM
    Date: 05-07-2014
'''

import numpy as np
from sklearn.linear_model import LinearRegression

# again lets say we train a naive system to calculate y = 4x - 1
# we will generate a bunch of randoms as training data

X = np.random.rand(1000, 1)
y = -1 + 4 * X + np.random.rand(1000, 1) # giving some random noise

# adding a bias (1)
X_b = np.c_[np.ones((1000, 1)), X]

lr = LinearRegression()
lr.fit(X, y)

print("intercept : ", lr.intercept_)
print("co efficient : ", lr.coef_)

# now lets predict from 0 to 9

X_new = np.arange(10).reshape(10, 1)
predicted = lr.predict(X_new)

print("Predicted values : ")
print(predicted)
