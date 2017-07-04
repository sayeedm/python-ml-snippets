'''
    Polynomial Regression using Scikit Learn
    We process it and then feed it to normal LinearRegression

    Author: SayeedM
    Date: 05-07-2014
'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# lets say we train a naive system to calculate y = x^2 - 9x + 20
# we will generate a bunch of randoms as training data

X = np.random.rand(1000, 1)
y = X ** 2 - 9 * X + 20 + np.random.randn(1000, 1) # giving some random noise

pf = PolynomialFeatures( degree = 2, include_bias = False)
X_poly = pf.fit_transform(X)

print("X[0] : ", X[0])
print("X_poly[0] : ", X_poly[0])

# we see that the poly contains both X and its degree 2
lr = LinearRegression()
lr.fit(X_poly, y)

print("intercept : ", lr.intercept_)
print("co efficient : ", lr.coef_)

# now lets predict from 0 to 9

X_new = np.array([[1, 10]])

predicted = lr.predict(X_new)

print("Predicted values : ")
print(predicted)
