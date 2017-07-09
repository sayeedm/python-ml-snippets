'''
    Running Logistic Regression on Iris Dataset and determine
    if a given row is Iris Virginica
    Simple Binary Classification

    Author: SayeedM
    Date: 06-07-2017
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron


def irisNameToInt(name):
    if name == 'Iris-setosa':
        return 0
    elif name == 'Iris-versicolor':
        return 1
    else:
        return 2


# loading data
iris = pd.read_csv('../data/iris.data', header = None)

# converting the name to an encoded int
# it will assign 2 to iris verginica
iris[4] = iris[4].apply(irisNameToInt)

# converting to numpy array
iris = iris.as_matrix()

# grab all the petal widths
X = iris[:, 2:4]
y = (iris[:, 4] == 0).reshape(-1, 1)
print(y)

# run Perceptron
per_clf = Perceptron(random_state = 42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)
