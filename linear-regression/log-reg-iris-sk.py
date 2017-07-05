'''
    Running Logistic Regression on Iris Dataset and determine
    if a given row is Iris Virginica
    Simple Binary Classification

    Author: SayeedM
    Date: 06-07-2017
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def irisNameToInt(name):
    if name == 'Iris-setosa':
        return 0
    elif name == 'Iris-versicolor':
        return 1
    else:
        return 2


# loading data
iris = pd.read_csv('data/iris.data', header = None)

# converting the name to an encoded int
# it will assign 2 to iris verginica
iris[4] = iris[4].apply(irisNameToInt)

# converting to numpy array
iris = iris.as_matrix()

# grab all the petal widths
X = iris[:, 3].reshape(-1, 1)
y = (iris[:, 4] == 2).reshape(-1, 1)

# running logistic regression
log_reg = LogisticRegression()
log_reg.fit(X, y)

# generating sample inputs
X_new = np.linspace(0, 3, 10).reshape(-1, 1)

# getting the probabilities
y_proba = log_reg.predict_proba(X_new)

# now print each inputs False and Truth probability
for i in range(len(X_new)):
    print("I : ", X_new[i], " F: ", y_proba[i][0], " , T: ", y_proba[i][1])
