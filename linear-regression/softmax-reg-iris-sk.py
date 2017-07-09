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
iris = pd.read_csv('../data/iris.data', header = None)

# converting the name to an encoded int
# it will assign 2 to iris verginica
iris[4] = iris[4].apply(irisNameToInt)

# converting to numpy array
iris = iris.as_matrix()

# grab all the petal lengths and widths
X = iris[:, 3:5].reshape(-1, 2)


y = iris[:, 4].reshape(-1, 1)

# running softmax regression
softmax_reg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", C = 10)
softmax_reg.fit(X, y)

# generating sample inputs
length_new = np.linspace(0, 7.5, 10).reshape(-1, 1)
width_new = np.linspace(0, 3, 10).reshape(-1, 1)
X_new = np.column_stack((length_new, width_new))

# getting the output class
y_proba = softmax_reg.predict(X_new)

# now print each inputs False and Truth probability
for i in range(len(X_new)):
    print("I : ", X_new[i][0], ", ", X_new[i][1], " O: ", y_proba[i])
