"""
    linear regression on the housing data
    author: SayeedM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# loading data from csv
def load_data():
    csv_path = '../data/housing.csv'
    return pd.read_csv(csv_path)

# spliting the data in training set and test set
def split_data_set(data):
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace = True)
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

    for train_index, test_index in split.split(data, data["income_cat"]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    return train_set, test_set

data = load_data()
train_set, test_set = split_data_set(data)
