"""
    linear regression on the housing data
    author: SayeedM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer


# useful column indexes for generating new column
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# transformer class for adding attributes to dataset which we think is legit
# used by scikit-learn pipeline
class CombinedAttributesAdder (BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_rooms = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_rooms]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# used for stripping text attributes from Padas DataFrame
class DataFrameSelector (BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return X[self.column_names].values


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

    # removing the synthetic attribute that was used for splitting
    train_set.drop(["income_cat"], axis = 1, inplace = True)
    test_set.drop(["income_cat"], axis = 1, inplace = True)

    return train_set, test_set




data = load_data()
train_set, test_set = split_data_set(data)



# seperating labels (prediction field) from train set

train_labels = train_set["median_house_value"].copy()
train_set.drop("median_house_value", axis = 1, inplace = True)

# we will be cleaning data, now that we need to fill in missing values we need an Imputer
# Imputer works only on numeric data, hence seperating again
train_numeric = train_set.drop("ocean_proximity", axis = 1)
train_text = train_set["ocean_proximity"]

numeric_attributes = list(train_numeric)
text_attributes = ["ocean_proximity"]

numeric_pipeline = Pipeline([
    ('selector', DataFrameSelector(numeric_attributes)),
    ('imputer', Imputer(strategy = 'median')),
    ('attributes_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

text_pipeline = Pipeline([
    ('selector', DataFrameSelector(text_attributes)),
    ('label_binarizer', LabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list = [
    ("numeric_pipeline", numeric_pipeline),
    ("text_pipeline", text_pipeline)
])

train_prepared = full_pipeline.fit_transform(train_set)




"""
train_set["rooms_per_household"] = train_set["total_rooms"] / train_set["households"]
train_set["bedrooms_per_rooms"] = train_set["total_bedrooms"] / train_set["total_rooms"]
train_set["population_per_household"] = train_set["population"] / train_set["households"]

corr_matrix = train_set.corr()
print(corr_matrix["median_house_value"].sort_values(ascending = False))
"""
