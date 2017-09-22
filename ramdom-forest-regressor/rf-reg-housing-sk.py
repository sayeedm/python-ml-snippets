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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



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


# utility method to display cross validation scores
def display_scores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("SD : ", scores.std())

# our flow begins here

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

# building sk-learn pipelines
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

# call the pipeline to preprocess data
train_prepared = full_pipeline.fit_transform(train_set)

# train the model
forest_reg = RandomForestRegressor()
forest_reg.fit(train_prepared, train_labels)

# now test some data
some_data = data.iloc[:5]
some_data_labels = train_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print(forest_reg.predict(some_data_prepared))
print(list(some_data_labels))

# lets calculate the error
train_prediction = forest_reg.predict(train_prepared)
mse = mean_squared_error(train_labels, train_prediction)
rmse = np.sqrt(mse)
print("Root Mean squared error : ", rmse)

scores = cross_val_score(forest_reg, train_prepared, train_labels,
                         scoring = "neg_mean_squared_error", cv = 10)

rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)

# lets see if we can improve - why not run a grid search
# to see if we find better parameters for RandomForestRegressor


param_grid = [
    {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]},
]

test_forest_reg = RandomForestRegressor();
# this param grid will train model 5 * (3 * 4 + 2 * 3) = 90 times and try to
# find a best result
grid_search = GridSearchCV(test_forest_reg, param_grid, cv = 5, scoring = "neg_mean_squared_error")
grid_search.fit(train_prepared, train_labels)

print("Best Params ", grid_search.best_params_)
print("Best Estimator ", grid_search.best_estimator_)

# also we may want to check feature importace
feature_importances = grid_search.best_estimator_.feature_importances_

final_model = grid_search.best_estimator_
X_test = test_set.drop("median_house_value", axis = 1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

test_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, test_predictions)

final_rmse = np.sqrt(final_mse)

print("RMSE Final : ", final_rmse)
