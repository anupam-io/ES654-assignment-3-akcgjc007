from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from classes.Unreg import Unreg
import pandas as pd


"""Generating training & test data"""
X, y =load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

obj = Unreg()
obj.fit(X_train, y_train)
print(obj.score(X_test, y_test))

# Cross-validation