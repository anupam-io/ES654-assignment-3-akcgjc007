from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from classes.Unreg import Unreg
import pandas as pd
import numpy as np


"""Generating training & test data"""
X, y =load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

obj = Unreg()
obj.fit(X_train, y_train)
# print(obj.score(X_test, y_test))

# Cross-validation
sX1, sX2, sX3 = X[:len(X)//3], X[len(X)//3:2*len(X)//3], X[2*len(X)//3:]
sy1, sy2, sy3 = y[:len(X)//3], y[len(X)//3:2*len(X)//3], y[2*len(X)//3:]

obj1, obj2, obj3 = Unreg(), Unreg(), Unreg()

obj1.fit(pd.concat([sX1, sX2]), pd.concat([sy1, sy2]))
obj2.fit(pd.concat([sX2, sX3]), pd.concat([sy2, sy3]))
obj3.fit(pd.concat([sX1, sX3]), pd.concat([sy1, sy3]))

print("Average accuracy for 3-cross-validation: ", np.mean([
    obj1.score(sX3, sy3),
    obj2.score(sX1, sy1),
    obj3.score(sX2, sy2),
]))