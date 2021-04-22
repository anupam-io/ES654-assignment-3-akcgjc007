from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from classes.Unreg import Unreg
import pandas as pd
from time import time


X, y =load_breast_cancer(as_frame=True, return_X_y=True)
obj = Unreg()

for i in range(1, 11):
    t = time()
    obj.fit(X[:(i*len(X.index))//10], y[:(i*len(y.index))//10])
    print("fit() time:", time()-t)

    t = time()
    obj.predict(X[:(i*len(X.index))//10])
    print("predict() time:", time()-t)
