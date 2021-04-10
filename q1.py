from sklearn.datasets import load_breast_cancer
from classes.Unreg import Unreg
import pandas as pd

"""Generating training & test data"""
raw_data = (load_breast_cancer(as_frame=True))
data = raw_data.data
data = data.sample(frac=1)
labels = raw_data.target
p = 80*len(data)//100
train_data, test_data = data[:p], data[p:]
train_labels, test_labels = [], []
for i in train_data.index:
    train_labels.append(labels.iloc[i])
for i in test_data.index:
    test_labels.append(labels.iloc[i])
test_labels = pd.Series(test_labels)
train_labels = pd.Series(train_labels)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
print(train_data.head())
print(test_data.head())

obj = Unreg()
obj.fit(train_data, train_labels)
