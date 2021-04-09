from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits

digits = (load_digits()).data
data = (load_breast_cancer()).data
X, y = load_boston(return_X_y=True)


print(digits[0])
print(data[0])
print(X[0])
print(y[0])
