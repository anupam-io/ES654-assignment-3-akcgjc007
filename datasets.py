from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

digits_data = load_digits(return_X_y=True)
cancer_data = load_breast_cancer(return_X_y=True)
home_data = load_boston(return_X_y=True)
