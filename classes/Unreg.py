import numpy as np
from numpy.random import rand

np.__version__


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Unreg:
    def cost_function(self, X, y, weights):
        z = np.dot(X, weights)
        positive_cost = y * np.log(sigmoid(z))
        negetive_cost = (1 - y) * np.log(1 - sigmoid(z))
        return -sum(positive_cost+negetive_cost) / len(X)

    def fit(self, X, y, epochs=1000, lr=0.05):
        self.weights = rand(X.shape[1])

        for _ in range(epochs):
            y_hat = sigmoid(np.dot(X, self.weights))
            self.weights -= lr * np.dot(X.T,  y_hat - y) / len(X)

    def predict(self, X):
        return [1 if i > 0.5 else 0 for i in sigmoid(np.dot(X, self.weights))]

    def score(self, X, y):
        y_hat = self.predict(X)
        y = y.reset_index(drop=True)
        p = 0
        for i in range(len(y)):
            if y[i] == y_hat[i]:
                p += 1
        return 100*p/len(y)
