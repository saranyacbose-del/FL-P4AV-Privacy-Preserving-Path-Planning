import numpy as np

class LocalModel:
    def __init__(self):
        self.w = np.zeros(4)
        self.b = 0

    def predict(self, x):
        return np.dot(self.w, x) + self.b

    def train(self, X, y, lr=0.01):
        for xi, yi in zip(X, y):
            pred = self.predict(xi)
            error = pred - yi
            self.w -= lr * error * xi
            self.b -= lr * error
