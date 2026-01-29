
from sklearn.linear_model import LinearRegression as SklearnLR
import numpy as np
import pandas as pd

class LinearRegression:
    """
    Wrapper for sklearn LinearRegression.
    """
    def __init__(self, **kwargs):
        self.model = SklearnLR(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
