import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DropUniqueColumns(BaseEstimator):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
    def fit(self, X, y=None):
        # drop columns with just one value
        X = pd.DataFrame(X)
        self.cols_to_drop = X.columns[X.nunique() == 1]
        # correlation = X.corr(numeric_only=True)
        # correlated_features = correlation.abs() > self.threshold
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        preprocessed = X.drop(self.cols_to_drop, axis=1)
        return preprocessed


class CustomEncoder(BaseEstimator):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.encoder = LabelEncoder()
        self.encoders = {}
        for col in X:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X, y=None):
        X_encoded = X.copy()
        for col in X:
            X_encoded[col] = self.encoders[col].transform(X[col])

        return pd.DataFrame(X_encoded, columns=X.columns)

# teste