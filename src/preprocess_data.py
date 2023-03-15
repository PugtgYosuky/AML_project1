"""
        ACA Project

        ATHORS:
            Joana Simões
            Pedro Carrasco
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# balance datasets
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest

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

#! Function to balancing de dataset. Receives X ,y , bmodel -> name of the model to use(SMOTE)
def dataset_balance(X ,y , bmodel):
    if bmodel == "SMOTE":
        smote = SMOTE(random_state=0)
        X_resampled, y_resampled = smote.fit_resample(X, y) 

    elif bmodel == "SMOTETomek":
        smotetomek = SMOTETomek(random_state=0)
        X_resampled, y_resampled = smotetomek.fit_resample(X, y)

    elif bmodel == "SMOTEENN":
        smoteenn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)

    return X_resampled,y_resampled


def create_fit_pipeline(config, X, y):
    # set parameters
    nm = config.get('norm_model', 'Standard') # try to get 'norm_model' from config files, uses 'Standart' if not founded

    # select which scaler to use
    if nm == 'MinMax':
        norm_model = MinMaxScaler()
    elif nm == 'Robust':
        norm_model = RobustScaler()
    else:
        norm_model = StandardScaler()

    # columns types
    categorical_columns = X.select_dtypes(include='object').columns.tolist() # list the categorical columns
    numerical_columns = [col for col in X if col not in categorical_columns] # list the numerical columns

    # pipeline of Preprocessing(Normalization, Feature Selection, Variance Selection, Scaler)
    pipeline = Pipeline(steps=[
        ('categories', ColumnTransformer(
            transformers=[
                ('cat', CustomEncoder(), categorical_columns),
                ('num', DropUniqueColumns(), numerical_columns)
            ], 
            remainder='passthrough'
        )),
        ('feature_selection', SelectKBest(k=config.get('number_best_features', 'all'))),
        ('variance_selct', VarianceThreshold(threshold=config.get('variance_threshold', 0))),
        ('scaler', norm_model)
    ])

    y_transformed = encode_target(y)

    X_transformed = pipeline.fit_transform(X, y_transformed)
    return pipeline, X_transformed, y_transformed
    
def encode_target(y):
    # dict with classes map
    classes_map = {
        'normal' : 0,
        'Dos'    : 1,
        'R2L'    : 2,
        'U2R'    : 3,
        'Probe'  : 4
    }

    # switch categories in y to the ones in classes_map
    y = y.map(classes_map).to_numpy()
    return y

    
