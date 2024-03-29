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

class DropColumns(BaseEstimator):
    """ Class to drop columns from the dataset"""
    def __init__(self, columns=[], threshold=0.8):
        self.threshold = threshold
        self.columns = columns
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns += list(X.columns[X.nunique() == 1])
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        preprocessed = X.drop(self.columns, axis=1)
        return preprocessed


class CustomEncoder(BaseEstimator):
    """ class to encode the categoracal variables"""
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
            # to handle unseen values
            column = X[col].to_numpy()
            indexes = np.isin(column, self.encoders[col].classes_)
            transformed = np.ones_like(column).astype(int) * -1
            # transform values
            transformed[indexes] = self.encoders[col].transform(column[indexes])
            X_encoded[col] = transformed

        return pd.DataFrame(X_encoded, columns=X.columns)


def dataset_balance(X ,y , model):
    if model == "SMOTE":
        smote = SMOTE(random_state=0)
        X_resampled, y_resampled = smote.fit_resample(X, y) 

    elif model == "SMOTETomek":
        smotetomek = SMOTETomek(random_state=0)
        X_resampled, y_resampled = smotetomek.fit_resample(X, y)

    elif model == "SMOTEENN":
        smoteenn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)

    return X_resampled,y_resampled


def create_fit_pipeline(config, X, y):
    """ Creates the pipeline and fits the training data"""
    # set parameters
    norm_model_name = config.get('norm_model', 'Standard') # try to get 'norm_model' from config files, uses 'Standart' if not founded

    # select which scaler to use
    if norm_model_name == 'MinMax':
        print('MinMaxScaler')
        norm_model = MinMaxScaler()
    elif   norm_model_name == 'Robust':
        print('RobustScaler')
        norm_model = RobustScaler()
    else:
        print('StandardScaler')
        norm_model = StandardScaler()

    # columns types
    categorical_columns = X.select_dtypes(include='object').columns.tolist() # list the categorical columns
    numerical_columns = [col for col in X if col not in categorical_columns] # list the numerical columns
    drop = DropColumns(config.get('columns_to_drop', []))
    # pipeline of Preprocessing(Normalization, Feature Selection, Variance Selection, Scaler)
    pipeline = Pipeline(steps=[
        ('categories', ColumnTransformer(
            transformers=[
                ('cat', CustomEncoder(), categorical_columns)
            ], 
            remainder='passthrough',
            verbose_feature_names_out=False
        )),
        ('remove_cols', drop),
        ('feature_selection', SelectKBest(k=config.get('number_best_features', 'all'))),
        ('variance_select', VarianceThreshold(threshold=config.get('variance_threshold', 0))),
        ('scaler', norm_model)
    ])

    y_transformed = encode_target(y)

    X_transformed = pipeline.fit_transform(X, y_transformed)
    return pipeline, X_transformed, y_transformed
    
def encode_target(y):
    """ Encode of the target variable"""
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

    
