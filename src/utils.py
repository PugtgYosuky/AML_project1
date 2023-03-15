import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#! 
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from skrebate import ReliefF
from sklearn.model_selection import cross_validate
import pprint

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

#! Function to balancing de dataset. Recieves X ,y , bmodel -> name of the model to use(SMOTE)
def dataset_balance(X ,y , bmodel):

    if bmodel == "SMOTE":
        #! using SMOTE to generate synthetic samples from the minority class by interpolating between existing samples
        smote = SMOTE(random_state=0)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    elif bmodel == "RandomUnderSampler":
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)   

    elif bmodel == "RandomOverSampler":
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled,y_resampled

#! After GridSearchCV this function will decide the best model to choose with cross_validate
def final_selection(X, y, models_list):
    scoring = ['f1_weighted' ,'accuracy' ,'balanced_accuracy' ,'matthews_corrcoef' ,'roc_auc_ovr_weighted']

    for model in models_list:
        # model = instanciate_model(model_name) 
        cross_val = cross_validate(model, X, y, cv=5, scoring=scoring)
 
        aux = cross_val.cv_results_
        pprint.pprint(aux)

        mean_accuracy = aux['test_accuracy'].mean()
        mean_precision = aux['test_precision_macro'].mean()
        mean_recall = aux['test_recall_macro'].mean()
        mean_f1 = aux['test_f1_macro'].mean()
        print(f'Model {model}: accuracy={mean_accuracy:.3f}, precision={mean_precision:.3f}, recall={mean_recall:.3f}, f1={mean_f1:.3f}')

#! Function to train the desired model
def train_model(X_train, y_train, X_test, y_test , modelToTrain):
        model = modelToTrain()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        pprint.pprint(score)

#! Function to select the best model based on 'mean_test_f1_weighted' and 'mean_test_matthews_corrcoef'
def train_model(path):
        data = pd.read_csv(path)
        best_model = data.sort_values(by=['mean_test_f1_weighted', 'mean_test_matthews_corrcoef'], ascending=False)
        print(best_model.iloc[0])