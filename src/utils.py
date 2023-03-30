"""
        ACA Project

        ATHORS:
            Joana Simões
            Pedro Carrasco
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate
import pprint
from sklearn import metrics

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier


def instanciate_model(model_name, settings={}):
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**settings, random_state=42)

    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**settings, random_state=42)

    elif model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier(**settings)

    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**settings, random_state=42)

    elif model_name == 'GaussianNB':
        model = GaussianNB(**settings)
    
    elif model_name == 'SVC':
        model = SVC(**settings, random_state=42)

    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(**settings, random_state=42)

    elif model_name == 'XGBClassifier':
        model = XGBClassifier(**settings, random_state=42)

    elif model_name == 'LGBMClassifier':
        model = LGBMClassifier(**settings, random_state=42)
    
    elif model_name == 'MLPClassifier':
        model = MLPClassifier(**settings, random_state=42)
    
    return model



#! After GridSearchCV this function will decide the best model to choose with cross_validate
def final_selection(X, y, models_list): # ????????????????????????????????????????????????????????
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

def calculate_metrics (y_test, y_pred):
    return {
          'balanced_accuracy' : metrics.balanced_accuracy_score(y_test, y_pred),
        #   'roc_auc_score' : metrics.roc_auc_score(y_test, y_pred),
          'recall_weighted' : metrics.recall_score(y_test, y_pred, average='weighted'), 
          'f1_weighted' : metrics.f1_score(y_test, y_pred, average='weighted'), 
          'precision_weighted' : metrics.precision_score(y_test, y_pred, average='weighted'), 
          'matthews_corrcoef' : metrics.matthews_corrcoef(y_test, y_pred)
    }