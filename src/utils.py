"""
        ACA Project

        ATHORS:
            Joana Simões
            Pedro Carrasco
"""
import pandas as pd
import numpy as np

#! 
from sklearn.datasets import make_classification # ????
#from skrebate import ReliefF # ????
from sklearn.model_selection import cross_validate
import pprint



#models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# TODO: move the function to utils.py
def instanciate_model(model_name, settings={"random_state":42}):
    # TODO: add option to use model's parameters
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**settings)

    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**settings)

    elif model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier(**settings)

    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**settings)

    elif model_name == 'GaussianNB':
        model = GaussianNB(**settings)
    
    elif model_name == "SVC":
        model = SVC(**settings)

    elif model_name == "AdaBoostClassifier":
        model = AdaBoostClassifier(**settings)

    elif model_name == 'StandardScaler': # ??????
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    
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


## TODO: function to calculate metrics and save predictions