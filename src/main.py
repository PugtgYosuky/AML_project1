import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import sys
import datetime
import json
import pprint

from utils import *

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest

#models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# plot parametars
plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3

# TODO: move the function to utils.py
def instanciate_model(model_name):
    if model_name == 'LogisticRegression':
        model = LogisticRegression()

    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier()
    return model

# main funtion of python
if __name__ == '__main__':
    # get experience from logs folder
    experiences = os.listdir(os.path.join('logs'))
    # if logs folder is empty
    if len(experiences) == 0:
        exp = 'exp00'
    else:
        # in case the logs files doesn´t contain the name "exp"
        try:
            num = int(sorted(experiences)[-1][3:]) + 1
            # in case num smaller than 10 to register the files in a scale of 00 to 09
            if num < 10:
                exp = f'exp0{num}'
            # otherwise
            else:
                exp = f'exp{num}'
        except:
            exp = f'exp{datetime.datetime.now()}'

    # crate the new experience folder
    LOGS_PATH = os.path.join('logs', exp)
    os.makedirs(LOGS_PATH)

    # read config 
    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)
        
    # save config in logs folder
    with open(os.path.join(LOGS_PATH, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)

    # read dataset
    data = pd.read_csv(config['dataset'])    

    # remove duplicates from dataset
    data.drop_duplicates(inplace=True)

    # preprocess data
    X = data.copy()

    # remove 'attack_type' column from X and saves it to 
    y = X.pop('attack_type')

    # dict with classes map
    classes_map = {
        'normal' : 0,
        'Dos' : 1,
        'R2L' : 2,
        'U2R' : 3,
        'Probe' : 4
    }

    # switch categories in y to the ones in classes_map
    y = y.map(classes_map).to_numpy()

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
    
    # pipeline of Preprocessing
    pipeline = Pipeline(steps=[
        ('categories', ColumnTransformer(
            transformers=[
                ('cat', CustomEncoder(), categorical_columns),
                ('num', DropUniqueColumns(), numerical_columns)
            ], 
            remainder='passthrough'
        )),
        ('feature_selection', SelectKBest(k=config.get('number_best_features', X.shape[1]))),
        ('variance_selct', VarianceThreshold(threshold=config.get('variance_threshold', 0))),
        ('scaler', norm_model)
    ])

    X_transformed = pipeline.fit_transform(X, y)
    # TODO: save dataset transformed
    
    # print("Shape before: ",X.shape,"\n")
    # print("Shape after: ",X_transformed.shape,"\n")
    # print("Info of the dataset after the preprocessing: ", pd.DataFrame(X_transformed).info(),"\n")

    # split in train - test
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y)

    # ["RandomForestClassifier","GradientBoostingClassifier","AdaBostClassifier","DecisionTree","LogisticRegression","ElasticNet","SVC","XGBoost","GaussianDB","LGBM","MLPClassifier"]
    # reads the models in config file
    models_names = config.get('models_names', ['LogisticRegression'])
    
    results = {}
    # sees which model to use
    for model_name in models_names:
        # get the model 
        model = instanciate_model(model_name)

        # use the cross validation to get the average metrics of the model
        cross_results = cross_validate(model, x_train, y_train, scoring=['accuracy', 'balanced_accuracy', 'f1_weighted', 'matthews_corrcoef', 'roc_auc', 'roc_auc_ovr_weighted'])
        results[model.__class__.__name__] = {
            'accuracy' : np.mean(cross_results['test_accuracy']),
            'balanced_accuracy' : np.mean(cross_results['test_balanced_accuracy']),
            'f1_weighted' : np.mean(cross_results['test_f1_weighted']),
            'matthews_corrcoef' : np.mean(cross_results['test_matthews_corrcoef']),
            'roc_auc_ovr_weighted' : np.mean(cross_results['test_roc_auc_ovr_weighted']),
        }

    pprint.pprint(results)