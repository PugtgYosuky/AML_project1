"""
        ACA Project

        ATHORS:
            Joana Simões
            Pedro Carrasco
"""

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
import time

from utils import *

# to ignore terminal warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest

#!
from sklearn.pipeline import make_pipeline

#models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

# plot parametars
plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3

# TODO: move the function to utils.py
def instanciate_model(model_name):
    # TODO: add option to use model's parameters
    if model_name == 'LogisticRegression':
        model = LogisticRegression()

    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier()

    #!
    elif model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier()

    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier()

    elif model_name == 'GaussianNB':
        model = GaussianNB()
    
    elif model_name == 'StandardScaler':
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    
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
    GRID_PATH = os.path.join(LOGS_PATH, 'grid_search')
    os.makedirs(GRID_PATH)

    # gets the name of the config file and read´s it
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
        'Dos'    : 1,
        'R2L'    : 2,
        'U2R'    : 3,
        'Probe'  : 4
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
    
    # pipeline of Preprocessing(Normalization, Feature Selection, Variance Selection, Scaler)
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
    
    # save the transformed dataset
    aux = pd.DataFrame(X_transformed)
    aux['target_y'] = y
    aux.to_csv(os.path.join(LOGS_PATH, 'transformed_dataset.csv'), index=False)

    # split in train - test
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, random_state=42)

    # reads the models in config file
    models_names = config.get('models_names', {'LogisticRegression' : {}})
    
    results = pd.DataFrame()

    scoring = {
        'f1_weighted'           : 'f1_weighted',
        'accuracy'              : 'accuracy',
        'balanced_accuracy'     : 'balanced_accuracy',
        'matthews_corrcoef'     : 'matthews_corrcoef',
        'roc_auc_ovr_weighted'  : 'roc_auc_ovr_weighted'
    }

    # for train, test in KFOLD
    # sees which model to use and the model´s parameters
    for model_name, params in models_names.items():
        # get the model 
        model = instanciate_model(model_name)

        # grid_search to find the best model(TODO) with the best parameters
        start = time.time()

        grid_search = GridSearchCV(
            estimator   = model, 
            param_grid  = params,
            scoring     = scoring,
            refit       = 'matthews_corrcoef',
            verbose     = 2,
            cv          = StratifiedKFold(n_splits = 5, random_state = 42, shuffle=True),
            
        )

        # fitting the model    
        grid_search.fit(x_train, y_train)

        #! priting the best params
        # best_params = grid_search.best_params_
        # print("\n\n\nbest_params: ",best_params,"\n\n\n")

        end = time.time()

        print(f'Time to test {model_name}: {(end - start) / 60} minutes')

        # store the results of grid search
        aux = grid_search.cv_results_

        pprint.pprint(aux)
        
        # create a pandas dataframe with the parameters and its means of the model tested
        df = pd.DataFrame(aux)
        df.to_csv(os.path.join(GRID_PATH, f'{model_name}_grid_search_{time.time()}.json'), index=False)
        df['model_name'] = model_name
        cols = [col for col in df.columns if col.startswith('std') or col.startswith('mean')]
        cols = ['model_name', 'params'] + cols
        df = pd.DataFrame(df[cols])
        print(df)

        results = pd.concat([results, df], ignore_index=True)
        # save the dataset with the parameters and its means of the model tested
    
    results.to_csv(os.path.join(LOGS_PATH, model_name+'_model_results.csv'), index=False)

