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
from preprocess_data import *

# to ignore terminal warnings
import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


# plot parameters
plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3


def grid_search(model, params, x_train, y_train, model_name):
    scoring = {
        'f1_weighted'           : 'f1_weighted',
        'accuracy'              : 'accuracy',
        'balanced_accuracy'     : 'balanced_accuracy',
        'matthews_corrcoef'     : 'matthews_corrcoef',
        'roc_auc_ovr_weighted'  : 'roc_auc_ovr_weighted'
    }
    start = time.time()
    grid_search = GridSearchCV(
        estimator   = model, 
        param_grid  = params,
        scoring     = scoring,
        refit       = 'matthews_corrcoef',
        cv          = StratifiedKFold(n_splits = 5, random_state = 42, shuffle=True),
        verbose     = 2,
        
    )

    # fitting the model    
    grid_search.fit(x_train, y_train)
    end = time.time()
    print(f'Time to test grid search {model_name}: {(end - start) / 60} minutes')

    # store the results of grid search
    aux = grid_search.cv_results_

    # create a pandas dataframe with the parameters and its means of the model tested
    df = pd.DataFrame(aux)
    df.to_csv(os.path.join(GRID_PATH, f'{model_name}_grid_search_{time.time()}.json'), index=False)
    df['model_name'] = model_name
    cols = [col for col in df.columns if col.startswith('std') or col.startswith('mean')]
    cols = ['model_name', 'params'] + cols
    df = pd.DataFrame(df[cols])
    print(df)

    return grid_search.best_estimaor, df
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
    PREDICTIONS_PATH = os.path.join(LOGS_PATH, 'predictions')
    os.makedirs(PREDICTIONS_PATH)

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

    
    pipeline, X_transformed, y_transformed = create_fit_pipeline(config, X, y)

    # save the transformed dataset
    aux = pd.DataFrame(X_transformed)
    aux['target_y'] = y_transformed
    aux.to_csv(os.path.join(LOGS_PATH, 'transformed_dataset.csv'), index=False)

    # split in train - test
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y_transformed, random_state=42)

    # to balance the dataset
    balance_method = config.get('balance_dataset', None)
    if balance_method:
        x_train, y_train = dataset_balance(x_train, y_train, balance_method)

    # reads the models in config file
    models_names = config.get('models_names', {'LogisticRegression' : {}})
    
    results = pd.DataFrame()
    best_models_results = pd.DataFrame()

    use_grid_search = config.get('grid_search', False)
    # for train, test in KFOLD
    # sees which model to use and the model´s parameters
    start = time.time()
    for model_name, params in models_names.items():
        print('MODEL:', model_name)
        if use_grid_search:
            # get the model 
            model = instanciate_model(model_name)
            model, df = grid_search(model, params, x_train, y_train, model_name)
            results = pd.concat([results, df], ignore_index=True)
        # save the dataset with the parameters and its means
        else:
            model = instanciate_model(model_name, params)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions = pd.DataFrame()
        predictions['y_test'] = y_test
        predictions['y_pred'] = y_pred
        predictions.to_csv(os.path.join(PREDICTIONS_PATH, f'{model_name}_predictions.csv'), index=False)

        # calculate metrics
        model_metrics = calculate_metrics(y_test, y_pred)
        pprint.pprint(model_metrics)
        metrics = pd.DataFrame(model_metrics, index=[model_name])
        best_models_results = pd.concat([best_models_results, metrics])
        best_models_results.to_csv(os.path.join(LOGS_PATH, 'model_metrics.csv'))

    end = time.time()
    print('Time to test all models: ', (end-start)/60, 'minutes')

    results.to_csv(os.path.join(LOGS_PATH, 'models_results.csv'), index=False)

