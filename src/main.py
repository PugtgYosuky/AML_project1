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
from scipy import stats
from sklearn import metrics
import os
import sys
import datetime
import json
import pprint
import time
from sklearn import set_config

set_config(transform_output='pandas')

from utils import *
from preprocess_data import *
from sklearn.ensemble import VotingClassifier

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

def majority_voting(results, weights):
    """ Function from the professor slides"""
    aux = np.bincount(results, weights=weights)
    return np.argmax(aux)


def grid_search(model, params, x_train, y_train, model_name, kfold):
    scoring = {
        'f1_weighted'           : 'f1_weighted',
        'balanced_accuracy'     : 'balanced_accuracy',
        'matthews_corrcoef'     : 'matthews_corrcoef',
        'recall_weighted'  : 'recall_weighted',
        'precision_weighted'  : 'precision_weighted',

    }
    start = time.time()
    grid_search = GridSearchCV(
        estimator   = model, 
        param_grid  = params,
        scoring     = scoring,
        refit       = 'matthews_corrcoef',
        cv          = kfold,
        verbose     = 2,
        
    )

    # fitting the model    
    grid_search.fit(x_train, y_train)
    end = time.time()
    print(f'Time to test grid search {model_name}: {(end - start) / 60} minutes')

    # store the results of grid search
    cv_results = grid_search.cv_results_

    # create a pandas dataframe with the parameters and its means of the model tested
    df = pd.DataFrame(cv_results)
    df.to_csv(os.path.join(GRID_PATH, f'{model_name}_grid_search_{time.time()}.json'), index=False)
    
    metrics = ['f1_weighted','matthews_corrcoef','balanced_accuracy','precision_weighted','recall_weighted']
    metrics_names = [f'mean_test_{metric}' for metric in metrics]
    results = pd.DataFrame()
    results['model'] = [model.__class__.__name__] * len(cv_results['params'])
    results['config'] = cv_results['params']
    results['fold'] = 0
    for metric, cv_name in zip(metrics, metrics_names):
        results[metric] = cv_results[cv_name]

    results['time(minutes)'] = cv_results['mean_fit_time'] + cv_results['mean_score_time']
    results['count'] = 5
    sorted_results = results.sort_values(by=['matthews_corrcoef', 'f1_weighted'], ascending=False)
    sorted_results = sorted_results.iloc[:min(len(sorted_results), 5)]
    return sorted_results

def get_best_models_config(data, best_num=5):
    """ returns the best config for each model tested according to the cross-validation results"""
    compare = data.groupby(['model', 'config']).mean()
    compare['count'] = data.groupby(['model', 'config'])['fold'].count()
    compare = compare.reset_index()
    compare = compare.sort_values(['count', 'matthews_corrcoef'], ascending=False)
    best_models_config = []
    best_weights = []

    best_models = pd.DataFrame()
    for model in compare.model.unique():
        if model == 'MajorityVoting':
            continue
        model_data = compare.loc[compare.model == model]
        model_data = model_data.sort_values(by=['matthews_corrcoef', 'f1_weighted'], ascending=False)
        num_samples = len(model_data)
        num_samples = min(num_samples, best_num) # save the best three models of each type
        for i in range(num_samples):
            best_models_config.append([model_data.iloc[i]['model'], model_data.iloc[i]['config']])
            best_weights.append(model_data.iloc[i]['matthews_corrcoef'])
            best_models = pd.concat([best_models, pd.DataFrame(model_data.iloc[i]).T], ignore_index=True)
    
    best_models.drop(['fold'], axis=1)
    return best_models_config, best_weights,  best_models

def train_predict_model(save_path, y_test, y_pred, fold, model_name, params, total_time):
    predictions = pd.DataFrame()
    predictions['y_test'] = y_test
    predictions['y_pred'] = y_pred
    predictions.to_csv(os.path.join(save_path, f'{model_name}_fold{fold+1}_predictions.csv'), index=False)
    # calculate metrics
    model_metrics = calculate_metrics(y_test, y_pred)
    model_metrics['time(minutes)'] = (total_time) / 60 # add fitting time to the evaluation
    
    model_metrics['fold'] = fold + 1 # save fold
    model_metrics['config'] = json.dumps(params) # save config 
    model_metrics['model'] = model_name
    pprint.pprint(model_metrics)
    metrics = pd.DataFrame(model_metrics, index=[0])
    #reorganize dataframe
    metrics = metrics[['model', 'config', 'fold', 'f1_weighted','matthews_corrcoef', 'balanced_accuracy','precision_weighted','recall_weighted','time(minutes)']]
    return metrics

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
    TARGET_PATH = os.path.join(LOGS_PATH, 'kaggle')
    os.makedirs(TARGET_PATH)

    # gets the name of the config file and read´s it
    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)
        
    # save config in logs folder
    with open(os.path.join(LOGS_PATH, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=1)

    # read dataset
    data = pd.read_csv(config['dataset'])    

    # remove duplicates from dataset
    data.drop_duplicates(inplace=True)

    # preprocess data
    X = data.copy()

    # remove 'attack_type' column from X and saves it to 
    y = X.pop('attack_type')

    print(X.shape)    
    pipeline, X_transformed, y_transformed = create_fit_pipeline(config, X, y)
    print(X_transformed.shape)
    print(X_transformed.columns)
    
    # save the transformed dataset
    aux = X_transformed.copy()
    aux['target_y'] = y_transformed
    aux.to_csv(os.path.join(LOGS_PATH, 'transformed_dataset.csv'), index=False)
    X_transformed = X_transformed.to_numpy()

    # split in train - test
    # x_train, x_test, y_train, y_test = train_test_split(X_transformed, y_transformed, random_state=42)
    models_to_test = config.get('models_names', [['LogisticRegression',  {}]])
    use_grid_search = config.get('grid_search', False)
    balance_method = config.get('balance_dataset', None)

    weights_models = [1/len(models_to_test)] * len(models_to_test)

    train_only = config.get('train_only', False)
    if not train_only:
        kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        if use_grid_search:
            print("GRID SEARCH")
            ranking = pd.DataFrame()
            start_total = time.time()
            if balance_method:
                x, y = dataset_balance(X_transformed, y_transformed, balance_method)
            else:
                x, y = X_transformed, y_transformed
            for model_name, params in models_to_test:
                # get the model 
                model = instanciate_model(model_name)
                df = grid_search(model, params, x, y, model_name, kfold)
                ranking = pd.concat([ranking, df], ignore_index=True)
            ranking.to_csv(os.path.join(LOGS_PATH, 'grid_results.csv'), index=False)
            weights_models = list(ranking['matthews_corrcoef'])
            models_to_test = [(model, config) for model, config in zip(ranking['model'], ranking['config'])]
            print(models_to_test)
        else:

            start_total = time.time()
            best_models_results = pd.DataFrame()
            for fold, (train_indexes, test_indexes) in enumerate(kfold.split(X_transformed, y_transformed)):
                print(f'############## {fold+1}-FOLD ##############')
                x_train = X_transformed[train_indexes]
                y_train = y_transformed[train_indexes]
                x_test = X_transformed[test_indexes]
                y_test = y_transformed[test_indexes]

                # to balance the dataset
                if balance_method:
                    x_train, y_train = dataset_balance(x_train, y_train, balance_method)

                # reads the models from config file
                

                # for train, test in KFOLD
                # sees which model to use and the model´s parameters
                start = time.time()
                ensemble_models = []
                models_weights = []
                ensemble_predictions = pd.DataFrame()
                for model_name, params in models_to_test:
                    model = instanciate_model(model_name, params)
                    # add the instanciated model to the list for ensemble
                    ensemble_models.append((model_name, instanciate_model(model_name, params)))
                    # metrics = train_predict_model(model, PREDICTIONS_PATH, x_train, y_train, x_test, y_test, fold, model_name, params)
                    print('MODEL:', model_name)
                    start_model = time.time()
                    model.fit(x_train, y_train)
                    end_model = time.time()
                    y_pred = model.predict(x_test)
                    ensemble_predictions[f'{model_name}_{time.time()}'] = y_pred
                    metrics = train_predict_model(PREDICTIONS_PATH, y_test, y_pred, fold, model_name, params, end_model - start_model)
                    models_weights.append(metrics['matthews_corrcoef'][0])
                    best_models_results = pd.concat([best_models_results, metrics], ignore_index=True)

                # soft normalization of the weights
                models_weights = np.array(models_weights)
                models_weights /= np.sum(models_weights)
                print('MODEL:', 'MajorityVoting')
                voting_preds = ensemble_predictions.apply(majority_voting, axis=1, weights=models_weights)
                metrics = train_predict_model(PREDICTIONS_PATH, y_test, voting_preds, fold, 'MajorityVoting', params, 0)
                best_models_results = pd.concat([best_models_results, metrics], ignore_index=True)

                end = time.time()
                print(f'\n[{fold+1}-fold] Time to test all models: ', (end-start)/60, 'minutes')


                best_models_results.to_csv(os.path.join(LOGS_PATH, 'model_metrics.csv'), index=False)

                models_to_test, weights_models,  ranking = get_best_models_config(best_models_results, config.get('num_best_models', 3))
                

        weights_models = np.array(weights_models)
        print(weights_models)
        weights_models /= np.sum(weights_models)
        print(weights_models)
        
        end_total = time.time()
        print('TOTAL TIME - 5-FOLDS:', (end_total - start_total) / 60, 'minutes')
        
        print("\n\n BEST CONFIGS \n")
        print(ranking)
        ranking.to_csv(os.path.join(LOGS_PATH, 'models_mean_results.csv'), index=False)
    
    # **************** PREDICT KAGGLE ****************
    print('**************** PREDICT KAGGLE ****************')
    # read target dataset
    target_data = pd.read_csv(config['target_dataset'])
    sample_ID = target_data.pop('SampleID')
    output_predictions = pd.DataFrame(sample_ID)
    # transform test dataset
    target_data = pipeline.transform(target_data)

    # to balance the dataset
    if balance_method:
        X_transformed, y_transformed = dataset_balance(X_transformed, y_transformed, balance_method)

    # select the best model
    ensemble_predictions = pd.DataFrame()
    ensemble_models = []
    count = 1
    for best_model_name, best_params in models_to_test:
        print('MODEL', best_model_name)
        if type(best_params) is not dict:
            best_params = json.loads(best_params)

        best_model = instanciate_model(best_model_name, best_params)
        ensemble_models.append((best_model_name, instanciate_model(best_model_name, best_params)))
        # fit best model
        best_model.fit(X_transformed, y_transformed)
        
        # predict target
        preds = best_model.predict(target_data)

        model_predictions = output_predictions.copy()
        model_predictions['Class'] = preds
        ensemble_predictions[best_model_name + str(count)] = preds
        # save model predictions
        model_predictions.to_csv(os.path.join(TARGET_PATH, f'{best_model_name + str(count)}.csv'), index=False)
        count += 1
    # ensemble the results
    voting_preds = ensemble_predictions.apply(majority_voting, axis=1, weights=weights_models)
    output_predictions['Class'] = voting_preds
    output_predictions.to_csv(os.path.join(TARGET_PATH, f'combined_preds.csv'), index=False)

