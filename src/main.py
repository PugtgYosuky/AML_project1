import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import sys
import datetime
import json

from utils import *

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest

# plot parametars
plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3

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
    y = y.map(classes_map)

    # # split in train - test
    # x_train, x_test, y_train, y_test = train_test_split(X, y)

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
    
    # pipeline
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

    res = pipeline.fit_transform(X, y)
    
    # test models
    print(res.shape)
    print(X.shape)