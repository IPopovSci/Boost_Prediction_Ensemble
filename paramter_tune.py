from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

import xgboost as xgb
import pandas as pd
import numpy as np
from data_process import pipeline_extra
from loss_functions import eval_metric_signs,assymetric_mse_train,assymetric_mse_sklearn
from pathlib import Path
from sklearn.metrics import make_scorer

# compare different numbers of features selected using mutual information
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

'''Module to search for optimal hyper-parameters, uses GridSearchCV'''

Path = Path.cwd() / 'Data'

def xb_hp_tune():
    df_x_train,df_x_test,df_y_train,df_y_test = pipeline_extra(split_percent=0.9)

    param_grid = {"max_depth":    [10,12,15,20,25],
                  "n_estimators": [700,800,900,1000,1100],
                  "learning_rate": [0.9,1,1.1,1.2,1.3],"reg_alpha": [0.05,0.03,0.01,0.075,0.1],"reg_lambda":[0.025,0.05,0.005,0.01]}
                                                #The best hyperparameters are  {'learning_rate': 1, 'max_depth': 10, 'n_estimators': 800, 'reg_alpha': 0.05, 'reg_lambda': 0.025}

    regressor=xgb.XGBRegressor(obj=assymetric_mse_train)

    search = GridSearchCV(regressor, param_grid, cv=5,scoring=make_scorer(assymetric_mse_sklearn)).fit(df_x_train, df_y_train)

    print("The best hyperparameters are ",search.best_params_) #The best hyperparameters are  {'learning_rate': 0.6, 'max_depth': 2, 'n_estimators': 800}

def xb_feature_count_tune():
    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(1,'btcusd','1h')

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the pipeline to evaluate
    model = xgb.XGBRegressor(obj=assymetric_mse_train)
    fs = SelectKBest(score_func=mutual_info_regression)
    pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
    # define the grid
    grid = dict()
    grid['sel__k'] = [i for i in range(85, 106)]
    # define the grid search
    search = GridSearchCV(pipeline, grid, scoring=make_scorer(assymetric_mse_sklearn,greater_is_better=False), n_jobs=-1, cv=cv)
    # perform the search
    results = search.fit(df_x_train, df_y_train)
    # summarize best
    print('Best AMSE: %.8f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.8f with: %r" % (mean, param))

xb_feature_count_tune()