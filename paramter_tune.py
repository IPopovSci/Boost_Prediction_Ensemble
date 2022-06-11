from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import sklearn
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
from sklearn.feature_selection import mutual_info_regression,f_classif,mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance

'''Module to search for optimal hyper-parameters, uses GridSearchCV'''

Path = Path.cwd() / 'Data'

def select_features(X_train, y_train, X_test,k):
	# configure to select a subset of features
	fs = SelectKBest(score_func=f_classif, k=k)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

def xb_hp_tune_regression():
    df_x_train,df_x_test,df_y_train,df_y_test = pipeline_extra(split_percent=0.9)

    param_grid = {"max_depth":    [10,12,15,20,25],
                  "n_estimators": [700,800,900,1000,1100],
                  "learning_rate": [0.9,1,1.1,1.2,1.3],"reg_alpha": [0.05,0.03,0.01,0.075,0.1],"reg_lambda":[0.025,0.05,0.005,0.01]}
                                                #The best hyperparameters are  {'learning_rate': 1, 'max_depth': 10, 'n_estimators': 800, 'reg_alpha': 0.05, 'reg_lambda': 0.025}

    regressor=xgb.XGBRegressor(obj=assymetric_mse_train)

    search = GridSearchCV(regressor, param_grid, cv=5,scoring=make_scorer(assymetric_mse_sklearn,greater_is_better=False)).fit(df_x_train, df_y_train)

    print("The best hyperparameters are ",search.best_params_) #The best hyperparameters are  {'learning_rate': 0.6, 'max_depth': 2, 'n_estimators': 800}

def xb_hp_tune_classification():
    df_x_train,df_x_test,df_y_train,df_y_test = pipeline_extra(0.8,'btcusd','1h')

    X_train_fs, X_test_fs, fs = select_features(df_x_train,df_y_train,df_x_test,k=30)

    param_grid = {"max_depth":    [15],
                  "n_estimators": [50],
                  "learning_rate": [1],'gamma':[0],"reg_alpha": [0.025],"reg_lambda":[1]}

    cv = RepeatedKFold(n_splits=10, n_repeats=3,random_state=1337)
    scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score),'f1 score':sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')}


    regressor=xgb.XGBClassifier(num_class=3,objective='multi:softmax',eval_metric='mlogloss')

    search = GridSearchCV(regressor, param_grid, cv=cv,scoring=scorer).fit(X_train_fs, df_y_train)

    print("The best hyperparameters are ",search.best_params_) #The best hyperparameters are  {'learning_rate': 0.6, 'max_depth': 2, 'n_estimators': 800}

def xb_feature_count_tune():
    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.8,'btcusd','1h')

    model = xgb.XGBClassifier(num_class=3, objective='multi:softmax',
                                  eval_metric='mlogloss')
    fs = SelectKBest(score_func=f_classif)
    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
    # define the grid
    grid = dict()
    grid['sel__k'] = [119,100,90,80,70,60,50,40,30] #30
    # define the grid search
    scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')

    search = GridSearchCV(pipeline, grid, scoring=scorer, n_jobs=-1, cv=cv)
    # perform the search
    results = search.fit(df_x_train, df_y_train)
    # summarize best
    print('Best f1: %.8f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.8f with: %r" % (mean, param))

def xgboost_test_classification():
    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.5, 'btcusd', '1h')

    #scorer = sklearn.metrics.f1_score()
    print(df_y_train)
    model = XGBClassifier(n_estimators=1000,max_depth=20,num_class=3)
    model.fit(df_x_train, df_y_train)
    # make predictions for test data
    y_pred = model.predict(df_x_test)
    plot_importance(model)
    pyplot.show()

xb_hp_tune_classification()
#xb_feature_count_tune()