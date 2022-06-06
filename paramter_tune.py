from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

import xgboost as xgb
import pandas as pd
import numpy as np
from data_process import small_pipeline
from loss_functions import eval_metric_signs,assymetric_mse_train,assymetric_mse_sklearn
from pathlib import Path
from sklearn.metrics import make_scorer

'''Module to search for optimal hyper-parameters, uses GridSearchCV'''

Path = Path.cwd() / 'Data'

df_x_train,df_x_test,df_y_train,df_y_test = small_pipeline(0.5)

param_grid = {"max_depth":    [2],
              "n_estimators": [800],
              "learning_rate": [0.6],"reg_alpha": [0.2,0.1,0.05,0.075,0.025],"reg_lambda":[0.2,0.1,0.05,0.075,0.025]}

regressor=xgb.XGBRegressor(obj=assymetric_mse_train)

search = GridSearchCV(regressor, param_grid, cv=5,scoring=make_scorer(assymetric_mse_sklearn)).fit(df_x_train, df_y_train)

print("The best hyperparameters are ",search.best_params_) #The best hyperparameters are  {'learning_rate': 0.6, 'max_depth': 2, 'n_estimators': 800}

