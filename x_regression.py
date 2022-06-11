import xgboost as xgb
import pandas as pd
import numpy as np
from data_process import pipeline_extra
from loss_functions import eval_metric_signs,assymetric_mse_train
from pathlib import Path
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
from feature_selection import select_features


'''Script to run XGBRegressor, fits on data and then predicts values and writes them to Boosted_predictions.csv'''

Path = Path.cwd() / 'Data'

def boost_ta():

    regressor=xgb.XGBRegressor(obj=assymetric_mse_train,random_state=1337) #,learning_rate=1.65,n_estimators=900,max_depth=20,reg_alpha=0.2,reg_lambda=0.05

    df_x_train,df_x_test,df_y_train,df_y_test = pipeline_extra(0.5,'btcusd','1h')

    df_x_train, df_x_test, fs = select_features(df_x_train, df_y_train, df_x_test,k=40)

    regressor.fit(df_x_train, df_y_train)

    y_pred = regressor.predict(df_x_test)

    y_true = df_y_test



    score = eval_metric_signs(y_true,y_pred)


    #print(y_true[-20:])
    #print(y_pred[-10:])
    print('Percent of signs guessed correctly is: ',score)

    pd.Series(y_pred,index=df_y_test.index).to_csv(f'{Path}/Boosted_Predictions_ta.csv')