import xgboost as xgb
import pandas as pd
import numpy as np
from data_process import small_pipeline
from loss_functions import eval_metric_signs,assymetric_mse_train
from pathlib import Path


'''Script to run XGBRegressor, fits on data and then predicts values and writes them to Boosted_predictions.csv'''

Path = Path.cwd() / 'Data'

regressor=xgb.XGBRegressor(obj=assymetric_mse_train,learning_rate=0.6,n_estimators=800,max_depth=2,reg_alpha=0.1,reg_lambda=0.025)

df_x_train,df_x_test,df_y_train,df_y_test = small_pipeline(0.5)

regressor.fit(df_x_train, df_y_train)

y_pred = regressor.predict(df_x_test)

y_true = df_y_test

score = eval_metric_signs(y_true,y_pred)

print(y_pred[-20:])
print('Percent of signs guessed correctly is: ',score)

pd.Series(y_pred,index=df_y_test.index).to_csv(f'{Path}/Boosted_Predictions.csv')

