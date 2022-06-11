import xgboost as xgb
import pandas as pd
import numpy as np
from data_process import pipeline_extra
from loss_functions import eval_metric_signs, assymetric_mse_train
from pathlib import Path
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from feature_selection import select_features
import sklearn
import matplotlib.pylab as plt
from matplotlib import pyplot

'''Script to run XGBRegressor, fits on data and then predicts values and writes them to Boosted_predictions.csv'''

Path = Path.cwd() / 'Data'


def boost_ta_regression():
    regressor = xgb.XGBRegressor(obj=assymetric_mse_train,
                                 random_state=1337)  # ,learning_rate=1.65,n_estimators=900,max_depth=20,reg_alpha=0.2,reg_lambda=0.05

    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.5, 'btcusd', '1h')

    df_x_train, df_x_test, fs = select_features(df_x_train, df_y_train, df_x_test, k=40)

    regressor.fit(df_x_train, df_y_train)

    y_pred = regressor.predict(df_x_test)

    y_fit = regressor.predict(df_x_train)

    y_true = df_y_test

    score = eval_metric_signs(y_true, y_pred)

    # print(y_true[-20:])
    # print(y_pred[-10:])
    print('Percent of signs guessed correctly is: ', score)

    pd.Series(y_pred, index=df_y_test.index).to_csv(f'{Path}/Boosted_Predictions_ta.csv')

    return y_pred,y_fit


def boost_ta_classification():



    regressor = xgb.XGBClassifier(gamma=0.1, learning_rate=0.2, max_depth=25, n_estimators=50, reg_alpha=0.25,
                                  reg_lambda=0.5, num_class=2, objective='multi:softprob',
                                  eval_metric='mlogloss')  # ,learning_rate=1.65,n_estimators=900,max_depth=20,reg_alpha=0.2,reg_lambda=0.05

    # [[90 43]
    #  [13 26]]
    # 0.5034965034965035
    # 0.4848580512055205



    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.1, 'btcusd', '1h')

    x_train, df_x_test, fs = select_features(df_x_train, df_y_train, df_x_test,k=30)

    class_weights = sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced',y = df_y_train)
    #print(class_weights)

    regressor.fit(x_train, df_y_train,sample_weight=class_weights)

    y_pred = regressor.predict(df_x_test)

    y_pred_probs = regressor.predict_proba(df_x_test)

    print(y_pred_probs[-20:])

    cm = confusion_matrix(df_y_test, y_pred)

    print(cm)

    accuracy = accuracy_score(df_y_test, y_pred)

    print(accuracy)

    fig1 = plt.gcf()

    (pd.Series(regressor.feature_importances_, index=fs.get_feature_names_out())
     .plot(kind='barh'))

    plt.show()

    f1 = sklearn.metrics.f1_score(df_y_test,y_pred,average='weighted')

    print(f1)

    pd.DataFrame(y_pred_probs, index=df_y_test.index).to_csv(f'{Path}/Boosted_Predictions_ta_classification.csv')
