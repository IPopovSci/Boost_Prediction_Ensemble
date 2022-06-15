from pathlib import Path
import matplotlib.pylab as plt
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from data_process import pipeline_extra
from feature_selection import select_features
from loss_functions import eval_metric_signs, assymetric_mse_train

'''Script to run XGBRegressor, fits on data and then predicts values and writes them to Boosted_predictions.csv'''

Path = Path.cwd() / 'Data'

'''Function to run regression
Performs feature selection via SelectKBest method of sklearn
Accepts: Csv name to load, number of features to select
Returns: Print out % of signs guessed correctly; Saves the predictions to csv file'''


def boost_ta_regression(csv_name='results.csv', k=30):
    regressor = xgb.XGBRegressor(obj=assymetric_mse_train,
                                 random_state=1337)  # ,learning_rate=1.65,n_estimators=900,max_depth=20,reg_alpha=0.2,reg_lambda=0.05

    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.75, 'btcusd', '1h', type='regression',
                                                                  csv_name=csv_name)

    df_x_train, df_x_test, fs = select_features(df_x_train, df_y_train, df_x_test, k=k, score_func='regression')

    regressor.fit(df_x_train, df_y_train)

    y_pred = regressor.predict(df_x_test)

    y_true = df_y_test

    score = eval_metric_signs(y_true[:48], y_pred[:48])

    print('Percent of signs guessed correctly is: ', score)

    pd.Series(y_pred, index=df_y_test.index).to_csv(f'{Path}/Boosted_Predictions_ta.csv')

    return y_pred


'''Function to run classification
Performs feature selection via SelectKBest method of sklearn
Weights the classes to avoid class imbalance
Accepts: Csv name to load, number of features to select
Returns: Print out of confusion matrix, accuracy, and f1 score; Saves the predictions to csv file'''


def boost_ta_classification(csv_name='results.csv', k=30):
    regressor = xgb.XGBClassifier(gamma=0.1, learning_rate=1, max_depth=30, n_estimators=85, reg_alpha=0.25,
                                  reg_lambda=0.5, num_class=2, objective='multi:softprob',
                                  eval_metric='mlogloss')  # ,learning_rate=1.65,n_estimators=900,max_depth=20,reg_alpha=0.2,reg_lambda=0.05

    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.9, 'btcusd', '1h', type='categorical',
                                                                  csv_name=csv_name)

    x_train, df_x_test, fs = select_features(df_x_train, df_y_train, df_x_test, k=k, score_func='classification')
    class_weights = sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced', y=df_y_train)

    regressor.fit(x_train, df_y_train, sample_weight=class_weights)

    y_pred = regressor.predict(df_x_test)
    y_pred_probs = regressor.predict_proba(df_x_test)

    cm = confusion_matrix(df_y_test[:-48],
                          y_pred[:-48])  # Last 48 points are meaningless, as we don't know their values yet
    print(cm)

    accuracy = accuracy_score(df_y_test, y_pred)
    print(accuracy)

    (pd.Series(regressor.feature_importances_, index=fs.get_feature_names_out())
     .plot(kind='barh'))

    plt.show()

    f1 = sklearn.metrics.f1_score(df_y_test, y_pred, average='weighted')
    print(f1)

    pd.DataFrame(y_pred_probs, index=df_y_test.index).to_csv(f'{Path}/Boosted_Predictions_ta_classification.csv')
