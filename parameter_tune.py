from pathlib import Path
import sklearn
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_classif
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from data_process import pipeline_extra
from loss_functions import assymetric_mse_sklearn

'''Module to search for optimal hyper-parameters, uses GridSearchCV'''

Path = Path.cwd() / 'Data'

'''Hyper-parameter tuning function
Looks for best hyper-parameters, including number of features to use with SelectKBest
Accepts: Type regression/categorical, csv name to load
Returns: Print out of best score and parameters found'''


def xb_tuner(type='regression', csv_name='results.csv'):
    df_x_train, df_x_test, df_y_train, df_y_test = pipeline_extra(0.9, 'btcusd', '1h', type=type, csv_name=csv_name)

    if type == 'categorical':
        model = xgb.XGBClassifier(num_class=2, objective='multi:softprob',
                                  eval_metric='mlogloss')
        fs = SelectKBest(score_func=f_classif)

    elif type == 'regression':
        fs = SelectKBest(score_func=mutual_info_regression)

    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
    # define the grid
    grid = dict()
    grid['sel__k'] = [120, 110, 100, 90, 80, 70, 60, 50, 40, 30]
    grid['lr__max_depth'] = [5, 10, 15, 20, 25, 30, 50, 75, 100]
    grid['lr__n_estimators'] = [25, 50, 75, 100, 150, 200]
    grid['lr__learning_rate'] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
    grid['lr__gamma'] = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    grid['lr__reg_alpha'] = [0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]
    grid['lr__reg_lambda'] = [1, 0.9, 0.8, 0.7, 0.6]
    # define the grid search
    if type == 'categorical':
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')
    elif type == 'regression':
        scorer = make_scorer(assymetric_mse_sklearn, greater_is_better=False)

    search = GridSearchCV(pipeline, grid, scoring=scorer, n_jobs=-1, cv=cv)
    # perform the search
    results = search.fit(df_x_train, df_y_train)
    # summarize best
    print('Best score: %.5f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
