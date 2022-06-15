from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_classif

'''Selects best k features, transforms train and test data to said number of features
Accepts: x/y train dataset, x  test dataset, number of features to use, score function to use regression/classification
Returns: x train and test fitted sets, SeleckKBest object'''


def select_features(X_train, y_train, X_test, k, score_func):
    # configure to select a subset of features
    if score_func == 'classification':
        fs = SelectKBest(score_func=f_classif, k=k)
    elif score_func == 'regression':
        fs = SelectKBest(score_func=mutual_info_regression, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
