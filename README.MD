# Boost_Prediction_Ensemble

This project uses XGBoost to perform classification/regression on outside predictions or original OHLCV data for cryptocurrency market data.

## Table of contents

* [Introduction](#Introduction)
* [Technologies](#Technologies)
* [Setup](#setup)
* [Features](#features)
* [Author](#Author)

### Introduction

This project is designed to further process predictions from outside sources, such as neural networks or to make predictions of future lagged returns in crypto markets.
It uses XGBoost to accomplish these tasks, via regression or classification.

For regression, predictions are done on lagged returns percentage change, and evaluated by computing number of direction signs that were guessed correctly.
Custom loss functions are used to create asymmetrical score for regression, to penalize wrong sign predictions more.

For classification, predictions are done on generated Signal, 0 for sell, 1 for buy. Evaluation is done via confusion matrix, accuracy and f1 score.

Hyper-parameter tuning is implemented for both tasks, and includes searching for optimal number of parameters.

Most of settings are hardcoded for 48-hour lagged interval, if you wish, feel free to clone the repository and change this.

Cryptowatch API data is used to receive most up to date data, and joined with selected csv.

### Technologies and libraries

* Anaconda 3
* Python 3.9
* XGBoost
* Pandas
* Numpy
* Scikit-learn
* Matplotlib
* Cryptowatch API

### Setup
Before running, assign your Cryptowatch API key in cryptowatch_data function in data_process.py

To run, clone the repository and launch one of the following from main.py:

boost_ta_regression() or boost_ta_classification()

Required parameters: 
csv_name - Either results.csv for ensembling outside predictions or crypto-pair.csv for custom cryptocurrency pair
k - number of best features to use
Make sure your loaded CSVs have dates within last 1000 hour, otherwise this will result in a date gap (Cryptowatch only returns last 1000 hours)

for tuning, run xb_tuner()

Required parameters:
type - regression or categorical
csv_name - Either results.csv for ensembling outside predictions or crypto-pair.csv for custom cryptocurrency pair
Make sure to modify xb_tuner() function in parameter_tune.py to include required search-grid settings

### Features

* Acquisition of crypto data through Rest-API.
* Data feature enhancement via technical analysis, OHLCV sums of the last n periods, moving averages of outside predictions, and day/month number augmentation.
* Classification and regression ensemble via XGBoost for outside predictions or OHLCV original data.
* Custom asymmetrical loss functions for regression.
* Hyper-parameter tuning for XGBoost models.
* Creation of target variables either via lagged returns for regression or signals for classification.

### Author

Created by Ivan Popov
