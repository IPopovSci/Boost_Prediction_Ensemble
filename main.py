from x_regression import boost_ta_classification, boost_ta_regression
from parameter_tune import xb_tuner


boost_ta_regression(csv_name='results.csv', k=30)
