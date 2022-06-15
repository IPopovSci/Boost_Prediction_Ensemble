from pathlib import Path
import cryptowatch as cw
import numpy as np
import pandas as pd
import pytz
import ta

pd.set_option('display.max_columns', None)

est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%d/%m/%Y %H:%M:%S'

'''Module to work with data'''

Path = Path.cwd() / 'Data'  # Grabs data path based on current working directory

'''Loads csv file and gets rid of old index
Has 2 modes: Either a file with predictions from another model/network called results.csv or
An OHLCV csv containing ticker information, with column names [time,open,high,low,close,volume]
If loading OHLCV ticker information, expected time format is Unix time (ms)
For OHLCV dataset, the expected candle interval is 1h
For results.csv file, time is expected to be in hourly intervals in US/Eastern timezone
'''


def load_csv(name='results.csv'):
    df = pd.read_csv(f'{Path}/{name}')

    if name == 'results.csv':
        df = df.iloc[:, 1:]  # Gets rid of existing index column
        df['time'] = pd.to_datetime(df['time'], dayfirst=True).round('60min')
        df.set_index('time', inplace=True)
        df.index = df.index.tz_localize("US/Eastern").tz_convert(utc)

    if name != 'results.csv':
        df = df
        df['time'] = pd.to_datetime(df['time'], unit='ms', dayfirst=True, utc=True)  # Unix to datetime conversion

        ohlcv = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df.set_index('time', inplace=True)

        df = df.resample(rule='1h', offset=0).apply(ohlcv)

        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M:%S', )

    return df


'''Adds moving averages for predictions 
Accepts: Dataframe w/ prediction columns (Must have Pred in name)
lagged: True/False argument to add lagged returns
Returns: Dataframe
Only for prediction ensembling use'''


def add_info(df):
    windows = [48, 24, 12, 8, 6, 4, 3, 1]

    for col in df.columns:
        if 'Pred' in col:
            numeric_filter = filter(str.isdigit, col)
            numeric_string = int("".join(numeric_filter))

            window_index = windows.index(numeric_string)

            for window in windows[window_index:]:
                df[f'{window}_MA_{numeric_string}h'] = df[col].rolling(window=window).mean()

    return df


'''Splits dataframe into x and y variables for regression on lagged interval
Accepts: Dataframe, Lag target (String)
Returns: Dataframe'''


def x_y_split_regression(df, lag='48'):
    df_y = df[f'change_{lag}h']

    df_x = df.drop([f'change_{lag}h'], axis=1)
    return df_x, df_y


'''Splits dataframe into x and y variables for classification on signal.
Accepts: Dataframe
Returns: Dataframe'''


def x_y_split_categorical(df):
    df_y = pd.Series(df['Signal'], name='Signal')

    df_x = df.drop(['Signal'], axis=1)
    return df_x, df_y


'''Percent based train/test split
Accepts: Dataframes for x and y data, percent of data to split (float)
Returns: Dataframes for training and testing, for both x and y'''


def train_test_split(df_x, df_y, split_percent):
    df_x_train = df_x.iloc[:int(split_percent * len(df_x))]
    df_x_test = df_x.iloc[int(split_percent * len(df_x)):]

    df_y_train = pd.Series(df_y.iloc[:int(split_percent * len(df_y))])
    df_y_test = pd.Series(df_y.iloc[int(split_percent * len(df_y)):])

    return df_x_train, df_x_test, df_y_train, df_y_test


'''Module for retrieving the last 1000 OHCLV candles of a ticker
Will convert the time column into UTC Datetime Index
Accepts:
Pair: Ticker to use
Periods: Time interval for tickers

Returns: OHLCV Dataframe'''


def cryptowatch_data(pair, periods):
    cw.api_key = 'LZKL7ULRG322Z0793KU3'

    # Sometimes data from Kaggle has different pair differentiator, need to implement custom swaps.
    if pair == 'btcusd':
        pair = 'btcusdt'
    elif pair == 'ethusd':
        pair = 'ethusdt'

    hist = cw.markets.get(f"BINANCE:{pair}", ohlc=True, periods=[f'{periods}'])

    hist_list = getattr(hist, f'of_{periods}')  # Calling a method on a class to get the desired interval

    col = ['time', 'Open', 'High', 'Low', 'Close', 'volume_a',
           'Volume']  # Volume is the volume in USDT in this case, volume_a is the volume in first currency (Currently using volume_a)
    df = pd.DataFrame(hist_list, columns=col)
    df.drop(['Volume'], axis=1, inplace=True)  # getting rid of first currency volume

    df.rename(columns={'volume_a': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'},
              inplace=True)

    df['time'] = pd.to_datetime(df['time'], unit='s')  # Unix to datetime conversion

    df.set_index('time', inplace=True)
    df.index = df.index.tz_localize("utc")

    return df


'''Feature augmentation function
Adds sums of OHLCV candles over several periods to create new features
if lagged=True will create lagged returns based on lag_list
if signal=True will create buy (1) or sell (0) signal for 48 hour lagged returns
Don't use both, as this will cause a data leak

Additionally adds Hour and Dayweek columns, which indicate hour of the day, and numeric day of the week feature
Accepts: Dataframe
Returns: Dataframe'''


def ohlcv_add_info(df, lagged=False, signal=True):
    sum_periods = [48, 42, 38, 32]
    lag_list = [48, 24, 12, 4, 1]

    for period in sum_periods:
        df[f'open_{period}h'] = df['open'].rolling(min_periods=1, window=period).sum()
        df[f'close_{period}h'] = df['close'].rolling(min_periods=1, window=period).sum()
        df[f'volume_{period}h'] = df['volume'].rolling(min_periods=1, window=period).sum()
        df[f'low_{period}h'] = df['low'].rolling(min_periods=1, window=period).sum()
        df[f'high_{period}h'] = df['high'].rolling(min_periods=1, window=period).sum()

    if lagged:
        for interval in lag_list:
            df[f'change_{interval}h'] = df['close'].pct_change(interval, axis=0).shift(
                -interval)  # Create lagged returns
    if signal:
        df.insert(loc=0, value=(np.where(df['close'].pct_change(periods=48).shift(-48) > 0., 1, 0)), column='Signal')

    df['Hour'] = df.index.hour
    df['DayWeek'] = df.index.dayofweek

    return df


'''Adds technical analysis
Accepts: Dataframe with OHLCV columns
Returns: Dataframe with technical analysis features'''


def add_ta(ticker_data):
    df = ta.add_all_ta_features(ticker_data, open=f"open", high=f"high", low=f"low", close=f"close", volume=f"volume",
                                fillna=True, vectorized=False)  # Add all the ta!

    return df


'''Pipeline function
Builds the required data
Accepts:
split_percent: float, how much data to use for fit/prediction, % wise
pair: currency pair (string)
type: categorical/regression data preparation
csv_name: Csv to load in addition to cryptowatch API

Returns: Train/Test data for x and y'''


def pipeline_extra(split_percent, pair, periods, type='categorical', csv_name='results.csv'):
    df_preds = load_csv(name=f'{csv_name}')
    df_preds = add_info(df_preds)
    df_preds.reset_index(inplace=True)

    df_apidata = cryptowatch_data(pair, periods)
    df_apidata.reset_index(inplace=True)

    df = df_apidata.append(df_preds).drop_duplicates(subset='time').sort_values('time')

    df.set_index(df['time'], inplace=True, drop=True)
    del df['time']

    if type == 'categorical':
        df = ohlcv_add_info(df, lagged=False, signal=True)
    elif type == 'regression':
        df = ohlcv_add_info(df, lagged=True, signal=False)

    df = add_ta(df)
    df.fillna(value=0, inplace=True)

    if type == 'regression':
        df_x, df_y = x_y_split_regression(df)
    elif type == 'categorical':
        df_x, df_y = x_y_split_categorical(df)

    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, split_percent)
    return df_x_train, df_x_test, df_y_train, df_y_test
