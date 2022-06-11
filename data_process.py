import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import cryptowatch as cw
import ta
from datetime import timezone, datetime, timedelta
import pytz

pd.set_option('display.max_columns', None)

est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%d/%m/%Y %H:%M:%S'


#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
'''Module to work with data'''

Path = Path.cwd() / 'Data' #Grabs data path based on current working directory

'''Loads csv file and gets rid of old index'''
def load_csv():
    df = pd.read_csv(f'{Path}/results.csv',index_col='time', encoding='utf8')
    df = df.iloc[:,1:] #Gets rid of existing index column
    return df

'''Separate loading of predictions
Rounds the time to the nearest hour, to align with cryptowatch API time'''
def load_csv_round_index():
    df = pd.read_csv(f'{Path}/results.csv')
    df = df.iloc[:,1:] #Gets rid of existing index column

    df['time'] = pd.to_datetime(df['time'],dayfirst=True).round('60min')
    #print(df['time'])
    df.set_index('time', inplace=True)

    df.index = df.index.tz_localize("US/Eastern").tz_convert(utc)
    #print(df.index)

    #df.index = df.index.strftime('%d/%m/%Y %H:%M:%S')
    #print(df.iloc[-1,:])
    #print(df.index)
    return df


'''Adds moving averages and lagged returns to the dataframe
Accepts: Dataframe w/ prediction columns (Must have Pred in name)
Returns: Dataframe'''

def add_info(df):
    windows = [48,24, 12, 8, 6, 4, 3, 1]



    for col in df.columns:
        if 'Pred' in col:
            numeric_filter = filter(str.isdigit, col)
            numeric_string = int("".join(numeric_filter))
            print(numeric_string)

            window_index = windows.index(numeric_string)
            #print(window_index)

            #df[f'change_{numeric_string}h'] = df['price'].pct_change(numeric_string, axis=0).shift(-numeric_string) #Create lagged returns
            #Not sure if we need this, this won't be avaliable for predicting anyways! Will need 48h for regression, create separately

            for window in windows[window_index:]:
                df[f'{window}_MA_{numeric_string}h'] = df[col].rolling(window=window).mean()
    df['Hour'] = df.index.hour
    df['DayWeek'] = df.index.dayofweek
    # print(df['Hour'])
    # print(df['DayWeek'])
    #print(df.head(n=10))
    return df

'''Splits dataframe into x and y variables
We are only interested in 48h returns, as being most statistically significant to our model performance'''
def x_y_split_regression(df):
    df_y = df['change_48h']

    df_x = df.drop(['change_48h'],axis=1)
    return df_x,df_y

def x_y_split_categorical(df):
    df_y = df['Signal']

    df_x = df.drop(['Signal'],axis=1)
    return df_x,df_y

'''Percent based train/test split
Accepts: Dataframes for x and y data, percent of data to split (float)
Returns: Dataframes for training and testing, for both x and y'''
def train_test_split(df_x,df_y,split_percent):
    df_x_train = df_x.iloc[:int(split_percent*len(df_x))]
    df_x_test = df_x.iloc[int(split_percent*len(df_x)):]

    df_y_train = df_y.iloc[:int(split_percent*len(df_x))]
    df_y_test = df_y.iloc[int(split_percent*len(df_x)):]

    return df_x_train,df_x_test,df_y_train,df_y_test

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

    df['time'] = pd.to_datetime(df['time'], unit='s')#.dt.strftime('%d/%m/%Y %H:%M:%S') # Unix to datetime conversion

    #print(type(df['time']))
    df.set_index('time', inplace=True)
    df.index = df.index.tz_localize("utc")

    #print(df.index)
    #df.index = df.index.strftime('%d/%m/%Y %H:%M:%S')

    return df

def ohlcv_add_info(df):
    df['open_48h'] = df['open'].rolling(min_periods=1, window=48).sum()
    df['close_48h'] = df['close'].rolling(min_periods=1, window=48).sum()
    df['volume_48h'] = df['volume'].rolling(min_periods=1, window=48).sum()
    df['low_48h'] = df['low'].rolling(min_periods=1, window=48).sum()
    df['high_48h'] = df['high'].rolling(min_periods=1, window=48).sum()


    return df

def add_ta(ticker_data):
    df = ta.add_all_ta_features(ticker_data, open=f"open", high=f"high", low=f"low", close=f"close", volume=f"volume",
                                fillna=True, vectorized=False)  # Add all the ta!

    return df


def pipeline_extra(split_percent,pair,periods,type='categorical'):
    df_preds = load_csv_round_index()
    df_preds = add_info(df_preds)

    df_apidata = cryptowatch_data(pair,periods)
    df_apidata = ohlcv_add_info(df_apidata)
    df_apidata = add_ta(df_apidata)

    df=pd.merge(df_preds,df_apidata, how='inner', left_index=True,right_index=True)
    df.fillna(value=0,inplace=True)

    if type == 'regression':

        df_x,df_y = x_y_split_regression(df)
    elif type == 'categorical':
        df_x, df_y = x_y_split_categorical(df)
    df_x_train,df_x_test,df_y_train,df_y_test = train_test_split(df_x,df_y,split_percent)
    return df_x_train, df_x_test, df_y_train, df_y_test


#Add borutashape feature selection after we figure out xgboost
