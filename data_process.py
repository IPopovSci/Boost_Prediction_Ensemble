import pandas as pd
import numpy as np
from pathlib import Path

'''Module to work with data'''

Path = Path.cwd() / 'Data' #Grabs data path based on current working directory

'''Loads csv file and gets rid of old index'''
def load_csv():
    df = pd.read_csv(f'{Path}/results.csv',index_col='time')
    df = df.iloc[:,1:] #Gets rid of existing index column
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
            #print(numeric_string)

            window_index = windows.index(numeric_string)

            df[f'price_{numeric_string}h'] = df['price'].pct_change(numeric_string, axis=0).shift(-numeric_string) #Create lagged returns

            for window in windows[window_index:]:
                df[f'{window}_MA'] = df[col].rolling(window=window).mean()
    #df = df.dropna() #Drops nan values created by moving averages and price lags #XGBOOST is resilient against missing values, maybe not needed?
    return df

'''Splits dataframe into x and y variables
We are only interested in 48h returns, as being most statistically significant to our model performance'''
def x_y_split(df):
    df_y = df['price_48h']
    df_x = df.drop(['price_48h'],axis=1)
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

'''A mini pipeline function to process data
Returns: Dataframes for training and testing, for both x and y'''
def small_pipeline(split_percent=0.5):
    df = load_csv()
    df = add_info(df)
    df_x,df_y = x_y_split(df)
    df_x_train,df_x_test,df_y_train,df_y_test = train_test_split(df_x,df_y,split_percent)
    return df_x_train,df_x_test,df_y_train,df_y_test

#Add borutashape feature selection after we figure out xgboost
