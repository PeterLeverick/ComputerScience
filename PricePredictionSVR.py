'''
Created on June 14, 2022
@author: peter leverick
'''

'''
    This program attempts to predict the future price of ETH using Machine Learning (SVR) 
    YouTube ComputerScience 
'''

import sys
import os

import pandas as pd
import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot


import kraken_libs

#--------------- new column with the future price
def newcol_future_price(asset_df, future_days):
    asset_df[str(future_days)+'_Interval_Price_Forecast'] = asset_df[['Close']].shift(-future_days)
    print(asset_df)

    return asset_df

#--------------- create X and y 
def create_X_y(asset_df, future_days):

    X = np.array(asset_df[['Close']])
    X = X[:asset_df.shape[0] - future_days]  # delete last 5 rows NaN from shift
    print(X); print()

    y = np.array(asset_df[[str(future_days)+'_Interval_Price_Forecast']])
    y = y[:asset_df.shape[0] - future_days]  # delete last 5 rows NaN from shift
    print(y)

    return X, y

#--------------- split dataset training & test  
def split_training_test(X, y):

    from sklearn.model_selection import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split (X, y, test_size = 0.2)

    print(); print(y_train)

    return x_train, x_test, y_train, y_test



#---------------  create SVR model   
def create_svr_model (x_train, y_train):

    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf', C =1e3, gamma=0.00001)
    svr_rbf.fit(x_train, y_train)

    return 

    ''' create the model  '''
    
    g = input("\ncreate_svr_model   press key : "); print (g)



 #---------------- back testing 
def back_testing(y, yhat, last_date_ohlc):

    tp_limit = 0.1
    sl_limit = -0.03

    print('')
    print(y)
    print('')
    print(yhat)

    back_testing_df = pd.DataFrame({
        'actual_price': y,
        'predicted_price': yhat
        })

    back_testing_df.dropna(inplace=True)
    back_testing_df['returns_actuals'] = back_testing_df.actual_price.pct_change()
    back_testing_df['positive_return_actuals'] = np.where(back_testing_df.returns_actuals > 0, 1, -1)
    back_testing_df['returns_actuals_limit'] = np.where(back_testing_df.returns_actuals > tp_limit, tp_limit, back_testing_df['returns_actuals'])
    back_testing_df['returns_actuals_limit'] = np.where(back_testing_df.returns_actuals < sl_limit, sl_limit, back_testing_df['returns_actuals_limit'])
    back_testing_df['returns_predicted'] = back_testing_df.predicted_price.pct_change()
    back_testing_df['positive_return_predicted'] = np.where(back_testing_df.returns_predicted > 0, 1, -1)
    back_testing_df['good_predict'] = np.where(back_testing_df['positive_return_actuals'] * back_testing_df['positive_return_predicted'] > 0, 1, 0)
    back_testing_df['returns_predicted'] = abs(back_testing_df['returns_actuals_limit']) * back_testing_df['positive_return_actuals'] * back_testing_df['positive_return_predicted']
    #back_testing_df.dropna(subset=['returns_actuals'], inplace=True)

    asset_initial_price = back_testing_df['actual_price'].iloc[0]
    initial_capital = 10000

    print(asset_initial_price)
    print(initial_capital)

    cum_return = initial_capital * (1 + back_testing_df['returns_predicted']).cumprod()
    print(cum_return)

    final_capital = cum_return.iloc[-1]
    asset_final_price = back_testing_df['actual_price'].iloc[-1]
    profit = final_capital - initial_capital

    print(f"\ninitial capital = {initial_capital}")
    print(f"final capital = {final_capital}")
    print(f"profit 30 candles = {profit}   {profit/initial_capital}")
    print("last date ohlc: ", last_date_ohlc)
    print(f"good predictions = {back_testing_df['good_predict'].sum()}")

    from datetime import date
    today = date.today()
    print("today date is: ", today)


#----------------
def main(crypto_pair='ETHUSDT', interval='1440'):

    future_days = 5
 
    ''' get ohlc from kraken '''
    asset_df = kraken_libs.kraken_ohlc_lib.main(crypto_pair, interval) 
    print(asset_df)
    g = input("\nohlc   press key : "); print (g)

    ''' create a new column with the future price '''
    asset_df = newcol_future_price(asset_df, future_days)
    g = input("\nnew column   press key : "); print (g)

    ''' create independent variable X and dependent variable y '''
    X, y = create_X_y(asset_df, future_days)
    g = input("\ncreate_X_y   press key : "); print (g)

    ''' split training and test sets  '''
    x_train, x_test, y_train, y_test = split_training_test(X, y)
    g = input("\nsplit   press key : "); print (g)

    ''' create the model  '''
    create_svr_model (x_train, y_train)
    g = input("\ncreate_svr_model   press key : "); print (g)

    return






#------------- 
if __name__ == "__main__":
  main()
  g = input("\n\n\n End Program .... Press any key : "); print (g)



