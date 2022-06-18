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

#--------------- get ohlc fom kraken 
def get_ohlc(crypto_pair, interval):
    asset_df = kraken_libs.kraken_ohlc_lib.main(crypto_pair, interval) 
    print(asset_df)

    return asset_df

#--------------- new column with the future price
def newcol_future_price(asset_df, future_days):
    asset_df[str(future_days)+'_Price_Forecast'] = asset_df[['Close']].shift(-future_days)
    asset_df.fillna(method='ffill', inplace=True) #  take the last value seen for a column
    #asset_df.fillna(0, inplace=True) #  take the last value seen for a column

    print(asset_df)
    print(f"\n asset_df.shape = {asset_df.shape}")


    return asset_df

#--------------- create X and y 
def create_X_y(asset_df, future_days):

    X = np.array(asset_df[['Close']])   # [[]] we want two dimensions array
    #X = X[:asset_df.shape[0] - future_days]  # delete last rows NaN from shift
    print(X); 
    print(X.shape)
    print()
    g = input("\nX   press key : "); print (g)

    y = np.array(asset_df[str(future_days)+'_Price_Forecast']) # [] we want one dimensions array
    #y = y[:-future_days]  # delete last rows NaN from shift
    print(y)
    print(y.shape)
    g = input("\ny   press key : "); print (g)

    return X, y

#--------------- split dataset training & test  
def split_training_test(X, y):

    from sklearn.model_selection import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split (X, y, test_size = 0.1, shuffle=False) 
    # test_size 10% better result than 20%
    # kraken provides 720 candels, maybe better to get the history with ticks? 
    # shuffle = false to keep sequential (big difference), we want x_test + y_test from the end
    # verify if shuffle = off takes takes x_test y_test  from the end (this is what we want)

    #x_test = np.array([[1724.10, 1794.95, 1813.35, 1997.56, 1938.75, 1817.91, 1833.12, 1774.62, 1804.22, 1804.35]])
    #x_test = x_test.reshape(10,1)   #1 col 10 rows
    #y_test = np.array([1860.14, 1815.60, 1792.58, 1789.21, 1661.66, 1528.13, 1434.11, 1210.85, 1212.60, 1237.54])

    #print(); print(y_train)
    print(f"\nx_train shape = {x_train.shape}")
    print(f"x_test shape = {x_test.shape}")
    print(f"y_train shape = {y_train.shape}")
    print(f"y_test shape = {y_test.shape}")
    print(f"x_test dim = {x_test.ndim}")
    print(f"y_test dim = {y_test.ndim}")
    g = input("shape  press key : "); print (g)

    return x_train, x_test, y_train, y_test



#---------------  create SVR model   
def create_svr_model (x_train, y_train, x_test, y_test):

    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf', C =1e3, gamma=0.00001)
    svr_rbf.fit(x_train, y_train)

    svr_rbf_confidence = svr_rbf.score(x_test, y_test)
    print(f"svr_rbf_confidence = {svr_rbf_confidence}")
    g = input("svr_rbf_confidence  press key : "); print (g)

    # manual entry of last prices, skip previous x_test
    # Each price will pridct as many days as future_days variable
    #x_test = np.array([[1815,1792,1789,1661,1528]])
    #x_test = np.array([[1434]])
    #x_test = x_test.reshape(5,1)   #1 col 10 rows
       
    #svr_prediction = svr_rbf.predict([[1661]]) #only valid for a single price?
    svr_prediction = svr_rbf.predict(x_test)
 
    return svr_prediction

#---------------  summary & plot   
def summary_plot (x_test, y_test, svr_prediction, future_days):

    summary_df = pd.DataFrame(data=x_test, columns=['x_test'])
    summary_df['y_test'] = y_test
    summary_df['forecast1'] = svr_prediction
    summary_df['forecast2'] = summary_df['forecast1'].shift(-future_days)
    summary_df['gap'] = summary_df['forecast2'] - summary_df['y_test']
    print(summary_df)

    plt.figure(figsize=(12,4))
    #plt.plot(summary_df['x_test'], label='x_test', lw=2, alpha=.7)
    plt.plot(summary_df['y_test'], label='y_test', lw=2, alpha=.7)
    #plt.plot(summary_df['forecast1'], label='forecast1', lw=2, alpha=.7)
    plt.plot(summary_df['forecast2'], label='forecast2', lw=2, alpha=.7)
    plt.title('Prediction vs Actual')
    plt.ylabel('Price in USDT')
    plt.xlabel('Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    return svr_prediction




#---------------- main
def main(crypto_pair='ETHUSDT', interval='1440'):

    future_days = 1
 
    ''' get ohlc from kraken '''
    asset_df = get_ohlc(crypto_pair, interval) 
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
    svr_prediction = create_svr_model (x_train, y_train, x_test, y_test)
    g = input("\ncreate_svr_model   press key : "); print (g)

    ''' summary & plot '''
    summary_plot (x_test, y_test, svr_prediction, future_days)
    g = input("\nsummary_plot   press key : "); print (g)

    return



#------------- 
if __name__ == "__main__":
  main()
  g = input("\n\n\n End Program .... Press any key : "); print (g)



'''
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


'''