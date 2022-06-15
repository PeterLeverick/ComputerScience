'''
Created on May 2021
New version error control Sept 21
@author: peter leverick
'''

""" 
    Kraken REST API  documentation  
    https://docs.kraken.com/rest/#operation/addStandardOrder
    https://docs.kraken.com/rest/#operation/getOHLCData
       
    ### --------- kraken Settling or closing a spot position on margin
    ### https://support.kraken.com/hc/en-us/articles/202966956-Settling-or-closing-a-spot-position-on-margin


"""

import pandas as pd 
import datetime
from datetime import date
import time, requests, sys, os
from requests.exceptions import HTTPError

import hmac
import hashlib
import json
import base64
import urllib



'''-----------------------------------------------------------------'''
'''  Kraken account --> balance, trades history ....   '''
'''-----------------------------------------------------------------'''
class Kraken_Account_Restapi(): 

    '''
    to access environment variables from whatever environment it is running in. 
    This could be either your local virtual environment for development or a service that you are hosting it on. 
    A useful package that simplifies this process is Python Decouple
        $ pip install python-decouple

    .env --> create a .env with environment variables.
    .gitignore -->  add .env to your .gitignore file so that you don’t commit this file of secrets to your code repository.

    in the .py file 
        from decouple import config
        API_USERNAME = config('USER')
        API_KEY = config('KEY')

    The benefit of using something like the above approach is that when you deploy your application to a cloud service, 
    you can set your environment variables using whatever method or interface the provider has and your Python code should still be able to access them. 
    Note that it is common convention to use capital letters for names of global constants in your code.
    '''

    ''' Kraken Account Balance '''
    def Account_Balance(self, Kraken_headers, Kraken_secret_key, coin_against):
        self.Kraken_headers = Kraken_headers
        self.Kraken_secret_key = Kraken_secret_key

        self.URI_path= '/0/private/Balance'
        self.URL_path = 'https://api.kraken.com/0/private/Balance'
        self.Kraken_nonce = str(int(time.time()*1000))
        self.Kraken_POST_data = {
        'nonce': self.Kraken_nonce
        } 

        for i in range(100):
            try:
                self.url_encoded_post_data = urllib.parse.urlencode(self.Kraken_POST_data) 
                self.encoded = (str(self.Kraken_POST_data['nonce'])+self.url_encoded_post_data).encode()  
                self.message = self.URI_path.encode() + hashlib.sha256(self.encoded).digest() 
                self.Kraken_signature = hmac.new(base64.b64decode(self.Kraken_secret_key), self.message, hashlib.sha512)
                self.Kraken_signature_digest = base64.b64encode(self.Kraken_signature.digest())
                self.Kraken_headers['API-Sign'] = self.Kraken_signature_digest.decode()
                self.response = requests.post(self.URL_path,data= self.Kraken_POST_data, headers = self.Kraken_headers)
        
                # If the response was successful, no Exception will be raised
                self.response.raise_for_status()
            except HTTPError as http_err:
                print(f'\nAccount_Balance --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'\nAccount_Balance --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nAccount_Balance --> Kraken API: Success')
                break
            print(f"--- Account_Balance --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Account_Balance --> Kraken API Error, we couldn't recover, 100 attempts") 
        
        self.result = self.response.json()
        
        print(f'\n --------- account balance ')
        print(self.result) 
        self.coin_against = coin_against
        self.balance_coin_against = self.result['result'][self.coin_against]
        print(f' \n current balance of {self.coin_against} is {self.balance_coin_against}')

        return self.balance_coin_against

        
    ''' Get last trade '''
    def Trades_History(self, Kraken_headers, Kraken_secret_key):
        self.Kraken_headers = Kraken_headers
        self.Kraken_secret_key = Kraken_secret_key

        self.URI_path= '/0/private/TradesHistory'
        self.URL_path = 'https://api.kraken.com/0/private/TradesHistory'
        self.Kraken_nonce = str(int(time.time()*1000))
        self.Kraken_POST_data = {
        'nonce': self.Kraken_nonce
        } 

        for i in range(100):
            try:
                self.url_encoded_post_data = urllib.parse.urlencode(self.Kraken_POST_data) 
                self.encoded = (str(self.Kraken_POST_data['nonce'])+self.url_encoded_post_data).encode()  
                self.message = self.URI_path.encode() + hashlib.sha256(self.encoded).digest() 
                self.Kraken_signature = hmac.new(base64.b64decode(self.Kraken_secret_key), self.message, hashlib.sha512)
                self.Kraken_signature_digest = base64.b64encode(self.Kraken_signature.digest())
                self.Kraken_headers['API-Sign'] = self.Kraken_signature_digest.decode()
                self.response = requests.post(self.URL_path,data= self.Kraken_POST_data, headers = self.Kraken_headers)
        
                # If the response was successful, no Exception will be raised
                self.response.raise_for_status()
            except HTTPError as http_err:
                print(f'\nTrades_History --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'\nTrades_History --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nTrades_History --> Kraken API: Success')
                break
            print(f"--- Trades_History --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Trades_History --> Kraken API Error, we couldn't recover, 100 attempts") 
        
        self.result = self.response.json()

        #print(self.result['result']['trades']) 
        # transform dic into df by selecting the trades without any header 
        # important trades are in columns not in rows, all rows of a column is a trade 
        self.trades_history_df = pd.DataFrame.from_dict(self.result['result']['trades']) 
        print(f'\n --------- last trade ')
        print(self.trades_history_df.iloc[:,0:1])     #this is the last trade --> all rows of the first column 

        return

    ''' Open orders '''
    def Open_Orders(self, Kraken_headers, Kraken_secret_key, stoploss_id, takeprofit_id):
        self.stoploss_id = stoploss_id
        self.takeprofit_id = takeprofit_id
        self.sloss_tf = True
        self.tprofit_tf = True 
        
        self.Kraken_headers = Kraken_headers
        self.Kraken_secret_key = Kraken_secret_key

        self.URI_path= '/0/private/OpenOrders'
        self.URL_path = 'https://api.kraken.com/0/private/OpenOrders'
        self.Kraken_nonce = str(int(time.time()*1000))
        self.Kraken_POST_data = {
        'nonce': self.Kraken_nonce
        } 

        for i in range(100):
            try:
                self.url_encoded_post_data = urllib.parse.urlencode(self.Kraken_POST_data) 
                self.encoded = (str(self.Kraken_POST_data['nonce'])+self.url_encoded_post_data).encode()  
                self.message = self.URI_path.encode() + hashlib.sha256(self.encoded).digest() 
                self.Kraken_signature = hmac.new(base64.b64decode(self.Kraken_secret_key), self.message, hashlib.sha512)
                self.Kraken_signature_digest = base64.b64encode(self.Kraken_signature.digest())
                self.Kraken_headers['API-Sign'] = self.Kraken_signature_digest.decode()
                self.response = requests.post(self.URL_path,data= self.Kraken_POST_data, headers = self.Kraken_headers)
        
                # If the response was successful, no Exception will be raised
                self.response.raise_for_status()
            except HTTPError as http_err:
                print(f'\nOpen_Orders --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'\nOpen_Orders --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nOpen_Orders --> Kraken API: Success')
                break
            print(f"--- Open_Orders --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Open_Orders --> Kraken API Error, we couldn't recover, 100 attempts") 
        
        self.result = self.response.json()

        print('\n ------- verifying if stoploss and takeprofit are still opened')
        try:
            print()
            print(self.result['result']['open'][self.stoploss_id]) 
            print(f'\nstoploss still opened {self.stoploss_id}')
            self.sloss_tf = True

        except Exception as err:
            print(f'\nstoploss triggered {self.stoploss_id}')  
            self.sloss_tf = False

        try:
            print()
            print(self.result['result']['open'][self.takeprofit_id]) 
            print(f'\ntakeprofit still opened {self.takeprofit_id}')
            self.tprofit_tf = True

        except Exception as err:
            print(f'\ntakeprofit triggered {self.takeprofit_id}') 
            self.tprofit_tf = False 

        #g = input("\n press key :")

        return self.sloss_tf, self.tprofit_tf


    ''' Last trade '''
    ''' add_order does not return price and id, so we read last trade just after creating it '''
    ''' in the main script that call this library we might need a latency sleep '''
    def Last_Trade(self, Kraken_headers, Kraken_secret_key):
        self.Kraken_headers = Kraken_headers
        self.Kraken_secret_key = Kraken_secret_key

        self.URI_path= '/0/private/TradesHistory'
        self.URL_path = 'https://api.kraken.com/0/private/TradesHistory'
        self.Kraken_nonce = str(int(time.time()*1000))
        self.Kraken_POST_data = {
        'nonce': self.Kraken_nonce
        } 

        for i in range(100):
            try:
                self.url_encoded_post_data = urllib.parse.urlencode(self.Kraken_POST_data) 
                self.encoded = (str(self.Kraken_POST_data['nonce'])+self.url_encoded_post_data).encode()  
                self.message = self.URI_path.encode() + hashlib.sha256(self.encoded).digest() 
                self.Kraken_signature = hmac.new(base64.b64decode(self.Kraken_secret_key), self.message, hashlib.sha512)
                self.Kraken_signature_digest = base64.b64encode(self.Kraken_signature.digest())
                self.Kraken_headers['API-Sign'] = self.Kraken_signature_digest.decode()
                self.response = requests.post(self.URL_path,data= self.Kraken_POST_data, headers = self.Kraken_headers)
        
                # If the response was successful, no Exception will be raised
                self.response.raise_for_status()
            except HTTPError as http_err:
                print(f'\nLast_Trade --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'\nLast_Trade --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nLast_Trade --> Kraken API: Success')
                break
            print(f"--- Last_Trade --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Last_Trade --> Kraken API Error, we couldn't recover, 100 attempts") 
        
        self.result = self.response.json()

        #self.result['result']['trades']['TBVVFF-HFZRY-EO2OKB']) # id header of ordertxid

        #self_result is a dic, first entry is the last trade, but we don't know the first key
        #we get all the keys into a list and pull the first 
        first_key_result_dic = list(self.result['result']['trades'].keys())[0] #get first key of dic
        print(first_key_result_dic)
        #print (self.result['result']['trades'][first_key_result_dic])
        #print (self.result['result']['trades'][first_key_result_dic]['price'])

        self.last_trade_dic = {
            'ordertxid': self.result['result']['trades'][first_key_result_dic]['ordertxid'],
            'pair' : self.result['result']['trades'][first_key_result_dic]['pair'],
            'time' : self.result['result']['trades'][first_key_result_dic]['time'],
            'type_buy_sell' : self.result['result']['trades'][first_key_result_dic]['type'],
            'order_type' : self.result['result']['trades'][first_key_result_dic]['ordertype'],
            'volume' : self.result['result']['trades'][first_key_result_dic]['vol'],
            'price' : self.result['result']['trades'][first_key_result_dic]['price']
        }
 
        print (self.last_trade_dic)

        return self.last_trade_dic


    
'''-----------------------------------------------------------------'''
'''   Kraken trading --->  add orders, cancel, OHLC ....            '''
'''-----------------------------------------------------------------'''
class Kraken_Trading_Restapi(): 

    ''' Get last ticker price  '''
    def Get_Ticker(self, pair, since=''):
        self.pair = pair
        self.since = since

        self.url = 'https://api.kraken.com/0/public/Trades' 

        for i in range(100):
            try:
                self.response_kraken = requests.post(self.url, 
                params=
                        {'pair':self.pair,    
                        'since':self.since}, 
                        headers={"content-type":"application/json"})
                
                # If the response was successful, no Exception will be raised
                self.response_kraken.raise_for_status()
            except HTTPError as http_err:
                print(f'\nGet_Ticker --> Kraken API: HTTP error occurred: {http_err}')   
            except Exception as err:
                print(f'\nGet_Ticker --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nGet_Ticker --> Kraken API: Success')
                break
            print(f"\n--- Get_Ticker --> Kraken API error, retry # {i}")
            #time.sleep(8)       

        if i == 99: sys.exit("Get_Ticker --> Kraken API Error, we couldn't recover, 100 attempts")     

        self.since = self.response_kraken.json()['result']['last']
        self.last_ticker_price = self.response_kraken.json()['result'][pair][-1][0]      #last ticker price

        return float(self.last_ticker_price) 


    ''' Get last close from kraken. The last close right now from the last candel (in contruction) '''
    def Get_OHLC(self, pair, interval):
        self.pair = pair
        self.interval = interval                    # 1440 = 1 day, 1 = 1 min
        self.since = ''                             # return las 720 datapoints
        
        self.response_json = {}
        self.url = 'https://api.kraken.com/0/public/OHLC'  #only last 720 datapoints 
           
        for i in range(100):
            try:
                self.response_kraken = requests.post(self.url, 
                params=
                        {'pair': self.pair,
                        'interval': self.interval,     
                        'since': self.since}, 
                        headers={"content-type":"application/json"})

                # If the response was successful, no Exception will be raised
                self.response_kraken.raise_for_status()
            except HTTPError as http_err:
                print(f'Get_OHLC --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'Get_OHLC --> Kraken API: Other error occurred: {err}')  
            else:
                print('Get_OHLC --> Kraken API: Success')
                break
            print(f"--- Get_OHLC --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Get_OHLC --> Kraken API Error, we couldn't recover, 100 attempts")     

        #print(self.response_kraken.json())
        #self.kraken_json = self.response_kraken.json()
        #print(self.kraken_json['result'][self.pair][-1][4])        # Close

        self.last_close = self.response_kraken.json()['result'][self.pair][-1][4]   # Close

        return self.last_close


    #--- Get TimeStamp of last processed order in Kraken, that we'll be the second last of the json file   
    def Get_TimeS_Last_Kraken(self, pair, interval):
        self.pair = pair
        self.interval = interval                    # 1440 = 1 day, 1 = 1 min
        self.since = ''                             # return last 720 datapoints
        
        self.response_json = {}
        self.url = 'https://api.kraken.com/0/public/OHLC'  #only last 720 datapoints 

           
        for i in range(100):
            try:
                self.response_kraken = requests.post(self.url, 
                params=
                        {'pair': self.pair,
                        'interval': self.interval,     
                        'since': self.since}, 
                        headers={"content-type":"application/json"})

                # If the response was successful, no Exception will be raised
                self.response_kraken.raise_for_status()
            except HTTPError as http_err:
                print(f'Get_TimeS_Last_Kraken --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'Get_TimeS_Last_Kraken --> Kraken API: Other error occurred: {err}')  
            else:
                print('Get_TimeS_Last_Kraken --> Kraken API: Success')
                break
            print(f"--- Get_TimeS_Last_Kraken --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Get_TimeS_Last_Kraken --> Kraken API Error, we couldn't recover, 100 attempts")     

        #print(self.response_kraken.json())
        self.last_candle_processed = self.response_kraken.json()['result']['last']
        #print(f"last candle processed kraken {self.last_candle_processed}")
        self.last_candle = self.response_kraken.json()['result'][self.pair][-1][0]   
        #print(f"last candle {self.last_candle}")
        #g = input("ohlc: "); print (g)

        return self.last_candle_processed, self.last_candle


    ''' ----- cancel order '''
    ### ---- documentation -->  https://docs.kraken.com/rest/#operation/cancelOrder 
    def Cancel_Order(self, order_to_cancel, Kraken_headers, Kraken_secret_key):
        self.txid = order_to_cancel

        self.Kraken_headers = Kraken_headers
        self.Kraken_secret_key = Kraken_secret_key
        self.URI_path = '/0/private/CancelOrder'
        self.URL_path = 'https://api.kraken.com/0/private/CancelOrder'     
        self.Kraken_nonce = str(int(time.time()*1000))
 
        self.Kraken_POST_data = {
            'nonce': self.Kraken_nonce,         # required, used in construction of API-Sign header
            'txid':self.txid                    # order to cancel id 
        } 

        for i in range(100):
            try:
                self.url_encoded_post_data = urllib.parse.urlencode(self.Kraken_POST_data) 
                self.encoded = (str(self.Kraken_POST_data['nonce'])+self.url_encoded_post_data).encode()  
                self.message = self.URI_path.encode() + hashlib.sha256(self.encoded).digest() 
                self.Kraken_signature = hmac.new(base64.b64decode(self.Kraken_secret_key), self.message, hashlib.sha512)
                self.Kraken_signature_digest = base64.b64encode(self.Kraken_signature.digest())
                self.Kraken_headers['API-Sign'] = self.Kraken_signature_digest.decode()
                self.response = requests.post(self.URL_path,data= self.Kraken_POST_data, headers = self.Kraken_headers)
        
                # If the response was successful, no Exception will be raised
                self.response.raise_for_status()
            except HTTPError as http_err:
                print(f'\nCancel_Order --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'\nCancel_Order --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nCancel_Order --> Kraken API: Success')
                break
            print(f"--- Cancel_Order --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Cancel_Order --> Kraken API Error, we couldn't recover, 100 attempts") 
        
        self.result = self.response.json()
        print(self.result) 
   
        print(f'just canceled order_id = {self.txid}')
        #g = input("Cancel_Order: "); print (g)

        return (self.result)

    
    ''' ----- add new order to kraken '''
    ### ---- documentation -->  https://docs.kraken.com/rest/#operation/addStandardOrder 
    ### ---- examples  https://support.kraken.com/hc/en-us/articles/360000920786-How-to-Add-standard-orders-with-different-parameters
    def Add_Order(self, new_trade_dic, Kraken_headers, Kraken_secret_key):
        self.new_trade_dic = new_trade_dic
        
        self.Kraken_headers = Kraken_headers
        self.Kraken_secret_key = Kraken_secret_key
        self.URI_path = '/0/private/AddOrder'
        self.URL_path = 'https://api.kraken.com/0/private/AddOrder'     
        self.Kraken_nonce = str(int(time.time()*1000))

        self.Kraken_POST_data = {
            'nonce': self.Kraken_nonce,                     # required, used in construction of API-Sign header
            'pair': self.new_trade_dic['pair'],             # pair
            'type': self.new_trade_dic['type_buy_sell'],    # "buy" "sell"
            'ordertype':self.new_trade_dic['order_type'],   # string, "market" "limit" "stop-loss" "take-profit" "stop-loss-limit" "take-profit-limit" "settle-position" 
            'leverage':self.new_trade_dic['leverage'],      # string, (default = none), I use 1 when short 
            'volume':self.new_trade_dic['volume'],          # string, Order quantity 
            'price':self.new_trade_dic['price'],             # string, Limit price for limit orders
            'price2':self.new_trade_dic['price2'], 
        } 

        for i in range(100):
            try:
                self.url_encoded_post_data = urllib.parse.urlencode(self.Kraken_POST_data) 
                self.encoded = (str(self.Kraken_POST_data['nonce'])+self.url_encoded_post_data).encode()  
                self.message = self.URI_path.encode() + hashlib.sha256(self.encoded).digest() 
                self.Kraken_signature = hmac.new(base64.b64decode(self.Kraken_secret_key), self.message, hashlib.sha512)
                self.Kraken_signature_digest = base64.b64encode(self.Kraken_signature.digest())
                self.Kraken_headers['API-Sign'] = self.Kraken_signature_digest.decode()
                self.response = requests.post(self.URL_path,data= self.Kraken_POST_data, headers = self.Kraken_headers)
        
                # If the response was successful, no Exception will be raised
                self.response.raise_for_status()
            except HTTPError as http_err:
                print(f'\nAdd_Order --> Kraken API: HTTP error occurred: {http_err}')  
            except Exception as err:
                print(f'\nAdd_Order --> Kraken API: Other error occurred: {err}')  
            else:
                print('\nAdd_Order --> Kraken API: Success')
                break
            print(f"--- Add_Order --> Kraken API error, retry # {i}")
            time.sleep(5)

        if i == 99: sys.exit("Add_Order --> Kraken API Error, we couldn't recover, 100 attempts") 
        
        self.result = self.response.json()
        print(self.result) 

        print(self.new_trade_dic['pair'])
        #g = input("Add_Order: just added order"); print (g)

        return (self.result)


    ### ------- decode result as the output of add order
    def Result_Decoding(self):

        # new order result (from trades history)
        self.result = {'ordertxid': 'OZYKU6-BBZQS-7JBAMO', 'pair': 'XXBTZUSD', 'time': 1619838038.7857, 'type_buy_sell': 'buy', 'order_type': 'market', 'volume': '0.00030000', 'price': '58186.40000'}
        print()
        print(self.result)
        print(self.result['ordertxid'])  
        print(self.result['pair'])    
        print(self.result['time'])     
        print(self.result['type_buy_sell'])    
        print(self.result['order_type'])
        print(self.result['volume'])
        print(self.result['price'])          
        #g = input("\n  new order : ")      

        # stop loss limit result (from add_order)
        self.result = {'error': [], 'result': {'descr': {'order': 'sell 0.00030000 XBTUSD @ stop loss 57604.5 -> limit 57593.0'}, 'txid': ['OY3EFD-HSPLA-HNKB5P']}}
        print()
        print(self.result)
        print(self.result['result']['descr']['order'])  #stop loss limit order description
        print(self.result['result']['txid'][0])         #it is an array, we take 1st element 
        #g = input("\n  stop loss : ")

         # take profit limit result (from add_order)
        self.result = {'error': [], 'result': {'descr': {'order': 'sell 0.00030000 XBTUSD @ take profit 62550.4 -> limit 62537.9'}, 'txid': ['OJEH2B-ZNHDS-UO56A6']}}
        print()
        print(self.result)
        print(self.result['result']['descr']['order'])  #take profit limit order description
        print(self.result['result']['txid'][0])         #it is an array, we take 1st element 
        #g = input("\n take profit : ")

        return 

 




'''-----------------------------------------------------------------'''
'''   Main function, only for standalone mode                       '''
'''-----------------------------------------------------------------'''
def main():
    
    #instructions for standalone mode here

    return 



#------------- 
if __name__ == "__main__":
  main()
  g = input("\n\n\n End Rest_API library .... Press any key : "); print (g)





  
