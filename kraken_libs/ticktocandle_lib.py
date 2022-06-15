'''
Created Oct0ber 2021 
@author: peter leverick

kraken --> https://docs.kraken.com/rest/#operation/getOHLCData

As kraken only provides the last 720 points we need to create an scrip to get a large range 
The objective for this script is to 
    Get ticks from Kraken
    Reshape ticks into candles 
    Export tiacks & candles into a csv files (ticks.csv, ohlcv.csv) 
Then other programs can get candles and process them 

Ticks storaged in a list (much faster than dataframe)

'''

#------------------------------------------------------------------------
#  
#           OHLC from Ticks 
#
# source: 
#  https://www.kraken.com/features/api#get-recent-trades
#
#  How to retrieve historical time and sales (trading history) using the REST API Trades endpoint.
#  https://support.kraken.com/hc/en-us/articles/218198197-How-to-retrieve-historical
#
#   url = 'https://api.kraken.com/0/public/OHLC'  #only 720
#   url = 'https://api.kraken.com/0/public/Trades' 
#
# https://api.kraken.com/0/public/Trades?pair=xbtusd&30&since=1559347200000000000
#
#------------------------------------------------------------------------


# ---------- imports 
import pandas as pd 
import numpy as np
import datetime
import time, json, requests, sys
from time import time, ctime
from datetime import datetime
from requests.exceptions import HTTPError


#------------ functions 

''' Get_Dates_Range <-- if we are running this script in standalone mode  '''
def Get_Dates_Range(interval):
    import time
    from time import ctime 
    from datetime import datetime
    
    # -- 1st since for kraken & now for the 1st iteration while 
    since = now = datetime(2021, 9, 16, 0, 0, 00000)  
    print(f"\nstart date    since {since}      now {now}")

    # -- since 
    since = datetime.timestamp(since)
    print(f"\nsince    timestamp, {since}     readable -->   {time.ctime(since)}")
    since = str(since)                  
    since = since[0:10] + '0'* 9        #since for kraken 

    # -- now is current in the while loop comparation    
    now = int(datetime.timestamp(now))
    print(f"now      timestamp, {now}       readable -->   {time.ctime(now)}")

    # -- end +1 interval (interval * 60sec (timestamp)) <-- to avoid "till end effect" and include last interval
    end = datetime(2021, 9, 25, 15, 00, 00000)      
    print(f"\nend date --> {end}")
    end = int(datetime.timestamp(end)) + int(interval) * 60         # the trick here 
    print(f"2nd end    timestamp, {end}     readable -->   {time.ctime(end)}")

    return since, now, end 

''' Process_Dates_Range <-- if acting as lib (called from another program that passed in the range of dates) '''
def Process_Dates_Range(interval, since, start_date, end_date):
    import time
    from time import ctime 
    import datetime
    from datetime import datetime
    
    since = now = start_date; end = end_date
    print(f"\nimport start date    since {since}      now {now}")
    print(f"import end date  {end}")

    # 1/ since for kraken & for now in the 1st iteration of while 
    format = "%Y-%m-%d %H:%M:%S"
    since = datetime.strptime(since, format)
    print(f"\nsince    datetime {since} ")

    since = datetime.timestamp(since)
    print(f"since    timestamp, {since}     readable -->   {time.ctime(since)}")
    since = str(since)                  
    since = since[0:10] + '0'* 9        #since for kraken 

    # 2/ now is current in the while loop    
    #now = int(datetime.timestamp(now))
    now = datetime.strptime(now, format)
    print(f"now    datetime {now} ")

    now = datetime.timestamp(now)
    print(f"now    timestamp, {now}     readable -->   {time.ctime(now)}")

    # 3/ end +1 interval (interval * 60sec (timestamp)) <-- to avoid "till end effect" and include last interval
    #end = datetime(2021, 9, 25, 15, 00, 00000)      
    end = datetime.strptime(end, format)
    print(f"end    datetime {end} ")

    end = datetime.timestamp(end) + int(interval) * 60         # the trick here
    print(f"end    timestamp, {end}     readable -->   {time.ctime(end)}")

    return since, now, end 


''' Get Ticks from Kraken '''
def Get_Ticks_Kraken(pair, interval, since):
        
    url = 'https://api.kraken.com/0/public/Trades' 

    for i in range(100):
        try:
            response_kraken = requests.post(url, 
            params=
                    {'pair':pair,
                    'interval':interval,     
                    'since':since}, 
                    headers={"content-type":"application/json"})
                
            # If the response was successful, no Exception will be raised
            response_kraken.raise_for_status()
        except HTTPError as http_err:
            print(f'\nGet_OHLC --> Kraken API: HTTP error occurred: {http_err}')   
        except Exception as err:
            print(f'\nGet_OHLC --> Kraken API: Other error occurred: {err}')  
        else:
            print('\nGet_OHLC --> Kraken API: Success')
            break
        print(f"\n--- Get_OHLC --> Kraken API error, retry # {i}")
        time.sleep(8)       

    if i == 99: sys.exit("Get_OHLC --> Kraken API Error, we couldn't recover, 100 attempts")     

    since = response_kraken.json()['result']['last']

    return response_kraken.json(), since


''' Process ticks json into a list '''
def Process_Kraken_Json(response_json, pair, l_ticks, i_ticks, end):

    while True:
        try:
            new_row = [ctime(response_json['result'][pair][i_ticks][2]),  #timestamp 
                       response_json['result'][pair][i_ticks][0],         #price
                       response_json['result'][pair][i_ticks][1],         #volume 
                       response_json['result'][pair][i_ticks][3]          #s/b 
                       ] 
            print (new_row)
            now = int((response_json['result'][pair][i_ticks][2]))
            if int(now) > int(end): break       # if we are over the 'end' timestamp 
            l_ticks += [new_row]
            i_ticks += 1
            
        except Exception as err:                # we only enter here if by chance last tick = end 
            print(f'exception Process_Kraken_Json: {err}')
            break 
        
    return l_ticks, now


''' transform price list into a DataFrame '''
def l_price_into_df_price(l_ticks):

    # list of ticks labels --> for df  
    l_labels = ['Date', 'Price','Volume', 'BS']

    # ' transform list into a df '
    df = pd.DataFrame.from_records(l_ticks, columns=l_labels)   # l_ticks --> list, l_labels --> column names 
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True) 

    df[['Price', 'Volume']] = df[['Price', 'Volume']].apply(pd.to_numeric)
    #df= df.iloc[:-1,:]       # no need 'end' manages it  # drop last row, we want yesterday, today is starting 

    return(df)


''' compute OHLCV from ticks df'''
def Compute_OHLCV(df_ticks, interval):

    # OHLC V resampling --> https://github.com/python-streamz/streamz/issues/119
    # https://techflare.blog/mastering-dataframe-how-to-aggregate-ohlcv-data-in-a-different-time-period/
    resample_interval = str(interval) + 'Min'       #e.g. '1440Min'  or '15Min'
    df_ohlc = pd.to_numeric(df_ticks['Price']).resample(resample_interval).ohlc()
    df_volume = pd.to_numeric(df_ticks['Volume']).resample(resample_interval).sum()   #sum all ticks to get volume

    df_ohlcv = pd.concat([df_ohlc, df_volume], axis=1)
    df_ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]         
    print(df_ohlcv.tail())

    return df_ohlcv

''' export df to CSV '''
def export_df_csv(df_ticks, df_ohlcv):

    # -- export OHLC to csv
    path_csv = './data/'
    name_csv = 'ohlcv.csv'
    ohlcv_csv = path_csv + name_csv
    df_ohlcv.to_csv(ohlcv_csv)

    # -- export ticks to csv
    t_path_csv = './data/'
    t_name_csv = 'ticks.csv'
    ticks_csv = t_path_csv + t_name_csv
    df_ticks.to_csv(ticks_csv)

    return 

''' how long did it take? <-- to compute this range of dates & ticks '''
def how_long_did_it_take(time_start, l_ticks):
    import time

    print (f"\n\n--- How long did it take processing the selected range of dates?")
    print (f"from            --->  {time_start}")
    time_end = time.ctime()
    print (f"to              --->  {time_end}")
    print (f"ticks processed --->  {len(l_ticks)}")

    return 

''' !!! it can be run standalone or been called as a lib (main) '''
# --------- Main 

def main(crypto_pair = 'ETHUSDT', interval = '15', since = '', start_date = '', end_date = ''):
    
    import time
    time_start = time.ctime()           # to compute how long the process took 
    i_ticks = 0
    l_ticks = [] 

    print(f"\nticktocandle_lib <-- crypto_pair {crypto_pair}  interval {interval}   since *{since}*")

    ''' Date range to get ticks from kraken '''
    # standalone --> if start_date is empty we call the function
    if not start_date: since, now, end = Get_Dates_Range(interval)
    # lib --> if start_date is full we've been called from another script
    else: since, now, end = Process_Dates_Range(interval, since, start_date, end_date)

    # end --> includes the last candle indicate in the end limit (we added 1 interval to end above) 
    while now <= end:       
        print(f"\nprocessing timestamp --> {now}     readable --> {ctime(now)}")
        time.sleep(1.5)         

        ''' Get ticks from kraken '''
        kraken_json, since = Get_Ticks_Kraken(crypto_pair, interval, since) #last since will become next since 

        ''' Process json into a list '''
        l_ticks, now = Process_Kraken_Json(kraken_json, crypto_pair, l_ticks, i_ticks, end)


    ''' Transform list into df '''
    df_ticks = l_price_into_df_price(l_ticks)

    ''' Compute ohlcv  '''
    df_ohlcv = Compute_OHLCV(df_ticks, interval)

    ''' export df to csv  '''
    export_df_csv(df_ticks, df_ohlcv)

    ''' How long did it take?  '''
    how_long_did_it_take(time_start, l_ticks)

    return df_ohlcv


#------------- 
if __name__ == "__main__":
    main()
    g = input("\n\n\n End Program .... Press any key : "); print (g)

