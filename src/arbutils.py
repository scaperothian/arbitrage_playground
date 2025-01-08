import os
import pickle

import numpy as np
import requests
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split


def etherscan_request(action, api_key, address, startblock=0, endblock=99999999, sort='desc'):

    """
    fetch transactions from address
    
    Return: 

    """
    base_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': action,
        'address': address,
        'startblock': startblock,
        'endblock': endblock,
        'sort': sort,
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        #st.error(f"API request failed with status code {response.status_code}")
        return None, f"API request failed with status code {response.status_code}"
    
    data = response.json()
    if data['status'] != '1':
        #st.error(f"API returned an error: {data['result']}")
        return None, f"API returned an error: {data['result']}"
    
    df = pd.DataFrame(data['result'])
    
    expected_columns = ['hash', 'blockNumber', 'timeStamp', 'from', 'to', 'gas', 'gasPrice', 'gasUsed', 'cumulativeGasUsed', 'confirmations', 'tokenSymbol', 'value', 'tokenName']
    
    for col in expected_columns:
        if col not in df.columns:
            raise Exception(f"Expected column '{col}' is missing from the response")
    
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Set Transaction Value in Appropriate Format
    df['og_value'] = df['value'].copy()
    df['value'] = np.where(df['tokenDecimal']=='6', df['value']/1e6, df['value']/1e18)

    # Sort by timestamp in descending order and select the most recent 10,000 trades
    df['timeStamp'] = pd.to_numeric(df['timeStamp'])
    df = df.sort_values(by='timeStamp', ascending=False)
    
    # Get the current timestamp (in Unix time)
    current_timestamp = df['timeStamp'].iloc[0]  # Current time in seconds since epoch

    # Calculate the timestamp for 24 hours ago
    threshold_timestamp = current_timestamp - (12 * 60 * 60)  # 24 hours in seconds

    # Filter the DataFrame to include only rows within the last 24 hours
    df = df[df['timeStamp'] >= threshold_timestamp]

    # Sort by timestamp in descending order (most recent first)
    df = df.sort_values(by='timeStamp', ascending=False)

    consolidated_data = {}

    for index, row in df.iterrows():
        tx_hash = row['hash']
        
        if tx_hash not in consolidated_data:
            consolidated_data[tx_hash] = {
                'blockNumber': row['blockNumber'],
                'timeStamp': row['timeStamp'],
                'hash': tx_hash,
                'from': row['from'],
                'to': row['to'],
                'WETH_value': 0,
                'USDC_value': 0,
                'tokenName_WETH': '',
                'tokenName_USDC': '',
                'gas': row['gas'],
                'gasPrice': row['gasPrice'],
                'gasUsed': row['gasUsed'],
                'cumulativeGasUsed': row['cumulativeGasUsed'],
                'confirmations': row['confirmations']
            }
        
        if row['tokenSymbol'] == 'WETH':
            consolidated_data[tx_hash]['WETH_value'] = row['value']
            consolidated_data[tx_hash]['tokenName_WETH'] = row['tokenName']
        elif row['tokenSymbol'] == 'USDC':
            consolidated_data[tx_hash]['USDC_value'] = row['value']
            consolidated_data[tx_hash]['tokenName_USDC'] = row['tokenName']

    return pd.DataFrame.from_dict(consolidated_data, orient='index')

def merge_pool_data(p0,p1):
    #Format P0 and P0 variables of interest
    p0['time'] = p0['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p0['p0.weth_to_usd_ratio'] = p0['WETH_value']/p0['USDC_value']
    p0['gasPrice'] = p0['gasPrice'].astype(float)
    p0['gasUsed']= p0['gasUsed'].astype(float)
    p0['p0.gas_fees_usd'] = ((p0['gasPrice']/1e9)*(p0['gasUsed']/1e9)*(1/p0['p0.weth_to_usd_ratio']))
    p1['time'] = p1['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p1['p1.weth_to_usd_ratio'] = p1['WETH_value']/p1['USDC_value']
    p1['gasPrice'] = p1['gasPrice'].astype(float)
    p1['gasUsed']= p1['gasUsed'].astype(float)
    p1['p1.gas_fees_usd'] = ((p1['gasPrice']/1e9)*(p1['gasUsed']/1e9)*(1/p1['p1.weth_to_usd_ratio']))

    #Merge Pool data
    both_pools = pd.merge(p0[['time','timeStamp','blockNumber','p0.weth_to_usd_ratio','p0.gas_fees_usd']],
                          p1[['time','timeStamp','blockNumber','p1.weth_to_usd_ratio','p1.gas_fees_usd']],
                          on=['time','timeStamp','blockNumber'], how='outer'
                         ).sort_values(by='timeStamp')
    both_pools = both_pools.ffill().reset_index(drop=True)
    both_pools = both_pools.dropna()
    both_pools['percent_change'] = (both_pools['p0.weth_to_usd_ratio'] - both_pools['p1.weth_to_usd_ratio'])/both_pools[['p0.weth_to_usd_ratio','p1.weth_to_usd_ratio']].min(axis=1)
    both_pools['total_gas_fees_usd'] = both_pools['p0.gas_fees_usd'] + both_pools['p1.gas_fees_usd']
    
    # Replace inf with NaN
    both_pools.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values (which were originally inf)
    both_pools.dropna(inplace=True)
    return both_pools

def find_closest_timestamp(df, time_col, label_key, minutes):
    # Ensure 'time' column is in datetime format
    df.loc[:, time_col] = pd.to_datetime(df[time_col])

    # Create a shifted version of the DataFrame with the target times
    shifted_df = df.copy()
    shifted_df[time_col] = shifted_df[time_col] - pd.Timedelta(minutes=minutes)

    # Merge the original DataFrame with the shifted DataFrame on the closest timestamps
    result_df = pd.merge_asof(df.sort_values(by=time_col),
                              shifted_df.sort_values(by=time_col),
                              on=time_col,
                              direction='forward',
                              suffixes=('', '_label'))

    # Select the required columns and rename them
    result_df = result_df[[time_col,label_key,label_key+'_label']]

    return result_df

def LGBM_Preprocessing(both_pools, forecast_window_min=10, objective='train',test_split=0.2):
    """
    Creating evaluation data from the pool.  The model for LGBM is predicting percent_change
    across the two pools.  

    input arguments:
    - both_pools (dataframe): raw dataframe of merged pool pair.
    - forecast_window_min: minutes to shift labels into the future.
    - objective: train / test / inference objectives changes the output arguments.  
                 train:
                 objective makes train/test splits with labels.  
                 test: 
                 objective returns the labeled data without splitting
                 inference:   
                 takes the latest timestamps (in this case samples without labels) to use as input into a 
                 trained model.  This is useful when pulling data and wanting to evaluate
                 the data (use a test split of the data), then perform inference on the same 
                 dataset just using a portion that is garunteed to not be used for training.
    - test_split: the proportion of the dataset to include in the test split (opposed to training).
                  therefore, 0.0 would be all training and 1.0 would be all test.  ignored if the 
                  objective is inference.
    
    Return (inference): 
    - (dataframe): data frame preserves all columns as int_df. i.e. the most recent shift_minutes of data.  
   
    Return (test): 
    - X (dataframe with time index): model features
    - y (dataframe with time index): labels   
   
    Return (train): 
    - X_train, X_test, y_train, y_test (dataframe with time index): standard sklearn train/test splits
    """
    int_df = both_pools.copy()
    int_df = int_df[['time','percent_change']]
    int_df = find_closest_timestamp(int_df, 'time', 'percent_change', forecast_window_min)

    #int_df = shift_column_by_time(int_df, 'time', 'percent_change', forecast_window_min)
    num_lags = 2  # Number of lags to create
    for i in range(1, num_lags + 1):
        int_df[f'lag_{i}'] = int_df['percent_change'].shift(i)
    int_df['rolling_mean_8'] = int_df['percent_change'].rolling(window=8).mean()
    int_df = int_df[['time','percent_change_label', 'percent_change','rolling_mean_8','lag_1','lag_2']]

    # Create time index for the dataframe to preserve timestamps with splits.
    int_df.index = int_df.pop('time')
    int_df.index = pd.to_datetime(int_df.index)

    if objective == 'inference':
        int_df = int_df[int_df['percent_change_label'].isna()]
        
        #remove labels from data in place.
        int_df.pop('percent_change_label') 
        return int_df
    
    elif objective == 'test':
        int_df.dropna(inplace=True)

        y = int_df.pop('percent_change_label')
        X = int_df.copy()
        return X,y
            
    elif objective == 'train':
        int_df.dropna(inplace=True)

        # Create labels and training...
        y = int_df.pop('percent_change_label')
        X = int_df.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42,shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        print(f"LGBM_Preprocessing: Error - wrong objective specified ({objective}). Should be train, test or inference")
        return None

def XGB_preprocessing(both_pools, params, objective='train'):
    
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE_LIST = params['GAS_FEES_N_WINDOW_AVERAGE']
    NUM_LAGS = params['GAS_FEES_NUM_LAGS']
    OBJECTIVE = params['MODEL_OBJECTIVE']
    TEST_SPLIT = params['TEST_SPLIT']

    int_df = both_pools.select_dtypes(include=['datetime64[ns]','int64', 'float64'])
    int_df = int_df[['time','total_gas_fees_usd']]

    df_int = find_closest_timestamp(int_df, 'time', 'total_gas_fees_usd', FORECAST_WINDOW_MIN)
    df_int.index = df_int.pop('time')
    df_int.index = pd.to_datetime(df_int.index)
    
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['total_gas_fees_usd'].shift(i)
    
    for i in N_WINDOW_AVERAGE_LIST:
        int_df[f'rolling_mean_{i}'] = int_df['total_gas_fees_usd'].rolling(window=i).mean()

    if objective == 'inference':
        df_int = df_int[df_int['total_gas_fees_usd_label'].isna()]

        #remove labels from data for inference.
        df_int.pop('total_gas_fees_usd_label') 
        return df_int
    
    elif objective == 'test':
        df_int.dropna(inplace=True)

        y = df_int.pop('total_gas_fees_usd_label')
        X = df_int.copy()
        return X,y
            
    elif objective == 'train':
        int_df.dropna(inplace=True)

        # Create labels and training...
        y = df_int.pop('total_gas_fees_usd_label')
        X = df_int.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42,shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        print(f"XGB_Preprocessing: Error - wrong objective specified ({OBJECTIVE}). Should be train, test or inference")
        return None

def calculate_min_investment(df,pool0_txn_fee_col, pool1_txn_fee_col, gas_fee_col,percent_change_col,min_investment_col='min_amount_to_invest'):

    # Percent Change is defined as P0 - P1 / min(P0,P1), 
    # where P0 is price of token in pool 0, P1 is price of token in pool 1.
    #
    # if percent_change (or ΔP) is positive then P0 > P1 and 
    #            first transaction is on pool 1
    #            second transaction is on pool 0 and 
    # if percent_change (or ΔP) is negative then P1 > P0 and 
    #            first transaction is on pool 0 
    #            second transaction is on pool 1 
    # 
    # Minimum Investment is defined as: G / [ (1+|ΔP|) x (1-T1) - (1-T0) ]
    # where ΔP is real, 0 < T1 < 1, 0 < T0 < 1 
    #       T0 is transaction fees for first pool transaction, T1 is transaction fees for second pool transaction.
    #       Note: use the pool with the smallest price for the first pool transaction
    #
    # To calculate the minimum investment, you must generate two terms conditional on the 
    # pool that is greater, which can be determined based on the sign.  
    #         
    # The first term is (1+ΔP) x (1 - T1) 
    # The second term is (1 - T0) 
    #
    # The minimum investment (using the two terms in the denominator) can have different outcomes depending 
    # on the configuration: 
    #    T0 > T1, positive outcome regardless of ΔP
    #    T1 > T0, negative outcome if |ΔP| < (1-T0)/(1-T1) - 1
    #    T1 > T0, positive outcome if |ΔP| > (1-T0)/(1-T1) - 1
    def min_investment(row):
        result = row[gas_fee_col] / (
            (1 + abs(row[percent_change_col])) *
            (1 - row[pool1_txn_fee_col] if row[percent_change_col] < 0 else 1 - row[pool0_txn_fee_col]) -
            (1 - row[pool0_txn_fee_col] if row[percent_change_col] < 0 else 1 - row[pool1_txn_fee_col])
        )
        if not np.isfinite(result):  # Check for inf or NaN
            return np.nan
        
        return result

    df[min_investment_col] = df.apply(min_investment, axis=1)


    # if the value in the dataframe is negative or inf, set the minimum investment to NaN to indicate 
    # ΔP is not large enough to overcome transaction fees or other race conditions.
    df[min_investment_col] = np.where(df[min_investment_col] < 0, np.nan, df[min_investment_col])

    return df

def load_model(model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    base_model_path = os.path.join(models_dir, model_name)
    
    # Check for different possible file extensions
    possible_extensions = ['', '.h5', '.pkl', '.joblib']
    model_path = next((base_model_path + ext for ext in possible_extensions if os.path.exists(base_model_path + ext)), None)
    
    if model_path is None:
        print(f"Model file not found for: {model_name}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model {model_name} loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None