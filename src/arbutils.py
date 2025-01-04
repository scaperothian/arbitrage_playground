import os
import pickle

import numpy as np
import requests
import pandas as pd
from datetime import datetime

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
    both_pools = pd.merge(p0[['time','timeStamp','p0.weth_to_usd_ratio','p0.gas_fees_usd']],
                          p1[['time','timeStamp','p1.weth_to_usd_ratio','p1.gas_fees_usd']],
                          on=['time','timeStamp'], how='outer'
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

def LGBM_Preprocessing(both_pools, forecast_window_min=10):
    """
    Creating evaluation data from the pool.  The model for LGBM is predicting percent_change
    across the two pools.  

    Return: 
    - int_df (nans dropped): all columns 'time','percent_change_label', 'percent_change','rolling_mean_8','lag_1','lag_2'
    - df_nan: data frame with all columns as above, but only keeping the rows where the percent_change_label is nan.  
                i.e. the first shift_minutes of data.
    - X_pct_test - only the input features for LGBM.
    - y_pct_test - only the labels column
    """
    int_df = both_pools.copy()
    int_df = int_df[['time','percent_change']]
    int_df = shift_column_by_time(int_df, 'time', 'percent_change', forecast_window_min)
    num_lags = 2  # Number of lags to create
    for i in range(1, num_lags + 1):
        int_df[f'lag_{i}'] = int_df['percent_change'].shift(i)
    int_df['rolling_mean_8'] = int_df['percent_change'].rolling(window=8).mean()
    int_df = int_df[['time','percent_change_label', 'percent_change','rolling_mean_8','lag_1','lag_2']]
    df_nan = int_df[int_df['percent_change_label'].isna()]
    
    int_df.dropna(inplace=True)
    
    X_pct_test = int_df[['percent_change','rolling_mean_8','lag_1','lag_2']]
    y_pct_test = int_df['percent_change_label']
    return int_df, df_nan, X_pct_test, y_pct_test

def XGB_preprocessing(both_pools, forecast_window_min=10):
    int_df = both_pools.select_dtypes(include=['datetime64[ns]','int64', 'float64'])
    int_df = int_df[['time','total_gas_fees_usd']]
    df_3M = shift_column_by_time(int_df, 'time', 'total_gas_fees_usd', forecast_window_min)
    df_3M.index = df_3M.pop('time')
    df_3M.index = pd.to_datetime(df_3M.index)

    num_lags = 9  # Number of lags to create
    for i in range(1, num_lags + 1):
        df_3M[f'lag_{i}'] = df_3M['total_gas_fees_usd'].shift(i)
    
    df_3M['rolling_mean_3'] = df_3M['total_gas_fees_usd'].rolling(window=3).mean()
    df_3M['rolling_mean_6'] = df_3M['total_gas_fees_usd'].rolling(window=6).mean()
    df_nan = df_3M[df_3M['total_gas_fees_usd_label'].isna()]
    
    df_3M.dropna(inplace=True)
    lag_features = [f'lag_{i}' for i in range(1, num_lags + 1)]
    X_gas_test = df_3M[['total_gas_fees_usd']+lag_features + ['rolling_mean_3', 'rolling_mean_6']]
    y_gas_test = df_3M['total_gas_fees_usd_label']
    
    df_nan = df_nan[['total_gas_fees_usd', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
       'lag_9', 'rolling_mean_3', 'rolling_mean_6']]

    return df_nan, X_gas_test, y_gas_test

def shift_column_by_time(df, time_col, value_col, shift_minutes):
    """
    The purpose of this method is to create a shifted set of columns
    that will act as labels downstream model.

    Returns: the same df with an additional column f"{value_col}_label"
    """
    # Ensure 'time_col' is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort the DataFrame by time
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Create an empty column for the shifted values
    df[f'{value_col}_label'] = None

    # Iterate over each row and find the appropriate value at least shift_minutes minutes later
    for i in range(len(df)):
        current_time = df.loc[i, time_col]
        future_time = current_time + pd.Timedelta(minutes=shift_minutes)
        
        # Find the first row where the time is greater than or equal to the future_time
        future_row = df[df[time_col] >= future_time]
        if not future_row.empty:
            df.at[i, f'{value_col}_label'] = future_row.iloc[0][value_col]
    
    return df

def calculate_min_investment_legacy(df,gas_fee_col,percent_change_col,min_investment_col='min_amount_to_invest'):
    """
    adds min_investment_col to a dataframe df.
    """
    df[min_investment_col] = df.apply(
        lambda row: row[gas_fee_col] /
                    (
                        (1 + abs(row[percent_change_col])) * (1 - 0.003 if row[percent_change_col] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if row[percent_change_col] < 0 else 1 - 0.003)
                    ),
        axis=1
    )

    return df


def calculate_min_investment(df,pool0_txn_fee_col, pool1_txn_fee_col, gas_fee_col,percent_change_col,min_investment_col='min_amount_to_invest'):
    """
    adds min_investment_col to a dataframe df.
    """
    # Assumption: 
    # Percent Change is defined as P0 - P1 / min(P0,P1), 
    # where P0 is price of token in pool 0, P1 is price of token in pool 1.
    #
    # Minimum Investment is defined as: G / [ (1+|ΔP|) x (1-T1) - (1-T0) ]
    # where ΔP is real, 0 < T1 < 1, 0 < T0 < 1 
    # and   T0 is transaction fees for Pool 0, T1 is transaction fees for Pool 1.

    # To calculate the minimum investment, you must generate two terms conditional on the 
    # pool that is greater, which can be determined based on the sign.  if percent_change 
    # is positive then P0 > P1 and transaction 1 is P1, if percent_change is negative then 
    # P1 > P0 and transaction 1 is P0.
    # 
    # The first term is (1+ΔP) x (1 - T1)
    # 
    # The second term is (1 - T0) 

    df[min_investment_col] = df.apply(
        lambda row: row[gas_fee_col] /
                    (
                        (1 + abs(row[percent_change_col])) * (1 - row[pool1_txn_fee_col] if row[percent_change_col] < 0 else 1 - row[pool0_txn_fee_col]) -
                        (1 - row[pool0_txn_fee_col] if row[percent_change_col] < 0 else 1 - row[pool1_txn_fee_col])
                    ),
        axis=1
    )

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