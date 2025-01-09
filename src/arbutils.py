import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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

def LGBM_Preprocessing(both_pools, params, objective='train'):
    """
    Creating evaluation data from the pool.  The model for LGBM is predicting percent_change
    across the two pools.  

    input arguments:
    - both_pools (dataframe): raw dataframe of merged pool pair.
    - params: model specific metadata
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
    
    Return (inference): 
    - (dataframe): data frame preserves all columns as int_df. i.e. the most recent shift_minutes of data.  
   
    Return (test): 
    - X (dataframe with time index): model features
    - y (dataframe with time index): labels   
   
    Return (train): 
    - X_train, X_test, y_train, y_test (dataframe with time index): standard sklearn train/test splits
    """
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE_LIST = params['PCT_CHANGE_N_WINDOW_AVERAGE']
    NUM_LAGS = params['PCT_CHANGE_NUM_LAGS']
    TEST_SPLIT = params['PCT_CHANGE_TEST_SPLIT']

    int_df = both_pools.copy()
    int_df = int_df[['time','percent_change']]
    int_df = find_closest_timestamp(int_df, 'time', 'percent_change', FORECAST_WINDOW_MIN)

    #int_df = shift_column_by_time(int_df, 'time', 'percent_change', forecast_window_min)
    num_lags = 2  # Number of lags to create
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['percent_change'].shift(i)

    for i in N_WINDOW_AVERAGE_LIST:
        int_df[f'rolling_mean_{i}'] = int_df['percent_change'].rolling(window=i).mean()

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42,shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        print(f"LGBM_Preprocessing: Error - wrong objective specified ({objective}). Should be train, test or inference")
        return None

def XGB_preprocessing(both_pools, params, objective='train'):
    
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE_LIST = params['GAS_FEES_N_WINDOW_AVERAGE']
    NUM_LAGS = params['GAS_FEES_NUM_LAGS']
    TEST_SPLIT = params['GAS_FEES_TEST_SPLIT']

    int_df = both_pools.copy()
    int_df = int_df[['time','total_gas_fees_usd']]

    int_df = find_closest_timestamp(int_df, 'time', 'total_gas_fees_usd', FORECAST_WINDOW_MIN)
    int_df.index = int_df.pop('time')
    int_df.index = pd.to_datetime(int_df.index)
    
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['total_gas_fees_usd'].shift(i)
    
    for i in N_WINDOW_AVERAGE_LIST:
        int_df[f'rolling_mean_{i}'] = int_df['total_gas_fees_usd'].rolling(window=i).mean()

    if objective == 'inference':
        int_df = int_df[int_df['total_gas_fees_usd_label'].isna()]

        #remove labels from data for inference.
        int_df.pop('total_gas_fees_usd_label') 
        return int_df
    
    elif objective == 'test':
        int_df.dropna(inplace=True)

        y = int_df.pop('total_gas_fees_usd_label')
        X = int_df.copy()
        return X,y
            
    elif objective == 'train':
        int_df.dropna(inplace=True)

        # Create labels and training...
        y = int_df.pop('total_gas_fees_usd_label')
        X = int_df.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42,shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        print(f"XGB_Preprocessing: Error - wrong objective specified ({objective}). Should be train, test or inference")
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
    #df[min_investment_col] = np.where(df[min_investment_col] < 0, np.nan, df[min_investment_col])

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