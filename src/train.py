import os
import re
import pickle
import logging

import numpy as np
import pandas as pd

import seaborn as sns

# To train the price prediction model...
import lightgbm as lgb
import xgboost as xgb

from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# local modules
import arbutils
import fetch

GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")

def find_pool_pairs(thegraph_api_key, location):
    """
    search in a directory with csv files with the naming convention pool_id_<address>.  Extract
    the address and then query for the metadata for the pool.  save the metadata for each file 
    and determine which ones are valid pairs (i.e. the token pairs).
    """
    pools = []

    for filename in [x for x in os.listdir(location) if x.find(f'.csv')!=-1]:

        #print(f"Reading: {filename}")
        pattern = r"pool_id_(.*?)_swap_final\.csv"
        match = re.search(pattern, filename)
        if match:
            address = match.group(1)
            #print(f"Found {address}")
            metadata = fetch.thegraph_request_pool_metadata(thegraph_api_key=thegraph_api_key, pool_address=address)
            pool = {
                'filename':f"{location}{filename}",
                'address':address,
                'feeTier':int(metadata['feeTier'])*1e-6,
                't0_symbol':metadata['token0']['symbol'],
                't1_symbol':metadata['token1']['symbol'],
                't0_decimal':metadata['token0']['decimals'],
                't1_decimal':metadata['token1']['decimals'],
            }
            pools.append(pool)
        else:
            #ignore this mysterious csv.
            print(f"Ignoring {filename}")
    
    #print(f"Found {len(pools)} pools.")

    pair_to_addresses = {}
    matching_addresses = []
    
    for pool in pools:
        # Create a pair (order doesn't matter, so we use a tuple and sort it)
        pair = tuple(sorted([pool['t0_symbol'], pool['t1_symbol']]))
        address = pool['address']
        
        if pair not in pair_to_addresses:
            pair_to_addresses[pair] = []
        
        # Add the current address to the list of addresses for this pair
        pair_to_addresses[pair].append(address)
    
    # For each token pair, generate all possible combinations of addresses
    for addresses in pair_to_addresses.values():
        if len(addresses) > 1:
            matching_addresses.extend(list(combinations(addresses, 2)))
    
    matching_pools = []
    for addr0,addr1 in matching_addresses:
        
        pool_pairs = {'pool0':dict, 'pool1':dict}
        for pool in pools:
            if addr0 == pool['address']:
                pool_pairs['pool0'] = pool
            elif addr1 == pool['address']:
                pool_pairs['pool1'] = pool
        matching_pools.append(pool_pairs)
            
    print(f"Found {len(matching_pools)} valid pool pairs.")

    return matching_pools


def import_data(location):
    """
    Function to extract transaction level files from Folder, iterates through all files looking for combination pool_id_XXXXXX
    """
    final_df = pd.DataFrame()
    
    for i in [x for x in os.listdir(location) if x.find(f'arbitrage_20240313_20240613_WETH_USDC')!=-1]:
        print(f"Reading: {i}")
        temp_df = pd.read_csv(f"{location}/{i}")
        try:
          temp_df['time']
          pass
        except:
          temp_df['DATE'] = temp_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).replace(hour=0, minute=0, second=0, microsecond=0))
        final_df = pd.concat([final_df,temp_df])
        #print('.',end='')
    
    final_df['time'] = pd.to_datetime(final_df['time'], format='ISO8601')
    final_df = final_df.sort_values(by='time', ascending=True)
    final_df = final_df.reset_index(drop=True)
    
    # Find the first row with NaNs...
    new_first_row = final_df['p0.eth_price_usd'].first_valid_index()
    final_df = final_df.iloc[new_first_row:]
    # remove NaNs from forward fill
    has_nans = final_df.isna().any().any()
    print("Are there any NaNs in the DataFrame?", has_nans)

    
    try:
        final_df['DATE'] = final_df['time'].apply(lambda x: datetime.datetime(int(x[:4]),int(x[5:7]),int(x[8:10])))
    except:
        pass
    
    try:
        return final_df.drop('Unnamed: 0',axis=1).reset_index(drop=True)
    except:
        return final_df.reset_index(drop=True)

def xgboost_gas_fees_train(X_train, X_test, y_train, y_test):
    """
    training for predicting gas fees for all pools.
    """

    xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 100)
    
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE':mse,
        'RMSE':np.sqrt(mse),
        'R Squared':r2
    }

    #return model and metrics
    return xgb_model, metrics

def lgbm_pct_change_train(X_train, X_test, y_train, y_test):
    """
    training for predicting percent change in price between pools.
    """
    metrics = {}

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.5,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    evals_result = {}
    
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.record_evaluation(evals_result)
            ]
    )
    
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'MSE':mse,
        'RMSE':np.sqrt(mse),
        'R Squared':r2
    }

    #return model and metrics
    return gbm, metrics

def configure_logs(log_file_name):
    # Configure logging
    # Remove all existing handlers (to reconfigure dynamically)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        filename=log_file_name,        # Log file name
        filemode='w',               # Append mode ('w' for overwrite, 'a' for append)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        level=logging.INFO          # Log level (INFO, DEBUG, WARNING, etc.)
    )

if __name__ == "__main__":

    # ################################
    # CONFIGURABLE PARAMETERS
    # ################################
    params = {
        'FORECAST_WINDOW_MIN':1,
        'TRAINING_DATA_PATH':"data/",
        'MODEL_PATH':"models/",
        # PCT_CHANGE model parameters (things that can be ablated using the same data)
        "PCT_CHANGE_MODEL_NAME":"LGBM",
        "PCT_CHANGE_NUM_LAGS":2,  # Number of lags to create
        "PCT_CHANGE_N_WINDOW_AVERAGE":[8], # rollling mean value
        "PCT_CHANGE_TEST_SPLIT":0.2,
        # GAS_FEES model parameters (things that can be ablated using the same data)
        "GAS_FEES_MODEL_NAME":"XGBoost",
        "GAS_FEES_NUM_LAGS":9,  # Number of lags to create
        "GAS_FEES_N_WINDOW_AVERAGE":[3,6], # rollling mean value
        "GAS_FEES_TEST_SPLIT":0.2
    }

    MODEL_PATH = params['MODEL_PATH']
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    TRAINING_DATA_PATH = params['TRAINING_DATA_PATH'] 
    N_WINDOW_AVERAGE = params['PCT_CHANGE_N_WINDOW_AVERAGE']
    NUM_LAGS = params['PCT_CHANGE_NUM_LAGS']
    PCT_CHANGE_MODEL_NAME = params['PCT_CHANGE_MODEL_NAME']
    GAS_FEES_MODEL_NAME = params['GAS_FEES_MODEL_NAME']
    
    pool_pairs_list = find_pool_pairs(GRAPH_API_KEY, TRAINING_DATA_PATH)
    for pool_pair in pool_pairs_list:
        print(f"Pair: {pool_pair['pool0']['address']}, {pool_pair['pool1']['address']}")

    if len(pool_pairs_list)>1:
        print("Cannot Support multiple pool pairs in training...yet!")
        exit

    print(pool_pairs_list)
    
    # Loading the files from the directory
    p0 = pd.read_csv(pool_pairs_list[0]['pool0']['filename'])
    p1 = pd.read_csv(pool_pairs_list[0]['pool1']['filename'])

    p0_fee_tier = pool_pairs_list[0]['pool0']['feeTier']
    p1_fee_tier = pool_pairs_list[0]['pool1']['feeTier']


    df = arbutils.merge_pool_data_v2(p0,p0_fee_tier,p1,p1_fee_tier)
    df['time'] = pd.to_datetime(df['time'])     
    
    # ################################
    # TRAINING PERCENT CHANGE MODEL
    # ################################   
    model_filename = f'{MODEL_PATH}percent_change_{FORECAST_WINDOW_MIN}min_forecast_{PCT_CHANGE_MODEL_NAME}.pkl'
    configure_logs(f'{model_filename}.perf')
    logging.info(f"{params}")

    # Training for Percent Change in Price.
    X_train, X_test, y_train, y_test = arbutils.LGBM_Preprocessing(df,params,objective='train')
    model, metrics = lgbm_pct_change_train(X_train, X_test, y_train, y_test)
    logging.info(f"Percent Change in Price RMSE:{metrics['RMSE']}")
    logging.info(f"Percent Change in Price R Squared:{metrics['R Squared']}")

    # Use this instead of gbm.save_model bc we are using pickle to load the model in our analysis.
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    # ################################
    # TRAINING GAS FEES MODEL
    # ################################ 
    model_filename = f'{MODEL_PATH}gas_fees_{FORECAST_WINDOW_MIN}min_forecast_{GAS_FEES_MODEL_NAME}.pkl'
    configure_logs(f'{model_filename}.perf')
    logging.info(f"{params}")
    # Training for Percent Change in Price.
    X_train, X_test, y_train, y_test = arbutils.XGB_preprocessing(df,params,objective='train')
    model, metrics = xgboost_gas_fees_train(X_train, X_test, y_train, y_test)
    logging.info(f"Gas Fees RMSE:{metrics['RMSE']}")
    logging.info(f"Gas Fees R Squared:{metrics['R Squared']}")
    
    # Use this instead of gbm.save_model bc we are using pickle to load the model in our analysis.
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    