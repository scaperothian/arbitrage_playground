import pandas as pd
import os
import numpy as np
import seaborn as sns
import pickle

# To train the price prediction model...
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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
                              direction='backward',
                              suffixes=('', '_label'))

    # Select the required columns and rename them
    result_df = result_df[[time_col,label_key,label_key+'_label']]

    return result_df

def create_gas_fees_splits(df_input, params):
    """
    Input: Dataframe with time and gas fees values.

    Creating splits using configurable model parameters.
    
    Returns: X_train, X_test, y_train, y_test
    """
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE = params['GAS_FEES_N_WINDOW_AVERAGE']
    NUM_LAGS = params['GAS_FEES_NUM_LAGS']

    int_df = df_input[['time','total_gas_fees_usd']]
    int_df = find_closest_timestamp(int_df, 'time', 'total_gas_fees_usd', FORECAST_WINDOW_MIN)
    
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['total_gas_fees_usd'].shift(i)
    
    int_df[f'rolling_mean_{N_WINDOW_AVERAGE}'] = int_df['total_gas_fees_usd'].rolling(window=N_WINDOW_AVERAGE).mean()
    int_df[f'rolling_mean_{N_WINDOW_AVERAGE*2}'] = int_df['total_gas_fees_usd'].rolling(window=N_WINDOW_AVERAGE*2).mean()

    # prune excess rows from lagging operation
    max_prune = max(N_WINDOW_AVERAGE*2,NUM_LAGS)
    int_df = int_df.iloc[max_prune:]

    
    has_nans = int_df.isna().any().any()
    print("Are there any NaNs in the DataFrame?", has_nans)
    if has_nans: 
        print(f"Found: {int_df.isna().sum()}")
    
    # Create time index foer the dataframe.
    int_df.index = int_df.pop('time')
    int_df.index = pd.to_datetime(int_df.index)
    
    # Create labels and training...
    y = int_df.pop('total_gas_fees_usd_label')
    X = int_df.copy()

    print(int_df.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
    return X_train, X_test, y_train, y_test


def create_pct_change_splits(df_input, params):
    """
    Input: Dataframe with time and percent_change values.

    Creating splits using configurable model parameters.
    
    Returns: X_train, X_test, y_train, y_test
    """

    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE = params['PCT_CHANGE_N_WINDOW_AVERAGE']
    NUM_LAGS = params['PCT_CHANGE_NUM_LAGS']
    
    int_df = df_input[['time','percent_change']]
    int_df = find_closest_timestamp(int_df, 'time', 'percent_change', FORECAST_WINDOW_MIN)
    
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['percent_change'].shift(i)
    
    int_df[f'rolling_mean_{N_WINDOW_AVERAGE}'] = int_df['percent_change'].rolling(window=N_WINDOW_AVERAGE).mean()
    
    # prune excess rows from lagging operation
    max_prune = max(N_WINDOW_AVERAGE,NUM_LAGS)
    int_df = int_df.iloc[max_prune:]

    
    has_nans = int_df.isna().any().any()
    print("Are there any NaNs in the DataFrame?", has_nans)
    if has_nans: 
        print(f"Found: {int_df.isna().sum()}")
    
    # Create time index foer the dataframe.
    int_df.index = int_df.pop('time')
    int_df.index = pd.to_datetime(int_df.index)
    
    # Create labels and training...
    y = int_df.pop('percent_change_label')
    X = int_df.copy()

    print(int_df.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
    return X_train, X_test, y_train, y_test

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

if __name__ == "__main__":

    # ################################
    # CONFIGURABLE PARAMETERS
    # ################################
    params = {
        'FORECAST_WINDOW_MIN':10,
        'TRAINING_DATA_PATH':"../../arbitrage_3M/",
        'MODEL_PATH':"../models/",
        # PCT_CHANGE model parameters (things that can be ablated using the same data)
        "PCT_CHANGE_MODEL_NAME":"LGBM",
        "PCT_CHANGE_NUM_LAGS":2,  # Number of lags to create
        "PCT_CHANGE_N_WINDOW_AVERAGE":8, # rollling mean value
        # GAS_FEES model parameters (things that can be ablated using the same data)
        "GAS_FEES_MODEL_NAME":"XGBoost",
        "GAS_FEES_NUM_LAGS":9,  # Number of lags to create
        "GAS_FEES_N_WINDOW_AVERAGE":3 # rollling mean value
    }

    MODEL_PATH = params['MODEL_PATH']
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    TRAINING_DATA_PATH = params['TRAINING_DATA_PATH'] 
    N_WINDOW_AVERAGE = params['PCT_CHANGE_N_WINDOW_AVERAGE']
    NUM_LAGS = params['PCT_CHANGE_NUM_LAGS']
    PCT_CHANGE_MODEL_NAME = params['PCT_CHANGE_MODEL_NAME']
    GAS_FEES_MODEL_NAME = params['GAS_FEES_MODEL_NAME']
    
    # ################################
    # IMPORT TRAINING DATA
    # ################################
    df = import_data(TRAINING_DATA_PATH)

    # ################################
    # TRAINING PERCENT CHANGE MODEL
    # ################################   
    # Training for Percent Change in Price.
    X_train, X_test, y_train, y_test = create_pct_change_splits(df,params)
    model, metrics = lgbm_pct_change_train(X_train, X_test, y_train, y_test)
    print("Percent Change in Price RMSE:", metrics['RMSE'])
    print("Percent Change in Price R Squared:", metrics['R Squared'])

    # Use this instead of gbm.save_model bc we are using pickle to load the model in our analysis.
    with open(f'{MODEL_PATH}percent_change_{FORECAST_WINDOW_MIN}min_forecast_{PCT_CHANGE_MODEL_NAME}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # ################################
    # TRAINING GAS FEES MODEL
    # ################################ 
    # Training for Percent Change in Price.
    X_train, X_test, y_train, y_test = create_gas_fees_splits(df,params)
    model, metrics = xgboost_gas_fees_train(X_train, X_test, y_train, y_test)
    print("Gas Fees RMSE:", metrics['RMSE'])
    print("Gas Fees R Squared:", metrics['R Squared'])
    
    # Use this instead of gbm.save_model bc we are using pickle to load the model in our analysis.
    with open(f'{MODEL_PATH}gas_fees_{FORECAST_WINDOW_MIN}min_forecast_{GAS_FEES_MODEL_NAME}.pkl', 'wb') as f:
        pickle.dump(model, f)