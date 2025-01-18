import math
import os
import unittest
import pandas as pd
import numpy as np
import pytz
import shutil

from datetime import datetime

# local modules
import src.arbutils as arbutils
import src.fetch as fetch

from sklearn.metrics import root_mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.")

# API inputs
pool0_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
pool1_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

class TestAppMethods(unittest.TestCase):

    def test_thegraph_request_validation(self):
        GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

        # Creates datetime objects.  
        #                     YYYY  MM  DD  HH  MM  SS
        old_date = datetime(*(2025,  1, 14,  0,  0,  0), tzinfo=pytz.UTC)
        new_date = datetime(*(2025,  1, 14,  1,  0,  0), tzinfo=pytz.UTC)


        df0 = fetch.thegraph_request(GRAPH_API_KEY,
                                        ETHERSCAN_API_KEY, 
                                        pool_address=pool0_address,
                                        old_date=old_date,
                                        new_date=new_date,
                                        batch_size=2)
        
        df1 = fetch.thegraph_request(GRAPH_API_KEY,
                                        ETHERSCAN_API_KEY, 
                                        pool_address=pool0_address,
                                        old_date=old_date,
                                        new_date=new_date,
                                        batch_size=3)
        
        df2 = fetch.thegraph_request(GRAPH_API_KEY,
                                        ETHERSCAN_API_KEY, 
                                        pool_address=pool0_address,
                                        old_date=old_date,
                                        new_date=new_date,
                                        batch_size=4)
        print(df0.shape, df1.shape, df2.shape)
        
        self.assertEqual(df0.shape,df1.shape,df2.shape)

    def test_thegraph_request_inference(self):
        #
        # Fetch the data and check the columns....
        # Note: this method assumes the pools are WETH/USDC pair.
        #
        GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

        df_results = fetch.thegraph_request(GRAPH_API_KEY,
                                               ETHERSCAN_API_KEY, 
                                      pool_address=pool0_address)

        valid_columns = ['transactionHash', 'datetime', 'timeStamp', 'sqrtPriceX96',
                'blockNumber', 'gasPrice', 'gasUsed', 'tick', 'amount0', 'amount1',
                'liquidity']
        
        print(df_results.dtypes)

        # Expected column types
        expected_dtypes = {
            'transactionHash': 'object',
            'datetime': 'datetime64[ns, UTC]',
            'timeStamp': 'int64',
            'sqrtPriceX96': 'float64',
            'blockNumber': 'int32',
            'gasPrice': 'float64',
            'gasUsed': 'float64',
            'tick': 'float64',
            'amount0': 'float64',
            'amount1': 'float64',
            'liquidity': 'float64',
        }

        # Validate column data types
        for col, expected_dtype in expected_dtypes.items():
            with self.subTest(col=col):
                self.assertEqual(df_results[col].dtype, expected_dtype, f"Column {col} has incorrect dtype")
        


        actual_columns = list(df_results.columns)
        
        self.assertEqual(actual_columns, valid_columns)

    def test_thegraph_request_analysis(self):
        #
        # Fetch the data and check the columns....
        # Note: this method assumes the pools are WETH/USDC pair.
        #
        GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

        # Creates datetime objects.  
        #                     YYYY  MM  DD  HH  MM  SS
        old_date = datetime(*(2025,  1, 14,  0,  0,  0), tzinfo=pytz.UTC)
        new_date = datetime(*(2025,  1, 18,  0,  0,  0), tzinfo=pytz.UTC)

        df_results = fetch.thegraph_request(GRAPH_API_KEY,
                                        ETHERSCAN_API_KEY, 
                                        pool_address=pool0_address,
                                        old_date=old_date,
                                        new_date=new_date,
                                        batch_size=1000)

        valid_columns = ['transactionHash', 'datetime', 'timeStamp', 'sqrtPriceX96',
                'blockNumber', 'gasPrice', 'gasUsed', 'tick', 'amount0', 'amount1',
                'liquidity']
        
        print(df_results.dtypes)

        # Expected column types
        expected_dtypes = {
            'transactionHash': 'object',
            'datetime': 'datetime64[ns, UTC]',
            'timeStamp': 'int64',
            'sqrtPriceX96': 'float64',
            'blockNumber': 'int32',
            'gasPrice': 'float64',
            'gasUsed': 'float64',
            'tick': 'float64',
            'amount0': 'float64',
            'amount1': 'float64',
            'liquidity': 'float64',
        }

        # Validate column data types
        for col, expected_dtype in expected_dtypes.items():
            with self.subTest(col=col):
                self.assertEqual(df_results[col].dtype, expected_dtype, f"Column {col} has incorrect dtype")
        


        actual_columns = list(df_results.columns)
        
        self.assertEqual(actual_columns, valid_columns)

    def test_thegraph_request_training(self):
        #
        # Fetch the data and check the columns....
        # Note: this method assumes the pools are WETH/USDC pair.
        #
        GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

        # Creates datetime objects.  
        #                     YYYY  MM  DD  HH  MM  SS
        old_date = datetime(*(2025,  1, 14,  0,  0,  0), tzinfo=pytz.UTC)
        new_date = datetime(*(2025,  1, 14,  2,  0,  0), tzinfo=pytz.UTC)
        
        data_path = f'data/'
        checkpoint_file = 'checkpoint.json'

        # Delete the file if it exists
        if os.path.isfile(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                print(f"Deleted file: {checkpoint_file}")
            except Exception as e:
                print(f"Error deleting file: {checkpoint_file}. Error: {e}")

        # Delete the directory if it exists
        if os.path.isdir(data_path):
            try:
                shutil.rmtree(data_path)
                print(f"Deleted directory: {data_path}")
            except Exception as e:
                print(f"Error deleting directory: {data_path}. Error: {e}")

        df_results = fetch.thegraph_request(GRAPH_API_KEY, 
                                    ETHERSCAN_API_KEY,
                                    pool_address=pool0_address,
                                    new_date=new_date, 
                                    old_date=old_date, 
                                    data_path=data_path, 
                                    checkpoint_file=checkpoint_file)

        print("Cleaning Up.")
        # Delete the file
        if os.path.isfile(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                print(f"Deleted file: {checkpoint_file}")
            except Exception as e:
                print(f"Error deleting file: {checkpoint_file}. Error: {e}")

        # Delete the directory
        if os.path.isdir(data_path):
            try:
                shutil.rmtree(data_path)
                print(f"Deleted directory: {data_path}")
            except Exception as e:
                print(f"Error deleting directory: {data_path}. Error: {e}")

        self.assertEqual(pd.DataFrame, type(df_results))


    def test_etherscan_request_v2(self):
        #
        # Fetch the data and check the columns....
        # Note: this method assumes the pools are WETH/USDC pair.
        #
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
        df_results = fetch.etherscan_request_v2(ETHERSCAN_API_KEY, 
                                      pool_address=pool0_address)

        valid_columns = ['transactionHash', 'datetime', 'timeStamp', 'sqrtPriceX96',
                'blockNumber', 'gasPrice', 'gasUsed', 'tick', 'amount0', 'amount1',
                'liquidity']
        
        print(df_results.dtypes)

        # Expected column types
        expected_dtypes = {
            'transactionHash': 'object',
            'datetime': 'datetime64[ns, UTC]',
            'timeStamp': 'int64',
            'sqrtPriceX96': 'float64',
            'blockNumber': 'int32',
            'gasPrice': 'float64',
            'gasUsed': 'float64',
            'tick': 'float64',
            'amount0': 'float64',
            'amount1': 'float64',
            'liquidity': 'float64',
        }

        # Validate column data types
        for col, expected_dtype in expected_dtypes.items():
            with self.subTest(col=col):
                self.assertEqual(df_results[col].dtype, expected_dtype, f"Column {col} has incorrect dtype")
        
        actual_columns = list(df_results.columns)
        
        self.assertEqual(actual_columns, valid_columns)

    def test_etherscan_request(self):
        #
        # Fetch the data and check the columns....
        # Note: this method assumes the pools are WETH/USDC pair.
        #
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

        df_results = fetch.etherscan_request_tokentx(ETHERSCAN_API_KEY, pool0_address)
        valid_columns = ['blockNumber',
                'timeStamp',
                'transactionHash',
                'from',
                'to',
                'WETH_value',
                'USDC_value',
                'tokenName_WETH',
                'tokenName_USDC',
                'gas',
                'gasPrice',
                'gasUsed',
                'cumulativeGasUsed',
                'confirmations']
        
        
        actual_columns = list(df_results.columns)
        
        self.assertEqual(actual_columns, valid_columns)
    
    def test_load_model(self):

        price_model_name = "percent_change_1min_forecast_LGBM"
        gasfee_model_name = "gas_fees_1min_forecast_XGBoost"
        # 
        # Check to make sure we can fetch the models
        #
        m0 = arbutils.load_model(price_model_name)
        self.assertNotEqual(None, m0)

        m1 = arbutils.load_model(gasfee_model_name)
        self.assertNotEqual(None, m1)

    def test_merge_pool_data(self):
        #
        # Test merging two pools of data for valid columns
        # Note: this assumes the pools are WETH/USDC pair.
        #
        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data_p0 = {
            'timeStamp': np.arange(1672531200, 1672531200 + num_rows * 3600, 3600),         # Sequential UNIX timestamps
            'blockNumber': np.arange(200000, 200000 + num_rows),                            # Sequential block numbers          
            'WETH_value': np.random.uniform(0, 10000, size=num_rows),       
            'USDC_value': np.random.uniform(0, 20000, size=num_rows),       
            'gas': np.random.uniform(0, 100000, size=num_rows),         
            'gasPrice': np.random.uniform(0, 100000, size=num_rows),
            'gasUsed': np.random.uniform(0, 100000, size=num_rows),
        }

        # Create dummy data
        data_p1 = {
            'timeStamp': np.arange(1672531200, 1672531200 + num_rows * 3600, 3600),         # Sequential UNIX timestamps
            'blockNumber': np.arange(200000, 200000 + num_rows),                            # Sequential block numbers          
            'WETH_value': np.random.uniform(0, 10000, size=num_rows),       
            'USDC_value': np.random.uniform(0, 20000, size=num_rows),       
            'gas': np.random.uniform(0, 100000, size=num_rows),         
            'gasPrice': np.random.uniform(0, 100000, size=num_rows),
            'gasUsed': np.random.uniform(0, 100000, size=num_rows),
        }

        p0 = pd.DataFrame(data_p0)
        p1 = pd.DataFrame(data_p1)

        valid_output_columns = ['time', 'timeStamp', 'blockNumber', 'p0.weth_to_usd_ratio', 'p0.gas_fees_usd',
       'p1.weth_to_usd_ratio', 'p1.gas_fees_usd', 'percent_change',
       'total_gas_fees_usd']

        df_results = arbutils.merge_pool_data(p0,p1)
        actual_columns = list(df_results.columns)

        self.assertEqual(actual_columns, valid_output_columns)

    def test_merge_pool_data_v2(self):
        """
                valid_input_columns = ['transaction_hash', 'timestamp', 'sqrtPriceX96', 'tick',
                    'eth_price_usd', 'usdc_amount0', 'eth_amount1', 'liquidity',
                    'block_number', 'gas_price', 'gas_used', 'sender', 'recipient']
        """
        # Define the number of rows for the dummy DataFrame
        num_rows = 10
        pool0_tx_fee = 0.01
        pool1_tx_fee = 0.01

        # Create dummy data
        data_p0 = {
            'transactionHash': ["0x45"]*num_rows,                    
            'datetime': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='1ME'),  # Generate hourly timestamps
            'timeStamp': np.random.uniform(-10000, 10000, size=num_rows),                            
            'sqrtPriceX96': np.random.uniform(-10000, 10000, size=num_rows),                            
            'blockNumber': np.arange(200000, 200000 + num_rows),                            # Sequential block numbers
            'gasPrice': np.random.uniform(0, 100000, size=num_rows),        
            'gasUsed': np.random.uniform(0, 100000, size=num_rows),        
            'tick': np.random.uniform(-10000, 10000, size=num_rows),                            
            'amount0': np.random.uniform(0, 20000, size=num_rows),       
            'amount1': np.random.uniform(0, 20000, size=num_rows),       
            'liquidity': np.random.uniform(0, 20000, size=num_rows),       
        }

        # Create dummy data
        data_p1 = {
            'transactionHash': ["0x45"]*num_rows,                    
            'datetime': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='1ME'),  # Generate hourly timestamps
            'timeStamp': np.random.uniform(-10000, 10000, size=num_rows),                            
            'sqrtPriceX96': np.random.uniform(-10000, 10000, size=num_rows),                            
            'blockNumber': np.arange(200000, 200000 + num_rows),                            # Sequential block numbers
            'gasPrice': np.random.uniform(0, 100000, size=num_rows),        
            'gasUsed': np.random.uniform(0, 100000, size=num_rows),        
            'tick': np.random.uniform(-10000, 10000, size=num_rows),                            
            'amount0': np.random.uniform(0, 20000, size=num_rows),       
            'amount1': np.random.uniform(0, 20000, size=num_rows),       
            'liquidity': np.random.uniform(0, 20000, size=num_rows),       
        }

        p0 = pd.DataFrame(data_p0)
        p1 = pd.DataFrame(data_p1)

        valid_output_columns = ['time', 'timestamp', 'blockNumber', 'p1.transaction_time',
                        'p1.transaction_epoch_time', 'p1.t0_amount', 'p1.t1_amount',
                        'p1.t0_token', 'p1.t1_token', 'p1.tick', 'p1.sqrtPriceX96',
                        'p1.gasUsed', 'p1.gasPrice', 'p1.transaction_id', 'p1.transaction_type',
                        'p1.transaction_rate', 'p1.eth_price_usd', 'p1.transaction_fees_usd',
                        'p1.gas_fees_usd', 'p1.total_fees_usd', 'p0.transaction_time',
                        'p0.transaction_epoch_time', 'p0.t0_amount', 'p0.t1_amount',
                        'p0.t0_token', 'p0.t1_token', 'p0.tick', 'p0.sqrtPriceX96',
                        'p0.gasUsed', 'p0.gasPrice', 'p0.transaction_id', 'p0.transaction_type',
                        'p0.transaction_rate', 'p0.eth_price_usd', 'p0.transaction_fees_usd',
                        'p0.gas_fees_usd', 'p0.total_fees_usd', 'percent_change',
                        'total_gas_fees_usd', 'total_transaction_rate',
                        'total_transaction_fees_used', 'total_fees_usd', 'swap_go_nogo']

        df_results = arbutils.merge_pool_data_v2(p0,pool0_tx_fee,p1,pool1_tx_fee)

        actual_columns = list(df_results.columns)

        self.assertEqual(actual_columns, valid_output_columns)

    def test_merge_pool_data_v3(self):
            
        GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
        ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
        
        # Define the number of rows for the dummy DataFrame
        num_rows = 10
        pool0_tx_fee = 0.01
        pool1_tx_fee = 0.01

        # Create dummy data
        data_p0 = {
            'datetime': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='1ME'),  # Generate hourly timestamps
            'timeStamp': np.random.uniform(-10000, 10000, size=num_rows),                            
            'blockNumber': np.arange(200000, 200000 + num_rows),                            # Sequential block numbers
            'gasPrice': np.random.uniform(0, 100000, size=num_rows),        
            'transactionHash': ["0x45"]*num_rows,                    
            'sqrtPriceX96': np.random.uniform(-10000, 10000, size=num_rows),                            
            'gasUsed': np.random.uniform(0, 100000, size=num_rows),        
            'tick': np.random.uniform(-10000, 10000, size=num_rows),                            
            'amount0': np.random.uniform(0, 20000, size=num_rows),       
            'amount1': np.random.uniform(0, 20000, size=num_rows),       
            'liquidity': np.random.uniform(0, 20000, size=num_rows),       
        }

        # Create dummy data
        data_p1 = {
            'datetime': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='1ME'),  # Generate hourly timestamps
            'timeStamp': np.random.uniform(0, 10000, size=num_rows),                            
            'blockNumber': np.arange(200000, 200000 + num_rows),                            # Sequential block numbers
            'gasPrice': np.random.uniform(0, 100000, size=num_rows),        
            'transactionHash': ["0x43"]*num_rows,                    
            'sqrtPriceX96': np.random.uniform(-10000, 10000, size=num_rows),                            
            'gasUsed': np.random.uniform(0, 100000, size=num_rows),        
            'tick': np.random.uniform(-10000, 10000, size=num_rows),                            
            'amount0': np.random.uniform(0, 20000, size=num_rows),       
            'amount1': np.random.uniform(0, 20000, size=num_rows),       
            'liquidity': np.random.uniform(0, 20000, size=num_rows),       
        }

        p0 = pd.DataFrame(data_p0)
        p1 = pd.DataFrame(data_p1)

        pool0_metadata = fetch.thegraph_request_pool_metadata(GRAPH_API_KEY, pool0_address)
        pool1_metadata = fetch.thegraph_request_pool_metadata(GRAPH_API_KEY, pool1_address)

        df = arbutils.merge_pool_data_v3(p0,pool0_metadata,p1,pool1_metadata)
        print(df.columns)
        print(df.dtypes)
        print(df.head(10))

    def test_model_lgbm_preprocessing_inference(self):
        #
        #  Test model preprocessing for LGBM
        #

        # ################################
        # CONFIGURABLE MODEL PARAMETERS
        # ################################
        model_params = {
            'FORECAST_WINDOW_MIN':1,
            'TRAINING_DATA_PATH':"../../arbitrage_3M/",
            'MODEL_PATH':"../models/",
            # PCT_CHANGE model parameters (things that can be ablated using the same data)
            "PCT_CHANGE_MODEL_NAME":"LGBM",
            "PCT_CHANGE_NUM_LAGS":2,  # Number of lags to create
            "PCT_CHANGE_N_WINDOW_AVERAGE":[8], # rollling mean value
            "PCT_CHANGE_TEST_SPLIT":0.2,
        }

        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'time': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='1ME'),  # Generate hourly timestamps
            'percent_change': np.random.uniform(-0.1, 0.1, size=num_rows),                  # Random percent changes between -10% and +10%
            'total_gas_fees_usd': np.random.uniform(10, 30, size=num_rows),                 # Random total gas fees
        }

        # Create the DataFrame
        merged_pool_data_df = pd.DataFrame(data)


        # LGBM Preprocessing
        lgbm_results = arbutils.LGBM_Preprocessing(merged_pool_data_df, model_params, objective='inference')

        # Check for 4 return objects.
        self.assertEqual(len(lgbm_results),1)

    def test_model_lgbm_preprocessing_test(self):
        #
        #  Test model preprocessing for LGBM
        #

        # ################################
        # CONFIGURABLE MODEL PARAMETERS
        # ################################
        model_params = {
            'FORECAST_WINDOW_MIN':1,
            'TRAINING_DATA_PATH':"../../arbitrage_3M/",
            'MODEL_PATH':"../models/",
            # PCT_CHANGE model parameters (things that can be ablated using the same data)
            "PCT_CHANGE_MODEL_NAME":"LGBM",
            "PCT_CHANGE_NUM_LAGS":2,  # Number of lags to create
            "PCT_CHANGE_N_WINDOW_AVERAGE":[8], # rollling mean value
            "PCT_CHANGE_TEST_SPLIT":0.2,
        }


        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'time': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='ME'),  # Generate hourly timestamps
            'percent_change': np.random.uniform(-0.1, 0.1, size=num_rows),                  # Random percent changes between -10% and +10%
            'total_gas_fees_usd': np.random.uniform(10, 30, size=num_rows),                 # Random total gas fees
        }

        # Create the DataFrame
        merged_pool_data_df = pd.DataFrame(data)


        # LGBM Preprocessing
        lgbm_results = arbutils.LGBM_Preprocessing(merged_pool_data_df, model_params, objective='test')

        # Check for 4 return objects.
        self.assertEqual(len(lgbm_results),2)


    def test_model_lgbm_preprocessing_train(self):
        #
        #  Test model preprocessing for LGBM
        #

        # ################################
        # CONFIGURABLE MODEL PARAMETERS
        # ################################
        model_params = {
            'FORECAST_WINDOW_MIN':1,
            'TRAINING_DATA_PATH':"../../arbitrage_3M/",
            'MODEL_PATH':"../models/",
            # PCT_CHANGE model parameters (things that can be ablated using the same data)
            "PCT_CHANGE_MODEL_NAME":"LGBM",
            "PCT_CHANGE_NUM_LAGS":2,  # Number of lags to create
            "PCT_CHANGE_N_WINDOW_AVERAGE":[8], # rollling mean value
            "PCT_CHANGE_TEST_SPLIT":0.2,
        }

        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'time': pd.date_range(start='2029-01-01 00:00:00', periods=num_rows, freq='ME'),  # Generate hourly timestamps
            'percent_change': np.random.uniform(-0.1, 0.1, size=num_rows),                  # Random percent changes between -10% and +10%
            'total_gas_fees_usd': np.random.uniform(10, 30, size=num_rows),                 # Random total gas fees
        }

        # Create the DataFrame
        merged_pool_data_df = pd.DataFrame(data)


        # LGBM Preprocessing
        lgbm_results = arbutils.LGBM_Preprocessing(merged_pool_data_df, model_params, objective='train')

        # Check for 4 return objects.
        self.assertEqual(len(lgbm_results),4)

    def test_model_xgb_preprocessing_train(self):
        #
        #  Test model preprocessing for XGBoost Model
        #
        # ################################
        # CONFIGURABLE PARAMETERS
        # ################################
        model_params = {
            'FORECAST_WINDOW_MIN':1,
            'TRAINING_DATA_PATH':"../../arbitrage_3M/",
            'MODEL_PATH':"../models/",
            # GAS_FEES model parameters (things that can be ablated using the same data)
            "GAS_FEES_MODEL_NAME":"XGBoost",
            "GAS_FEES_NUM_LAGS":9,  # Number of lags to create
            "GAS_FEES_N_WINDOW_AVERAGE":[3,6], # rollling mean value
            "GAS_FEES_TEST_SPLIT":0.2
        }

        # Define the number of rows for the dummy DataFrame
        num_rows = 20

        # Create dummy data
        data = {
            'time': pd.date_range(start='2025-01-01 00:00:00', periods=num_rows, freq='ME'),  # Generate hourly timestamps
            'timeStamp': np.arange(1672531200, 1672531200 + num_rows * 3600, 3600),         # Sequential UNIX timestamps
            'blockNumber': np.arange(100000, 100000 + num_rows),                            # Sequential block numbers
            'p0.weth_to_usd_ratio': np.random.uniform(1900, 2100, size=num_rows),           # Random WETH to USD ratios
            'p0.gas_fees_usd': np.random.uniform(5, 15, size=num_rows),                     # Random gas fees in USD for p0
            'p1.weth_to_usd_ratio': np.random.uniform(1900, 2100, size=num_rows),           # Random WETH to USD ratios for p1
            'p1.gas_fees_usd': np.random.uniform(5, 15, size=num_rows),                     # Random gas fees in USD for p1
            'percent_change': np.random.uniform(-0.1, 0.1, size=num_rows),                  # Random percent changes between -10% and +10%
            'total_gas_fees_usd': np.random.uniform(10, 30, size=num_rows),                 # Random total gas fees
        }

        # Create the DataFrame
        merged_pool_data_df = pd.DataFrame(data)

        # XGB Preprocessing
        xgb_results = arbutils.XGB_preprocessing(merged_pool_data_df, model_params, objective='train')

        # For training, there are four outputs.
        self.assertEqual(len(xgb_results),4)

    def test_model_xgb_preprocessing_test(self):
        #
        #  Test model preprocessing for XGBoost Model
        #
        # ################################
        # CONFIGURABLE PARAMETERS
        # ################################
        model_params = {
            'FORECAST_WINDOW_MIN':1,
            'TRAINING_DATA_PATH':"../../arbitrage_3M/",
            'MODEL_PATH':"../models/",
            # GAS_FEES model parameters (things that can be ablated using the same data)
            "GAS_FEES_MODEL_NAME":"XGBoost",
            "GAS_FEES_NUM_LAGS":9,  # Number of lags to create
            "GAS_FEES_N_WINDOW_AVERAGE":[3,6], # rollling mean value
            "GAS_FEES_TEST_SPLIT":0.2
        }

        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'time': pd.date_range(start='2025-01-01 00:00:00', periods=num_rows, freq='ME'),  # Generate hourly timestamps
            'percent_change': np.random.uniform(-0.1, 0.1, size=num_rows),                  # Random percent changes between -10% and +10%
            'total_gas_fees_usd': np.random.uniform(10, 30, size=num_rows),                 # Random total gas fees
        }

        # Create the DataFrame
        merged_pool_data_df = pd.DataFrame(data)

        # XGB Preprocessing
        xgb_results = arbutils.XGB_preprocessing(merged_pool_data_df, model_params, objective='test')

        # For training, there are four outputs.
        self.assertEqual(len(xgb_results),2)

    def test_model_xgb_preprocessing_inference(self):
        #
        #  Test model preprocessing for XGBoost Model
        #
        # ################################
        # CONFIGURABLE PARAMETERS
        # ################################
        model_params = {
            'FORECAST_WINDOW_MIN':1,
            'TRAINING_DATA_PATH':"../../arbitrage_3M/",
            'MODEL_PATH':"../models/",
            # GAS_FEES model parameters (things that can be ablated using the same data)
            "GAS_FEES_MODEL_NAME":"XGBoost",
            "GAS_FEES_NUM_LAGS":9,  # Number of lags to create
            "GAS_FEES_N_WINDOW_AVERAGE":[3,6], # rollling mean value
            "GAS_FEES_TEST_SPLIT":0.2
        }

        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'time': pd.date_range(start='2025-01-01 00:00:00', periods=num_rows, freq='ME'),  # Generate hourly timestamps
            'percent_change': np.random.uniform(-0.1, 0.1, size=num_rows),                  # Random percent changes between -10% and +10%
            'total_gas_fees_usd': np.random.uniform(10, 30, size=num_rows),                 # Random total gas fees
        }

        # Create the DataFrame
        merged_pool_data_df = pd.DataFrame(data)

        # XGB Preprocessing
        xgb_results = arbutils.XGB_preprocessing(merged_pool_data_df, model_params, objective='inference')

        # For training, there are four outputs.
        self.assertEqual(len(xgb_results),1)

    def test_model_pricing_inference(self):
        #
        #  Test model preprocessing for XGBoost Model
        #

        price_model_name = "percent_change_1min_forecast_LGBM"

        # Columns required for inference: 
        # percent_change    
        # rolling_mean_8    
        # lag_1             
        # lag_2            
        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'percent_change': np.random.uniform(-10, 10.0, size=num_rows),           # Random WETH to USD ratios
            'rolling_mean_8': np.random.uniform(-10, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_1': np.random.uniform(-10, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_2': np.random.uniform(-10, 10.0, size=num_rows),           # Random WETH to USD ratios
        }
        x_pct_test = pd.DataFrame(data)

        model = arbutils.load_model(price_model_name)

        y_pct_pred = model.predict(x_pct_test)

        self.assertEqual(len(x_pct_test),len(y_pct_pred))

    def test_model_gas_fee_inference(self):
        #
        #  Test model preprocessing for XGBoost Model
        #

        gas_fees_model_name = "gas_fees_1min_forecast_XGBoost"

        # Columns required for inference: 
        #total_gas_fees_usd    .
        #lag_1                 .
        #lag_2                 .
        #lag_3                 .
        #lag_4                 .
        #lag_5                 .
        #lag_6                 .
        #lag_7                 .
        #lag_8                 .
        #lag_9                 .
        #rolling_mean_3        .
        #rolling_mean_6        .          
        # Define the number of rows for the dummy DataFrame
        num_rows = 10

        # Create dummy data
        data = {
            'total_gas_fees_usd': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_1': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_2': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_3': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_4': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_5': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_6': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_7': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_8': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'lag_9': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'rolling_mean_3': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios
            'rolling_mean_6': np.random.uniform(0, 10.0, size=num_rows),           # Random WETH to USD ratios

        }
        x_pct_test = pd.DataFrame(data)

        model = arbutils.load_model(gas_fees_model_name)

        y_pct_pred = model.predict(x_pct_test)

        self.assertEqual(len(x_pct_test),len(y_pct_pred))

    def test_min_investment_scenario_1(self):
        """
        T0 > T1, positive outcome

        |ΔP| > (1-T0)/(1-T1) - 1
        
        where T0 is the transaction fee on the first transaction
              T1 is the transaction fee on the second transaction

        Note: 
            percent_change is defined as (P0 - P1) / min(P0,P1), 

            if percent_change is positive then P0 > P1 and 
                first transaction is on pool 1
                second transaction is on pool 0 and 
            if percent_change is negative then P1 > P0 and 
                first transaction is on pool 0 
                second transaction is on pool 1 
        
        where P0 is token price in pool 0, P1 is token price in pool 1
        """
        GAS_FEES_COL_NAME = 'total_gas_fees'
        PERCENT_CHANGE_COL_NAME = 'percent_change'
        POOL0_TXN_FEE_COL_NAME = 'pool0_txn_fee'
        POOL1_TXN_FEE_COL_NAME = 'pool1_txn_fee'
        
        # positive percent_change, T0 is for Pool 1, T1 is for Pool 0
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[0.321],
            POOL0_TXN_FEE_COL_NAME:[0.1],
            POOL1_TXN_FEE_COL_NAME:[0.3]
        }

        df = arbutils.calculate_min_investment(pd.DataFrame(test_dict),
                                        POOL0_TXN_FEE_COL_NAME,
                                        POOL1_TXN_FEE_COL_NAME,
                                        GAS_FEES_COL_NAME,
                                        PERCENT_CHANGE_COL_NAME,
                                        min_investment_col='min_amount_to_invest')
        self.assertAlmostEqual(df.iloc[0]['min_amount_to_invest'],40.908161,places=6)

    def test_min_investment_scenario_2(self):
        """
        T1 > T0, positive outcome

        |ΔP| > (1-T0)/(1-T1) - 1
        
        where T0 is the transaction fee on the first transaction
              T1 is the transaction fee on the second transaction

        Note: 
            percent_change is defined as (P0 - P1) / min(P0,P1), 

            if percent_change is positive then P0 > P1 and 
                first transaction is on pool 1
                second transaction is on pool 0 and 
            if percent_change is negative then P1 > P0 and 
                first transaction is on pool 0 
                second transaction is on pool 1 
        
        where P0 is token price in pool 0, P1 is token price in pool 1
        """
        GAS_FEES_COL_NAME = 'total_gas_fees'
        PERCENT_CHANGE_COL_NAME = 'percent_change'
        POOL0_TXN_FEE_COL_NAME = 'pool0_txn_fee'
        POOL1_TXN_FEE_COL_NAME = 'pool1_txn_fee'
        
        # positive percent_change, T0 is for Pool 1, T1 is for Pool 0
        #print(f"T1 > T0, and percent_change ({0.321}) is greater than {(1-0.1)/(1-0.3)-1}")
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[0.321],
            POOL0_TXN_FEE_COL_NAME:[0.3],
            POOL1_TXN_FEE_COL_NAME:[0.1]
        }

        df = arbutils.calculate_min_investment(pd.DataFrame(test_dict),
                                        POOL0_TXN_FEE_COL_NAME,
                                        POOL1_TXN_FEE_COL_NAME,
                                        GAS_FEES_COL_NAME,
                                        PERCENT_CHANGE_COL_NAME,
                                        min_investment_col='min_amount_to_invest')
        
        self.assertAlmostEqual(df.iloc[0]['min_amount_to_invest'],809.716599,places=6)

    def test_min_investment_scenario_3(self):
        """
        T1 > T0, negative outcome

        |ΔP| < (1-T0)/(1-T1) - 1
        
        where T0 is the transaction fee on the first transaction
              T1 is the transaction fee on the second transaction

        Note: 
            percent_change is defined as (P0 - P1) / min(P0,P1), 

            if percent_change is positive then P0 > P1 and 
                first transaction is on pool 1
                second transaction is on pool 0 and 
            if percent_change is negative then P1 > P0 and 
                first transaction is on pool 0 
                second transaction is on pool 1 
        
        where P0 is token price in pool 0, P1 is token price in pool 1
        """
        GAS_FEES_COL_NAME = 'total_gas_fees'
        PERCENT_CHANGE_COL_NAME = 'percent_change'
        POOL0_TXN_FEE_COL_NAME = 'pool0_txn_fee'
        POOL1_TXN_FEE_COL_NAME = 'pool1_txn_fee'
        
        # positive percent_change, T0 is for Pool 1, T1 is for Pool 0
        #print(f"T1 > T0, and percent_change ({0.0321}) is less than {(1-0.1)/(1-0.3)-1}")
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[0.0321],
            POOL0_TXN_FEE_COL_NAME:[0.3],
            POOL1_TXN_FEE_COL_NAME:[0.1]
        }

        df = arbutils.calculate_min_investment(pd.DataFrame(test_dict),
                                        POOL0_TXN_FEE_COL_NAME,
                                        POOL1_TXN_FEE_COL_NAME,
                                        GAS_FEES_COL_NAME,
                                        PERCENT_CHANGE_COL_NAME,
                                        min_investment_col='min_amount_to_invest')

        self.assertAlmostEqual(df.iloc[0]['min_amount_to_invest'],-112.657016,places=6)

    def test_min_investment_scenario_4(self):
        """
        |ΔP| = (1-T0)/(1-T1) - 1, infinite outcome
        
        where T0 is the transaction fee on the first transaction
              T1 is the transaction fee on the second transaction

        Note: 
            percent_change is defined as (P0 - P1) / min(P0,P1), 

            if percent_change is positive then P0 > P1 and 
                first transaction is on pool 1
                second transaction is on pool 0 and 
            if percent_change is negative then P1 > P0 and 
                first transaction is on pool 0 
                second transaction is on pool 1 
        
        where P0 is token price in pool 0, P1 is token price in pool 1
        """
        GAS_FEES_COL_NAME = 'total_gas_fees'
        PERCENT_CHANGE_COL_NAME = 'percent_change'
        POOL0_TXN_FEE_COL_NAME = 'pool0_txn_fee'
        POOL1_TXN_FEE_COL_NAME = 'pool1_txn_fee'
        
        # positive percent_change, T0 is for Pool 1, T1 is for Pool 0
        #print(f"T1 > T0, and percent_change ({0.0321}) is equal to {(1-0.1)/(1-0.3)-1}")
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[(1-0.1)/(1-0.3)-1],
            POOL0_TXN_FEE_COL_NAME:[0.3],
            POOL1_TXN_FEE_COL_NAME:[0.1]
        }

        df = arbutils.calculate_min_investment(pd.DataFrame(test_dict),
                                        POOL0_TXN_FEE_COL_NAME,
                                        POOL1_TXN_FEE_COL_NAME,
                                        GAS_FEES_COL_NAME,
                                        PERCENT_CHANGE_COL_NAME,
                                        min_investment_col='min_amount_to_invest')
        self.assertTrue(math.isnan(df.iloc[0]['min_amount_to_invest']))

if __name__ == "__main__":
    unittest.main()

