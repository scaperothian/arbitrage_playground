import unittest
import pandas as pd
import numpy as np

from src.arbutils import load_model, etherscan_request, merge_pool_data, LGBM_Preprocessing, XGB_preprocessing, calculate_min_investment

from sklearn.metrics import root_mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.")


# API inputs: TODO - put these in a config file...
api_key = "16FCD3FTVWC3KDK17WS5PTWRQX1E2WEYV2"
pool0_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
pool1_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

price_model_name = "LGBM_Percent_Change_v1"
gasfee_model_name = "XGB_Gas_Prices_v2"

class TestAppMethods(unittest.TestCase):
    
    def test_etherscan_request(self):
        #
        # Fetch the data and check the columns....
        # Note: this method assumes the pools are WETH/USDC pair.
        #
        df_results = etherscan_request('tokentx', api_key, address=pool0_address)
        valid_columns = ['blockNumber', 'timeStamp', 'hash', 'from', 'to', 'WETH_value',
       'USDC_value', 'tokenName_WETH', 'tokenName_USDC', 'gas', 'gasPrice',
       'gasUsed', 'cumulativeGasUsed', 'confirmations']
        
        actual_columns = list(df_results.columns)
        
        self.assertEqual(actual_columns, valid_columns)
    
    def test_load_model(self):
        # 
        # Check to make sure we can fetch the models
        #
        m0 = load_model(price_model_name)
        self.assertNotEqual(None, m0)

        m1 = load_model(gasfee_model_name)
        self.assertNotEqual(None, m1)

    def test_merge_pool_data(self):
        #
        # Test merging two pools of data for valid columns
        # Note: this assumes the pools are WETH/USDC pair.
        #
        p0 = etherscan_request('tokentx', api_key, address=pool0_address)
        p1 = etherscan_request('tokentx', api_key, address=pool1_address)

        valid_columns = ['time', 'timeStamp', 'p0.weth_to_usd_ratio', 'p0.gas_fees_usd',
       'p1.weth_to_usd_ratio', 'p1.gas_fees_usd', 'percent_change',
       'total_gas_fees_usd']

        df_results = merge_pool_data(p0,p1)
        actual_columns = list(df_results.columns)

        self.assertEqual(actual_columns, valid_columns)

    def test_model_lgbm_preprocessing(self):
        #
        #  Test model preprocessing for LGBM
        #

        # fetch data from etherscan.io.
        p0 = etherscan_request('tokentx', api_key, address=pool0_address)
        p1 = etherscan_request('tokentx', api_key, address=pool1_address)
        
        # merge data from both pools.
        both_pools = merge_pool_data(p0,p1)

        # LGBM Preprocessing
        lgbm_results = LGBM_Preprocessing(both_pools)

        # Check for 4 return objects.
        self.assertEqual(len(lgbm_results),4)

    def test_model_xgb_preprocessing(self):
        #
        #  Test model preprocessing for XGBoost Model
        #

        # fetch data from etherscan.io.
        p0 = etherscan_request('tokentx', api_key, address=pool0_address)
        p1 = etherscan_request('tokentx', api_key, address=pool1_address)
        
        # merge data from both pools.
        both_pools = merge_pool_data(p0,p1)

        # XGB Preprocessing
        xgb_results = XGB_preprocessing(both_pools)

        # Check for 3 return objects.
        self.assertEqual(len(xgb_results),3)
    
    def test_model_pricing_inference(self):
        #
        #  Test model preprocessing for XGBoost Model
        #

        # fetch data from etherscan.io.
        p0 = etherscan_request('tokentx', api_key, address=pool0_address)
        p1 = etherscan_request('tokentx', api_key, address=pool1_address)
        
        # merge data from both pools.
        both_pools = merge_pool_data(p0,p1)

        # LGBM Preprocessing
        _, _, X_pct_test, y_pct_test = LGBM_Preprocessing(both_pools)
        
        model = load_model(price_model_name)


        y_pct_pred = model.predict(X_pct_test)
        rmse = root_mean_squared_error(y_pct_test, y_pct_pred)
        r2 = r2_score(y_pct_test, y_pct_pred)
        
        print(f"test_model_pricing_inference: Root Mean Squared Error: {rmse:.4f}")
        print(f"test_model_pricing_inference: R² Score: {r2:.4f}")

        self.assertEqual(None,None)
    

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

        df = calculate_min_investment(pd.DataFrame(test_dict),
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
        print(f"T1 > T0, and percent_change ({0.321}) is greater than {(1-0.1)/(1-0.3)-1}")
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[0.321],
            POOL0_TXN_FEE_COL_NAME:[0.3],
            POOL1_TXN_FEE_COL_NAME:[0.1]
        }

        df = calculate_min_investment(pd.DataFrame(test_dict),
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
        print(f"T1 > T0, and percent_change ({0.0321}) is less than {(1-0.1)/(1-0.3)-1}")
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[0.0321],
            POOL0_TXN_FEE_COL_NAME:[0.3],
            POOL1_TXN_FEE_COL_NAME:[0.1]
        }

        df = calculate_min_investment(pd.DataFrame(test_dict),
                                        POOL0_TXN_FEE_COL_NAME,
                                        POOL1_TXN_FEE_COL_NAME,
                                        GAS_FEES_COL_NAME,
                                        PERCENT_CHANGE_COL_NAME,
                                        min_investment_col='min_amount_to_invest')
        
        self.assertAlmostEqual(df.iloc[0]['min_amount_to_invest'],np.nan,places=6)

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
        print(f"T1 > T0, and percent_change ({0.0321}) is equal to {(1-0.1)/(1-0.3)-1}")
        test_dict = {
            GAS_FEES_COL_NAME:[20],
            PERCENT_CHANGE_COL_NAME:[(1-0.1)/(1-0.3)-1],
            POOL0_TXN_FEE_COL_NAME:[0.3],
            POOL1_TXN_FEE_COL_NAME:[0.1]
        }

        df = calculate_min_investment(pd.DataFrame(test_dict),
                                        POOL0_TXN_FEE_COL_NAME,
                                        POOL1_TXN_FEE_COL_NAME,
                                        GAS_FEES_COL_NAME,
                                        PERCENT_CHANGE_COL_NAME,
                                        min_investment_col='min_amount_to_invest')
        
        self.assertAlmostEqual(df.iloc[0]['min_amount_to_invest'],np.nan,places=6)

if __name__ == "__main__":
    unittest.main()

