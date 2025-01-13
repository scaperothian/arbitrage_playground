import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
import pytz

from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb

#import tensorflow as tf
import random
import time
import lightgbm as lgb

import seaborn as sns
sns.set_style('darkgrid')

import arbutils
import etherscanutils
import alchemyutils

# Fetch the API key from environment variables
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
#ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
#ALCHEMY_URL = f'https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}'


# ################################
# CONFIGURABLE PARAMETERS
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
    # GAS_FEES model parameters (things that can be ablated using the same data)
    "GAS_FEES_MODEL_NAME":"XGBoost",
    "GAS_FEES_NUM_LAGS":9,  # Number of lags to create
    "GAS_FEES_N_WINDOW_AVERAGE":[3,6], # rollling mean value
    "GAS_FEES_TEST_SPLIT":0.2
}

forecast_window_minutes = model_params['FORECAST_WINDOW_MIN']
pct_change_model_name = model_params['PCT_CHANGE_MODEL_NAME']
gas_fees_model_name = model_params['GAS_FEES_MODEL_NAME']


# fetch data from mainnet
@st.cache_data(ttl=60)
def fetch_data(api_key, address, method='etherscan_legacy'):
    """
    etherscan_legacy: legacy method (fast mode).  does not use sqrtPriceX96 (training data for models use this value).
                    percent changes are inflated.  much, much faster to return data for analysis and
                    inference because the fetch does not rely on decoding actual transaction data blocks.
    etherscan (preferred): uses sqrtPriceX96 value.  Note: today you cannot do 12 hour analysis on minimum
                    investment because of the time to fetch that much data.
    """
    if method == 'etherscan':
        results = arbutils.etherscan_request_v2(api_key, address)
    elif method == 'etherscan_legacy':
        results = arbutils.etherscan_request(api_key, address)
    elif method == 'alchemy':
        raise NotImplementedError("fetch_data: Alchemy method not implemented.")
    elif method == 'thegraph':
        raise NotImplementedError("fetch_data: The Graph method not implemented.")
    elif method == 'file':
        raise NotImplementedError("fetch_data: From file not implemented.")
    else:
        raise NotImplementedError(f"fetch_data: Unknown fetch method requested: {method}")
    
    if type(results)!=pd.DataFrame: 
        st.error(results)
    else:
        return results

# 
@st.cache_data
def percent_change_preprocessing(both_pools, model_parameters, objective='test'):
    return arbutils.LGBM_Preprocessing(both_pools, params=model_parameters, objective=objective)

# 
@st.cache_data
def gas_fee_preprocessing(both_pools, model_parameters, objective='test'):
    return arbutils.XGB_preprocessing(both_pools, params=model_parameters, objective=objective)

@st.cache_data
def merge_pool_data(p0,p1):
    return arbutils.merge_pool_data(p0,p1)
    
def calculate_profit(y_pct_test, y_pct_pred, y_gas_test, y_gas_pred, pool0_txn_fees,pool1_txn_fees):
    """
    similar to the minimum investment calculation.  using labels to evaluate the
    prediction.
    """

    df_percent_change = pd.DataFrame({
        'time': y_pct_test.index,
        'percent_change_label': y_pct_test.to_numpy(),
        'percent_change_prediction': y_pct_pred,
    })

    df_gas = pd.DataFrame({
        'time': y_gas_test.index,
        'total_gas_fees_usd_label':y_gas_test.to_numpy(),
        'gas_fees_prediction': y_gas_pred,
    })
    
    df = pd.merge(df_percent_change, df_gas, how = 'left', on = 'time')
    df['pool0_txn_fees'] = pool0_txn_fees
    df['pool1_txn_fees'] = pool1_txn_fees
    df = df.dropna()

    df = arbutils.calculate_min_investment(df, 
                                        pool0_txn_fee_col='pool0_txn_fees',
                                        pool1_txn_fee_col='pool1_txn_fees',
                                        gas_fee_col='gas_fees_prediction', 
                                        percent_change_col='percent_change_prediction', 
                                        min_investment_col='min_amount_to_invest_prediction')


    def profit_calculation(row):
        """
        Calculate profit for a given row.
        """
        try:
            # if the model guesses the sign on the percent change wrong, the result is
            # a loss in return.
            if np.sign(row['percent_change_label']) == np.sign(row['percent_change_prediction']):
                SIGN = 1.0
            else: 
                SIGN = -1.0

            # Calculate the profit.
            profit = (
                threshold * 
                (1 + SIGN * np.abs(row['percent_change_label'])) * 
                (1 - pool1_txn_fee if row['percent_change_prediction'] < 0 else 1 - pool0_txn_fee) -
                threshold * 
                (1 - pool0_txn_fee if row['percent_change_prediction'] < 0 else 1 - pool1_txn_fee) -
                row['total_gas_fees_usd_label']
            )
            return profit
        except Exception as e:
            # Handle any unexpected errors
            print(f"Error calculating profit for row: {row}, Error: {e}")
            return np.nan

    # Use apply with the standalone function
    df['Profit'] = df.apply(profit_calculation, axis=1)
    
    return df

def create_min_investment_breakout(last_sample, pred_sample, model_forecast_window_min, pool0_txn_fee, pool1_txn_fee):
    """
    last_sample (DataSeries): 
    pred_sample (DataSeries)
    """

    T_lt = f"{last_sample['time']}"
    block_num = f"{last_sample['blockNumber']}"
    block_num_pred = f"{int(last_sample['blockNumber']) + 5 * model_forecast_window_min}" # blocks created every 12s 
    T_pred = f"{pred_sample['time']+timedelta(minutes=1)}"
    A = f"{last_sample['p0.eth_price_usd']:.2f}"    # Pool 0 ETH / USDC
    B = f"{pool0_txn_fee:.5f}"                               # Pool 0 Transaction Fee
    C = f"{last_sample['p0.gas_fees_usd']:.2f}"           # Pool 0 Gas Fees
    D = f"{last_sample['p1.eth_price_usd']:.2f}"    # Pool 1 ETH / USDC
    E = f"{pool1_txn_fee:.5f}"                               # Pool 1 Transaction Fee
    F = f"{last_sample['p1.gas_fees_usd']:.2f}"           # Pool 1 Gas Fees
    G = f"{last_sample['percent_change']:.5f}"            # Percent Change in Transaction Price
    H = f"{last_sample['total_gas_fees_usd']:.2f}"       # Total Gas Fees
    J = f"{pred_sample['percent_change_prediction']:.5f}"
    K = f"{pred_sample['gas_fees_prediction']:.2f}"

    L = last_sample['total_gas_fees_usd'] / \
                (
                    (1 + abs(last_sample['percent_change'])) * (1 - pool1_txn_fee if last_sample['percent_change'] < 0 else 1 - pool0_txn_fee) -
                    (1 - pool0_txn_fee if last_sample['percent_change'] < 0 else 1 - pool1_txn_fee)
                )
    L = f"{L:.2f}"
    M = pred_sample['gas_fees_prediction'] / \
                (
                    (1 + abs(pred_sample['percent_change_prediction'])) * (1 - pool1_txn_fee if pred_sample['percent_change_prediction'] < 0 else 1 - pool0_txn_fee) -
                    (1 - pool0_txn_fee if pred_sample['percent_change_prediction'] < 0 else 1 - pool1_txn_fee)
                )
    M = f"{M:.2f}"
    
    table_dict = {
        'Last Transaction': [T_lt,block_num,A, B, C, D, E, F, G, H, L],
        'Predicted': [T_pred,block_num_pred,'', B, '', '', E, '', J, K, M],
        'Index': [
            'Timestamp',
            'Block Number',
            'Pool 1 Transaction',
            'Pool 1 Transaction Fee',
            'Pool 1 Gas Fee',
            'Pool 2 Transaction',
            'Pool 2 Transaction Fee',
            'Pool 2 Gas Fee',
            'Percent Change',
            'Total Gas Fees',
            'Minimum Investment'
        ]
    }

    # Convert the dictionary to a DataFrame
    table_df = pd.DataFrame(table_dict)
    table_df.set_index('Index', inplace=True)

    return table_df


def calculate_min_investment(y_pct_time, y_pct_pred,y_gas_time,y_gas_pred,pool0_txn_fees,pool1_txn_fees):

    df_percent_change = pd.DataFrame({
        'time': y_pct_time,
        'percent_change_prediction': y_pct_pred,
    })

    df_gas = pd.DataFrame({
        'time': y_gas_time,
        'gas_fees_prediction': y_gas_pred,
    })
    
    df = pd.merge(df_percent_change, df_gas, how = 'left', on = 'time')
    df['pool0_txn_fees'] = pool0_txn_fees
    df['pool1_txn_fees'] = pool1_txn_fees
    df = df.dropna()

    df = arbutils.calculate_min_investment(df, 
                                        pool0_txn_fee_col='pool0_txn_fees',
                                        pool1_txn_fee_col='pool1_txn_fees',
                                        gas_fee_col='gas_fees_prediction', 
                                        percent_change_col='percent_change_prediction', 
                                        min_investment_col='min_amount_to_invest_prediction')
    
    return df

@st.cache_resource
def load_model(model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    base_model_path = os.path.join(models_dir, model_name)
    
    # Check for different possible file extensions
    possible_extensions = ['', '.h5', '.pkl', '.joblib']
    model_path = next((base_model_path + ext for ext in possible_extensions if os.path.exists(base_model_path + ext)), None)
    
    if model_path is None:
        st.error(f"Model file not found for: {model_name}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model {model_name} loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


#
#  Start of Streamlit UI code...
#
st.set_page_config(page_title="Arbitrage Recommender", page_icon=None, layout="centered", initial_sidebar_state="collapsed")

#
# Sidebar
#
st.sidebar.header("API Configuration")
pool0_address = st.sidebar.text_input("Pool 0 Address", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640")
pool0_txn_fee = float(st.sidebar.selectbox(label="Pool 0 Transaction Fee (${T_0}$)",options=[0.01, 0.003,0.0005,0.0001],index=2)) #select 0.0005 by default.

pool1_address = st.sidebar.text_input("Pool 1 Address", "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8")
pool1_txn_fee = float(st.sidebar.selectbox(label="Pool 1 Transaction Fee (${T_1}$)",options=[0.01, 0.003,0.0005,0.0001],index=1)) #select 0.003 by default.

analysis_strategy = st.sidebar.selectbox(label="How to analyze data: ",options=["fast etherscan (legacy)", "etherscan v2 (recommended)"],index=1) 

st.sidebar.markdown(
    '[Back to Main Page (mydiamondhands)](https://mydiamondhands.io/)',
    unsafe_allow_html=True
)

#
# Main screen
#
st.title("Arbitrage Recommender")
st.write(
    "Use this app to experiment with different cross-liquidity pool arbitrage scenarios in WETH/USDC liquidity pools. Enter an Etherscan API key, your budget,  and click run to simulate performance."
)

####################################################
# Run Inference Upfront
####################################################
# Fetch and process data for both pools
p0 = fetch_data(ETHERSCAN_API_KEY, address=pool0_address, method='etherscan')
p1 = fetch_data(ETHERSCAN_API_KEY, address=pool1_address, method='etherscan')

if p0 is None or p1 is None:
    st.error("Failed to fetch data from Etherscan. Please check your API key and try again.")
    raise Exception

both_pools = arbutils.merge_pool_data_v2(p0, pool0_txn_fee, p1, pool1_txn_fee)

# percent_change Preprocessing for model prediction
X_pct_change_infer = percent_change_preprocessing(both_pools, model_params, objective='inference')

if X_pct_change_infer is None:
    st.error("Preprocessing for percent_change data failed. Cannot proceed with analysis.")
    raise Exception

# gas_fee Preprocessing for model prediction
X_gas_fee_infer = gas_fee_preprocessing(both_pools, model_params, objective='inference')

if X_gas_fee_infer is None:
    st.error("Preprocessing for gas fees data failed. Cannot proceed with analysis.")
    raise Exception

# Run models
with st.spinner("Running Percent Change model..."):
    model_name_str = f"percent_change_{forecast_window_minutes}min_forecast_{pct_change_model_name}"
    pct_change_model = load_model(model_name_str)    
    try: 
        # Predictions!
        y_pct_change_pred_latest = pct_change_model.predict(X_pct_change_infer)
    except Exception as e:
        err_msg = f"Error running percent change model ({model_name_str}): {str(e)}" 
        st.error(err_msg)
        raise Exception(err_msg)

with st.spinner("Running Gas model..."):
    gas_fee_model = load_model(f"gas_fees_{forecast_window_minutes}min_forecast_{gas_fees_model_name}")
    try: 
        # Predictions!
        y_gas_fee_pred_latest = gas_fee_model.predict(X_gas_fee_infer)
    except Exception as e:
        err_msg = f"Error running gas fees model (gas_fees_{forecast_window_minutes}min_forecast_{gas_fees_model_name}): {str(e)}" 
        st.error(err_msg)
        raise Exception(err_msg)

df_min = calculate_min_investment(X_pct_change_infer.index,
                                y_pct_change_pred_latest,
                                X_gas_fee_infer.index,
                                y_gas_fee_pred_latest,
                                pool0_txn_fee,
                                pool1_txn_fee)

# Get the current time with the desired timezone
timezone = pytz.timezone("UTC")  # Replace "UTC" with your desired timezone (e.g., "America/New_York")
now = datetime.now(timezone)

# Difference between current time and last timestamp
time_difference = now - df_min['time'].iloc[-1]

# Check if the difference is less than FORCAST_WINDOW_MIN minutes
is_less_than_x_minutes = time_difference < timedelta(minutes=forecast_window_minutes)
x_minutes = timedelta(minutes=forecast_window_minutes)
if is_less_than_x_minutes:
    if df_min['min_amount_to_invest_prediction'].iloc[-1] < 0:
        st.write(f'Arbitrage Opportunity is not expected {forecast_window_minutes} minute(s) from now.')
    elif  np.isnan(df_min['min_amount_to_invest_prediction'].iloc[-1]):
        print("minimum amount to invest should not be NaN unless the units are so small they induce infinity...")
        print(df_min.iloc[-1])
        st.write(f'Arbitrage Opportunity is not expected {forecast_window_minutes} minute(s) from now.')
    else:
        if df_min['percent_change_prediction'].iloc[-1] < 0:
            st.write(f'**BUY**: Pool 0 ({pool0_address})')
            st.write(f'**SELL**: Pool 1 ({pool1_address})')
            st.write(f'**Minimum amount to invest**: ${df_min["min_amount_to_invest_prediction"].iloc[-1]:.2f} at time: {df_min["time"].iloc[-1]+x_minutes}')
        else:
            st.write(f'**BUY**: Pool 1 ({pool1_address})')
            st.write(f'**SELL**: Pool 0 ({pool0_address})')
            st.write(f'**Minimum amount to invest**: ${df_min["min_amount_to_invest_prediction"].iloc[-1]:.2f} at time: {df_min["time"].iloc[-1]+x_minutes}')
    
        st.write(
                "*Disclaimer: The creators of this app are not licensed to provide, offer or recommend financial instruments in any way shape or form. This information and the information within the site should not be considered unique financial advice and if you consider utilizing the model, or investing in crypto you should first seek financial advice from a trained professional, to ensure that you fully understand the risk. Further, while modelling efforts have been undertaken in an effort to avoid risk through the application of the principles of arbitrage, the model has not been empirically tested, and should be approached with extreme caution, care and be utilized at one’s own risk (do not make trades to which you would be unable to fulfill, or would be in a detrimental financial position if it was to not complete as expected).*"
        )                                
else:
    st.write(f"Last Data point received from query was at {df_min['time'].iloc[-1]}\nData queried is greater than {forecast_window_minutes} minute(s) old, unable to provide minimum amount to invest")



"""
if st.button("Run Analysis"):
    with st.spinner("Fetching and processing data..."):
        #
        # fetch data, preprocess, load model, perform inference 
        #
        if analysis_strategy == "etherscan (recommended)":
            
        
        # Run percent_change model
        with st.spinner("Running Percent Change model..."):
            pct_change_model = load_model(f"percent_change_{forecast_window_minutes}min_forecast_LGBM")
            if pct_change_model is not None:
                try:
                    y_pct_pred = pct_change_model.predict(X_pct_test, num_iteration=pct_change_model.best_iteration)
                    
                    if y_pct_pred is None: 
                        st.error("Error Running percent_change model")

                    y_pct_rmse = root_mean_squared_error(y_pct_test, y_pct_pred)
                    y_pct_r2 = r2_score(y_pct_test, y_pct_pred)
                    
                except Exception as e:
                    st.error(f"Error running percent_change model: {str(e)}")
                    raise Exception
            else:
                st.error("Failed to load percent_change model. Skipping percent_change analysis.")
                raise Exception

        # Run gas_fee model
        with st.spinner("Running Gas model..."):
            gas_fee_model = load_model(f"gas_fees_{forecast_window_minutes}min_forecast_XGBoost")
            if gas_fee_model is not None:
                try:
                    y_gas_pred = gas_fee_model.predict(X_gas_test)
                    
                    if y_gas_pred is None: 
                        st.error("Error Running gas fees model")
                        raise Exception

                    y_gas_rmse = root_mean_squared_error(y_gas_test, y_gas_pred)
                    y_gas_r2 = r2_score(y_gas_test, y_gas_pred)


                except Exception as e:
                    st.error(f"Error running gas fees model: {str(e)}")
            else:
                st.error("Failed to load gas fees model. Skipping gas fees analysis.")
                raise Exception

        # Process final results
        df_final = calculate_profit(y_pct_test,
                                    y_pct_pred,
                                    y_gas_test,
                                    y_gas_pred,
                                    pool0_txn_fee,
                                    pool1_txn_fee)

        experiment_duration = df_final['time'].iloc[-1] - df_final['time'].iloc[0]
        avg_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].mean()
        median_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].median()

        #avg_profit = df_final[(df_final['min_amount_to_invest_prediction'] > 0) 
        #                    & (df_final['min_amount_to_invest_prediction'] < threshold)]['Profit'].mean()
        #med_profit = df_final[(df_final['min_amount_to_invest_prediction'] > 0) 
        #                    & (df_final['min_amount_to_invest_prediction'] < threshold)]['Profit'].median()

        number_of_simulated_swaps = df_final.shape[0]

        # ##########################################################################
        #
        #  Section 3: Recommended investment based on latest transactions.
        #
        # ##########################################################################
        st.subheader(f'Recommended Minimum Investment Prediction ({forecast_window_minutes} minute forecast)')

 


        # ##########################################################################
        #
        #  Section 1: Rollup stats from past transactions
        #
        # ##########################################################################
        st.subheader(f'Results from Previous {np.round(experiment_duration.total_seconds() / 3600):.0f} hour(s)')

        st.write(f"Number of Transactions: {number_of_simulated_swaps}")
        st.write(f"Average Recommended Minimum Investment: ${avg_positive_min_investment:.2f}")
        st.write(f"Median Recommended Minimum Investment: ${median_positive_min_investment:.2f}")
        
        df_valid_txns = df_final[(df_final['min_amount_to_invest_prediction'] > 0) 
                            & (df_final['min_amount_to_invest_prediction'] < threshold)]
        
        df_gain = df_valid_txns[(df_valid_txns['Profit'] > 0)]

        st.write(f"Percent of All Transactions with Detected Arbitrage Opportunites: {df_valid_txns.shape[0]/df_final.shape[0]*100:.1f}%")
        st.write(f"Percent of Transactions with Detected Arbitrage Opportunites that the models predict a Return: {df_gain.shape[0]/df_valid_txns.shape[0]*100:.1f}%")
        
        st.write("*Return / Profit is defined as the the hypothetical return from the actual percent_change and actual fees from past transactions 
                    using three new inputs: (1) the initial investment 'budget' provided by the user above, (2) the calculation for minimum investment 
                    that indicates if percent_change is large enough to perform arbitrage to overcome fees, (3) the decision by the model on which 
                    pool to use which impacts performance.*
                    ")

        if df_final[df_final['min_amount_to_invest_prediction']>0].shape[0] != 0:

            avg_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].mean()
            median_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].median()

            avg_profit = df_final[(df_final['min_amount_to_invest_prediction'] > 0) 
                                & (df_final['min_amount_to_invest_prediction'] < threshold)]['Profit'].mean()
            med_profit = df_final[(df_final['min_amount_to_invest_prediction'] > 0) 
                                & (df_final['min_amount_to_invest_prediction'] < threshold)]['Profit'].median()



            df_valid_txns = df_final[(df_final['min_amount_to_invest_prediction'] > 0)]
            
            df_gain = df_valid_txns[(df_valid_txns['Profit'] > 0)]

            st.write(f"Average Recommended Minimum Investment: ${avg_positive_min_investment:.2f}")
            st.write(f"Median Recommended Minimum Investment: ${median_positive_min_investment:.2f}")
            st.write(f"Percent of All Transactions with Detected Arbitrage Opportunites: {df_valid_txns.shape[0]/df_final.shape[0]*100:.1f}%")

            if df_valid_txns.shape[0] > 0:
                st.write(f"Percent of Transactions with Detected Arbitrage Opportunites that the models predict a Return: {df_gain.shape[0]/df_valid_txns.shape[0]*100:.1f}%")
            
            st.write("*Return / Profit is defined as the the hypothetical return from the actual percent_change and actual fees from past transactions 
                        using three new inputs: (1) the initial investment 'budget' provided by the user above, (2) the calculation for minimum investment 
                        that indicates if percent_change is large enough to perform arbitrage to overcome fees, (3) the decision by the model on which 
                        pool to use which impacts performance.*
                        ")

            #TODO: attempt to understand how really large price flucuations should be taken into account for 
            #      scenarios where there is losses (i.e. profit is largely negative - greater than 100000 as calculated).
            #      But you can't loose more money than you put in, so what's wrong with the calculus?
            #st.write(f"Average Return on Transactions with Detected Arbitrage Opportunites: ${avg_profit:.2f}")
            #st.write(f"Median Return on Transactions with Detected Arbitrage Opportunites: ${med_profit:.2f}")


            st.subheader(f"Recommended Minimum Investment in the last {np.round(experiment_duration.total_seconds() / 3600):.0f} hour(s).")
            
            # ##########################################################################
            #
            #  Section 2a: Scatter plot of Recommended Minimum Investment Amount over time
            #
            # ##########################################################################
            fig2, axs2 = plt.subplots(2, 1, figsize=(14, 10))

            axs2[0].scatter(df_final['time'], df_final['min_amount_to_invest_prediction'], marker='o')
            axs2[0].set_xlabel('time')
            axs2[0].set_ylabel('Recommended Minimum Amount to Invest')
            max_ylim = df_final['min_amount_to_invest_prediction'].max()

            # Add horizontal line at the threshold
            axs2[0].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
            # Add annotation to the line
            axs2[0].annotate(
                "Selected Budget",
                xy=(df_final['time'].min(), threshold),  # Far right of the plot
                xytext=(10, 5),  # Offset to position the text just above the line
                textcoords="offset points",
                fontsize=8,  # Small font size
                color='black',  # Same color as the line
                ha='left',  # Align text to the left
                va='bottom'  # Align text above the line
            )

            #axs[0].set_ylim(0, max_ylim * 1.2)
            axs2[0].set_yscale('log')
            axs2[0].set_title(f'Scatter Plot of Recommended Minimum Investment in last {experiment_duration.total_seconds() / 3600:.1f} hour(s)')

            # Format the x-axis to show HH:MM:SS
            axs2[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            axs2[0].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency
            #fig.autofmt_xdate()  # Rotates the labels to prevent overlap

            # ##########################################################################
            #
            #  Section 2b: Histogram Recommended Minimum Investment Amount bounded by
            #            "budget" threshold
            #
            # ##########################################################################
            df_final_fig2b = df_final[
                (df_final['min_amount_to_invest_prediction'] > 0) & 
                (df_final['min_amount_to_invest_prediction'] < threshold)
            ][['min_amount_to_invest_prediction']]

            # Calculate histogram data
            data = df_final_fig2b['min_amount_to_invest_prediction']
            hist_values, bin_edges = np.histogram(data, bins=50)

            # Plot the histogram
            axs2[1].hist(data, bins=50, alpha=0.7)
            axs2[1].set_xlabel('Recommended Minimum Amount to Invest')
            axs2[1].set_title(f'Distribution of Recommended Minimum Investment in last {experiment_duration.total_seconds() / 3600:.1f} hour(s)')
            axs2[1].set_xlim(0, threshold)                
            
            # Display the figure in Streamlit
            st.pyplot(fig2)
        else:
            st.write(f"There were no arbitrage opportunites in the last {np.round(experiment_duration.total_seconds() / 3600):.0f} hour(s)")




        # ##########################################################################
        #
        #  Section 4: Raw data for deeper dive.
        #
        # ##########################################################################

        st.subheader("Raw Results")
        st.write(df_final.sort_values(by='time',ascending=False))

        st.subheader("LGBM Price Model Results")
        st.write(f"Root Mean Squared Error: {y_pct_rmse:.4f}")
        st.write(f"R² Score: {y_pct_r2:.4f}")

        st.subheader("XGB Gas Fee Model Results")
        st.write(f"Root Mean Squared Error: {y_gas_rmse:.4f}")
        st.write(f"R² Score: {y_gas_r2:.4f}")


        # Price Plots
        st.subheader(f'Pool Price Data')
        fig3, axs3 = plt.subplots(2, 1, figsize=(14, 10))
        axs3[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs3[0].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency
        axs3[0].scatter(both_pools['time'],both_pools['p0.eth_price_usd'], marker='o')
        axs3[0].scatter(both_pools['time'],both_pools['p1.eth_price_usd'], marker='x')
        axs3[0].set_title('Token Price for Pool 0 / Pool 1')
        axs3[0].legend(['Pool 0','Pool 1'])
        axs3[0].set_yscale('log')
        axs3[0].set_xlabel('time')
        axs3[0].set_ylabel('USD per Eth (log scale)')


        both_pools_trunc = both_pools.iloc[-50:]
        axs3[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs3[1].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency
        axs3[1].scatter(both_pools_trunc['time'],both_pools_trunc['p0.eth_price_usd'], marker='o')
        axs3[1].scatter(both_pools_trunc['time'],both_pools_trunc['p1.eth_price_usd'], marker='x')
        axs3[1].set_title('Zoomed in Token Price for Pool 0 / Pool 1')
        axs3[1].legend(['Pool 0','Pool 1'])
        axs3[1].set_xlabel('time')
        axs3[1].set_ylabel('USD per Eth (log scale)')
        st.pyplot(fig3)


        # Gas Price Plots
        st.subheader(f'Gas Price Data')
        fig4, axs4 = plt.subplots(1, 1, figsize=(14, 10))
        axs4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs4.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency
        axs4.scatter(both_pools['time'],both_pools['p0.gas_fees_usd'], marker='o')
        axs4.scatter(both_pools['time'],both_pools['p1.gas_fees_usd'], marker='x')
        axs4.set_title('Gas Price for Pool 0 / Pool 1')
        axs4.legend(['Pool 0','Pool 1'])
        axs4.set_yscale('log')
        axs4.set_xlabel('time')
        axs4.set_ylabel('Gas prices in USD (log scale)')
        st.pyplot(fig4)

        
        # Build Minimum Investment Breakout Table
        st.subheader('Minimum Investment Calculation Breakout')
        last_sample_ds = both_pools.iloc[-1]
        last_sample_ds_pred = df_min.iloc[-1]

        table_df = create_min_investment_breakout(last_sample_ds, last_sample_ds_pred, forecast_window_minutes,pool0_txn_fee, pool1_txn_fee)

        st.table(table_df)
"""

