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
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb

#import tensorflow as tf
import random
import time
import lightgbm as lgb

import seaborn as sns
sns.set_style('darkgrid')

import arbutils

st.set_page_config(page_title="Arbitrage Playground", page_icon=None, layout="centered", initial_sidebar_state="collapsed")

FORECAST_WINDOW_MIN=1

st.title("Arbitrage Playground")
st.write(
    "Use this app to experiment with different cross-liquidity pool arbitrage scenarios in WETH/USDC liquidity pools. Enter an Etherscan API key, your budget,  and click run to simulate performance."
)


threshold = st.slider('Select Budget', min_value=1000, max_value=40000, value=10000, step=500)
st.write(f'Selected Budget: {threshold}')

# fetch data from Etherscan API
@st.cache_data(ttl=60)
def etherscan_request(action, api_key, address, startblock=0, endblock=99999999, sort='desc'):
    results = arbutils.etherscan_request(action, api_key, address, startblock, endblock, sort)
    if type(results)!=pd.DataFrame: 
        st.error(results[1])
    else:
        return results

# 
@st.cache_data
def LGBM_Preprocessing(both_pools, forecast_window_min=10):
    return arbutils.LGBM_Preprocessing(both_pools, forecast_window_min)

# 
@st.cache_data
def XGB_preprocessing(both_pools, forecast_window_min=10):
    return arbutils.XGB_preprocessing(both_pools, forecast_window_min)

@st.cache_data
def merge_pool_data(p0,p1):
    return arbutils.merge_pool_data(p0,p1)
    
def Final_results_processing(dates_test,y_test,test_predictions,y_gas_test,y_gas_pred):
    df_percent_change = pd.DataFrame({
        'time': dates_test,
        'percent_change_actual': y_test,
        'percent_change_prediction': test_predictions
    })
    
    df_gas = y_gas_test.to_frame()
    df_gas = df_gas.reset_index()
    df_gas['gas_fees_prediction'] = y_gas_pred
    
    df_final = pd.merge(df_percent_change, df_gas, how = 'left', on = 'time')
    df_final = df_final.dropna()

    df_final = arbutils.calculate_min_investment(df_final, 
                                        gas_fee_col='gas_fees_prediction', 
                                        percent_change_col='percent_change_prediction', 
                                        min_investment_col='min_amount_to_invest_prediction')
    
    df_final['Profit'] = df_final.apply(
        lambda row: (row['min_amount_to_invest_prediction'] * 
                     (1 + row['percent_change_actual']) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                     row['min_amount_to_invest_prediction'] * (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003) -
                     row['total_gas_fees_usd_label']),
        axis=1
    )
    
    return df_final



def final_min_amt_invest(dates_test, test_predictions, index, gas_predictions):
    df_percent_change = pd.DataFrame({
            'time': dates_test,
            'percent_change_prediction': test_predictions
        })
    df_gas = pd.DataFrame({
            'time': index,
            'gas_fees_prediction': gas_predictions
        })
    df_final = pd.merge(df_percent_change, df_gas, how = 'left', on = 'time')
    df_final = df_final.dropna()
    df_final['min_amount_to_invest_prediction'] = df_final.apply(
        lambda row: row['gas_fees_prediction'] /
                    (
                        (1 + abs(row['percent_change_prediction'])) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003)
                    ),
        axis=1
    )
    return df_final

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
        st.success(f"Model {model_name} loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Sidebar
st.sidebar.header("API Configuration")

# API key input
api_key = st.sidebar.text_input("Etherscan API Key", "16FCD3FTVWC3KDK17WS5PTWRQX1E2WEYV2")
pool0_address = st.sidebar.text_input("Pool 0 Address", "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8")
pool1_address = st.sidebar.text_input("Pool 1 Address", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640")


st.sidebar.markdown(
    '[Back to Main Page (mydiamondhands)](https://mydiamondhands.io/)',
    unsafe_allow_html=True
)

if st.button("Run Analysis"):
    with st.spinner("Fetching and processing data..."):
        # Fetch and process data for both pools
        p0 = etherscan_request('tokentx', api_key, address=pool0_address)
        p1 = etherscan_request('tokentx', api_key, address=pool1_address)
        
        if p0 is None or p1 is None:
            st.error("Failed to fetch data from Etherscan. Please check your API key and try again.")
        else:
            both_pools = merge_pool_data(p0, p1)

       
        
            # LGBM Preprocessing
            LGBM_org_data, df_final_LGBM_X_test, X_pct_test, y_pct_test = LGBM_Preprocessing(both_pools, FORECAST_WINDOW_MIN)
        
            # XGB Preprocessing
            df_final_XGB_X_test, X_gas_test, y_gas_test = XGB_preprocessing(both_pools, FORECAST_WINDOW_MIN)
        
            if df_final_LGBM_X_test is None or X_pct_test is None or y_pct_test is None or df_final_XGB_X_test is None or X_gas_test is None or y_gas_test is None:
                st.error("Preprocessing failed. Cannot proceed with analysis.")
            else:
                test_predictions = None
                y_gas_pred = None

                # Run LGBM model
                with st.spinner("Running LGBM model..."):
                    LGBM = load_model(f"percent_change_{FORECAST_WINDOW_MIN}min_forecast_LGBM")
                    if LGBM is not None:
                        try:
                            y_pct_pred = LGBM.predict(X_pct_test, num_iteration=LGBM.best_iteration)
                            y_pct_rmse = root_mean_squared_error(y_pct_test, y_pct_pred)
                            y_pct_r2 = r2_score(y_pct_test, y_pct_pred)
                            
                        except Exception as e:
                            st.error(f"Error running LGBM model: {str(e)}")
                    else:
                        st.error("Failed to load LGBM model. Skipping LGBM analysis.")

                # Run XGB model
                with st.spinner("Running XGB model..."):
                    XGB = load_model(f"gas_fees_{FORECAST_WINDOW_MIN}min_forecast_XGBoost")
                    if XGB is not None:
                        try:
                            y_gas_pred = XGB.predict(X_gas_test)
                            y_gas_rmse = root_mean_squared_error(y_gas_test, y_gas_pred)
                            y_gas_r2 = r2_score(y_gas_test, y_gas_pred)


                        except Exception as e:
                            st.error(f"Error running XGB model: {str(e)}")
                    else:
                        st.error("Failed to load XGB model. Skipping XGB analysis.")

                # Process final results
                if y_pct_pred is not None and y_gas_pred is not None:
                    df_final = Final_results_processing(LGBM_org_data['time'],y_pct_test,y_pct_pred,y_gas_test,y_gas_pred)

                    #print(df_final.columns)

                    experiment_duration = df_final['time'].iloc[-1] - df_final['time'].iloc[0]
                    avg_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].mean()
                    median_positive_min_investment = df_final[df_final['min_amount_to_invest_prediction']>0]['min_amount_to_invest_prediction'].median()

                    avg_profit = df_final['Profit'].mean()
                    med_profit = df_final['Profit'].median()

                    number_of_simulated_swaps = df_final.shape[0]

                    # ##########################################################################
                    #
                    #  Section 1: Rollup stats from past transactions
                    #
                    # ##########################################################################
                    df_gain = df_final[(df_final['Profit'] > 0) 
                                        & (df_final['min_amount_to_invest_prediction'] > 0) 
                                        & (df_final['min_amount_to_invest_prediction'] < threshold)]

                    st.subheader(f'Results from Previous {np.round(experiment_duration.total_seconds() / 3600):.0f} hour(s)')
                    st.write(f"Number of Transactions: {number_of_simulated_swaps}")
                    st.write(f"Average Recommended Minimum Investment: ${avg_positive_min_investment:.2f}")
                    st.write(f"Median Recommended Minimum Investment: ${median_positive_min_investment:.2f}")
                    st.write(f"Percent of Transactions with Profitable Arbitrage Opportunities: {df_gain.shape[0]/df_final.shape[0]*100:.1f}%")
                    st.write(f"Median Profit Per Transaction (when gain is actualized): ${med_profit:.2f}")

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

                    #
                    # Find the largest bin
                    # Why? its an interesting artifact of the data that the histogram reveals a 
                    #      peak 'mode' where minimum investment appears.  may be derived from 
                    #      transaction fees and gas fees at the time?
                    #
                    max_bin_index = np.argmax(hist_values)
                    max_bin_count = hist_values[max_bin_index]

                    # Calculate the center and width of the largest bin
                    bin_width = bin_edges[1] - bin_edges[0]
                    bin_center = bin_edges[max_bin_index] + bin_width / 2

                    #  Annotate the largest bin
                    axs2[1].annotate(
                        f"Bin Center: {bin_center:.2f} +/- {bin_width/2:.2f}\n",
                        xy=(bin_center, max_bin_count),
                        xytext=(bin_center + bin_width, max_bin_count*0.8),  # Middle of the plot, slightly to the right
                        textcoords='data',
                        #arrowprops=dict(facecolor='black', arrowstyle='--'),
                        fontsize=8,
                        ha='left'
                    )

                    # Plot the histogram
                    axs2[1].hist(data, bins=50, alpha=0.7)
                    axs2[1].set_xlabel('Recommended Minimum Amount to Invest')
                    axs2[1].set_title(f'Distribution of Recommended Minimum Investment in last {experiment_duration.total_seconds() / 3600:.1f} hour(s)')
                    axs2[1].set_xlim(0, threshold)                
                    
                    # Display the figure in Streamlit
                    st.pyplot(fig2)

                    # ##########################################################################
                    #
                    #  Section 3: Recommended investment based on latest transactions.
                    #
                    # ##########################################################################
                    st.subheader(f'Recommended Minimum Investment Prediction ({FORECAST_WINDOW_MIN} minute forecast)')

                    # Predictions
                    df_final_LGBM_X_test_dates = df_final_LGBM_X_test['time']
                    df_final_LGBM_X_test = df_final_LGBM_X_test[['percent_change', 'rolling_mean_8', 'lag_1', 'lag_2']]
                    y_final_pct_pred = LGBM.predict(df_final_LGBM_X_test, num_iteration=LGBM.best_iteration)
                    y_final_gas_pred = XGB.predict(df_final_XGB_X_test)
                    
                    df_min = final_min_amt_invest(df_final_LGBM_X_test_dates, y_final_pct_pred, df_final_XGB_X_test.index, y_final_gas_pred)
                    #import pdb
                    #pdb.set_trace()
                    now = datetime.now()

                    # Difference between current time and last timestamp
                    time_difference = now - df_min['time'].iloc[-1]

                    # Check if the difference is less than FORCAST_WINDOW_MIN minutes
                    is_less_than_x_minutes = time_difference < timedelta(minutes=FORECAST_WINDOW_MIN)
                    x_minutes = timedelta(minutes=FORECAST_WINDOW_MIN)
                    if is_less_than_x_minutes:
                        if df_min['min_amount_to_invest_prediction'].iloc[-1] < 0:
                            st.write(f'Arbitrage Opportunity is not expected {FORECAST_WINDOW_MIN} minute(s) from now.')
                        else:
                            if df_min['percent_change_prediction'].iloc[-1] < 0:
                                st.write(f'**BUY**: Pool 0 ({pool0_address})')
                                st.write(f'**SELL**: Pool 1 ({pool1_address})')
                                st.write(f'**Minimum amount to invest**: ${df_min["min_amount_to_invest_prediction"].iloc[-1]:.2f} at time: {df_min["time"].iloc[-1]+x_minutes}')
                            else:
                                st.write(f'**BUY**: Pool 1 ({pool1_address})')
                                st.write(f'**SELL**: Pool 0 ({pool0_address})')
                                st.write(f'**Minimum amount to invest**: ${df_min["min_amount_to_invest_prediction"].iloc[-1]:.2f} at time: {df_min["time"].iloc[-1]+x_minutes}')
                                
                    else:
                        st.write(f"Last Data point received from query was at {df_min['time'].iloc[-1]}\nData queried is greater than {FORECAST_WINDOW_MIN} minute(s) old, unable to provide minimum amount to invest")
                    


                    
                    
                else:
                    st.error("Cannot proceed with final processing. Some model predictions are missing.")

        st.write(
            "*Disclaimer: The creators of this app are not licensed to provide, offer or recommend financial instruments in any way shape or form. This information and the information within the site should not be considered unique financial advice and if you consider utilizing the model, or investing in crypto you should first seek financial advice from a trained professional, to ensure that you fully understand the risk. Further, while modelling efforts have been undertaken in an effort to avoid risk through the application of the principles of arbitrage, the model has not been empirically tested, and should be approached with extreme caution, care and be utilized at one’s own risk (do not make trades to which you would be unable to fulfill, or would be in a detrimental financial position if it was to not complete as expected).*"
        )

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

        #figa, axsa = plt.subplots(1, 1, figsize=(14, 10))
        #print(len(y_pct_pred), both_pools.shape[0])
        #print(y_pct_pred[:30])
        #axsa.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        #axsa.scatter(y_pct_test, y_pct_pred, marker='o')
        #st.pyplot(figa)

        st.subheader("XGB Gas Fee Model Results")
        st.write(f"Root Mean Squared Error: {y_gas_rmse:.4f}")
        st.write(f"R² Score: {y_gas_r2:.4f}")




        # Price Plots
        st.subheader(f'Pool Price Data')
        fig3, axs3 = plt.subplots(2, 1, figsize=(14, 10))
        axs3[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs3[0].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency

        axs3[0].scatter(both_pools['time'],1/both_pools['p0.weth_to_usd_ratio'], marker='o')
        axs3[0].scatter(both_pools['time'],1/both_pools['p1.weth_to_usd_ratio'], marker='x')

        axs3[0].set_title('Token Price for Pool 0 / Pool 1')
        axs3[0].legend(['Pool 0','Pool 1'])
        axs3[0].set_yscale('log')
        axs3[0].set_xlabel('time')
        axs3[0].set_ylabel('USD per Eth (log scale)')


        both_pools_concat = both_pools.iloc[-50:]
        axs3[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs3[1].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts tick frequency

        axs3[1].scatter(both_pools_concat['time'],1/both_pools_concat['p0.weth_to_usd_ratio'], marker='o')
        axs3[1].scatter(both_pools_concat['time'],1/both_pools_concat['p1.weth_to_usd_ratio'], marker='x')
        
        axs3[1].set_title('Close in Token Price for Pool 0 / Pool 1')
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


        # Display the figure in Streamlit
        st.pyplot(fig4)

        
        # Build Minimum Investment Breakout Table
        st.subheader('Minimum Investment CalculationBreakout')
        last_sample_ds = both_pools.iloc[-1]
        last_sample_ds_pred = df_min.iloc[-1]

        A = 1/last_sample_ds['p0.weth_to_usd_ratio']    # Pool 0 ETH / USDC
        B = 0.005                                        # Pool 0 Transaction Fee
        C = last_sample_ds['p0.gas_fees_usd']           # Pool 0 Gas Fees
        D = 1/last_sample_ds['p1.weth_to_usd_ratio']    # Pool 1 ETH / USDC
        E = 0.03                                        # Pool 1 Transaction Fee
        F = last_sample_ds['p1.gas_fees_usd']           # Pool 1 Gas Fees
        G = last_sample_ds['percent_change']            # Percent Change in Transaction Price
        H = last_sample_ds['total_gas_fees_usd']        # Total Gas Fees
        J = last_sample_ds_pred['percent_change_prediction']
        K = last_sample_ds_pred['gas_fees_prediction']

        L = last_sample_ds['total_gas_fees_usd'] / \
                    (
                        (1 + abs(last_sample_ds['percent_change'])) * (1 - 0.003 if last_sample_ds['percent_change'] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if last_sample_ds['percent_change'] < 0 else 1 - 0.003)
                    )
        M = last_sample_ds_pred['gas_fees_prediction'] / \
                    (
                        (1 + abs(last_sample_ds_pred['percent_change_prediction'])) * (1 - 0.003 if last_sample_ds_pred['percent_change_prediction'] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if last_sample_ds_pred['percent_change_prediction'] < 0 else 1 - 0.003)
                    )

        
        table_dict = {
            'Last Transaction': [A, B, C, D, E, F, G, H, L],
            'Predicted': ['', B, '', '', E, '', J, K, M],
            'Index': [
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
        table_df['Last Transaction'] = table_df['Last Transaction'].apply(lambda x: f"{x:.2e}" if isinstance(x, (int, float)) else x)
        table_df['Predicted'] = table_df['Predicted'].apply(lambda x: f"{x:.2e}" if isinstance(x, (int, float)) else x)

        st.table(table_df)

        #min_invest_last_sample = 

        #print(min_invest_last_sample)

        #min_invest_pred = 

        #print(min_invest_pred)



