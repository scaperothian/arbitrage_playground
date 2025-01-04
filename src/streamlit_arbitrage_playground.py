import streamlit as st
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta

import random
import time

import seaborn as sns
sns.set_style('darkgrid')

import arbutils

st.set_page_config(page_title="Arbitrage Plots", page_icon=None, layout="centered", initial_sidebar_state="expanded")

st.title("Liquidity Pool Arbitrage Playground (version 2)")

st.write("The minimum investment required must satisfy the following criteria.")
st.latex(r'P_{min}=\frac{G}{(1+\Delta{P})(1-T_1) - (1-T_0)}')

st.write(
    """ 
    Use Sliders and Text inputs Fees to see how that effects the investment optionsn for Uniswap liquidity pools.
    """
)

# Function to generate the sine and cosine waves
def generate_plot(investment, profit):
    
    fig, ax = plt.subplots()
    ax.plot(investment, profit, label="profit")
    ax.legend()
    ax.axhline(y=profit[-1], color='black', linestyle='--', label=f'Max Profit = {profit[-1]}')
    ax.set_ylabel("Estimated Profit (number of tokens)")
    ax.set_xlabel("Initial Investment (number of tokens)")
    ax.set_title("Profit from Liquidity Pools")
    ax.grid(True)

    return fig

st.sidebar.subheader("Pool 0 Parameters")

transaction_budget = float(st.sidebar.text_input("Transaction Budget (i.e. how many tokens to use on the first swap)", value=10000))

pool0_price = st.sidebar.slider('Pool 0 Token Price', min_value=3000., max_value=3500., value=3100., step=0.01)
#pool0_transaction_fee = float(st.sidebar.text_input("Pool 0 Transaction Fee", value=0.003))
pool0_transaction_fee = float(st.sidebar.selectbox(label="Pool 0 Transaction Fee",options=[0.01, 0.003,0.0005,0.0001],index=1)) #select 0.003 by default.

pool0_gas_fees = st.sidebar.slider('Pool 0 Gas Fee (i.e. to execute a swap)', min_value=0., max_value=100., value=10., step=0.001)

st.sidebar.subheader("Pool 1 Parameters")

pool1_price = st.sidebar.slider('Pool 1 Token Price', min_value=3000., max_value=3500., value=3200., step=0.01)
#pool1_transaction_fee = float(st.sidebar.text_input("Pool 1 Transaction Fee", value=0.0005))
pool1_transaction_fee = float(st.sidebar.selectbox(label="Pool 1 Transaction Fee",options=[0.01, 0.003,0.0005,0.0001],index=2)) #select 0.0005 by default.
pool1_gas_fees = st.sidebar.slider('Pool 1 Gas Fees (i.e. to execute a swap)', min_value=0., max_value=100., value=10., step=0.001)

st.subheader("Results")

percent_change = (pool0_price - pool1_price) / np.minimum(pool0_price, pool1_price)
total_gas_fees = pool0_gas_fees + pool1_gas_fees
st.write(f'Percent Difference Between Pool Prices (scalar): {percent_change}')
st.write(f'Total Gas Prices (token x): {total_gas_fees}')

input_dict = {
    'percent_change':[percent_change],
    'total_gas_fees':[total_gas_fees],
    'pool0_transaction_fee':[pool0_transaction_fee],
    'pool1_transaction_fee':[pool1_transaction_fee], 
}

df = pd.DataFrame(input_dict)

df = arbutils.calculate_min_investment(df,
                                  'pool0_transaction_fee',
                                  'pool1_transaction_fee',
                                  'total_gas_fees',
                                  'percent_change',
                                  min_investment_col='min_amount_to_invest')

minimum_amount_to_invest = df.iloc[0]['min_amount_to_invest']

st.subheader("Profit Estimate")
if minimum_amount_to_invest > 0:
    st.write(f"Minimum Amount to Invest (token x): {minimum_amount_to_invest:.2f}")

    # Create a plot of Investments from minimum to budget
    #if percent_change is positive then P0 > P1 and transaction 0 is P1, if percent_change is negative then 
    # P1 > P0 and transaction 0 is P0.
    if pool0_price > pool1_price:
        t0_fees = pool1_transaction_fee
        t1_fees = pool0_transaction_fee
    else: 
        t0_fees = pool0_transaction_fee
        t1_fees = pool1_transaction_fee

    if minimum_amount_to_invest < transaction_budget:
        investment = np.arange(minimum_amount_to_invest,transaction_budget,1.0)
        profit  = investment * (1 + np.abs(percent_change)) * (1 - t1_fees) - investment * (1-t0_fees) - total_gas_fees
        st.write(f"Maximum estimated profit for budget is: {profit[-1]:.2f}")
        st.pyplot(generate_plot(investment, profit))

    else: 
        st.write("Budget is too low to make a profit based on the liquidity pool attributes.")

else: 
        st.write(f"Minimum Amount to Invest (token x): N/A")

        st.write("Price differences cannot overcome the fees.")