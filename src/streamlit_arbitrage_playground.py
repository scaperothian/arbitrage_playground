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

st.subheader("Getting Started")

st.markdown(
    "New to liquidity pool arbitrage?  Check out a primer at: [mydiamondhands.io](https://mydiamondhands.io/2024/05/09/a-basic-theoretical-arbitrage-approach/)",
    unsafe_allow_html=True
)

st.subheader("Instructions")
st.write(
    """ 
    Use Sliders and Pull downs in sidebar on the left to explore how prices and fees effects the investment options for Uniswap liquidity pools.
    """
)
st.write(
    """
    *Inuitions: (1) if prices are closer together the minimum amount to invest is higher.  (2) if the prices are closer together, the difference in transaction rates can effect whether you can overcome fees with the price difference.*
    """
)

st.sidebar.subheader("Pool 0 Parameters")

pool0_price = st.sidebar.slider('Pool 0 Token Price (${P_0}$)', min_value=2995., max_value=3005., value=3000., step=0.01)
#pool0_transaction_fee = float(st.sidebar.text_input("Pool 0 Transaction Fee", value=0.003))
pool0_transaction_fee = float(st.sidebar.selectbox(label="Pool 0 Transaction Fee (${T_0}$)",options=[0.01, 0.003,0.0005,0.0001],index=1)) #select 0.003 by default.

pool0_gas_fees = st.sidebar.slider('Pool 0 Gas Fee (${G_0}$)', min_value=0., max_value=100., value=10., step=0.001)

st.sidebar.subheader("Pool 1 Parameters")

pool1_price = st.sidebar.slider('Pool 1 Token Price (${P_1}$)', min_value=2995., max_value=3005., value=3002., step=0.01)
#pool1_transaction_fee = float(st.sidebar.text_input("Pool 1 Transaction Fee", value=0.0005))
pool1_transaction_fee = float(st.sidebar.selectbox(label="Pool 1 Transaction Fee (${T_1}$)",options=[0.01, 0.003,0.0005,0.0001],index=2)) #select 0.0005 by default.
pool1_gas_fees = st.sidebar.slider('Pool 1 Gas Fees (${G_1}$)', min_value=0., max_value=100., value=10., step=0.001)

st.subheader("Relevant Equations")
st.write("Percent Difference (ΔP) Equation: ${\\frac{P_0-P_1}{min(P_0,P_1)}}$")
st.write("Total Gas Fees (G) Equation: ${G_0+G_1}$")
st.write("Minimum Investment without slippage Equation: ${\\frac{G}{(1+|\\Delta{P}|)(1-T_1)-(1-T_0))}}$")
st.write("Profit without slippage Equation: ${A(1+|\\Delta{P}|)(1-T_1)-A(1-T_0) - G}$")

st.subheader("Calculated Profit")

percent_change = (pool0_price - pool1_price) / np.minimum(pool0_price, pool1_price)
total_gas_fees = pool0_gas_fees + pool1_gas_fees
st.write(f'Calculated Percent Difference (ΔP) Between Pool Prices (scalar): {percent_change}')
st.write(f'Calculated Total Gas Fees (G) (token x): {total_gas_fees}')


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

if minimum_amount_to_invest > 0:
    st.write(f"Calculated Minimum Amount to Invest (token x): {minimum_amount_to_invest:.2f}")

    st.write("*Use the slider to see how going above and below the minimimum amount above effects the Profit.*")
    transaction_budget = float(st.slider("Investment Budget (A) i.e. how many tokens to use on the first swap", min_value=0., max_value=20000., value=10000., step=10.))


    # Create a plot of Investments from minimum to budget
    #if per cent_change is positive then P0 > P1 and transaction 0 is P1, if percent_change is negative then 
    # P1 > P0 and transaction 0 is P0.
    if pool0_price > pool1_price:
        t0_fees = pool1_transaction_fee
        t1_fees = pool0_transaction_fee
    else: 
        t0_fees = pool0_transaction_fee
        t1_fees = pool1_transaction_fee

    if minimum_amount_to_invest < transaction_budget:
        profit  = transaction_budget * (1 + np.abs(percent_change)) * (1 - t1_fees) - transaction_budget * (1-t0_fees) - total_gas_fees

        st.write(f"***Calculated Profit without slippage: {profit:.2f}***")
        #st.pyplot(generate_plot(investment, profit))

    else: 
        st.write("Budget is too low to make a profit based on the liquidity pool attributes.")

else: 
        st.write(f"Minimum Amount to Invest (token x): N/A")

        st.write("Price differences cannot overcome the fees.")