import pandas as pd
import requests
from datetime import datetime

import numpy as np

def etherscan_request(api_key, address, startblock=0, endblock=99999999, sort='desc'):

    """
    fetch transactions from address
    
    Return: 

    """
    base_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': 'tokentx',
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
    """
    compatible with etherscan_request(...)

    Return: 
    
    both_pools.columns - ['time', 'timeStamp', 'blockNumber', 'p0.weth_to_usd_ratio', 'p0.gas_fees_usd',
       'p1.weth_to_usd_ratio', 'p1.gas_fees_usd', 'percent_change',
       'total_gas_fees_usd']
    """
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
    both_pools = pd.merge(p0[['time','timeStamp','blockNumber','p0.weth_to_usd_ratio','p0.gas_fees_usd']],
                          p1[['time','timeStamp','blockNumber','p1.weth_to_usd_ratio','p1.gas_fees_usd']],
                          on=['time','timeStamp','blockNumber'], how='outer'
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

if __name__ == "__main__":
    ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEYKEY")
    POOL0_ADDRESS="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" # USDC / WETH (0.05%) 
    POOL0_TXN_FEE = 0.0005
    POOL1_ADDRESS="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8" # USDC / WETH (0.3%)
    POOl1_TXN_FEE = 0.003

    # Fetch and process data for both pools
    p0 = etherscan_request(ETHERSCAN_API_KEY, address=POOL0_ADDRESS)
    if p0 is None:
        print("Failed to fetch data from Etherscan for Pool 0. Please check your API key and try again.")
        raise Exception
    p1 = etherscan_request(ETHERSCAN_API_KEY, address=POOL1_ADDRESS)
    if p1 is None:
        print("Failed to fetch data from Etherscan for Pool 1. Please check your API key and try again.")
        raise Exception       


    both_pools = merge_pool_data(p0, p1)