import numpy as np
import requests
import pandas as pd
from datetime import datetime

def etherscan_request(action, api_key, address, startblock=0, endblock=99999999, sort='desc'):

    """
    fetch transactions from address
    
    Return: 

    """
    base_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': action,
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
    df = df.sort_values(by='timeStamp', ascending=False).head(10000)
    
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
    #Format P0 and P0 variables of interest
    p0['time'] = p0['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p0['p0.weth_to_usd_ratio'] = p0['WETH_value']/p0['USDC_value']
    p0['gasPrice'] = p0['gasPrice'].astype(float)
    p0['gasUsed']= p0['gasUsed'].astype(float)
    p0['p0.gas_fees_usd'] = (p0['gasPrice']/1e9)*(p0['gasUsed']/1e9)*p0['p0.weth_to_usd_ratio']
    p1['time'] = p1['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p1['p1.weth_to_usd_ratio'] = p1['WETH_value']/p1['USDC_value']
    p1['gasPrice'] = p1['gasPrice'].astype(float)
    p1['gasUsed']= p1['gasUsed'].astype(float)
    p1['p1.gas_fees_usd'] = (p1['gasPrice']/1e9)*(p1['gasUsed']/1e9)*p1['p1.weth_to_usd_ratio']

    #Merge Pool data
    both_pools = pd.merge(p0[['time','timeStamp','p0.weth_to_usd_ratio','p0.gas_fees_usd']],
                          p1[['time','timeStamp','p1.weth_to_usd_ratio','p1.gas_fees_usd']],
                          on=['time','timeStamp'], how='outer'
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