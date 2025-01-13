import os
import pickle
import requests
import datetime
import time
import pytz
import numpy as np
import pandas as pd


from web3 import Web3

from sklearn.model_selection import train_test_split

def fetch_latest_block(etherscan_api_key):
    """
    Used with etherscan_requests_v2: u
    sed as a support method to find a block range for requests for inference.

    """
    # Parameters for the API request
    params = {
        "module": "proxy",
        "action": "eth_blockNumber",
        "apikey": etherscan_api_key
    }
    
    # Make the request
    base_url = "https://api.etherscan.io/api"

    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        #st.error(f"API request failed with status code {response.status_code}")
        (f"API request failed with status code {response.status_code}") 
        return None
    
    # Handle the response
    if response.status_code == 200:
        result = response.json()
        if result.get("status") != "0" and "result" in result:

            latest_block = int(result['result'], 16)  # Convert hex to int
            #print(f"Latest Block Number: {latest_block}")
            return latest_block
        else:
            print(f"Unexpected response or error: {result.get('message', 'Unknown error')}")
            return None
    else:
        print(f"Failed to fetch the latest block. Status code: {response.status_code}")
        return None

# Decode the sqrtPriceX96 from the logs
def decode_data(logs,transaction_hashes):
    base_url = "https://api.etherscan.io/api"
    w3 = Web3(Web3.HTTPProvider(base_url))

    # Uniswap V3 Swap event ABI
    # The topic for the Swap event in Uniswap V3 (Swap event signature)
    # https://www.4byte.directory/event-signatures/?bytes_signature=0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67
    swap_event = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
    swap_event_abi = {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
            {"indexed": False, "internalType": "int256", "name": "amount0", "type": "int256"},
            {"indexed": False, "internalType": "int256", "name": "amount1", "type": "int256"},
            {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
            {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"}
        ],
        "name": "Swap",
        "type": "event"
    }
    # For Decoding Data Field
    datatype_list = [i['type'] for i in swap_event_abi['inputs'] if not i['indexed']]
    raw_data = []
    for log in logs:
        if swap_event in log["topics"] and log["transactionHash"] in transaction_hashes:
            block = int(log['blockNumber'],16)
            transaction_hash=log['transactionHash']
            data = log['data']
            amount0, amount1, sqrt_price_x96, liquidity, tick = w3.eth.codec.decode(datatype_list,bytes.fromhex(data[2:]))
            raw_data.append((block,amount0, amount1, sqrt_price_x96, liquidity, tick,transaction_hash))
    return raw_data

# Function to fetch logs from a block
# Function to fetch logs from a block
def fetch_logs_for_block(api_key, block_number, pool_address):

    base_url = "https://api.etherscan.io/api"

    params = {
        "module": "logs",
        "action": "getLogs",
        "fromBlock": block_number,
        "toBlock": block_number,
        "address": pool_address,
        "apikey": api_key,
    }
    data = {'result':[]}
    #while not data['result']:
    response = requests.get(base_url, params=params)
    data = response.json()

    if data.get("status") == "1" and "result" in data:
        return data["result"]
    else:
        raise ValueError(f"ValueError fetching logs for block {block_number}: {data.get('message', 'Unknown error')}")

def fetch_block_data(etherscan_api_key, block_numbers, time_stamps, transaction_hashes, pool_address):
    """
    works for etherscan_requests_v2
    """
    results = {'timeStamp':[], 'blockNumber':[], 'sqrtPriceX96':[], 'transactionHash':[],'tick':[],'amount0':[],'amount1':[],'liquidity':[]}

    for block,timestamp in zip(block_numbers,time_stamps):
        try:
            logs = fetch_logs_for_block(etherscan_api_key, block, pool_address)
            
        except Exception as e:
            print(f"Error fetching logs: {e} (ignoring for now)")
        
        try:
            data_in_block = decode_data(logs,transaction_hashes)

            for block,amount0, amount1, sqrt_price_x96, liquidity, tick,transaction_hash in data_in_block:
                results['timeStamp'].append(timestamp)
                results['blockNumber'].append(block)
                results['sqrtPriceX96'].append(sqrt_price_x96)
                results['transactionHash'].append(transaction_hash)
                results['tick'].append(tick)
                results['liquidity'].append(liquidity)
                results['amount0'].append(amount0)
                results['amount1'].append(amount1)
            print(".",end='')

        except Exception as e:
            print(f"Error decoding logs {block}: {e} (ignoring for now)")
        
    return  pd.DataFrame(results)

def etherscan_request_v2(etherscan_api_key, pool_address):
    """
    
    """
    MAX_SAMPLES = 10

    # fetch block numbers for the last hour.
    # Assumes that there is at least one transaction from both pools
    # and more than ten transactions total.
    endblock = fetch_latest_block(etherscan_api_key)
    startblock = endblock-5 * 60
    if endblock: 
        print(f"Start Block: {startblock}")
        print(f"Stop Block: {endblock}")

    try: 
        pool = fetch_tokentx_data(etherscan_api_key, pool_address)

        endtime = int(pool['timeStamp'].iloc[0])
        beginningtime = int(pool['timeStamp'].iloc[-1])
        print(f"Successfully fetched {pool.shape[0]} swaps in the last {(endtime -beginningtime)/60:.2f} minutes.")

        # truncate to MAX_SAMPLES
        pool = pool.iloc[-1*MAX_SAMPLES:]
        #print(f"{pool.shape}")

        pool_data = fetch_block_data(etherscan_api_key, 
                                               list(pool['blockNumber'].unique()), 
                                               list(pool['timeStamp'].unique()),
                                                list(pool['transactionHash']),
                                               pool_address)
        pool = pool.merge(pool_data,on=['timeStamp','blockNumber','transactionHash'],how='left').dropna().reset_index(drop=True)
        pool['datetime'] = pool['timeStamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x), tz=pytz.UTC))
        print(f"{pool.blockNumber}")
        return pool[['transactionHash','datetime','timeStamp','sqrtPriceX96','blockNumber','gasPrice','gasUsed','tick','amount0','amount1','liquidity']]

    except Exception as e: 
        print(e)

def fetch_tokentx_data(etherscan_api_key, pool_address, start_block=0, end_block=99999999):

    base_url = 'https://api.etherscan.io/api'

    params = {
        'module': 'account',
        'action': 'tokentx',
        'address': pool_address,
        'startblock': start_block,
        'endblock': end_block,
        'sort': 'desc',
        'apikey': etherscan_api_key
    }
    
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        #st.error(f"API request failed with status code {response.status_code}")
        raise Exception(f"API request failed with status code {response.status_code}")
    
    data = response.json()
    if data['status'] != '1':
        #st.error(f"API returned an error: {data['result']}")
        raise Exception(f"API returned an error: {data['result']}")
    
    df = pd.DataFrame(data['result'])
    
    expected_columns = ['hash', 'blockNumber', 'timeStamp', 'from', 'to', 'gas', 'gasPrice', 'gasUsed', 'cumulativeGasUsed', 'confirmations', 'tokenSymbol', 'value', 'tokenName']
    
    for col in expected_columns:
        if col not in df.columns:
            raise Exception(f"Expected column '{col}' is missing from the response")
    
    df.sort_values(by='timeStamp')
    
    consolidated_data = {}

    for index, row in df.iterrows():
        tx_hash = row['hash']
        
        if tx_hash not in consolidated_data:
            consolidated_data[tx_hash] = {
                'blockNumber': np.int32(row['blockNumber']),
                'timeStamp': int(row['timeStamp']),
                'transactionHash': tx_hash,
                'from': row['from'],
                'to': row['to'],
                'WETH_value': 0,
                'USDC_value': 0,
                'tokenName_WETH': '',
                'tokenName_USDC': '',
                'gas': int(row['gas']),
                'gasPrice': int(row['gasPrice']),
                'gasUsed': int(row['gasUsed']),
                'cumulativeGasUsed': int(row['cumulativeGasUsed']),
                'confirmations': row['confirmations']
            }
        
        if row['tokenSymbol'] == 'WETH':
            consolidated_data[tx_hash]['WETH_value'] = row['value']
            consolidated_data[tx_hash]['tokenName_WETH'] = row['tokenName']
        elif row['tokenSymbol'] == 'USDC':
            consolidated_data[tx_hash]['USDC_value'] = row['value']
            consolidated_data[tx_hash]['tokenName_USDC'] = row['tokenName']

    final_df = pd.DataFrame.from_dict(consolidated_data, orient='index').reset_index(drop=True)

    return final_df.sort_values(by='timeStamp')

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
    threshold_timestamp = current_timestamp - (24 * 60 * 60)  # 24 hours in seconds

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

def create_pool_df(pool_swap_df,transaction_rate,t0_res=18, t1_res=18, num=0):
    """
    used with merge_pool_data_v2 (only)

    TODO: combine this with merge_pool_data_v2 somehow.
    
    TODO: right now, this only works with WETH / USDC.  
    - Implement t#_res to work with other pools.
    - rename the _usd calculated values to work relative to a token 
    
    Create dataframe from extracted query data for a single pool.
    pool_id (string) - hash for uniswap swap pool
    transaction_rate (float) - found on uniswap.com for the specific pool.  
               Can be four different values in uniswap v3: 1%, 0.3%, 0.05%, 0.01%
    t0_res - token 0 resolution - see ethereum contract to understand what the res of the token is.
    t1_res - token 1 resolution 
    num (int) - what number to call the pool informally (Default: Pool 0)

    Columns: 
    Swaps Pool (raw data)
        p#.transaction_time (string) - used to preserve the original timestamp 
        p#.transaciton_epoch_time (long) - used to preserve the original timestamp
        p#.t0_amount (float)
        p#.t1_amount (float)
        p#.t0_token (string)
        p#.t1_token (string)
        p#.tick (long)
        p#.sqrtPriceX96 (long)
        p#.gasUsed (long)
        p#.gasPrice (long)
        p#.blockNumber (long)
        p#.sender (string)
        p#.recipient (string)
        p#.transaction_id (string)
    
    Swaps Pool (calculated)
        p#.transaction_type (int) - 0 is SWAP, 1 is BURN, 2 is MINT
        p#.transaction_rate (float)
        p#.eth_price_usd - conversion from sqrtPriceX96.
        p#.transaction_fee_usd (float) - |p0.transaction_rate * p0.t1_amount|*eth_price_usd
        p#.gas_fees_usd - total gas fees for Pool 0 calculated from gasUsed*gasPrice.
        p#.total_fees_usd - sum of p0.gas_fees_usd and p0.transaction_fee_usd.
    """
    # Extract Raw Pool data from CSV files.
    # Local files are expected in the same directory as this notebook.
    #pool_swap_df = extract_swap_df(pool_id,'./pools/.')
    
    # Reorders things in time series
    pool_swap_df = pool_swap_df.sort_values(by='timestamp')
    
    # Creating columns directly from extracted data...
    pool_swap_df[f'p{num}.transaction_time'] = pool_swap_df['time']
    pool_swap_df[f'p{num}.transaction_epoch_time'] = pool_swap_df['timestamp']
    pool_swap_df[f'p{num}.t0_amount'] = pool_swap_df['amount0']
    pool_swap_df[f'p{num}.t1_amount'] = pool_swap_df['amount1']
    pool_swap_df[f'p{num}.t0_token'] = pool_swap_df['pool.token0.name']
    pool_swap_df[f'p{num}.t1_token'] = pool_swap_df['pool.token1.name']
    pool_swap_df[f'p{num}.tick'] = pool_swap_df['tick']
    pool_swap_df[f'p{num}.sqrtPriceX96'] = pool_swap_df['sqrtPriceX96'].astype(float)
    pool_swap_df[f'p{num}.gasUsed'] = pool_swap_df['transaction.gasUsed']
    pool_swap_df[f'p{num}.gasPrice'] = pool_swap_df['transaction.gasPrice']
    pool_swap_df[f'p{num}.blockNumber'] = pool_swap_df['transaction.blockNumber']
    #pool_swap_df[f'p{num}.sender'] = pool_swap_df['sender']
    #pool_swap_df[f'p{num}.recipient'] = pool_swap_df['recipient']
    pool_swap_df[f'p{num}.transaction_id'] = pool_swap_df['transaction.id']

    # Create new columns with new calculations...
    pool_swap_df[f'p{num}.transaction_type']=0
    pool_swap_df[f'p{num}.transaction_rate']=transaction_rate
    pool_swap_df[f'p{num}.eth_price_usd'] = ((pool_swap_df[f'p{num}.sqrtPriceX96'] / 2**96)**2 / 1e12) **-1
    pool_swap_df[f'p{num}.gas_fees_usd'] = (pool_swap_df[f'p{num}.gasPrice'] / 1e9 )*(pool_swap_df[f'p{num}.gasUsed'] / 1e9) * pool_swap_df[f'p{num}.eth_price_usd']
    pool_swap_df[f'p{num}.transaction_fees_usd'] = np.abs(pool_swap_df[f'p{num}.t1_amount'] * pool_swap_df[f'p{num}.transaction_rate']) * pool_swap_df[f'p{num}.eth_price_usd']
    pool_swap_df[f'p{num}.total_fees_usd'] = pool_swap_df[f'p{num}.gas_fees_usd'] + pool_swap_df[f'p{num}.transaction_fees_usd']

    # Filtering out zero dollar transactions
    pool_swap_df = pool_swap_df[pool_swap_df[f'p{num}.t0_amount'] != 0]
    pool_swap_df = pool_swap_df[pool_swap_df[f'p{num}.t1_amount'] != 0]
    
    # Reseting index
    pool_swap_df.reset_index(drop=False)

    p_df = pool_swap_df[['time',
                        'timestamp',
                        f'p{num}.transaction_time',
                        f'p{num}.transaction_epoch_time',
                        f'p{num}.t0_amount',
                        f'p{num}.t1_amount',
                        f'p{num}.t0_token',
                        f'p{num}.t1_token',
                        f'p{num}.tick',
                        f'p{num}.sqrtPriceX96',
                        f'p{num}.gasUsed',
                        f'p{num}.gasPrice',
                        f'p{num}.blockNumber',
                        #f'p{num}.sender',
                        #f'p{num}.recipient',
                        f'p{num}.transaction_id',
                        f'p{num}.transaction_type',
                        f'p{num}.transaction_rate',        
                        f'p{num}.eth_price_usd',
                        f'p{num}.transaction_fees_usd',
                        f'p{num}.gas_fees_usd',
                        f'p{num}.total_fees_usd']]

    
    return p_df   

def merge_pool_data_v2(p0, p0_txn_fee, p1, p1_txn_fee):
    """
    compatible with etherscan_request_v2(...)
    
    Return: 
    both_pools.columns = ['time', 'timestamp', 'p1.transaction_time', 'p1.transaction_epoch_time',
       'p1.t0_amount', 'p1.t1_amount', 'p1.t0_token', 'p1.t1_token', 'p1.tick',
       'p1.sqrtPriceX96', 'p1.gasUsed', 'p1.gasPrice', 'p1.blockNumber',
       'p1.sender', 'p1.recipient', 'p1.transaction_id', 'p1.transaction_type',
       'p1.transaction_rate', 'p1.eth_price_usd', 'p1.transaction_fees_usd',
       'p1.gas_fees_usd', 'p1.total_fees_usd', 'p0.transaction_time',
       'p0.transaction_epoch_time', 'p0.t0_amount', 'p0.t1_amount',
       'p0.t0_token', 'p0.t1_token', 'p0.tick', 'p0.sqrtPriceX96',
       'p0.gasUsed', 'p0.gasPrice', 'p0.blockNumber', 'p0.sender',
       'p0.recipient', 'p0.transaction_id', 'p0.transaction_type',
       'p0.transaction_rate', 'p0.eth_price_usd', 'p0.transaction_fees_usd',
       'p0.gas_fees_usd', 'p0.total_fees_usd', 'percent_change',
       'total_gas_fees_usd', 'total_transaction_rate',
       'total_transaction_fees_used', 'total_fees_usd', 'swap_go_nogo']
    """
    p0 = p0.copy()
    p1 = p1.copy()

    # Renaming columns to be compatible with data dictionary.
    p0.columns = ['transaction.id','time','timestamp','sqrtPriceX96','transaction.blockNumber','transaction.gasPrice','transaction.gasUsed','tick','amount0','amount1','liquidity']

    # TODO: how do i get this programmatically?
    p0['pool.token0.name'] = 'USDC'
    p0['pool.token1.name'] = 'WETH'

    # Renaming columsn to be compatible with data dictionary.
    p1.columns = ['transaction.id','time','timestamp','sqrtPriceX96','transaction.blockNumber','transaction.gasPrice','transaction.gasUsed','tick','amount0','amount1','liquidity']
    
    # TODO: how do i get this programmatically?
    p1['pool.token0.name'] = 'USDC'
    p1['pool.token1.name'] = 'WETH'

    pool0_swap_df = create_pool_df(p0,transaction_rate=p0_txn_fee,num=0)
    pool1_swap_df = create_pool_df(p1,transaction_rate=p1_txn_fee,num=1)

    # Merge with Forward Fill in Time
    both_pools = pd.merge(pool1_swap_df, pool0_swap_df, on=['time','timestamp'], how='outer').sort_values(by='timestamp')
    both_pools = both_pools.ffill().reset_index(drop=True)
    ###########
    # Add columns that include information from both pools.
    #Both Pools<br>
    #- percent_change - (p0.eth_price_usd-p1.eth_price_usd)/min(p1.eth_price_usd,p0.eth_price_usd)<br>
    #- total_gas_fees_usd - sum of <br>
    #- total_transaction_rate - p0.transaction_rate + p1.transaction_rate
    #- total_transaction_fees_usd - sum of p0.transaction_fee_usd and p1.transaction_fee_usd<br>
    #- total_fees_usd - sum of total_gas_fees_usd and total_transaction_fees_usd<br>
    #- swap_go_nogo (1 or 0) - 1 if total_gas_fees_usd / (|percent_change| - total_transaction_rate) > 0 <br>
    ######################
    eth_price_min = both_pools[['p0.eth_price_usd','p1.eth_price_usd']].min(axis=1)
    both_pools['percent_change'] = (both_pools['p0.eth_price_usd'] - both_pools['p1.eth_price_usd']) / eth_price_min
    both_pools['total_gas_fees_usd'] = both_pools['p0.gas_fees_usd']+both_pools['p1.gas_fees_usd']
    both_pools['total_transaction_rate'] = both_pools['p0.transaction_rate']+both_pools['p1.transaction_rate']
    both_pools['total_transaction_fees_used'] = both_pools['p0.transaction_fees_usd']+both_pools['p1.transaction_fees_usd']
    both_pools['total_fees_usd'] = both_pools['p0.gas_fees_usd']+both_pools['p1.gas_fees_usd']+both_pools['p0.transaction_fees_usd']+both_pools['p1.transaction_fees_usd']
    both_pools['swap_go_nogo'] = (both_pools['total_gas_fees_usd'] / (np.abs(both_pools['percent_change']) - both_pools['total_transaction_rate']))>0
    #TODO: the fill forward creates NaNs in columns which alters the data type of the column.  need to go back recast back to the expected type.
    # remove first 40 rows with NaNs.
    # both_pools = both_pools.iloc[40:]
    # Before the transactions from the 'slower' pool have their first transaction,
    # so of the fields for that pool (pool 0) will be NaNs.  We should attempt to filter those out.
    #both_pools['time'] = pd.to_datetime(both_pools['time'],tz='UTC')
    
    # rows in a randomly selected day:
    # num_rows = (both_pools[both_pools['time'].dt.date == pd.to_datetime("2024-03-13").date()]).shape[0]
    
    # Find the first row with NaNs...
    new_first_row = both_pools['p1.eth_price_usd'].first_valid_index()
    both_pools = both_pools.iloc[new_first_row:]

    # Find the first row with NaNs...
    new_first_row = both_pools['p0.eth_price_usd'].first_valid_index()
    both_pools = both_pools.iloc[new_first_row:]
    
    has_nans = both_pools.isna().any().any()
    print("Are there any NaNs in the DataFrame?", has_nans)

    return both_pools

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

def find_closest_timestamp(df, time_col, label_key, minutes):
    # Note: 'time' must be in a timezone aware  datetime format
    #df.loc[:, time_col] = pd.to_datetime(df[time_col])

    # Create a shifted version of the DataFrame with the target times
    shifted_df = df.copy()
    shifted_df[time_col] = shifted_df[time_col] - pd.Timedelta(minutes=minutes)

    # Merge the original DataFrame with the shifted DataFrame on the closest timestamps
    result_df = pd.merge_asof(df.sort_values(by=time_col),
                              shifted_df.sort_values(by=time_col),
                              on=time_col,
                              direction='forward',
                              suffixes=('', '_label'))

    # Select the required columns and rename them
    result_df = result_df[[time_col,label_key,label_key+'_label']]

    return result_df

def LGBM_Preprocessing(both_pools, params, objective='train'):
    """
    Creating evaluation data from the pool.  The model for LGBM is predicting percent_change
    across the two pools.  

    input arguments:
    - both_pools (dataframe): raw dataframe of merged pool pair.
    - params: model specific metadata
    - objective: train / test / inference objectives changes the output arguments.  
                 train:
                 objective makes train/test splits with labels.  
                 test: 
                 objective returns the labeled data without splitting
                 inference:   
                 takes the latest timestamps (in this case samples without labels) to use as input into a 
                 trained model.  This is useful when pulling data and wanting to evaluate
                 the data (use a test split of the data), then perform inference on the same 
                 dataset just using a portion that is garunteed to not be used for training.
    
    Return (inference): 
    - (dataframe): data frame preserves all columns as int_df. i.e. the most recent shift_minutes of data.  
   
    Return (test): 
    - X (dataframe with time index): model features
    - y (dataframe with time index): labels   
   
    Return (train): 
    - X_train, X_test, y_train, y_test (dataframe with time index): standard sklearn train/test splits
    """
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE_LIST = params['PCT_CHANGE_N_WINDOW_AVERAGE']
    NUM_LAGS = params['PCT_CHANGE_NUM_LAGS']
    TEST_SPLIT = params['PCT_CHANGE_TEST_SPLIT']

    int_df = both_pools.copy()
    int_df = int_df[['time','percent_change']]
    int_df = find_closest_timestamp(int_df, 'time', 'percent_change', FORECAST_WINDOW_MIN)

    #int_df = shift_column_by_time(int_df, 'time', 'percent_change', forecast_window_min)
    num_lags = 2  # Number of lags to create
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['percent_change'].shift(i)

    for i in N_WINDOW_AVERAGE_LIST:
        int_df[f'rolling_mean_{i}'] = int_df['percent_change'].rolling(window=i).mean()

    # Create time index for the dataframe to preserve timestamps with splits.
    int_df.index = int_df.pop('time')

    if objective == 'inference':
        int_df = int_df[int_df['percent_change_label'].isna()]
        
        #remove labels from data in place.
        int_df.pop('percent_change_label') 
        return int_df
    
    elif objective == 'test':
        int_df.dropna(inplace=True)

        y = int_df.pop('percent_change_label')
        X = int_df.copy()
        return X,y
            
    elif objective == 'train':
        int_df.dropna(inplace=True)

        # Create labels and training...
        y = int_df.pop('percent_change_label')
        X = int_df.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42,shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        print(f"LGBM_Preprocessing: Error - wrong objective specified ({objective}). Should be train, test or inference")
        return None

def XGB_preprocessing(both_pools, params, objective='train'):
    
    FORECAST_WINDOW_MIN = params['FORECAST_WINDOW_MIN']
    N_WINDOW_AVERAGE_LIST = params['GAS_FEES_N_WINDOW_AVERAGE']
    NUM_LAGS = params['GAS_FEES_NUM_LAGS']
    TEST_SPLIT = params['GAS_FEES_TEST_SPLIT']

    int_df = both_pools.copy()
    int_df = int_df[['time','total_gas_fees_usd']]

    int_df = find_closest_timestamp(int_df, 'time', 'total_gas_fees_usd', FORECAST_WINDOW_MIN)
    int_df.index = int_df.pop('time')
    #int_df.index = pd.to_datetime(int_df.index)
    
    for i in range(1, NUM_LAGS + 1):
        int_df[f'lag_{i}'] = int_df['total_gas_fees_usd'].shift(i)
    
    for i in N_WINDOW_AVERAGE_LIST:
        int_df[f'rolling_mean_{i}'] = int_df['total_gas_fees_usd'].rolling(window=i).mean()

    if objective == 'inference':
        int_df = int_df[int_df['total_gas_fees_usd_label'].isna()]

        #remove labels from data for inference.
        int_df.pop('total_gas_fees_usd_label') 
        return int_df
    
    elif objective == 'test':
        int_df.dropna(inplace=True)

        y = int_df.pop('total_gas_fees_usd_label')
        X = int_df.copy()
        return X,y
            
    elif objective == 'train':
        int_df.dropna(inplace=True)

        # Create labels and training...
        y = int_df.pop('total_gas_fees_usd_label')
        X = int_df.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42,shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        print(f"XGB_Preprocessing: Error - wrong objective specified ({objective}). Should be train, test or inference")
        return None

def calculate_min_investment(df,pool0_txn_fee_col, pool1_txn_fee_col, gas_fee_col,percent_change_col,min_investment_col='min_amount_to_invest'):

    # Percent Change is defined as P0 - P1 / min(P0,P1), 
    # where P0 is price of token in pool 0, P1 is price of token in pool 1.
    #
    # if percent_change (or ΔP) is positive then P0 > P1 and 
    #            first transaction is on pool 1
    #            second transaction is on pool 0 and 
    # if percent_change (or ΔP) is negative then P1 > P0 and 
    #            first transaction is on pool 0 
    #            second transaction is on pool 1 
    # 
    # Minimum Investment is defined as: G / [ (1+|ΔP|) x (1-T1) - (1-T0) ]
    # where ΔP is real, 0 < T1 < 1, 0 < T0 < 1 
    #       T0 is transaction fees for first pool transaction, T1 is transaction fees for second pool transaction.
    #       Note: use the pool with the smallest price for the first pool transaction
    #
    # To calculate the minimum investment, you must generate two terms conditional on the 
    # pool that is greater, which can be determined based on the sign.  
    #         
    # The first term is (1+ΔP) x (1 - T1) 
    # The second term is (1 - T0) 
    #
    # The minimum investment (using the two terms in the denominator) can have different outcomes depending 
    # on the configuration: 
    #    T0 > T1, positive outcome regardless of ΔP
    #    T1 > T0, negative outcome if |ΔP| < (1-T0)/(1-T1) - 1
    #    T1 > T0, positive outcome if |ΔP| > (1-T0)/(1-T1) - 1
    def min_investment(row):
        result = row[gas_fee_col] / (
            (1 + abs(row[percent_change_col])) *
            (1 - row[pool1_txn_fee_col] if row[percent_change_col] < 0 else 1 - row[pool0_txn_fee_col]) -
            (1 - row[pool0_txn_fee_col] if row[percent_change_col] < 0 else 1 - row[pool1_txn_fee_col])
        )
        if not np.isfinite(result):  # Check for inf or NaN
            return np.nan
        
        return result

    df[min_investment_col] = df.apply(min_investment, axis=1)


    # if the value in the dataframe is negative or inf, set the minimum investment to NaN to indicate 
    # ΔP is not large enough to overcome transaction fees or other race conditions.
    #df[min_investment_col] = np.where(df[min_investment_col] < 0, np.nan, df[min_investment_col])

    return df

def load_model(model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    base_model_path = os.path.join(models_dir, model_name)
    
    # Check for different possible file extensions
    possible_extensions = ['', '.h5', '.pkl', '.joblib']
    model_path = next((base_model_path + ext for ext in possible_extensions if os.path.exists(base_model_path + ext)), None)
    
    if model_path is None:
        print(f"Model file not found for: {model_name}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model {model_name} loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None