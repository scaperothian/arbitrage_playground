import requests
import time
import os
import pytz
import json

import numpy as np
import pandas as pd

from pandas import json_normalize
from datetime import datetime, timedelta

from web3 import Web3

# Decode the sqrtPriceX96 from the logs
def etherscan_decode_data(logs,transaction_hashes):
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
def etherscan_logs_for_block(api_key, block_number, pool_address):

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

def etherscan_request_block_data(etherscan_api_key, block_numbers, time_stamps, transaction_hashes, pool_address):
    results = {'timeStamp':[], 'blockNumber':[], 'sqrtPriceX96':[], 'transactionHash':[],'tick':[],'amount0':[],'amount1':[],'liquidity':[]}
    token0_resolution = 1e6  # right now this is USDC
    token1_resolution = 1e18 #right now this is WETH


    for block,timestamp in zip(block_numbers,time_stamps):
        try:
            logs = etherscan_logs_for_block(etherscan_api_key, block, pool_address)
            
        except Exception as e:
            print(f"Error fetching logs: {e} (ignoring for now)")
        
        try:
            data_in_block = etherscan_decode_data(logs,transaction_hashes)

            for block,amount0, amount1, sqrt_price_x96, liquidity, tick,transaction_hash in data_in_block:
                results['timeStamp'].append(int(timestamp))
                results['blockNumber'].append(np.int32(block))
                results['sqrtPriceX96'].append(float(sqrt_price_x96))
                results['transactionHash'].append(transaction_hash)
                results['tick'].append(float(tick))
                results['liquidity'].append(float(liquidity))
                results['amount0'].append(float(amount0)/token0_resolution)
                results['amount1'].append(float(amount1)/token1_resolution)
            print(".",end='')

        except Exception as e:
            print(f"Error decoding logs {block}: {e} (ignoring for now)")
        
    return  pd.DataFrame(results)

def etherscan_request_tokentx(etherscan_api_key, pool_address, start_block=0, end_block=99999999):

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
        raise Exception(f"API request failed with status code {response.status_code}: {response}")
    
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
                'gas': float(row['gas']),
                'gasPrice': float(row['gasPrice']),
                'gasUsed': float(row['gasUsed']),
                'cumulativeGasUsed': float(row['cumulativeGasUsed']),
                'confirmations': row['confirmations']
            }
        
        if row['tokenSymbol'] == 'WETH':
            consolidated_data[tx_hash]['WETH_value'] = float(row['value'])
            consolidated_data[tx_hash]['tokenName_WETH'] = row['tokenName']
        elif row['tokenSymbol'] == 'USDC':
            consolidated_data[tx_hash]['USDC_value'] = float(row['value'])
            consolidated_data[tx_hash]['tokenName_USDC'] = row['tokenName']

    final_df = pd.DataFrame.from_dict(consolidated_data, orient='index').reset_index(drop=True)

    return final_df.sort_values(by='timeStamp')

def etherscan_get_latest_block(etherscan_api_key):
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

def etherscan_request_v2(etherscan_api_key, pool_address, max_samples=10, lookback_minutes=60):
    """
    Three big steps: 
    1. Get the latest timestamp for etherscan
    2. Get the transfer token information so we have a blocknumber with a finalized timestamp
    3. Get block raw data with valid transactions (i.e. from #1 and are labeled as swap events)
       a. fetch the logs
       b. decode the logs
    4. Merge the transfer token data with the raw block data (i.e. sqrtPrice, liquidity, etc.)
       
    Note: to encapsulate this function, I am not currently using a class.  Instead i'm doing
          embedding methods within a method.  TODO: fix this.
    """
    # magic number used to estimate how far 
    # we should look back from latest timestamp
    MAINNET_TRANSACTIONS_PER_MINUTE = 5

    # this is an estimate of how many transactions we care about to 
    # make the model work for example, we need to support moving 
    # average and lags from past data for model inference.
    # set to 10 by default for inference.
    MAX_SAMPLES = max_samples

    # Based on some empircal data, lookback_minutes=60 minutes tends to give pretty
    # good results for getting at least one transaction for each
    # pool (only tested with two pools).  
    # This approach anchors forward filling data.
    NUMBER_OF_ESTIMATED_MINUTES_LOOKBACK = lookback_minutes

    ##############################################################
    #
    #     1. Get the latest timestamp for etherscan
    # 
    ##############################################################
    endblock = etherscan_get_latest_block(etherscan_api_key)
    startblock = endblock-MAINNET_TRANSACTIONS_PER_MINUTE * NUMBER_OF_ESTIMATED_MINUTES_LOOKBACK
    if endblock: 
        print(f"Start Block: {startblock}")
        print(f"Stop Block: {endblock}")
    

    try: 
    ##############################################################
    #
    #         2. Get the transfer token information so we have a 
    #            blocknumber and timestamp (finalized)
    # 
    ##############################################################
        pool = etherscan_request_tokentx(etherscan_api_key, pool_address, start_block=startblock, end_block=endblock)

        endtime = int(pool['timeStamp'].iloc[-1])
        beginningtime = int(pool['timeStamp'].iloc[0])
        print(f"Successfully fetched {pool.shape[0]} swaps in the last {(endtime -beginningtime)/60:.2f} minutes.")

        # otherwise preserve the number of samples...
        if MAX_SAMPLES != 'all':
            # truncate to MAX_SAMPLES to avoid decoding unnecessary transactions...
            pool = pool.iloc[-1*MAX_SAMPLES:]
        
    ##############################################################
    #
    #    3. Get block raw data with valid transactions 
    #       (i.e. from #1 and are labeled as swap events)
    # 
    ##############################################################

        pool_data = etherscan_request_block_data(etherscan_api_key, 
                                               list(pool['blockNumber'].unique()), 
                                               list(pool['timeStamp'].unique()),
                                                list(pool['transactionHash']),
                                               pool_address)


    ##############################################################
    #
    #    4. Merge the transfer token data with the raw block data 
    #      (i.e. sqrtPrice, liquidity, etc.)
    # 
    ##############################################################
    
        pool = pool.merge(pool_data,on=['timeStamp','blockNumber','transactionHash'],how='left').dropna().reset_index(drop=True)
        pool['datetime'] = pool['timeStamp'].apply(lambda x: datetime.fromtimestamp(int(x), tz=pytz.UTC))
        print(f"{pool.blockNumber}")
        return pool[['transactionHash','datetime','timeStamp','sqrtPriceX96','blockNumber','gasPrice','gasUsed','tick','amount0','amount1','liquidity']]

    except Exception as e: 
        print(e)

def etherscan_request_tokentx(etherscan_api_key, pool_address, start_block=0, end_block=99999999):

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
                'gas': float(row['gas']),
                'gasPrice': float(row['gasPrice']),
                'gasUsed': float(row['gasUsed']),
                'cumulativeGasUsed': float(row['cumulativeGasUsed']),
                'confirmations': row['confirmations']
            }
        
        if row['tokenSymbol'] == 'WETH':
            consolidated_data[tx_hash]['WETH_value'] = float(row['value'])
            consolidated_data[tx_hash]['tokenName_WETH'] = row['tokenName']
        elif row['tokenSymbol'] == 'USDC':
            consolidated_data[tx_hash]['USDC_value'] = float(row['value'])
            consolidated_data[tx_hash]['tokenName_USDC'] = row['tokenName']

    final_df = pd.DataFrame.from_dict(consolidated_data, orient='index').reset_index(drop=True)

    return final_df.sort_values(by='timeStamp')

def etherscan_get_chunks(start_block_num, end_block_num):
    """
    Calculates the block ranges to pull data from etherscan.  Anecdotally, a range of blocks that 
    results in more than 5000 transactions will fail.  So, based on a request for range of blocks 
    of 2500 or more, calculate the block ranges for a request that will more reliably succeed.    
    """
    MAGIC_NUMBER=2500
    chunks = []
    while start_block_num <= end_block_num:
        s = start_block_num
        e = start_block_num+MAGIC_NUMBER
        if e > end_block_num: 
            e = end_block_num
        start_block_num = e+1
        chunks.append((s,e))
    return chunks

def thegraph_request_with_pagination(thegraph_api_key, pool_address, old_date, new_date, data_path=None, checkpoint_file='checkpoint.json',batch_size=1000):
    """
    Note: batch_size must be greater than 1!  Why would do a batch_size of 1?

    Keys based on default query template: 
         ['amount0', 'amount1', 'amountUSD', 'id', 'recipient', 'sender',
                            'sqrtPriceX96', 'tick', 'timestamp', 'pool.id', 'pool.token0.id',
                            'pool.token0.name', 'pool.token0.symbol', 'pool.token1.id',
                            'pool.token1.name', 'pool.token1.symbol', 'transaction.blockNumber',
                            'transaction.gasPrice', 'transaction.gasUsed', 'transaction.id',
                            'time']


    Return:      
    all_data.keys() - ['transactionHash', 'datetime', 'timeStamp', 'sqrtPriceX96',
                'blockNumber', 'gasPrice', 'gasUsed', 'tick', 'amount0', 'amount1',
                'liquidity']
    
    """
    if batch_size <= 1:
        raise Exception("Batch Size must be greater than one!  Why would do a batch size of 1?")

    query_template = """
    {
        swaps(
            first: %s,
            where: {
                pool: "%s",
                timestamp_gt: "%s",
                timestamp_lt: "%s"
            },
            orderBy: timestamp
        ) {
            id
            timestamp
            sender
            recipient
            amount0
            amount1
            amountUSD
            sqrtPriceX96
            tick
            pool {
                id
                token0 {
                    id
                    symbol
                    name
                }
                token1 {
                    id
                    symbol
                    name
                }
            }
            transaction {
                id
                blockNumber
                gasUsed
                gasPrice
            }
        }
    }
    """
    newest_id = int(new_date.timestamp())

    if data_path and os.path.exists(checkpoint_file):
        
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        oldest_id = int(checkpoint['last_timestamp'])
        batch_num = checkpoint['batch_num']
        all_data = checkpoint.get('accumulated_data', [])
        print(f"Get data from checkpoint ({checkpoint_file}).  Processing data from timestamp: {oldest_id} to {newest_id}")
    else:
        oldest_id = int(old_date.timestamp())
        batch_num = 0
        all_data = []
        print(f"Processing data from timestamp: {oldest_id} to {newest_id}")

    #
    #  Discussion of loop:
    #  The goal of the loop is to extract all relevant transactions into 
    #  all_data list.  The bounds are oldest_id to newest_id.  These are 
    #  epoch timestamps (i.e. seconds based).  When a batch of 'batch_size'
    #  or less transactions is returned from the response, the oldest_id
    #  is changed in two ways (1) its updated to the newest timestamp in the
    #  "new_data" batch of responses and (2) the timestamp found is decremented
    #  by 1 second for a specific race condition (see next paragraph).  The oldest_id
    #  is thus approaching newest_id until oldest_id == newest_id.  The program breaks 
    #  at this point.
    #
    #  Race condition: it is observed that there are multiple transactions that occur
    #  within the same timestamp (i.e. multiple transaction can occur in the same block
    #  from within the same pool).  Because of this, when we use "first: 1000" in the GraphQL
    #  if there is a case where multiple transactions occur with the same timestamp and just
    #  happens to occur at the end of the list of 1000, the transactions afterward may be 
    #  excluded if care is not taken for this case in managing timestmaps.  The decrement by 1
    #  is intended to pick up where the request left off assuming the last timestamp could have
    #  had multiple transactions.  
    #
    while True:    
        query = query_template % (batch_size, pool_address, oldest_id, newest_id)

        try:
            response = requests.post(
                f'https://gateway.thegraph.com/api/{thegraph_api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV',
                json={'query': query}
            )  
        except Exception as e:
            print(f"Post error occurred: {e}")
            break
    
        data = response.json()
        if 'errors' in data:
            raise Exception(f"GraphQL query error: {data['errors']}")
            
        new_data = data['data']['swaps'] if 'swaps' in data['data'] else []
        if not new_data:
            # Exit Loop?
            if len(all_data)==0:
                print(f"No data found {oldest_id} and {newest_id}.")
                return None
                            
        # Save the batch immediately to CSV
        batch_df = json_normalize(new_data)
        batch_df['datetime'] = batch_df['timestamp'].apply(lambda x: datetime.fromtimestamp(int(x),tz=pytz.UTC))

        # add batch to all_data
        all_data.extend(new_data)

        # Update oldest_id (see note above about Race Condition)
        if (int(new_data[0]['timestamp']) - int(new_data[-1]['timestamp'])) != 0: 
            oldest_id = int(new_data[-1]['timestamp'])-1
        else:
            # Exit Loop
            break

        
        print(f"{batch_num}: [{oldest_id}-{newest_id}]found {len(new_data)} swaps from {new_data[0]['timestamp']} to {new_data[-1]['timestamp']}")
        batch_num += 1

        if data_path:
            batch_df.to_csv(f'{data_path}/{pool_address}/pool_id_{pool_address}_swap_batch_{batch_num}.csv')
            checkpoint = {
                'last_timestamp': oldest_id,
                'batch_num': batch_num
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
    
    print(f"Found {len(all_data)} swaps from {all_data[0]['timestamp']} to {all_data[-1]['timestamp']}")

    return all_data

def thegraph_request(thegraph_api_key, etherscan_api_key, pool_address, old_date=None, new_date=None, data_path=None, checkpoint_file='checkpoint.json',batch_size=1000):
    """
    assume that data_path == None means that we are not saving data...


    Note: make this a class.  currently using sub-methods which is just to encapsulate 
          operations that are suited for Python methods. 
    """

    if new_date == None or old_date == None: 
        # Create timestamps based on the latest timestamp...for inference
        new_date = datetime.now(pytz.UTC)
        old_date = new_date - timedelta(hours=1)
    else:
        print(f"Using given dates: {old_date} to {new_date}")

    full_start = time.time()
    
    if data_path: 
        # Create data directory if it doesn't exist
        os.makedirs(f'{data_path}/{pool_address}/', exist_ok=True)
    
    swaps_data = thegraph_request_with_pagination(
        thegraph_api_key=thegraph_api_key,
        pool_address=pool_address,
        old_date=old_date,
        new_date=new_date,
        data_path=data_path,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size
    )

    if swaps_data is None:
        print("Error processing data.")
        return None
    
    # Create DataFrame
    swaps_df = json_normalize(swaps_data)
    

    swaps_df['transactionHash'] = swaps_df['transaction.id']
    swaps_df['datetime'] = swaps_df['timestamp'].apply(lambda x: datetime.fromtimestamp(int(x),tz=pytz.UTC))
    swaps_df['timeStamp'] = swaps_df['timestamp'].astype('int64')
    swaps_df['sqrtPriceX96'] = swaps_df['sqrtPriceX96'].astype('float')
    swaps_df['blockNumber'] = swaps_df['transaction.blockNumber'].astype('int32')
    swaps_df['gasPrice'] = swaps_df['transaction.gasPrice'].astype('float')
    swaps_df['gasUsed'] = swaps_df['transaction.gasUsed'].astype('float')
    swaps_df['tick'] = swaps_df['tick'].astype('float')
    swaps_df['amount0'] = swaps_df['amount0'].astype('float')
    swaps_df['amount1'] = swaps_df['amount1'].astype('float')
    swaps_df['liquidity'] = -1.0  #not implemented

    if swaps_df['gasUsed'].sum()==0:
        print(f"Swaps Found (Prior to Merge, Prior to drop_duplicates): {swaps_df.shape}")
        swaps_df = swaps_df.drop_duplicates()
        #########################################
        #    If the gasUsed is all zero, then 
        #    we need to get the gasUsed from the 
        #    token transfers...
        #########################################
        pool_startblock = swaps_df['blockNumber'].iloc[0]
        pool_endblock = swaps_df['blockNumber'].iloc[-1]

        # must throttle the requests for tokentx.  it will give you 
        # less than 5000 transactions regardless of what block range you 
        # give it.  so we break it up into 2500 block chunks.
        block_chunks = etherscan_get_chunks(pool_startblock, pool_endblock)
        swaps_tokentx_df = pd.DataFrame()
        for startblock, endblock in block_chunks:
            #throttle requests...
            time.sleep(0.01)
            swaps_tokentx_df = pd.concat([swaps_tokentx_df, etherscan_request_tokentx(etherscan_api_key, 
                                                                   pool_address, 
                                                                   start_block=startblock, 
                                                                   end_block=endblock)])

        #swaps_tokentx_df = etherscan_request_tokentx(etherscan_api_key, pool_address, start_block=startblock-1, end_block=endblock+1)
        print(f"Swaps Found (Prior to Merge): {swaps_df.shape}")
        print(f"Found {swaps_tokentx_df.shape[0]} Token Tx between {swaps_df['timeStamp'].iloc[0]} and {swaps_df['timeStamp'].iloc[-1]}")
        swaps_df = pd.merge(swaps_df.drop(labels=['gasUsed'],axis=1), swaps_tokentx_df[['timeStamp','blockNumber','transactionHash','gasUsed']], on=['timeStamp','blockNumber','transactionHash'], how='left')
        print(f"Swaps Found (After Merge): {swaps_df.shape}")   

    if data_path: 
        swaps_df.to_csv(f'{data_path}/{pool_address}/pool_id_{pool_address}_swap_final.csv')
 
    return swaps_df[['transactionHash', 'datetime', 'timeStamp', 'sqrtPriceX96',
                'blockNumber', 'gasPrice', 'gasUsed', 'tick', 'amount0', 'amount1',
                'liquidity']]


if __name__ == "__main__":
    print("Not Implemented")
    
    