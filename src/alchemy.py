import requests
import time
import os

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from web3 import Web3

# Add environmental variable ALCHEMY_API_KEY
ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY')
ALCHEMY_URL = f'https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}'

w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))

def fetch_latest_block(url):
    # Request payload
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_blockNumber",
        "params": [],
        "id": 1
    }
    
    # Make the request
    response = requests.post(ALCHEMY_URL, json=payload)
    
    # Handle the response
    if response.status_code == 200:
        result = response.json()
        if "result" in result:
            latest_block = int(result['result'], 16)  # Convert hex to int
            print(f"Latest Block Number: {latest_block}")
        else:
            print(f"Unexpected response format: {result}")
    else:
        print(f"Failed to fetch the latest block. Status code: {response.status_code}")
    
    return latest_block

def fetch_block_range(url, pool_address, latest_block, nblocks):
    
    # The topic for the Swap event in Uniswap V3 (Swap event signature)
    # https://www.4byte.directory/event-signatures/?bytes_signature=0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67
    swap_event_topic = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'

    #
    # Parameters for the API requestv - fetch blocks in the range that include swap events in any transaction
    # i.e. you need to filter out other transactions if you don't care about them...
    #
    # You can see the example of this call here: 
    # https://docs.alchemy.com/reference/eth-getlogs
    params = {
        'jsonrpc': '2.0',
        'method': 'eth_getLogs',
        'params': [{
            'fromBlock': str(hex(latest_block-nblocks)),
            'toBlock': str(hex(latest_block)),
            'address': pool_address,
            'topics': [swap_event_topic]
        }],
        'id': 1
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=params)
        data = response.json()
        #print(f"Fetched: {len(data['result'])} Transactions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

    
    # filter out transactions that are not a SWAP event.
    # final_data = [d for d in data['result'] if swap_event_topic in d['topics']]
    
    #print(f"\nFiltered out: {len(data['result'])-len(final_data)} transactions out of {len(data['result'])}.")
    #print(data)
    
    return data['result']

def decode_block_data(swap_tx_data, block_number,transaction_hash,datatype_list,verbose=False):
    """
    
    # Call for a single block
    print(f"Block Number: {block_number}")
    swap_tx_data = pool1_tx_data[0]['data']
    block_number = int(pool1_tx_data[0]['blockNumber'],16)
    block_data = decode_block_data(swap_tx_data,block_number,verbose=True)
    
    """
    
    ETH_RES  = 1e18
    USDC_RES = 1e6
    amount0, amount1, sqrtPriceX96, liquidity, tick = w3.eth.codec.decode(datatype_list,bytes.fromhex(swap_tx_data[2:]))
    
    # Convert and Print transaction details
    eth_price_usd = (((sqrtPriceX96 / 2**96)**2) / 1e12)**-1

    usdc_amount0 = amount0 / USDC_RES
    eth_amount1 = amount1 / ETH_RES

    block = w3.eth.get_block(block_number)
    timestamp = block['timestamp']
    #datetime_object = datetime.utcfromtimestamp(timestamp)
    datetime_object = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    if verbose:
        print(f"ETH Price: ${eth_price_usd:.2f}")
        print(f"USDC Tokens: ${usdc_amount0:.2f}")
        print(f"ETH Tokens: ${eth_amount1:.2f}")
        print(f"Liquidity: {liquidity}")
        print(f"Time Stamp: {datetime_object}")
    
    return {
        'transaction_hash':transaction_hash,
        'timestamp':datetime_object,
        'sqrtPriceX96':sqrtPriceX96,
        'tick':tick,
        'eth_price_usd':eth_price_usd,
        'usdc_amount0': usdc_amount0,
        'eth_amount1': eth_amount1,
        'liquidity':liquidity,
        'block_number':block_number,
    }

def fetch_swap_pool(url, pool_address, nblocks,latest_block_number=0):
    """
    Return: DataFrame with df.columns = ['transaction_hash', 'timestamp', 'sqrtPriceX96', 'tick',
       'eth_price_usd', 'usdc_amount0', 'eth_amount1', 'liquidity',
       'block_number']
    """
    # If at first, the request fails, sleep and try again.

    # Est. Requests: 1 API request Per function call (if needed)
    while latest_block_number==0:
        time.sleep(1)
        print('.',end='',flush=True)
        latest_block_number = fetch_latest_block_v2(url)
    
    print(f"")
    print(f"Latest Block: {latest_block_number}")
    start = time.time()
    # Now that I know the latest block number I can pull pool transactions by block range.
    # Est. Requests: 1 API request Per function call
    pool_tx_rawdata = fetch_block_range(url, pool_address, latest_block_number, nblocks)
    ntxs_found = len(pool_tx_rawdata)
    finish = time.time()
    print(f"Found {ntxs_found} Swap Transactions from {latest_block_number-nblocks} to {latest_block_number}")
    print(f"Time to fetch blocks: {finish-start}")

    
    # Parse the raw data, decode blocks.  Each call the decode_block_data has one call in it, 
    # to fetch the timestamp, so beware rate limit gods.
    # 
    # Est. Requests: ntxs_found requests per decode function call
    # Uniswap V3 Swap event ABI
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
    
    start = time.time()
    raw_data = []
    for i, d in enumerate(pool_tx_rawdata):
        block = int(d['blockNumber'],16)
        data = d['data']
        transaction_hash=d['transactionHash']
        raw_data.append(decode_block_data(data,block,transaction_hash,datatype_list))
        #if (i % 100)==0: print(f"{i}, ",end='')
    
    pool_tx_data = pd.DataFrame(raw_data)
        
    finish = time.time()
    #print(f"Time to create Dataframe: {finish-start}")
    
    return pool_tx_data, ntxs_found, latest_block_number

# Based on trial and error.  There is a 150MB limit on the request.  
# Larger than 700 was bad, but 300 seems to be working now.
MAX_BLOCKS = 300

def get_blocks_in_chunks(df, chunk_size=MAX_BLOCKS):
    blocks_list = []
    for i in range(0, len(df), chunk_size):
        # Get a chunk of 'blocks' column
        chunk = df['block_number'].iloc[i:i + chunk_size].tolist()
        blocks_list.append(chunk)
    return blocks_list
    
class TooManyRequestsError(Exception):
    pass

def fetch_with_retries(url, batch_request, max_retries=5, initial_wait=1, max_wait=60, factor=2):
    attempt = 0
    wait_time = initial_wait

    while attempt < max_retries:
        try:
            response = requests.post(url, json=batch_request)
            response.raise_for_status()  # Raise an error if the request was unsuccessful
            json_response = response.json()

            # If we receive a 429 error (Too Many Requests), log an error and retry
            if 'error' in json_response and json_response['error'].get('code') == 429:
                print("HTTP error 429: Too Many Requests, retrying...")
                raise TooManyRequestsError("HTTP error 429: Too Many Requests, retrying...")

            # Otherwise, return the response JSON
            return json_response

        except TooManyRequestsError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(wait_time)
            wait_time = min(wait_time * factor, max_wait)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

    raise Exception("Failed to fetch data after multiple retries")


def fetch_block_details(pool_df,include_all_objects=True):

    # Get the blocks in chunks of MAX_BLOCKS
    all_blockdata = []
    block_number_chunks = get_blocks_in_chunks(pool_df)
    for block_number_chunk in block_number_chunks:
        start = time.time()
        # Prepare batch request
        batch_request = [
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "eth_getBlockByNumber",
                "params": [str(hex(block_number)), include_all_objects]  # True to include full transaction objects
            } for i, block_number in enumerate(block_number_chunk)
        ]
    
        # Fetch the result with retries
        try:
            blockdata_rsp = fetch_with_retries(ALCHEMY_URL, batch_request)
        except Exception as e:
            print(f"Request failed after multiple retries: {e}")
        finish = time.time()
        print(f"Time to process {len(block_number_chunk)} blocks: {finish-start}")
    
    
        #display(pool0_merge.shape
        all_blockdata.append(blockdata_rsp)

    #
    # Take the MAX_BLOCKS chunks and flatten them... 
    #
    flat_pool_blocks = [pool_block for pool_block_chunk in all_blockdata for pool_block in pool_block_chunk]
    
    # Parse binary data
    pool_block_details = parse_block_details(flat_pool_blocks)
    
    return pool_block_details

def parse_block_details(pool_blocks,verbose=False):
    # Fetch all transactions.
    pool_block_details = []
    for block_details in pool_blocks:
        if 'result' in block_details:
            txs_details = block_details['result']['transactions']
            block_number = block_details['result']['number']
            timestamp = int(block_details['result']['timestamp'],16)
            #datetime_object = datetime.utcfromtimestamp(timestamp)
            datetime_object = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            for tx in txs_details:
                gas_price = int(tx['gasPrice'], 16) / 1e9  # Convert from wei to Gwei
                gas_used = int(tx['gas'], 16)
                tx_hash  = tx['hash']
                sender = tx['from']
                receiver = tx['to']
                tx_data = {
                    'timestamp': datetime_object,
                    'transaction_hash': tx_hash,
                    'block_number': int(block_number,16),
                    'gas_price': gas_price,
                    'gas_used': gas_used,
                    'sender': sender,
                    'recipient':receiver
                }
                if verbose: 
                    print(f"Timestamp: {datetime_object}")
                    print(f"Transaction Hash: {tx_hash}")
                    print(f"Gas Price (Gwei): {gas_price}")
                    print(f"Gas Used: {gas_used}")
                pool_block_details.append(tx_data)
                
        else:
            print(f"Error fetching data: {block_details}")
    return pd.DataFrame(pool_block_details)

def create_pool_df(pool_swap_df,transaction_rate,t0_res=18, t1_res=18, num=0):
    """
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
    pool_swap_df[f'p{num}.sender'] = pool_swap_df['sender']
    pool_swap_df[f'p{num}.recipient'] = pool_swap_df['recipient']
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
                        f'p{num}.sender',
                        f'p{num}.recipient',
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
    From the original data dictionary...
    
    pool_swap_df['time']
    pool_swap_df['timestamp']
    pool_swap_df['amount0']
    ['amount1']
    pool_swap_df[f'p{num}.t0_token'] = pool_swap_df['pool.token0.name']
    pool_swap_df[f'p{num}.t1_token'] = pool_swap_df['pool.token1.name']
    pool_swap_df[f'p{num}.tick'] = pool_swap_df['tick']
    pool_swap_df[f'p{num}.sqrtPriceX96'] = pool_swap_df['sqrtPriceX96'].astype(float)
    pool_swap_df[f'p{num}.gasUsed'] = pool_swap_df['transaction.gasUsed']
    pool_swap_df[f'p{num}.gasPrice'] = pool_swap_df['transaction.gasPrice']
    pool_swap_df[f'p{num}.blockNumber'] = pool_swap_df['transaction.blockNumber']
    pool_swap_df[f'p{num}.sender'] = pool_swap_df['sender']
    pool_swap_df[f'p{num}.recipient'] = pool_swap_df['recipient']
    pool_swap_df[f'p{num}.transaction_id'] = pool_swap_df['transaction.id']
    """
    p0.columns = ['transaction.id','time','sqrtPriceX96','tick','eth_price_usd','amount0','amount1','liquidity','transaction.blockNumber','transaction.gasPrice','transaction.gasUsed','sender','recipient']

    # add other columns.
    p0['timestamp'] = p0['time'].apply(lambda x: x.timestamp())
    p0['pool.token0.name'] = 'USDC'
    p0['pool.token1.name'] = 'WETH'

    
    p1.columns = ['transaction.id','time','sqrtPriceX96','tick','eth_price_usd','amount0','amount1','liquidity','transaction.blockNumber','transaction.gasPrice','transaction.gasUsed','sender','recipient']
    # add other columns.
    p1['timestamp'] = p1['time'].apply(lambda x: x.timestamp())
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
    both_pools['time'] = pd.to_datetime(both_pools['time'])
    
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

def alchemy_request(url, pool_address, blocks_to_look_back=40, latest_block=0):
   
    #
    # Look for Swap Events in the duration from "latest block" to "latest block - blocks_to_look_back"
    #
    # pool_df.columns = ['transaction_hash', 'timestamp', 'sqrtPriceX96', 'tick',
    #   'eth_price_usd', 'usdc_amount0', 'eth_amount1', 'liquidity',
    #   'block_number']
    pool_df, npool_txs, _ = fetch_swap_pool(url, 
                                            pool_address, 
                                            blocks_to_look_back, 
                                            latest_block)
    
    print("")
    if npool_txs == 0:
        print("Pool: Did not find any transactions in the window specified.")
        return None
        
    else:
        print(f"Latest Block Timestamp: {pool_df['timestamp'].max()-pd.Timedelta(hours=4)}")
        pool_total_minutes = (pool_df['timestamp'].max()-pool_df['timestamp'].min()).total_seconds() / 60
        print(f"Duration of Pool Data (in minutes): {pool_total_minutes} minutes")

    #
    # Get detailed block data from block numbers...
    #
    # pool_block_details.columns = ['timestamp', 'transaction_hash', 'block_number', 'gas_price',
    #   'gas_used', 'sender', 'recipient']
    pool_block_details = fetch_block_details(pool_df)

    # merge 
    pool_final = pool_df.merge(pool_block_details,
                                 how='left',
                                 on=['transaction_hash','timestamp','block_number']).drop_duplicates()
    if (pool_final['gas_price'].isna().sum())==0 and (pool_df.shape[0]==pool_final.shape[0]):
        print("All transactions merged Successfully")
    else:
        print("Issue with creating final dataframe with gas prices.")
        return None

    return pool_final

if __name__ == "__main__":
    
    POOL0_ADDRESS="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" # USDC / WETH (0.05%) 
    POOL0_TXN_FEE = 0.0005
    POOL1_ADDRESS="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8" # USDC / WETH (0.3%)
    POOl1_TXN_FEE = 0.003

    # Ethereum is estimated to have a transaction speed of one block per 12 seconds (5 times a minute).
    LAST_60MIN_BLOCKS= int(np.floor(5 * 60))

    # Use the same latest block for both requests...
    # better to do the least frequent of the two second
    latest_block_number = fetch_latest_block(ALCHEMY_URL)
    
    pool0_df = alchemy_request(ALCHEMY_URL, 
                               POOL0_ADDRESS, 
                               blocks_to_look_back=LAST_60MIN_BLOCKS,
                               latest_block=latest_block_number)    
    pool1_df = alchemy_request(ALCHEMY_URL, 
                               POOL1_ADDRESS, 
                               blocks_to_look_back=LAST_60MIN_BLOCKS,
                               latest_block=latest_block_number)    
    
    both_pools = merge_pool_data_v2(pool0_df, POOL0_TXN_FEE, pool1_df, POOl1_TXN_FEE)
    
    