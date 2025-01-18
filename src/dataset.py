import os
import pytz
import shutil
from datetime import datetime, timedelta

# local file
import fetch

# Fetch the API key from environment variables
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def delete_batch_files(directory):
    """
    Deletes files in the specified directory that contain the given word in their name.
    
    Args:
    - directory (str): Path to the directory to search for files.
    """
    word = "batch"
    file_cnt = 0
    print("Deleting Batch Files...")
    try:
        # List all files in the directory
        for filename in os.listdir(directory):
            # Check if the word is in the filename
            if word in filename:
                file_path = os.path.join(directory, filename)
                # Check if it's a file (not a directory)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    #print(f"Deleted file: {file_path}")
                    print(".",end='')
                    file_cnt += 1 
        print(f"Deleted {file_cnt} files.")

    except Exception as e:
        print(f"An error occurred: {e}")

def remove_checkpoint_file(file):
    # Delete the file if it exists
    if os.path.isfile(file):
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except Exception as e:
            print(f"Error deleting file: {file}. Error: {e}")

def remove_data_directory(file_dir):
    # Delete the directory if it exists
    if os.path.isdir(file_dir):
        try:
            shutil.rmtree(file_dir)
            print(f"Deleted directory: {file_dir}")
        except Exception as e:
            print(f"Error deleting directory: {file_dir}. Error: {e}")

if __name__ == "__main__":

    pool0_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    pool0_txn_fee = 0.0005

    pool1_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    pool1_txn_fee = 0.003

    # Creates datetime objects for the fetch
    #                     YYYY  MM  DD  HH  MM  SS
    new_date = datetime(*(2025,  1, 14,  2,  0,  0), tzinfo=pytz.UTC)
    old_date = new_date - timedelta(weeks=26)
    print(f"Attempting to fetch data from {old_date} to {new_date}")

    base_path = "data/"
    pool0_data_path = f'{base_path}/'
    pool0_checkpoint_file = f'{pool0_data_path}/checkpoint.pool0.json'
    pool1_data_path = f'{base_path}/'
    pool1_checkpoint_file = f'{pool1_data_path}/checkpoint.pool1.json'

    clean_install = True

    if clean_install:
        remove_data_directory(base_path)
    
    print("--------------------")
    print("Creating Pool 0")
    print("--------------------")
    p0 = fetch.thegraph_request(thegraph_api_key=GRAPH_API_KEY,
                           etherscan_api_key=ETHERSCAN_API_KEY,
                           pool_address=pool0_address,
                           data_path=pool0_data_path,
                           checkpoint_file=pool0_checkpoint_file,
                           old_date=old_date,
                           new_date=new_date)
    
    
    print("--------------------")
    print("Creating Pool 1")
    print("--------------------")
    p1 = fetch.thegraph_request(thegraph_api_key=GRAPH_API_KEY,
                        etherscan_api_key=ETHERSCAN_API_KEY,
                        pool_address=pool1_address,
                        data_path=pool1_data_path,
                        checkpoint_file=pool1_checkpoint_file,                        
                        old_date=old_date,
                        new_date=new_date)

    # detele the batch files...
    for dataset_directory in [f"{pool0_data_path}", f"{pool1_data_path}"]:
        print(f"Deleting the batch files in {dataset_directory}")
        delete_batch_files(dataset_directory)

    # detele the batch files...
    for checkpoint_file in [f"{pool0_checkpoint_file}", f"{pool1_checkpoint_file}"]:
        print(f"Deleting checkpoint files in {checkpoint_file}")
        remove_checkpoint_file(checkpoint_file)

    