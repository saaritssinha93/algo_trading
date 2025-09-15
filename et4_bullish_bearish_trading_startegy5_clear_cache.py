# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:36:05 2024

@author: Saarit
"""

# ================================
# Import Necessary Modules
# ================================

import os
import logging
import threading
import shutil  # Import shutil for high-level file operations
from datetime import datetime as datetime


# ================================
# Setup Logging
# ================================

# Define the correct working directory path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)  # Change the current working directory to 'cwd'

# Set up logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.FileHandler("trading_script_cache_clear.log"),  # Log messages will be written to 'trading_script.log'
        logging.StreamHandler()  # Log messages will also be printed to the console
    ]
)

# Test logging initialization by logging an info message
logging.info("Logging has been initialized successfully.")
print("Logging initialized and script execution started.")  # Print to console

# ================================
# Define Global Variables
# ================================

# Known market holidays for 2024
market_holidays = [
    datetime(2024, 1, 26).date(),  # Republic Day
    datetime(2024, 3, 8).date(),   # Mahashivratri
    datetime(2024, 3, 25).date(),  # Holi
    datetime(2024, 3, 29).date(),  # Good Friday
    datetime(2024, 4, 11).date(),  # Id-Ul-Fitr (Ramadan Eid)
    datetime(2024, 4, 17).date(),  # Shri Ram Navami
    datetime(2024, 5, 1).date(),   # Maharashtra Day
    datetime(2024, 6, 17).date(),  # Bakri Id
    datetime(2024, 7, 17).date(),  # Moharram
    datetime(2024, 8, 15).date(),  # Independence Day
    datetime(2024, 10, 2).date(),  # Gandhi Jayanti
    datetime(2024, 11, 1).date(),  # Diwali Laxmi Pujan
    datetime(2024, 11, 15).date(), # Gurunanak Jayanti
    datetime(2024, 11, 20).date(), # Maha elections
    datetime(2024, 12, 25).date(), # Christmas
]

# Directory for storing cached data
CACHE_DIR = "data_cache"  # Name of the cache directory
os.makedirs(CACHE_DIR, exist_ok=True)  # Create the cache directory if it doesn't exist

# Directory for storing main indicators data
MAIN_INDICATORS_DIR = "main_indicators"  # Name of the main indicators directory
os.makedirs(MAIN_INDICATORS_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Create a semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(2)  # Limits to 2 concurrent API calls; adjust as needed

# ================================
# Define Helper Functions
# ================================

def clear_cache():
    """
    Clears the cache directories by deleting all files in them.
    Specifically targets both 'data_cache' and 'main_indicators' directories.
    """
    # List of directories to clear
    directories_to_clear = [CACHE_DIR, MAIN_INDICATORS_DIR]
    
    for directory in directories_to_clear:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete the file or link
                        logging.info(f"Deleted cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete the directory and its contents
                        logging.info(f"Deleted cache directory and its contents: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}. Reason: {e}")
            logging.info(f"Cache has been cleared for directory: {directory}")
        else:
            logging.info(f"Cache directory does not exist: {directory}")

def cache_file_path(ticker, data_type):
    """
    Generate a unique file path for caching data, differentiating between historical and intraday data.
    
    Parameters:
        ticker (str): The trading symbol of the stock.
        data_type (str): Type of data ('historical' or 'intraday').
    
    Returns:
        str: The file path for the cache file.
    """
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")  # e.g., 'data_cache/GLAND_historical.csv'

# Clear caches from both 'data_cache' and 'main_indicators' directories
clear_cache()
