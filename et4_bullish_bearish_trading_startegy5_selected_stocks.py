# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:15:12 2024

Author: Saarit
"""

# ================================
# Import Necessary Modules
# ================================

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time

import pandas as pd
import numpy as np
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException, KiteException
from requests.exceptions import HTTPError

# Import the shares list from et4_filtered_stocks_market_cap.py
from et4_filtered_stocks_market_cap1 import shares  # 'shares' should be a list of stock tickers

# ================================
# Setup Logging
# ================================

# Define the correct working directory path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)  # Change the current working directory to 'cwd'

# Set up logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO to capture informational messages and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the format of log messages
    handlers=[
        logging.FileHandler("trading_script.log"),  # Log messages will be written to 'trading_script.log' file
        logging.StreamHandler()  # Log messages will also be printed to the console
    ]
)

# Test logging initialization by logging an info message
logging.info("Logging has been initialized successfully.")  # Log an informational message
print("Logging initialized and script execution started.")  # Print a message to the console

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

# Create a semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(2)  # Limits to 2 concurrent API calls; adjust as needed

# ================================
# Define Helper Functions
# ================================


def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day.
    If today is a trading day (even if the market is currently closed), return today.
    If today is not a trading day (weekend or holiday), return the previous trading day.

    Parameters:
        reference_date (date, optional): The date to start checking from. Defaults to today.

    Returns:
        date: The last trading day.
    """
    if reference_date is None:
        reference_date = datetime.now().date()  # Start from today if no reference_date provided

    logging.info(f"Checking for the last trading day. Reference date: {reference_date}.")
    print(f"Reference Date: {reference_date}")

    # If today is a weekday and not a holiday, it's the last trading day
    if reference_date.weekday() < 5 and reference_date not in market_holidays:
        last_trading_day = reference_date
    else:
        # Go back to the previous trading day
        last_trading_day = reference_date - timedelta(days=1)
        while last_trading_day.weekday() >= 5 or last_trading_day in market_holidays:
            last_trading_day -= timedelta(days=1)

    logging.info(f"Last trading day determined as: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")
    return last_trading_day

def clear_cache():
    """
    Clears the cache directory by deleting all files in it.
    """
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)  # Full path of the file
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Delete the file
                    logging.info(f"Deleted cache file: {file_path}")  # Log file deletion
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")  # Log any errors during deletion
        logging.info("Cache has been cleared.")  # Log successful cache clearance
    else:
        logging.info("Cache directory does not exist.")  # Log if cache directory is absent

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

def is_cache_up_to_date(combined_data, to_date_intra, required_threshold=timedelta(minutes=5)):
    """
    Check if the cached data covers up to 'to_date_intra' within a required threshold.

    Parameters:
        combined_data (pd.DataFrame): The combined intraday data.
        to_date_intra (datetime): The end datetime for intraday data.
        required_threshold (timedelta): The acceptable difference between the latest cached data and 'to_date_intra'.

    Returns:
        bool: True if cache is up-to-date, False otherwise.
    """
    if combined_data.empty:
        return False  # Return False if there's no cached data
    last_cached_time = combined_data['date'].max()  # Latest timestamp in the cached data
    current_time = to_date_intra  # Current time to compare against
    return (current_time - last_cached_time) <= required_threshold  # Check if within threshold

def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=10)):
    """
    Check if the cache file is fresh based on the freshness_threshold.

    Parameters:
        cache_file (str): Path to the cache file.
        freshness_threshold (timedelta): Time delta to determine freshness.

    Returns:
        bool: True if cache is fresh, False otherwise.
    """
    if not os.path.exists(cache_file):
        logging.debug(f"Cache file {cache_file} does not exist.")  # Log if cache file is missing
        return False  # Return False if cache file doesn't exist
    
    # Get the last modified time of the cache file in UTC
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)  # Last modified time
    current_time = datetime.now(pytz.UTC)  # Current UTC time
    
    # Calculate freshness
    is_fresh = (current_time - cache_mtime) < freshness_threshold  # Determine if cache is fresh
    logging.debug(f"Cache freshness for {cache_file}: {is_fresh} (Last Modified: {cache_mtime}, Current Time: {current_time})")
    return is_fresh  # Return the freshness status

def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Parameters:
        series (pd.Series): Series of closing prices.
        window (int, optional): Number of periods to use for RSI calculation. Defaults to 14.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()  # Calculate price differences between consecutive periods
    gain = delta.clip(lower=0).rolling(window).mean()  # Calculate average gains over the window
    loss = (-delta.clip(upper=0)).rolling(window).mean()  # Calculate average losses over the window
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi.fillna(0)  # Replace NaN values with 0 to handle initial periods

def get_required_from_date(last_trading_day, required_trading_days, buffer_days=10):
    """
    Calculate the from_date required to fetch at least 'required_trading_days' trading days.
    
    Parameters:
        last_trading_day (date): The last trading day.
        required_trading_days (int): Number of trading days required.
        buffer_days (int): Additional days to account for non-trading days.
    
    Returns:
        date: The calculated from_date.
    """
    # Start with a buffer to ensure we cover non-trading days
    from_date = last_trading_day - timedelta(days=required_trading_days + buffer_days)
    
    # Generate business days between from_date and last_trading_day
    potential_dates = pd.bdate_range(from_date, last_trading_day, freq='C', holidays=market_holidays)
    
    # If potential_dates are sufficient, return
    if len(potential_dates) >= required_trading_days:
        return from_date  # Removed .date()
    
    # Otherwise, increase buffer_days and recalculate
    while len(potential_dates) < required_trading_days:
        buffer_days += 5  # Increment buffer
        from_date = last_trading_day - timedelta(days=required_trading_days + buffer_days)
        potential_dates = pd.bdate_range(from_date, last_trading_day, freq='C', holidays=market_holidays)
    
    return from_date  # Removed .date()





