# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:25:53 2024

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:58:21 2024

Updated by: [Your Name]

Description:
This script fetches intraday stock data, calculates technical indicators,
detects entry signals based on defined trend conditions, and manages data
storage and logging. All references to "Time" have been standardized to "date"
for consistency and clarity.
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
import pytz
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
import traceback


# Import the shares list from et1_select_stocklist.py
from et1_stock_tickers import shares  # 'shares' should be a list of stock tickers

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
        logging.FileHandler("trading_script.log"),  # Log messages will be written to 'trading_script.log'
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

# Create a semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(2)  # Limits to 2 concurrent API calls; adjust as needed

# ================================
# Define Helper Functions
# ================================

def clear_cache():
    """
    Clears the cache directory by deleting all files in it.
    """
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Delete the file
                    logging.info(f"Deleted cache file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
        logging.info("Cache has been cleared.")
    else:
        logging.info("Cache directory does not exist.")

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
        return False
    last_cached_time = combined_data['date'].max()
    current_time = to_date_intra
    return (current_time - last_cached_time) <= required_threshold

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
        logging.debug(f"Cache file {cache_file} does not exist.")
        return False  # Cache file doesn't exist
    
    # Get the last modified time of the cache file in UTC
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)
    current_time = datetime.now(pytz.UTC)
    
    # Calculate freshness
    is_fresh = (current_time - cache_mtime) < freshness_threshold
    logging.debug(f"Cache freshness for {cache_file}: {is_fresh} (Last Modified: {cache_mtime}, Current Time: {current_time})")
    return is_fresh

def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Parameters:
        series (pd.Series): Series of closing prices.
        window (int, optional): Number of periods to use for RSI calculation. Defaults to 14.

    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()  # Calculate price differences
    gain = delta.clip(lower=0).rolling(window).mean()  # Calculate average gains
    loss = (-delta.clip(upper=0)).rolling(window).mean()  # Calculate average losses
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi.fillna(0)  # Replace NaN values with 0

def calculate_atr(data, window=14):
    """
    Calculate the Average True Range (ATR).

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        window (int, optional): Window size for ATR calculation. Defaults to 14.

    Returns:
        pd.Series: ATR values.
    """
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr.fillna(0)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate the MACD and Signal line.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' column.
        fast (int, optional): Fast EMA period. Defaults to 12.
        slow (int, optional): Slow EMA period. Defaults to 26.
        signal (int, optional): Signal line period. Defaults to 9.

    Returns:
        tuple: MACD and Signal line Series.
    """
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.fillna(0), signal_line.fillna(0)

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' column.
        window (int, optional): Window size for SMA. Defaults to 20.
        num_std (int, optional): Number of standard deviations for the bands. Defaults to 2.

    Returns:
        tuple: Upper and Lower Bollinger Bands Series.
    """
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band.fillna(0), lower_band.fillna(0)

def calculate_stochastic(data, window=14):
    """
    Calculate Stochastic Oscillator.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        window (int, optional): Window size for Stochastic calculation. Defaults to 14.

    Returns:
        pd.Series: Stochastic Oscillator values.
    """
    low_min = data['low'].rolling(window=window).min()
    high_max = data['high'].rolling(window=window).max()
    stochastic = 100 * (data['close'] - low_min) / (high_max - low_min)
    return stochastic.fillna(0)

# ================================
# Kite Connect Session Setup
# ================================

def setup_kite_session():
    """
    Initialize Kite Connect session using API key and access token.

    Returns:
        KiteConnect: An instance of KiteConnect with an active session.

    Raises:
        FileNotFoundError: If 'access_token.txt' or 'api_key.txt' is not found.
        Exception: For other errors during session setup.
    """
    try:
        # Read access token from file
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()

        # Read API key from file
        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        # Initialize KiteConnect with API key
        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)  # Set access token for the session
        logging.info("Kite session established successfully.")
        print("Kite Connect session initialized successfully.")
        return kite  # Return the KiteConnect instance

    except FileNotFoundError as e:
        # Log and print error if files are not found
        logging.error(f"File not found: {e}")
        print(f"Error: {e}")
        raise  # Re-raise the exception to halt execution
    except Exception as e:
        # Log and print any other exceptions
        logging.error(f"Error setting up Kite session: {e}")
        print(f"Error: {e}")
        raise  # Re-raise the exception to halt execution

# Setup Kite session by initializing the KiteConnect instance
kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    """
    Fetch instrument tokens for the provided stock symbols.

    Parameters:
        stocks (list): List of stock ticker symbols.

    Returns:
        dict: Dictionary mapping ticker symbols to their instrument tokens.
    """
    try:
        logging.info("Fetching tokens for the provided stocks.")
        instrument_dump = kite.instruments("NSE")  # Fetch all NSE instruments
        instrument_df = pd.DataFrame(instrument_dump)  # Convert to DataFrame
        # Filter instruments that are in the 'stocks' list and select relevant columns
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][['tradingsymbol', 'instrument_token']]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))  # Return as dictionary
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        raise  # Re-raise the exception to halt execution

# Read tokens for shares using the list from et1_select_stocklist.py
shares_tokens = get_tokens_for_stocks(shares)

# ================================
# Data Fetching Functions
# ================================

def fetch_historical_data(ticker, token, from_date, to_date, interval="15minute", use_cached_data=True):
    """
    Fetch historical or intraday data for a given ticker, utilizing cache if available.

    Parameters:
        ticker (str): The trading symbol of the stock.
        token (int): The instrument token for the stock.
        from_date (datetime): Start datetime for data fetching.
        to_date (datetime): End datetime for data fetching.
        interval (str): Data interval (default is "15minute").
        use_cached_data (bool): Whether to use cached data if available.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """
    max_retries = 5  # Maximum number of retry attempts
    delay = 1  # Initial delay in seconds for retrying

    today = datetime.now().date()  # Current date
    if to_date.date() < today:
        # Fetching historical data
        cache_file = cache_file_path(ticker, 'historical')
    else:
        # Fetching intraday data
        cache_file = cache_file_path(ticker, 'intraday')

    # Check if cached data exists, is fresh, and caching is enabled
    if use_cached_data and os.path.exists(cache_file) and is_cache_fresh(cache_file):
        logging.info(f"Loading cached data for {ticker}")
        cached_data = pd.read_csv(cache_file, parse_dates=['date'])  # Read cached data
        # Ensure 'date' column is timezone-aware
        cached_data['date'] = pd.to_datetime(cached_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
        return cached_data  # Return cached data

    else:
        # Fetch data from Kite Connect API and cache it
        logging.info(f"Fetching data for {ticker}")
        for attempt in range(max_retries):
            with api_semaphore:  # Ensure limited concurrent API calls
                try:
                    data = pd.DataFrame(
                        kite.historical_data(
                            instrument_token=token,
                            from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                            to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                            interval=interval
                        )
                    )
                    if not data.empty:
                        # Convert 'date' to timezone-aware datetime in IST
                        data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert('Asia/Kolkata')               
                        #data.to_csv(cache_file, index=False)  # Save fetched data to cache
                        #logging.info(f"Cached data for {ticker}")
                    time.sleep(0.2)  # Brief pause to respect API rate limits
                    return data  # Return fetched data
                except Exception as e:
                    if "Too many requests" in str(e):
                        # Handle API rate limiting with exponential backoff
                        logging.warning(f"Rate limit exceeded for {ticker}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Double the delay for next retry
                    else:
                        # Log other exceptions and abort retries
                        logging.error(f"Error fetching data for {ticker}: {e}")
                        return pd.DataFrame()
        # If all retries fail, log the failure
        logging.error(f"Failed to fetch data for {ticker} after {max_retries} retries.")
        return pd.DataFrame()
'''
def fetch_intraday_data_partial(ticker, token, from_date_intra, to_date_intra, interval="15minute", cache_dir=CACHE_DIR, freshness_threshold=timedelta(minutes=10)):
    """
    Fetch intraday data for a ticker with partial caching.
    
    Parameters:
        ticker (str): The trading symbol of the stock.
        token (int): The instrument token for the stock.
        from_date_intra (datetime): Start datetime for intraday data.
        to_date_intra (datetime): End datetime for intraday data.
        interval (str): Data interval (default is "15minute").
        cache_dir (str): Directory where cached data is stored.
        freshness_threshold (timedelta): Threshold to determine cache freshness.
    
    Returns:
        pd.DataFrame: Combined historical and intraday data.
    """
    import os
    import pandas as pd
    import logging
    import pytz
    from datetime import datetime, timedelta

    # Define cache file path for intraday data
    cache_file = cache_file_path(ticker, 'intraday')
    
    historical_cache_file = cache_file_path(ticker, 'historical')

    # Initialize empty DataFrame for combined data
    combined_data = pd.DataFrame()

    # Check if cached intraday data exists and is fresh
    cache_exists = os.path.exists(cache_file)
    cache_fresh = is_cache_fresh(cache_file, freshness_threshold) if cache_exists else False

    if cache_exists and cache_fresh:
        logging.info(f"Loading cached intraday data for {ticker}")
        cached_data = pd.read_csv(cache_file, parse_dates=['date'])  # Read cached intraday data
        cached_data['date'] = pd.to_datetime(cached_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')  # Ensure timezone
        combined_data = cached_data  # Assign to combined_data
        logging.debug(f"Loaded cached data shape: {combined_data.shape}")
    else:
        if cache_exists:
            logging.info(f"Cache exists but is stale for {ticker}")
            logging.debug(f"Cache freshness: {cache_fresh}")
        else:
            logging.info(f"No cached intraday data found for {ticker}")
        # Initialize combined_data with existing cache if any, to append new data
        if cache_exists:
            cached_data = pd.read_csv(cache_file, parse_dates=['date'])
            cached_data['date'] = pd.to_datetime(cached_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
            combined_data = cached_data
        else:
            combined_data = pd.DataFrame()

    # Determine the start time for fetching new data
    if not combined_data.empty:
        last_cached_time = combined_data['date'].max()  # Last timestamp in cached data
        # Set fetch_from to the next interval after last_cached_time to avoid overlaps
        fetch_from = last_cached_time + timedelta(minutes=15)
        logging.debug(f"Last cached time for {ticker}: {last_cached_time}")
    else:
        # If no cached data, fetch from the start of intraday period
        fetch_from = from_date_intra
        logging.debug(f"No cached data for {ticker}. Setting fetch_from to {fetch_from}")

    # If fetch_from is greater than or equal to to_date_intra, no new data to fetch
    if fetch_from >= to_date_intra:
        logging.info(f"Cache is up-to-date for {ticker}. No new intraday data to fetch.")
        return combined_data  # Return existing cached data

    # Fetch new intraday data from Kite Connect API
    logging.info(f"Fetching new intraday data for {ticker} from {fetch_from} to {to_date_intra}")
    max_retries = 5  # Maximum number of retry attempts
    delay = 1  # Initial delay in seconds for retrying

    for attempt in range(max_retries):
        with api_semaphore:  # Ensure limited concurrent API calls
            try:
                data = pd.DataFrame(
                    kite.historical_data(
                        instrument_token=token,
                        from_date=fetch_from.strftime("%Y-%m-%d %H:%M:%S"),
                        to_date=to_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
                        interval=interval
                    )
                )
                if not data.empty:
                    # Convert 'date' to timezone-aware datetime in IST
                    data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
                    
                    # Add indicators
                    data['RSI'] = calculate_rsi(data['close'])
                    data['ATR'] = calculate_atr(data)
                    data['MACD'], data['Signal_Line'] = calculate_macd(data)
                    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
                    data['Stochastic'] = calculate_stochastic(data)
                    
                    # Add VWAP
                    data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

                    
                    # Append new data to existing cached data
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
                    # Remove any potential duplicates based on 'date'
                    combined_data.drop_duplicates(subset='date', inplace=True)
                    # Sort the data by 'date' in ascending order
                    combined_data.sort_values(by='date', inplace=True)
                    # Save the updated cache with combined data
                    combined_data.to_csv(cache_file, index=False)
                    logging.info(f"Updated cached intraday data for {ticker}")
                    logging.debug(f"Combined data shape after fetching: {combined_data.shape}")
                else:
                    # Log if no new data was fetched
                    logging.warning(f"No new intraday data fetched for {ticker}")
                break  # Exit retry loop upon successful fetch
            except Exception as e:
                if "Too many requests" in str(e):
                    # Handle API rate limiting with exponential backoff
                    logging.warning(f"Rate limit exceeded for {ticker}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Double the delay for next retry
                else:
                    # Log other exceptions and abort retries
                    logging.error(f"Error fetching intraday data for {ticker}: {e}")
                    break  # Exit retry loop on non-rate limit errors

    return combined_data  # Return combined historical and intraday data
'''


# Define a custom exception for API call failures
class APICallError(Exception):
    pass

def make_api_call_with_retry(kite, token, from_date, to_date, interval, max_retries=5, backoff_factor=4):
    """
    Makes the API call to fetch historical data with retry logic.
    
    Parameters:
        kite: The KiteConnect instance.
        token (str): Instrument token.
        from_date (str): Start datetime in "YYYY-MM-DD HH:MM:SS" format.
        to_date (str): End datetime in "YYYY-MM-DD HH:MM:SS" format.
        interval (str): Interval for data (e.g., "15minute").
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (int): Factor by which the delay increases after each retry.
    
    Returns:
        pd.DataFrame: DataFrame containing historical data.
    
    Raises:
        APICallError: If all retry attempts fail.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            if not data:
                raise APICallError("Received empty data from API.")
            return pd.DataFrame(data)
        except Exception as e:
            attempt += 1
            wait_time = backoff_factor * (2 ** (attempt - 1))
            logging.error(f"API call failed on attempt {attempt}: {e}")
            logging.error(traceback.format_exc())
            if attempt < max_retries:
                logging.info(f"Retrying in {wait_time} seconds...")
                print(f"API call failed on attempt {attempt}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} retry attempts failed for API call.")
                print(f"All {max_retries} retry attempts failed for API call.")
                raise APICallError(f"API call failed after {max_retries} attempts.") from e

def fetch_intraday_data_with_historical(
    ticker, token, from_date_intra, to_date_intra, interval="15minute", 
    cache_dir=CACHE_DIR, freshness_threshold=timedelta(minutes=10)
):
    """
    Fetch intraday data for a ticker with historical data included for better indicator calculations.
    Implements retries with exponential backoff for API calls.
    
    Parameters:
        ticker (str): The trading symbol of the stock.
        token (str): The instrument token for the ticker.
        from_date_intra (datetime): Start datetime for intraday data.
        to_date_intra (datetime): End datetime for intraday data.
        interval (str): Interval for data (default "15minute").
        cache_dir (str): Directory path for caching data.
        freshness_threshold (timedelta): Time threshold to determine if cache is fresh.
    
    Returns:
        pd.DataFrame: Combined DataFrame with historical and intraday data, including indicators.
    """
    combined_cache_file = os.path.join(cache_dir, f"{ticker}_combined.csv")
    historical_cache_file = os.path.join(cache_dir, f"{ticker}_historical.csv")
    intraday_cache_file = os.path.join(cache_dir, f"{ticker}_intraday.csv")
    
    # Step 1: Check Combined Cache
    logging.info(f"Checking combined data cache for {ticker}.")
    combined_cache_exists = os.path.exists(combined_cache_file)
    combined_cache_fresh = is_cache_fresh(combined_cache_file, freshness_threshold) if combined_cache_exists else False

    if combined_cache_exists and combined_cache_fresh:
        logging.info(f"Loading cached combined data for {ticker}.")
        combined_data = pd.read_csv(combined_cache_file, parse_dates=['date'])
        combined_data['date'] = pd.to_datetime(combined_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
        return combined_data  # Return combined data directly if cache is fresh

    # Step 2: Fetch Historical Data
    historical_cache_exists = os.path.exists(historical_cache_file)
    historical_cache_fresh = is_cache_fresh(historical_cache_file, freshness_threshold) if historical_cache_exists else False

    if historical_cache_exists and historical_cache_fresh:
        logging.info(f"Loading cached historical data for {ticker}.")
        historical_data = pd.read_csv(historical_cache_file, parse_dates=['date'])
        historical_data['date'] = pd.to_datetime(historical_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
    else:
        logging.info(f"Fetching historical data for {ticker}.")
        from_date_hist = from_date_intra - timedelta(days=5)
        try:
            historical_data = make_api_call_with_retry(
                kite=kite,
                token=token,
                from_date=from_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=(from_date_intra - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval
            )
            historical_data['date'] = pd.to_datetime(historical_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
            historical_data.to_csv(historical_cache_file, index=False)
            logging.info(f"Saved historical data for {ticker} to cache.")
        except APICallError as e:
            logging.error(f"Failed to fetch historical data for {ticker}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure

    # Step 3: Fetch Intraday Data
    intraday_cache_exists = os.path.exists(intraday_cache_file)
    intraday_cache_fresh = is_cache_fresh(intraday_cache_file, freshness_threshold) if intraday_cache_exists else False

    if intraday_cache_exists and intraday_cache_fresh:
        logging.info(f"Loading cached intraday data for {ticker}.")
        intraday_data = pd.read_csv(intraday_cache_file, parse_dates=['date'])
        intraday_data['date'] = pd.to_datetime(intraday_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
    else:
        intraday_data = pd.DataFrame()

    if not intraday_data.empty:
        last_cached_time = intraday_data['date'].max()
        fetch_from = last_cached_time + timedelta(minutes=15)
    else:
        fetch_from = from_date_intra

    if fetch_from < to_date_intra:
        logging.info(f"Fetching new intraday data for {ticker} from {fetch_from} to {to_date_intra}.")
        try:
            new_intraday_data = make_api_call_with_retry(
                kite=kite,
                token=token,
                from_date=fetch_from.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval
            )
            if not new_intraday_data.empty:
                new_intraday_data['date'] = pd.to_datetime(new_intraday_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
                intraday_data = pd.concat([intraday_data, new_intraday_data]).drop_duplicates(subset='date')
                intraday_data.sort_values(by='date', inplace=True)
                intraday_data.to_csv(intraday_cache_file, index=False)
                logging.info(f"Updated intraday cache for {ticker}.")
        except APICallError as e:
            logging.error(f"Failed to fetch intraday data for {ticker}: {e}")

    # Step 4: Combine Historical and Intraday Data
    if historical_data.empty and intraday_data.empty:
        logging.warning(f"No data available for {ticker}. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Combining historical and intraday data for {ticker}.")
    combined_data = pd.concat([historical_data, intraday_data]).drop_duplicates(subset='date').sort_values(by='date')

    # Step 5: Calculate Indicators if Combined Data is Not Cached
    if not combined_data.empty:
        logging.info(f"Calculating indicators on combined data for {ticker}.")
        combined_data['RSI'] = calculate_rsi(combined_data['close'])
        combined_data['ATR'] = calculate_atr(combined_data)
        combined_data['MACD'], combined_data['Signal_Line'] = calculate_macd(combined_data)
        combined_data['Upper_Band'], combined_data['Lower_Band'] = calculate_bollinger_bands(combined_data)
        combined_data['Stochastic'] = calculate_stochastic(combined_data)
        combined_data['VWAP'] = (combined_data['close'] * combined_data['volume']).cumsum() / combined_data['volume'].cumsum()
        combined_data['20_SMA'] = combined_data['close'].rolling(window=20).mean()
        combined_data['Recent_High'] = combined_data['high'].rolling(window=5).max()
        combined_data['Recent_Low'] = combined_data['low'].rolling(window=5).min()

    # Save combined data to cache
    combined_data.to_csv(combined_cache_file, index=False)
    logging.info(f"Saved combined data with indicators for {ticker} to cache.")

    return combined_data


# ================================
# Market Status and Trading Day
# ================================

def is_market_open():
    """
    Check if the market is currently open based on the current time and known holidays.

    Returns:
        bool: True if market is open, False otherwise.
    """
    now = datetime.now(pytz.timezone('Asia/Kolkata'))  # Current time in IST
    current_time = now.time()  # Current time without date
    market_start = datetime_time(9, 15)  # Market opens at 9:15 AM
    market_end = datetime_time(15, 30)   # Market closes at 3:30 PM
    # Market is open if current time is between market_start and market_end and today is not a holiday
    return market_start <= current_time <= market_end and now.date() not in market_holidays

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

def sort_dataframe_by_date(df):
    """
    Sorts the DataFrame by the 'date' column in ascending order.

    Parameters:
        df (pd.DataFrame): The DataFrame to sort.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    if 'date' not in df.columns:
        logging.error("DataFrame does not contain a 'date' column.")
        raise KeyError("DataFrame does not contain a 'date' column.")
    
    # Ensure 'date' is in datetime format and timezone-aware
    df = normalize_time(df)
    
    # Sort by 'date' in ascending order
    df.sort_values(by='date', inplace=True)
    
    # Reset index after sorting
    df.reset_index(drop=True, inplace=True)
    
    logging.debug("DataFrame sorted by 'date' in ascending order.")
    return df


# ================================
# Price Action Entry Detection
# ================================

from selected_stocks import selected_stocks  # Import selected_stocks from the file

def find_price_action_entry(last_trading_day):
    india_tz = pytz.timezone('Asia/Kolkata')
    
    # Prepare historical and intraday time ranges
    from_date_hist, to_date_hist, from_date_intra, to_date_intra = prepare_time_ranges(
        last_trading_day, india_tz
    )

    logging.info(f"Fetching historical data from {from_date_hist} to {to_date_hist}")
    logging.info(f"Fetching intraday data from {from_date_intra} to {to_date_intra}")

    # Backup and remove corrupted CSVs before processing
    #clean_up_corrupted_csvs(selected_stocks)

    # Define the wrapper function with necessary arguments
    def process_ticker_wrapper(ticker):
        return process_ticker(
            ticker, from_date_intra, to_date_intra, last_trading_day, india_tz
        )

    # Process all tickers in parallel
    entries_dict, processed_data_dict = process_all_tickers(selected_stocks, process_ticker_wrapper)

    logging.info("Entry signal detection completed.")

    # Aggregate all entries from entries_dict
    all_entries = []
    for ticker, entries in entries_dict.items():
        for entry in entries:
            entry_record = {
                "Ticker": entry.get("Ticker"),
                "date": entry.get("date"),  # Ensure 'date' is correctly set
                "Entry Type": entry.get("Entry Type"),
                "Trend Type": entry.get("Trend Type"),
                "Price": entry.get("Price"),
                "Daily Change %": entry.get("Daily Change %"),
                "VWAP": entry.get("VWAP"),
            }
            all_entries.append(entry_record)

    if all_entries:
        entries_df = pd.DataFrame(all_entries)
        
        # **Apply normalize_time to parse 'date' column correctly**
        try:
            entries_df = normalize_time(entries_df)
        except Exception as e:
            logging.error(f"Normalization failed for aggregated entries: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure
        
        # **New Logging Added Here:**
        logging.debug(f"Aggregated Entries DataFrame Columns: {entries_df.columns.tolist()}")
        print(f"Aggregated Entries Columns: {entries_df.columns.tolist()}")

        # Validate 'date' column presence
        if 'date' not in entries_df.columns:
            logging.error("Aggregated Entries DataFrame is missing the 'date' column.")
            raise KeyError("Aggregated Entries DataFrame is missing the 'date' column.")

        # Validate DataFrame structure
        if not validate_dataframe(entries_df, ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP"]):
            logging.error("Aggregated Entries DataFrame failed validation.")
            raise ValueError("Aggregated Entries DataFrame failed validation.")

        # **Log sample data**
        logging.debug(f"Sample Aggregated Entries:\n{entries_df.head()}")

        # Save the entries to individual CSV files
        for ticker in entries_df['Ticker'].unique():
            ticker_entries = entries_df[entries_df['Ticker'] == ticker]
            today_str = last_trading_day.strftime("%Y-%m-%d")  # Format date as string
            entries_filename = f"{ticker}_price_action_entries_15min_{today_str}.csv"  # Define filename
            ticker_entries.to_csv(entries_filename, index=False)  # Save DataFrame to CSV
            logging.info(f"Saved price action entries for {ticker} to {entries_filename}")  # Log saving

        # Save the aggregated entries to a general CSV
        general_entries_filename = f"price_action_entries_15min_{last_trading_day.strftime('%Y-%m-%d')}.csv"
        entries_df.to_csv(general_entries_filename, index=False)
        logging.info(f"Saved aggregated price action entries to {general_entries_filename}")

        # **Log columns and sample data of the aggregated CSV**
        logging.debug(f"Aggregated Entries DataFrame Columns: {entries_df.columns.tolist()}")
        logging.debug(f"Sample Aggregated Entries DataFrame:\n{entries_df.head()}")

        # Optionally, return the entries_df for further processing
        return entries_df
    else:
        # Log and return an empty DataFrame if no entries were detected
        logging.info("No entry signals detected across all tickers.")
        return pd.DataFrame(columns=[
            "Ticker",
            "date",
            "Entry Type",
            "Trend Type",
            "Price",
            "Daily Change %",
            "VWAP",
        ])





def prepare_time_ranges(last_trading_day, india_tz):
    """
    Prepare the historical and intraday time ranges based on the last trading day.

    Parameters:
        last_trading_day (date): The last trading day to consider.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.

    Returns:
        tuple: from_date_hist, to_date_hist, from_date_intra, to_date_intra
    """
    # Set from_date and to_date for historical data (5 days back)
    past_n_days = last_trading_day - timedelta(days=5)
    from_date_hist = datetime.combine(past_n_days, datetime_time.min).replace(tzinfo=india_tz)
    to_date_hist = datetime.combine(last_trading_day - timedelta(days=1), datetime_time.max).replace(tzinfo=india_tz)

    # Set from_date and to_date for intraday data (market hours)
    from_date_intra = india_tz.localize(
        datetime.combine(last_trading_day, datetime_time(9, 15))
    )  # Market open time
    to_date_intra = india_tz.localize(
        datetime.combine(last_trading_day, datetime_time(15, 30))
    )  # Market close time

    # Adjust to_date_intra if current time is before market close
    now = datetime.now(india_tz)  # Current time in IST
    if now.date() == last_trading_day and now.time() < datetime_time(15, 30):
        to_date_intra = now  # Set to current time to avoid fetching future data

    return from_date_hist, to_date_hist, from_date_intra, to_date_intra

def process_all_tickers(selected_stocks, process_function):
    """
    Process all selected tickers using a provided processing function in parallel.

    Parameters:
        selected_stocks (list): List of selected stock tickers.
        process_function (function): Function to process each ticker.

    Returns:
        tuple: (entries_dict, processed_data_dict)
            - entries_dict: Dictionary mapping tickers to their list of entries.
            - processed_data_dict: Dictionary mapping tickers to their processed DataFrames.
    """
    entries_dict = {}
    processed_data_dict = {}
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
            future_to_ticker = {
                executor.submit(process_function, ticker): ticker
                for ticker in selected_stocks
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        data, ticker_entries = result
                        processed_data_dict[ticker] = data
                        if ticker_entries:
                            entries_dict[ticker] = ticker_entries
                except Exception as e:
                    logging.error(f"Error processing ticker {ticker}: {e}")
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")

    return entries_dict, processed_data_dict

def get_token(ticker):
    """
    Retrieve the instrument token for a given ticker.

    Parameters:
        ticker (str): The trading symbol of the stock.

    Returns:
        str or None: The instrument token if found, else None.
    """
    token = shares_tokens.get(ticker)
    if not token:
        logging.warning(f"No token found for ticker {ticker}. Skipping.")
    return token

def fetch_intraday_data(ticker, token, from_date_intra, to_date_intra):
    """
    Fetch intraday data with historical context for a given ticker.

    Parameters:
        ticker (str): The trading symbol of the stock.
        token (str): The instrument token for the ticker.
        from_date_intra (datetime): Start datetime for intraday data.
        to_date_intra (datetime): End datetime for intraday data.

    Returns:
        pd.DataFrame: DataFrame containing intraday data.
    """
    try:
        intraday_data = fetch_intraday_data_with_historical(
            ticker=ticker,
            token=token,
            from_date_intra=from_date_intra,
            to_date_intra=to_date_intra,
            interval="15minute",
            freshness_threshold=timedelta(minutes=10),  # Adjust as needed
        )
        return intraday_data
    except Exception as e:
        logging.error(f"Error fetching intraday data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def combine_and_validate_data(intraday_data, ticker, india_tz):
    if intraday_data.empty:
        logging.warning(f"No data available for {ticker}. Skipping.")
        return None

    # Remove duplicates based on 'date' and reset index
    data = intraday_data.drop_duplicates(subset="date").reset_index(drop=True)

    # Ensure required columns exist
    required_columns = [
        "date", "open", "high", "low", "close", "volume",
        "RSI", "ATR", "MACD", "Signal_Line",
        "Upper_Band", "Lower_Band", "Stochastic",
        "VWAP", "20_SMA", "Recent_High", "Recent_Low"
    ]
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        logging.warning(f"Missing required columns for {ticker}: {missing_columns}. Skipping.")
        return None

    # Convert 'date' column to timezone-aware datetime
    try:
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(india_tz)
    except Exception as e:
        logging.error(f"Error converting 'date' to timezone-aware datetime for {ticker}: {e}")
        logging.error(traceback.format_exc())
        return None

    # Sort the data by 'date' in ascending order without setting it as index
    data.sort_values(by="date", inplace=True)

    # Initialize `Entry Signal` column as "No" by default
    if "Entry Signal" not in data.columns:
        data["Entry Signal"] = "No"
    else:
        # Ensure no invalid data exists in 'Entry Signal' column
        data["Entry Signal"] = data["Entry Signal"].apply(lambda x: x if x in ["Yes", "No"] else "No")

    # **New Logging Added Here:**
    logging.debug(f"DataFrame for {ticker} after validation and sorting:\n{data.head()}")

    return data



def calculate_daily_change(today_data):
    """
    Calculate the daily percentage change based on today's data.

    Parameters:
        today_data (pd.DataFrame): DataFrame containing today's intraday data.

    Returns:
        float: Daily percentage change.
    """
    try:
        first_open = today_data["open"].iloc[0]
        latest_close = today_data["close"].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
        return daily_change
    except Exception as e:
        logging.error(f"Error calculating daily change: {e}")
        return 0.0  # Default to 0% change on failure

def apply_trend_conditions(today_data, daily_change, ticker, last_trading_day):
    """
    Apply trend conditions to detect Bullish or Bearish entries.

    Parameters:
        today_data (pd.DataFrame): DataFrame containing today's intraday data.
        daily_change (float): Daily percentage change.
        ticker (str): The trading symbol of the stock.
        last_trading_day (date): The last trading day to consider.

    Returns:
        list: List of detected entry dictionaries.
    """
    ticker_entries = []
    try:
        latest_data = today_data.iloc[-1]
        if (
            daily_change > 1.5
            and latest_data["RSI"] > 55
            and latest_data["close"] > latest_data["VWAP"]
        ):
            # Criteria for Bullish trend
            ticker_entries.extend(
                screen_trend(
                    today_data, "Bullish", daily_change, 1.5, ticker, last_trading_day
                )
            )
        elif (
            daily_change < -1.5
            and latest_data["RSI"] < 45
            and latest_data["close"] < latest_data["VWAP"]
        ):
            # Criteria for Bearish trend
            ticker_entries.extend(
                screen_trend(
                    today_data, "Bearish", daily_change, 1.5, ticker, last_trading_day
                )
            )
    except Exception as e:
        logging.error(f"Error applying trend conditions for {ticker}: {e}")

    return ticker_entries  # Return list of trend entries

def mark_entry_signals(data, entry_times, india_tz):
    """
    Mark 'Entry Signal' as 'Yes' for detected entry times in the data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data.
        entry_times (list): List of datetime objects representing entry times.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.
    """
    try:
        if entry_times:
            # Use 'entry_time' instead of 'time' to avoid shadowing
            localized_times = [
                entry_time if entry_time.tzinfo else india_tz.localize(entry_time) for entry_time in entry_times
            ]
            data.loc[data['date'].isin(localized_times), "Entry Signal"] = "Yes"
            logging.debug(f"Marked 'Entry Signal' for times: {localized_times}")
    except Exception as e:
        logging.error(f"Error marking entry signals: {e}")
        logging.error(traceback.format_exc())  # Now correctly references traceback



def screen_trend(data, trend_type, change, volume_multiplier, ticker, last_trading_day):
    """
    Screen data for bullish or bearish trend entries based on defined conditions.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with indicators.
        trend_type (str): "Bullish" or "Bearish".
        change (float): Daily percentage change.
        volume_multiplier (float): Multiplier for volume comparison.
        ticker (str): The trading symbol of the stock.
        last_trading_day (date): The last trading day to consider.

    Returns:
        list: List of dictionaries containing entry details.
    """
    trend_entries = []  # Initialize list to store trend entries

    try:
        if trend_type == "Bullish":
            # Define conditions for a Bullish trend
            pullback_condition = (
                (data["low"] >= data["VWAP"] * 0.99)  # Price not dipping below 1% of VWAP
                & (data["close"] > data["open"])  # Close price higher than open price
                & (data["MACD"] > data["Signal_Line"])  # MACD line above Signal line
                & (data["close"] > data["20_SMA"])  # Close price above 20 SMA
                & (data["Stochastic"] < 50)  # Stochastic oscillator below 50
                & (data["Stochastic"].diff() > 0)  # Stochastic oscillator increasing
            )
            breakout_condition = (
                (data["close"] > data["high"].rolling(window=10).max() * 1.02)  # Breakout above 2% of 10-period high
                & (
                    data["volume"]
                    > (volume_multiplier * 1.5)
                    * data["volume"].rolling(window=10).mean()
                )  # Volume higher than 1.5x average
                & (data["MACD"] > data["Signal_Line"])
                & (data["close"] > data["20_SMA"])
                & (data["Stochastic"] > 60)  # Stochastic oscillator above 60
                & (data["Stochastic"].diff() > 0)
            )
        else:  # Bearish conditions
            # Define conditions for a Bearish trend
            pullback_condition = (
                (data["high"] <= data["VWAP"] * 1.01)  # Price not rising above 1% of VWAP
                & (data["close"] < data["open"])  # Close price lower than open price
                & (data["MACD"] < data["Signal_Line"])  # MACD line below Signal line
                & (data["close"] < data["20_SMA"])  # Close price below 20 SMA
                & (data["Stochastic"] > 50)  # Stochastic oscillator above 50
                & (data["Stochastic"].diff() < 0)  # Stochastic oscillator decreasing
            )
            breakdown_condition = (
                (data["close"] < data["low"].rolling(window=10).min() * 0.98)  # Breakdown below 2% of 10-period low
                & (
                    data["volume"]
                    > (volume_multiplier * 1.5)
                    * data["volume"].rolling(window=10).mean()
                )  # Volume higher than 1.5x average
                & (data["MACD"] < data["Signal_Line"])
                & (data["close"] < data["20_SMA"])
                & (data["Stochastic"] < 40)  # Stochastic oscillator below 40
                & (data["Stochastic"].diff() < 0)
            )

        # Define conditions dictionary based on trend type
        conditions = (
            {"Pullback": pullback_condition, "Breakout": breakout_condition}
            if trend_type == "Bullish"
            else {"Pullback": pullback_condition, "Breakdown": breakdown_condition}
        )

        # Iterate over each condition and collect entries
        for entry_type, condition in conditions.items():
            selected = data[condition]  # Apply condition

            # **Corrected Filtering Based on 'date' Column:**
            selected = selected[selected['date'].dt.date == last_trading_day]

            logging.debug(f"Entries for {entry_type} condition: {len(selected)}")

            for _, row in selected.iterrows():
                # Ensure 'date' is a datetime object
                if isinstance(row["date"], pd.Timestamp):
                    formatted_date = row["date"].strftime("%Y-%m-%d %H:%M:%S%z")  # Include timezone offset
                else:
                    logging.error(f"'date' is not a datetime object for ticker {ticker}. Value: {row['date']}")
                    continue  # **Skip appending this entry**

                trend_entries.append(
                    {
                        "Ticker": ticker,
                        "date": formatted_date,  # Corrected line with timezone
                        "Entry Type": entry_type,  # "Pullback", "Breakout", etc.
                        "Trend Type": trend_type,  # "Bullish" or "Bearish"
                        "Price": round(row["close"], 2),  # Closing price rounded to 2 decimals
                        "Daily Change %": round(change, 2),  # Daily percentage change
                        "VWAP": round(row["VWAP"], 2) if "Pullback" in entry_type else None,  # VWAP if Pullback
                    }
                )

    except Exception as e:
        logging.error(f"Error screening trends for {ticker}: {e}")
        logging.error(traceback.format_exc())  # Logs the full traceback

    return trend_entries  # Return list of trend entries






def validate_dataframe(df, expected_columns):
    """
    Validate that the DataFrame contains the expected columns with correct data types.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (list): List of expected column names.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    # Check for missing columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        return False

    # Check data types (example: 'date' should be datetime)
    try:
        # **Remove 'format' to allow pandas to infer the datetime format with timezone**
        df['date'] = pd.to_datetime(df['date'], errors='raise')
    except Exception as e:
        logging.error(f"'date' column validation failed: {e}")
        return False

    return True


def update_combined_csv(ticker, entry_times, india_tz):
    """
    Update the '{ticker}_combined.csv' with detected entry signals for the specific ticker.

    Parameters:
        ticker (str): The trading symbol of the stock.
        entry_times (list): List of datetime objects representing entry times.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.
    """
    combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
    temp_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined_temp.csv")
    
    if entry_times:
        signal_data = pd.DataFrame(
            {
                "Ticker": [ticker] * len(entry_times),
                "date": [time.strftime("%Y-%m-%d %H:%M:%S") for time in entry_times],  # Replaced "Time" with "date"
                "Entry Signal": ["Yes"] * len(entry_times),
            }
        )
        
        # Validate 'date' column format
        try:
            pd.to_datetime(signal_data["date"], format="%Y-%m-%d %H:%M:%S")
        except ValueError as ve:
            logging.error(f"Invalid 'date' format in signal_data for {ticker}: {ve}")
            return  # Skip updating if validation fails
        
        logging.debug(f"Signal Data for {ticker}:\n{signal_data.head()}")
    else:
        logging.debug(f"No new entry signals for {ticker}. Skipping CSV update.")
        return  # No new entries to update

    if os.path.exists(combined_csv_path):
        try:
            existing_signals = pd.read_csv(combined_csv_path, parse_dates=["date"])
            existing_signals["date"] = pd.to_datetime(
                existing_signals["date"], utc=True
            ).dt.tz_convert(india_tz)
            logging.debug(f"Existing Signals for {ticker}:\n{existing_signals.head()}")

            updated_signals = pd.concat([existing_signals, signal_data], ignore_index=True)
            updated_signals = updated_signals.drop_duplicates(subset=["Ticker", "date"], keep="last")
            logging.debug(f"Updated Signals for {ticker}:\n{updated_signals.head()}")

        except Exception as e:
            logging.error(f"Error reading existing CSV for {ticker}: {e}. Recreating the file.")
            updated_signals = signal_data
    else:
        updated_signals = signal_data

    # Ensure correct column order
    updated_signals = updated_signals[["Ticker", "date", "Entry Signal"]]

    # Validate DataFrame before writing
    if not validate_dataframe(updated_signals, ["Ticker", "date", "Entry Signal"]):
        logging.error(f"DataFrame validation failed for {ticker}. Skipping CSV write.")
        return  # Skip writing invalid DataFrame

    # Save to temporary CSV first
    try:
        updated_signals.to_csv(temp_csv_path, index=False)
        logging.debug(f"Temporary CSV saved for {ticker}: {temp_csv_path}")
    except Exception as e:
        logging.error(f"Error writing temporary CSV for {ticker}: {e}")
        return  # Skip renaming if write fails

    # Rename temporary CSV to final CSV
    try:
        os.replace(temp_csv_path, combined_csv_path)
        logging.info(f"Updated entry signals for {ticker} in {combined_csv_path}.")
    except Exception as e:
        logging.error(f"Error renaming temporary CSV for {ticker}: {e}")

def process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day, india_tz):
    """
    Process a single ticker to identify potential price action entries.

    Parameters:
        ticker (str): The trading symbol of the stock.
        from_date_intra (datetime): Start datetime for intraday data.
        to_date_intra (datetime): End datetime for intraday data.
        last_trading_day (date): The last trading day to consider.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.

    Returns:
        tuple: (data, ticker_entries) where data is the combined DataFrame and ticker_entries is the list of detected entries.
    """
    token = shares_tokens.get(ticker)
    if not token:
        logging.warning(f"No token found for ticker {ticker}. Skipping.")
        return None, []  # Skip if token is not found

    intraday_data = fetch_intraday_data(ticker, token, from_date_intra, to_date_intra)
    data = combine_and_validate_data(intraday_data, ticker, india_tz)

    if data is None:
        return None, []  # Skip if data is invalid

    # **Corrected Filtering Based on 'date' Column:**
    today_data = data[data['date'] >= from_date_intra]

    if today_data.empty:
        logging.warning(
            f"No intraday data available after combining for {ticker}. Skipping."
        )
        return data, []  # Return data with 'Entry Signal' as 'No'

    # Calculate daily percentage change using today's data
    daily_change = calculate_daily_change(today_data)

    # Exclude entries after 3:30 PM IST
    cutoff_time = india_tz.localize(
        datetime.combine(last_trading_day, datetime_time(15, 30))
    )
    today_data = today_data[today_data['date'] < cutoff_time]

    # Apply trend conditions to detect entries
    ticker_entries = apply_trend_conditions(
        today_data, daily_change, ticker, last_trading_day
    )

    # Collect entry times
    entry_times = []
    for entry in ticker_entries:
        try:
            # Parse the 'date' string back to datetime with timezone
            entry_time = pd.to_datetime(entry["date"])
            # If 'date' is timezone-naive, localize it; else, convert to 'Asia/Kolkata'
            if entry_time.tzinfo is None:
                entry_time = india_tz.localize(entry_time)
            else:
                entry_time = entry_time.astimezone(india_tz)
            entry_times.append(entry_time)
        except Exception:
            logging.error(f"Invalid date format for entry in {ticker}: {entry['date']}")
            logging.error(traceback.format_exc())

    # Mark 'Entry Signal' as 'Yes' for detected entries
    mark_entry_signals(data, entry_times, india_tz)

    # Save the updated combined data to CSV
    combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
    temp_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined_temp.csv")

    try:
        # Write to a temporary CSV first
        data.to_csv(temp_csv_path, index=False)  # Changed index=True to index=False
        logging.debug(f"Temporary CSV saved for {ticker} at {temp_csv_path}")

        # Replace the old CSV with the new one
        os.replace(temp_csv_path, combined_csv_path)
        logging.info(f"Saved combined data with indicators and entry signals for {ticker} to cache.")
    except Exception as e:
        logging.error(f"Error saving combined CSV for {ticker}: {e}")
        # Attempt to remove the temporary CSV if exists
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            logging.info(f"Removed temporary CSV for {ticker} due to error.")

    return data, ticker_entries  # Return the DataFrame and list of entries





# ================================
# Remaining Functions and Imports
# ================================

# Ensure that the following helper functions are defined elsewhere in your script:
# - fetch_intraday_data_with_historical
# - calculate_rsi
# - calculate_atr
# - calculate_macd
# - calculate_bollinger_bands
# - calculate_stochastic

# Additionally, ensure that `shares_tokens` and `CACHE_DIR` are defined globally or passed appropriately.
'''
import shutil
import os

def backup_and_remove_corrupted_csv(ticker):
    """
    Backup and remove the corrupted {ticker}_combined.csv file.

    Parameters:
        ticker (str): The trading symbol of the stock.
    """
    combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
    backup_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined_backup.csv")
    
    if os.path.exists(combined_csv_path):
        try:
            shutil.copy(combined_csv_path, backup_csv_path)
            logging.info(f"Backed up the corrupted CSV for {ticker} to {backup_csv_path}")
            os.remove(combined_csv_path)
            logging.info(f"Removed the corrupted CSV file: {combined_csv_path}")
        except Exception as e:
            logging.error(f"Error backing up or removing corrupted CSV for {ticker}: {e}")

def clean_up_corrupted_csvs(selected_stocks):
    """
    Backup and remove corrupted CSVs for all selected stocks.

    Parameters:
        selected_stocks (list): List of selected stock tickers.
    """
    for ticker in selected_stocks:
        backup_and_remove_corrupted_csv(ticker)
'''
# ================================
# Time Normalization
# ================================

def normalize_time(df):
    """
    Normalize the 'date' column in a DataFrame to ensure consistent datetime format with timezone 'Asia/Kolkata'.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with normalized 'date' column.
    """
    if 'date' not in df.columns:
        logging.error("The DataFrame does not contain a 'date' column.")
        raise KeyError("The DataFrame does not contain a 'date' column.")

    logging.debug(f"Normalizing 'date' column for DataFrame with columns: {df.columns.tolist()}")

    try:
        # **Option 1: Remove 'format' to allow pandas to infer the format**
        df['date'] = pd.to_datetime(df['date'], errors='raise')
        
        # **Option 2: Include timezone in the format string**
        # df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S%z", errors='raise')
    except Exception as e:
        logging.error(f"Error converting 'date' to datetime: {e}")
        logging.error(traceback.format_exc())
        raise

    # Check if 'date' is timezone-aware
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')  # Localize to IST if naive
    else:
        # If 'date' has a timezone, convert it to 'Asia/Kolkata'
        df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')

    logging.debug(f"'date' column after normalization:\n{df['date'].head()}")
    return df

# ================================
# Main Processing Functions
# ================================

def run():
    """
    Fetch data, identify entries, and create combined paper trade files for selected stocks.
    """
    # Get the last trading day
    last_trading_day = get_last_trading_day()
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")  # Added for clarity

    # Run find_price_action_entry for the current trading day to detect entries
    price_action_entries = find_price_action_entry(last_trading_day)

    today_str = last_trading_day.strftime('%Y-%m-%d')  # Format date as string
    entries_filename = f"price_action_entries_15min_{today_str}.csv"  # Define entries filename
    papertrade_filename = f"papertrade_{today_str}.csv"  # Define papertrade filename

    if not price_action_entries.empty:
        # **New Logging Added Here:**
        logging.debug(f"Price Action Entries DataFrame Columns: {price_action_entries.columns.tolist()}")
        print(f"Price Action Entries Columns: {price_action_entries.columns.tolist()}")

        # Check if 'date' column exists
        if 'date' not in price_action_entries.columns:
            logging.error(f"The entries DataFrame does not contain a 'date' column.")
            print(f"Error: The entries DataFrame does not contain a 'date' column.")
            return  # Exit the function to prevent further errors

        # Normalize 'date' in new entries
        price_action_entries = normalize_time(price_action_entries)

        if os.path.exists(entries_filename):
            # Read existing entries from CSV
            existing_entries = pd.read_csv(entries_filename)
            # Normalize 'date' in existing entries
            existing_entries = normalize_time(existing_entries)
            # Combine existing entries with new entries
            combined_entries = pd.concat([existing_entries, price_action_entries], ignore_index=True)
            # Drop duplicates based on 'Ticker' and 'date' columns
            combined_entries = combined_entries.drop_duplicates(subset=['Ticker', 'date'])
            # Save combined entries back to CSV
            combined_entries.to_csv(entries_filename, index=False)
        else:
            # If entries file doesn't exist, save new entries
            price_action_entries.to_csv(entries_filename, index=False)

        logging.info(f"Entries detected. Saved results to {entries_filename}.")
        print(f"Entries detected and saved to '{entries_filename}'.")

        # Print the entries for today to the console
        print("\nPrice Action Entries for Today:")
        print(price_action_entries)

        # Create or append to the combined papertrade file
        create_papertrade_file(
            input_file=entries_filename,
            output_file=papertrade_filename,
            last_trading_day=last_trading_day
        )
        print(f"\nCombined papertrade data saved to '{papertrade_filename}'.")
        # Optionally, read and print the papertrade data
        try:
            papertrade_data = pd.read_csv(papertrade_filename)
            # **New Logging Added Here:**
            logging.debug(f"Papertrade DataFrame Columns: {papertrade_data.columns.tolist()}")
            print("\nPapertrade Data for Today:")
            print(papertrade_data)
        except Exception as e:
            logging.error(f"Error reading papertrade file {papertrade_filename}: {e}")
            print(f"Error reading papertrade file '{papertrade_filename}': {e}")
    else:
        # Log and print if no entries were detected
        logging.info(f"No entries detected for {last_trading_day}.")
        print(f"No entries detected for {last_trading_day}.")



def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads price action entries from a CSV file and creates a papertrade file for bullish and bearish strategies.
    Includes only entries after 9:20 AM on the last trading day.

    Parameters:
        input_file (str): Path to the price action entries CSV file.
        output_file (str): Path to the papertrade CSV file.
        last_trading_day (date): The last trading day to consider.
    """
    try:
        india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST

        logging.info(f"Reading data from {input_file}.")
        price_action_entries = pd.read_csv(input_file)

        # Log columns to verify 'date' existence
        logging.debug(f"Price Action Entries Columns: {price_action_entries.columns.tolist()}")
        print(f"Price Action Entries Columns: {price_action_entries.columns.tolist()}")

        # Validate 'date' column presence before normalization
        if 'date' not in price_action_entries.columns:
            logging.error(f"The input file {input_file} does not contain a 'date' column.")
            raise KeyError(f"The input file {input_file} does not contain a 'date' column.")

        # Ensure 'date' is parsed as timezone-aware datetime
        price_action_entries = normalize_time(price_action_entries)

        # Sort the entire DataFrame by 'date'
        price_action_entries = sort_dataframe_by_date(price_action_entries)

        # Define 9:20 AM threshold for filtering (timezone-aware)
        threshold_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 20)))

        # Filter entries for the last trading day and after 9:20 AM
        filtered_entries = price_action_entries[
            (price_action_entries["date"].dt.date == last_trading_day) &
            (price_action_entries["date"] >= threshold_time)
        ]

        # Log the filtered entries
        logging.debug(f"Filtered Entries for {last_trading_day} after 9:20 AM: {filtered_entries.shape}")
        print(f"Filtered Entries for {last_trading_day} after 9:20 AM: {filtered_entries.shape}")

        if filtered_entries.empty:
            # Log and print warning if no valid entries are found
            logging.warning(f"No valid entries for {last_trading_day} after 9:20 AM.")
            print(f"No valid entries for {last_trading_day} after 9:20 AM.")
            return  # Exit the function

        # Sort filtered entries by 'date'
        filtered_entries = sort_dataframe_by_date(filtered_entries)

        # Sort by 'date' and get the earliest entry for each ticker
        earliest_entries = filtered_entries.sort_values(by="date").groupby("Ticker").first().reset_index()

        # Log columns of earliest_entries
        logging.debug(f"Earliest Entries Columns: {earliest_entries.columns.tolist()}")
        print(f"Earliest Entries Columns: {earliest_entries.columns.tolist()}")

        # Ensure 'date' column exists in earliest_entries
        if 'date' not in earliest_entries.columns:
            logging.error(f"'date' column is missing in earliest_entries for {last_trading_day}.")
            raise KeyError('date')

        # Validate the DataFrame
        if not validate_dataframe(earliest_entries, ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP"]):
            logging.error("Aggregated Entries DataFrame failed validation.")
            raise ValueError("Aggregated Entries DataFrame failed validation.")

        # **Log sample data**
        logging.debug(f"Sample Earliest Entries:\n{earliest_entries.head()}")

        # Calculate target prices, quantities, and total values based on trend type
        if "Trend Type" in earliest_entries.columns:
            bullish_entries = earliest_entries[earliest_entries["Trend Type"] == "Bullish"]  # Select Bullish entries
            bearish_entries = earliest_entries[earliest_entries["Trend Type"] == "Bearish"]  # Select Bearish entries

            # Create copies of the slices to modify them without SettingWithCopyWarning
            bullish_entries = bullish_entries.copy()
            bearish_entries = bearish_entries.copy()

            # Bullish processing: Calculate target price, quantity, and total value
            if not bullish_entries.empty:
                bullish_entries["Target Price"] = bullish_entries["Price"] * 1.01  # Target price is 1% above current price
                bullish_entries["Quantity"] = (20000 / bullish_entries["Price"]).astype(int)  # Quantity based on capital allocation
                bullish_entries["Total value"] = bullish_entries["Quantity"] * bullish_entries["Price"]  # Total value of the position

            # Bearish processing: Calculate target price, quantity, and total value
            if not bearish_entries.empty:
                bearish_entries["Target Price"] = bearish_entries["Price"] * 0.99  # Target price is 1% below current price
                bearish_entries["Quantity"] = (20000 / bearish_entries["Price"]).astype(int)  # Quantity based on capital allocation
                bearish_entries["Total value"] = bearish_entries["Quantity"] * bearish_entries["Price"]  # Total value of the position

            # Combine bullish and bearish entries into one DataFrame
            combined_entries = pd.concat([bullish_entries, bearish_entries], ignore_index=True)

            # Log combined_entries before date parsing
            logging.debug(f"Combined Entries Columns before date parsing: {combined_entries.columns.tolist()}")
            print(f"Combined Entries Columns before date parsing: {combined_entries.columns.tolist()}")

            # **Ensure 'date' is datetime with timezone info for accurate deduplication**
            try:
                # **Remove 'format' to allow pandas to infer the datetime format with timezone**
                combined_entries['date'] = pd.to_datetime(combined_entries['date'], errors='raise')
            except Exception as e:
                logging.error(f"Error converting 'date' in combined_entries: {e}")
                logging.error(traceback.format_exc())
                raise

            # Sort combined_entries by 'date'
            combined_entries = sort_dataframe_by_date(combined_entries)

            # Log combined_entries after date parsing
            logging.debug(f"Combined Entries after date parsing:\n{combined_entries.head()}")
            print(f"Combined Entries after date parsing:\n{combined_entries.head()}")

            # Validate the combined_entries DataFrame
            if not validate_dataframe(combined_entries, ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP"]):
                logging.error(f"Combined Entries DataFrame validation failed for {last_trading_day}.")
                raise ValueError(f"Combined Entries DataFrame validation failed for {last_trading_day}.")

            # Append or create the papertrade file
            if os.path.exists(output_file):
                # Read existing papertrade entries from CSV
                existing_papertrade = pd.read_csv(output_file)
                # Normalize 'date' in existing entries
                existing_papertrade = normalize_time(existing_papertrade)
                # Sort existing entries by 'date'
                existing_papertrade = sort_dataframe_by_date(existing_papertrade)
                # Combine existing entries with new combined entries
                all_papertrade_entries = pd.concat([existing_papertrade, combined_entries], ignore_index=True)
                # Drop duplicates based on 'Ticker' and 'date' columns
                all_papertrade_entries = all_papertrade_entries.drop_duplicates(subset=['Ticker', 'date'])
                # Sort by 'date'
                all_papertrade_entries = sort_dataframe_by_date(all_papertrade_entries)
                # Save the combined entries back to the papertrade CSV
                all_papertrade_entries.to_csv(output_file, index=False)
            else:
                # If papertrade file doesn't exist, save the combined entries directly
                combined_entries.to_csv(output_file, index=False)

            logging.info(f"Combined papertrade data saved to {output_file}")  # Log saving
            print(f"Combined papertrade data saved to '{output_file}'")        # Print confirmation

        else:
            # Log and print warning if 'Trend Type' column is missing
            logging.warning("Trend Type column is missing in the input file. Skipping processing.")
            print("Trend Type column is missing in the input file. Skipping processing.")
            return  # Exit the function

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        logging.error(traceback.format_exc())
        print(f"KeyError: {e}")
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        logging.error(traceback.format_exc())
        print(f"ValueError: {ve}")
    except Exception as e:
        logging.error(f"Error creating combined papertrade file: {e}")
        logging.error(traceback.format_exc())  # Now correctly references traceback
        print(f"Error creating combined papertrade file: {e}")







# ================================
# Main Execution Function
# ================================

def main():
    """
    Main function to execute the trading script.
    - Determines the last trading day.
    - Selects stocks based on RSI criteria.
    - Schedules the entry detection to run every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.
    """
    import pytz
    from datetime import datetime, timedelta, time as datetime_time
    import logging
    import time
    import signal
    import sys

    # ================================
    # Setup Signal Handling for Graceful Shutdown
    # ================================

    def signal_handler(sig, frame):
        logging.info("Interrupt received. Shutting down gracefully...")
        print("Interrupt received. Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ================================
    # Timezone Configuration
    # ================================

    india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST
    now = datetime.now(india_tz)  # Current time in IST

    # ================================
    # Define the Start and End Times
    # ================================

    # Define the start and end times with timezone awareness
    start_time_naive = datetime.combine(now.date(), datetime_time(9, 15, 5))
    end_time_naive = datetime.combine(now.date(), datetime_time(18, 0, 0))
    
    start_time = india_tz.localize(start_time_naive)
    end_time = india_tz.localize(end_time_naive)
    
    # ================================
    # Handle Past Start Times
    # ================================

    if now > end_time:
        logging.warning("Current time is past the trading window. Exiting the script.")
        print("Current time is past the trading window. Exiting the script.")
        return
    elif now > start_time:
        logging.info("Current time is within the trading window. Scheduling remaining jobs.")
    else:
        logging.info("Current time is before the trading window. Waiting to start scheduling.")
    
    # ================================
    # Generate Scheduled Times at 15-Minute Intervals
    # ================================

    # Generate the list of scheduled times at 15-minute intervals between start_time and end_time
    scheduled_times = []
    current_time = start_time

    while current_time <= end_time:
        scheduled_times.append(current_time.strftime("%H:%M:%S"))
        current_time += timedelta(minutes=15)
    
    # ================================
    # Define the Job Function
    # ================================

    def job():
        """
        Job function to execute the trading strategy.
        """
        try:
            run()
            logging.info("Executed trading strategy successfully.")
            print("Executed trading strategy successfully.")
        except Exception as e:
            logging.error(f"Error executing job: {e}")
            print(f"Error executing job: {e}")

    # ================================
    # Schedule the Jobs
    # ================================

    for scheduled_time in scheduled_times:
        # Convert scheduled_time string back to datetime object for comparison
        scheduled_datetime_naive = datetime.strptime(scheduled_time, "%H:%M:%S")
        scheduled_datetime = india_tz.localize(datetime.combine(now.date(), scheduled_datetime_naive.time()))
        
        if scheduled_datetime > now:
            schedule.every().day.at(scheduled_time).do(job)
            logging.info(f"Scheduled job at {scheduled_time} IST for ticker(s): {selected_stocks}")
        else:
            logging.debug(f"Skipped scheduling job at {scheduled_time} IST as it's already past.")

    logging.info("Scheduled `run` to execute every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    print("Scheduled `run` to execute every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    
    # ================================
    # Immediate Execution Check
    # ================================

    # Optionally, execute the job immediately if within trading hours
    if start_time <= now <= end_time:
        logging.info("Executing initial run of trading strategy.")
        print("Executing initial run of trading strategy.")
        job()
    
    # ================================
    # Keep the Script Running to Allow Scheduled Jobs
    # ================================

    while True:
        schedule.run_pending()  # Run any pending scheduled jobs
        time.sleep(1)  # Sleep to prevent high CPU usage

if __name__ == "__main__":
    main()  # Execute the main function when the script is run
