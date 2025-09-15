# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:15:12 2024

@author: Saarit
"""

# Import necessary modules
import datetime as dt  # For handling dates and times
from kiteconnect import KiteConnect  # Kite Connect API for trading data
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
from datetime import datetime, timedelta, time as datetime_time  # Specific datetime classes
import logging  # Logging events for debugging and monitoring
import os  # Operating system interface
import sys  # For command-line arguments
import time as time_module  # Time-related functions
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
import threading  # Thread-based parallelism
import pytz  # Timezone calculations
import schedule  # Job scheduling for periodic tasks
import time

# Create a semaphore to limit concurrent API calls
api_semaphore = threading.Semaphore(2)  # Limits to 2 concurrent API calls; adjust as needed

# Importing the shares list from et1_select_stocklist.py
from et1_stock_tickers import shares  # 'shares' should be a list of stock tickers

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

# Define known market holidays for 2024
market_holidays = [
    dt.date(2024, 1, 26),  # Republic Day
    dt.date(2024, 3, 8),   # Mahashivratri
    dt.date(2024, 3, 25),  # Holi
    dt.date(2024, 3, 29),  # Good Friday
    dt.date(2024, 4, 11),  # Id-Ul-Fitr (Ramadan Eid)
    dt.date(2024, 4, 17),  # Shri Ram Navami
    dt.date(2024, 5, 1),   # Maharashtra Day
    dt.date(2024, 6, 17),  # Bakri Id
    dt.date(2024, 7, 17),  # Moharram
    dt.date(2024, 8, 15),  # Independence Day
    dt.date(2024, 10, 2),  # Gandhi Jayanti
    dt.date(2024, 11, 1),  # Diwali Laxmi Pujan
    dt.date(2024, 11, 15), # Gurunanak Jayanti
    dt.date(2024, 11, 20), # Maha elections
    dt.date(2024, 12, 25), # Christmas
]

# Directory for storing cached data
CACHE_DIR = "data_cache"  # Name of the cache directory
os.makedirs(CACHE_DIR, exist_ok=True)  # Create the cache directory if it doesn't exist

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
    """
    if not os.path.exists(cache_file):
        return False  # Cache file doesn't exist
    cache_mtime = dt.datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)  # Last modified time in UTC
    current_time = dt.datetime.now(pytz.UTC)  # Current time in UTC
    is_fresh = (current_time - cache_mtime) < freshness_threshold
    logging.debug(f"Cache freshness for {cache_file}: {is_fresh} (Last Modified: {cache_mtime}, Current Time: {current_time})")
    return is_fresh


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
                        data.to_csv(cache_file, index=False)  # Save fetched data to cache
                        logging.info(f"Cached data for {ticker}")
                    time.sleep(0.1)  # Brief pause to respect API rate limits
                    return data  # Return fetched data
                except Exception as e:
                    if "Too many requests" in str(e):
                        # Handle API rate limiting with exponential backoff
                        logging.warning(f"Rate limit exceeded for {ticker}. Retrying in {delay} seconds...")
                        time_module.sleep(delay)
                        delay *= 2  # Double the delay for next retry
                    else:
                        # Log other exceptions and abort retries
                        logging.error(f"Error fetching data for {ticker}: {e}")
                        return pd.DataFrame()
        # If all retries fail, log the failure
        logging.error(f"Failed to fetch data for {ticker} after {max_retries} retries.")
        return pd.DataFrame()

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




def is_market_open():
    """
    Check if the market is currently open based on the current time and known holidays.

    Returns:
        bool: True if market is open, False otherwise.
    """
    now = dt.datetime.now(pytz.timezone('Asia/Kolkata'))  # Current time in IST
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
        reference_date = dt.date.today()  # Start from today if no reference_date provided

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

def select_stocks(last_trading_day):
    """
    Select stocks to monitor based on historical RSI criteria.

    Parameters:
        last_trading_day (date): The last trading day to consider for selection.

    Returns:
        list: List of selected stock tickers.
    """
    # Set from_date and to_date for historical data
    past_5_days = last_trading_day - timedelta(days=5)  # Look back 5 days
    from_date_hist = datetime.combine(past_5_days, datetime_time.min)  # Start of the first day
    to_date_hist = datetime.combine(last_trading_day - timedelta(days=1), datetime_time.max)  # End of the day before last_trading_day

    logging.info(f"Fetching historical data from {from_date_hist} to {to_date_hist}")

    selected_stocks = []  # Initialize list for selected stocks

    def process_ticker(ticker_token):
        """
        Process a single ticker to determine if it meets selection criteria.

        Parameters:
            ticker_token (tuple): Tuple containing (ticker, token).

        Returns:
            str or None: Ticker symbol if selected, else None.
        """
        ticker, token = ticker_token
        try:
            # Fetch historical data with caching enabled
            historical_data = fetch_historical_data(
                ticker=ticker,
                token=token,
                from_date=from_date_hist,
                to_date=to_date_hist,
                interval="15minute",
                use_cached_data=True  # Use cached data if available
            )

            if historical_data.empty:
                logging.warning(f"No historical data available for {ticker}. Skipping.")
                return None  # Skip if no data

            # Ensure required columns exist
            required_columns = ['date', 'close', 'volume', 'high', 'low', 'open']
            if not all(col in historical_data.columns for col in required_columns):
                logging.warning(f"Missing required columns for {ticker}. Skipping.")
                return None  # Skip if essential data is missing

            # Convert 'date' column to timezone-aware datetime
            historical_data['date'] = pd.to_datetime(historical_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
            historical_data.set_index('date', inplace=True)  # Set 'date' as index

            # Calculate RSI indicator
            historical_data['RSI'] = calculate_rsi(historical_data['close'])

            # Adjusted selection criteria: RSI between 20 and 80
            if 20 < historical_data['RSI'].iloc[-1] < 80:
                return ticker  # Select the ticker
            else:
                return None  # Do not select

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")
            return None  # Skip on error

    # Use ThreadPoolExecutor to process tickers in parallel for efficiency
    selected_stocks = []
    with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 threads; adjust as needed
        # Submit all tickers for processing
        future_to_ticker = {executor.submit(process_ticker, item): item for item in shares_tokens.items()}
        for future in as_completed(future_to_ticker):
            result = future.result()
            if result:
                selected_stocks.append(result)  # Append if ticker was selected

    logging.info(f"Stock selection completed. Selected stocks: {selected_stocks}")
    print(f"Selected stocks: {selected_stocks}")
    return selected_stocks  # Return the list of selected stocks

def find_price_action_entry(last_trading_day, selected_stocks):
    """
    Identify potential trading entries for selected stocks, classifying them as Bullish or Bearish.
    Calculations are performed on combined historical and intraday data to ensure sufficient data points.

    Parameters:
        last_trading_day (date): The last trading day to consider.
        selected_stocks (list): List of selected stock tickers.

    Returns:
        pd.DataFrame: DataFrame containing detected price action entries.
    """
    import pytz
    from datetime import datetime, timedelta, time as datetime_time
    import pandas as pd
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    # Helper functions for technical indicators
    def calculate_rsi(series, window=14):
        """Calculate the Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)  # Replace NaN values with 0

    def calculate_atr(data, window=14):
        """Calculate the Average True Range (ATR)."""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr.fillna(0)

    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate the MACD and Signal line."""
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.fillna(0), signal_line.fillna(0)

    def calculate_bollinger_bands(data, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band.fillna(0), lower_band.fillna(0)

    def calculate_stochastic(data, window=14):
        """Calculate Stochastic Oscillator."""
        low_min = data['low'].rolling(window=window).min()
        high_max = data['high'].rolling(window=window).max()
        stochastic = 100 * (data['close'] - low_min) / (high_max - low_min)
        return stochastic.fillna(0)

    def calculate_daily_change(data, last_trading_day):
        """Calculate the daily percentage change for the last trading day."""
        # Use today's data only
        today_start = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
        today_end = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
        today_data = data.loc[today_start:today_end]

        if today_data.empty:
            return 0  # No data available

        open_price = today_data['open'].iloc[0]  # Opening price
        close_price = today_data['close'].iloc[-1]  # Closing price
        return ((close_price - open_price) / open_price) * 100  # Percentage change

    # Timezone object for 'Asia/Kolkata'
    india_tz = pytz.timezone('Asia/Kolkata')

    # Set from_date and to_date for historical data (5 days back)
    past_n_days = last_trading_day - timedelta(days=5)
    from_date_hist = datetime.combine(past_n_days, datetime_time.min)  # Start of the first day
    to_date_hist = datetime.combine(last_trading_day - timedelta(days=1), datetime_time.max)  # End of the day before last_trading_day

    # Set from_date and to_date for intraday data (market hours)
    from_date_intra = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))  # Market open time
    to_date_intra = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))    # Market close time

    # Adjust to_date_intra if current time is before market close
    now = datetime.now(india_tz)  # Current time in IST
    if now.date() == last_trading_day and now.time() < datetime_time(15, 30):
        to_date_intra = now  # Set to current time to avoid fetching future data

    logging.info(f"Fetching historical data from {from_date_hist} to {to_date_hist}")
    logging.info(f"Fetching intraday data from {from_date_intra} to {to_date_intra}")
    entries = []  # Initialize list to store entries

    def process_ticker(ticker):
        """
        Process a single ticker to identify potential price action entries.

        Parameters:
            ticker (str): The trading symbol of the stock.

        Returns:
            list: List of dictionaries containing entry details.
        """
        token = shares_tokens.get(ticker)  # Get the instrument token for the ticker
        if not token:
            logging.warning(f"No token found for ticker {ticker}. Skipping.")
            return []  # Skip if token is not found

        try:
            # Fetch historical data with caching enabled
            historical_data = fetch_historical_data(
                ticker=ticker,
                token=token,
                from_date=from_date_hist,
                to_date=to_date_hist,
                interval="15minute",
                use_cached_data=True  # Use cached data for historical
            )

            # Fetch intraday data with partial caching
            intraday_data = fetch_intraday_data_partial(
                ticker=ticker,
                token=token,
                from_date_intra=from_date_intra,
                to_date_intra=to_date_intra,
                interval="15minute",
                cache_dir=CACHE_DIR,
                freshness_threshold=timedelta(minutes=10)  # Adjust as needed
            )

            # Combine historical and intraday data, removing duplicates based on 'date'
            data = pd.concat([historical_data, intraday_data]).drop_duplicates(subset='date').reset_index(drop=True)

            if data.empty:
                logging.warning(f"No data available for {ticker}. Skipping.")
                return []  # Skip if no data is available

            # Ensure required columns exist
            required_columns = ['date', 'close', 'volume', 'high', 'low', 'open']
            if not all(col in data.columns for col in required_columns):
                logging.warning(f"Missing required columns for {ticker}. Skipping.")
                return []  # Skip if essential data is missing

            # Convert 'date' column to timezone-aware datetime in IST
            data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert(india_tz)
            data.set_index('date', inplace=True)  # Set 'date' as index

            # Sort data by index (date) in ascending order
            data.sort_index(inplace=True)

            # Calculate technical indicators on the combined data
            data['RSI'] = calculate_rsi(data['close'])  # Relative Strength Index
            data['ATR'] = calculate_atr(data)           # Average True Range
            data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()  # Volume Weighted Average Price
            data['20_SMA'] = data['close'].rolling(window=20).mean()  # 20-period Simple Moving Average
            data['Recent_High'] = data['high'].rolling(window=5).max()  # 5-period Recent High
            data['Recent_Low'] = data['low'].rolling(window=5).min()    # 5-period Recent Low
            data['MACD'], data['Signal_Line'] = calculate_macd(data)    # Moving Average Convergence Divergence
            data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)  # Bollinger Bands
            data['Stochastic'] = calculate_stochastic(data)  # Stochastic Oscillator

            # Extract today's data for entry signal detection
            today_data = data.loc[from_date_intra:]

            if today_data.empty:
                logging.warning(f"No intraday data available after combining for {ticker}. Skipping.")
                return []  # Skip if no intraday data is available

            # Calculate daily percentage change using today's data
            daily_change = calculate_daily_change(data, last_trading_day)

            # Exclude entries after 3:30 PM IST
            cutoff_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
            today_data = today_data[today_data.index < cutoff_time]

            # Define entry signal criteria based on trend type
            def screen_trend(data, trend_type, change, volume_multiplier):
                """
                Screen data for bullish or bearish trend entries based on defined conditions.

                Parameters:
                    data (pd.DataFrame): DataFrame containing stock data with indicators.
                    trend_type (str): "Bullish" or "Bearish".
                    change (float): Daily percentage change.
                    volume_multiplier (float): Multiplier for volume comparison.

                Returns:
                    list: List of dictionaries containing entry details.
                """
                trend_entries = []  # Initialize list to store trend entries

                if trend_type == "Bullish":
                    # Define conditions for a Bullish trend
                    pullback_condition = (
                        (data['low'] >= data['VWAP'] * 0.99) &  # Price not dipping below 1% of VWAP
                        (data['close'] > data['open']) &        # Close price higher than open price
                        (data['MACD'] > data['Signal_Line']) &  # MACD line above Signal line
                        (data['close'] > data['20_SMA']) &      # Close price above 20 SMA
                        (data['Stochastic'] < 50) &             # Stochastic oscillator below 50
                        (data['Stochastic'].diff() > 0)         # Stochastic oscillator increasing
                    )
                    breakout_condition = (
                        (data['close'] > data['high'].rolling(window=10).max() * 1.02) &  # Breakout above 2% of 10-period high
                        (data['volume'] > (volume_multiplier * 1.5) * data['volume'].rolling(window=10).mean()) &  # Volume higher than 1.5x average
                        (data['MACD'] > data['Signal_Line']) &
                        (data['close'] > data['20_SMA']) &
                        (data['Stochastic'] > 60) &             # Stochastic oscillator above 60
                        (data['Stochastic'].diff() > 0)
                    )
                else:  # Bearish conditions
                    # Define conditions for a Bearish trend
                    pullback_condition = (
                        (data['high'] <= data['VWAP'] * 1.01) &  # Price not rising above 1% of VWAP
                        (data['close'] < data['open']) &         # Close price lower than open price
                        (data['MACD'] < data['Signal_Line']) &   # MACD line below Signal line
                        (data['close'] < data['20_SMA']) &       # Close price below 20 SMA
                        (data['Stochastic'] > 50) &              # Stochastic oscillator above 50
                        (data['Stochastic'].diff() < 0)          # Stochastic oscillator decreasing
                    )
                    breakdown_condition = (
                        (data['close'] < data['low'].rolling(window=10).min() * 0.98) &  # Breakdown below 2% of 10-period low
                        (data['volume'] > (volume_multiplier * 1.5) * data['volume'].rolling(window=10).mean()) &  # Volume higher than 1.5x average
                        (data['MACD'] < data['Signal_Line']) &
                        (data['close'] < data['20_SMA']) &
                        (data['Stochastic'] < 40) &             # Stochastic oscillator below 40
                        (data['Stochastic'].diff() < 0)
                    )

                # Define conditions dictionary based on trend type
                conditions = {
                    "Pullback": pullback_condition,
                    "Breakout": breakout_condition
                } if trend_type == "Bullish" else {
                    "Pullback": pullback_condition,
                    "Breakdown": breakdown_condition
                }

                # Iterate over each condition and collect entries
                for entry_type, condition in conditions.items():
                    selected = data[condition]  # Apply condition
                    selected = selected[selected.index.date == last_trading_day]  # Keep only today's data
                    for idx, row in selected.iterrows():
                        trend_entries.append({
                            "Ticker": ticker,
                            "Time": idx.strftime('%Y-%m-%d %H:%M:%S'),  # Format timestamp
                            "Entry Type": entry_type,                   # "Pullback", "Breakout", etc.
                            "Trend Type": trend_type,                   # "Bullish" or "Bearish"
                            "Price": round(row['close'], 2),           # Closing price rounded to 2 decimals
                            "Daily Change %": round(change, 2),         # Daily percentage change
                            "VWAP": round(row['VWAP'], 2) if 'Pullback' in entry_type else None  # VWAP if Pullback
                        })

                return trend_entries  # Return list of trend entries

            ticker_entries = []  # Initialize list to store entries for this ticker

            # Use the latest data point to determine trend
            latest_data = today_data.iloc[-1]
            if daily_change > 1.5 and latest_data['RSI'] > 55 and latest_data['close'] > latest_data['VWAP']:
                # Criteria for Bullish trend
                ticker_entries.extend(screen_trend(today_data, "Bullish", daily_change, 1.5))
            elif daily_change < -1.5 and latest_data['RSI'] < 45 and latest_data['close'] < latest_data['VWAP']:
                # Criteria for Bearish trend
                ticker_entries.extend(screen_trend(today_data, "Bearish", daily_change, 1.5))

            # **Optional:** Save the combined data with indicators to a CSV file for analysis
            # output_dir = "stock_data_with_indicators"
            # os.makedirs(output_dir, exist_ok=True)
            # output_file = os.path.join(output_dir, f"{ticker}_data_with_indicators.csv")
            # data.to_csv(output_file)
            # logging.info(f"Saved data with indicators for {ticker} to {output_file}")

            return ticker_entries  # Return list of entries for this ticker

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")
            return []  # Return empty list on error

    # Use ThreadPoolExecutor to process multiple tickers in parallel
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:  # Allow up to 5 concurrent threads
            # Submit all tickers to the executor
            future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in selected_stocks}
            for future in as_completed(future_to_ticker):
                result = future.result()  # Get the result from the future
                entries.extend(result)    # Append the entries to the main list
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")

    logging.info("Entry signal detection completed.")  # Log completion
    entries_df = pd.DataFrame(entries)  # Convert list of entries to DataFrame

    # Ensure the entries DataFrame has the correct columns
    expected_columns = ["Ticker", "Time", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP"]
    for col in expected_columns:
        if col not in entries_df.columns:
            entries_df[col] = None  # Add missing columns with None values

    # **Deduplicate entries based on 'Ticker' and 'Time'**
    entries_df = entries_df.drop_duplicates(subset=['Ticker', 'Time'])

    # Save the entries to a CSV file
    today_str = last_trading_day.strftime('%Y-%m-%d')  # Format date as string
    entries_filename = f"price_action_entries_15min_{today_str}.csv"  # Define filename
    entries_df.to_csv(entries_filename, index=False)  # Save DataFrame to CSV
    logging.info(f"Saved price action entries to {entries_filename}")  # Log saving
    return entries_df  # Return the entries DataFrame

def normalize_time(df):
    """
    Normalize the 'Time' column in a DataFrame to ensure consistent datetime format with timezone 'Asia/Kolkata'.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Time' column.

    Returns:
        pd.DataFrame: DataFrame with normalized 'Time' column.
    """
    # Convert 'Time' to datetime without specifying UTC, assuming it's already in IST
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Check if 'Time' is naive (no timezone)
    if df['Time'].dt.tz is None:
        df['Time'] = df['Time'].dt.tz_localize('Asia/Kolkata')  # Localize to IST if naive
    else:
        # If 'Time' has a timezone, convert it to 'Asia/Kolkata'
        df['Time'] = df['Time'].dt.tz_convert('Asia/Kolkata')
    
    return df  # Return the DataFrame with normalized 'Time'

def run_for_last_30_trading_days(selected_stocks):
    """
    Fetch data, identify entries, and create combined paper trade files for selected stocks.

    Parameters:
        selected_stocks (list): List of selected stock tickers.
    """
    # Get the last trading day
    last_trading_day = get_last_trading_day()
    logging.info(f"Last trading day: {last_trading_day}")

    # Run find_price_action_entry for the current trading day to detect entries
    price_action_entries = find_price_action_entry(last_trading_day, selected_stocks)

    today_str = last_trading_day.strftime('%Y-%m-%d')  # Format date as string
    entries_filename = f"price_action_entries_15min_{today_str}.csv"  # Define entries filename
    papertrade_filename = f"papertrade_{today_str}.csv"  # Define papertrade filename

    if not price_action_entries.empty:
        # Normalize 'Time' in new entries
        price_action_entries = normalize_time(price_action_entries)

        if os.path.exists(entries_filename):
            # Read existing entries from CSV
            existing_entries = pd.read_csv(entries_filename)
            # Normalize 'Time' in existing entries
            existing_entries = normalize_time(existing_entries)
            # Combine existing entries with new entries
            combined_entries = pd.concat([existing_entries, price_action_entries], ignore_index=True)
            # Drop duplicates based on 'Ticker' and 'Time' columns
            combined_entries = combined_entries.drop_duplicates(subset=['Ticker', 'Time'])
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
        papertrade_data = pd.read_csv(papertrade_filename)
        print("\nPapertrade Data for Today:")
        print(papertrade_data)
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
        price_action_entries = pd.read_csv(input_file)  # Read price action entries from CSV

        # Ensure 'Time' is parsed as timezone-aware datetime
        price_action_entries = normalize_time(price_action_entries)

        # Define 9:20 AM threshold for filtering (timezone-aware)
        threshold_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 20)))

        # Filter entries for the last trading day and after 9:20 AM
        filtered_entries = price_action_entries[
            (price_action_entries["Time"].dt.date == last_trading_day) &
            (price_action_entries["Time"] >= threshold_time)
        ]

        if filtered_entries.empty:
            # Log and print warning if no valid entries are found
            logging.warning(f"No valid entries for {last_trading_day} after 9:20 AM.")
            print(f"No valid entries for {last_trading_day} after 9:20 AM.")
            return  # Exit the function

        # Sort by 'Time' and get the earliest entry for each ticker
        earliest_entries = filtered_entries.sort_values(by="Time").groupby("Ticker").first().reset_index()

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

        else:
            # Log and print warning if 'Trend Type' column is missing
            logging.warning("Trend Type column is missing in the input file. Skipping processing.")
            print("Trend Type column is missing in the input file. Skipping processing.")
            return  # Exit the function

        # **Ensure 'Time' is datetime with timezone info for accurate deduplication**
        combined_entries['Time'] = pd.to_datetime(combined_entries['Time'])  # Convert to datetime

        # Append or create the papertrade file
        if os.path.exists(output_file):
            # Read existing papertrade entries from CSV
            existing_papertrade = pd.read_csv(output_file)
            # Ensure 'Time' is datetime with timezone info
            existing_papertrade = normalize_time(existing_papertrade)
            # Combine existing entries with new combined entries
            all_papertrade_entries = pd.concat([existing_papertrade, combined_entries], ignore_index=True)
            # Drop duplicates based on 'Ticker' and 'Time' columns
            all_papertrade_entries = all_papertrade_entries.drop_duplicates(subset=['Ticker', 'Time'])
            # Save the combined entries back to the papertrade CSV
            all_papertrade_entries.to_csv(output_file, index=False)
        else:
            # If papertrade file doesn't exist, save the combined entries directly
            combined_entries.to_csv(output_file, index=False)

        logging.info(f"Combined papertrade data saved to {output_file}")  # Log saving
        print(f"Combined papertrade data saved to {output_file}")        # Print confirmation

    except Exception as e:
        # Log and print error if any exception occurs
        logging.error(f"Error creating combined papertrade file: {e}")
        print(f"Error creating combined papertrade file: {e}")

# Main execution function
def main():
    """
    Main function to execute the trading script.
    - Determines the last trading day.
    - Selects stocks based on RSI criteria.
    - Runs the price action entry detection.
    - Schedules the entry detection to run every 15 minutes starting at 9:15:05 AM IST.
    """
    today = dt.datetime.today()  # Current datetime
    india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST
    current_time = dt.datetime.now(india_tz)  # Current time in IST

    # Define start time at 9:15:05 AM IST
    start_time = dt.datetime.combine(today.date(), datetime_time(9, 15, 5, tzinfo=india_tz))

    # Get last trading day and select stocks based on selection criteria
    last_trading_day = get_last_trading_day()
    selected_stocks = select_stocks(last_trading_day)

    # Run the function once immediately after selecting stocks
    run_for_last_30_trading_days(selected_stocks)

    # Define a job function to run the entry detection
    def job():
        run_for_last_30_trading_days(selected_stocks)

    # Calculate the delay until the first scheduled run
    if current_time < start_time:
        delay_seconds = (start_time - current_time).total_seconds()  # Seconds until start_time
        logging.info(f"Waiting for {delay_seconds} seconds until the first scheduled run.")
        time_module.sleep(delay_seconds)  # Sleep until start_time

    # Schedule the job to run every 15 minutes
    schedule.every(15).minutes.do(job)

    logging.info("Scheduled `run_for_last_30_trading_days` to run every 15 minutes starting at 9:15:05 AM IST.")
    print("Scheduled `run_for_last_30_trading_days` to run every 15 minutes starting at 9:15:05 AM IST.")

    # Keep the script running to allow scheduled jobs to execute
    while True:
        schedule.run_pending()  # Run any pending scheduled jobs
        time_module.sleep(1)     # Sleep for a short time to prevent high CPU usage


if __name__ == "__main__":
    main()  # Execute the main function when the script is run
    
   
#clear_cache()
