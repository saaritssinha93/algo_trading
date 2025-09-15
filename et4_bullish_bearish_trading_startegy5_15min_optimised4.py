# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:58:21 2024
Updated by: Saarit Shankar Sinha

Description:
Fetches intraday stock data, calculates technical indicators,
detects entry signals based on trend conditions, and manages data
storage and logging. "Time" references standardized to "date".
Only new data is added every 15 minutes without refreshing or changing older data.
Calculations are performed using a temporary historical CSV, and only the latest indicators
are appended to the main indicators CSV.
"""

# ================================
# Import Modules
# ================================

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect.exceptions import NetworkException, KiteException
import pandas as pd
import numpy as np
import pytz
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
import traceback
from requests.exceptions import HTTPError
import shutil

# ================================
# Global Configuration
# ================================

# Define the timezone object for IST globally
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST

# Import selected stock tickers
from et4_filtered_stocks_market_cap import selected_stocks  # List of stock tickers

# Define cache directories
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Limit concurrent API calls
api_semaphore = threading.Semaphore(2)

# ================================
# Setup Logging
# ================================

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Set the lowest level to WARNING

# Create handlers
file_handler = logging.FileHandler("trading_script_main.log")
file_handler.setLevel(logging.WARNING)  # Log only warnings and errors to the file

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Log only warnings and errors to the console

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Log initialization message as WARNING since INFO won't be logged
logging.warning("Logging initialized successfully.")
print("Logging initialized and script execution started.")



'''

# ================================
# Configure Logging to File and Console
# ================================

# Define log format with more verbosity
log_format = (
    '%(asctime)s - %(levelname)s - %(name)s - '
    '%(funcName)s - %(lineno)d - %(message)s'
)

# Initialize logging configuration before any logging calls
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for full verbosity
    format=log_format,
    handlers=[
        logging.FileHandler("trading_script_main.log", mode='a'),
        logging.StreamHandler()
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Log initialization message
logger.info("Logging initialized successfully.")
print("Logging initialized and script execution started.")

# ================================
# Set Working Directory
# ================================

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info(f"Changed working directory to {cwd}.")
except FileNotFoundError:
    logger.error(f"Working directory {cwd} not found.")
    logger.debug(traceback.format_exc())
    raise
except Exception as e:
    logger.error(f"Unexpected error changing directory: {e}")
    logger.debug(traceback.format_exc())
    raise

'''

# ================================
# Rest of Your Script...
# ================================

# Example of additional logging within your functions
def example_function(param1, param2):
    logger.debug(f"Entering example_function with param1={param1}, param2={param2}")
    try:
        # Your function logic here
        result = param1 + param2
        logger.debug(f"Computed result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in example_function: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        logger.debug("Exiting example_function")
# ================================
# Define Market Holidays for 2024
# ================================

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

# ================================
# Helper Functions
# ================================

def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Normalize the 'date' column in a DataFrame to ensure consistent datetime format with the specified timezone.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the 'date' column.
        timezone (str): Timezone to localize/convert the 'date' column to. Defaults to 'Asia/Kolkata'.
        
    Returns:
        pd.DataFrame: DataFrame with the normalized 'date' column.
    """
    if 'date' not in df.columns:
        logging.error("The DataFrame does not contain a 'date' column.")
        raise KeyError("The DataFrame does not contain a 'date' column.")
    
    logging.debug(f"Normalizing 'date' column for DataFrame with columns: {df.columns.tolist()}")
    
    try:
        df['date'] = pd.to_datetime(df['date'], errors='raise')
    except Exception as e:
        logging.error(f"Error converting 'date' to datetime: {e}")
        logging.error(traceback.format_exc())
        raise
    
    # Ensure the timezone string is valid
    try:
        target_tz = pytz.timezone(timezone)
    except pytz.UnknownTimeZoneError:
        logging.error(f"Unknown timezone provided: {timezone}")
        raise
    
    # Check if 'date' is timezone-aware
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
    else:
        df['date'] = df['date'].dt.tz_convert(target_tz)
    
    # Check for any NaT (Not a Time) values after localization/conversion
    if df['date'].isnull().any():
        logging.warning(f"'date' column contains NaT values after normalization for timezone {timezone}.")
        # Decide on a strategy: drop these rows, fill them, or handle otherwise
        df = df.dropna(subset=['date'])
    
    logging.debug(f"'date' column after normalization:\n{df['date'].head()}")
    return df

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

def validate_dataframe(data, required_columns):
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        logging.error(f"DataFrame is missing required columns: {missing_columns}")
        return False
    return True


def send_notification(subject, message):
    """
    Send an email notification. Configure SMTP settings as per your email provider.

    Parameters:
        subject (str): Subject of the email.
        message (str): Body of the email.
    """
    import smtplib
    from email.mime.text import MIMEText

    # SMTP Configuration
    SMTP_SERVER = 'smtp.example.com'
    SMTP_PORT = 587
    SMTP_USERNAME = 'your_email@example.com'
    SMTP_PASSWORD = 'your_password'

    # Email Content
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = SMTP_USERNAME
    msg['To'] = 'recipient_email@example.com'

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info("Notification email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send notification email: {e}")

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
        kite = KiteConnect(api_key=key_secret[0])  # Create KiteConnect instance with API key
        kite.set_access_token(access_token)  # Set access token for the session
        logging.info("Kite session established successfully.")  # Log successful session setup
        print("Kite Connect session initialized successfully.")  # Print confirmation to console
        return kite  # Return the KiteConnect instance

    except FileNotFoundError as e:
        # Log and print error if files are not found
        logging.error(f"File not found: {e}")  # Log the missing file error
        print(f"Error: {e}")  # Print error to console
        raise  # Re-raise the exception to halt execution
    except Exception as e:
        # Log and print any other exceptions
        logging.error(f"Error setting up Kite session: {e}")  # Log any other setup errors
        print(f"Error: {e}")  # Print error to console
        raise  # Re-raise the exception to halt execution

# Initialize Kite session by setting up the KiteConnect instance
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

# Read tokens for shares using the list from et4_filtered_stocks_market_cap.py
shares_tokens = get_tokens_for_stocks(selected_stocks)

class APICallError(Exception):
    """Custom exception raised when API calls fail after all retry attempts."""
    pass

def make_api_call_with_retry(kite, token, from_date, to_date, interval, max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Make an API call with retries and exponential backoff.

    Parameters:
        kite (KiteConnect): KiteConnect instance.
        token (int): Instrument token.
        from_date (str): Start datetime for data fetching.
        to_date (str): End datetime for data fetching.
        interval (str): Data interval.
        max_retries (int): Maximum number of retries.
        initial_delay (int): Initial delay in seconds.
        backoff_factor (int): Backoff multiplier.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.

    Raises:
        APICallError: If all retries fail.
    """
    attempt = 0
    delay = initial_delay

    while attempt < max_retries:
        try:
            logging.info(f"API call attempt {attempt + 1} for token {token}.")
            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            if not data:
                logging.warning(f"No data returned from API for token {token} on attempt {attempt + 1}.")
                raise ValueError("Empty data received from API.")
            logging.info(f"Successfully fetched data for token {token} on attempt {attempt + 1}.")
            return pd.DataFrame(data)
        except NetworkException as net_exc:
            error_message = str(net_exc).lower()
            if 'too many requests' in error_message or 'rate limit' in error_message:
                logging.warning(f"Rate limit hit for token {token}. Attempt {attempt + 1} of {max_retries}. Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                logging.error(f"Network error for token {token}: {net_exc}")
                logging.error(traceback.format_exc())
                raise
        except KiteException as kite_exc:
            error_message = str(kite_exc).lower()
            if 'too many requests' in error_message or 'rate limit' in error_message:
                logging.warning(f"Rate limit exceeded for token {token}. Attempt {attempt + 1} of {max_retries}. Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                logging.error(f"KiteException for token {token}: {kite_exc}")
                logging.error(traceback.format_exc())
                raise
        except HTTPError as http_err:
            if http_err.response.status_code == 429:
                logging.warning(f"HTTP 429 Too Many Requests for token {token}. Attempt {attempt + 1} of {max_retries}. Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                logging.error(f"HTTP error occurred: {http_err} - Token: {token}")
                raise
        except Exception as e:
            logging.error(f"Unexpected error fetching data for token {token} on attempt {attempt + 1}: {e}")
            logging.error(traceback.format_exc())
            if isinstance(e, ValueError):
                logging.warning(f"Empty data received for token {token}. Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                raise

    logging.error(f"All {max_retries} retries failed for token {token}.")
    raise APICallError(f"Failed to fetch data for token {token} after {max_retries} retries.")

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

def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=5)):
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

def cache_update(ticker, data_type, fetch_function, cache_dir=CACHE_DIR, freshness_threshold=None, bypass_cache=False):
    """
    Update the cache for a given ticker and data type.
    If the cache is not fresh or bypass_cache is True, fetch fresh data and update the cache.

    Parameters:
        ticker (str): The trading symbol of the stock.
        data_type (str): Type of data ('historical' or 'intraday').
        fetch_function (function): Function to fetch the data.
        cache_dir (str): Directory path for caching data.
        freshness_threshold (timedelta, optional): Time threshold to determine freshness.
            Defaults to None, which assigns default based on data_type.
        bypass_cache (bool): Whether to bypass cache and fetch fresh data.

    Returns:
        pd.DataFrame: The cached or newly fetched data.
    """
    cache_path = cache_file_path(ticker, data_type)  # e.g., 'data_cache/GLAND_historical.csv'

    # Assign default freshness_threshold based on data_type if not provided
    if freshness_threshold is None:
        default_thresholds = {
            "historical": timedelta(minutes=600),  # 10 hours
            "intraday": timedelta(minutes=15)      # 15 minutes
        }
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=15))  # Default to 15 minutes if unknown type

    # **Modification: Only fetch new data based on last timestamp**
    if not bypass_cache and is_cache_fresh(cache_path, freshness_threshold):
        try:
            logging.info(f"Loading cached {data_type} data for {ticker}.")
            cached_data = pd.read_csv(cache_path, parse_dates=['date'])
            cached_data['date'] = pd.to_datetime(cached_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
            return cached_data
        except Exception as e:
            logging.error(f"Failed to load cache for {ticker} ({data_type}): {e}")
            logging.error(traceback.format_exc())
            # Proceed to fetch fresh data if cache loading fails

    # Fetch fresh data since cache is not fresh or bypass_cache is True
    try:
        logging.info(f"Fetching fresh {data_type} data for {ticker}.")
        fresh_data = fetch_function(ticker)
        if fresh_data is not None and not fresh_data.empty:
            # **Modification: Append only new data**
            if os.path.exists(cache_path):
                existing_data = pd.read_csv(cache_path, parse_dates=['date'])
                existing_data['date'] = pd.to_datetime(existing_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
                combined_data = pd.concat([existing_data, fresh_data]).drop_duplicates(subset='date').sort_values(by='date')
                combined_data.to_csv(cache_path, index=False)
                logging.info(f"Appended new {data_type} data for {ticker} to cache.")
                return combined_data
            else:
                fresh_data.to_csv(cache_path, index=False)
                logging.info(f"Cached fresh {data_type} data for {ticker} at {cache_path}.")
                return fresh_data
        else:
            logging.warning(f"No fresh {data_type} data fetched for {ticker}.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching fresh {data_type} data for {ticker}: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def fetch_intraday_data_with_historical(
    ticker, token, from_date_intra, to_date_intra, interval="15minute", 
    cache_dir=CACHE_DIR, freshness_threshold=timedelta(minutes=15), bypass_cache=False
):
    """
    Fetch intraday data for a ticker with historical data included for better indicator calculations.
    Implements retries with exponential backoff for API calls.
    """
    combined_indicators_file = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    
    logging.info(f"Checking combined indicators cache for {ticker}.")
    combined_cache_exists = os.path.exists(combined_indicators_file)
    
    # Step 1: Fetch Intraday Data using cache_update
    def fetch_intraday(ticker):
        """
        Fetch intraday data for the given ticker.

        Parameters:
            ticker (str): The trading symbol of the stock.

        Returns:
            pd.DataFrame: Intraday data DataFrame with indicators.
        """
        try:
            new_intraday_df = make_api_call_with_retry(
                kite=kite,
                token=token,
                from_date=from_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval
            )
            if new_intraday_df.empty:
                logging.warning(f"No new intraday data fetched for {ticker}.")
                return pd.DataFrame()
            else:
                intraday_data = pd.to_datetime(new_intraday_df['date']).dt.tz_convert('Asia/Kolkata').to_frame(name='date')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    intraday_data[col] = new_intraday_df[col]
                return intraday_data
        except Exception as e:
            logging.error(f"Error fetching intraday data for {ticker}: {e}")
            logging.error(traceback.format_exc())
            send_notification("Trading Script Error", f"Failed to fetch intraday data for {ticker}: {e}")
            return pd.DataFrame()
    
    intraday_data = cache_update(
        ticker=ticker,
        data_type="intraday",
        fetch_function=fetch_intraday,
        cache_dir=cache_dir,
        freshness_threshold=timedelta(minutes=15),  # 15 minutes freshness
        bypass_cache=True
    )
        
    # Step 2: Fetch Historical Data using cache_update
    def fetch_historical(ticker):
        """
        Fetch historical data for the given ticker.

        Parameters:
            ticker (str): The trading symbol of the stock.

        Returns:
            pd.DataFrame: Historical data DataFrame.
        """
        from_date_hist = from_date_intra - timedelta(days=30)  # Increased buffer for more comprehensive historical data
        to_date_hist = from_date_intra - timedelta(minutes=1)
        try:
            historical_df = make_api_call_with_retry(
                kite=kite,
                token=token,
                from_date=from_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval
            )
            if historical_df.empty:
                logging.warning(f"No historical data fetched for {ticker}.")
                return pd.DataFrame()
            else:
                historical_data = pd.to_datetime(historical_df['date']).dt.tz_convert('Asia/Kolkata').to_frame(name='date')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    historical_data[col] = historical_df[col]
                return historical_data
        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {e}")
            logging.error(traceback.format_exc())
            send_notification("Trading Script Error", f"Failed to fetch historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    historical_data = cache_update(
        ticker=ticker,
        data_type="historical",
        fetch_function=fetch_historical,
        cache_dir=cache_dir,
        freshness_threshold=timedelta(minutes=600),  # 10 hours
        bypass_cache=False  # Allow using fresh historical data when needed
    )
    
    # Step 3: Combine Historical and Intraday Data
    if historical_data.empty and intraday_data.empty:
        logging.warning(f"No data available for {ticker}. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Combining historical and intraday data for {ticker}.")
    combined_data = pd.concat([historical_data, intraday_data]).drop_duplicates(subset='date').sort_values(by='date')

    # Step 4: Recalculate Indicators using Temporary Historical CSV
    if not combined_data.empty:
        logging.info(f"Calculating indicators on combined data for {ticker}.")
        combined_data = sort_dataframe_by_date(combined_data)
        combined_data = recalc_indicators(combined_data)
        logging.info(f"Indicators calculated for {ticker}.")
    
    # Step 5: Extract the Latest Indicator Values (without 'Entry Signal')
    if not combined_data.empty:
        latest_indicators = combined_data.tail(1)[[
            'date', 'RSI', 'ATR', 'MACD', 'Signal_Line', 'Histogram',
            'Upper_Band', 'Lower_Band', 'Stochastic', 'VWAP',
            '20_SMA', 'Recent_High', 'Recent_Low', 'ADX'
        ]].copy()
        # **Add 'Entry Signal' with default value 'No'**
        latest_indicators['Entry Signal'] = 'No'
        logging.debug(f"Latest indicators for {ticker}:\n{latest_indicators}")
    else:
        latest_indicators = pd.DataFrame()

    # Step 6: Append Latest Indicators to Main Indicators CSV
    main_indicators_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    if not latest_indicators.empty:
        try:
            if os.path.exists(main_indicators_path):
                # Read existing main indicators
                existing_indicators = pd.read_csv(main_indicators_path, parse_dates=['date'])
                existing_indicators['date'] = pd.to_datetime(existing_indicators['date'], utc=True).dt.tz_convert('Asia/Kolkata')
                # Append the latest indicators
                combined_indicators = pd.concat([existing_indicators, latest_indicators], ignore_index=True)
                # Drop duplicates based on 'date'
                combined_indicators = combined_indicators.drop_duplicates(subset=['date'], keep='last')
                # Save back to main indicators CSV
                combined_indicators.to_csv(main_indicators_path, index=False)
                logging.info(f"Appended latest indicators for {ticker} to main indicators CSV.")
            else:
                # If main indicators CSV doesn't exist, create it with latest indicators
                latest_indicators.to_csv(main_indicators_path, index=False)
                logging.info(f"Created main indicators CSV for {ticker} with latest indicators.")
        except Exception as e:
            logging.error(f"Error appending indicators to main indicators CSV for {ticker}: {e}")
            logging.error(traceback.format_exc())
    else:
        logging.warning(f"No indicators to append for {ticker}.")
    
    return combined_data



def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    roll_up = up.rolling(window=timeperiod).mean()
    roll_down = down.rolling(window=timeperiod).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_atr(data: pd.DataFrame, timeperiod: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    data = data.copy()  # Ensure we're working on a copy
    data.loc[:, 'previous_close'] = data['close'].shift(1)
    data.loc[:, 'high_low'] = data['high'] - data['low']
    data.loc[:, 'high_pc'] = abs(data['high'] - data['previous_close'])
    data.loc[:, 'low_pc'] = abs(data['low'] - data['previous_close'])
    tr = data[['high_low', 'high_pc', 'low_pc']].max(axis=1)
    atr = tr.rolling(window=timeperiod, min_periods=1).mean()
    data = data.drop(['previous_close', 'high_low', 'high_pc', 'low_pc'], axis=1)
    return atr



def calculate_macd(data: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
    """Calculate MACD and Signal Line and return as a DataFrame."""
    ema_fast = data['close'].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd - signal_line
    return pd.DataFrame({
        'MACD': macd,
        'Signal_Line': signal_line,
        'Histogram': histogram
    })

def calculate_bollinger_bands(data: pd.DataFrame, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2, matype: int = 0) -> pd.DataFrame:
    """Calculate Bollinger Bands and return as a DataFrame."""
    sma = data['close'].rolling(window=timeperiod, min_periods=1).mean()
    std = data['close'].rolling(window=timeperiod, min_periods=1).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return pd.DataFrame({
        'Upper_Band': upper_band,
        'Lower_Band': lower_band
    })

def calculate_stochastic(data: pd.DataFrame, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> pd.Series:
    """Calculate Stochastic Oscillator (%K)."""
    low_min = data['low'].rolling(window=fastk_period, min_periods=1).min()
    high_max = data['high'].rolling(window=fastk_period, min_periods=1).max()
    percent_k = 100 * ((data['close'] - low_min) / (high_max - low_min + 1e-10))  # Avoid division by zero
    slowk = percent_k.rolling(window=slowk_period, min_periods=1).mean()
    # Optionally, calculate %D as a moving average of %K
    # percent_d = slowk.rolling(window=slowd_period, min_periods=1).mean()
    return slowk  # Returning %K; modify if %D is needed

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) for the given DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
    - period (int): The number of periods to use for ADX calculation.

    Returns:
    - pd.Series: ADX values.
    """
    data = data.copy()  # Ensure we're working on a copy
    
    # Ensure necessary columns are present
    if not {'high', 'low', 'close'}.issubset(data.columns):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
        
    # 1. Calculate True Range (TR)
    data['previous_close'] = data['close'].shift(1)
    data['TR'] = data[['high', 'low', 'previous_close']].apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['previous_close']),
            abs(row['low'] - row['previous_close'])
        ),
        axis=1
    )

    # 2. Calculate Directional Movement (+DM and -DM)
    data['+DM'] = np.where(
        (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
        np.where((data['high'] - data['high'].shift(1)) > 0, data['high'] - data['high'].shift(1), 0),
        0
    )

    data['-DM'] = np.where(
        (data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
        np.where((data['low'].shift(1) - data['low']) > 0, data['low'].shift(1) - data['low'], 0),
        0
    )

    # 3. Smooth TR, +DM, and -DM using Wilder's smoothing method
    # Initialize smoothed values
    data['TR_smooth'] = data['TR'].rolling(window=period, min_periods=period).sum()
    data['+DM_smooth'] = data['+DM'].rolling(window=period, min_periods=period).sum()
    data['-DM_smooth'] = data['-DM'].rolling(window=period, min_periods=period).sum()

    # After the initial period, continue smoothing
    for i in range(period, len(data)):
        if i == period:
            # Already calculated the initial sum
            continue
        else:
            data.at[data.index[i], 'TR_smooth'] = data.at[data.index[i-1], 'TR_smooth'] - (data.at[data.index[i-1], 'TR_smooth'] / period) + data.at[data.index[i], 'TR']
            data.at[data.index[i], '+DM_smooth'] = data.at[data.index[i-1], '+DM_smooth'] - (data.at[data.index[i-1], '+DM_smooth'] / period) + data.at[data.index[i], '+DM']
            data.at[data.index[i], '-DM_smooth'] = data.at[data.index[i-1], '-DM_smooth'] - (data.at[data.index[i-1], '-DM_smooth'] / period) + data.at[data.index[i], '-DM']

    # 4. Calculate +DI and -DI
    data['+DI'] = (data['+DM_smooth'] / data['TR_smooth']) * 100
    data['-DI'] = (data['-DM_smooth'] / data['TR_smooth']) * 100

    # 5. Calculate DX
    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'] + 1e-10)) * 100  # Avoid division by zero

    # 6. Calculate ADX
    data.loc[:, 'ADX'] = data['DX'].rolling(window=period, min_periods=period).mean()

    # After the initial period, continue smoothing ADX
    for i in range(2 * period, len(data)):
        data.at[data.index[i], 'ADX'] = (
            (data.at[data.index[i-1], 'ADX'] * (period - 1)) + data.at[data.index[i], 'DX']
        ) / period

    # Clean up intermediate columns
    data = data.drop(['previous_close', 'TR', '+DM', '-DM', 'TR_smooth', '+DM_smooth', '-DM_smooth', '+DI', '-DI', 'DX'], axis=1)
    return data['ADX']

def recalc_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate all required indicators for the given data.
    This function recalculates indicators only for new data entries.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing price and volume data.
    
    Returns:
        pd.DataFrame: DataFrame with recalculated indicators.
    """
    # Reset index to ensure uniqueness
    data = data.reset_index(drop=True)
    
    # Create a copy to avoid modifying the original DataFrame
    data = data.copy()

    # Calculate indicators only for new data (assumed to be at the end)
    new_rows = data.tail(100).copy()  # Ensure new_rows is a copy

    logging.debug(f"Number of new rows: {len(new_rows)}")

    # Check for NaNs in critical columns
    critical_columns = ['close', 'high', 'low', 'volume']
    if new_rows[critical_columns].isnull().any().any():
        logging.warning("Found NaN values in critical columns. Applying forward fill.")
        new_rows[critical_columns] = new_rows[critical_columns].fillna(method='ffill')

    # Ensure all critical columns are numeric
    for col in critical_columns:
        if not pd.api.types.is_numeric_dtype(new_rows[col]):
            logging.error(f"Column '{col}' must be numeric. Found dtype: {new_rows[col].dtype}")
            raise TypeError(f"Column '{col}' must be numeric.")

    # Calculate RSI
    try:
        data.loc[new_rows.index, 'RSI'] = calculate_rsi(new_rows['close'])
        logging.debug("RSI calculated successfully.")
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        raise

    # Calculate ATR
    try:
        data.loc[new_rows.index, 'ATR'] = calculate_atr(new_rows)
        logging.debug("ATR calculated successfully.")
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        raise

    # Calculate MACD and assign directly
    try:
        macd_df = calculate_macd(new_rows)  # Returns DataFrame with 'MACD', 'Signal_Line', 'Histogram'
        logging.debug(f"MACD DataFrame shape: {macd_df.shape}, Expected: {len(new_rows)} rows.")
        assert macd_df.shape[0] == len(new_rows), "Mismatch in MACD DataFrame row count."
        data.loc[new_rows.index, ['MACD', 'Signal_Line', 'Histogram']] = macd_df
        logging.debug("MACD assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        raise

    # Calculate Bollinger Bands and assign directly
    try:
        bollinger_df = calculate_bollinger_bands(new_rows)  # Returns DataFrame with 'Upper_Band', 'Lower_Band'
        logging.debug(f"Bollinger Bands DataFrame shape: {bollinger_df.shape}, Expected: {len(new_rows)} rows.")
        assert bollinger_df.shape[0] == len(new_rows), "Mismatch in Bollinger Bands DataFrame row count."
        data.loc[new_rows.index, ['Upper_Band', 'Lower_Band']] = bollinger_df
        logging.debug("Bollinger Bands assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
        raise

    # Calculate Stochastic Oscillator
    try:
        stochastic = calculate_stochastic(new_rows)
        logging.debug(f"Stochastic Oscillator Series length: {len(stochastic)}, Expected: {len(new_rows)} rows.")
        assert len(stochastic) == len(new_rows), "Mismatch in Stochastic Oscillator Series row count."
        data.loc[new_rows.index, 'Stochastic'] = stochastic
        logging.debug("Stochastic Oscillator assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating Stochastic Oscillator: {e}")
        raise

    # Calculate VWAP
    try:
        vwap = (new_rows['close'] * new_rows['volume']).cumsum() / new_rows['volume'].cumsum()
        logging.debug(f"VWAP Series length: {len(vwap)}, Expected: {len(new_rows)} rows.")
        assert len(vwap) == len(new_rows), "Mismatch in VWAP Series row count."
        data.loc[new_rows.index, 'VWAP'] = vwap
        logging.debug("VWAP assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating VWAP: {e}")
        raise

    # Calculate 20 SMA
    try:
        sma20 = new_rows['close'].rolling(window=20, min_periods=1).mean()
        logging.debug(f"20_SMA Series length: {len(sma20)}, Expected: {len(new_rows)} rows.")
        assert len(sma20) == len(new_rows), "Mismatch in 20_SMA Series row count."
        data.loc[new_rows.index, '20_SMA'] = sma20
        logging.debug("20_SMA assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating 20_SMA: {e}")
        raise

    # Calculate Recent High and Low
    try:
        recent_high = new_rows['high'].rolling(window=5, min_periods=1).max()
        recent_low = new_rows['low'].rolling(window=5, min_periods=1).min()
        logging.debug(f"Recent_High Series length: {len(recent_high)}, Expected: {len(new_rows)} rows.")
        logging.debug(f"Recent_Low Series length: {len(recent_low)}, Expected: {len(new_rows)} rows.")
        assert len(recent_high) == len(new_rows), "Mismatch in Recent_High Series row count."
        assert len(recent_low) == len(new_rows), "Mismatch in Recent_Low Series row count."
        data.loc[new_rows.index, 'Recent_High'] = recent_high
        data.loc[new_rows.index, 'Recent_Low'] = recent_low
        logging.debug("Recent High and Low assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating Recent High/Low: {e}")
        raise

    # Calculate ADX
    try:
        adx = calculate_adx(new_rows, period=14)
        logging.debug(f"ADX Series length: {len(adx)}, Expected: {len(new_rows)} rows.")
        assert len(adx) == len(new_rows), "Mismatch in ADX Series row count."
        data.loc[new_rows.index, 'ADX'] = adx
        logging.debug("ADX assigned successfully.")
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        raise
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}")
        raise

    # Handle any potential NaN values resulting from indicator calculations
    data = data.ffill().bfill()
    return data

def check_duplicates(data: pd.DataFrame, ticker: str):
    """
    Check and handle duplicate index labels and duplicate rows based on 'date'.

    Parameters:
        data (pd.DataFrame): DataFrame to check.
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: DataFrame with duplicates handled.
    """
    # Check for duplicate index labels
    if data.index.duplicated().any():
        duplicated_indices = data.index[data.index.duplicated()].unique()
        logging.error(f"Duplicate index labels found for ticker {ticker}: {duplicated_indices}")
        # Optionally, remove duplicates
        data = data[~data.index.duplicated(keep='last')]
        logging.info(f"Removed duplicate indices for ticker {ticker}.")

    # Check for duplicate rows based on 'date'
    if data.duplicated(subset=['date']).any():
        duplicated_dates = data['date'][data.duplicated(subset=['date'])].unique()
        logging.error(f"Duplicate date entries found for ticker {ticker}: {duplicated_dates}")
        # Optionally, remove duplicates or aggregate
        data = data.drop_duplicates(subset=['date'], keep='last')
        logging.info(f"Removed duplicate date entries for ticker {ticker}.")

    return data



import logging
import pandas as pd
from datetime import datetime, time as datetime_time
import pytz

# Assuming these functions and variables are defined elsewhere in your script
# def fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra):
# def calculate_daily_change(today_data):
# def apply_trend_conditions(today_data, daily_change, ticker, last_trading_day):
# def mark_entry_signals(combined_data, entry_times, india_tz):
# def send_notification(subject, message):
# api_semaphore = ...
# shares_tokens = {...}

def process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day, india_tz):
    try:
        with api_semaphore:
            token = shares_tokens.get(ticker)
            if not token:
                logging.warning(f"No token found for ticker {ticker}. Skipping.")
                return None, []
            
            # Fetch and update combined data
            combined_data = fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra)
            if combined_data.empty:
                logging.warning(f"No combined data available for {ticker}. Skipping.")
                return None, []
            
            # Ensure 'Entry Signal' column exists
            if "Entry Signal" not in combined_data.columns:
                combined_data["Entry Signal"] = "No"
                logging.info(f"'Entry Signal' column missing for {ticker}. Initialized to 'No'.")
            else:
                # Ensure no invalid data exists in 'Entry Signal' column
                valid_signals = {"Yes", "No"}
                invalid_signals = set(combined_data["Entry Signal"].unique()) - valid_signals
                if invalid_signals:
                    logging.warning(f"Invalid 'Entry Signal' values {invalid_signals} in {ticker}. Correcting to 'No'.")
                    combined_data["Entry Signal"] = combined_data["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
            
            # Detect entry signals
            today_start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
            today_end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
            today_data = combined_data[(combined_data['date'] >= today_start_time) & (combined_data['date'] <= today_end_time)].copy()
            
            if today_data.empty:
                logging.warning(f"No intraday data available after combining for {ticker}. Skipping.")
                return combined_data, []
            
            # Calculate daily percentage change using today's data
            daily_change = calculate_daily_change(today_data)
            
            # Apply trend conditions to detect entries
            ticker_entries = apply_trend_conditions(today_data, daily_change, ticker, last_trading_day)
            
            # Collect entry times
            entry_times = []
            for entry in ticker_entries:
                try:
                    entry_time = pd.to_datetime(entry["date"])
                    if entry_time.tzinfo is None:
                        entry_time = india_tz.localize(entry_time)
                    else:
                        entry_time = entry_time.astimezone(india_tz)
                    entry_times.append(entry_time)
                except Exception as e:
                    logging.error(f"Invalid date format for entry in {ticker}: {entry['date']}. Error: {e}")
                    continue
            
            # Mark 'Entry Signal' only for new entries
            mark_entry_signals(combined_data, entry_times, india_tz)
            
            # Validation Before Saving
            if "Entry Signal" not in combined_data.columns:
                logging.error(f"'Entry Signal' column is missing after marking for {ticker}.")
                raise KeyError(f"'Entry Signal' column is missing after marking for {ticker}.")
            
            # Log the unique values in 'Entry Signal' before saving
            logging.debug(f"'Entry Signal' values for {ticker} before saving: {combined_data['Entry Signal'].unique()}")
            
            # **Save the entire combined_data back to main_indicators.csv to persist 'Entry Signal'**
            main_indicators_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            combined_data.to_csv(main_indicators_path, index=False)
            logging.info(f"Saved combined data with 'Entry Signal' for {ticker} to main indicators CSV.")
            
            return combined_data, ticker_entries  # Return the DataFrame and list of entries

    except KeyError as ke:
        logging.error(f"Error processing ticker {ticker}: {ke}")
        send_notification("Trading Script Error", f"Error processing ticker {ticker}: {ke}")
        return None, []
    except Exception as e:
        logging.error(f"Unexpected error processing ticker {ticker}: {e}")
        logging.error(traceback.format_exc())
        send_notification("Trading Script Error", f"Unexpected error processing ticker {ticker}: {e}")
        return None, []



def validate_and_correct_main_indicators(selected_stocks, india_tz):
    """
    Validate that all main_indicators CSV files include the 'Entry Signal' column.
    If missing, add the column with default value 'No'.
    
    Parameters:
        selected_stocks (list): List of stock tickers.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.
    """
    for ticker in selected_stocks:
        main_indicators_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        if os.path.exists(main_indicators_path):
            try:
                data = pd.read_csv(main_indicators_path, parse_dates=['date'])
                if "Entry Signal" not in data.columns:
                    data["Entry Signal"] = "No"
                    data.to_csv(main_indicators_path, index=False)
                    logging.info(f"Added missing 'Entry Signal' column for {ticker} in {main_indicators_path}.")
                else:
                    # Ensure only 'Yes' or 'No' are present
                    valid_signals = {"Yes", "No"}
                    invalid_signals = set(data["Entry Signal"].unique()) - valid_signals
                    if invalid_signals:
                        data["Entry Signal"] = data["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
                        data.to_csv(main_indicators_path, index=False)
                        logging.info(f"Corrected invalid 'Entry Signal' values for {ticker} in {main_indicators_path}.")
            except Exception as e:
                logging.error(f"Error validating {main_indicators_path}: {e}")
                logging.error(traceback.format_exc())
                send_notification("Trading Script Error", f"Error validating {main_indicators_path}: {e}")
        else:
            logging.warning(f"Main indicators file for {ticker} does not exist at {main_indicators_path}.")


def calculate_indicators(data, period):
    """
    Calculate technical indicators for the given DataFrame.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing stock data.
    - period (int): The rolling window period for calculations.
    
    Returns:
    - pd.DataFrame: The DataFrame with new indicators added and unnecessary columns dropped.
    """
    try:
        # Ensure you're working with a copy to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Calculate smoothed DM values
        data.loc[:, '-DM_smooth'] = data['-DM'].rolling(window=period, min_periods=period).sum()
        data.loc[:, '+DI'] = (data['+DM_smooth'] / data['TR_smooth']) * 100
        data.loc[:, '-DI'] = (data['-DM_smooth'] / data['TR_smooth']) * 100
        data.loc[:, 'DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'] + 1e-10)) * 100  # Avoid division by zero
        data.loc[:, 'ADX'] = data['DX'].rolling(window=period, min_periods=period).mean()
        
        # Drop unnecessary columns by assigning back to data
        columns_to_drop = ['previous_close', 'TR', '+DM', '-DM', 'TR_smooth', '+DM_smooth', '-DM_smooth', '+DI', '-DI', 'DX']
        data = data.drop(columns=columns_to_drop)
        
        return data
    
    except KeyError as ke:
        logging.error(f"Missing expected column: {ke}")
        send_notification("Trading Script Error", f"Missing expected column: {ke}")
        raise
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        logging.error(traceback.format_exc())
        send_notification("Trading Script Error", f"Error calculating indicators: {e}")
        raise

def mark_entry_signals(data, entry_times, india_tz):
    """
    Mark 'Entry Signal' as 'Yes' for detected entry times in the data.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing stock data.
    - entry_times (list of datetime): List of datetime objects representing entry times.
    - india_tz (pytz.timezone): The timezone for localization.
    
    Returns:
    - None: Modifies the DataFrame in place.
    """
    try:
        if entry_times:
            # Ensure 'date' column is timezone-aware
            if data['date'].dt.tz is None:
                data['date'] = data['date'].dt.tz_localize(india_tz)
            else:
                data['date'] = data['date'].dt.tz_convert(india_tz)
            
            # Log entry_times and data['date'] for debugging
            logging.debug(f"Entry Times for marking 'Yes': {entry_times}")
            logging.debug(f"Sample 'date' entries:\n{data['date'].head()}")
            
            # Mark 'Entry Signal' as 'Yes' where the 'date' matches any of the entry times
            data.loc[data['date'].isin(entry_times), 'Entry Signal'] = 'Yes'
            logging.debug(f"Marked 'Entry Signal' for times: {entry_times}")
        else:
            logging.debug("No entry times detected; no 'Entry Signal' marked.")
    
    except Exception as e:
        logging.error(f"Error marking entry signals: {e}")
        logging.error(traceback.format_exc())
        send_notification("Trading Script Error", f"Error marking entry signals: {e}")
        raise






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

def apply_trend_conditions(data: pd.DataFrame, daily_change: float, ticker: str, last_trading_day: pd.Timestamp) -> list:
    """
    Detects Bullish or Bearish trend entries based on defined criteria.
    """
    trend_entries = []
    try:
        latest_data = data.iloc[-1]

        # Define common condition for strong trend using ADX
        strong_trend = latest_data["ADX"] > 25
        logging.debug(f"{ticker} - Latest ADX: {latest_data['ADX']}, Strong Trend: {strong_trend}")

        if strong_trend:
            if (
                daily_change > 1.0
                and latest_data["RSI"] > 25
                and latest_data["close"] > latest_data["VWAP"]
            ):
                # Bullish trend detected
                logging.debug(f"{ticker} - Bullish trend detected.")
                trend_entries.extend(
                    screen_trend(
                        data, "Bullish", daily_change, 1.5, ticker, last_trading_day
                    )
                )
            elif (
                daily_change < -1.25
                and latest_data["RSI"] < 75
                and latest_data["close"] < latest_data["VWAP"]
            ):
                # Bearish trend detected
                logging.debug(f"{ticker} - Bearish trend detected.")
                trend_entries.extend(
                    screen_trend(
                        data, "Bearish", daily_change, 1.5, ticker, last_trading_day
                    )
                )
    except Exception as e:
        logging.error(f"Error applying trend conditions for {ticker}: {e}")
        logging.error(traceback.format_exc())

    return trend_entries



def screen_trend(data: pd.DataFrame, trend_type: str, change: float, volume_multiplier: float, ticker: str, last_trading_day: pd.Timestamp) -> list:
    """
    Identifies specific trend entry points (Pullback/Breakout or Pullback/Breakdown) based on trend type,
    ensuring that ADX > 25 is a prerequisite for all conditions.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing price and indicator data.
    - trend_type (str): "Bullish" or "Bearish".
    - change (float): Daily percentage change.
    - volume_multiplier (float): Multiplier for average volume.
    - ticker (str): Stock ticker symbol.
    - last_trading_day (pd.Timestamp): The date of the last trading day.
    
    Returns:
    - list: List of detected trend entries.
    """
    trend_entries = []
    try:
        conditions = {}

        if trend_type == "Bullish":
            conditions = {
                "Pullback": (
                    (data["low"] >= data["VWAP"] * 0.975)
                    & (data["close"] > data["open"])
                    & (data["MACD"] > data["Signal_Line"])
                    & (data["MACD"] > 0)
                    & (data["close"] > data["20_SMA"])
                    & (data["Stochastic"] < 45)
                    & (data["Stochastic"].diff() > 0)
                    & (data["ADX"] > 25)  # ADX condition added
                ),
                "Breakout": (
                    (data["close"] > data["high"].rolling(window=10).max() * 1.015)
                    & (data["volume"] > (volume_multiplier * 2.0) * data["volume"].rolling(window=10).mean())
                    & (data["MACD"] > data["Signal_Line"])
                    & (data["MACD"] > 0)
                    & (data["close"] > data["20_SMA"])
                    & (data["Stochastic"] > 55)
                    & (data["Stochastic"].diff() > 0)
                    & (data["ADX"] > 25)  # ADX condition added
                ),
            }
        else:  # Bearish
            conditions = {
                "Pullback": (
                    (data["high"] <= data["VWAP"] * 1.01)
                    & (data["close"] < data["open"])
                    & (data["MACD"] < data["Signal_Line"])
                    & (data["MACD"] < 0)
                    & (data["close"] < data["20_SMA"])
                    & (data["Stochastic"] > 60)
                    & (data["Stochastic"].diff() < 0)
                    & (data["ADX"] > 25)  # ADX condition added
                ),
                "Breakdown": (
                    (data["close"] < data["low"].rolling(window=10).min() * 0.97)
                    & (data["volume"] > (volume_multiplier * 2.0) * data["volume"].rolling(window=10).mean())
                    & (data["MACD"] < data["Signal_Line"])
                    & (data["MACD"] < 0)
                    & (data["close"] < data["20_SMA"])
                    & (data["Stochastic"] < 40)
                    & (data["Stochastic"].diff() < 0)
                    & (data["ADX"] > 25)  # ADX condition added
                ),
            }

        # Iterate through each condition type
        for entry_type, condition in conditions.items():
            # Filter data based on condition and last trading day
            selected = data.loc[condition & (data['date'].dt.date == last_trading_day)]

            logging.debug(f"Entries for {entry_type} condition: {len(selected)}")

            if not selected.empty:
                # Vectorized creation of trend entries
                entries = selected.apply(lambda row: {
                    "Ticker": ticker,
                    "date": row["date"].strftime("%Y-%m-%d %H:%M:%S%z") if isinstance(row["date"], pd.Timestamp) else logging.error(f"'date' is not a datetime object for ticker {ticker}. Value: {row['date']}"),
                    "Entry Type": entry_type,
                    "Trend Type": trend_type,
                    "Price": round(row["close"], 2),
                    "Daily Change %": round(change, 2),
                    "VWAP": round(row["VWAP"], 2) if "Pullback" in entry_type else None,
                    "ADX": round(row["ADX"], 2),
                }, axis=1).tolist()

                # Filter out any entries where 'date' was not a Timestamp
                entries = [entry for entry in entries if isinstance(entry["date"], str)]
                trend_entries.extend(entries)

    except Exception as e:
        logging.error(f"Error screening trends for {ticker}: {e}")
        logging.error(traceback.format_exc())

    return trend_entries





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
        with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers as needed
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
                    send_notification("Trading Script Error", f"Error processing ticker {ticker}: {e}")
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")
        send_notification("Trading Script Error", f"Error during parallel processing: {e}")

    return entries_dict, processed_data_dict



LAST_RECALC_TIMESTAMP_FILE = os.path.join(CACHE_DIR, "last_signal_id_timestamp.txt")

# ------------------ Helper Functions ------------------

def read_last_signal_id_timestamp() -> datetime:
    """
    Placeholder function to read the last Signal_ID timestamp.
    Implement this function based on your storage mechanism.
    """
    try:
        with open('last_signal_id_timestamp.txt', 'r') as file:
            timestamp_str = file.read().strip()
            return datetime.fromisoformat(timestamp_str)
    except FileNotFoundError:
        return None

def write_last_signal_id_timestamp():
    """
    Placeholder function to write the current timestamp as the last Signal_ID timestamp.
    Implement this function based on your storage mechanism.
    """
    now = datetime.now(pytz.UTC).isoformat()
    with open('last_signal_id_timestamp.txt', 'w') as file:
        file.write(now)
# ------------------ Ensuring Indicators ------------------

def ensure_all_indicators(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Modified logic to only recalculate indicators for rows without Signal_ID if within 15 minutes of last run.

    Parameters:
    - data (pd.DataFrame): DataFrame containing price and volume data.
    - ticker (str): Stock ticker symbol.

    Returns:
    - pd.DataFrame: DataFrame with indicators calculated.
    """
    required_columns = {
        'RSI', 'ATR', 'MACD', 'Signal_Line',
        'Upper_Band', 'Lower_Band', 'VWAP',
        '20_SMA', 'Recent_High', 'Recent_Low', 'Stochastic', 'ADX'
    }

    # Ensure all required columns are present; if not, initialize them
    for col in required_columns:
        if col not in data.columns:
            data[col] = np.nan

    # If Signal_ID not present, all rows are considered new
    if 'Signal_ID' not in data.columns:
        data['Signal_ID'] = None

    last_timestamp = read_last_signal_id_timestamp()
    india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST
    now = datetime.now(india_tz)  # Current time in IST
    within_15_min = False
    if last_timestamp and (now - last_timestamp) < timedelta(minutes=15):
        within_15_min = True

    # Separate old rows (with Signal_ID) and new rows (without Signal_ID)
    old_rows = data[data['Signal_ID'].notna()].copy()
    new_rows = data[data['Signal_ID'].isna()].copy()

    if within_15_min:
        # Within 15 min of last full recalculation
        if new_rows.empty:
            # No new rows, no recalculation needed
            logging.info(f"No new rows for {ticker} within 15 minutes, skipping recalculation.")
            return data
        else:
            # Recalculate indicators only for new rows
            # However, some indicators need historical context. To correctly calculate them,
            # combine old and new data, then recalculate indicators for the entire set.
            combined = pd.concat([old_rows, new_rows]).sort_values(by='date').reset_index(drop=True)

            # Check and remove duplicates
            combined = check_duplicates(combined, ticker)

            # Calculate indicators for entire combined data
            combined = recalc_indicators(combined)

            # Assign Signal_ID only to new rows
            combined['Signal_ID'] = combined.apply(
                lambda row: row['Signal_ID'] if pd.notnull(row['Signal_ID']) else f"{ticker}-{row['date'].isoformat()}-{row.name}",
                axis=1
            )

            # Update timestamp
            write_last_signal_id_timestamp()
            return combined
    else:
        # More than 15 minutes have passed or first run
        # Option: Recalculate for all data or just new rows.
        # Here, let's do full recalculation for all rows to simplify:
        combined = data.sort_values(by='date').reset_index(drop=True)

        # Check and remove duplicates
        combined = check_duplicates(combined, ticker)

        # Recalculate indicators
        combined = recalc_indicators(combined)

        # Assign Signal_ID to any rows that don't have it
        combined['Signal_ID'] = combined.apply(
            lambda row: row['Signal_ID'] if pd.notnull(row['Signal_ID']) else f"{ticker}-{row['date'].isoformat()}-{row.name}",
            axis=1
        )

        # Update timestamp after full recalculation
        write_last_signal_id_timestamp()
        return combined

def combine_and_validate_data(intraday_data, ticker, india_tz):
    """
    Combine historical and intraday data, ensure data integrity, and calculate missing indicators.
    
    Parameters:
        intraday_data (pd.DataFrame): Intraday data fetched for the ticker.
        ticker (str): The trading symbol of the stock.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.
    
    Returns:
        pd.DataFrame: Combined and validated data.
    """
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
        logging.warning(f"Missing required columns for {ticker}: {missing_columns}. Recalculating indicators.")
        # Attempt to calculate missing indicators
        data = ensure_all_indicators(data, ticker)
        if data is None:
            logging.error(f"Failed to calculate indicators for {ticker}. Skipping.")
            return None

    # Convert 'date' column to timezone-aware datetime
    try:
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(india_tz)
    except Exception as e:
        logging.error(f"Error converting 'date' to timezone-aware datetime for {ticker}: {e}")
        logging.error(traceback.format_exc())
        return None

    # Sort the data by 'date' in ascending order without setting it as index
    data = sort_dataframe_by_date(data)

    # Initialize `Entry Signal` column as "No" by default if missing
    if "Entry Signal" not in data.columns:
        data["Entry Signal"] = "No"
        logging.info(f"'Entry Signal' column missing for {ticker}. Initialized to 'No'.")
    else:
        # Ensure no invalid data exists in 'Entry Signal' column
        data["Entry Signal"] = data["Entry Signal"].apply(lambda x: x if x in ["Yes", "No"] else "No")

    # **New Logging Added Here:**
    logging.debug(f"DataFrame for {ticker} after validation and sorting:\n{data.head()}")

    return data



def find_price_action_entry(last_trading_day):
    india_tz = pytz.timezone('Asia/Kolkata')

    # Prepare historical and intraday time ranges
    from_date_hist, to_date_hist, from_date_intra, to_date_intra = prepare_time_ranges(last_trading_day, india_tz)
    logging.info(f"Fetching historical data from {from_date_hist} to {to_date_hist}")
    logging.info(f"Fetching intraday data from {from_date_intra} to {to_date_intra}")

    # Wrapper for processing each ticker
    def process_ticker_wrapper(ticker):
        return process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day, india_tz)

    entries_dict, processed_data_dict = process_all_tickers(selected_stocks, process_ticker_wrapper)
    logging.info("Entry signal detection completed.")

    # Define final columns for consistent CSV structure
    final_columns = ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "ADX", "Signal_ID"]
    today_str = last_trading_day.strftime("%Y-%m-%d")

    # Aggregate all entries into a single DataFrame
    all_entries = []
    for ticker, entries in entries_dict.items():
        for entry in entries:
            all_entries.append({
                "Ticker": entry.get("Ticker"),
                "date": entry.get("date"),  # Ensure 'date' is correctly set
                "Entry Type": entry.get("Entry Type"),
                "Trend Type": entry.get("Trend Type"),
                "Price": entry.get("Price"),
                "Daily Change %": entry.get("Daily Change %"),
                "VWAP": entry.get("VWAP"),
                "ADX": entry.get("ADX"),
                "Signal_ID": f"{ticker}-{entry.get('date')}"  # Generate unique Signal_ID
            })

    # If no entries found, return empty DataFrame
    if not all_entries:
        logging.info("No entry signals detected across all tickers.")
        return pd.DataFrame(columns=final_columns)

    entries_df = pd.DataFrame(all_entries)

    # Normalize 'date'
    try:
        entries_df = normalize_time(entries_df)
    except Exception as e:
        logging.error(f"Normalization failed for aggregated entries: {e}")
        return pd.DataFrame()

    # Validate 'date' column presence
    if 'date' not in entries_df.columns:
        logging.error("Aggregated Entries DataFrame is missing the 'date' column.")
        raise KeyError("Aggregated Entries DataFrame is missing the 'date' column.")

    # Validate DataFrame structure
    expected_columns = final_columns  # Same as final_columns in this context
    if not validate_dataframe(entries_df, expected_columns):
        logging.error("Aggregated Entries DataFrame failed validation.")
        raise ValueError("Aggregated Entries DataFrame failed validation.")

    # Reassign Signal_ID to ensure uniqueness if needed
    entries_df["Signal_ID"] = entries_df.apply(
        lambda row: f"{row['Ticker']}-{row['date'].astimezone(pytz.UTC).isoformat()}",
        axis=1
    )

    # Ensure all final columns are present, then reorder
    for col in final_columns:
        if col not in entries_df.columns:
            entries_df[col] = ""
    entries_df = entries_df[final_columns]

    # Function to unify existing CSVs with final_columns and Signal_ID
    def unify_existing_csv(filename):
        if os.path.exists(filename):
            existing = pd.read_csv(filename, parse_dates=['date'])
            existing = normalize_time(existing)
    
            for c in final_columns:
                if c not in existing.columns:
                    existing[c] = ""
    
            existing = existing[final_columns]
    
            # If Signal_ID missing, create it
            if 'Signal_ID' not in existing.columns:
                existing["Signal_ID"] = existing.apply(
                    lambda row: f"{row['Ticker']}-{row['date'].astimezone(pytz.UTC).isoformat()}",
                    axis=1
                )
    
            # Rewrite file to ensure consistent columns and Signal_ID
            existing.to_csv(filename, index=False)
            return pd.read_csv(filename, parse_dates=['date'])
        return None

    # Save to individual ticker CSV files
    for ticker in entries_df['Ticker'].unique():
        ticker_filename = f"papertrade_{ticker}_{today_str}.csv"  # Modified filename to include ticker
        ticker_entries = entries_df[entries_df['Ticker'] == ticker].copy()

        for c in final_columns:
            if c not in ticker_entries.columns:
                ticker_entries[c] = ""
        ticker_entries = ticker_entries[final_columns]

        existing_ticker = unify_existing_csv(ticker_filename)
        if existing_ticker is not None:
            existing_signals = set(existing_ticker["Signal_ID"].values)
            new_ticker_entries = ticker_entries[~ticker_entries["Signal_ID"].isin(existing_signals)]
        else:
            new_ticker_entries = ticker_entries

        if not new_ticker_entries.empty:
            write_mode = 'a' if os.path.exists(ticker_filename) else 'w'
            header = not os.path.exists(ticker_filename)
            new_ticker_entries.to_csv(ticker_filename, index=False, mode=write_mode, header=header, columns=final_columns)
            logging.info(f"Saved unique price action entries for {ticker} to {ticker_filename}")

    # Save aggregated entries
    general_entries_filename = f"price_action_entries_15min_{today_str}.csv"
    existing_general = unify_existing_csv(general_entries_filename)
    if existing_general is not None:
        existing_signals = set(existing_general["Signal_ID"].values)
        new_entries_df = entries_df[~entries_df["Signal_ID"].isin(existing_signals)]
    else:
        new_entries_df = entries_df

    if not new_entries_df.empty:
        write_mode = 'a' if os.path.exists(general_entries_filename) else 'w'
        header = not os.path.exists(general_entries_filename)
        new_entries_df.to_csv(general_entries_filename, index=False, mode=write_mode, header=header, columns=final_columns)
        logging.info(f"Saved aggregated unique price action entries to {general_entries_filename}")

    return entries_df

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
        expected_columns = ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "ADX", "Signal_ID"]
        if not validate_dataframe(earliest_entries, expected_columns):
            logging.error(f"Combined Entries DataFrame validation failed for {last_trading_day}.")
            raise ValueError(f"Combined Entries DataFrame validation failed for {last_trading_day}.")

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
                # Remove 'format' to allow pandas to infer the datetime format with timezone
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
            if not validate_dataframe(combined_entries, expected_columns):
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

            # **Optional:** Read and print the papertrade data
            try:
                papertrade_data = pd.read_csv(output_file)
                # **New Logging Added Here:**
                logging.debug(f"Papertrade DataFrame Columns: {papertrade_data.columns.tolist()}")
                print("\nPapertrade Data for Today:")
                print(papertrade_data)
            except Exception as e:
                logging.error(f"Error reading papertrade file {output_file}: {e}")
                print(f"Error reading papertrade file '{output_file}': {e}")

            # **New Validation Step Added Here:**
            logging.info("Validating all combined CSV files for 'Entry Signal' integrity.")
            validate_entry_signals(selected_stocks, last_trading_day, india_tz)
            logging.info("Validation of combined CSV files completed.")
            print("Validation of combined CSV files completed.")
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

def prepare_time_ranges(last_trading_day, india_tz):
    """
    Prepare the historical and intraday time ranges based on the last trading day.

    Parameters:
        last_trading_day (date): The last trading day to consider.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.

    Returns:
        tuple: from_date_hist, to_date_hist, from_date_intra, to_date_intra
    """
    # Set from_date and to_date for historical data (30 days back)
    past_n_days = last_trading_day - timedelta(days=30)
    from_date_hist = india_tz.localize(
        datetime.combine(past_n_days, datetime_time.min)
    )
    to_date_hist = india_tz.localize(
        datetime.combine(last_trading_day - timedelta(days=1), datetime_time.max)
    )

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

def run():
    """
    Fetch data, identify entries, and create combined paper trade files for selected stocks.
    Only new data is added every 15 minutes without refreshing or changing older data.
    Calculations are performed solely on new data entries.
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
            combined_entries = combined_entries.drop_duplicates(subset=['Ticker', 'date'], keep='last')
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

        # **Optional:** Read and print the papertrade data
        try:
            papertrade_data = pd.read_csv(papertrade_filename)
            # **New Logging Added Here:**
            logging.debug(f"Papertrade DataFrame Columns: {papertrade_data.columns.tolist()}")
            print("\nPapertrade Data for Today:")
            print(papertrade_data)
        except Exception as e:
            logging.error(f"Error reading papertrade file {papertrade_filename}: {e}")
            print(f"Error reading papertrade file '{papertrade_filename}': {e}")

        # **New Validation Step Added Here:**
        logging.info("Validating all combined CSV files for 'Entry Signal' integrity.")
        validate_entry_signals(selected_stocks, last_trading_day, india_tz)
        logging.info("Validation of combined CSV files completed.")
        print("Validation of combined CSV files completed.")
    else:
        # Log and print if no entries were detected
        logging.info(f"No entries detected for {last_trading_day}.")
        print(f"No entries detected for {last_trading_day}.")

def validate_entry_signals(selected_stocks, last_trading_day, india_tz, validation_log_path='validation_results.txt'):
    """
    Validate the 'Entry Signal' column in all combined.csv files to ensure correctness.
    Recalculate if missing.

    Parameters:
        selected_stocks (list): List of selected stock tickers.
        last_trading_day (date): The last trading day to consider.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.
        validation_log_path (str): Path to the validation log file.
    """
    # Setup a separate logger for validation results
    validation_logger = logging.getLogger('ValidationLogger')
    validation_logger.setLevel(logging.INFO)
    
    # Add handler only if not already added
    if not validation_logger.handlers:
        fh = logging.FileHandler(validation_log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        validation_logger.addHandler(fh)
    
    for ticker in selected_stocks:
        combined_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        if not os.path.exists(combined_csv_path):
            logging.warning(f"Combined CSV for {ticker} does not exist at {combined_csv_path}. Skipping validation.")
            validation_logger.warning(f"Combined CSV for {ticker} does not exist. Skipping validation.")
            continue  # Skip if combined.csv does not exist

        try:
            # Load combined.csv
            data = pd.read_csv(combined_csv_path, parse_dates=['date'])
            data = normalize_time(data)  # Ensure 'date' is timezone-aware

            # Ensure 'Entry Signal' column exists
            if "Entry Signal" not in data.columns:
                logging.warning(f"'Entry Signal' column missing in {combined_csv_path}. Recalculating.")
                validation_logger.warning(f"'Entry Signal' column missing in {combined_csv_path}. Recalculating.")
                data["Entry Signal"] = "No"

            # **Modification: Indicators already calculated for new data, skip recalculating**
            # Recalculate indicators for new data only if needed
            # data = ensure_all_indicators(data, ticker)  # Not needed

            # Reapply trend conditions to detect expected entries
            # Filter today's data within market hours
            today_start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
            today_end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
            today_data = data[(data['date'] >= today_start_time) & (data['date'] <= today_end_time)]

            if today_data.empty:
                logging.warning(f"No intraday data available for {ticker} on {last_trading_day}. Skipping validation.")
                validation_logger.warning(f"No intraday data for {ticker} on {last_trading_day}. Skipping validation.")
                continue

            # Calculate daily percentage change using today's data
            daily_change = calculate_daily_change(today_data)

            # Detect expected entries based on trend conditions
            expected_entries = apply_trend_conditions(today_data, daily_change, ticker, last_trading_day)

            # Collect expected entry times with tolerance
            tolerance = timedelta(minutes=1)  # 1-minute tolerance
            expected_entry_times = []
            for entry in expected_entries:
                try:
                    entry_time = pd.to_datetime(entry["date"])
                    if entry_time.tzinfo is None:
                        entry_time = india_tz.localize(entry_time)
                    else:
                        entry_time = entry_time.astimezone(india_tz)
                    expected_entry_times.append(entry_time)
                except Exception:
                    logging.error(f"Invalid date format for entry in {ticker}: {entry['date']}")
                    validation_logger.error(f"Invalid date format for entry in {ticker}: {entry['date']}")
                    logging.error(traceback.format_exc())
                    validation_logger.error(traceback.format_exc())

            # Create a copy of data to compare without modifying the original
            comparison_data = data.copy()

            # Reset 'Entry Signal' to 'No' before re-marking
            comparison_data["Entry Signal"] = "No"

            # Mark expected entries as 'Yes' with tolerance
            for entry_time in expected_entry_times:
                mask = (comparison_data['date'] >= entry_time - tolerance) & (comparison_data['date'] <= entry_time + tolerance)
                comparison_data.loc[mask, "Entry Signal"] = "Yes"

            # Identify discrepancies where 'Entry Signal' was 'Yes' but not expected
            actual_yes = set(data[data["Entry Signal"] == "Yes"]['date'])
            expected_yes = set(comparison_data[comparison_data["Entry Signal"] == "Yes"]['date'])

            # Entries that should be 'Yes' but are not
            missing_yes = expected_yes - actual_yes

            # Entries that are 'Yes' but should not be
            unexpected_yes = actual_yes - expected_yes

            # Log discrepancies to the validation log file
            if missing_yes:
                validation_logger.warning(f"For ticker {ticker}, 'Entry Signal' missing for the following times:")
                for time_point in sorted(missing_yes):
                    validation_logger.warning(f" - {time_point}")
            
            if unexpected_yes:
                validation_logger.warning(f"For ticker {ticker}, 'Entry Signal' incorrectly set to 'Yes' for the following times:")
                for time_point in sorted(unexpected_yes):
                    validation_logger.warning(f" - {time_point}")

            if not missing_yes and not unexpected_yes:
                validation_logger.info(f"'Entry Signal' correctly calculated for {ticker}.")
            else:
                # **Modification: Correct the 'Entry Signal' column based on expected entries**
                validation_logger.info(f"Correcting 'Entry Signal' for {ticker} based on expected entries.")
                data["Entry Signal"] = comparison_data["Entry Signal"]
                data.to_csv(combined_csv_path, index=False)
                validation_logger.info(f"'Entry Signal' column corrected and saved for {ticker}.")
                send_notification("Trading Script Alert", f"'Entry Signal' was corrected for {ticker}.")

        except Exception as e:
            logging.error(f"Error validating 'Entry Signal' for {ticker}: {e}")
            logging.error(traceback.format_exc())
            validation_logger.error(f"Error validating 'Entry Signal' for {ticker}: {e}")
            validation_logger.error(traceback.format_exc())
            send_notification("Trading Script Error", f"Error validating 'Entry Signal' for {ticker}: {e}")

def backup_and_remove_corrupted_csv(ticker):
    """
    Backup and remove the corrupted {ticker}_combined.csv file.

    Parameters:
        ticker (str): The trading symbol of the stock.
    """
    combined_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    backup_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_backup.csv")
    
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
        
def validate_all_combined_csv(selected_stocks):
    """
    Validate all combined CSV files to ensure the 'Entry Signal' column exists.
    Recalculate if missing.
    """
    for ticker in selected_stocks:
        combined_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        if os.path.exists(combined_csv_path):
            try:
                data = pd.read_csv(combined_csv_path, parse_dates=['date'])
                data = normalize_time(data)  # Ensure 'date' is timezone-aware

                # Ensure 'Entry Signal' column exists
                if "Entry Signal" not in data.columns:
                    logging.warning(f"'Entry Signal' column missing in {combined_csv_path}. Recalculating.")
                    # Recalculate 'Entry Signal' by setting to 'No'
                    data["Entry Signal"] = "No"
                    # Save back to CSV
                    data.to_csv(combined_csv_path, index=False)
                    logging.info(f"Recalculated 'Entry Signal' for {ticker}.")
                    send_notification("Trading Script Alert", f"'Entry Signal' was missing for {ticker}. It has been initialized to 'No'.")
                else:
                    # Optionally, validate the values in 'Entry Signal'
                    valid_signals = {"Yes", "No"}
                    invalid_signals = set(data["Entry Signal"].unique()) - valid_signals
                    if invalid_signals:
                        logging.warning(f"Invalid 'Entry Signal' values {invalid_signals} in {combined_csv_path}. Correcting to 'No'.")
                        data["Entry Signal"] = data["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
                        data.to_csv(combined_csv_path, index=False)
                        logging.info(f"Corrected invalid 'Entry Signal' values for {ticker}.")
                        send_notification("Trading Script Alert", f"Invalid 'Entry Signal' values were corrected for {ticker}.")
            except Exception as e:
                logging.error(f"Error validating {combined_csv_path}: {e}")
                send_notification("Trading Script Error", f"Error validating {combined_csv_path}: {e}")



import os
import glob
from datetime import datetime
import pandas as pd
import argparse
import pytz

def get_latest_expected_interval(current_time=None, tz='Asia/Kolkata'):
    """
    Given a current_time (datetime), find the most recent 15-minute interval
    in the specified timezone.

    For example:
    - If current_time is 9:40, the interval is 9:30.
    - If current_time is 10:10, the interval is 10:00.
    - If current_time is 12:20, the interval is 12:15.
    """
    timezone = pytz.timezone(tz)
    if current_time is None:
        current_time = datetime.now(timezone)
    else:
        if current_time.tzinfo is None:
            current_time = timezone.localize(current_time)
        else:
            current_time = current_time.astimezone(timezone)

    minute_block = (current_time.minute // 15) * 15
    expected_interval = current_time.replace(minute=minute_block, second=0, microsecond=0)
    return expected_interval

def check_csv_intervals(data_dir='main_indicators', current_time=None, tz='Asia/Kolkata'):
    """
    Reads all *_main_indicators.csv files in the given directory, extracts the ticker from the filename,
    and checks if the last expected 15-min interval entry exists in the CSV.

    Assumes CSV dates either:
    - Contain timezone info compatible with 'Asia/Kolkata', or
    - Are naive and represent local time in 'Asia/Kolkata'.

    In the latter case, we localize them to 'Asia/Kolkata'.

    Returns a dictionary: { 'TICKER': True/False, ... } indicating whether the latest expected
    entry was found for each ticker.
    """
    expected_interval = get_latest_expected_interval(current_time, tz=tz)

    pattern = os.path.join(data_dir, '*_main_indicators.csv')
    files = glob.glob(pattern)
    
    results = {}
    timezone = pytz.timezone(tz)

    for f in files:
        ticker = os.path.basename(f).replace('_main_indicators.csv', '')
        
        # Read the CSV
        try:
            df = pd.read_csv(f)
        except Exception:
            results[ticker] = False
            continue
        
        if 'date' not in df.columns:
            results[ticker] = False
            continue
        
        try:
            df['date'] = pd.to_datetime(df['date'], utc=False)
            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize(timezone)
            else:
                df['date'] = df['date'].dt.tz_convert(timezone)
        except Exception:
            results[ticker] = False
            continue
        
        df['floored_date'] = df['date'].dt.floor('15min')
        interval_exists = (df['floored_date'] == expected_interval).any()
        
        results[ticker] = interval_exists
    
    return results

def main1():
    parser = argparse.ArgumentParser(description="Check CSV intervals for tickers (timezone aware).")
    parser.add_argument("--data_dir", type=str, default="main_indicators",
                        help="Directory containing *_main_indicators.csv files.")
    parser.add_argument("--timezone", type=str, default="Asia/Kolkata",
                        help="Timezone for matching intervals.")
    args = parser.parse_args()
    
    current_time = datetime.now()  # naive current time
    results = check_csv_intervals(data_dir=args.data_dir, current_time=current_time, tz=args.timezone)
    
    # Print only the missing ones
    for ticker, exists in results.items():
        if not exists:
            print(ticker)



def main():
    """
    Fetch data, identify entries, and create combined paper trade files for selected stocks.
    Executes functions twice every 15 minutes with a 15-second pause between executions.
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
    end_time_naive = datetime.combine(now.date(), datetime_time(19, 0, 5))  # Changed end time to 3:00:05 PM
    
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
    # Define the Job Functions
    # ================================
    
    def job_run_twice():
        """
        Executes the 'run' function twice with a 15-second pause between executions.
        """
        try:
            run()
            logging.info("Executed trading strategy successfully (First Run).")
            print("Executed trading strategy successfully (First Run).")
            
            time.sleep(15)  # Pause for 15 seconds
            
            run()
            logging.info("Executed trading strategy successfully (Second Run).")
            print("Executed trading strategy successfully (Second Run).")
        except Exception as e:
            logging.error(f"Error executing job_run_twice: {e}")
            print(f"Error executing job_run_twice: {e}")
            # Optionally, uncomment the line below to send notifications on failure
            # send_notification("Trading Script Error", f"Error executing job_run_twice: {e}")
    
    def job_validate_twice():
        """
        Executes the 'validate_entry_signals' function twice with a 15-second pause between executions.
        """
        try:
            last_trading_day = get_last_trading_day()
            validate_entry_signals(selected_stocks, last_trading_day, india_tz)
            logging.info("Entry signal validation executed successfully (First Validation).")
            print("Entry signal validation executed successfully (First Validation).")
            
            time.sleep(15)  # Pause for 15 seconds
            
            validate_entry_signals(selected_stocks, last_trading_day, india_tz)
            logging.info("Entry signal validation executed successfully (Second Validation).")
            print("Entry signal validation executed successfully (Second Validation).")
        except Exception as e:
            logging.error(f"Error executing job_validate_twice: {e}")
            print(f"Error executing job_validate_twice: {e}")
            # Optionally, uncomment the line below to send notifications on failure
            # send_notification("Trading Script Error", f"Error executing job_validate_twice: {e}")
    
    # ================================
    # Schedule the Jobs
    # ================================
    
    for scheduled_time in scheduled_times:
        # Convert scheduled_time string back to datetime object for comparison
        scheduled_datetime_naive = datetime.strptime(scheduled_time, "%H:%M:%S")
        scheduled_datetime = india_tz.localize(datetime.combine(now.date(), scheduled_datetime_naive.time()))
        
        if scheduled_datetime > now:
            # Schedule the 'run' function to execute twice every 15 minutes
            schedule.every().day.at(scheduled_time).do(job_run_twice)
            logging.info(f"Scheduled run job at {scheduled_time} IST for ticker(s): {selected_stocks}")
        else:
            logging.debug(f"Skipped scheduling run job at {scheduled_time} IST as it's already past.")
    
    # Schedule the validation job to run immediately after the last run job
    schedule_time = (end_time + timedelta(seconds=5)).strftime("%H:%M:%S")
    schedule.every().day.at(schedule_time).do(job_validate_twice)
    logging.info(f"Scheduled validation job at {schedule_time} IST for ticker(s): {selected_stocks}")
    
    logging.info("Scheduled `run` to execute twice every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    print("Scheduled `run` to execute twice every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    
    # ================================
    # Immediate Execution Check
    # ================================
    
    # Optionally, execute the job_run_twice immediately if within trading hours
    if start_time <= now <= end_time:
        logging.info("Executing initial run of trading strategy twice.")
        print("Executing initial run of trading strategy twice.")
        job_run_twice()
    
    # ================================
    # Keep the Script Running to Allow Scheduled Jobs
    # ================================
    
    # Perform validation before running the main process
    validate_and_correct_main_indicators(selected_stocks, india_tz)
    
    while True:
        schedule.run_pending()  # Run any pending scheduled jobs
        time.sleep(1)  # Sleep to prevent high CPU usage


if __name__ == "__main__":
    # Perform validation before running the main process
    validate_and_correct_main_indicators(selected_stocks, india_tz)
    main() # Execute the main function when the script is run
    main1()       
    
    
    



