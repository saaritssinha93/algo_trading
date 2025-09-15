# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:03:21 2024

@author: Saarit

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
import csv

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
'''
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

'''
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
'''
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
    combined_indicators_file = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    
    logging.info(f"Checking combined indicators cache for {ticker}.")
    combined_cache_exists = os.path.exists(combined_indicators_file)
    
    # Step 1: Fetch Intraday Data
    def fetch_intraday(ticker):
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
                intraday_data = pd.DataFrame(new_intraday_df)
                intraday_data['date'] = pd.to_datetime(intraday_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
                return intraday_data
        except Exception as e:
            logging.error(f"Error fetching intraday data for {ticker}: {e}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    intraday_data = cache_update(
        ticker=ticker,
        data_type="intraday",
        fetch_function=fetch_intraday,
        cache_dir=cache_dir,
        freshness_threshold=timedelta(minutes=15),
        bypass_cache=True
    )
        
    # Step 2: Fetch Historical Data
    def fetch_historical(ticker):
        from_date_hist = from_date_intra - timedelta(days=5)  # buffer for historical
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
                historical_data = pd.DataFrame(historical_df)
                historical_data['date'] = pd.to_datetime(historical_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
                return historical_data
        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {e}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    historical_data = cache_update(
        ticker=ticker,
        data_type="historical",
        fetch_function=fetch_historical,
        cache_dir=cache_dir,
        freshness_threshold=timedelta(minutes=600),  # 10 hours
        bypass_cache=False
    )
    
    # Step 3: Combine Historical and Intraday Data
    if historical_data.empty and intraday_data.empty:
        logging.warning(f"No data available for {ticker}. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Combining historical and intraday data for {ticker}.")
    combined_data = pd.concat([historical_data, intraday_data]).drop_duplicates(subset='date').sort_values(by='date')
    
    # Step 4: Recalculate Indicators if combined_data not empty
    if not combined_data.empty:
        combined_data = sort_dataframe_by_date(combined_data)
        combined_data = recalc_indicators(combined_data)
        logging.info(f"Indicators recalculated for {ticker}.")
    else:
        logging.warning(f"Combined data is empty after merging historical and intraday for {ticker}.")
        return combined_data

    # Ensure Signal_ID column exists
    if 'Signal_ID' not in combined_data.columns:
        combined_data['Signal_ID'] = combined_data.apply(
            lambda row: f"{ticker}-{row['date'].isoformat()}", axis=1
        )
    else:
        # For any rows without a Signal_ID, assign one
        combined_data['Signal_ID'] = combined_data.apply(
            lambda row: row['Signal_ID'] if pd.notnull(row['Signal_ID']) else f"{ticker}-{row['date'].isoformat()}",
            axis=1
        )

    # Ensure 'Entry Signal' column exists before marking entries
    if 'Entry Signal' not in combined_data.columns:
        combined_data['Entry Signal'] = 'No'

    # *** NEW LOGIC STARTS HERE ***
    # Ensure 'Entry Signal' column exists and default to 'No'
    # Set all rows to 'No' initially
    combined_data['Entry Signal'] = 'No'

    # Identify today's timeframe
    last_trading_day = combined_data['date'].dt.date.max()
    today_start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    today_end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
    today_data = combined_data[(combined_data['date'] >= today_start_time) & (combined_data['date'] <= today_end_time)].copy()

    if not today_data.empty:
        # Calculate daily change
        daily_change = calculate_daily_change(today_data)
        # Apply trend conditions
        ticker_entries = apply_trend_conditions(ticker, last_trading_day, india_tz)
        # If entries found, mark them as 'Yes'
        if ticker_entries:
            entry_times = []
            for entry in ticker_entries:
                entry_time = pd.to_datetime(entry['date'])
                if entry_time.tzinfo is None:
                    entry_time = india_tz.localize(entry_time)
                else:
                    entry_time = entry_time.astimezone(india_tz)
                entry_times.append(entry_time)
            
            # Mark these specific times as 'Yes'
            mark_entry_signals(combined_data, entry_times, india_tz)
    # *** NEW LOGIC ENDS HERE ***

    # Step 6: Append Only Missing Rows to Main Indicators CSV
    main_indicators_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    if os.path.exists(main_indicators_path):
        existing_indicators = pd.read_csv(main_indicators_path, parse_dates=['date'])
        # Ensure 'date' is timezone-aware
        if existing_indicators['date'].dt.tz is None:
            existing_indicators['date'] = existing_indicators['date'].dt.tz_localize('Asia/Kolkata')
        else:
            existing_indicators['date'] = existing_indicators['date'].dt.tz_convert('Asia/Kolkata')
    else:
        existing_indicators = pd.DataFrame()

    # Identify new rows not present in the main indicators file
    if not existing_indicators.empty:
        new_rows = combined_data[~combined_data['Signal_ID'].isin(existing_indicators['Signal_ID'])]
    else:
        new_rows = combined_data.copy()

    if not new_rows.empty:
        # Ensure 'logtime' column exists
        if 'logtime' not in new_rows.columns:
            new_rows['logtime'] = ""

        # Assign current logtime to new rows
        current_logtime = datetime.now(india_tz).isoformat()
        new_rows['logtime'] = current_logtime

        # Assign current logtime to new rows
        current_logtime = datetime.now(india_tz).isoformat()
        new_rows['logtime'] = current_logtime

        # Combine and remove duplicates
        final_data = pd.concat([existing_indicators, new_rows], ignore_index=True)
        final_data = final_data.drop_duplicates(subset=['Signal_ID'], keep='first')
        final_data = sort_dataframe_by_date(final_data)

        # Save final_data back to the main_indicators CSV
        final_data.to_csv(main_indicators_path, index=False)
        logging.info(f"Appended {len(new_rows)} new rows of indicator data for {ticker}, with logtime added.")
    # Removed the extra 'else:' here
    # else:
    #     logging.info(f"No new rows to append for {ticker}.")  # This was the problematic extra 'else:'

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
                logging.warning(f"No token found for {ticker}. Skipping.")
                return None, []
            
            combined_data = fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra)

            if combined_data is None or combined_data.empty:
                logging.warning(f"No combined data available for {ticker}.")
                return None, []
            
            # Ensure 'Entry Signal' exists
            if 'Entry Signal' not in combined_data.columns:
                combined_data['Entry Signal'] = 'No'
            
            # Get today's data and daily_change
            today_start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
            today_end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
            today_data = combined_data[(combined_data['date'] >= today_start_time) & (combined_data['date'] <= today_end_time)].copy()

            if today_data.empty:
                logging.warning(f"No intraday data available for {ticker}. Skipping.")
                return combined_data, []
            
            daily_change = calculate_daily_change(today_data)
            
            # Detect entries
            ticker_entries = apply_trend_conditions(ticker, last_trading_day, india_tz)
            
            # If there are no entries, just return combined_data and empty list
            if not ticker_entries:
                return combined_data, []
            
            # Mark entries in combined_data
            entry_times = []
            for entry in ticker_entries:
                # Convert entry['date'] to a proper datetime
                entry_time = pd.to_datetime(entry['date'])
                if entry_time.tzinfo is None:
                    entry_time = india_tz.localize(entry_time)
                else:
                    entry_time = entry_time.astimezone(india_tz)
                entry_times.append(entry_time)

            mark_entry_signals(combined_data, entry_times, india_tz)
            
            # Now we want to add the Signal_ID from combined_data to our entries
            # Convert ticker_entries to a DataFrame to merge on 'date'
            ticker_entries_df = pd.DataFrame(ticker_entries)
            
            # Ensure ticker_entries_df['date'] is datetime & localized
            ticker_entries_df['date'] = pd.to_datetime(ticker_entries_df['date'], errors='coerce')
            ticker_entries_df['date'] = ticker_entries_df['date'].dt.tz_localize(india_tz) if ticker_entries_df['date'].dt.tz is None else ticker_entries_df['date'].dt.tz_convert(india_tz)
            
            # Merge with combined_data to get Signal_ID
            merged_df = pd.merge(
                ticker_entries_df,
                combined_data[['date', 'Signal_ID']],
                on='date',
                how='left'
            )

            # Convert back to list of dicts
            ticker_entries_with_id = merged_df.to_dict('records')

            # Save combined_data back to CSV
            main_indicators_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            combined_data.to_csv(main_indicators_path, index=False)
            logging.info(f"Saved combined data for {ticker} to main indicators CSV.")
            
            return combined_data, ticker_entries_with_id

    except KeyError as ke:
        logging.error(f"Error processing ticker {ticker}: {ke}")
        return None, []
    except Exception as e:
        logging.error(f"Unexpected error processing ticker {ticker}: {e}")
        logging.error(traceback.format_exc())
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
#                send_notification("Trading Script Error", f"Error validating {main_indicators_path}: {e}")
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
#        send_notification("Trading Script Error", f"Missing expected column: {ke}")
        raise
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        logging.error(traceback.format_exc())
#        send_notification("Trading Script Error", f"Error calculating indicators: {e}")
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
#        send_notification("Trading Script Error", f"Error marking entry signals: {e}")
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

def apply_trend_conditions(ticker: str, last_trading_day: pd.Timestamp, india_tz) -> list:
    """
    Detects Bullish or Bearish trend entries based solely on the data stored in
    the existing *_main_indicators.csv file for the given ticker and last_trading_day.

    Parameters:
        ticker (str): The stock ticker symbol.
        last_trading_day (pd.Timestamp): The last trading day as a date object.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.

    Returns:
        list: A list of entries if conditions are met, otherwise an empty list.
    """
    main_indicators_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    if not os.path.exists(main_indicators_path):
        logging.warning(f"No main indicators file found for {ticker}.")
        return []

    # Load the data from the CSV
    df = pd.read_csv(main_indicators_path, parse_dates=['date'])

    # Ensure date is timezone-aware
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize(india_tz)
    else:
        df['date'] = df['date'].dt.tz_convert(india_tz)

    # Filter for the given trading day between 9:15 and 15:30
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
    today_data = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()

    if today_data.empty:
        logging.warning(f"No intraday data for {ticker} on {last_trading_day}.")
        return []

    # Calculate daily percentage change using today's data from CSV
    try:
        first_open = today_data["open"].iloc[0]
        latest_close = today_data["close"].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except Exception as e:
        logging.error(f"Error calculating daily change for {ticker}: {e}")
        return []

    # Check the latest row for trend conditions
    try:
        latest_data = today_data.iloc[-1]
        strong_trend = latest_data["ADX"] > 20

        if strong_trend:
            # Bullish conditions
            if (
                daily_change > 0.5
                and latest_data["RSI"] > 20
                and latest_data["close"] > latest_data["VWAP"]
            ):
                # Bullish trend detected
                return screen_trend(ticker, last_trading_day, india_tz, "Bullish", daily_change, today_data)
            # Bearish conditions
            elif (
                daily_change < -0.75
                and latest_data["RSI"] < 80
                and latest_data["close"] < latest_data["VWAP"]
            ):
                # Bearish trend detected
                return screen_trend(ticker, last_trading_day, india_tz, "Bearish", daily_change, today_data)

        # If no conditions met
        return []

    except Exception as e:
        logging.error(f"Error applying trend conditions for {ticker}: {e}")
        logging.error(traceback.format_exc())
        return []


def screen_trend(ticker: str, last_trading_day: pd.Timestamp, india_tz, trend_type: str, change: float, today_data: pd.DataFrame) -> list:
    """
    Identifies specific trend entry points (Pullback/Breakout or Pullback/Breakdown) based on the trend_type,
    using only the data loaded from the *_main_indicators.csv file passed as `today_data`.

    Parameters:
        ticker (str): The stock ticker symbol.
        last_trading_day (pd.Timestamp): The last trading day as a date.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.
        trend_type (str): "Bullish" or "Bearish".
        change (float): Daily percentage change.
        today_data (pd.DataFrame): The filtered data for the current trading day loaded from CSV.

    Returns:
        list: A list of entries (dictionaries) if conditions are met, otherwise an empty list.
    """
    trend_entries = []
    volume_multiplier = 1.2  # As per the original logic

    # Conditions dictionary similar to the original code but no recalculations in-memory
    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (today_data["low"] >= today_data["VWAP"] * 0.98)
                & (today_data["close"] > today_data["open"])
                & (today_data["MACD"] > today_data["Signal_Line"])
                & (today_data["MACD"] > -0.5)
                & (today_data["close"] > today_data["20_SMA"])
                & (today_data["Stochastic"] < 50)
                & (today_data["Stochastic"].diff() > 0)
                & (today_data["ADX"] > 20)
            ),
            "Breakout": (
                (today_data["close"] > today_data["high"].rolling(window=10).max() * 1.005)
                & (today_data["volume"] > (volume_multiplier * 1.5) * today_data["volume"].rolling(window=10).mean())
                & (today_data["MACD"] > today_data["Signal_Line"])
                & (today_data["MACD"] > -0.5)
                & (today_data["close"] > today_data["20_SMA"])
                & (today_data["Stochastic"] > 50)
                & (today_data["Stochastic"].diff() > 0)
                & (today_data["ADX"] > 20)
            ),
        }
    else:  # Bearish
        conditions = {
            "Pullback": (
                (today_data["high"] <= today_data["VWAP"] * 1.02)
                & (today_data["close"] < today_data["open"])
                & (today_data["MACD"] < today_data["Signal_Line"])
                & (today_data["MACD"] < 0.5)
                & (today_data["close"] < today_data["20_SMA"])
                & (today_data["Stochastic"] > 55)
                & (today_data["Stochastic"].diff() < 0)
                & (today_data["ADX"] > 20)
            ),
            "Breakdown": (
                (today_data["close"] < today_data["low"].rolling(window=10).min() * 0.98)
                & (today_data["volume"] > (volume_multiplier * 1.5) * today_data["volume"].rolling(window=10).mean())
                & (today_data["MACD"] < today_data["Signal_Line"])
                & (today_data["MACD"] < 0.5)
                & (today_data["close"] < today_data["20_SMA"])
                & (today_data["Stochastic"] < 45)
                & (today_data["Stochastic"].diff() < 0)
                & (today_data["ADX"] > 20)
            ),
        }

    for entry_type, condition in conditions.items():
        selected = today_data.loc[condition & (today_data['date'].dt.date == last_trading_day)]

        if not selected.empty:
            entries = selected.apply(lambda row: {
                "Ticker": ticker,
                "date": row["date"].strftime("%Y-%m-%d %H:%M:%S%z") if isinstance(row["date"], pd.Timestamp) else None,
                "Entry Type": entry_type,
                "Trend Type": trend_type,
                "Price": round(row["close"], 2),
                "Daily Change %": round(change, 2),
                "VWAP": round(row["VWAP"], 2) if "Pullback" in entry_type else None,
                "ADX": round(row["ADX"], 2),
            }, axis=1).tolist()

            # Filter out entries with invalid date formatting
            entries = [entry for entry in entries if entry["date"] is not None]
            trend_entries.extend(entries)

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
#                    send_notification("Trading Script Error", f"Error processing ticker {ticker}: {e}")
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")
#        send_notification("Trading Script Error", f"Error during parallel processing: {e}")

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



import pandas as pd
import os
from datetime import datetime
import pytz
import logging
import csv
import traceback

# Define expected columns including 'logtime'
expected_columns = [
    "Ticker", "date", "Entry Type", "Trend Type",
    "Price", "Daily Change %", "VWAP", "ADX",
    "Signal_ID", "logtime"
]

def read_and_normalize_csv(file_path, expected_columns, timezone='Asia/Kolkata'):
    """
    Reads a CSV file, ensures it has the expected columns, and normalizes the 'date' column.

    Parameters:
        file_path (str): Path to the CSV file.
        expected_columns (list): List of expected column names.
        timezone (str): Timezone for 'logtime'.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    try:
        logging.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(
            file_path,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines='warn',  # Log and skip malformed lines
            engine='python'        # Use Python engine for better handling
        )
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing columns {missing_cols} in {file_path}. Adding them with default values.")
            for col in missing_cols:
                if col == 'logtime':
                    india_tz = pytz.timezone(timezone)
                    current_logtime = datetime.now(india_tz).isoformat()
                    df[col] = current_logtime
                else:
                    df[col] = ""  # Assign default empty string or appropriate default
        
        # Handle extra columns by keeping only the expected ones
        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            df = df.drop(columns=extra_columns)
            logging.warning(f"Dropped extra columns {extra_columns} from {file_path}")
        
        # Normalize 'date' column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df['date'] = df['date'].dt.tz_convert(timezone)
        else:
            logging.error(f"'date' column missing in {file_path}. Cannot normalize time.")
            df['date'] = pd.NaT  # Assign Not-a-Time
        
        # Drop rows with NaT in 'date' after normalization
        before_drop = len(df)
        df = df.dropna(subset=['date'])
        after_drop = len(df)
        if before_drop != after_drop:
            logging.warning(f"Dropped {before_drop - after_drop} rows with invalid 'date' in {file_path}")
        
        # Assign unique Signal_ID if missing
        if 'Signal_ID' not in df.columns:
            df['Signal_ID'] = df.apply(
                lambda row: f"{row['Ticker']}-{row['date'].isoformat()}",
                axis=1
            )
            logging.info(f"Assigned 'Signal_ID' for all rows in {file_path}")
        else:
            # For rows with missing Signal_ID, assign one
            missing_signal_ids = df['Signal_ID'].isna()
            if missing_signal_ids.any():
                df.loc[missing_signal_ids, 'Signal_ID'] = df.loc[missing_signal_ids].apply(
                    lambda row: f"{row['Ticker']}-{row['date'].isoformat()}",
                    axis=1
                )
                logging.info(f"Assigned missing 'Signal_ID' in {file_path}")
        
        # Reorder columns to match expected_columns
        df = df[expected_columns]
        
        return df
    
    except pd.errors.ParserError as e:
        logging.error(f"Pandas ParserError while reading {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"Unexpected error while reading {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise

def write_csv(df, file_path, expected_columns):
    """
    Writes a DataFrame to a CSV file with consistent formatting.

    Parameters:
        df (pd.DataFrame): The DataFrame to write.
        file_path (str): Destination CSV file path.
        expected_columns (list): List of expected column names.
    """
    try:
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Reorder columns
        df = df[expected_columns]
        
        # Write to CSV with consistent quoting
        df.to_csv(
            file_path,
            index=False,
            quoting=csv.QUOTE_ALL,    # Enclose all fields in quotes
            escapechar='\\',
            doublequote=True
        )
        logging.info(f"Successfully wrote to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write to {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise



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
    
    expected_columns = [
        "Ticker", "date", "Entry Type", "Trend Type",
        "Price", "Daily Change %", "VWAP", "ADX",
        "Signal_ID", "logtime"
    ]
    
    # Define final columns for consistent CSV structure
    final_columns = expected_columns
    today_str = last_trading_day.strftime("%Y-%m-%d")

    # Aggregate all entries into a single DataFrame with existing Signal_IDs
    all_entries = []
    for ticker, entries in entries_dict.items():
        for entry in entries:
            all_entries.append({
                "Ticker": entry.get("Ticker"),
                "date": entry.get("date"),
                "Entry Type": entry.get("Entry Type"),
                "Trend Type": entry.get("Trend Type"),
                "Price": entry.get("Price"),
                "Daily Change %": entry.get("Daily Change %"),
                "VWAP": entry.get("VWAP"),
                "ADX": entry.get("ADX"),
                "Signal_ID": entry.get("Signal_ID"),  # Use existing Signal_ID from main indicators
                "logtime": entry.get("logtime")
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
    expected_columns = final_columns
    if not validate_dataframe(entries_df, expected_columns):
        logging.error("Aggregated Entries DataFrame failed validation.")
        raise ValueError("Aggregated Entries DataFrame failed validation.")

    # Ensure all final columns are present, then reorder
    for col in final_columns:
        if col not in entries_df.columns:
            entries_df.loc[:, col] = ""  # Use .loc to avoid SettingWithCopyWarning
    entries_df = entries_df[final_columns]

    # Function to unify existing CSVs with final_columns and Signal_ID
    def unify_existing_csv(filename):
        if os.path.exists(filename):
            existing = read_and_normalize_csv(filename, final_columns, timezone='Asia/Kolkata')
            return existing
        return None

    # Save to individual ticker CSV files
    for ticker in entries_df['Ticker'].unique():
        ticker_filename = f"papertrade_{ticker}_{today_str}.csv"
        ticker_entries = entries_df[entries_df['Ticker'] == ticker].copy()

        # Ensure all final columns in ticker_entries
        for c in final_columns:
            if c not in ticker_entries.columns:
                ticker_entries.loc[:, c] = ""  # Use .loc to avoid warnings

        # Assign current logtime to all rows (not just new) using .loc
        current_logtime = datetime.now(india_tz).isoformat()
        ticker_entries.loc[:, 'logtime'] = current_logtime

        # Reorder columns (adding 'logtime' at the end)
        if 'logtime' not in final_columns:
            final_columns_with_logtime = final_columns + ['logtime']
        else:
            final_columns_with_logtime = final_columns

        if 'logtime' not in final_columns_with_logtime:
            final_columns_with_logtime.append('logtime')

        ticker_entries = ticker_entries[final_columns_with_logtime]

        existing_ticker = unify_existing_csv(ticker_filename)
        if existing_ticker is not None:
            existing_signals = set(existing_ticker["Signal_ID"].values)
            new_ticker_entries = ticker_entries[~ticker_entries["Signal_ID"].isin(existing_signals)]
        else:
            new_ticker_entries = ticker_entries

        if not new_ticker_entries.empty:
            write_mode = 'a' if os.path.exists(ticker_filename) else 'w'
            header = not os.path.exists(ticker_filename)
            write_csv(
                new_ticker_entries,
                ticker_filename,
                final_columns_with_logtime
            )
            logging.info(f"Saved unique price action entries for {ticker} to {ticker_filename}")
            print(f"Saved unique price action entries for {ticker} to {ticker_filename}")

    # Save aggregated entries
    general_entries_filename = f"price_action_entries_15min_{today_str}.csv"
    existing_general = unify_existing_csv(general_entries_filename)
    if existing_general is not None:
        existing_signals = set(existing_general["Signal_ID"].values)
        new_entries_df = entries_df[~entries_df["Signal_ID"].isin(existing_signals)]
    else:
        new_entries_df = entries_df

    if not new_entries_df.empty:
        # Also add logtime to aggregated entries using .loc
        current_logtime = datetime.now(india_tz).isoformat()
        new_entries_df.loc[:, 'logtime'] = current_logtime

        # Ensure final columns with 'logtime'
        if 'logtime' not in final_columns:
            final_columns_with_logtime = final_columns + ['logtime']
        else:
            final_columns_with_logtime = final_columns
            if 'logtime' not in final_columns_with_logtime:
                final_columns_with_logtime.append('logtime')

        # Reorder columns for new_entries_df
        for c in final_columns_with_logtime:
            if c not in new_entries_df.columns:
                new_entries_df.loc[:, c] = ""
        new_entries_df = new_entries_df[final_columns_with_logtime]

        write_csv(
            new_entries_df,
            general_entries_filename,
            final_columns_with_logtime
        )
        logging.info(f"Saved aggregated unique price action entries to {general_entries_filename}")
        print(f"Saved aggregated unique price action entries to {general_entries_filename}")

    return entries_df







import os
import glob
import pytz
import pandas as pd
from datetime import datetime, timedelta, time as datetime_time

def validate_entry_signals(selected_stocks, last_trading_day, india_tz, validation_log_path='validation_results.txt'):
    """
    Validate the 'Entry Signal' column in all main_indicators CSV files to ensure correctness.
    Now uses strict interval matching logic similar to analyze_latest_interval().
    If discrepancies are found, it corrects them.
    """
    validation_logger = logging.getLogger('ValidationLogger')
    validation_logger.setLevel(logging.INFO)
    
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
            continue

        try:
            data = pd.read_csv(combined_csv_path, parse_dates=['date'])
            data = normalize_time(data)  # Ensure 'date' is timezone-aware

            # Ensure 'Entry Signal' column exists
            if "Entry Signal" not in data.columns:
                logging.warning(f"'Entry Signal' column missing in {combined_csv_path}. Initializing to 'No'.")
                validation_logger.warning(f"'Entry Signal' column missing in {combined_csv_path}. Initializing to 'No'.")
                data["Entry Signal"] = "No"

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

            # Detect expected entries based on trend conditions (exact matching)
            expected_entries = apply_trend_conditions(ticker, last_trading_day, india_tz)

            # Rebuild the expected signal assignment based on exact interval matches
            comparison_data = data.copy()
            comparison_data["Entry Signal"] = "No"

            # Convert expected entry times to timezone-aware datetimes
            entry_times = []
            for entry in expected_entries:
                entry_time = pd.to_datetime(entry["date"], errors='coerce')
                if entry_time is not None and not pd.isnull(entry_time):
                    if entry_time.tzinfo is None:
                        entry_time = india_tz.localize(entry_time)
                    else:
                        entry_time = entry_time.astimezone(india_tz)
                    # Mark these exact times as 'Yes'
                    comparison_data.loc[comparison_data['date'] == entry_time, "Entry Signal"] = "Yes"

            # Identify discrepancies
            actual_yes = set(data[data["Entry Signal"] == "Yes"]['date'])
            expected_yes = set(comparison_data[comparison_data["Entry Signal"] == "Yes"]['date'])

            missing_yes = expected_yes - actual_yes
            unexpected_yes = actual_yes - expected_yes

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
                # Correct the 'Entry Signal' column based on expected entries
                validation_logger.info(f"Correcting 'Entry Signal' for {ticker} based on expected entries.")
                data["Entry Signal"] = comparison_data["Entry Signal"]
                data.to_csv(combined_csv_path, index=False)
                validation_logger.info(f"'Entry Signal' column corrected and saved for {ticker}.")

        except Exception as e:
            logging.error(f"Error validating 'Entry Signal' for {ticker}: {e}")
            logging.error(traceback.format_exc())
            validation_logger.error(f"Error validating 'Entry Signal' for {ticker}: {e}")
            validation_logger.error(traceback.format_exc())


def analyze_latest_interval(data_dir=INDICATORS_DIR, tz='Asia/Kolkata'):
    """
    Analyze the latest 15-minute interval row for each ticker in the main_indicators directory.
    Determine the entry signal ("Yes" or "No") using apply_trend_conditions and exact interval matching.
    """
    timezone = pytz.timezone(tz)
    current_time = datetime.now(timezone)
    minute_block = (current_time.minute // 15) * 15
    expected_interval = current_time.replace(minute=minute_block, second=0, microsecond=0)

    pattern = os.path.join(data_dir, '*_main_indicators.csv')
    files = glob.glob(pattern)

    for f in files:
        ticker = os.path.basename(f).replace('_main_indicators.csv', '')

        try:
            df = pd.read_csv(f, parse_dates=['date'])
            if df.empty:
                print(f"[{ticker}] No data found.")
                continue

            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize(timezone)
            else:
                df['date'] = df['date'].dt.tz_convert(timezone)

            latest_rows = df[df['date'].dt.floor('15min') == expected_interval]
            if latest_rows.empty:
                print(f"[{ticker}] No row found for the expected interval ({expected_interval}).")
                continue

            # Take the last row for this interval
            latest_row_index = latest_rows.index[-1]

            last_trading_day = df['date'].dt.date.max()
            today_start_time = timezone.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
            today_end_time = timezone.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))

            today_data = df[(df['date'] >= today_start_time) & (df['date'] <= today_end_time)].copy()
            if today_data.empty:
                df.at[latest_row_index, 'Entry Signal'] = "No"
                df.to_csv(f, index=False)
                print(f"[{ticker}] No intraday data for today; Entry Signal set to No.")
                continue

            daily_change = calculate_daily_change(today_data)
            ticker_entries = apply_trend_conditions(ticker, last_trading_day, india_tz)

            # Identify if expected_interval matches any entry time exactly
            entry_times = []
            for entry in ticker_entries:
                entry_time = pd.to_datetime(entry['date'], errors='coerce')
                if entry_time is not None and not pd.isnull(entry_time):
                    if entry_time.tzinfo is None:
                        entry_time = timezone.localize(entry_time)
                    else:
                        entry_time = entry_time.astimezone(timezone)
                    entry_times.append(entry_time)

            matched_entry = any(t == expected_interval for t in entry_times)
            entry_signal = "Yes" if matched_entry else "No"
            df.at[latest_row_index, 'Entry Signal'] = entry_signal

            df.to_csv(f, index=False)
            print(f"[{ticker}] Latest interval {expected_interval}: Entry Signal set to {entry_signal}")

        except Exception as e:
            print(f"[{ticker}] Error analyzing latest interval: {e}")




'''
# In your main() function, you would call analyze_latest_interval before create_papertrade_file or other steps:

def main():
    # ... previous setup code ...

    # Call the function to analyze and print entry signals before creating any files
    analyze_latest_interval()

    # Now proceed with whatever steps you have:
    # For example, calling create_papertrade_file() or similar functions
    # ...
    # create_papertrade_file(...)
    # ...
'''

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

        # Add 'logtime' to the expected columns
        expected_columns = ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "ADX", "Signal_ID", "logtime"]

        # Validate the DataFrame
        if not validate_dataframe(earliest_entries, ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "ADX", "Signal_ID", "logtime"]):
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

            logging.debug(f"Combined Entries Columns before date parsing: {combined_entries.columns.tolist()}")
            print(f"Combined Entries Columns before date parsing: {combined_entries.columns.tolist()}")

            # Ensure 'date' is datetime with timezone info for accurate deduplication
            try:
                combined_entries['date'] = pd.to_datetime(combined_entries['date'], errors='raise')
            except Exception as e:
                logging.error(f"Error converting 'date' in combined_entries: {e}")
                logging.error(traceback.format_exc())
                raise

            # Sort combined_entries by 'date'
            combined_entries = sort_dataframe_by_date(combined_entries)
            logging.debug(f"Combined Entries after date parsing:\n{combined_entries.head()}")
            print(f"Combined Entries after date parsing:\n{combined_entries.head()}")

            # Re-validate with the added 'logtime' column
            for col in expected_columns:
                if col not in combined_entries.columns:
                    combined_entries[col] = ""

            # Add the logtime for when the entry is actually written to the CSV
            # This will record the exact time the entry is appended
            current_logtime = datetime.now(india_tz).isoformat()
            combined_entries['logtime'] = current_logtime

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
                all_papertrade_entries.to_csv(output_file, index=False)
            else:
                combined_entries.to_csv(output_file, index=False)

            logging.info(f"Combined papertrade data saved to {output_file}")
            print(f"Combined papertrade data saved to '{output_file}'")

            # **Optional:** Read and print the papertrade data
            try:
                papertrade_data = pd.read_csv(output_file)
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
            return

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
        logging.error(traceback.format_exc())
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

    # Call the function to analyze and print entry signals before creating any files
    analyze_latest_interval()

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
#                    send_notification("Trading Script Alert", f"'Entry Signal' was missing for {ticker}. It has been initialized to 'No'.")
                else:
                    # Optionally, validate the values in 'Entry Signal'
                    valid_signals = {"Yes", "No"}
                    invalid_signals = set(data["Entry Signal"].unique()) - valid_signals
                    if invalid_signals:
                        logging.warning(f"Invalid 'Entry Signal' values {invalid_signals} in {combined_csv_path}. Correcting to 'No'.")
                        data["Entry Signal"] = data["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
                        data.to_csv(combined_csv_path, index=False)
                        logging.info(f"Corrected invalid 'Entry Signal' values for {ticker}.")
#                       send_notification("Trading Script Alert", f"Invalid 'Entry Signal' values were corrected for {ticker}.")
            except Exception as e:
                logging.error(f"Error validating {combined_csv_path}: {e}")
#               send_notification("Trading Script Error", f"Error validating {combined_csv_path}: {e}")



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
    end_time_naive = datetime.combine(now.date(), datetime_time(23, 59, 5))  # Changed end time to 3:00:05 PM
    
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

'''

from datetime import datetime, time
import time as t  # Renamed the time module to 't'

if __name__ == "__main__":
    # Obtain the current time in the given timezone
    now = datetime.now(india_tz)

    # Define the target time (9:15 AM local time) using datetime.time
    target_time = datetime.combine(now.date(), time(9, 15, 0, tzinfo=india_tz))

    # If it's before 9:15 AM, wait until that time
    if now < target_time:
        wait_seconds = (target_time - now).total_seconds()
        # Sleep until we reach the target time
        t.sleep(wait_seconds)

    # Perform validation before running the main processes
    validate_and_correct_main_indicators(selected_stocks, india_tz)

    # Execute the main functions
    main()
    main1()
'''

   
if __name__ == "__main__":    
    # Perform validation before running the main processes
    validate_and_correct_main_indicators(selected_stocks, india_tz)

    # Execute the main functions
    main()
    main1()


