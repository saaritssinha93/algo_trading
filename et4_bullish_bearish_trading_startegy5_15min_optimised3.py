# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:58:21 2024
Updated by: Saarit Shankar Sinha

Description:
Fetches intraday stock data, calculates technical indicators,
detects entry signals based on trend conditions, and manages data
storage and logging. "Time" references standardized to "date".
"""

# ================================
# Import Modules
# ================================

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect.exceptions import NetworkException  # Import specific exceptions
from kiteconnect.exceptions import KiteException# Import available exceptions
import pandas as pd
import numpy as np
import pytz
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
import traceback
from requests.exceptions import HTTPError
import shutil

# Define the timezone object for IST globally
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST

from et4_filtered_stocks_market_cap import selected_stocks  # List of stock tickers

# ================================
# Setup Logging
# ================================

# Set working directory
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logging.info(f"Changed working directory to {cwd}.")
except FileNotFoundError:
    logging.error(f"Working directory {cwd} not found.")
    raise

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_script_main.log"),
        logging.StreamHandler()
    ]
)

logging.info("Logging initialized successfully.")
print("Logging initialized and script execution started.")

# ================================
# Define Global Variables
# ================================

# Market holidays for 2024
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

# Ensure cache directory exists
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Limit concurrent API calls
api_semaphore = threading.Semaphore(2)

# ================================
# Define Helper Functions
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
        # Attempt to parse 'date' column
        df['date'] = pd.to_datetime(df['date'], errors='raise')
    except Exception as e:
        logging.error(f"'date' column validation failed: {e}")
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
shares_tokens = get_tokens_for_stocks(selected_stocks)

# ================================
# Data Fetching Functions
# ================================

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



# Assuming CACHE_DIR and other dependencies are already defined
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
            "intraday": timedelta(minutes=5)
        }
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=5))  # Default to 10 minutes if unknown type

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
    cache_dir=CACHE_DIR, freshness_threshold=timedelta(minutes=5), bypass_cache=False
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
        bypass_cache (bool): Whether to bypass cache and fetch fresh data.

    Returns:
        pd.DataFrame: Combined DataFrame with historical and intraday data, including indicators.
    """
    combined_cache_file = os.path.join(cache_dir, f"{ticker}_combined.csv")
    
    logging.info(f"Checking combined data cache for {ticker}.")
    combined_cache_exists = os.path.exists(combined_cache_file)
    
    
    # Step 1: Fetch Intraday Data using cache_update
    def fetch_intraday(ticker):
        """
        Fetch intraday data for the given ticker.

        Parameters:
            ticker (str): The trading symbol of the stock.

        Returns:
            pd.DataFrame: Intraday data DataFrame.
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
                # Recalculate missing indicators
                # intraday_data = ensure_all_indicators(intraday_data, ticker)
                if intraday_data.empty:
                    logging.error(f"Failed to calculate indicators for intraday data of {ticker}.")
                    return pd.DataFrame()
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
        freshness_threshold=timedelta(minutes=5),  # Uses the parameter passed to the main function
        bypass_cache=True
    )
        

    # Step 3: Fetch Historical Data using cache_update
    def fetch_historical(ticker):
        """
        Fetch historical data for the given ticker.

        Parameters:
            ticker (str): The trading symbol of the stock.

        Returns:
            pd.DataFrame: Historical data DataFrame.
        """
        from_date_hist = from_date_intra - timedelta(days=5)  # Increased buffer for more comprehensive historical data
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
                # Recalculate indicators if missing
                # historical_data = ensure_all_indicators(historical_data, ticker)
                if historical_data.empty:
                    logging.error(f"Failed to calculate indicators for historical data of {ticker}.")
                    return pd.DataFrame()
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
        bypass_cache=bypass_cache
    )
    

    
    # Step 4: Combine Historical and Intraday Data
    if historical_data.empty and intraday_data.empty:
        logging.warning(f"No data available for {ticker}. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Combining historical and intraday data for {ticker}.")
    combined_data = pd.concat([historical_data, intraday_data]).drop_duplicates(subset='date').sort_values(by='date')
    
    
    
    # Step 5: Calculate Indicators if Combined Data is Not Cached
    if not combined_cache_exists or bypass_cache:
        if not combined_data.empty:
            logging.info(f"Calculating indicators on combined data for {ticker}.")
            combined_data = ensure_all_indicators(combined_data, ticker)
            if combined_data is None:
                logging.error(f"Failed to calculate all indicators for {ticker}.")
                return pd.DataFrame()

    # Step 6: Save Combined Data to Cache
    combined_data.to_csv(combined_cache_file, index=False)
    logging.info(f"Saved combined data with indicators for {ticker} to cache.")

    return combined_data

# ================================
# Entry Signal Detection Functions
# ================================

def process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day, india_tz):
    """
    Process a single ticker:
    - Fetch data
    - Combine and validate data
    - Detect entry signals
    - Save updated data to cache

    Parameters:
        ticker (str): The trading symbol of the stock.
        from_date_intra (datetime): Start datetime for intraday data.
        to_date_intra (datetime): End datetime for intraday data.
        last_trading_day (date): The last trading day to consider.
        india_tz (pytz.timezone): Timezone object for 'Asia/Kolkata'.

    Returns:
        tuple: (data, ticker_entries)
            - data (pd.DataFrame): Processed data with indicators.
            - ticker_entries (list): List of detected entry signals.
    """
    with api_semaphore:  # Acquire semaphore before making API call
        token = shares_tokens.get(ticker)
        if not token:
            logging.warning(f"No token found for ticker {ticker}. Skipping.")
            return None, []  # Skip if token is not found

        intraday_data = fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra)
        data = combine_and_validate_data(intraday_data, ticker, india_tz)

        if data is None:
            return None, []  # Skip if data is invalid

        # Ensure all indicators are present
        data = ensure_all_indicators(data, ticker)
        if data is None:
            logging.error(f"Failed to ensure all indicators for {ticker}. Skipping.")
            return None, []  # Skip if indicators are still missing

        # Continue with processing (entry detection, etc.)
        # Filter today's data within market hours
        today_start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
        today_end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
        today_data = data[(data['date'] >= today_start_time) & (data['date'] <= today_end_time)]

        if today_data.empty:
            logging.warning(f"No intraday data available after combining for {ticker}. Skipping.")
            return data, []  # Return data with 'Entry Signal' as 'No'

        # Calculate daily percentage change using today's data
        daily_change = calculate_daily_change(today_data)

        # Apply trend conditions to detect entries
        ticker_entries = apply_trend_conditions(today_data, daily_change, ticker, last_trading_day)

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

        # **Recalculate 'Entry Signal' if missing**
        if "Entry Signal" not in data.columns:
            data["Entry Signal"] = "No"
            logging.info(f"'Entry Signal' column missing for {ticker}. Initialized to 'No' and recalculating.")

        # Mark 'Entry Signal' as 'Yes' for detected entries
        mark_entry_signals(data, entry_times, india_tz)

        # **Validation Before Saving**
        if "Entry Signal" not in data.columns:
            logging.error(f"'Entry Signal' column is missing after marking for {ticker}.")
            raise KeyError(f"'Entry Signal' column is missing after marking for {ticker}.")

        # Log the unique values in 'Entry Signal' before saving
        logging.debug(f"'Entry Signal' values for {ticker} before saving: {data['Entry Signal'].unique()}")

        # Save the updated combined data to CSV
        combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
        temp_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined_temp.csv")

        try:
            # Write to a temporary CSV first
            data.to_csv(temp_csv_path, index=False)
            logging.debug(f"Temporary CSV saved for {ticker} at {temp_csv_path}")

            # Replace the old CSV with the new one atomically
            os.replace(temp_csv_path, combined_csv_path)
            logging.info(f"Saved combined data with indicators and entry signals for {ticker} to cache.")

            # **Post-Save Validation**
            # Reload the saved combined CSV to verify 'Entry Signal'
            saved_data = pd.read_csv(combined_csv_path, parse_dates=['date'])
            if "Entry Signal" not in saved_data.columns:
                logging.error(f"'Entry Signal' column is missing in the saved CSV for {ticker}.")
                raise KeyError(f"'Entry Signal' column is missing in the saved CSV for {ticker}.")
            else:
                # Optionally, validate the values in 'Entry Signal'
                valid_signals = {"Yes", "No"}
                invalid_signals = set(saved_data["Entry Signal"].unique()) - valid_signals
                if invalid_signals:
                    logging.warning(f"Invalid 'Entry Signal' values {invalid_signals} in saved CSV for {ticker}. Correcting to 'No'.")
                    saved_data["Entry Signal"] = saved_data["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
                    saved_data.to_csv(combined_csv_path, index=False)
                    logging.info(f"Corrected invalid 'Entry Signal' values in saved CSV for {ticker}.")
                    send_notification("Trading Script Alert", f"Invalid 'Entry Signal' values were corrected for {ticker}.")
                else:
                    logging.debug(f"'Entry Signal' column successfully saved for {ticker}.")

        except Exception as e:
            logging.error(f"Error saving or validating combined CSV for {ticker}: {e}")
            # Attempt to remove the temporary CSV if exists
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                logging.info(f"Removed temporary CSV for {ticker} due to error.")

        return data, ticker_entries  # Return the DataFrame and list of entries


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
    data['previous_close'] = data['close'].shift(1)
    data['high_low'] = data['high'] - data['low']
    data['high_pc'] = abs(data['high'] - data['previous_close'])
    data['low_pc'] = abs(data['low'] - data['previous_close'])
    tr = data[['high_low', 'high_pc', 'low_pc']].max(axis=1)
    atr = tr.rolling(window=timeperiod, min_periods=1).mean()
    data.drop(['previous_close', 'high_low', 'high_pc', 'low_pc'], axis=1, inplace=True)
    return atr

def calculate_macd(data: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
    """Calculate MACD and Signal Line."""
    ema_fast = data['close'].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data: pd.DataFrame, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2, matype: int = 0):
    """Calculate Bollinger Bands."""
    sma = data['close'].rolling(window=timeperiod, min_periods=1).mean()
    std = data['close'].rolling(window=timeperiod, min_periods=1).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band, lower_band

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
    data['ADX'] = data['DX'].rolling(window=period, min_periods=period).mean()

    # After the initial period, continue smoothing ADX
    for i in range(2 * period, len(data)):
        data.at[data.index[i], 'ADX'] = (
            (data.at[data.index[i-1], 'ADX'] * (period - 1)) + data.at[data.index[i], 'DX']
        ) / period

    # Clean up intermediate columns
    data.drop(['previous_close', 'TR', '+DM', '-DM', 'TR_smooth', '+DM_smooth', '-DM_smooth', '+DI', '-DI', 'DX'], axis=1, inplace=True)

    return data['ADX']

# ------------------ Indicator Recalculation ------------------

def recalc_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate all required indicators for the given data.
    This is a helper function to keep the ensure_all_indicators() logic clean.

    Parameters:
    - data (pd.DataFrame): DataFrame containing price and volume data.

    Returns:
    - pd.DataFrame: DataFrame with recalculated indicators.
    """
    # Reset index to ensure uniqueness
    data = data.reset_index(drop=True)

    data['RSI'] = calculate_rsi(data['close'])
    data['ATR'] = calculate_atr(data)
    data['MACD'], data['Signal_Line'], _ = calculate_macd(data)
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    data['Stochastic'] = calculate_stochastic(data)
    data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    data['20_SMA'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['Recent_High'] = data['high'].rolling(window=5, min_periods=1).max()
    data['Recent_Low'] = data['low'].rolling(window=5, min_periods=1).min()
    
    # Calculate ADX and add to DataFrame
    data['ADX'] = calculate_adx(data, period=14)
    
    # Handle any potential NaN values resulting from indicator calculations
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    return data


def check_duplicates(data: pd.DataFrame, ticker: str):
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


# ------------------ Trend Conditions ------------------

def apply_trend_conditions(data: pd.DataFrame, daily_change: float, ticker: str, last_trading_day: pd.Timestamp) -> list:
    """
    Detects Bullish or Bearish trend entries based on defined criteria.

    Parameters:
    - data (pd.DataFrame): DataFrame containing price and indicator data.
    - daily_change (float): Daily percentage change.
    - ticker (str): Stock ticker symbol.
    - last_trading_day (pd.Timestamp): The date of the last trading day.

    Returns:
    - list: List of detected trend entries.
    """
    trend_entries = []
    try:
        latest_data = data.iloc[-1]

        # Define common condition for strong trend using ADX
        strong_trend = latest_data["ADX"] > 25

        if strong_trend:
            if (
                daily_change > 1.0
                and latest_data["RSI"] > 25
                and latest_data["close"] > latest_data["VWAP"]
            ):
                # Bullish trend detected
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
                trend_entries.extend(
                    screen_trend(
                        data, "Bearish", daily_change, 1.5, ticker, last_trading_day
                    )
                )
    except Exception as e:
        logging.error(f"Error applying trend conditions for {ticker}: {e}")
        validation_logger.error(f"Error applying trend conditions for {ticker}: {e}")
    
    return trend_entries

def screen_trend(data: pd.DataFrame, trend_type: str, change: float, volume_multiplier: float, ticker: str, last_trading_day: pd.Timestamp) -> list:
    """
    Identifies specific trend entry points (Pullback/Breakout or Pullback/Breakdown) based on trend type.

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
                ),
                "Breakout": (
                    (data["close"] > data["high"].rolling(window=10).max() * 1.015)
                    & (data["volume"] > (volume_multiplier * 2.0) * data["volume"].rolling(window=10).mean())
                    & (data["MACD"] > data["Signal_Line"])
                    & (data["MACD"] > 0)
                    & (data["close"] > data["20_SMA"])
                    & (data["Stochastic"] > 55)
                    & (data["Stochastic"].diff() > 0)
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
                ),
                "Breakdown": (
                    (data["close"] < data["low"].rolling(window=10).min() * 0.97)
                    & (data["volume"] > (volume_multiplier * 2.0) * data["volume"].rolling(window=10).mean())
                    & (data["MACD"] < data["Signal_Line"])
                    & (data["MACD"] < 0)
                    & (data["close"] < data["20_SMA"])
                    & (data["Stochastic"] < 40)
                    & (data["Stochastic"].diff() < 0)
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
            # Localize entry times if not already timezone-aware
            localized_times = [
                entry_time if entry_time.tzinfo else india_tz.localize(entry_time) for entry_time in entry_times
            ]
            # Ensure 'date' column is in the same timezone
            data['date'] = data['date'].dt.tz_convert(india_tz)
            # Mark 'Entry Signal' as 'Yes' where the 'date' matches any of the entry times
            data.loc[data['date'].isin(localized_times), "Entry Signal"] = "Yes"
            logging.debug(f"Marked 'Entry Signal' for times: {localized_times}")
        else:
            logging.debug("No entry times detected; no 'Entry Signal' marked.")
    except Exception as e:
        logging.error(f"Error marking entry signals: {e}")
        logging.error(traceback.format_exc())


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


# ================================
# Error Handling and Notification
# ================================

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
        
        
def validate_all_combined_csv(selected_stocks):
    """
    Validate all combined CSV files to ensure the 'Entry Signal' column exists.
    Recalculate if missing.
    """
    for ticker in selected_stocks:
        combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
        if os.path.exists(combined_csv_path):
            try:
                data = pd.read_csv(combined_csv_path, parse_dates=['date'])
                if "Entry Signal" not in data.columns:
                    logging.warning(f"'Entry Signal' missing in {combined_csv_path}. Recalculating.")
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

                send_notification("Trading Script Error", f"Error validating {combined_csv_path}: {e}")


# ================================
# Entry Signal Detection Functions
# ================================

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
    final_columns = ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "Signal_ID"]
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
                "Signal_ID": entry.get("Signal_ID"),
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
            existing = pd.read_csv(filename, parse_dates=['date'], dtype=str)
            existing = normalize_time(existing)

            for c in final_columns:
                if c not in existing.columns:
                    existing[c] = ""

            existing = existing[final_columns]

            # If Signal_ID missing, create it
            if 'Signal_ID' not in existing.columns:
                existing["Signal_ID"] = existing.apply(
                    lambda row: f"{row['Ticker']}-{pd.to_datetime(row['date']).astimezone(pytz.UTC).isoformat()}",
                    axis=1
                )

            # Rewrite file to ensure consistent columns and Signal_ID
            existing.to_csv(filename, index=False)
            return pd.read_csv(filename, parse_dates=['date'], dtype=str)
        return None

    # Save to individual ticker CSV files
    for ticker in entries_df['Ticker'].unique():
        ticker_filename = f"{ticker}_price_action_entries_15min_{today_str}.csv"
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
        if not validate_dataframe(earliest_entries, ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "Signal_ID"]):
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
            if not validate_dataframe(combined_entries, ["Ticker", "date", "Entry Type", "Trend Type", "Price", "Daily Change %", "VWAP", "Signal_ID"]):
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
    # Set from_date and to_date for historical data (5 days back)
    past_n_days = last_trading_day - timedelta(days=5)
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

        # **New Validation Step Added Here:**
        logging.info("Validating all combined CSV files for 'Entry Signal' integrity.")
        validate_entry_signals(selected_stocks, last_trading_day, india_tz)
        logging.info("Validation of combined CSV files completed.")
        print("Validation of combined CSV files completed.")
    else:
        # Log and print if no entries were detected
        logging.info(f"No entries detected for {last_trading_day}.")
        print(f"No entries detected for {last_trading_day}.")



# Configure the main logger
logging.basicConfig(
    filename='trading_script.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a separate logger for validation results
validation_logger = logging.getLogger('ValidationLogger')
validation_logger.setLevel(logging.INFO)

# Create a file handler for validation results
validation_handler = logging.FileHandler('validation_results.txt')
validation_handler.setLevel(logging.INFO)

# Define a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
validation_handler.setFormatter(formatter)

# Add the handler to the validation logger
validation_logger.addHandler(validation_handler)



def validate_entry_signals(selected_stocks, last_trading_day, india_tz, validation_log_path='validation_results.txt'):
    """
    Validate the 'Entry Signal' column in all combined.csv files to ensure correctness.
    It recalculates indicators, re-applies trend conditions, and compares the expected
    entries with the existing 'Entry Signal' values.

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
        combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
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
                logging.warning(f"'Entry Signal' column missing in {combined_csv_path}. Initializing to 'No'.")
                validation_logger.warning(f"'Entry Signal' column missing in {combined_csv_path}. Initialized to 'No'.")
                data["Entry Signal"] = "No"

            # Recalculate indicators to ensure they are up-to-date
            data = ensure_all_indicators(data, ticker)
            if data is None:
                logging.error(f"Failed to recalculate indicators for {ticker}. Skipping validation.")
                validation_logger.error(f"Failed to recalculate indicators for {ticker}. Skipping validation.")
                continue

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
                # Optionally, correct the 'Entry Signal' column based on expected entries
                validation_logger.info(f"Correcting 'Entry Signal' for {ticker} based on expected entries.")
                data["Entry Signal"] = comparison_data["Entry Signal"]
                data.to_csv(combined_csv_path, index=False)
                validation_logger.info(f"'Entry Signal' column corrected and saved for {ticker}.")

        except Exception as e:
            logging.error(f"Error validating 'Entry Signal' for {ticker}: {e}")
            logging.error(traceback.format_exc())
            validation_logger.error(f"Error validating 'Entry Signal' for {ticker}: {e}")
            validation_logger.error(traceback.format_exc())
            send_notification("Trading Script Error", f"Error validating 'Entry Signal' for {ticker}: {e}")

# ================================
# Scheduling the Script
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
    end_time_naive = datetime.combine(now.date(), datetime_time(20, 59, 0))
    
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

    def job_run():
        """
        Job function to execute the trading strategy.
        """
        try:
            run()
            logging.info("Executed trading strategy successfully.")
            print("Executed trading strategy successfully.")
        except Exception as e:
            logging.error(f"Error executing job: {e}")
            #send_notification("Trading Script Error", f"Error executing job: {e}")
            print(f"Error executing job: {e}")

    def job_validate():
        """
        Job function to execute the validation of entry signals.
        """
        try:
            last_trading_day = get_last_trading_day()
            logging.info("Executing entry signal validation.")
            validate_entry_signals(selected_stocks, last_trading_day, india_tz)
            logging.info("Entry signal validation executed successfully.")
            print("Entry signal validation executed successfully.")
        except Exception as e:
            logging.error(f"Error executing validation job: {e}")
            #send_notification("Trading Script Error", f"Error executing validation job: {e}")
            print(f"Error executing validation job: {e}")

    # ================================
    # Schedule the Jobs
    # ================================

    for scheduled_time in scheduled_times:
        # Convert scheduled_time string back to datetime object for comparison
        scheduled_datetime_naive = datetime.strptime(scheduled_time, "%H:%M:%S")
        scheduled_datetime = india_tz.localize(datetime.combine(now.date(), scheduled_datetime_naive.time()))
        
        if scheduled_datetime > now:
            schedule.every().day.at(scheduled_time).do(job_run)
            logging.info(f"Scheduled run job at {scheduled_time} IST for ticker(s): {selected_stocks}")
        else:
            logging.debug(f"Skipped scheduling run job at {scheduled_time} IST as it's already past.")

    # Schedule the validation job to run immediately after the last run job
    schedule_time = (end_time + timedelta(seconds=5)).strftime("%H:%M:%S")
    schedule.every().day.at(schedule_time).do(job_validate)
    logging.info(f"Scheduled validation job at {schedule_time} IST for ticker(s): {selected_stocks}")

    logging.info("Scheduled `run` to execute every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    print("Scheduled `run` to execute every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    
    # ================================
    # Immediate Execution Check
    # ================================

    # Optionally, execute the job immediately if within trading hours
    if start_time <= now <= end_time:
        logging.info("Executing initial run of trading strategy.")
        print("Executing initial run of trading strategy.")
        job_run()
    
    # ================================
    # Keep the Script Running to Allow Scheduled Jobs
    # ================================

    while True:
        schedule.run_pending()  # Run any pending scheduled jobs
        time.sleep(1)  # Sleep to prevent high CPU usage

if __name__ == "__main__":
    main()  # Execute the main function when the script is run

