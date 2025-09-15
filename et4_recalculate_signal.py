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
from requests.exceptions import HTTPError
# Define the timezone object for IST globally
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST

# Import the shares list from et1_select_stocklist.py
from et4_filtered_stocks_market_cap import selected_stocks  # 'shares' should be a list of stock tickers

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
        logging.FileHandler("trading_script_main.log"),  # Log messages will be written to 'trading_script.log'
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
api_semaphore = threading.Semaphore(2)  # Reduced from 4 to 2; adjust as needed


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
        logging.info(f"Dropped rows with NaT in 'date' column.")
    
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

def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=2)):
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
    rs = rs.replace([float('inf'), -float('inf')], 0)  # Replace infinities with 0
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




from kiteconnect.exceptions import NetworkException  # Import specific exceptions

import logging
from kiteconnect.exceptions import KiteException# Import available exceptions

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




import os
import logging
import pandas as pd
from datetime import timedelta
import traceback

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
            "intraday": timedelta(minutes=2)
        }
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=2))  # Default to 10 minutes if unknown type

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
    cache_dir=CACHE_DIR, freshness_threshold=timedelta(minutes=2), bypass_cache=False
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
    historical_cache_file = os.path.join(cache_dir, f"{ticker}_historical.csv")
    intraday_cache_file = os.path.join(cache_dir, f"{ticker}_intraday.csv")
    
    logging.info(f"Checking combined data cache for {ticker}.")
    combined_cache_exists = os.path.exists(combined_cache_file)
    combined_cache_fresh = is_cache_fresh(combined_cache_file, freshness_threshold) if combined_cache_exists else False
    
    
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
        freshness_threshold=timedelta(minutes=2),  # Uses the parameter passed to the main function
        bypass_cache=True
    )
    
    '''
    # Step 2: Check Combined Cache
    logging.info(f"Checking combined data cache for {ticker}.")
    combined_cache_exists = os.path.exists(combined_cache_file)
    combined_cache_fresh = is_cache_fresh(combined_cache_file, freshness_threshold) if combined_cache_exists else False

    if combined_cache_exists and combined_cache_fresh and not bypass_cache:
        logging.info(f"Loading cached combined data for {ticker}.")
        combined_data = pd.read_csv(combined_cache_file, parse_dates=['date'])
        combined_data['date'] = pd.to_datetime(combined_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
        
        # **Ensure 'Entry Signal' exists**
        if "Entry Signal" not in combined_data.columns:
            combined_data["Entry Signal"] = "No"
            logging.info(f"'Entry Signal' column missing in cached combined data for {ticker}. Initialized to 'No'.")
        else:
            # Ensure no invalid data exists in 'Entry Signal' column
            combined_data["Entry Signal"] = combined_data["Entry Signal"].apply(lambda x: x if x in ["Yes", "No"] else "No")
            logging.debug(f"'Entry Signal' column already present for {ticker}.")
        
        return combined_data  # Return combined data directly if cache is fresh
    '''    

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



# EMAIL

def send_notification(subject, message):
    """
    Send an email notification. Configure SMTP settings as per your email provider.

    Parameters:
        subject (str): Subject of the email.
        message (str): Body of the email.
    """
    import smtplib
    from email.mime.text import MIMEText

    # SMTP Configuration - Update these with your actual SMTP server details
    SMTP_SERVER = 'smtp.example.com'          # e.g., 'smtp.gmail.com' for Gmail
    SMTP_PORT = 587                           # Common ports: 587 (TLS), 465 (SSL)
    SMTP_USERNAME = 'your_email@example.com'  # Your email address
    SMTP_PASSWORD = 'your_password'           # Your email password or app-specific password

    # Recipient Email
    RECIPIENT_EMAIL = 'recipient_email@example.com'  # Where notifications will be sent

    # Email Content
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = SMTP_USERNAME
    msg['To'] = RECIPIENT_EMAIL

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Upgrade the connection to secure
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info("Notification email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send notification email: {e}")
        
def ensure_all_indicators(data, ticker):
    """
    Ensure that all required technical indicators are present in the DataFrame.
    If any are missing, calculate and add them.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data.
        ticker (str): The trading symbol of the stock.

    Returns:
        pd.DataFrame: DataFrame with all required indicators or None if failure.
    """
    required_columns = {
        'RSI', 'ATR', 'MACD', 'Signal_Line',
        'Upper_Band', 'Lower_Band', 'VWAP',
        '20_SMA', 'Recent_High', 'Recent_Low', 'Stochastic'
    }
    missing_columns = required_columns - set(data.columns)
    logging.debug(f"Current columns for {ticker}: {data.columns.tolist()}")
    logging.debug(f"Missing columns for {ticker}: {missing_columns}")

    if missing_columns:
        logging.info(f"Missing columns for {ticker}: {missing_columns}. Recalculating indicators.")
        try:
            # Calculate each missing indicator
            if 'RSI' in missing_columns:
                data['RSI'] = calculate_rsi(data['close'])
                logging.debug(f"Calculated RSI for {ticker}:\n{data['RSI'].tail()}")
            if 'ATR' in missing_columns:
                data['ATR'] = calculate_atr(data)
                logging.debug(f"Calculated ATR for {ticker}:\n{data['ATR'].tail()}")
            if 'MACD' in missing_columns or 'Signal_Line' in missing_columns:
                data['MACD'], data['Signal_Line'] = calculate_macd(data)
                logging.debug(f"Calculated MACD and Signal_Line for {ticker}:\n{data[['MACD', 'Signal_Line']].tail()}")
            if 'Upper_Band' in missing_columns or 'Lower_Band' in missing_columns:
                data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
                logging.debug(f"Calculated Bollinger Bands for {ticker}:\n{data[['Upper_Band', 'Lower_Band']].tail()}")
            if 'Stochastic' in missing_columns:
                data['Stochastic'] = calculate_stochastic(data)
                logging.debug(f"Calculated Stochastic for {ticker}:\n{data['Stochastic'].tail()}")
            if 'VWAP' in missing_columns:
                data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
                logging.debug(f"Calculated VWAP for {ticker}:\n{data['VWAP'].tail()}")
            if '20_SMA' in missing_columns:
                data['20_SMA'] = data['close'].rolling(window=20, min_periods=1).mean()
                logging.debug(f"Calculated 20_SMA for {ticker}:\n{data['20_SMA'].tail()}")
            if 'Recent_High' in missing_columns:
                data['Recent_High'] = data['high'].rolling(window=5, min_periods=1).max()
                logging.debug(f"Calculated Recent_High for {ticker}:\n{data['Recent_High'].tail()}")
            if 'Recent_Low' in missing_columns:
                data['Recent_Low'] = data['low'].rolling(window=5, min_periods=1).min()
                logging.debug(f"Calculated Recent_Low for {ticker}:\n{data['Recent_Low'].tail()}")

            # Handle NaN values after indicator calculations
            data.ffill(inplace=True)
            data.bfill(inplace=True)

            # Recheck for missing columns
            missing_columns = required_columns - set(data.columns)
            if missing_columns:
                logging.error(f"After recalculating, still missing columns for {ticker}: {missing_columns}.")
                return None
            logging.info(f"All indicators are now present for {ticker}.")
            return data
        except Exception as e:
            logging.error(f"Error recalculating indicators for {ticker}: {e}")
            logging.error(traceback.format_exc())
            return None
    else:
        logging.info(f"All required indicators are present for {ticker}.")
        return data

def apply_trend_conditions(data, daily_change, ticker, last_trading_day):
    """
    Apply trend conditions to detect Bullish or Bearish entries with stricter criteria.
    """
    ticker_entries = []
    try:
        latest_data = data.iloc[-1]
        # Stricter Bullish Conditions
        if (
            daily_change > 1.5  # Increased from 1.0 to 1.5%
            and latest_data["RSI"] > 40  # Increased from 20 to 40
            and latest_data["close"] > latest_data["VWAP"]
        ):
            # Criteria for Bullish trend
            ticker_entries.extend(
                screen_trend(
                    data, "Bullish", daily_change, 2.0, ticker, last_trading_day
                )
            )
        # Stricter Bearish Conditions
        elif (
            daily_change < -1.5  # Increased from -1.0 to -1.5%
            and latest_data["RSI"] < 60  # Decreased from 80 to 60
            and latest_data["close"] < latest_data["VWAP"]
        ):
            # Criteria for Bearish trend
            ticker_entries.extend(
                screen_trend(
                    data, "Bearish", daily_change, 2.0, ticker, last_trading_day
                )
            )
    except Exception as e:
        logging.error(f"Error applying trend conditions for {ticker}: {e}")
        #validation_logger.error(f"Error applying trend conditions for {ticker}: {e}")
    
    return ticker_entries


def screen_trend(data, trend_type, change, volume_multiplier, ticker, last_trading_day):
    """
    Screen data for bullish or bearish trend entries with stricter conditions.
    """
    trend_entries = []

    try:
        if trend_type == "Bullish":
            # Bullish Pullback Conditions (More Strict)
            pullback_condition = (
                (data["low"] >= data["VWAP"] * 0.985)  # Slightly tighter than 0.98
                & (data["close"] > data["open"])
                & (data["MACD"] > data["Signal_Line"]) & (data["MACD"] > 0)  # MACD positive and above Signal
                & (data["close"] > data["20_SMA"])
                & (data["Stochastic"] < 40)  # Lowered from 50 to 40 for more oversold pullback
                & (data["Stochastic"].diff() > 0)
            )

            # Bullish Breakout Conditions (More Strict)
            breakout_condition = (
                (data["close"] > data["high"].rolling(window=10).max() * 1.03)  # From 1.02 to 1.03
                & (
                    data["volume"]
                    > (volume_multiplier * 2.0)
                    * data["volume"].rolling(window=10).mean()
                )  # Increased from 1.5x to 2.0x average volume
                & (data["MACD"] > data["Signal_Line"]) & (data["MACD"] > 0)
                & (data["close"] > data["20_SMA"])
                & (data["Stochastic"] > 60)  # Raised from 50 to 60 for stronger momentum
                & (data["Stochastic"].diff() > 0)
            )

            conditions = {"Pullback": pullback_condition, "Breakout": breakout_condition}

        else:  # Bearish conditions (More Strict)
            pullback_condition = (
                (data["high"] <= data["VWAP"] * 1.01)  # Tighter than 1.02
                & (data["close"] < data["open"])
                & (data["MACD"] < data["Signal_Line"]) & (data["MACD"] < 0)
                & (data["close"] < data["20_SMA"])
                & (data["Stochastic"] > 60)  # Raised from 50 to 60 for a more overbought scenario
                & (data["Stochastic"].diff() < 0)
            )

            breakdown_condition = (
                (data["close"] < data["low"].rolling(window=10).min() * 0.97)  # From 0.98 to 0.97
                & (
                    data["volume"]
                    > (volume_multiplier * 2.0)
                    * data["volume"].rolling(window=10).mean()
                )  # Increased volume factor from 1.5 to 2.0
                & (data["MACD"] < data["Signal_Line"]) & (data["MACD"] < 0)
                & (data["close"] < data["20_SMA"])
                & (data["Stochastic"] < 40)  # Lower than 50 to ensure stronger downward momentum
                & (data["Stochastic"].diff() < 0)
            )

            conditions = {"Pullback": pullback_condition, "Breakdown": breakdown_condition}

        # Apply conditions and filter by last trading day
        for entry_type, condition in conditions.items():
            selected = data[condition]
            selected = selected[selected['date'].dt.date == last_trading_day]

            logging.debug(f"Entries for {entry_type} condition: {len(selected)}")

            for _, row in selected.iterrows():
                if isinstance(row["date"], pd.Timestamp):
                    formatted_date = row["date"].strftime("%Y-%m-%d %H:%M:%S%z")
                else:
                    logging.error(f"'date' is not a datetime object for ticker {ticker}. Value: {row['date']}")
                    continue

                trend_entries.append(
                    {
                        "Ticker": ticker,
                        "date": formatted_date,
                        "Entry Type": entry_type,
                        "Trend Type": trend_type,
                        "Price": round(row["close"], 2),
                        "Daily Change %": round(change, 2),
                        "VWAP": round(row["VWAP"], 2) if "Pullback" in entry_type else None,
                    }
                )

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

def recalculate_entry_signal(ticker, last_trading_day, india_tz=pytz.timezone('Asia/Kolkata'), tolerance_minutes=1):
    """
    Recalculate 'Entry Signal' for a given ticker's combined data.

    Steps:
    1. Load the ticker's *_combined.csv file.
    2. Ensure all required indicators are present; recalculate if missing.
    3. Filter today's intraday data (9:15 AM to 3:30 PM) for the last trading day.
    4. Apply trend conditions to find the expected entries.
    5. Reset and mark 'Entry Signal' as 'Yes' for these expected entries (with a small time tolerance).
    6. Save the updated DataFrame back to the CSV.

    Parameters:
        ticker (str): The trading symbol of the stock.
        last_trading_day (date): The last trading day to consider.
        india_tz (pytz.timezone): Timezone for 'Asia/Kolkata'. Defaults to India Standard Time.
        tolerance_minutes (int): Time tolerance in minutes around expected entry times.

    Returns:
        pd.DataFrame: Updated DataFrame with recalculated 'Entry Signal'.
    """
    combined_csv_path = os.path.join(CACHE_DIR, f"{ticker}_combined.csv")
    if not os.path.exists(combined_csv_path):
        logging.warning(f"Combined CSV for {ticker} does not exist at {combined_csv_path}. Cannot recalculate entry signals.")
        return pd.DataFrame()

    # Load combined CSV
    try:
        data = pd.read_csv(combined_csv_path, parse_dates=['date'])
    except Exception as e:
        logging.error(f"Error reading combined CSV for {ticker}: {e}")
        return pd.DataFrame()

    # Normalize time to ensure timezone awareness
    try:
        data = normalize_time(data)
    except Exception as e:
        logging.error(f"Error normalizing time for {ticker}: {e}")
        return pd.DataFrame()

    # Ensure all indicators
    data = ensure_all_indicators(data, ticker)
    if data is None:
        logging.error(f"Failed to recalculate indicators for {ticker}.")
        return pd.DataFrame()

    # Ensure 'Entry Signal' column exists
    if "Entry Signal" not in data.columns:
        data["Entry Signal"] = "No"
        logging.info(f"'Entry Signal' column was missing for {ticker}. Initialized to 'No'.")

    # Filter today's intraday data within market hours
    today_start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    today_end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
    today_data = data[(data['date'] >= today_start_time) & (data['date'] <= today_end_time)]

    if today_data.empty:
        logging.warning(f"No intraday data available for {ticker} on {last_trading_day}. No recalculation performed.")
        return data

    # Calculate daily percentage change
    daily_change = calculate_daily_change(today_data)

    # Apply trend conditions to detect expected entries
    expected_entries = apply_trend_conditions(today_data, daily_change, ticker, last_trading_day)

    # Collect expected entry times
    tolerance = timedelta(minutes=tolerance_minutes)
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
            logging.error(f"Invalid date format for expected entry in {ticker}: {entry['date']}")
            logging.error(traceback.format_exc())

    # Reset all 'Entry Signal' to 'No' before remarking
    data["Entry Signal"] = "No"

    # Mark expected entries as 'Yes' within tolerance
    for entry_time in expected_entry_times:
        mask = (data['date'] >= entry_time - tolerance) & (data['date'] <= entry_time + tolerance)
        data.loc[mask, "Entry Signal"] = "Yes"

    # Save the updated DataFrame back to CSV
    try:
        data.to_csv(combined_csv_path, index=False)
        logging.info(f"Recalculated 'Entry Signal' and saved updated data for {ticker} to {combined_csv_path}.")
    except Exception as e:
        logging.error(f"Error saving updated combined CSV for {ticker}: {e}")

    return data



def main1():
    # Determine the last trading day
    last_trading_day = get_last_trading_day()
    logging.info(f"Recalculating entry signals for selected stocks on last trading day: {last_trading_day}")
    print(f"Recalculating entry signals for selected stocks on last trading day: {last_trading_day}")

    # Loop over each stock and recalculate the entry signals
    for ticker in selected_stocks:
        logging.info(f"Starting recalculation for {ticker}...")
        print(f"Starting recalculation for {ticker}...")
        
        updated_data = recalculate_entry_signal(ticker, last_trading_day, india_tz)
        
        if updated_data.empty:
            logging.warning(f"No data updated for {ticker}. Check logs for errors or missing data.")
            print(f"No data updated for {ticker}.")
        else:
            logging.info(f"Recalculation complete for {ticker}.")
            print(f"Recalculation complete for {ticker}.")

    logging.info("Recalculation of entry signals completed for all selected stocks.")
    print("Recalculation of entry signals completed for all selected stocks.")


if __name__ == "__main__":
    main1()
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:58:21 2024

Updated by: [Your Name]

Description:
This script fetches intraday stock data, calculates technical indicators,
detects entry signals based on defined trend conditions, and manages data
storage and logging. All references to "Time" have been standardized to "date"
for consistency and clarity.

All scheduling logic has been removed. Running this script will execute the
`extract_entry_signals()` function once and then exit.
"""

import os
import glob
import pandas as pd
import logging
import traceback

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
        logging.FileHandler("trading_script.log"),  # Log messages to 'trading_script.log'
        logging.StreamHandler()  # Also print log messages to the console
    ]
)

# Test logging initialization
logging.info("Logging has been initialized successfully.")
print("Logging initialized and script execution started.")

def extract_entry_signals(cache_dir='data_cache', 
                          output_file='signal.csv', 
                          unique_output_file='signal_unique.csv'):
    """
    Scans all *_combined.csv files in the specified cache directory, filters rows where 'Entry Signal' is 'Yes',
    and aggregates them into a single 'signal.csv' file. Also creates and prints 'signal_unique.csv' containing the 
    first 'Yes' Entry Signal for each ticker, sorted by date.
    """
    pattern = os.path.join(cache_dir, "*_combined.csv")
    combined_files = glob.glob(pattern)

    if not combined_files:
        logging.warning(f"No combined CSV files found in directory: {cache_dir}")
        print(f"No combined CSV files found in directory: {cache_dir}")
        return

    logging.info(f"Found {len(combined_files)} combined CSV files in directory: {cache_dir}")
    print(f"Found {len(combined_files)} combined CSV files in directory: {cache_dir}")

    filtered_rows = []

    required_columns = [
        'Ticker',
        'date', 'open', 'high', 'low', 'close', 'volume',
        'RSI', 'ATR', 'MACD', 'Signal_Line',
        'Upper_Band', 'Lower_Band', 'Stochastic',
        'VWAP', '20_SMA', 'Recent_High', 'Recent_Low', 'Entry Signal'
    ]

    for file in combined_files:
        try:
            logging.info(f"Processing file: {file}")
            print(f"Processing file: {file}")
            df = pd.read_csv(file)

            if 'Ticker' not in df.columns:
                basename = os.path.basename(file)
                ticker = basename.split('_combined.csv')[0]
                df['Ticker'] = ticker
                logging.info(f"Added 'Ticker' column with value '{ticker}' to DataFrame.")
                print(f"Added 'Ticker' column with value '{ticker}' to DataFrame.")
            else:
                unique_tickers = df['Ticker'].unique()
                if len(unique_tickers) != 1:
                    logging.warning(f"Multiple tickers found in {file}: {unique_tickers}. Skipping this file.")
                    print(f"Multiple tickers found in {file}: {unique_tickers}. Skipping this file.")
                    continue
                else:
                    ticker = unique_tickers[0]

            if not set(required_columns).issubset(df.columns):
                missing = set(required_columns) - set(df.columns)
                logging.error(f"File {file} is missing columns: {missing}. Skipping this file.")
                print(f"File {file} is missing columns: {missing}. Skipping this file.")
                continue

            df['Entry Signal'] = df['Entry Signal'].astype(str)
            filtered = df[df['Entry Signal'].str.lower() == 'yes']

            if not filtered.empty:
                logging.info(f"Found {len(filtered)} 'Yes' Entry Signals in {file}.")
                print(f"Found {len(filtered)} 'Yes' Entry Signals in {file}.")
                filtered_rows.append(filtered)
            else:
                logging.debug(f"No 'Yes' Entry Signals found in {file}.")
                print(f"No 'Yes' Entry Signals found in {file}.")

        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            logging.error(traceback.format_exc())
            print(f"Error processing file {file}: {e}")
            continue

    if filtered_rows:
        signals_df = pd.concat(filtered_rows, ignore_index=True)
        signals_df = signals_df[required_columns]

        if not pd.api.types.is_datetime64_any_dtype(signals_df['date']):
            try:
                signals_df['date'] = pd.to_datetime(signals_df['date'], errors='raise')
                logging.info("Converted 'date' column to datetime.")
                print("Converted 'date' column to datetime.")
            except Exception as e:
                logging.error(f"Error converting 'date' to datetime: {e}")
                logging.error(traceback.format_exc())
                print(f"Error converting 'date' to datetime: {e}")
                return

        signals_df = signals_df.sort_values(by='date').reset_index(drop=True)
        logging.info("Sorted the aggregated signals by 'date' in increasing order.")
        print("Sorted the aggregated signals by 'date' in increasing order.")

        try:
            signals_df.to_csv(output_file, index=False)
            logging.info(f"Successfully saved {len(signals_df)} entry signals to '{output_file}'.")
            print(f"Successfully saved {len(signals_df)} entry signals to '{output_file}'.")
        except Exception as e:
            logging.error(f"Error saving to '{output_file}': {e}")
            logging.error(traceback.format_exc())
            print(f"Error saving to '{output_file}': {e}")

        try:
            signal_unique_df = signals_df.drop_duplicates(subset=['Ticker'], keep='first').sort_values(by='date').reset_index(drop=True)
            signal_unique_df.to_csv(unique_output_file, index=False)
            logging.info(f"Successfully saved unique entry signals to '{unique_output_file}'.")
            print(f"Successfully saved unique entry signals to '{unique_output_file}'.")

            print("\nUnique Entry Signals (First 'Yes' per Ticker, Sorted by Date):")
            print(signal_unique_df)
        except Exception as e:
            logging.error(f"Error creating or saving '{unique_output_file}': {e}")
            logging.error(traceback.format_exc())
            print(f"Error creating or saving '{unique_output_file}': {e}")
    else:
        logging.info("No 'Yes' Entry Signals found across all combined CSV files.")
        print("No 'Yes' Entry Signals found across all combined CSV files.")


def main():
    # Running on-demand: Just call extract_entry_signals and exit
    extract_entry_signals()


if __name__ == "__main__":
    main()
    