# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:48:52 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:20:29 2024

@author: Saarit
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time
import pandas as pd
import pytz
from kiteconnect import KiteConnect
import traceback
import glob
import signal
import sys
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

# ================================
# Global Configuration
# ================================

india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for all date operations

# We import a list of stocks from another file
from et4_filtered_stocks_market_cap import selected_stocks

# Directories for caching data and storing main indicators
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Limit how many threads can make API calls simultaneously
api_semaphore = threading.Semaphore(2)

# Setup logging with rotation
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Base level

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Rotating File Handler
file_handler = RotatingFileHandler("trading_script_signal1.log", maxBytes=5*1024*1024, backupCount=5)  # 5 MB per file, keep 5 backups
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging initialized successfully.")
print("Logging initialized and script execution started.")

# Attempt to set working directory
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

# Define Market Holidays for 2024 - These are days no trading occurs.
market_holidays = [
    datetime(2024, 1, 26).date(),
    datetime(2024, 3, 8).date(),
    datetime(2024, 3, 25).date(),
    datetime(2024, 3, 29).date(),
    datetime(2024, 4, 11).date(),
    datetime(2024, 4, 17).date(),
    datetime(2024, 5, 1).date(),
    datetime(2024, 6, 17).date(),
    datetime(2024, 7, 17).date(),
    datetime(2024, 8, 15).date(),
    datetime(2024, 10, 2).date(),
    datetime(2024, 11, 1).date(),
    datetime(2024, 11, 15).date(),
    datetime(2024, 11, 20).date(),
    datetime(2024, 12, 25).date(),
]

class APICallError(Exception):
    """Custom Exception for API call failures after retries"""
    pass



SIGNALS_DB = "generated_signals.json"

import json

def load_generated_signals():
    """
    Load the set of already generated Signal_IDs from a JSON file.
    """
    if os.path.exists(SIGNALS_DB):
        with open(SIGNALS_DB, 'r') as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                logging.error("Signals DB JSON is corrupted. Starting with an empty set.")
                return set()
    else:
        return set()

def save_generated_signals(generated_signals):
    """
    Save the set of generated Signal_IDs to a JSON file.
    """
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f)


def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Ensure 'date' column is in datetime format and set to the correct timezone.
    If the 'date' column is naive (no timezone), we first treat it as UTC and then convert.
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("The 'date' column could not be converted to datetime.")

    # If no timezone, assume UTC then convert to given timezone
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(timezone)

    return df

def load_and_normalize_csv(file_path, expected_cols=None, timezone='Asia/Kolkata'):
    """
    Load a CSV, ensure it has a 'date' column in proper timezone,
    and if some expected columns are missing, add them.
    """
    if not os.path.exists(file_path):
        # If file not found, return empty DataFrame with the expected columns
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, timezone=timezone)
        df = df.sort_values(by='date').reset_index(drop=True)
    else:
        # If no 'date' column, just ensure other columns exist
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""

    # Ensure all expected columns are present
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[expected_cols]

    return df

def setup_kite_session():
    """
    Setup the KiteConnect session using saved API keys and access tokens.
    Without proper keys, the script cannot fetch data.
    """
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()
        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        print("Kite Connect session initialized successfully.")
        return kite
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        print(f"Error: {e}")
        raise

kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    """
    Given a list of stock symbols, fetch their corresponding instrument tokens from the Kite API.
    Tokens are needed to request historical data for each stock.
    """
    try:
        logging.info("Fetching tokens for provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        # Convert tradingsymbol to uppercase for consistency
        instrument_df['tradingsymbol'] = instrument_df['tradingsymbol'].str.upper()
        stocks_upper = [stock.upper() for stock in stocks]
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks_upper)][['tradingsymbol','instrument_token']]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        raise

shares_tokens = get_tokens_for_stocks(selected_stocks)

def build_signal_id(ticker, row):
    """
    Generate a stable Signal_ID based on (ticker + row['date']).
    The 'date' must be a localized (Asia/Kolkata) datetime so it's consistent.
    Example result: 'INFY-2024-06-01T09:30:00+05:30'
    """
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def detect_signals_in_memory(ticker: str, df_today: pd.DataFrame, existing_signal_ids: set) -> list:
    """
    Detects trading signals based on indicators, avoiding duplicates using existing_signal_ids.

    Parameters:
    - ticker (str): The stock ticker symbol.
    - df_today (pd.DataFrame): DataFrame containing today's data for the ticker.
    - existing_signal_ids (set): Set of already processed Signal_IDs.

    Returns:
    - list: List of detected signals as dictionaries.
    """
    signals_detected = []

    if df_today.empty:
        return signals_detected  # No data -> no signals

    # -- 1) Daily change
    try:
        first_open = df_today["open"].iloc[0]
        latest_close = df_today["close"].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except Exception as e:
        logging.error(f"[{ticker}] Error computing daily_change in detect_signals_in_memory: {e}")
        return signals_detected

    # -- 2) Check if ADX indicates momentum (e.g., > 35 for focused momentum)
    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get("ADX", 0)
    if adx_val <= 20:  # Further lowered ADX threshold for pullbacks
        return signals_detected

    # Decide if it's Bullish or Bearish scenario:
    if (
        daily_change > 1.25
        and latest_row.get("RSI", 0) > 35
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.02
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.25
        and latest_row.get("RSI", 0) < 65
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.98
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # -- 3) Define "Pullback" & "Breakout/Breakdown" conditions
    volume_multiplier = 1.5
    rolling_window = 20
    MACD_BULLISH_OFFSET = 0.5
    MACD_BEARISH_OFFSET = -0.5
    MACD_BULLISH_OFFSETp = -1.0
    MACD_BEARISH_OFFSETp = 1.0

    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["low"] >= df_today["VWAP"] * 0.95)  # Further loosened proximity to VWAP by allowing 5% below
                & (df_today["close"] > df_today["open"])
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSETp)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] < 60)  # Further relaxed stochastic threshold from 45 to 50
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 20)  # Maintained lowered ADX threshold
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.97)  # Tightened price movement from 0.99 to 0.97
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 60)  # Increased stochastic threshold from 60 to 65
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 30)  # Increased ADX threshold from 40 to 45
            ),
        }

    else:  # Bearish
        conditions = {
            "Pullback": (
                (df_today["high"] <= df_today["VWAP"] * 1.05)  # Further loosened proximity to VWAP by allowing 5% above
                & (df_today["close"] < df_today["open"])
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSETp)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] > 40)  # Further relaxed stochastic threshold from 55 to 60
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 20)  # Maintained lowered ADX threshold
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.03)  # Tightened price movement from 1.01 to 1.03
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 40)  # Increased stochastic threshold from 40 to 35
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 30)  # Increased ADX threshold from 40 to 45
            ),
        }

    # Collect rows matching conditions
    for entry_type, cond in conditions.items():
        selected = df_today.loc[cond]
        if not selected.empty:
            for _, row in selected.iterrows():
                # Generate or retrieve Signal_ID
                sig_id = row.get('Signal_ID', "")
                if not sig_id:
                    sig_id = build_signal_id(ticker, row)

                # Check if the signal has already been generated
                if sig_id in existing_signal_ids:
                    continue  # Skip duplicate signal

                # Append the new signal
                signals_detected.append({
                    "Ticker": ticker,
                    "date": row["date"],
                    "Entry Type": entry_type,
                    "Trend Type": trend_type,
                    "Price": round(row.get("close", 0), 2),
                    "Daily Change %": round(daily_change, 2),
                    "VWAP": round(row.get("VWAP", 0), 2),
                    "ADX": round(row.get("ADX", 0), 2),
                    "MACD": row.get("MACD", None),
                    "Signal_ID": sig_id,
                    "logtime": row.get("logtime", datetime.now(india_tz).isoformat()),
                    "Entry Signal": "Yes"
                })

    return signals_detected


def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day:
    - If today is trading day, return today
    - Else, go back until we find the last valid trading day.
    """
    if reference_date is None:
        reference_date = datetime.now(india_tz).date()
    logging.info(f"Checking last trading day from {reference_date}.")
    print(f"Reference Date: {reference_date}")
    if reference_date.weekday() < 5 and reference_date not in market_holidays:
        last_trading_day = reference_date
    else:
        last_trading_day = reference_date - timedelta(days=1)
        while last_trading_day.weekday() >= 5 or last_trading_day in market_holidays:
            last_trading_day -= timedelta(days=1)
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")
    return last_trading_day

def signal_handler(sig, frame):
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_csv_with_retries(file_path, ticker, max_retries=3, delay=5):
    """
    Attempt to process a CSV file with retries in case of transient errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = load_and_normalize_csv(file_path, timezone='Asia/Kolkata')
            return df
        except Exception as e:
            logging.error(f"[{ticker}] Attempt {attempt}: Error loading CSV - {e}")
            logging.debug(traceback.format_exc())
            if attempt < max_retries:
                logging.info(f"[{ticker}] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"[{ticker}] Failed to load CSV after {max_retries} attempts.")
                raise

from filelock import FileLock

def mark_signals_in_main_csv(ticker, signals_list, main_path, india_tz):
    """
    Update 'Entry Signal' in main_indicators.csv with synchronization.
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No existing main_indicators.csv to update.")
        return

    lock_path = f"{main_path}.lock"
    lock = FileLock(lock_path, timeout=10)  # Timeout after 10 seconds

    try:
        with lock:
            # Load existing main CSV
            df_main = load_and_normalize_csv(main_path, timezone='Asia/Kolkata')
            if df_main.empty:
                logging.warning(f"[{ticker}] main_indicators.csv is empty. No signals updated.")
                return

            if 'Entry Signal' not in df_main.columns:
                df_main['Entry Signal'] = 'No'

            # Initialize counters
            updated_signals = 0

            # Iterate over each signal and update the corresponding row
            for signal in signals_list:
                sig_id = signal.get('Signal_ID', "")
                sig_date = signal.get('date', None)

                # Option A: Use Signal_ID if available
                if sig_id and 'Signal_ID' in df_main.columns:
                    mask = (df_main['Signal_ID'] == str(sig_id))
                else:
                    # Option B: Match exact date
                    mask = (df_main['date'] == sig_date)

                # Check if any rows match the mask and Entry Signal is not already 'Yes'
                if mask.any():
                    # Only update if 'Entry Signal' is not already 'Yes'
                    mask_to_update = mask & (df_main['Entry Signal'] != 'Yes')
                    if mask_to_update.any():
                        # Update 'Entry Signal' and 'logtime' for matched rows
                        df_main.loc[mask_to_update, 'Entry Signal'] = 'Yes'
                        df_main.loc[mask_to_update, 'logtime'] = signal.get('logtime', datetime.now(india_tz).isoformat())
                        updated_signals += mask_to_update.sum()
                else:
                    logging.warning(f"[{ticker}] No matching row found for signal with Signal_ID={sig_id} and date={sig_date}.")

            # Write back the updated CSV if any updates occurred
            if updated_signals > 0:
                df_main.sort_values('date', inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] Updated 'Entry Signal'='Yes' for {updated_signals} signals.")
            else:
                logging.info(f"[{ticker}] No 'Entry Signal' updates performed.")
    except FileLock.Timeout:
        logging.error(f"[{ticker}] Timeout acquiring lock for {main_path}.")
        print(f"[{ticker}] Timeout acquiring lock for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] Error updating signals in main CSV: {e}")
        
        
import json

TRACKING_FILE = "last_processed_timestamps.json"

def load_last_processed_timestamps():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_last_processed_timestamps(timestamps):
    with open(TRACKING_FILE, 'w') as f:
        json.dump(timestamps, f)
        


def find_price_action_entries_for_today(last_trading_day):
    india_tz = pytz.timezone('Asia/Kolkata')
    # Define today's start and end times
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,25)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14,50)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        logging.warning(f"No CSV files found in directory {INDICATORS_DIR}.")
        print(f"No CSV files found in directory {INDICATORS_DIR}.")
        return pd.DataFrame()

    expected_tickers = set([ticker.upper() for ticker in selected_stocks])
    processed_tickers = set()
    all_signals = []
    total_files = len(csv_files)
    total_rows_scanned = 0
    per_ticker_signals = {}

    logging.info(f"Starting to process {total_files} CSV files.")
    print(f"Processing {total_files} CSV files.")

    # Load existing generated signals
    generated_signals = load_generated_signals()

    # Initialize tqdm progress bar
    for file_path in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','').upper()
        processed_tickers.add(ticker)
        try:
            df = process_csv_with_retries(file_path, ticker)
            if df.empty:
                logging.info(f"[{ticker}] main_indicators.csv is empty. Skipping.")
                print(f"[{ticker}] main_indicators.csv is empty. Skipping.")
                continue

            # Slice today's data
            df_today = df[(df['date'] >= start_time)&(df['date'] <= end_time)].copy()
            if df_today.empty:
                logging.info(f"[{ticker}] No data for {last_trading_day}. Skipping.")
                print(f"[{ticker}] No data for {last_trading_day}. Skipping.")
                continue

            # Additional logging
            total_rows = len(df_today)
            total_rows_scanned += total_rows
            logging.info(f"[{ticker}] Processing {total_rows} rows.")
            print(f"[{ticker}] Processing {total_rows} rows.")

            # Validate time intervals
            outside_window = df_today[(df_today['date'] < start_time) | (df_today['date'] > end_time)]
            if not outside_window.empty:
                logging.warning(f"[{ticker}] Found {len(outside_window)} rows outside the trading window.")
                print(f"[{ticker}] Found {len(outside_window)} rows outside the trading window.")

            # Check for gaps and overlaps
            df_today_sorted = df_today.sort_values('date').reset_index(drop=True)
            time_diffs = df_today_sorted['date'].diff().dropna()
            gaps = time_diffs[time_diffs > timedelta(minutes=15)]
            overlaps = time_diffs[time_diffs < timedelta(minutes=15)]

            if not gaps.empty:
                logging.warning(f"[{ticker}] Found {len(gaps)} gaps in time intervals.")
                print(f"[{ticker}] Found {len(gaps)} gaps in time intervals.")

            if not overlaps.empty:
                logging.warning(f"[{ticker}] Found {len(overlaps)} overlaps in time intervals.")
                print(f"[{ticker}] Found {len(overlaps)} overlaps in time intervals.")

            # Check for missing or invalid data
            critical_columns = ['date', 'open', 'close', 'low', 'high', 'volume', 'RSI', 'VWAP', 'ADX', 'MACD', 'Signal_Line', '20_SMA', 'Stochastic', 'Signal_ID', 'logtime', 'Entry Signal']
            missing_data = df_today[critical_columns].isnull().sum()

            if missing_data.any():
                missing_cols = missing_data[missing_data > 0].index.tolist()
                logging.warning(f"[{ticker}] Missing data in columns: {missing_cols}")
                print(f"[{ticker}] Missing data in columns: {missing_cols}")

            # Validate data types
            try:
                numeric_columns = ['open', 'close', 'low', 'high', 'volume', 'RSI', 'VWAP', 'ADX', 'MACD', 'Signal_Line', '20_SMA', 'Stochastic']
                df_today[numeric_columns] = df_today[numeric_columns].apply(pd.to_numeric, errors='coerce')
            except Exception as e:
                logging.error(f"[{ticker}] Error converting data types: {e}")
                print(f"[{ticker}] Error converting data types: {e}")

            # Validate RSI ranges (0-100)
            invalid_rsi = df_today[(df_today['RSI'] < 0) | (df_today['RSI'] > 100)]
            if not invalid_rsi.empty:
                logging.warning(f"[{ticker}] Found {len(invalid_rsi)} rows with invalid RSI values.")
                print(f"[{ticker}] Found {len(invalid_rsi)} rows with invalid RSI values.")

            # Validate ADX ranges (0-100)
            invalid_adx = df_today[(df_today['ADX'] < 0) | (df_today['ADX'] > 100)]
            if not invalid_adx.empty:
                logging.warning(f"[{ticker}] Found {len(invalid_adx)} rows with invalid ADX values.")
                print(f"[{ticker}] Found {len(invalid_adx)} rows with invalid ADX values.")

            # Detect signals
            signals_list = detect_signals_in_memory(ticker, df_today, generated_signals)
            signal_count = len(signals_list)
            logging.info(f"[{ticker}] Detected {signal_count} signals.")
            print(f"[{ticker}] Detected {signal_count} signals.")

            if signals_list:
                for signal in signals_list:
                    generated_signals.add(signal['Signal_ID'])  # Add new Signal_ID
                all_signals.extend(signals_list)
                per_ticker_signals[ticker] = signal_count
        except Exception as e:
            logging.error(f"[{ticker}] Error processing file: {e}")
            logging.debug(traceback.format_exc())
            print(f"[{ticker}] Error processing file: {e}")
            continue  # Continue with the next file

    # Identify missing tickers
    missing_tickers = expected_tickers - processed_tickers
    if missing_tickers:
        logging.warning(f"The following tickers have no corresponding CSV files: {missing_tickers}")
        print(f"The following tickers have no corresponding CSV files: {missing_tickers}")

    # Logging summary
    logging.info(f"Processed {total_files} CSV files.")
    logging.info(f"Scanned a total of {total_rows_scanned} rows across all CSV files.")
    logging.info(f"Total signals detected: {len(all_signals)}")
    for ticker, count in per_ticker_signals.items():
        logging.info(f"[{ticker}] - {count} signals detected.")

    print(f"Processed {total_files} CSV files.")
    print(f"Scanned a total of {total_rows_scanned} rows across all CSV files.")
    print(f"Total signals detected: {len(all_signals)}")
    for ticker, count in per_ticker_signals.items():
        print(f"[{ticker}] - {count} signals detected.")

    if missing_tickers:
        print(f"The following tickers have no corresponding CSV files: {missing_tickers}")

    # Save the updated generated signals
    save_generated_signals(generated_signals)

    # Return them as a DataFrame
    if not all_signals:
        logging.info("No signals detected across all CSV files.")
        return pd.DataFrame()
    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values(by='date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df



def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads the full-day 'price_action_entries' from input_file,
    ensures only the *first* (earliest) intraday entry of each Ticker
    is kept, calculates target price & quantity, then merges
    into 'papertrade_{YYYY-MM-DD}.csv' (no duplicates).
    Adds a 'time' column recording the current time for new entries.
    """
    india_tz = pytz.timezone('Asia/Kolkata')
    if not os.path.exists(input_file):
        print(f"No entries file {input_file}, skipping papertrade creation.")
        logging.warning(f"No entries file {input_file}, skipping papertrade creation.")
        return

    try:
        df_entries = pd.read_csv(input_file)
        if 'date' not in df_entries.columns:
            logging.error(f"No 'date' column in {input_file}")
            raise KeyError("No 'date' in entries file.")

        # Convert 'date' to Asia/Kolkata time
        df_entries = normalize_time(df_entries, timezone='Asia/Kolkata')
        df_entries.sort_values('date', inplace=True)

        # Filter strictly to today's (9:25–14:50)
        start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 25)))
        end_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14, 50)))
        todays_entries = df_entries[(df_entries['date'] >= start_time) & (df_entries['date'] <= end_time)].copy()

        if todays_entries.empty:
            print(f"No valid entries found for {last_trading_day}.")
            logging.info(f"No valid entries found for {last_trading_day}.")
            return

        # ---------------------------
        # Only keep *earliest* entry
        # of the day for each Ticker
        # ---------------------------
        # 1) Sort by 'date' ascending
        todays_entries.sort_values('date', inplace=True)
        # 2) Drop duplicates on ['Ticker'], keep='first'
        #    so we only have the earliest row for each Ticker
        todays_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

            # Ensure required columns
            required_cols = [
                "Ticker", "date", "Entry Type", "Trend Type", "Price",
                "Daily Change %", "VWAP", "ADX", "Signal_ID", "logtime"
            ]
            for col in required_cols:
                if col not in todays_entries.columns:
                    todays_entries[col] = ""

            # Separate Bullish vs Bearish to assign targets
            if "Trend Type" in todays_entries.columns:
                bullish = todays_entries[todays_entries["Trend Type"] == "Bullish"].copy()
                bearish = todays_entries[todays_entries["Trend Type"] == "Bearish"].copy()

                # Example logic
                if not bullish.empty:
                    bullish["Target Price"] = bullish["Price"] * 1.01
                    bullish["Quantity"] = (20000 / bullish["Price"]).astype(int)
                    bullish["Total value"] = bullish["Quantity"] * bullish["Price"]

                if not bearish.empty:
                    bearish["Target Price"] = bearish["Price"] * 0.99
                    bearish["Quantity"] = (20000 / bearish["Price"]).astype(int)
                    bearish["Total value"] = bearish["Quantity"] * bearish["Price"]

                combined_entries = pd.concat([bullish, bearish], ignore_index=True)
            else:
                combined_entries = todays_entries.copy()

            # Convert 'date' again and re-sort
            combined_entries['date'] = pd.to_datetime(combined_entries['date'], errors='coerce')
            combined_entries = normalize_time(combined_entries, timezone='Asia/Kolkata')
            combined_entries.sort_values('date', inplace=True)

            # If 'logtime' missing, fill
            if 'logtime' not in combined_entries.columns:
                # If 'logtime' missing in any row, retain existing 'logtime' or leave empty
                combined_entries['logtime'] = combined_entries['logtime'].fillna('')

        # Prepare 'time' column with current time
        current_time = datetime.now(india_tz).strftime('%H:%M:%S')
        combined_entries['time'] = current_time

        # Merge into existing papertrade CSV
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, timezone='Asia/Kolkata')
            existing.sort_values('date', inplace=True)

            # Check if 'time' column exists in existing, if not, add it
            if 'time' not in existing.columns:
                existing['time'] = ''

            # Combine existing and new entries
            all_paper = pd.concat([existing, combined_entries], ignore_index=True)

            # Drop duplicates on [Ticker, date], keeping the last (new entry)
            all_paper.drop_duplicates(subset=['Ticker', 'date'], keep='last', inplace=True)

            # Sort by 'date' ascending
            all_paper.sort_values('date', inplace=True)

            # Save to CSV
            all_paper.to_csv(output_file, index=False)
        else:
            # If output_file does not exist, ensure 'time' column is present
            if 'time' not in combined_entries.columns:
                combined_entries['time'] = current_time
            combined_entries.to_csv(output_file, index=False)

        print(f"Papertrade data (only earliest Ticker entries) saved to '{output_file}'")
        logging.info(f"Papertrade data (only earliest Ticker entries) saved to '{output_file}'")

        # Debug output
        try:
            papertrade_data = pd.read_csv(output_file)
            print("\nPapertrade Data for Today (Earliest Ticker entries):")
            print(papertrade_data)
        except Exception as e:
            logging.error(f"Error reading {output_file}: {e}")
            print(f"Error reading {output_file}: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

            

def main():
    """
    1) Determine today's (or last) trading day
    2) Find signals for each ticker from existing main_indicators
    3) Write them to price_action_entries_15min_{today}.csv
    4) Update 'Entry Signal' in main_indicators.csv
    5) Then create papertrade_{today}.csv from those signals
    """
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')

    logging.info(f"Starting main process for trading day: {last_trading_day}")
    print(f"Starting main process for trading day: {last_trading_day}")

    # Step 2: gather signals
    signals_df = find_price_action_entries_for_today(last_trading_day)
    entries_filename = f"price_action_entries_15min_{today_str}.csv"
    papertrade_filename = f"papertrade_{today_str}.csv"

    if signals_df.empty:
        logging.info(f"No signals detected for {last_trading_day}.")
        print(f"No signals detected for {last_trading_day}. Exiting.")
        return

    # Step 3: write them to entries CSV
    try:
        signals_df.to_csv(entries_filename, index=False)
        logging.info(f"Wrote {len(signals_df)} signals to '{entries_filename}'.")
        print(f"Wrote {len(signals_df)} signals to '{entries_filename}'.")
        print("\nDetected Signals for Today:")
        print(signals_df)
    except Exception as e:
        logging.error(f"Error writing signals to CSV: {e}")
        print(f"Error writing signals to CSV: {e}")
        return

    # Step 4: Update 'Entry Signal' in main_indicators.csv
    try:
        unique_signals = signals_df.drop_duplicates(subset=['Ticker', 'date'])
        for _, signal in unique_signals.iterrows():
            ticker = signal['Ticker']
            main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, [signal.to_dict()], main_path, india_tz)
        logging.info("Updated 'Entry Signal' in all main_indicators.csv files.")
        print("Updated 'Entry Signal' in all main_indicators.csv files.")
    except Exception as e:
        logging.error(f"Error updating 'Entry Signal' in main CSVs: {e}")
        print(f"Error updating 'Entry Signal' in main CSVs: {e}")
        return

    # Step 5: create papertrade
    try:
        create_papertrade_file(entries_filename, papertrade_filename, last_trading_day)
        logging.info(f"Created/Updated papertrade file '{papertrade_filename}'.")
        print(f"\nPapertrade file => {papertrade_filename}")
    except Exception as e:
        logging.error(f"Error creating/updating papertrade file: {e}")
        print(f"Error creating/updating papertrade file: {e}")
        return

    logging.info("Main process completed successfully.")
    print("Main process completed successfully.")

def continuous_run():
    """
    Runs the 'main()' function in a continuous loop every 15 seconds,
    until user interrupts (Ctrl+C) or an exception breaks out.
    """
    while True:
        try:
            main()          # the function that does your daily signals logic
            time.sleep(15)  # wait 15 seconds
        except KeyboardInterrupt:
            print("User interrupted. Exiting continuous run.")
            logging.info("User interrupted. Exiting continuous run.")
            break
        except Exception as e:
            logging.error(f"Error in continuous_run: {e}")
            logging.debug(traceback.format_exc())
            # If some error occurs, wait briefly and then try again
            time.sleep(15)

if __name__=="__main__":

    continuous_run()
