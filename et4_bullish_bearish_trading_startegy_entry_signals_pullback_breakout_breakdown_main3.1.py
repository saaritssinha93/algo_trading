# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:57:31 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Rewritten code3 with improved scheduling, logging, 
immediate writes to papertrade, and stricter detection logic.

Created on Wed Jan  1 12:54:52 2025
@author: Saarit
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import threading
import atexit
from datetime import datetime, timedelta, time as datetime_time
import pandas as pd
import pytz
from kiteconnect import KiteConnect
import traceback
import glob
import signal
import sys
from filelock import FileLock, Timeout
import json
from tqdm import tqdm
import schedule  # For improved scheduling

# ================================
# Global Configuration
# ================================

india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for all date/time operations

# We import a list of stocks from another file
from et4_filtered_stocks_market_cap import selected_stocks

# Directories for caching data and storing main indicators
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Limit how many threads can make API calls simultaneously
api_semaphore = threading.Semaphore(2)

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

# ================================
# Logging Setup (Single Handler)
# ================================

def setup_logging_for_code3(log_filename="trading_script_signal3.log"):
    """
    Initialize a single TimedRotatingFileHandler to avoid concurrency issues (WinError 32).
    Uses 'delay=True' so file is only opened upon first log write.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # or DEBUG if desired

    # Remove any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Timed rotating file handler
    handler = TimedRotatingFileHandler(
        filename=log_filename,
        when="M",               # rotate every "M" minutes
        interval=30,            # every 30 minutes
        backupCount=0,          # no old files
        delay=True              # open file only when needed
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optional console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging initialized for code3.")

def shutdown_logging():
    """ Gracefully shutdown logging at program exit. """
    logging.shutdown()

atexit.register(shutdown_logging)

# ================================
# Attempt to set working directory
# ================================
def set_working_directory():
    cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
    try:
        os.chdir(cwd)
        logging.info(f"Changed working directory to {cwd}.")
    except FileNotFoundError:
        logging.error(f"Working directory {cwd} not found.")
        logging.debug(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"Unexpected error changing directory: {e}")
        logging.debug(traceback.format_exc())
        raise

# ================================
# Kite/Session Setup
# ================================
def setup_kite_session():
    """
    Setup the KiteConnect session using saved API keys and access tokens.
    """
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()
        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        kite_obj = KiteConnect(api_key=key_secret[0])
        kite_obj.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        print("Kite Connect session initialized successfully.")
        return kite_obj
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        print(f"Error: {e}")
        raise

# Single global kite instance
kite = None

# ================================
# JSON-based Signal Storage
# ================================
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

# ================================
# Helper Functions
# ================================
def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Ensure 'date' column is in datetime format and localized to 'Asia/Kolkata'.
    If the 'date' column is naive, we localize as UTC then convert.
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("The 'date' column could not be converted to datetime.")

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
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, timezone=timezone)
        df = df.sort_values(by='date').reset_index(drop=True)
    else:
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""

    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[expected_cols]

    return df

def get_tokens_for_stocks(stocks):
    """
    Map stock symbols to instrument tokens from the Kite API.
    """
    try:
        logging.info("Fetching tokens for provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
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

def build_signal_id(ticker, row):
    """
    Generate a stable Signal_ID for a row, using ticker + row['date'].
    """
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def signal_handler(sig, frame):
    """Handle Ctrl+C or kill signals gracefully."""
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================================
# get_last_trading_day
# ================================
def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day:
    - If reference_date is a trading weekday and not in market_holidays, use it.
    - Otherwise, go back until we find a valid trading day.
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

# ================================
# detect_signals_in_memory (Refined)
# ================================
def detect_signals_in_memory(ticker: str, df_today: pd.DataFrame, existing_signal_ids: set) -> list:
    """
    Detects trading signals based on refined indicators, avoiding duplicates using existing_signal_ids.
    This version checks only the latest row (partial detection), applying stricter thresholds 
    for daily_change and ADX, then references bullish/bearish sub-routines (pullback/breakout/breakdown).

    Parameters:
    -----------
    ticker : str
        The stock ticker symbol.
    df_today : pd.DataFrame
        DataFrame containing today's data for the ticker (only the latest row is used).
    existing_signal_ids : set
        Set of already processed Signal_IDs to avoid duplicates.

    Returns:
    --------
    signals_detected : list
        A list of detected signals (dicts), each containing relevant fields
        (Ticker, date, Entry Type, Trend Type, Price, daily_change, etc.).
    """
    signals_detected = []

    # If no data, no signals
    if df_today.empty:
        return signals_detected

    # 1) Calculate daily_change
    try:
        first_open = df_today["open"].iloc[0]
        latest_close = df_today["close"].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except Exception as e:
        logging.error(f"[{ticker}] Error computing daily_change in detect_signals_in_memory: {e}")
        return signals_detected

    # 2) Check ADX momentum on the latest row
    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get("ADX", 0)
    # Example: if we want ADX > 30 for "strong momentum"
    if adx_val < 30:
        return signals_detected  # skip if momentum is not strong enough

    # Distinguish bullish vs. bearish scenario based on daily_change, RSI, close vs. VWAP
    # (These thresholds are more relaxed or more strict based on your strategy preferences.)
    if (
        daily_change > 2
        and latest_row.get("RSI", 0) > 40
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.01
    ):
        # Bullish scenario
        signals_detected.extend(
            detect_bullish_scenarios(ticker, df_today, existing_signal_ids, daily_change)
        )

    elif (
        daily_change < -2
        and latest_row.get("RSI", 100) < 60
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.99
    ):
        # Bearish scenario
        signals_detected.extend(
            detect_bearish_scenarios(ticker, df_today, existing_signal_ids, daily_change)
        )

    return signals_detected


def detect_bullish_scenarios(ticker, df_today, existing_signal_ids, daily_change):
    """
    Checks for bullish Pullback or Breakout conditions in the latest row,
    referencing the partial row detection approach. Expand or adjust as needed.
    """
    signals = []
    latest_row = df_today.iloc[-1]

    # Example offsets / thresholds used in your original logic:
    #   MACD_BULLISH_OFFSET = 0.5
    #   MACD_BULLISH_OFFSETp = -1.0
    #   etc.

    # 1) Pullback Condition
    #    For instance, "Pullback" in a bullish scenario might require a mild correction
    #    but still above VWAP, RSI condition, and ADX above some threshold.
    pullback_condition = (
        (df_today["low"] >= df_today["VWAP"] * 0.95)
        & (df_today["close"] > df_today["open"])
        & (df_today["MACD"] > df_today["Signal_Line"])
        & (df_today["MACD"] > -1.0)  # e.g., MACD_BULLISH_OFFSETp
        & (df_today["close"] > df_today["20_SMA"])
        & (df_today["Stochastic"] < 60)
        & (df_today["Stochastic"].diff() > 0)
        & (df_today["ADX"] > 20)
    )

    if not df_today.empty and pullback_condition.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(
                create_signal_dict(
                    ticker, latest_row, sig_id, "Pullback", "Bullish", daily_change
                )
            )

    # 2) Breakout Condition
    #    For example, if close > rolling max * 0.97, volume surge, MACD above a certain offset, etc.
    breakout_condition = (
        (df_today["close"] > df_today["high"].rolling(window=20).max() * 0.97)
        & (df_today["volume"] > 1.5 * df_today["volume"].rolling(window=20).mean())
        & (df_today["MACD"] > df_today["Signal_Line"])
        & (df_today["MACD"] > 0.5)  # e.g., MACD_BULLISH_OFFSET
        & (df_today["close"] > df_today["20_SMA"])
        & (df_today["Stochastic"] > 60)
        & (df_today["Stochastic"].diff() > 0)
        & (df_today["ADX"] > 30)  # e.g., ADX threshold
    )

    if not df_today.empty and breakout_condition.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(
                create_signal_dict(
                    ticker, latest_row, sig_id, "Breakout", "Bullish", daily_change
                )
            )

    return signals


def detect_bearish_scenarios(ticker, df_today, existing_signal_ids, daily_change):
    """
    Checks for bearish Pullback or Breakdown conditions in the latest row,
    referencing the partial row detection approach.
    """
    signals = []
    latest_row = df_today.iloc[-1]

    # 1) Pullback Condition (bearish)
    pullback_condition = (
        (df_today["high"] <= df_today["VWAP"] * 1.05)
        & (df_today["close"] < df_today["open"])
        & (df_today["MACD"] < df_today["Signal_Line"])
        & (df_today["MACD"] < 1.0)   # e.g. MACD_BEARISH_OFFSETp
        & (df_today["close"] < df_today["20_SMA"])
        & (df_today["Stochastic"] > 40)
        & (df_today["Stochastic"].diff() < 0)
        & (df_today["ADX"] > 20)
    )

    if not df_today.empty and pullback_condition.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(
                create_signal_dict(
                    ticker, latest_row, sig_id, "Pullback", "Bearish", daily_change
                )
            )

    # 2) Breakdown Condition
    breakdown_condition = (
        (df_today["close"] < df_today["low"].rolling(window=20).min() * 1.03)
        & (df_today["volume"] > 1.5 * df_today["volume"].rolling(window=20).mean())
        & (df_today["MACD"] < df_today["Signal_Line"])
        & (df_today["MACD"] < -0.5)  # e.g. MACD_BEARISH_OFFSET
        & (df_today["close"] < df_today["20_SMA"])
        & (df_today["Stochastic"] < 40)
        & (df_today["Stochastic"].diff() < 0)
        & (df_today["ADX"] > 30)
    )

    if not df_today.empty and breakdown_condition.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(
                create_signal_dict(
                    ticker, latest_row, sig_id, "Breakdown", "Bearish", daily_change
                )
            )

    return signals


def create_signal_dict(ticker, row, sig_id, entry_type, trend_type, daily_change):
    """
    Helper to build a consistent signal dictionary. Adjust fields/keys if needed.
    """
    return {
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
    }


# ================================
# CSV and Locking Helpers
# ================================
def process_csv_with_retries(file_path, ticker, max_retries=3, delay_sec=5):
    """
    Attempt to load a CSV file with retries in case of transient errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = load_and_normalize_csv(file_path, timezone='Asia/Kolkata')
            return df
        except Exception as e:
            logging.error(f"[{ticker}] Attempt {attempt}: Error loading CSV - {e}")
            logging.debug(traceback.format_exc())
            if attempt < max_retries:
                logging.info(f"[{ticker}] Retrying in {delay_sec} seconds...")
                time.sleep(delay_sec)
            else:
                logging.error(f"[{ticker}] Failed to load CSV after {max_retries} attempts.")
                raise

def mark_signals_in_main_csv(ticker, signals_list, main_path, india_tz):
    """
    Update 'Entry Signal' in main_indicators.csv with synchronization (FileLock).
    Only sets 'Entry Signal'='Yes' if not already set.
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No existing main_indicators.csv to update.")
        return

    lock_path = f"{main_path}.lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, timezone='Asia/Kolkata')
            if df_main.empty:
                logging.warning(f"[{ticker}] main_indicators.csv is empty. No signals updated.")
                return

            if 'Entry Signal' not in df_main.columns:
                df_main['Entry Signal'] = 'No'

            updated_signals = 0
            for signal in signals_list:
                sig_id = signal.get('Signal_ID', "")
                sig_date = signal.get('date', None)

                # Match by Signal_ID if present, else by date
                if sig_id and 'Signal_ID' in df_main.columns:
                    mask = (df_main['Signal_ID'] == str(sig_id))
                else:
                    mask = (df_main['date'] == sig_date)

                if mask.any():
                    mask_to_update = mask & (df_main['Entry Signal'] != 'Yes')
                    if mask_to_update.any():
                        df_main.loc[mask_to_update, 'Entry Signal'] = 'Yes'
                        df_main.loc[mask_to_update, 'logtime'] = signal.get(
                            'logtime',
                            datetime.now(india_tz).isoformat()
                        )
                        updated_signals += mask_to_update.sum()
                else:
                    logging.warning(f"[{ticker}] No matching row for Signal_ID={sig_id} and date={sig_date}.")

            if updated_signals > 0:
                df_main.sort_values('date', inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] Updated 'Entry Signal'='Yes' for {updated_signals} signals.")
            else:
                logging.info(f"[{ticker}] No 'Entry Signal' updates performed.")
    except Timeout:
        logging.error(f"[{ticker}] Timeout acquiring lock for {main_path}.")
        print(f"[{ticker}] Timeout acquiring lock for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] Error updating signals in main CSV: {e}")

# ================================
# find_price_action_entries_for_today
# ================================
def find_price_action_entries_for_today(last_trading_day):
    """
    Scans each *_main_indicators.csv in INDICATORS_DIR for the last trading day's data,
    detects signals from the final row, merges them into a single DataFrame.
    """
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,45)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14,20)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        logging.warning(f"No CSV files found in {INDICATORS_DIR}.")
        print(f"No CSV files found in {INDICATORS_DIR}.")
        return pd.DataFrame()

    expected_tickers = set([ticker.upper() for ticker in selected_stocks])
    processed_tickers = set()
    all_signals = []
    total_files = len(csv_files)
    total_rows_scanned = 0
    per_ticker_signals = {}

    logging.info(f"Processing {total_files} CSV files for signals.")
    print(f"Processing {total_files} CSV files.")

    generated_signals = load_generated_signals()

    for file_path in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','').upper()
        processed_tickers.add(ticker)
        try:
            df = process_csv_with_retries(file_path, ticker)
            if df.empty:
                logging.info(f"[{ticker}] main_indicators.csv is empty. Skipping.")
                print(f"[{ticker}] main_indicators.csv is empty. Skipping.")
                continue

            df_today = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()
            if df_today.empty:
                logging.info(f"[{ticker}] No data in window for {last_trading_day}. Skipping.")
                print(f"[{ticker}] No data in window for {last_trading_day}. Skipping.")
                continue

            total_rows = len(df_today)
            total_rows_scanned += total_rows
            logging.info(f"[{ticker}] Processing {total_rows} rows.")
            print(f"[{ticker}] Processing {total_rows} rows.")

            # Check for final row signals
            latest_df = df_today.iloc[-1:].copy()  # last row
            signals_list = detect_signals_in_memory(ticker, latest_df, generated_signals)
            signal_count = len(signals_list)
            logging.info(f"[{ticker}] Detected {signal_count} signals.")
            print(f"[{ticker}] Detected {signal_count} signals.")

            if signals_list:
                for sig in signals_list:
                    generated_signals.add(sig['Signal_ID'])
                all_signals.extend(signals_list)
                per_ticker_signals[ticker] = signal_count
        except Exception as e:
            logging.error(f"[{ticker}] Error processing file: {e}")
            logging.debug(traceback.format_exc())
            print(f"[{ticker}] Error processing file: {e}")
            continue

    missing_tickers = expected_tickers - processed_tickers
    if missing_tickers:
        logging.warning(f"The following tickers have no *_main_indicators.csv files: {missing_tickers}")
        print(f"The following tickers have no CSV files: {missing_tickers}")

    logging.info(f"Processed {total_files} CSV files. Rows scanned: {total_rows_scanned}. "
                 f"Signals detected: {len(all_signals)}.")
    print(f"Processed {total_files} CSV files.")
    print(f"Scanned a total of {total_rows_scanned} rows.")
    print(f"Total signals detected: {len(all_signals)}")

    for ticker, count in per_ticker_signals.items():
        print(f"[{ticker}] - {count} signals")

    save_generated_signals(generated_signals)

    if not all_signals:
        logging.info("No signals detected across all CSV files.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values(by='date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df

# ================================
# create_papertrade_file
# ================================
def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads from 'input_file' (signals CSV),
    ensures earliest daily entry per Ticker,
    appends to 'output_file' (papertrade).
    """
    if not os.path.exists(input_file):
        logging.warning(f"No entries file {input_file}. Skipping papertrade creation.")
        print(f"No entries file {input_file}, skipping papertrade creation.")
        return

    india_tz = pytz.timezone('Asia/Kolkata')
    try:
        df_entries = pd.read_csv(input_file)
        if 'date' not in df_entries.columns:
            logging.error(f"No 'date' column in {input_file}")
            raise KeyError("No 'date' in entries file.")

        df_entries = normalize_time(df_entries, timezone='Asia/Kolkata')
        df_entries.sort_values('date', inplace=True)

        # Filter strictly to today's 9:45–20:00
        start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,45)))
        end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(20)))
        todays_entries = df_entries[(df_entries['date'] >= start_time) & (df_entries['date'] <= end_time)].copy()

        if todays_entries.empty:
            print(f"No valid entries found for {last_trading_day}.")
            logging.info(f"No valid entries for {last_trading_day}.")
            return

        # Keep earliest daily entry per Ticker
        todays_entries.sort_values('date', inplace=True)
        todays_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

        required_cols = ["Ticker", "date", "Entry Type", "Trend Type", "Price",
                         "Daily Change %", "VWAP", "ADX", "Signal_ID", "logtime"]
        for col in required_cols:
            if col not in todays_entries.columns:
                todays_entries[col] = ""

        if "Trend Type" in todays_entries.columns:
            bullish = todays_entries[todays_entries["Trend Type"] == "Bullish"].copy()
            bearish = todays_entries[todays_entries["Trend Type"] == "Bearish"].copy()

            # Sample target/quantity logic
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

        combined_entries['date'] = pd.to_datetime(combined_entries['date'], errors='coerce')
        combined_entries = normalize_time(combined_entries, timezone='Asia/Kolkata')
        combined_entries.sort_values('date', inplace=True)

        # Ensure 'logtime' + 'time'
        if 'logtime' not in combined_entries.columns:
            combined_entries['logtime'] = combined_entries['logtime'].fillna('')
        current_time_str = datetime.now(india_tz).strftime('%H:%M:%S')
        combined_entries['time'] = current_time_str

        # Locking for concurrency
        lock_path = f"{output_file}.lock"
        lock = FileLock(lock_path, timeout=10)

        try:
            with lock:
                if os.path.exists(output_file):
                    existing = pd.read_csv(output_file)
                    existing = normalize_time(existing, timezone='Asia/Kolkata')
                    existing.sort_values('date', inplace=True)
                    if 'time' not in existing.columns:
                        existing['time'] = ''

                    all_paper = pd.concat([existing, combined_entries], ignore_index=True)
                    # Drop duplicates on [Ticker, date]
                    all_paper.drop_duplicates(subset=['Ticker','date'], keep='last', inplace=True)
                    all_paper.sort_values('date', inplace=True)
                    all_paper.to_csv(output_file, index=False)
                else:
                    if 'time' not in combined_entries.columns:
                        combined_entries['time'] = ''
                    combined_entries.to_csv(output_file, index=False)

            logging.info(f"Papertrade data saved to '{output_file}'")
            print(f"Papertrade data saved to '{output_file}'")

            # Optional debug print
            try:
                papertrade_data = pd.read_csv(output_file)
                print("\nPapertrade Data for Today:")
                print(papertrade_data)
            except Exception as e:
                logging.error(f"Error reading {output_file}: {e}")
                print(f"Error reading {output_file}: {e}")
        except Timeout:
            logging.error(f"Timeout acquiring lock for {output_file}.")
            print(f"Timeout acquiring lock for {output_file}.")
        except Exception as e:
            logging.error(f"Error creating papertrade file: {e}")
            print(f"Error creating papertrade file: {e}")
    except Exception as e:
        logging.error(f"Error in create_papertrade_file: {e}")

# ================================
# Main Logic
# ================================
def main():
    """
    1) Determine last trading day
    2) Gather signals into a single DataFrame
    3) Write them out to 'price_action_entries_15min_{YYYY-MM-DD}.csv'
    4) Mark 'Entry Signal' in main_indicators
    5) Create/Update papertrade file immediately
    """
    last_trading_day = get_last_trading_day()
    date_str = last_trading_day.strftime('%Y-%m-%d')

    logging.info(f"Starting main process for trading day: {last_trading_day}")
    print(f"Starting main process for trading day: {last_trading_day}")

    signals_df = find_price_action_entries_for_today(last_trading_day)
    if signals_df.empty:
        logging.info(f"No signals for {last_trading_day}. Exiting.")
        print(f"No signals detected for {last_trading_day}. Exiting.")
        return

    # Immediately write them to price_action_entries CSV
    entries_file = f"price_action_entries_15min_{date_str}.csv"
    try:
        signals_df.to_csv(entries_file, index=False)
        logging.info(f"Wrote {len(signals_df)} signals to '{entries_file}'.")
        print(f"\nDetected Signals for Today:\n{signals_df}")
    except Exception as e:
        logging.error(f"Error writing signals to CSV '{entries_file}': {e}")
        print(f"Error writing signals to '{entries_file}': {e}")
        return

    # Mark 'Entry Signal' = 'Yes' in main_indicators.csv
    try:
        unique_signals = signals_df.drop_duplicates(subset=['Ticker','date'])
        for _, sig_row in unique_signals.iterrows():
            ticker = sig_row['Ticker']
            main_file = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, [sig_row.to_dict()], main_file, india_tz)
        logging.info("Updated 'Entry Signal' in main_indicators.")
        print("Updated 'Entry Signal' in main_indicators CSVs.")
    except Exception as e:
        logging.error(f"Error updating 'Entry Signal': {e}")
        print(f"Error updating 'Entry Signal': {e}")
        return

    # Now create/append to papertrade file
    papertrade_file = f"papertrade_{date_str}.csv"
    try:
        create_papertrade_file(entries_file, papertrade_file, last_trading_day)
        logging.info(f"Papertrade file updated: {papertrade_file}")
        print(f"\nPapertrade file => {papertrade_file}")
    except Exception as e:
        logging.error(f"Error in create_papertrade_file: {e}")
        print(f"Error in create_papertrade_file: {e}")

    logging.info("Main process completed successfully.")
    print("Main process completed successfully.")

# ================================
# Scheduling Instead of 15s Sleep
# ================================
last_quarter_block = None

def check_and_run_main():
    """
    Runs 'main()' if we've hit a new 15-min boundary 
    (e.g., 09:15, 09:30, 09:45, 10:00, etc.).
    """
    global last_quarter_block
    now = datetime.now(india_tz)
    minute_block = (now.minute // 15) * 15
    if minute_block != last_quarter_block:
        last_quarter_block = minute_block
        main()

def continuous_run():
    """
    Replaces the older infinite loop that slept 15 seconds.
    We schedule check_and_run_main() every 1 minute. 
    If it's a new 15-min block, we run main().
    """
    schedule.every(1).minutes.do(check_and_run_main)
    while True:
        schedule.run_pending()
        time.sleep(1)

# ================================
# Single-Instance Lock
# ================================
def acquire_lock(lock_file_path, timeout=10):
    """
    Acquire a lock to ensure single instance execution.
    """
    lock = FileLock(lock_file_path, timeout=timeout)
    try:
        lock.acquire()
        logging.info("Global lock acquired successfully.")
        return lock
    except Timeout:
        logging.error("Another instance is running. Exiting.")
        print("Another instance is running. Exiting.")
        sys.exit(1)

# ================================
# Main Entry Point
# ================================
if __name__=="__main__":
    # 1) Setup logging
    setup_logging_for_code3("trading_script_signal3.log")

    # 2) Try setting working directory
    set_working_directory()

    # 3) Setup Kite session
    kite = setup_kite_session()

    # 4) Possibly fetch instrument tokens
    shares_tokens = get_tokens_for_stocks(selected_stocks)

    # 5) Acquire lock to prevent multiple instances
    global_lock = acquire_lock("trading_script_signal3.lock")

    try:
        # Instead of an infinite 15-sec loop, do scheduled checks
        continuous_run()
    finally:
        global_lock.release()
        logging.info("Global lock released.")
