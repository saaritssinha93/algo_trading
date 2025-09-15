# -*- coding: utf-8 -*-
"""
Fetch 15-minute data for the last 200 days for each ticker in `selected_stocks`,
calculate indicators, and store the output in CSV files under `main_indicators_test`.
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect.exceptions import NetworkException, KiteException
import pandas as pd
import numpy as np
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
import traceback
from requests.exceptions import HTTPError
import glob
import argparse
import signal
import sys

# ================================
# Global Configuration
# ================================
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for all date operations

# Import the list of stocks
from et4_filtered_stocks_market_cap import selected_stocks

CACHE_DIR = "data_cache1"
INDICATORS_DIR = "main_indicators_test"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

api_semaphore = threading.Semaphore(2)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Reduce verbosity

file_handler = logging.FileHandler("logs\\trading_data.log")
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging initialized successfully.")
print("Logging initialized and script execution started.")

# Change to a valid working directory
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info(f"Changed working directory to {cwd}.")
except FileNotFoundError:
    logger.error(f"Working directory {cwd} not found.")
    raise
except Exception as e:
    logger.error(f"Unexpected error changing directory: {e}")
    raise

# List of market holidays
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

def setup_kite_session():
    """
    Reads local `access_token.txt` and `api_key.txt` to establish a KiteConnect session.
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
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

kite = setup_kite_session()

def refresh_instrument_list():
    """
    Refresh the instrument list from the Kite API (NSE exchange).
    """
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        updated_tokens = dict(zip(instrument_df['tradingsymbol'], instrument_df['instrument_token']))
        logging.info("Instrument list refreshed successfully.")
        return updated_tokens
    except Exception as e:
        logging.error(f"Error refreshing instrument list: {e}")
        return {}

try:
    shares_tokens = refresh_instrument_list()
except:
    shares_tokens = {}

def get_tokens_for_stocks(stocks):
    """
    Fetch the instrument tokens for given `stocks` from the instrument dump.
    """
    try:
        logging.info("Fetching tokens for provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][['tradingsymbol','instrument_token']]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        raise

def get_tokens_for_stocks_with_retry(stocks, retries=3, delay=5):
    """
    Repeatedly tries to fetch instrument tokens for the given `stocks`.
    """
    for attempt in range(1, retries + 1):
        try:
            return get_tokens_for_stocks(stocks)
        except Exception as e:
            logging.error(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Exiting token fetch.")
                raise

if not shares_tokens:
    # If the global refresh failed, try direct fetch for selected stocks
    shares_tokens = get_tokens_for_stocks_with_retry(selected_stocks)

def make_api_call_with_retry(kite, token, from_date, to_date, interval,
                             max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Safely attempts an API call to fetch historical data with a retry mechanism.
    """
    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            logging.info(f"API call attempt {attempt+1} for token {token}. Interval: {interval}")
            data = kite.historical_data(token, from_date, to_date, interval)
            if not data:
                logging.warning(f"No data from API for token {token}, retrying...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
                continue
            logging.info(f"Data fetched for token {token}, attempt {attempt+1}.")
            return pd.DataFrame(data)
        except NetworkException as net_exc:
            if 'too many requests' in str(net_exc).lower():
                logging.warning(f"Rate limit hit, retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                raise
        except (KiteException, HTTPError) as e:
            if 'too many requests' in str(e).lower():
                logging.warning(f"Rate limit exceeded, retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                raise
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {token}: {e}")
            time.sleep(delay)
            attempt += 1
            delay *= backoff_factor
    raise APICallError(f"Failed to fetch data for {token} after {max_retries} retries.")

def cache_file_path(ticker, data_type):
    """
    Returns the filepath where data for (ticker, data_type) is cached.
    """
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")

def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=5)):
    """
    Checks if an existing cache file is 'fresh' (modified within `freshness_threshold`).
    """
    if not os.path.exists(cache_file):
        return False
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)
    current_time = datetime.now(pytz.UTC)
    return (current_time - cache_mtime) < freshness_threshold

def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Converts the 'date' column in a DataFrame to a timezone-aware datetime in `timezone`.
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Remove invalid dates

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("The 'date' column could not be converted to datetime.")

    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(timezone)

    return df

def cache_update(ticker, data_type, fetch_function, freshness_threshold=None, bypass_cache=False):
    """
    A general caching mechanism: 
      - Checks if a local CSV is fresh.
      - If not (or if bypass_cache=True), fetches fresh data from `fetch_function`.
      - Merges new data with existing, de-duplicates by 'date', saves to CSV.
    """
    cache_path = cache_file_path(ticker, data_type)
    if freshness_threshold is None:
        default_thresholds = {"historical": timedelta(hours=10), "15min": timedelta(minutes=15)}
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=15))

    if not bypass_cache and is_cache_fresh(cache_path, freshness_threshold):
        try:
            logging.info(f"Loading cached {data_type} data for {ticker}.")
            df = pd.read_csv(cache_path, parse_dates=['date'])
            df = normalize_time(df)
            return df
        except Exception:
            logging.warning("Cache load failed, fetching fresh data...")

    logging.info(f"Fetching fresh {data_type} data for {ticker}.")
    fresh_data = fetch_function(ticker)
    if fresh_data is None or fresh_data.empty:
        logging.warning(f"No fresh {data_type} data for {ticker}. Returning empty.")
        return pd.DataFrame()

    # Merge with existing data if file exists
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, parse_dates=['date'])
        existing = normalize_time(existing)
        combined = pd.concat([existing, fresh_data]).drop_duplicates(subset='date').sort_values('date')
        combined.to_csv(cache_path, index=False)
        return combined
    else:
        fresh_data.to_csv(cache_path, index=False)
        return fresh_data

# ==========================
# Technical Indicator Calculations
# ==========================
def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=timeperiod).mean()
    roll_down = down.rolling(window=timeperiod).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_atr(data: pd.DataFrame, timeperiod: int = 14) -> pd.Series:
    d = data.copy()
    d['previous_close'] = d['close'].shift(1)
    d['high_low'] = d['high'] - d['low']
    d['high_pc'] = abs(d['high'] - d['previous_close'])
    d['low_pc'] = abs(d['low'] - d['previous_close'])
    tr = d[['high_low', 'high_pc', 'low_pc']].max(axis=1)
    atr = tr.rolling(window=timeperiod, min_periods=1).mean()
    return atr

def calculate_macd(data: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({'MACD': macd, 'Signal_Line': sig, 'Histogram': hist})

def calculate_bollinger_bands(data: pd.DataFrame, period=20, up=2, dn=2):
    sma = data['close'].rolling(period, min_periods=1).mean()
    std = data['close'].rolling(period, min_periods=1).std()
    upper = sma + (std * up)
    lower = sma - (std * dn)
    return pd.DataFrame({'Upper_Band': upper, 'Lower_Band': lower})

def calculate_stochastic(data: pd.DataFrame, fastk=14, slowk=3):
    low_min = data['low'].rolling(window=fastk, min_periods=1).min()
    high_max = data['high'].rolling(window=fastk, min_periods=1).max()
    percent_k = 100.0 * (data['close'] - low_min) / (high_max - low_min + 1e-10)
    slow_k = percent_k.rolling(window=slowk, min_periods=1).mean()
    return slow_k

def calculate_adx(data: pd.DataFrame, period=14):
    d = data.copy()
    if not {'high', 'low', 'close'}.issubset(d.columns):
        raise ValueError("Missing required columns for ADX")
    d['previous_close'] = d['close'].shift(1)
    d['TR'] = d.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['previous_close']),
        abs(row['low'] - row['previous_close'])
    ), axis=1)
    d['+DM'] = np.where(
        (d['high'] - d['high'].shift(1)) > (d['low'].shift(1) - d['low']),
        np.where((d['high'] - d['high'].shift(1)) > 0, d['high'] - d['high'].shift(1), 0),
        0
    )
    d['-DM'] = np.where(
        (d['low'].shift(1) - d['low']) > (d['high'] - d['high'].shift(1)),
        np.where((d['low'].shift(1) - d['low']) > 0, d['low'].shift(1) - d['low'], 0),
        0
    )
    d['TR_smooth'] = d['TR'].rolling(period, min_periods=period).sum()
    d['+DM_smooth'] = d['+DM'].rolling(period, min_periods=period).sum()
    d['-DM_smooth'] = d['-DM'].rolling(period, min_periods=period).sum()

    # Wilder's smoothing
    for i in range(period, len(d)):
        d.at[d.index[i], 'TR_smooth'] = (
            d.at[d.index[i-1], 'TR_smooth']
            - (d.at[d.index[i-1], 'TR_smooth'] / period)
            + d.at[d.index[i], 'TR']
        )
        d.at[d.index[i], '+DM_smooth'] = (
            d.at[d.index[i-1], '+DM_smooth']
            - (d.at[d.index[i-1], '+DM_smooth'] / period)
            + d.at[d.index[i], '+DM']
        )
        d.at[d.index[i], '-DM_smooth'] = (
            d.at[d.index[i-1], '-DM_smooth']
            - (d.at[d.index[i-1], '-DM_smooth'] / period)
            + d.at[d.index[i], '-DM']
        )

    d['+DI'] = (d['+DM_smooth'] / d['TR_smooth']) * 100
    d['-DI'] = (d['-DM_smooth'] / d['TR_smooth']) * 100
    d['DX'] = (abs(d['+DI'] - d['-DI']) / (d['+DI'] + d['-DI'] + 1e-10)) * 100
    d['ADX'] = d['DX'].rolling(window=period, min_periods=period).mean()

    for i in range(2 * period, len(d)):
        d.at[d.index[i], 'ADX'] = (
            (d.at[d.index[i-1], 'ADX'] * (period - 1)) + d.at[d.index[i], 'DX']
        ) / period

    return d['ADX']

# ==========================
# Indicators for 15-Min Data
# ==========================
def recalc_indicators_for_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all required indicators for the full DataFrame.
    Also adds a 'Daily Change' column:
      'close' - the day's first 'open', grouped by the date portion.
    """
    if df.empty:
        return df.copy()

    df = df.copy().reset_index(drop=True)

    try:
        df['RSI'] = calculate_rsi(df['close'])
        df['ATR'] = calculate_atr(df)

        macd_df = calculate_macd(df)
        df['MACD'] = macd_df['MACD']
        df['Signal_Line'] = macd_df['Signal_Line']
        df['Histogram'] = macd_df['Histogram']

        bb_df = calculate_bollinger_bands(df)
        df['Upper_Band'] = bb_df['Upper_Band']
        df['Lower_Band'] = bb_df['Lower_Band']

        df['Stochastic'] = calculate_stochastic(df)

        # VWAP (cumulative)
        vwap_series = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['VWAP'] = vwap_series

        df['20_SMA'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['Recent_High'] = df['high'].rolling(window=5, min_periods=1).max()
        df['Recent_Low'] = df['low'].rolling(window=5, min_periods=1).min()
        df['ADX'] = calculate_adx(df, period=14)
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        raise

    # Forward/back fill initial rolling windows
    df = df.ffill().bfill()

    # "Daily Change": for each calendar date, difference between the bar's close and that day's first open
    df['Daily Change'] = df['close'] - df.groupby(df['date'].dt.date)['open'].transform('first')
    return df

def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def get_last_trading_day(reference_date=None):
    """
    Returns the most recent trading day relative to `reference_date`.
    Skips weekends and `market_holidays`.
    """
    if reference_date is None:
        reference_date = datetime.now().date()
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

# ============================
# Fetch 15-Min Data, Last 200 Days
# ============================
def fetch_15min_data_with_indicators(ticker, token, from_date, to_date, interval="15minute"):
    """
    Fetches 15-minute historical data for the given date range,
    calculates all indicators, and returns the full DataFrame.
    """
    def fetch_15m(_):
        df = make_api_call_with_retry(
            kite, token,
            from_date.strftime("%Y-%m-%d %H:%M:%S"),
            to_date.strftime("%Y-%m-%d %H:%M:%S"),
            interval=interval
        )
        return normalize_time(df) if not df.empty else df

    # Force a fresh fetch for 15min data 
    bars_15m = cache_update(ticker, "15min", fetch_15m, bypass_cache=True)
    if bars_15m.empty:
        logging.warning(f"[{ticker}] No 15min data found between {from_date} and {to_date}.")
        return pd.DataFrame()

    # Calculate indicators
    full_with_indicators = recalc_indicators_for_all_rows(bars_15m)

    # Assign some default columns
    if 'Entry Signal' not in full_with_indicators.columns:
        full_with_indicators['Entry Signal'] = 'No'
    else:
        full_with_indicators['Entry Signal'] = full_with_indicators['Entry Signal'].fillna("No")

    now_str = datetime.now(india_tz).isoformat()
    full_with_indicators['logtime'] = now_str

    if 'Signal_ID' not in full_with_indicators.columns:
        full_with_indicators['Signal_ID'] = full_with_indicators.apply(lambda r: build_signal_id(ticker, r), axis=1)
    else:
        mask_missing = full_with_indicators['Signal_ID'].isna() | (full_with_indicators['Signal_ID'] == "")
        full_with_indicators.loc[mask_missing, 'Signal_ID'] = full_with_indicators.loc[mask_missing].apply(
            lambda r: build_signal_id(ticker, r), axis=1
        )
    return full_with_indicators

def process_ticker_15min(ticker, last_trading_day, tz):
    """
    Fetches 15-minute data for 200 calendar days back from `last_trading_day` up to `last_trading_day`,
    computes indicators, and saves the result to a CSV in `main_indicators_test`.
    """
    try:
        with api_semaphore:
            token = shares_tokens.get(ticker)
            if not token:
                logging.warning(f"[{ticker}] No token found.")
                return None

            # We'll define from_date as 200 days before last_trading_day
            # to_date as last_trading_day end-of-day
            from_date = tz.localize(datetime.combine(last_trading_day - timedelta(days=200), datetime_time(0, 0)))
            to_date = tz.localize(datetime.combine(last_trading_day, datetime_time(23, 59)))

            df_15m = fetch_15min_data_with_indicators(ticker, token, from_date, to_date, interval="15minute")
            if df_15m.empty:
                logging.warning(f"[{ticker}] 15min data is empty over the last 200 days.")
                return None

        # Sort by date and save
        df_15m.sort_values('date', inplace=True)
        out_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        df_15m.to_csv(out_path, index=False)

        logging.info(f"[{ticker}] Saved 15min data with indicators to {out_path}.")
        return df_15m
    except Exception as e:
        logging.error(f"[{ticker}] Error processing 15min data: {e}")
        logging.error(traceback.format_exc())
        return None

# =====================
# Main Execution
# =====================
def signal_handler(sig, frame):
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """
    For each ticker in `selected_stocks`:
      1) Determine the last trading day.
      2) Fetch 15-minute data for the last 200 days.
      3) Calculate indicators.
      4) Save the final DataFrame in `main_indicators_test`.
    """
    last_trading_day = get_last_trading_day()
    processed_data_dict = {}

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_ticker = {
                executor.submit(process_ticker_15min, t, last_trading_day, india_tz): t
                for t in selected_stocks
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        processed_data_dict[ticker] = data
                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")

        if not processed_data_dict:
            print("No data fetched for any ticker.")
        else:
            print("15-minute data with indicators for the last 200 days has been saved in:", INDICATORS_DIR)
    except Exception as e:
        logging.error(f"Error in processing tickers: {e}")

if __name__ == "__main__":
    main()
