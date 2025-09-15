# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:44:28 2025

@author: Saarit
"""

"""
Modified script without 15-minute scheduling.
 - No repeated scheduling using schedule.
 - The 'run()' function is still intact for a single pass of data fetching & indicator updates.
 - 'Entry Signal' is always initialized to "No" in new rows.
 - Data fetching, indicator calculations, and incremental row additions remain the same.

Edited to run every 15 minutes from 9:30:15 AM to 3:30:15 PM.
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

# We import a list of stocks from another file
from et4_filtered_stocks_market_cap import selected_stocks
#from et4_filtered_stocks import selected_stocks

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

api_semaphore = threading.Semaphore(2)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Base level to avoid too verbose logs

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

def refresh_instrument_list():
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        shares_tokens_updated = dict(zip(instrument_df['tradingsymbol'], instrument_df['instrument_token']))
        logging.info("Instrument list refreshed successfully.")
        return shares_tokens_updated
    except Exception as e:
        logging.error(f"Error refreshing instrument list: {e}")
        return shares_tokens  # Return existing tokens if refresh fails

def normalize_time(df, timezone='Asia/Kolkata'):
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

def setup_kite_session():
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

def get_tokens_for_stocks_with_retry(stocks, retries=3, delay=5):
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

def get_tokens_for_stocks(stocks):
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

refresh_instrument_list()
shares_tokens = get_tokens_for_stocks_with_retry(selected_stocks)

def make_api_call_with_retry(kite, token, from_date, to_date, interval, max_retries=5, initial_delay=1, backoff_factor=2):
    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            logging.info(f"API call attempt {attempt+1} for token {token}.")
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
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")

def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=5)):
    if not os.path.exists(cache_file):
        return False
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)
    current_time = datetime.now(pytz.UTC)
    return (current_time - cache_mtime) < freshness_threshold

def cache_update(ticker, data_type, fetch_function, freshness_threshold=None, bypass_cache=False):
    cache_path = cache_file_path(ticker, data_type)
    if freshness_threshold is None:
        default_thresholds = {"historical": timedelta(hours=10), "intraday": timedelta(minutes=15)}
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

    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, parse_dates=['date'])
        existing = normalize_time(existing)
        combined = pd.concat([existing, fresh_data]).drop_duplicates(subset='date').sort_values('date')
        combined.to_csv(cache_path, index=False)
        return combined
    else:
        fresh_data.to_csv(cache_path, index=False)
        return fresh_data

def fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra, interval="15minute"):
    def fetch_intraday(_):
        df = make_api_call_with_retry(
            kite, token,
            from_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
            to_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
            interval
        )
        return normalize_time(df) if not df.empty else df

    def fetch_historical(_):
        from_date_hist = from_date_intra - timedelta(days=10)
        to_date_hist   = from_date_intra - timedelta(minutes=1)
        df = make_api_call_with_retry(
            kite, token,
            from_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
            to_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
            interval
        )
        return normalize_time(df) if not df.empty else df

    intraday_data = cache_update(ticker, "intraday", fetch_intraday, bypass_cache=True)
    historical_data = cache_update(ticker, "historical", fetch_historical)

    combined = pd.concat([historical_data, intraday_data], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    combined.drop_duplicates(subset=['date'], keep='first', inplace=True)
    combined.sort_values('date', inplace=True)

    latest_row = recalc_indicators_for_latest_row(combined)
    if latest_row.empty:
        return pd.DataFrame()

    if 'Entry Signal' not in latest_row.columns:
        latest_row['Entry Signal'] = 'No'
    if 'logtime' not in latest_row.columns:
        latest_row['logtime'] = datetime.now(india_tz).isoformat()

    if 'Signal_ID' not in latest_row.columns:
        latest_row['Signal_ID'] = latest_row.apply(lambda r: build_signal_id(ticker, r), axis=1)
    else:
        mask_missing = latest_row['Signal_ID'].isna() | (latest_row['Signal_ID'] == "")
        latest_row.loc[mask_missing, 'Signal_ID'] = latest_row.loc[mask_missing].apply(
            lambda r: build_signal_id(ticker, r), axis=1
        )

    return latest_row

def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

# =====================
# Indicator Calculations
# =====================
def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=timeperiod).mean()
    roll_down = down.rolling(window=timeperiod).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0/(1.0 + rs))
    return rsi

def calculate_atr(data: pd.DataFrame, timeperiod: int =14) -> pd.Series:
    d = data.copy()
    d['previous_close'] = d['close'].shift(1)
    d['high_low'] = d['high'] - d['low']
    d['high_pc'] = abs(d['high'] - d['previous_close'])
    d['low_pc'] = abs(d['low'] - d['previous_close'])
    tr = d[['high_low','high_pc','low_pc']].max(axis=1)
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
    percent_k = 100 * (data['close'] - low_min) / (high_max - low_min + 1e-10)
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
        abs(row['low'] - row['previous_close'])),
        axis=1
    )
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

    # Smooth values further as per Wilder’s method
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

def recalc_indicators_for_latest_row(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    df = data.copy().reset_index(drop=True)
    last_100 = df.tail(100).copy()

    try:
        df.loc[last_100.index, 'RSI'] = calculate_rsi(last_100['close'])
        df.loc[last_100.index, 'ATR'] = calculate_atr(last_100)
        macd_df = calculate_macd(last_100)
        df.loc[last_100.index, ['MACD','Signal_Line','Histogram']] = macd_df
        boll_df = calculate_bollinger_bands(last_100)
        df.loc[last_100.index, ['Upper_Band','Lower_Band']] = boll_df
        stoch = calculate_stochastic(last_100)
        df.loc[last_100.index, 'Stochastic'] = stoch
        vwap_series = (last_100['close'] * last_100['volume']).cumsum() / last_100['volume'].cumsum()
        df.loc[last_100.index, 'VWAP'] = vwap_series
        df.loc[last_100.index, '20_SMA'] = last_100['close'].rolling(window=20, min_periods=1).mean()
        df.loc[last_100.index, 'Recent_High'] = last_100['high'].rolling(window=5, min_periods=1).max()
        df.loc[last_100.index, 'Recent_Low']  = last_100['low'].rolling(window=5, min_periods=1).min()
        adx_series = calculate_adx(last_100, period=14)
        df.loc[last_100.index, 'ADX'] = adx_series
    except Exception as e:
        logging.error(f"Indicator calculation error in recalc_indicators_for_latest_row: {e}")
        raise

    df = df.ffill().bfill()
    latest_row = df.iloc[[-1]].copy()
    return latest_row


# ===========================
# Main Process Logic (Single)
# ===========================
def process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day, india_tz):
    try:
        with api_semaphore:
            token = shares_tokens.get(ticker)
            if not token:
                logging.warning(f"[{ticker}] No token found.")
                return None, []

            latest_row = fetch_intraday_data_with_historical(
                ticker, token, from_date_intra, to_date_intra, interval="15minute"
            )
            if latest_row.empty:
                logging.warning(f"[{ticker}] No new row returned.")
                return None, []

        main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_updated.csv")
        if os.path.exists(main_path):
            existing = pd.read_csv(main_path, parse_dates=['date'])
            existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
        else:
            existing = pd.DataFrame()

        latest_row = normalize_time(latest_row, timezone='Asia/Kolkata')

        if 'Signal_ID' not in latest_row.columns:
            logging.error(f"[{ticker}] Latest row missing 'Signal_ID'.")
            return None, []

        if not existing.empty and 'Signal_ID' in existing.columns:
            all_ids = set(existing['Signal_ID'].astype(str).unique())
            new_id = str(latest_row.iloc[0]['Signal_ID'])
            if new_id in all_ids:
                logging.info(f"[{ticker}] Row with Signal_ID={new_id} already exists. Skipping.")
                return None, []

        # Always set Entry Signal to "No"
        latest_row.at[latest_row.index[0], 'Entry Signal'] = 'No'
        latest_row['logtime'] = datetime.now(india_tz).isoformat()

        combined = pd.concat([existing, latest_row], ignore_index=True)
        #if 'Signal_ID' in combined.columns:
        #    combined.drop_duplicates(subset=['Signal_ID'], keep='first', inplace=True)
        combined.drop_duplicates(subset=['date'], keep='first', inplace=True)
        combined.sort_values('date', inplace=True)
        combined.to_csv(main_path, index=False)
        logging.info(f"[{ticker}] Appended new row date={latest_row.iloc[0]['date']} to {main_path}.")

        return latest_row, []
    except Exception as e:
        logging.error(f"[{ticker}] Error processing: {e}")
        logging.error(traceback.format_exc())
        return None, []

def validate_and_correct_main_indicators(selected_stocks, india_tz):
    for ticker in selected_stocks:
        path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_updated.csv")
        if not os.path.exists(path):
            logging.warning(f"No main indicators for {ticker}.")
            continue
        try:
            df = load_and_normalize_csv(path, timezone='Asia/Kolkata')
            if "Entry Signal" not in df.columns:
                df["Entry Signal"] = "No"
                df.to_csv(path, index=False)
                logging.info(f"Added 'Entry Signal' to {ticker}.")
            else:
                valid_signals = {"Yes", "No"}
                df["Entry Signal"] = df["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
                df.to_csv(path, index=False)
        except Exception as e:
            logging.error(f"Error validating {path}: {e}")

def get_last_trading_day(reference_date=None):
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

def prepare_time_ranges(last_trading_day, india_tz):
    past_n_days = last_trading_day - timedelta(days=7)
    from_date_hist = india_tz.localize(datetime.combine(past_n_days, datetime_time.min))
    to_date_hist = india_tz.localize(datetime.combine(last_trading_day - timedelta(days=1), datetime_time.max))
    from_date_intra = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    to_date_intra = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
    now = datetime.now(india_tz)
    if now.date() == last_trading_day and now.time() < datetime_time(15, 30):
        to_date_intra = now
    return from_date_hist, to_date_hist, from_date_intra, to_date_intra

def get_latest_expected_interval(current_time=None, tz='Asia/Kolkata'):
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
    expected_interval = get_latest_expected_interval(current_time, tz=tz)
    pattern = os.path.join(data_dir, '*_main_indicators_updated.csv')
    files = glob.glob(pattern)
    timezone = pytz.timezone(tz)

    results = {}
    for f in files:
        ticker = os.path.basename(f).replace('_main_indicators_updated.csv', '')
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

def check_missing_signals(data_dir='main_indicators', tz='Asia/Kolkata'):
    current_time = datetime.now()
    results = check_csv_intervals(data_dir=data_dir, current_time=current_time, tz=tz)
    timezone = pytz.timezone(tz)
    expected_interval = get_latest_expected_interval(current_time, tz=tz)

    for ticker, exists in results.items():
        if not exists:
            print(f"[{ticker}] No entry found for the expected interval ({expected_interval}).")

def signal_handler(sig, frame):
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run():
    """
    A single-pass function that:
    - Determines last trading day
    - Prepares intraday time range
    - For each ticker, fetches new row with updated indicators and appends to CSV
    """
    last_trading_day = get_last_trading_day()
    _, _, from_date_intra, to_date_intra = prepare_time_ranges(last_trading_day, india_tz)

    def wrapper(tkr):
        return process_ticker(tkr, from_date_intra, to_date_intra, last_trading_day, india_tz)

    processed_data_dict = {}

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_ticker = {executor.submit(wrapper, t): t for t in selected_stocks}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    res = future.result()
                    if res is not None:
                        data, _ = res
                        if data is not None:
                            processed_data_dict[ticker] = data
                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")
    except Exception as e:
        logging.error(f"Error in parallel processing: {e}")

    if not processed_data_dict:
        logging.info("No new data for any ticker.")
        print("No new data fetched.")
    else:
        print("Data fetching and indicator updates done for tickers.")

def main():
    initialize_calc_flag_in_main_indicators()
    validate_and_correct_main_indicators(selected_stocks, india_tz)
    run()
    check_missing_signals(data_dir=INDICATORS_DIR, tz='Asia/Kolkata')
    initialize_calc_flag_in_main_indicators()
    print("1st iteration is complete")
    time.sleep(10)
    validate_and_correct_main_indicators(selected_stocks, india_tz)
    run()
    initialize_calc_flag_in_main_indicators()
    check_missing_signals(data_dir=INDICATORS_DIR, tz='Asia/Kolkata')
    print("2nd iteration is complete")
    time.sleep(10)
    validate_and_correct_main_indicators(selected_stocks, india_tz)
    run()
    initialize_calc_flag_in_main_indicators()
    check_missing_signals(data_dir=INDICATORS_DIR, tz='Asia/Kolkata')
    print("3rd iteration is complete")
    time.sleep(10) 

def main1():
    parser = argparse.ArgumentParser(description="Check CSV intervals")
    parser.add_argument("--data_dir", type=str, default="main_indicators", help="Directory containing main indicators")
    parser.add_argument("--timezone", type=str, default="Asia/Kolkata", help="Timezone")
    args = parser.parse_args()
    current_time = datetime.now()
    results = check_csv_intervals(data_dir=args.data_dir, current_time=current_time, tz=args.timezone)
    for t, e in results.items():
        if not e:
            print(t)

def initialize_calc_flag_in_main_indicators(indicators_dir="main_indicators"):
    import glob
    import os
    import pandas as pd
    import logging
    import traceback

    pattern = os.path.join(indicators_dir, "*_main_indicators_updated.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No main_indicator CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            if "CalcFlag" not in df.columns:
                df["CalcFlag"] = "No"
                logging.info(f"Added 'CalcFlag' column to {file_path} with default 'No'.")
            else:
                df["CalcFlag"] = df["CalcFlag"].fillna("")
                df.loc[df["CalcFlag"].eq(""), "CalcFlag"] = "No"

            df.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Error processing CalcFlag for {file_path}: {e}")
            logging.error(traceback.format_exc())


# =====================================================
# Run the script EVERY 15 MINUTES from 9:30:15 to 15:30:15
# =====================================================
def run_periodically():
    """
    Loop from 9:30:15 AM to 3:30:15 PM in 15-minute increments:
      1) Wait until the next scheduled time.
      2) Run the `main()` logic once.
      3) Repeat until past 3:30:15.
    """
    # Define today's date in Indian timezone
    today = datetime.now(india_tz).date()

    # Start at 9:30:15 today
    start_dt = datetime.combine(today, datetime_time(9, 30, 15))
    start_dt = india_tz.localize(start_dt)

    # End at 15:30:15 today
    end_dt = datetime.combine(today, datetime_time(15, 30, 15))
    end_dt = india_tz.localize(end_dt)

    # If code starts before 9:30:15, we wait until 9:30:15
    # If code starts after 9:30:15, we begin as soon as possible
    next_run = max(datetime.now(india_tz), start_dt)

    while True:
        now = datetime.now(india_tz)

        # If we're past 3:30:15, break
        if now > end_dt:
            print("Reached 15:30:15; stopping periodic runs.")
            break

        # If now < next_run, just sleep until next_run
        if now < next_run:
            time.sleep((next_run - now).total_seconds())

        # --- RUN once ---
        main()  # or just call run() if you want a lighter call

        # Schedule the next run time = current + 15 min
        next_run = next_run + timedelta(minutes=15)
        # But if next_run is already beyond 3:30:15, we will break in next loop iteration.


# Kick off the repeated scheduling when we run this script
if __name__ == "__main__":
    run_periodically()
    # Optionally call main1 at the very end if you like:
    # main1()

