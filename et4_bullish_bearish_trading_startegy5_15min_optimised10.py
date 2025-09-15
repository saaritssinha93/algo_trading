# -*- coding: utf-8 -*-
"""
This script automates fetching intraday stock data, updating and calculating technical indicators,
detecting trend-based entry signals, and maintaining records (main_indicators CSV files) without overwriting old data.

Key points:
- Data is fetched every 15 minutes (intraday intervals).
- Old data is never refreshed or overwritten. Only new data points are appended.
- Indicators are recalculated on a recent subset of data to save computation time.
- Entry signals are determined from the existing indicators in the main CSV files.
- A unique Signal_ID prevents reprocessing old rows.

The code also handles logging, error scenarios, and schedules tasks during market hours.
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect.exceptions import NetworkException, KiteException, DataException
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

# Directories for caching data and storing main indicators
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Limit how many threads can make API calls simultaneously
api_semaphore = threading.Semaphore(2)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # We use WARNING as the base level to avoid too verbose logs

file_handler = logging.FileHandler("trading_script_main.log")
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


def write_csv_safely(df, file_path, expected_columns):
    """
    Write DataFrame to CSV ensuring all expected columns are present and proper quoting is used.
    """
    df = df.copy()
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_columns]

    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL,
              escapechar='\\', doublequote=True)
    logging.info(f"Wrote DataFrame to {file_path}")


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
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][['tradingsymbol','instrument_token']]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        raise


shares_tokens = get_tokens_for_stocks(selected_stocks)


def make_api_call_with_retry(kite, token, from_date, to_date, interval,
                             max_retries=5, initial_delay=2, backoff_factor=2):
    """
    Attempt to fetch data from the Kite API with retries and exponential backoff.
    If rate limits or network errors occur, we'll retry up to max_retries times.
    """
    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            logging.info(f"API call attempt {attempt+1} for token {token}.")
            data = kite.historical_data(token, from_date, to_date, interval)
            if not data:
                logging.warning(f"No data from API for token {token}, retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
                continue
            logging.info(f"Data fetched for token {token}, attempt {attempt+1}.")
            return pd.DataFrame(data)

        except DataException as e:
            # check if we have the '502 Bad Gateway' text
            if "502 Bad Gateway" in str(e):
                logging.warning(f"[{token}] 502 Bad Gateway encountered. Retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                # some other data exception
                raise

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

    # if we exhausted attempts
    raise APICallError(f"Failed to fetch data for {token} after {max_retries} retries.")



def cache_file_path(ticker, data_type):
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")


def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=5)):
    """
    Check if the cache file was recently updated. If yes, no need to refetch data.
    """
    if not os.path.exists(cache_file):
        return False
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)
    current_time = datetime.now(pytz.UTC)
    return (current_time - cache_mtime) < freshness_threshold


def cache_update(ticker, data_type, fetch_function, freshness_threshold=None, bypass_cache=False):
    """
    Try to load cached data from CSV. If cache is outdated or bypass_cache=True, fetch fresh data and update cache.
    Only append new data (no old data overwrite).
    """
    cache_path = cache_file_path(ticker, data_type)
    if freshness_threshold is None:
        default_thresholds = {"historical": timedelta(hours=10),
                              "intraday": timedelta(minutes=15)}
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=15))

    if not bypass_cache and is_cache_fresh(cache_path, freshness_threshold):
        # Cache is fresh, load it
        try:
            logging.info(f"Loading cached {data_type} data for {ticker}.")
            df = pd.read_csv(cache_path, parse_dates=['date'])
            df = normalize_time(df)
            return df
        except Exception:
            logging.warning("Cache load failed, fetching fresh data...")

    # Fetch fresh data if no fresh cache
    logging.info(f"Fetching fresh {data_type} data for {ticker}.")
    fresh_data = fetch_function(ticker)
    if fresh_data is None or fresh_data.empty:
        # No new data
        logging.warning(f"No fresh {data_type} data for {ticker}. Returning empty.")
        return pd.DataFrame()

    # If we have old cache, append new data to it
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, parse_dates=['date'])
        existing = normalize_time(existing)
        combined = pd.concat([existing, fresh_data]).drop_duplicates(subset='date').sort_values('date')
        combined.to_csv(cache_path, index=False)
        return combined
    else:
        # No cache yet, just save fresh_data
        fresh_data.to_csv(cache_path, index=False)
        return fresh_data


def fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra, interval="15minute"):
    """
    1. Fetch historical + today's intraday.
    2. Combine and deduplicate.
    3. Recalculate indicators only on last 100, but *return only* the newest row.
    """
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

    # Combine them in memory
    combined = pd.concat([historical_data, intraday_data], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    combined.drop_duplicates(subset=['date'], keep='first', inplace=True)
    combined.sort_values('date', inplace=True)

    # Recalc only for the final bar
    latest_row = recalc_indicators_for_latest_row(combined)
    if latest_row.empty:
        return pd.DataFrame()

    # Ensure minimal columns on that single row
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


def get_latest_expected_interval(current_time=None, tz='Asia/Kolkata'):
    """
    Given a current_time (datetime), find the most recent 15-minute interval
    in the specified timezone. For example, if current_time is 11:31:05,
    the floored interval is 11:30:00.
    """
    timezone = pytz.timezone(tz)
    if current_time is None:
        current_time = datetime.now(timezone)
    else:
        if current_time.tzinfo is None:
            current_time = timezone.localize(current_time)
        else:
            current_time = current_time.astimezone(timezone)

    # Floor minutes to the nearest 15-min block: 0, 15, 30, 45
    minute_block = (current_time.minute // 15) * 15
    expected_interval = current_time.replace(minute=minute_block, second=0, microsecond=0)
    return expected_interval


def build_signal_id(ticker, row):
    """
    Generate a stable Signal_ID based on (ticker + row['date']).
    The 'date' must be a localized (Asia/Kolkata) datetime so it's consistent.
    Example result: 'INFY-2024-06-01T09:30:00+05:30'
    """
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"


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
    if not {'high','low','close'}.issubset(d.columns):
        raise ValueError("Missing required columns for ADX")
    d['previous_close'] = d['close'].shift(1)
    d['TR'] = d.apply(
        lambda row: max(row['high'] - row['low'],
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

    # Smooth as per Wilder’s method
    for i in range(period, len(d)):
        d.at[d.index[i], 'TR_smooth'] = (
            d.at[d.index[i-1], 'TR_smooth'] - (d.at[d.index[i-1], 'TR_smooth'] / period) + d.at[d.index[i], 'TR']
        )
        d.at[d.index[i], '+DM_smooth'] = (
            d.at[d.index[i-1], '+DM_smooth'] - (d.at[d.index[i-1], '+DM_smooth'] / period) + d.at[d.index[i], '+DM']
        )
        d.at[d.index[i], '-DM_smooth'] = (
            d.at[d.index[i-1], '-DM_smooth'] - (d.at[d.index[i-1], '-DM_smooth'] / period) + d.at[d.index[i], '-DM']
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
    """
    Recalc indicators only on the last 100 rows and return just the final row.
    """
    if data.empty:
        return pd.DataFrame()

    df = data.copy().reset_index(drop=True)
    last_100 = df.tail(100).copy()

    try:
        # RSI
        df.loc[last_100.index, 'RSI'] = calculate_rsi(last_100['close'])

        # ATR
        df.loc[last_100.index, 'ATR'] = calculate_atr(last_100)

        # MACD
        macd_df = calculate_macd(last_100)
        df.loc[last_100.index, ['MACD','Signal_Line','Histogram']] = macd_df

        # Bollinger
        boll_df = calculate_bollinger_bands(last_100)
        df.loc[last_100.index, ['Upper_Band','Lower_Band']] = boll_df

        # Stochastic
        stoch = calculate_stochastic(last_100)
        df.loc[last_100.index, 'Stochastic'] = stoch

        # VWAP
        vwap_series = (last_100['close'] * last_100['volume']).cumsum() / last_100['volume'].cumsum()
        df.loc[last_100.index, 'VWAP'] = vwap_series

        # 20 SMA
        df.loc[last_100.index, '20_SMA'] = last_100['close'].rolling(window=20, min_periods=1).mean()

        # 5-bar high/low
        df.loc[last_100.index, 'Recent_High'] = last_100['high'].rolling(window=5, min_periods=1).max()
        df.loc[last_100.index, 'Recent_Low']  = last_100['low'].rolling(window=5, min_periods=1).min()

        # ADX
        adx_series = calculate_adx(last_100, period=14)
        df.loc[last_100.index, 'ADX'] = adx_series

    except Exception as e:
        logging.error(f"Indicator calculation error in recalc_indicators_for_latest_row: {e}")
        raise

    df = df.ffill().bfill()
    latest_row = df.iloc[[-1]].copy()
    return latest_row

def apply_trend_conditions(ticker: str, last_trading_day: pd.Timestamp, india_tz):
    """
    Check the main indicators CSV for today's data and detect if any bullish/bearish signals appear,
    replicating the moderately strict logic from the detect_signals_in_memory snippet:
      - daily_change>1.0 for bullish or < -1.0 for bearish
      - ADX>30 for "strong" trend
      - RSI, VWAP multipliers, etc. as used in your detect_signals_in_memory
    Then mark those signals in the CSV.

    Returns:
        A list of signal dictionaries that were newly detected & marked in the CSV.
    """

    main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    if not os.path.exists(main_path):
        logging.warning(f"No main indicators for {ticker}.")
        return []

    df = load_and_normalize_csv(main_path, timezone='Asia/Kolkata')

    # Filter for today's intraday data
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
    today_data = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()
    if today_data.empty:
        logging.warning(f"No intraday data for {ticker} on {last_trading_day}.")
        return []

    # 1) Daily change
    try:
        first_open   = today_data['open'].iloc[0]
        latest_close = today_data['close'].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except Exception as e:
        logging.error(f"Error calculating daily change for {ticker}: {e}")
        return []

    # 2) Check ADX > 30
    latest_row = today_data.iloc[-1]
    adx_val    = latest_row.get("ADX", 0)
    if adx_val <= 22:  # replicate the "moderate" threshold
        return []

    # 3) Decide if it's Bullish or Bearish scenario
    #    Using the same logic: daily_change>1.0 for bullish, < -1.0 for bearish, etc.
    signals_detected = []
    if (
        daily_change > 1.5
        and latest_row.get("RSI", 0) > 25
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.01
    ):
        # Bullish scenario => replicate "pullback" / "breakout" checks
        signals_detected = screen_trend_for_bullish(ticker, today_data, last_trading_day, daily_change)
    elif (
        daily_change < -1.5
        and latest_row.get("RSI", 0) < 75
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.99
    ):
        # Bearish scenario => replicate "pullback" / "breakdown" checks
        signals_detected = screen_trend_for_bearish(ticker, today_data, last_trading_day, daily_change)
    else:
        # no scenario => empty
        return []

    if not signals_detected:
        return []

    # 4) Mark signals in the main CSV
    mark_signals_in_main_csv(ticker, signals_detected, main_path, india_tz, last_trading_day)
    return signals_detected




def screen_trend_for_bullish(ticker, df_today, last_day, daily_change):
    """
    Replicates the moderate logic for 'Pullback' and 'Breakout' conditions
    (Bullish scenario). Returns a list of signals that match those conditions.
    """

    signals = []
    volume_multiplier    = 1.8
    rolling_window       = 10
    MACD_BULLISH_OFFSET  = 0.1  # from your snippet

    # Define conditions from your detect_signals_in_memory snippet:
    cond_pullback = (
        (df_today["low"] >= df_today["VWAP"] * 0.99)
        & (df_today["close"] > df_today["open"])
        & (df_today["MACD"] > df_today["Signal_Line"])
        & (df_today["MACD"] > MACD_BULLISH_OFFSET)
        & (df_today["close"] > df_today["20_SMA"])
        & (df_today["Stochastic"] < 47)
        & (df_today["Stochastic"].diff() > 0)
        & (df_today["ADX"] > 22)
    )

    cond_breakout = (
        (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.995)
        & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
        & (df_today["MACD"] > df_today["Signal_Line"])
        & (df_today["MACD"] > MACD_BULLISH_OFFSET)
        & (df_today["close"] > df_today["20_SMA"])
        & (df_today["Stochastic"] > 60)
        & (df_today["Stochastic"].diff() > 0)
        & (df_today["ADX"] > 42)
    )

    # Filter the final day
    pullback_rows = df_today.loc[cond_pullback & (df_today['date'].dt.date == last_day)]
    breakout_rows = df_today.loc[cond_breakout & (df_today['date'].dt.date == last_day)]

    # Construct signal dict
    for _, row in pullback_rows.iterrows():
        signals.append({
            "date": row["date"],
            "Entry Type": "Pullback",
            "Trend Type": "Bullish",
            "Entry Signal": "Yes",
            "Signal_ID": row.get("Signal_ID", None),
            "Price": round(row.get("close", 0), 2),
            "Daily Change %": round(daily_change, 2),
            "VWAP": round(row.get("VWAP", 0), 2),
            "ADX": round(row.get("ADX", 0), 2),
            "MACD": row.get("MACD", None),
        })

    for _, row in breakout_rows.iterrows():
        signals.append({
            "date": row["date"],
            "Entry Type": "Breakout",
            "Trend Type": "Bullish",
            "Entry Signal": "Yes",
            "Signal_ID": row.get("Signal_ID", None),
            "Price": round(row.get("close", 0), 2),
            "Daily Change %": round(daily_change, 2),
            "VWAP": round(row.get("VWAP", 0), 2),
            "ADX": round(row.get("ADX", 0), 2),
            "MACD": row.get("MACD", None),
        })

    return signals



def screen_trend_for_bearish(ticker, df_today, last_day, daily_change):
    """
    Replicates the moderate logic for 'Pullback' and 'Breakdown' conditions
    (Bearish scenario). Returns a list of signals that match those conditions.
    """

    signals = []
    volume_multiplier     = 1.8
    rolling_window        = 10
    MACD_BEARISH_OFFSET   = -0.1  # from your snippet

    # Define conditions from your snippet:
    cond_pullback = (
        (df_today["high"] <= df_today["VWAP"] * 1.01)
        & (df_today["close"] < df_today["open"])
        & (df_today["MACD"] < df_today["Signal_Line"])
        & (df_today["MACD"] < MACD_BEARISH_OFFSET)
        & (df_today["close"] < df_today["20_SMA"])
        & (df_today["Stochastic"] > 53)
        & (df_today["Stochastic"].diff() < 0)
        & (df_today["ADX"] > 22)
    )

    cond_breakdown = (
        (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.005)
        & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
        & (df_today["MACD"] < df_today["Signal_Line"])
        & (df_today["MACD"] < MACD_BEARISH_OFFSET)
        & (df_today["close"] < df_today["20_SMA"])
        & (df_today["Stochastic"] < 40)
        & (df_today["Stochastic"].diff() < 0)
        & (df_today["ADX"] > 42)
    )

    pullback_rows  = df_today.loc[cond_pullback & (df_today['date'].dt.date == last_day)]
    breakdown_rows = df_today.loc[cond_breakdown & (df_today['date'].dt.date == last_day)]

    # Construct signal dict
    for _, row in pullback_rows.iterrows():
        signals.append({
            "date": row["date"],
            "Entry Type": "Pullback",
            "Trend Type": "Bearish",
            "Entry Signal": "Yes",
            "Signal_ID": row.get("Signal_ID", None),
            "Price": round(row.get("close", 0), 2),
            "Daily Change %": round(daily_change, 2),
            "VWAP": round(row.get("VWAP", 0), 2),
            "ADX": round(row.get("ADX", 0), 2),
            "MACD": row.get("MACD", None),
        })

    for _, row in breakdown_rows.iterrows():
        signals.append({
            "date": row["date"],
            "Entry Type": "Breakdown",
            "Trend Type": "Bearish",
            "Entry Signal": "Yes",
            "Signal_ID": row.get("Signal_ID", None),
            "Price": round(row.get("close", 0), 2),
            "Daily Change %": round(daily_change, 2),
            "VWAP": round(row.get("VWAP", 0), 2),
            "ADX": round(row.get("ADX", 0), 2),
            "MACD": row.get("MACD", None),
        })

    return signals



def mark_signals_in_main_csv(ticker, signals_list, main_path, india_tz, last_trading_day):
    """
    Given a list of signals (each with 'date', 'Entry Signal', 'Signal_ID', etc.),
    update the corresponding rows in `*_main_indicators.csv` so that 'Entry Signal'='Yes'
    if and only if it is currently 'No' or missing.
    We do NOT re-update if it's already 'Yes'.
    """

    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No existing main_indicators.csv to update.")
        return

    try:
        df_main = load_and_normalize_csv(main_path, timezone='Asia/Kolkata')
        if df_main.empty:
            logging.warning(f"[{ticker}] main_indicators.csv is empty. No signals updated.")
            return

        # Ensure columns exist
        required_cols = ["Entry Signal", "Trend Type", "Entry Type", "logtime"]
        for c in required_cols:
            if c not in df_main.columns:
                df_main[c] = ""

        # Force these columns to string/object dtype to avoid warnings
        df_main["Entry Signal"] = df_main["Entry Signal"].astype(str)
        df_main["Trend Type"]   = df_main["Trend Type"].astype(str)
        df_main["Entry Type"]   = df_main["Entry Type"].astype(str)
        # logtime can stay string as well
        df_main["logtime"]      = df_main["logtime"].astype(str)

        # For each signal in signals_list, match row by Signal_ID (preferred) or date
        for signal in signals_list:
            sig_id   = signal.get('Signal_ID', "")
            sig_date = signal.get('date', None)
            ent_type = signal.get('Entry Type', "")
            trd_type = signal.get('Trend Type', "")

            # Build our row mask
            if sig_id and 'Signal_ID' in df_main.columns:
                # Unique ID match, cast both sides to str
                mask = (df_main['Signal_ID'].astype(str) == str(sig_id))
            else:
                # Fallback on exact date match
                mask = (df_main['date'] == sig_date)

            # Only if row is from last_trading_day
            date_mask = (df_main['date'].dt.date == last_trading_day)
            final_mask = mask & date_mask

            # Check if it's already 'Yes'
            already_yes = df_main.loc[final_mask, 'Entry Signal'].eq("Yes").any()
            if already_yes:
                # If the row is already "Yes", skip updating
                logging.info(f"[{ticker}] Row already 'Yes'; skipping re-update for {sig_id}.")
                continue

            # Otherwise, set fields for the first time
            df_main.loc[final_mask, 'Entry Signal'] = 'Yes'
            df_main.loc[final_mask, 'Trend Type']   = trd_type
            df_main.loc[final_mask, 'Entry Type']   = ent_type
            df_main.loc[final_mask, 'logtime']      = datetime.now(india_tz).isoformat()

        # Save updated CSV
        df_main.sort_values('date', inplace=True)
        df_main.to_csv(main_path, index=False)
        logging.info(f"[{ticker}] Updated signals (no re-update if already Yes).")

    except Exception as e:
        logging.error(f"[{ticker}] Error in mark_signals_in_main_csv: {e}")






def mark_entry_signals(data: pd.DataFrame, entry_times: list, india_tz, last_trading_day: pd.Timestamp):
    if not entry_times:
        return None
    
    data = data.copy()
    if 'Entry Signal' not in data.columns:
        data['Entry Signal'] = 'No'

    localized_entry_times = []
    for et in entry_times:
        if et.tzinfo is None:
            localized_entry_times.append(india_tz.localize(et))
        else:
            localized_entry_times.append(et.astimezone(india_tz))

    for etz in localized_entry_times:
        if etz.date() == last_trading_day:
            mask = (data['date'] == etz)
            data.loc[mask, 'Entry Signal'] = 'Yes'

    return data


def process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day, india_tz):
    """
    - Fetch just the *latest* updated row with fresh indicators.
    - Only append that row to main_indicators.csv if truly new (by 'Signal_ID').
    - Do not alter older rows or reassign older indicators.
    """
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
                logging.warning(f"[{ticker}] No new row returned for {ticker}.")
                return None, []

        # Load existing CSV (as-is)
        main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        if os.path.exists(main_path):
            existing = pd.read_csv(main_path, parse_dates=['date'])
            existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
        else:
            existing = pd.DataFrame()

        # Normalize the single new row
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

        row_obj = latest_row.iloc[0]
        if row_obj.get('ADX', 0) > 24 and row_obj.get('MACD', 0) > row_obj.get('Signal_Line', 0):
            latest_row.at[latest_row.index[0], 'Entry Signal'] = 'Yes'
        else:
            latest_row.at[latest_row.index[0], 'Entry Signal'] = 'No'

        latest_row['logtime'] = datetime.now(india_tz).isoformat()

        combined = pd.concat([existing, latest_row], ignore_index=True)

        if 'Signal_ID' in combined.columns:
            combined.drop_duplicates(subset=['Signal_ID'], keep='first', inplace=True)

        combined.sort_values('date', inplace=True)
        combined.to_csv(main_path, index=False)
        logging.info(f"[{ticker}] Appended new row date={latest_row.iloc[0]['date']} to {main_path}.")

        signals_out = []
        if latest_row.iloc[0]['Entry Signal'] == 'Yes':
            signals_out.append({
                "Ticker": ticker,
                "date": str(latest_row.iloc[0]['date']),
                "ADX": row_obj['ADX'],
                "MACD": row_obj['MACD'],
                "Signal_ID": str(latest_row.iloc[0]['Signal_ID']),
                "Entry Signal": "Yes",
                "logtime": latest_row.iloc[0]['logtime']
            })

        return latest_row, signals_out

    except Exception as e:
        logging.error(f"[{ticker}] Error processing: {e}")
        logging.error(traceback.format_exc())
        return None, []


def validate_and_correct_main_indicators(selected_stocks, india_tz):
    """
    Ensure that for each ticker's main indicators file:
    - 'Entry Signal' column exists
    - If invalid values, correct them to 'No'
    """
    for ticker in selected_stocks:
        path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
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
    """
    Determine the last valid trading day:
    - If today is trading day, return today
    - Else, go back until we find the last valid trading day.
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


def prepare_time_ranges(last_trading_day, india_tz):
    """
    Prepare start/end dates for historical and intraday data requests.
    - Past 7 days for historical
    - Today's market hours for intraday
    Adjust intraday end time if current time < market close
    """
    past_n_days = last_trading_day - timedelta(days=7)
    from_date_hist = india_tz.localize(datetime.combine(past_n_days, datetime_time.min))
    to_date_hist   = india_tz.localize(datetime.combine(last_trading_day - timedelta(days=1), datetime_time.max))
    from_date_intra= india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,15)))
    to_date_intra  = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15,30)))
    now = datetime.now(india_tz)
    if now.date() == last_trading_day and now.time() < datetime_time(15,30):
        to_date_intra = now
    return from_date_hist, to_date_hist, from_date_intra, to_date_intra


def analyze_latest_interval(data_dir=INDICATORS_DIR, tz='Asia/Kolkata'):
    """
    Analyze latest 15-min interval in main_indicators for each ticker.
    Check if entry signals match conditions and update 'Entry Signal' accordingly.
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
            df = normalize_time(df, timezone=india_tz)
            if df.empty:
                print(f"[{ticker}] No data found.")
                continue

            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize(timezone)
            else:
                df['date'] = df['date'].dt.tz_convert(timezone)

            latest_rows = df[df['date'].dt.floor('15min') == expected_interval]
            if latest_rows.empty:
                print(f"[{ticker}] No row found for expected interval {expected_interval}.")
                continue

            latest_row_index = latest_rows.index[-1]

            last_trading_day = df['date'].dt.date.max()
            today_start_time = timezone.localize(datetime.combine(last_trading_day, datetime_time(9,15)))
            today_end_time   = timezone.localize(datetime.combine(last_trading_day, datetime_time(15,30)))

            today_data = df[(df['date'] >= today_start_time) & (df['date'] <= today_end_time)].copy()
            if today_data.empty:
                df.at[latest_row_index,'Entry Signal'] = "No"
                df.to_csv(f, index=False)
                print(f"[{ticker}] No intraday data today; Entry Signal=No.")
                continue

            ticker_entries = apply_trend_conditions(ticker, last_trading_day, india_tz)
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
            df.at[latest_row_index,'Entry Signal'] = entry_signal

            df.to_csv(f, index=False)
            print(f"[{ticker}] Latest interval {expected_interval}: Entry Signal={entry_signal}")

        except Exception as e:
            print(f"[{ticker}] Error analyzing latest interval: {e}")


def check_csv_intervals(data_dir='main_indicators', current_time=None, tz='Asia/Kolkata'):
    """
    Reads all *_main_indicators.csv files in the given directory, extracts the ticker from the filename,
    and checks if the last expected 15-min interval entry exists in the CSV.
    """
    expected_interval = get_latest_expected_interval(current_time, tz=tz)
    pattern = os.path.join(data_dir, '*_main_indicators.csv')
    files = glob.glob(pattern)
    timezone = pytz.timezone(tz)

    results = {}
    for f in files:
        ticker = os.path.basename(f).replace('_main_indicators.csv', '')
        
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
    """
    After we've run the main strategy, find which tickers are missing
    the latest expected 15-min interval or have no 'Entry Signal' there.
    """
    current_time = datetime.now()  # naive current time
    results = check_csv_intervals(data_dir=data_dir, current_time=current_time, tz=tz)

    timezone = pytz.timezone(tz)
    expected_interval = get_latest_expected_interval(current_time, tz=tz)

    for ticker, exists in results.items():
        file_path = os.path.join(data_dir, f"{ticker}_main_indicators.csv")
        
        if not exists:
            print(f"[{ticker}] No entry found for the expected interval ({expected_interval}), so missing Entry Signal.")
        else:
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'], utc=False)
                if df['date'].dt.tz is None:
                    df['date'] = df['date'].dt.tz_localize(timezone)
                else:
                    df['date'] = df['date'].dt.tz_convert(timezone)

                df['floored_date'] = df['date'].dt.floor('15min')
                matched = df[df['floored_date'] == expected_interval]

                if matched.empty:
                    print(f"[{ticker}] Interval found previously, but matching row is now empty?")
                    continue

                latest_row = matched.iloc[-1]
                entry_signal = latest_row.get('Entry Signal', None)

                if not entry_signal or str(entry_signal).strip().lower() not in ["yes"]:
                    print(f"[{ticker}] has the interval but missing or not 'Yes' at {expected_interval}.")
            except Exception as e:
                print(f"[{ticker}] Error processing file for missing signal check: {e}")


def backup_and_remove_corrupted_csv(ticker):
    combined_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    backup_path   = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_backup.csv")
    if os.path.exists(combined_path):
        try:
            shutil.copy(combined_path, backup_path)
            logging.info(f"Backed up corrupted CSV for {ticker} to {backup_path}")
            os.remove(combined_path)
            logging.info(f"Removed corrupted CSV {combined_path}")
        except Exception as e:
            logging.error(f"Error backing up/removing corrupted CSV for {ticker}: {e}")


def clean_up_corrupted_csvs(selected_stocks):
    for t in selected_stocks:
        backup_and_remove_corrupted_csv(t)


def validate_all_combined_csv(selected_stocks):
    for t in selected_stocks:
        path = os.path.join(INDICATORS_DIR, f"{t}_main_indicators.csv")
        if os.path.exists(path):
            try:
                df = load_and_normalize_csv(path, timezone='Asia/Kolkata')
                if "Entry Signal" not in df.columns:
                    df["Entry Signal"] = "No"
                    df.to_csv(path, index=False)
                    logging.info(f"Recalculated 'Entry Signal' for {t}.")
                else:
                    valid_signals = {"Yes","No"}
                    invalid = set(df["Entry Signal"].unique()) - valid_signals
                    if invalid:
                        df["Entry Signal"] = df["Entry Signal"].apply(
                            lambda x: x if x in valid_signals else "No"
                        )
                        df.to_csv(path, index=False)
                        logging.info(f"Corrected invalid 'Entry Signal' for {t}.")
            except Exception as e:
                logging.error(f"Error validating {path}: {e}")


def signal_handler(sig, frame):
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def validate_entry_signals(selected_stocks,last_trading_day,india_tz, validation_log_path='validation_results.txt'):
    """
    After entries are detected, we validate them by comparing expected signals from conditions
    vs actual 'Entry Signal' columns and correct if needed.
    """
    validation_logger=logging.getLogger('ValidationLogger')
    validation_logger.setLevel(logging.INFO)
    if not validation_logger.handlers:
        fh=logging.FileHandler(validation_log_path)
        fh.setLevel(logging.INFO)
        formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        validation_logger.addHandler(fh)

    for ticker in selected_stocks:
        path=os.path.join(INDICATORS_DIR,f"{ticker}_main_indicators.csv")
        if not os.path.exists(path):
            validation_logger.warning(f"No main indicators for {ticker}.")
            continue
        try:
            data=load_and_normalize_csv(path,timezone='Asia/Kolkata')
            if "Entry Signal" not in data.columns:
                data["Entry Signal"]="No"
            today_start_time=india_tz.localize(datetime.combine(last_trading_day,datetime_time(9,15)))
            today_end_time=india_tz.localize(datetime.combine(last_trading_day,datetime_time(15,30)))
            today_data=data[(data['date']>=today_start_time)&(data['date']<=today_end_time)]
            if today_data.empty:
                validation_logger.warning(f"No intraday data for {ticker} on {last_trading_day}.")
                continue

            expected_entries=apply_trend_conditions(ticker,last_trading_day,india_tz)
            comparison_data=data.copy()
            comparison_data["Entry Signal"]="No"

            # Mark expected entries in comparison_data
            for entry in expected_entries:
                et=pd.to_datetime(entry["date"], errors='coerce')
                if et is not None and not pd.isnull(et):
                    if et.tzinfo is None:
                        et=india_tz.localize(et)
                    else:
                        et=et.astimezone(india_tz)
                    comparison_data.loc[comparison_data['date']==et,"Entry Signal"]="Yes"

            actual_yes=set(data[data["Entry Signal"]=="Yes"]['date'])
            expected_yes=set(comparison_data[comparison_data["Entry Signal"]=="Yes"]['date'])

            missing_yes=expected_yes-actual_yes
            unexpected_yes=actual_yes-expected_yes

            if missing_yes:
                validation_logger.warning(f"{ticker}: Missing Yes for {missing_yes}")
            if unexpected_yes:
                validation_logger.warning(f"{ticker}: Unexpected Yes for {unexpected_yes}")

            if missing_yes or unexpected_yes:
                data["Entry Signal"]=comparison_data["Entry Signal"]
                data.to_csv(path,index=False)
                validation_logger.info(f"Corrected Entry Signals for {ticker}.")
            else:
                validation_logger.info(f"Entry Signals correct for {ticker}.")
        except Exception as e:
            validation_logger.error(f"Error validating {ticker}: {e}")
            validation_logger.error(traceback.format_exc())

def detect_signals_in_memory(ticker: str, df_today: pd.DataFrame) -> list:
    """
    Signal detection with relaxed thresholds for pullbacks while maintaining quality breakouts/breakdowns.
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
    if adx_val <= 22:  # Maintain ADX threshold for general momentum
        return signals_detected

    # Decide if it's Bullish or Bearish scenario:
    if (
        daily_change > 1.5
        and latest_row.get("RSI", 0) > 35
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.01
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.5
        and latest_row.get("RSI", 0) < 65
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.99
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # -- 3) Define "Pullback" & "Breakout/Breakdown" conditions
    volume_multiplier = 1.8
    rolling_window = 10
    MACD_BULLISH_OFFSET = 0.1
    MACD_BEARISH_OFFSET = -0.1

    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["low"] >= df_today["VWAP"] * 0.99)  # Loosened proximity to VWAP
                & (df_today["close"] > df_today["open"])
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] < 47)  # Slightly relaxed stochastic threshold
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 22)  # Reduced ADX threshold for pullbacks
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.995)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 60)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 42)
            ),
        }

    else:  # Bearish
        conditions = {
            "Pullback": (
                (df_today["high"] <= df_today["VWAP"] * 1.01)  # Loosened proximity to VWAP
                & (df_today["close"] < df_today["open"])
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] > 53)  # Slightly relaxed stochastic threshold
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 22)  # Reduced ADX threshold for pullbacks
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.005)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 40)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 42)
            ),
        }

    # -- 4) Collect rows matching conditions
    for entry_type, cond in conditions.items():
        selected = df_today.loc[cond]
        if not selected.empty:
            for _, row in selected.iterrows():
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
                    "Signal_ID": row.get("Signal_ID", ""),
                    "logtime": row.get("logtime", ""),
                    "Entry Signal": "Yes"
                })

    return signals_detected

def find_price_action_entry(last_trading_day):
    """
    Reads each *_main_indicators.csv for today's data (9:15 - 15:30),
    calls detect_signals_in_memory(ticker, df_today) to gather signals,
    and combines them into a DataFrame of 'Entry Signal' rows.
    
    Returns a DataFrame of signals with columns like:
       [Ticker, date, Entry Type, Trend Type, Price, Daily Change %, VWAP, ADX, Signal_ID, logtime, ...]
    """

    india_tz = pytz.timezone('Asia/Kolkata')
    # Define today's start/end for slicing
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    csv_files = glob.glob(pattern)

    all_signals = []
    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','')
        df = load_and_normalize_csv(file_path, timezone='Asia/Kolkata')
        if df.empty:
            continue

        # slice data for today
        df_today = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()
        if df_today.empty:
            continue

        # detect signals in memory
        signals_list = apply_trend_conditions(ticker, last_trading_day, india_tz)
        if signals_list:
            all_signals.extend(signals_list)

    # Convert all signals to DataFrame
    if not all_signals:
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    # Sort by date, reset index
    signals_df = normalize_time(signals_df, timezone='Asia/Kolkata')
    signals_df = signals_df.sort_values(by='date').reset_index(drop=True)
    return signals_df

def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads the full-day 'price_action_entries' from input_file,
    ensures only the *first* (earliest) intraday entry of each Ticker
    is kept, calculates target price & quantity, then merges
    into 'papertrade_{YYYY-MM-DD}.csv' (no duplicates).
    """

    india_tz = pytz.timezone('Asia/Kolkata')
    if not os.path.exists(input_file):
        print(f"No entries file {input_file}, skipping papertrade creation.")
        return

    df_entries = pd.read_csv(input_file)
    if 'date' not in df_entries.columns:
        logging.error(f"No 'date' column in {input_file}")
        raise KeyError("No 'date' in entries file.")

    # Convert 'date' to Asia/Kolkata time
    df_entries = normalize_time(df_entries, timezone='Asia/Kolkata')
    df_entries.sort_values('date', inplace=True)

    # Filter strictly to today's (9:25–15:50)
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,25)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14,50)))
    todays_entries = df_entries[(df_entries['date'] >= start_time) & (df_entries['date'] <= end_time)].copy()

    if todays_entries.empty:
        print(f"No valid entries found for {last_trading_day}.")
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
        "Ticker","date","Entry Type","Trend Type","Price",
        "Daily Change %","VWAP","ADX","Signal_ID","logtime"
    ]
    for col in required_cols:
        if col not in todays_entries.columns:
            todays_entries[col] = ""

    # Separate Bullish vs Bearish to assign targets
    if "Trend Type" in todays_entries.columns:
        bullish = todays_entries[todays_entries["Trend Type"]=="Bullish"].copy()
        bearish = todays_entries[todays_entries["Trend Type"]=="Bearish"].copy()

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
        combined_entries['logtime'] = datetime.now(india_tz).isoformat()

    # Merge into existing papertrade CSV
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        existing = normalize_time(existing, timezone='Asia/Kolkata')
        existing.sort_values('date', inplace=True)

        # Combine, drop duplicates on [Ticker, date]
        all_paper = pd.concat([existing, combined_entries], ignore_index=True)
        all_paper.drop_duplicates(subset=['Ticker','date'], keep='last', inplace=True)
        all_paper.sort_values('date', inplace=True)
        all_paper.to_csv(output_file, index=False)
    else:
        combined_entries.to_csv(output_file, index=False)

    print(f"Papertrade data (only earliest Ticker entries) saved to '{output_file}'")

    # Debug output
    try:
        papertrade_data = pd.read_csv(output_file)
        print("\nPapertrade Data for Today (Earliest Ticker entries):")
        print(papertrade_data)
    except Exception as e:
        logging.error(f"Error reading {output_file}: {e}")
        print(f"Error reading {output_file}: {e}")

def run():
    """
    The main run function:
    - Determine last trading day
    - Find price action entries
    - Analyze latest interval for signals
    - Save entries to a file
    - Create papertrade file if applicable
    - Validate signals
    """
    last_trading_day=get_last_trading_day()
    price_action_entries=find_price_action_entry(last_trading_day)
    analyze_latest_interval()
    today_str=last_trading_day.strftime('%Y-%m-%d')
    entries_filename=f"price_action_entries_15min_{today_str}.csv"
    papertrade_filename=f"papertrade_{today_str}.csv"

    if not price_action_entries.empty:
        if 'date' not in price_action_entries.columns:
            return
        price_action_entries=normalize_time(price_action_entries)
        if os.path.exists(entries_filename):
            existing=load_and_normalize_csv(entries_filename)
            combined=pd.concat([existing,price_action_entries],ignore_index=True)
            combined=combined.drop_duplicates(subset=['Ticker','date'],keep='last')
            combined=combined.sort_values('date').reset_index(drop=True)
            combined.to_csv(entries_filename,index=False)
        else:
            price_action_entries.to_csv(entries_filename,index=False)
        print(f"Entries detected and saved to '{entries_filename}'.")
        print("\nPrice Action Entries for Today:")
        print(price_action_entries)
        create_papertrade_file(entries_filename,papertrade_filename,last_trading_day)
        print(f"\nCombined papertrade data saved to '{papertrade_filename}'.")

        try:
            p_data=pd.read_csv(papertrade_filename)
            print("\nPapertrade Data for Today:")
            print(p_data)
        except Exception as e:
            logging.error(f"Error reading {papertrade_filename}: {e}")
            print(f"Error reading {papertrade_filename}: {e}")

        logging.info("Validating combined CSV files for 'Entry Signal'")
        validate_entry_signals(selected_stocks, last_trading_day, india_tz)
        print("Validation of combined CSV files completed.")
    else:
        logging.info(f"No entries detected for {last_trading_day}.")
        print(f"No entries detected for {last_trading_day}.")


def job_run_twice():
    """
    Job scheduled to run twice within each 15-minute block:
    once at the start, once after a delay to let the new candle finalize.
    """
    try:
        run()
        logging.info("Executed trading strategy (First Run).")
        print("Executed trading strategy (First Run).")

        time.sleep(60)

        run()
        logging.info("Executed trading strategy (Second Run).")
        print("Executed trading strategy (Second Run).")

    except Exception as e:
        logging.error(f"Error in job_run_twice: {e}")
        print(f"Error in job_run_twice: {e}")


def job_validate_twice():
    """
    Job scheduled to validate entry signals twice after end-of-day, separated by a 15 second pause.
    """
    try:
        last_trading_day = get_last_trading_day()
        validate_entry_signals(selected_stocks, last_trading_day, india_tz)
        print("Entry signal validation executed successfully (First Validation).")
        time.sleep(15)
        validate_entry_signals(selected_stocks, last_trading_day, india_tz)
        print("Entry signal validation executed successfully (Second Validation).")
    except Exception as e:
        logging.error(f"Error in job_validate_twice: {e}")


def main():
    """
    Main function sets up a schedule:
    - Every 15 mins during trading window, run 'job_run_twice'.
    - After trading ends, run 'job_validate_twice'.
    - If starting during trading hours, run immediately.
    - Continuously checks schedule.
    """
    now = datetime.now(india_tz)
    start_time_naive = datetime.combine(now.date(), datetime_time(9,15,0))
    end_time_naive   = datetime.combine(now.date(), datetime_time(23,59,5))
    start_time = india_tz.localize(start_time_naive)
    end_time   = india_tz.localize(end_time_naive)

    if now > end_time:
        logging.warning("Current time past trading window, exiting.")
        print("Past trading window, exiting.")
        return
    elif now > start_time:
        logging.info("Current time within trading window.")
    else:
        logging.info("Current time before trading window.")

    scheduled_times = []
    current_time = start_time
    while current_time <= end_time:
        scheduled_times.append(current_time.strftime("%H:%M:%S"))
        current_time += timedelta(minutes=15)

    for st in scheduled_times:
        scheduled_dt_naive = datetime.strptime(st, "%H:%M:%S")
        scheduled_dt = india_tz.localize(datetime.combine(now.date(), scheduled_dt_naive.time()))
        if scheduled_dt > now:
            schedule.every().day.at(st).do(job_run_twice)
            logging.info(f"Scheduled run job at {st} IST.")
        else:
            logging.debug(f"Skipped scheduling {st}, time passed.")

    schedule_time = (end_time + timedelta(seconds=5)).strftime("%H:%M:%S")
    schedule.every().day.at(schedule_time).do(job_validate_twice)
    logging.info(f"Scheduled validation job at {schedule_time} IST.")

    if start_time <= now <= end_time:
        logging.info("Executing initial run of trading strategy twice.")
        job_run_twice()

    validate_and_correct_main_indicators(selected_stocks, india_tz)

    while True:
        schedule.run_pending()
        check_missing_signals(data_dir=INDICATORS_DIR, tz='Asia/Kolkata')
        time.sleep(1)


def main1():
    """
    This function checks CSV intervals for a given directory and prints any tickers missing the latest interval.
    Useful for debugging and confirming data arrival.
    """
    parser = argparse.ArgumentParser(description="Check CSV intervals")
    parser.add_argument("--data_dir", type=str, default="main_indicators", help="Directory containing main indicators")
    parser.add_argument("--timezone", type=str, default="Asia/Kolkata", help="Timezone")
    args = parser.parse_args()
    current_time = datetime.now()
    results = check_csv_intervals(data_dir=args.data_dir, current_time=current_time, tz=args.timezone)
    for t, e in results.items():
        if not e:
            print(t)


if __name__ == "__main__":
    validate_and_correct_main_indicators(selected_stocks, india_tz)
    main()
    main1()
