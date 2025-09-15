# -*- coding: utf-8 -*-
"""
Script to run every 15 minutes from 9:30:15 AM to 3:30:15 PM,
fetch intraday (5min) data + some historical data for continuity,   # CHANGED: 15min -> 5min in doc
calculate a suite of technical indicators (including ATR),
and append a new row with all indicators to a CSV.

Indicators included:
- RSI
- ATR (Average True Range)
- MACD (with MACD_Signal & MACD_Hist)                     # CHANGED: naming
- Bollinger Bands (upper & lower)
- Stochastic (Stoch_%K, Stoch_%D)                          # CHANGED: two outputs
- ADX
- VWAP
- 20-SMA
- Recent High/Low
- EMA_50, EMA_200
- CCI
- MFI
- OBV
- Parabolic SAR
- Ichimoku (Tenkan_Sen, Kijun_Sen, Senkou_Span_A, Senkou_Span_B, Chikou_Span)
- Williams %R
- Rate of Change (ROC)
- Bollinger Band Width (BB_Width)
- Daily_Change
- Intra_Change                                             # CHANGED: newly added
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
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for all date/time operations

# Load your own stock list here
# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks_MIS import selected_stocks

CACHE_DIR = "data_cache_5min"
INDICATORS_DIR = "main_indicators_5min"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

api_semaphore = threading.Semaphore(2)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Set base logging level

file_handler = logging.FileHandler("logs\\trading_data_dc.log")
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Avoid adding handlers repeatedly if re-run
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging initialized successfully.")
print("Logging initialized and script execution started.")

# Change directory if needed
cwd = r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info(f"Changed working directory to {cwd}.")
except FileNotFoundError:
    logger.error(f"Working directory {cwd} not found.")
    raise
except Exception as e:
    logger.error(f"Unexpected error changing directory: {e}")
    raise

# Market holiday list (example set)
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
    """Custom Exception for repeated API call failures."""
    pass

# ================================
# Kite Session & Instruments
# ================================
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
    Refresh the instrument list from the Kite API (NSE).
    """
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        shares_tokens_updated = dict(zip(instrument_df['tradingsymbol'], instrument_df['instrument_token']))
        logging.info("Instrument list refreshed successfully.")
        return shares_tokens_updated
    except Exception as e:
        logging.error(f"Error refreshing instrument list: {e}")
        return shares_tokens  # fallback if refresh fails

def get_tokens_for_stocks(stocks):
    """
    Fetch instrument tokens for given `stocks`.
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
    Fetch tokens with limited retries.
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

# Attempt to refresh instrument list, else fetch tokens
try:
    shares_tokens = refresh_instrument_list()
except:
    shares_tokens = {}
if not shares_tokens:
    shares_tokens = get_tokens_for_stocks_with_retry(selected_stocks)

# ================================
# Data Caching Utilities
# ================================
def cache_file_path(ticker, data_type):
    """
    Returns something like: data_cache/TICKER_intraday.csv
    """
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")

def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=5)):
    """
    Check if a given cache file has been modified within `freshness_threshold`.
    """
    if not os.path.exists(cache_file):
        return False
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)
    current_time = datetime.now(pytz.UTC)
    return (current_time - cache_mtime) < freshness_threshold

def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Convert 'date' column to a timezone-aware datetime. Returns an empty DataFrame if 'date' missing.
    """
    df = df.copy()
    if 'date' not in df.columns:
        logging.error("DataFrame missing 'date' column. Returning empty DataFrame.")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("The 'date' column could not be converted to datetime.")
    # Assign or convert to the specified timezone
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(timezone)
    return df

def cache_update(ticker, data_type, fetch_function,
                 freshness_threshold=None,
                 bypass_cache=False):
    """
    General caching mechanism for any 'data_type' (e.g., 'intraday', 'historical'):
      - If cache is fresh (and not bypassed), load it.
      - Else, fetch new data, merge with existing cache, and save.
    """
    cache_path = cache_file_path(ticker, data_type)
    if freshness_threshold is None:
        default_thresholds = {
            "historical": timedelta(hours=8),
            "intraday"  : timedelta(minutes=5)   # CHANGED: 15 -> 5 to reflect 5-min candles
        }
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=5))

    # If we trust the cache, load it
    if (not bypass_cache) and is_cache_fresh(cache_path, freshness_threshold):
        try:
            logging.info(f"Loading cached {data_type} data for {ticker}.")
            if os.path.getsize(cache_path) > 0:
                existing = pd.read_csv(cache_path)
                if 'date' not in existing.columns:
                    logging.warning(f"Cache file missing 'date' column for {ticker}; removing.")
                    os.remove(cache_path)
                    raise ValueError("Missing date column in cache file")
                existing['date'] = pd.to_datetime(existing['date'])
                existing = normalize_time(existing)
                return existing
            else:
                raise ValueError("Cache file is empty")
        except Exception as e:
            logging.warning(f"Cache load failed for {ticker}: {e}. Fetching fresh data...")

    # Either bypassed or stale cache
    logging.info(f"Fetching fresh {data_type} data for {ticker}.")
    fresh_data = fetch_function(ticker)
    if fresh_data is None or fresh_data.empty:
        logging.warning(f"No fresh {data_type} data for {ticker}. Returning empty.")
        return pd.DataFrame()

    # Merge with existing if it exists
    if os.path.exists(cache_path):
        try:
            existing = pd.read_csv(cache_path)
            existing = normalize_time(existing)
        except Exception as e:
            logging.warning(f"Existing cache load failed for {ticker}: {e}. Overwriting.")
            existing = pd.DataFrame()

        combined = pd.concat([existing, fresh_data]).drop_duplicates(subset='date').sort_values('date')
        combined.to_csv(cache_path, index=False)
        return combined
    else:
        fresh_data.to_csv(cache_path, index=False)
        return fresh_data

# ================================
# API Calls & Indicator Functions
# ================================
def make_api_call_with_retry(kite, token, from_date, to_date, interval,
                             max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Attempt a KiteConnect historical_data call with retries/backoff.
    """
    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            logging.info(f"API call attempt {attempt+1} for token={token}, interval={interval}")
            data = kite.historical_data(token, from_date, to_date, interval)
            if not data:
                logging.warning(f"No data from API for token {token}, retrying...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
                continue
            logging.info(f"Data fetched for token={token}, attempt {attempt+1}.")
            return pd.DataFrame(data)
        except NetworkException as net_exc:
            # Rate-limit or network failures
            if 'too many requests' in str(net_exc).lower():
                logging.warning(f"Rate limit hit; retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                raise
        except (KiteException, HTTPError) as e:
            if 'too many requests' in str(e).lower():
                logging.warning(f"Rate limit exceeded; retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                raise
        except Exception as e:
            logging.error(f"Unexpected error for token={token}: {e}")
            time.sleep(delay)
            attempt += 1
            delay *= backoff_factor

    raise APICallError(f"Failed to fetch data for token={token} after {max_retries} retries.")

# ---- Core Indicator Calculations ----
def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """
    RSI calculation using average gains/losses.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(timeperiod).mean()
    roll_down = down.rolling(timeperiod).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0/(1.0 + rs))
    return rsi

def calculate_atr(data: pd.DataFrame, timeperiod: int = 14) -> pd.Series:
    """
    ATR = Smoothed moving average of True Range over `timeperiod`.
    True Range = max( high-low, abs(high-previous_close), abs(low-previous_close) ).
    """
    d = data.copy()
    d['previous_close'] = d['close'].shift(1)
    d['high_low'] = d['high'] - d['low']
    d['high_pc'] = (d['high'] - d['previous_close']).abs()
    d['low_pc'] = (d['low'] - d['previous_close']).abs()
    tr = d[['high_low', 'high_pc', 'low_pc']].max(axis=1)
    # Simple rolling average used (Wilder's smoothing can also be used)
    atr = tr.rolling(window=timeperiod, min_periods=1).mean()
    return atr

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    MACD on close series. Returns DataFrame with columns:
    MACD, MACD_Signal, MACD_Hist                               # CHANGED: names
    """
    close = data['close'] if isinstance(data, pd.DataFrame) else data  # CHANGED
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({'MACD': macd, 'MACD_Signal': sig, 'MACD_Hist': hist})  # CHANGED

def calculate_bollinger_bands(data: pd.DataFrame, period=20, up=2, dn=2):
    sma = data['close'].rolling(period, min_periods=period).mean()              # CHANGED: min_periods=period
    std = data['close'].rolling(period, min_periods=period).std()               # CHANGED
    upper = sma + (std * up)
    lower = sma - (std * dn)
    return pd.DataFrame({'Upper_Band': upper, 'Lower_Band': lower})

def calculate_stochastic(data: pd.DataFrame, k_period=14, k_smooth=3, d_period=3):
    """
    Slow stochastic aligned to reference:
    returns Stoch_%K (smoothed) and Stoch_%D                          # CHANGED: new behavior + names
    """
    low_min  = data['low'].rolling(k_period, min_periods=k_period).min()
    high_max = data['high'].rolling(k_period, min_periods=k_period).max()
    rng = (high_max - low_min)
    k_fast = pd.Series(0.0, index=data.index)
    valid = rng > 0
    k_fast.loc[valid] = 100.0 * (data['close'].loc[valid] - low_min.loc[valid]) / rng.loc[valid]
    k_fast = k_fast.clip(0.0, 100.0)
    # slow %K and %D
    k_slow = k_fast.rolling(k_smooth, min_periods=k_smooth).mean().clip(0.0, 100.0)
    d = k_slow.rolling(d_period, min_periods=d_period).mean().clip(0.0, 100.0)
    return pd.DataFrame({'Stoch_%K': k_slow, 'Stoch_%D': d})

def calculate_adx(data: pd.DataFrame, period=14):
    """
    Classic ADX (Wilder's DMI).
    """
    d = data.copy()
    if not {'high','low','close'}.issubset(d.columns):
        raise ValueError("Missing columns for ADX calculation")

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

    # Wilder smoothing method
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

# ---- Additional Indicator Functions ----
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return cci

def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    tp_change = typical_price.diff()
    pos_mf = money_flow.where(tp_change > 0, 0)
    neg_mf = money_flow.where(tp_change < 0, 0)
    pos_mf_sum = pos_mf.rolling(window=period).sum()
    neg_mf_sum = neg_mf.rolling(window=period).sum().abs()
    mfi = 100 - (100 / (1 + pos_mf_sum / (neg_mf_sum + 1e-10)))
    return mfi

def calculate_obv(data: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i-1]:
            obv.append(obv[-1] + data['volume'].iloc[i])
        elif data['close'].iloc[i] < data['close'].iloc[i-1]:
            obv.append(obv[-1] - data['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=data.index)

def calculate_parabolic_sar(data: pd.DataFrame, initial_af=0.02, step_af=0.02, max_af=0.2) -> pd.Series:
    length = len(data)
    psar = data['low'].iloc[0]
    ep = data['high'].iloc[0]
    af = initial_af
    trend = 1  # 1 for uptrend, -1 for downtrend
    sar_list = [psar]

    for i in range(1, length):
        prev_psar = psar
        if trend == 1:
            psar = prev_psar + af * (ep - prev_psar)
            psar = min(psar, data['low'].iloc[i-1], data['low'].iloc[i])
            if data['low'].iloc[i] < psar:
                trend = -1
                psar = ep
                ep = data['low'].iloc[i]
                af = initial_af
        else:
            psar = prev_psar + af * (ep - prev_psar)
            psar = max(psar, data['high'].iloc[i-1], data['high'].iloc[i])
            if data['high'].iloc[i] > psar:
                trend = 1
                psar = ep
                ep = data['high'].iloc[i]
                af = initial_af

        if trend == 1:
            if data['high'].iloc[i] > ep:
                ep = data['high'].iloc[i]
                af = min(af + step_af, max_af)
        else:
            if data['low'].iloc[i] < ep:
                ep = data['low'].iloc[i]
                af = min(af + step_af, max_af)

        sar_list.append(psar)

    return pd.Series(sar_list, index=data.index)

def calculate_ichimoku(data: pd.DataFrame, conversion_line_period=9, base_line_period=26,
                       leading_span_b_period=52, displacement=26):
    high = data['high']
    low = data['low']

    tenkan_sen = (high.rolling(window=conversion_line_period).max() +
                  low.rolling(window=conversion_line_period).min()) / 2
    kijun_sen = (high.rolling(window=base_line_period).max() +
                 low.rolling(window=base_line_period).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = ((high.rolling(window=leading_span_b_period).max() +
                      low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)
    chikou_span = data['close'].shift(-displacement)

    return pd.DataFrame({
        'Tenkan_Sen': tenkan_sen,
        'Kijun_Sen': kijun_sen,
        'Senkou_Span_A': senkou_span_a,
        'Senkou_Span_B': senkou_span_b,
        'Chikou_Span': chikou_span
    })

def calculate_williams_r(data: pd.DataFrame, period=14) -> pd.Series:
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()
    willr = -100 * (highest_high - data['close']) / (highest_high - lowest_low + 1e-10)
    return willr

def calculate_roc(data: pd.DataFrame, period=12) -> pd.Series:
    return ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100

def calculate_bb_width(data: pd.DataFrame, period=20, up=2, dn=2) -> pd.Series:
    sma = data['close'].rolling(period, min_periods=1).mean()
    std = data['close'].rolling(period, min_periods=1).std()
    upper = sma + (std * up)
    lower = sma - (std * dn)
    bb_width = upper - lower
    return bb_width

# ---- Recompute Indicators for the Latest Row ----
def recalc_indicators_for_latest_row(data: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate all indicators on the last ~100 rows for performance efficiency
    and return ONLY the final row with updated columns.
    """
    if data.empty:
        return pd.DataFrame()

    df = data.copy().reset_index(drop=True)
    last_100 = df.tail(100).copy()

    try:
        # RSI / ATR
        df.loc[last_100.index, 'RSI'] = calculate_rsi(last_100['close'])
        df.loc[last_100.index, 'ATR'] = calculate_atr(last_100)

        # MACD (MACD, MACD_Signal, MACD_Hist)                         # CHANGED: names
        macd_df = calculate_macd(last_100['close'])
        df.loc[last_100.index, ['MACD','MACD_Signal','MACD_Hist']] = macd_df

        # Bollinger Bands with min_periods=20                          # CHANGED
        boll_df = calculate_bollinger_bands(last_100)
        df.loc[last_100.index, ['Upper_Band','Lower_Band']] = boll_df

        # Stochastic slow: Stoch_%K / Stoch_%D                         # CHANGED
        stoch_df = calculate_stochastic(last_100)
        df.loc[last_100.index, ['Stoch_%K','Stoch_%D']] = stoch_df

        # VWAP: cumulative on full df, then mapped to last_100         # CHANGED
        vwap_full = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df.loc[last_100.index, 'VWAP'] = vwap_full.loc[last_100.index]

        # 20_SMA with min_periods=20                                   # CHANGED
        df.loc[last_100.index, '20_SMA'] = last_100['close'].rolling(window=20, min_periods=20).mean()

        # Recent High/Low with min_periods=5                            # CHANGED
        df.loc[last_100.index, 'Recent_High'] = last_100['high'].rolling(window=5, min_periods=5).max()
        df.loc[last_100.index, 'Recent_Low']  = last_100['low'].rolling(window=5, min_periods=5).min()

        # ADX
        adx_series = calculate_adx(last_100, period=14)
        df.loc[last_100.index, 'ADX'] = adx_series

        # Remaining indicators (unchanged)
        df.loc[last_100.index, 'EMA_50'] = calculate_ema(last_100['close'], span=50)
        df.loc[last_100.index, 'EMA_200'] = calculate_ema(last_100['close'], span=200)
        df.loc[last_100.index, 'CCI'] = calculate_cci(last_100)
        df.loc[last_100.index, 'MFI'] = calculate_mfi(last_100)
        df.loc[last_100.index, 'OBV'] = calculate_obv(last_100)
        df.loc[last_100.index, 'Parabolic_SAR'] = calculate_parabolic_sar(last_100)
        ichimoku_df = calculate_ichimoku(last_100)
        for col in ichimoku_df.columns:
            df.loc[last_100.index, col] = ichimoku_df[col]
        df.loc[last_100.index, 'Williams_%R'] = calculate_williams_r(last_100)
        df.loc[last_100.index, 'ROC'] = calculate_roc(last_100)
        df.loc[last_100.index, 'BB_Width'] = calculate_bb_width(last_100)
    except Exception as e:
        logging.error(f"Indicator calculation error: {e}")
        raise

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Return the final row with updated columns
    latest_row = df.iloc[[-1]].copy()
    return latest_row

def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

# ================================
# Fetch Intraday + Historical
# ================================
def fetch_intraday_data_with_historical(ticker, token, from_date_intra, to_date_intra, interval="5minute"):  # CHANGED default
    """
    Fetches today's intraday data plus ~10 days of historical data for continuity,
    merges them, recalculates all indicators, returns the newest row only.
    Also calculates 'Daily_Change' vs. last candle of the previous trading day,
    and 'Intra_Change' vs previous 5-min bar (same day).                       # CHANGED: mention Intra_Change
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

    # Use caching for intraday and historical
    intraday_data = cache_update(ticker, "intraday", fetch_intraday, bypass_cache=False)
    historical_data = cache_update(ticker, "historical", fetch_historical, bypass_cache=False)

    combined = pd.concat([historical_data, intraday_data], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    combined.drop_duplicates(subset=['date'], inplace=True)
    combined.sort_values('date', inplace=True)

    # Recompute indicators and keep only latest row
    latest_row = recalc_indicators_for_latest_row(combined)
    if latest_row.empty:
        return pd.DataFrame()

    # Daily_Change (vs previous trading day's last bar)
    try:
        combined_before_today = combined[combined['date'] < from_date_intra]
        if not combined_before_today.empty:
            prev_close = combined_before_today.iloc[-1]['close']
            latest_close = latest_row.iloc[0]['close']
            daily_change = ((latest_close - prev_close) / prev_close) * 100
            latest_row['Daily_Change'] = daily_change
        else:
            latest_row['Daily_Change'] = np.nan
    except Exception as e:
        logging.error(f"[{ticker}] Error computing Daily_Change: {e}")
        latest_row['Daily_Change'] = np.nan

    # Intra_Change (vs previous 5-min bar, same day)                     # CHANGED: new calc
    try:
        if len(combined) >= 2:
            prev_row = combined.iloc[-2]
            latest_dt = latest_row.iloc[0]['date']
            prev_dt = prev_row['date']
            # normalize to IST date for same-day check
            latest_day = pd.to_datetime(latest_dt).tz_convert(india_tz).date() if hasattr(latest_dt, 'tz_convert') else pd.to_datetime(latest_dt).date()
            prev_day   = pd.to_datetime(prev_dt).tz_convert(india_tz).date()   if hasattr(prev_dt, 'tz_convert') else pd.to_datetime(prev_dt).date()
            if latest_day == prev_day and prev_row['close'] != 0:
                latest_row['Intra_Change'] = ((latest_row.iloc[0]['close'] - prev_row['close']) / prev_row['close']) * 100.0
            else:
                latest_row['Intra_Change'] = np.nan
        else:
            latest_row['Intra_Change'] = np.nan
    except Exception as e:
        logging.error(f"[{ticker}] Error computing Intra_Change: {e}")
        latest_row['Intra_Change'] = np.nan

    # Ensure minimal columns
    if 'Entry Signal' not in latest_row.columns:
        latest_row['Entry Signal'] = 'No'
    if 'logtime' not in latest_row.columns:
        latest_row['logtime'] = datetime.now(india_tz).isoformat()

    # Create or fill Signal_ID
    if 'Signal_ID' not in latest_row.columns:
        latest_row['Signal_ID'] = latest_row.apply(lambda r: build_signal_id(ticker, r), axis=1)
    else:
        mask_missing = latest_row['Signal_ID'].isna() | (latest_row['Signal_ID'] == "")
        latest_row.loc[mask_missing, 'Signal_ID'] = latest_row.loc[mask_missing].apply(
            lambda r: build_signal_id(ticker, r), axis=1
        )

    return latest_row

# ================================
# Main Ticker Processing
# ================================
def process_ticker(ticker, from_date_intra, to_date_intra, last_trading_day):
    """
    Fetch the new candle + indicators, append row to {ticker}_main_indicators.csv.
    """
    try:
        with api_semaphore:
            token = shares_tokens.get(ticker)
            if not token:
                logging.warning(f"[{ticker}] No token found.")
                return None, []

            latest_row = fetch_intraday_data_with_historical(
                ticker, token, from_date_intra, to_date_intra, interval="5minute"   # CHANGED: 15minute -> 5minute
            )
            if latest_row.empty:
                logging.warning(f"[{ticker}] No new row returned.")
                return None, []

        main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        if os.path.exists(main_path):
            existing = pd.read_csv(main_path, parse_dates=['date'], keep_default_na=False)
        else:
            existing = pd.DataFrame()

        # Normalize the date
        latest_row = normalize_time(latest_row, timezone='Asia/Kolkata')
        if latest_row.empty or 'Signal_ID' not in latest_row.columns:
            logging.error(f"[{ticker}] Missing Signal_ID or empty row.")
            return None, []

        # Check for duplicates by Signal_ID
        if not existing.empty and 'Signal_ID' in existing.columns:
            all_ids = set(existing['Signal_ID'].astype(str).unique())
            new_id = str(latest_row.iloc[0]['Signal_ID'])
            if new_id in all_ids:
                logging.info(f"[{ticker}] Signal_ID={new_id} already exists. Skipping.")
                return None, []

        # Force "Entry Signal" to "No"
        latest_row.at[latest_row.index[0], 'Entry Signal'] = 'No'
        latest_row['logtime'] = datetime.now(india_tz).isoformat()

        combined = pd.concat([existing, latest_row], ignore_index=True)
        combined.drop_duplicates(subset=['date'], keep='first', inplace=True)
        combined.sort_values('date', inplace=True)
        combined.to_csv(main_path, index=False)
        logging.info(f"[{ticker}] Appended new row (date={latest_row.iloc[0]['date']}) to {main_path}.")

        return latest_row, []
    except Exception as e:
        logging.error(f"[{ticker}] Error processing: {e}")
        logging.error(traceback.format_exc())
        return None, []

# ================================
# Utility & Validation Functions
# ================================
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
        df = df.reset_index(drop=True)

    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[expected_cols]
    return df

def validate_and_correct_main_indicators(selected_stocks):
    """
    Ensures each {ticker}_main_indicators.csv has 'Entry Signal' column (defaults 'No').
    """
    for ticker in selected_stocks:
        path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        if not os.path.exists(path):
            logging.warning(f"No main indicators CSV for {ticker}.")
            continue
        try:
            df = load_and_normalize_csv(path, timezone='Asia/Kolkata')
            if "Entry Signal" not in df.columns:
                df["Entry Signal"] = "No"
                df.to_csv(path, index=False)
                logging.info(f"[{ticker}] Added 'Entry Signal' column.")
            else:
                valid_signals = {"Yes", "No"}
                df["Entry Signal"] = df["Entry Signal"].apply(lambda x: x if x in valid_signals else "No")
                df.to_csv(path, index=False)
        except Exception as e:
            logging.error(f"[{ticker}] Error validating {path}: {e}")

def get_last_trading_day(reference_date=None):
    """
    Return the most recent trading day skipping weekends & known holidays.
    """
    if reference_date is None:
        reference_date = datetime.now(india_tz).date()
    logging.info(f"Checking last trading day from {reference_date}")
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

def prepare_time_ranges(last_trading_day):
    """
    Prepare the from-to times for intraday portion (9:15 to 15:30).
    Adjust if the day is 'today' and it’s not yet 15:30.
    """
    from_date_intra = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 15)))
    to_date_intra = india_tz.localize(datetime.combine(last_trading_day, datetime_time(15, 30)))
    now_ist = datetime.now(india_tz)

    if now_ist.date() == last_trading_day and now_ist.time() < datetime_time(15, 30):
        to_date_intra = now_ist
    return from_date_intra, to_date_intra

# ================================
# Main Single-Pass & Periodic Runs
# ================================
def run():
    """
    One pass:
    - Determine last trading day
    - For each ticker, fetch intraday/historical data (with caching),
      recalc indicators, append new row to CSV.
    """
    last_td = get_last_trading_day()
    from_date_intra, to_date_intra = prepare_time_ranges(last_td)

    def wrapper(tkr):
        return process_ticker(tkr, from_date_intra, to_date_intra, last_td)

    processed_data_dict = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_ticker = {executor.submit(wrapper, t): t for t in selected_stocks}
            for fut in as_completed(future_to_ticker):
                ticker = future_to_ticker[fut]
                try:
                    res = fut.result()
                    if res is not None:
                        data, _ = res
                        if data is not None:
                            processed_data_dict[ticker] = data
                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")
    except Exception as e:
        logging.error(f"Error in parallel processing: {e}")

    if processed_data_dict:
        print("Data fetching and indicator updates done for tickers.")
    else:
        print("No new data fetched for any ticker.")

def initialize_calc_flag_in_main_indicators(indicators_dir=INDICATORS_DIR):
    """
    Example of adding a 'CalcFlag' column if missing, set default 'No'.
    """
    pattern = os.path.join(indicators_dir, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No main_indicator CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            if "CalcFlag" not in df.columns:
                df["CalcFlag"] = "No"
                logging.info(f"Added 'CalcFlag' to {file_path}.")
            else:
                df["CalcFlag"] = df["CalcFlag"].fillna("")
                df.loc[df["CalcFlag"].eq(""), "CalcFlag"] = "No"
            df.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Error adding CalcFlag to {file_path}: {e}")
            logging.error(traceback.format_exc())

def signal_handler(sig, frame):
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """
    Example function that runs multiple times with short pauses.
    """
    initialize_calc_flag_in_main_indicators()
    validate_and_correct_main_indicators(selected_stocks)
    run()
    print("1st iteration is complete")
    time.sleep(10)

    validate_and_correct_main_indicators(selected_stocks)
    run()
    print("2nd iteration is complete")
    time.sleep(10)

    validate_and_correct_main_indicators(selected_stocks)
    run()
    print("3rd iteration is complete")

def run_periodically():
    """
    Runs in a 15-minute loop from 9:30:15 to 15:30:15 local time.
    (Loop unchanged by request; now fetching 5-min bars each run.)       # NOTE
    """
    today = datetime.now(india_tz).date()
    start_dt = india_tz.localize(datetime.combine(today, datetime_time(9, 15, 15)))
    end_dt   = india_tz.localize(datetime.combine(today, datetime_time(15, 30, 15)))

    next_run = max(datetime.now(india_tz), start_dt)
    while True:
        now = datetime.now(india_tz)
        if now > end_dt:
            print("Reached 15:30:15; stopping periodic runs.")
            break

        if now < next_run:
            time.sleep((next_run - now).total_seconds())

        main()
        next_run += timedelta(minutes=5)

if __name__ == "__main__":
    # Uncomment whichever approach you prefer:
    # run_periodically()  # to schedule repeated runs
    # main()              # or do a single-run demonstration
    run_periodically()
