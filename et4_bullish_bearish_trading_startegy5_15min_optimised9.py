# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:49:07 2024

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
This script automates fetching intraday stock data, updating technical indicators for only the newest rows,
detecting any new entry signals (one-time), and appending new data/entries to CSV files (main_indicators, papertrade, etc.)
without modifying old data.
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

# ==============================
# Global Configuration
# ==============================
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for all date operations

from et4_filtered_stocks_market_cap import selected_stocks  # List of ticker symbols

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

api_semaphore = threading.Semaphore(2)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

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

# Market holidays in 2024
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
    """Custom Exception for repeated API failures."""
    pass


# ==============================
# Basic Helpers (Normalization)
# ==============================
def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Convert 'date' column to datetime with the specified timezone. 
    If no timezone is detected, assume UTC -> convert to Asia/Kolkata.
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("The 'date' column could not be converted to datetime.")

    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')

    df['date'] = df['date'].dt.tz_convert(timezone)
    return df.sort_values('date').reset_index(drop=True)


def load_and_normalize_csv(filepath, expected_cols=None, tz='Asia/Kolkata'):
    """
    Loads a CSV, normalizes the 'date' column, and ensures expected_cols exist.
    Returns a sorted DataFrame (by 'date').
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = normalize_time(df, tz)
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

    return df.sort_values('date').reset_index(drop=True)


def write_csv_safely(df, filepath, expected_cols):
    """
    Writes a DataFrame to CSV ensuring columns match expected_cols and quoting is consistent.
    """
    df = df.copy()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_cols]
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL,
              escapechar='\\', doublequote=True)
    logging.info(f"[write_csv_safely] Wrote {len(df)} rows to {filepath}")



def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day:
    - If today is trading day, return today
    - Else, go back until we find the last valid trading day.
    """
    if reference_date is None:
        reference_date=datetime.now().date()
    logging.info(f"Checking last trading day from {reference_date}.")
    print(f"Reference Date: {reference_date}")
    if reference_date.weekday()<5 and reference_date not in market_holidays:
        last_trading_day=reference_date
    else:
        last_trading_day=reference_date-timedelta(days=1)
        while last_trading_day.weekday()>=5 or last_trading_day in market_holidays:
            last_trading_day-=timedelta(days=1)
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")
    return last_trading_day


# ==============================
# Kite / API Setup
# ==============================
def setup_kite_session():
    try:
        with open("access_token.txt") as tk:
            access_token = tk.read().strip()
        with open("api_key.txt") as keyf:
            key_secret = keyf.read().split()
        kite_ = KiteConnect(api_key=key_secret[0])
        kite_.set_access_token(access_token)
        print("Kite session established.")
        return kite_
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

kite = setup_kite_session()


def get_tokens_for_stocks(stocks):
    """
    Fetch instrument tokens for the provided list of stocks.
    """
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        chosen = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][['tradingsymbol','instrument_token']]
        return dict(zip(chosen['tradingsymbol'], chosen['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        raise


shares_tokens = get_tokens_for_stocks(selected_stocks)


def make_api_call_with_retry(kite_, token, from_date, to_date, interval,
                             max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Repeatedly call kite.historical_data with backoff if we get 'too many requests' or network issues.
    """
    attempt = 0
    delay = initial_delay

    while attempt < max_retries:
        try:
            df_ = kite_.historical_data(token, from_date, to_date, interval)
            if not df_:
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
                continue
            return pd.DataFrame(df_)
        except (NetworkException, HTTPError, KiteException) as e:
            if 'too many requests' in str(e).lower():
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
            else:
                raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(delay)
            attempt += 1
            delay *= backoff_factor

    raise APICallError(f"Failed to fetch data for token {token} after {max_retries} retries.")


# ==============================
# Caching
# ==============================
def cache_file_path(ticker, data_type):
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")


def is_cache_fresh(filepath, threshold=timedelta(minutes=5)):
    if not os.path.exists(filepath):
        return False
    ctime = datetime.fromtimestamp(os.path.getmtime(filepath), pytz.UTC)
    return (datetime.now(pytz.UTC) - ctime) < threshold


def cache_update(ticker, data_type, fetch_function, freshness_threshold=None, bypass_cache=False):
    """
    Try to load cached data from CSV. If cache is outdated or bypass_cache=True, fetch fresh data and update cache.
    Only append new data (no old data overwrite).
    """
    cache_path = cache_file_path(ticker, data_type)
    if freshness_threshold is None:
        default_thresholds = {"historical": timedelta(hours=10), "intraday": timedelta(minutes=15)}
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
    For a given ticker, fetch both:
    - Historical data (past 7 days)
    - Intraday data (up to current time)
    Combine them, recalc indicators, and ensure no old data is overwritten.
    """

    def fetch_intraday(tkr):
        # Fetch today's intraday data
        df = make_api_call_with_retry(
            kite, token,
            from_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
            to_date_intra.strftime("%Y-%m-%d %H:%M:%S"),
            interval
        )
        if df.empty:
            return df
        df = normalize_time(df)
        return df

    def fetch_historical(tkr):
        # Fetch historical data for past 7 days as a buffer for indicator calc
        from_date_hist = from_date_intra - timedelta(days=10)
        to_date_hist = from_date_intra - timedelta(minutes=1)
        df = make_api_call_with_retry(
            kite, token,
            from_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
            to_date_hist.strftime("%Y-%m-%d %H:%M:%S"),
            interval
        )
        if df.empty:
            return df
        return normalize_time(df)

    # Use caching mechanism so we don't refetch repeatedly
    intraday_data = cache_update(ticker, "intraday", fetch_intraday, freshness_threshold=timedelta(minutes=15), bypass_cache=True)
    historical_data = cache_update(ticker, "historical", fetch_historical, freshness_threshold=timedelta(hours=10))

    # If no data at all, return empty
    if intraday_data.empty and historical_data.empty:
        logging.warning(f"No data available for {ticker}.")
        return pd.DataFrame()

    # Combine both sets
    combined_data = pd.concat([historical_data, intraday_data]).drop_duplicates('date').sort_values('date')

    # Recalculate indicators for new data
    combined_data = recalc_indicators(combined_data)

    # Ensure required columns exist
    if 'Signal_ID' not in combined_data.columns:
        combined_data['Signal_ID'] = combined_data.apply(lambda row: f"{ticker}-{row['date'].isoformat()}", axis=1)
    if 'Entry Signal' not in combined_data.columns:
        combined_data['Entry Signal'] = 'No'
    if 'logtime' not in combined_data.columns:
        combined_data['logtime'] = datetime.now(india_tz).isoformat()

    return combined_data
# ==============================
# Indicators
# ==============================
def calculate_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    return 100.0 - (100.0/(1.0+rs))


def calculate_macd(df_, fast=12, slow=26, signal=9):
    ema_fast = df_['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df_['close'].ewm(span=slow, adjust=False).mean()
    macd_ = ema_fast - ema_slow
    sig = macd_.ewm(span=signal, adjust=False).mean()
    hist_ = macd_ - sig
    return pd.DataFrame({'MACD':macd_, 'Signal_Line':sig, 'Histogram':hist_})


def calculate_atr(df_, period=14):
    d = df_.copy()
    d['prev_close'] = d['close'].shift(1)
    d['hl'] = d['high'] - d['low']
    d['hc'] = (d['high'] - d['prev_close']).abs()
    d['lc'] = (d['low'] - d['prev_close']).abs()
    tr = d[['hl','hc','lc']].max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def calculate_boll_bands(df_, period=20, up=2, dn=2):
    sma = df_['close'].rolling(period, min_periods=1).mean()
    std = df_['close'].rolling(period, min_periods=1).std()
    upper = sma + std*up
    lower = sma - std*dn
    return pd.DataFrame({'Upper_Band':upper,'Lower_Band':lower})


def calculate_stochastic(df_, fastk=14, slowk=3):
    low_min = df_['low'].rolling(fastk, min_periods=1).min()
    high_max = df_['high'].rolling(fastk, min_periods=1).max()
    p_k = 100*(df_['close'] - low_min)/(high_max - low_min + 1e-9)
    return p_k.rolling(slowk, min_periods=1).mean()


def calculate_adx(df_, period=14):
    d = df_.copy()
    d['prev_close'] = d['close'].shift(1)
    d['TR'] = d.apply(lambda row: max(row['high']-row['low'],
                                      abs(row['high']-row['prev_close']),
                                      abs(row['low']-row['prev_close'])), axis=1)
    d['+DM'] = np.where((d['high']-d['high'].shift(1))>(d['low'].shift(1)-d['low']),
                        np.where((d['high']-d['high'].shift(1))>0, d['high']-d['high'].shift(1),0),0)
    d['-DM'] = np.where((d['low'].shift(1)-d['low'])>(d['high']-d['high'].shift(1)),
                        np.where((d['low'].shift(1)-d['low'])>0, d['low'].shift(1)-d['low'],0),0)
    d['TR_smooth'] = d['TR'].rolling(period, min_periods=period).sum()
    d['+DM_smooth'] = d['+DM'].rolling(period, min_periods=period).sum()
    d['-DM_smooth'] = d['-DM'].rolling(period, min_periods=period).sum()

    for i in range(period, len(d)):
        d.at[d.index[i], 'TR_smooth'] = d.at[d.index[i-1],'TR_smooth'] - (d.at[d.index[i-1],'TR_smooth']/period) + d.at[d.index[i],'TR']
        d.at[d.index[i], '+DM_smooth'] = d.at[d.index[i-1],'+DM_smooth'] - (d.at[d.index[i-1],'+DM_smooth']/period) + d.at[d.index[i],'+DM']
        d.at[d.index[i], '-DM_smooth'] = d.at[d.index[i-1],'-DM_smooth'] - (d.at[d.index[i-1],'-DM_smooth']/period) + d.at[d.index[i],'-DM']

    d['+DI'] = (d['+DM_smooth']/d['TR_smooth'])*100
    d['-DI'] = (d['-DM_smooth']/d['TR_smooth'])*100
    d['DX'] = (abs(d['+DI'] - d['-DI'])/(d['+DI']+d['-DI']+1e-9))*100
    d['ADX'] = d['DX'].rolling(period, min_periods=period).mean()

    for i in range(2*period, len(d)):
        d.at[d.index[i],'ADX'] = ((d.at[d.index[i-1],'ADX']*(period-1)) + d.at[d.index[i],'DX'])/period

    return d['ADX']


def recalc_indicators(df_):
    """
    Only recalc indicators for last ~100 rows to avoid re-analysis of old data.
    """
    if df_.empty:
        return df_

    df_ = df_.copy().reset_index(drop=True)
    last_n = df_.tail(100).copy()

    # RSI
    df_.loc[last_n.index, 'RSI'] = calculate_rsi(last_n['close'])

    # ATR
    df_.loc[last_n.index, 'ATR'] = calculate_atr(last_n)

    # MACD
    macd_ = calculate_macd(last_n)
    df_.loc[last_n.index, ['MACD','Signal_Line','Histogram']] = macd_

    # Bollinger
    boll_ = calculate_boll_bands(last_n)
    df_.loc[last_n.index, ['Upper_Band','Lower_Band']] = boll_

    # Stochastic
    stoch_ = calculate_stochastic(last_n)
    df_.loc[last_n.index, 'Stochastic'] = stoch_

    # VWAP
    df_.loc[last_n.index, 'VWAP'] = (last_n['close']*last_n['volume']).cumsum() / last_n['volume'].cumsum()

    # 20_SMA
    df_.loc[last_n.index, '20_SMA'] = last_n['close'].rolling(20, min_periods=1).mean()

    # Recent high/low
    df_.loc[last_n.index, 'Recent_High'] = last_n['high'].rolling(5, min_periods=1).max()
    df_.loc[last_n.index, 'Recent_Low']  = last_n['low'].rolling(5, min_periods=1).min()

    # ADX
    adx_ = calculate_adx(last_n, 14)
    df_.loc[last_n.index, 'ADX'] = adx_

    return df_.ffill().bfill()


# ==============================
# Trend Conditions & Processing
# ==============================
def apply_trend_conditions(ticker, last_trading_day, tz):
    """
    Load main_indicators for ticker, slice today's data, check final row for signals.
    Return a list of signals (0 or more).
    """
    path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    if not os.path.exists(path):
        return []

    df = load_and_normalize_csv(path, tz=tz)
    if df.empty:
        return []

    # Slice to today's data
    start_ = tz.localize(datetime.combine(last_trading_day, datetime_time(9,15)))
    end_   = tz.localize(datetime.combine(last_trading_day, datetime_time(15,30)))
    today_ = df[(df['date']>=start_)&(df['date']<=end_)].copy()
    if today_.empty:
        return []

    # daily change
    try:
        first_ = today_['open'].iloc[0]
        last_  = today_['close'].iloc[-1]
        dchg = (last_ - first_)/first_*100
    except:
        return []

    # last row
    latest = today_.iloc[-1]
    adx_ = latest.get('ADX', 0)
    rsi_ = latest.get('RSI', 0)
    close_ = latest.get('close',0)
    vwap_ = latest.get('VWAP',0)

    strong_trend = (adx_>24)

    signals = []
    # check bullish or bearish
    if strong_trend:
        if (dchg>0.9) and (rsi_>18) and (close_>vwap_):
            # Bullish
            signals = _screen_bullish(ticker, last_trading_day, dchg, today_)
        elif (dchg<-0.9) and (rsi_<82) and (close_<vwap_):
            # Bearish
            signals = _screen_bearish(ticker, last_trading_day, dchg, today_)

    return signals

def _screen_bullish(ticker, day_, dchg, df_):
    # simplified from your logic
    # returns a list of dictionary signals
    # ...
    # For brevity, assume only 1 "Pullback" or "Breakout"
    return []

def _screen_bearish(ticker, day_, dchg, df_):
    # ...
    return []


def process_ticker(ticker, from_intra, to_intra, last_trading_day, tz):
    """
    Only append brand-new 15-min row. 
    Not re-writing old data, not re-checking old signals.
    """
    try:
        with api_semaphore:
            token = shares_tokens.get(ticker, None)
            if not token:
                return None, []

            # fetch new intraday, historical
            combined = fetch_intraday_data_with_historical(ticker, token, from_intra, to_intra)
            if combined.empty:
                return None, []

            # load existing main_indicators
            main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            if os.path.exists(main_path):
                existing = load_and_normalize_csv(main_path, tz=tz)
            else:
                existing = pd.DataFrame()

            # identify only the newest row(s)
            # e.g. the last row in combined
            if not existing.empty:
                existing_ids = set(existing['Signal_ID'])
            else:
                existing_ids = set()

            combined['Signal_ID'] = combined['Signal_ID'].astype(str)
            new_rows = combined[~combined['Signal_ID'].isin(existing_ids)].copy()
            new_rows = new_rows.sort_values('date')
            if new_rows.empty:
                return combined, []

            # We'll assume typically it's 1 row (the newly-formed 15-min bar).
            # But in edge cases, it might be multiple bars if script was off for a while.
            # So we keep them all but do not recalc old.
            # For the last 100 subset, we already recalc in fetch step.

            # Now we run apply_trend_conditions once for the "new" data
            # But by your design, we do not re-check older rows
            # We'll produce signals only if the "new" row(s) meet conditions 
            # => Actually we need to unify new rows into main_indicators first
            final_data = pd.concat([existing, new_rows], ignore_index=True)
            final_data.drop_duplicates(subset=['Signal_ID'], keep='first', inplace=True)
            final_data.sort_values('date', inplace=True)

            # Save final_data
            final_data.to_csv(main_path, index=False)
            logging.info(f"Appended {len(new_rows)} rows for {ticker} to main_indicators.csv")

            # Now detect signals from final_data
            # But only for the newly added row(s) - 
            #    or, since apply_trend_conditions slices by today's date anyway,
            #    we can run it and see if the new row triggered a signal
            signals = apply_trend_conditions(ticker, last_trading_day, tz)

            # If any signals found, we unify them 
            # and note that 'logtime' will be assigned for the new signals
            # We can append these signals to an "entries" object
            if signals:
                # We unify them with final_data to get the same Signal_ID and 'logtime'
                df_signals = pd.DataFrame(signals)
                if not df_signals.empty:
                    # attach the 'Signal_ID' from final_data
                    df_signals['date'] = pd.to_datetime(df_signals['date'], errors='coerce')
                    df_signals['date'] = df_signals['date'].dt.tz_localize(tz, nonexistent='shift_forward') \
                                                   .dt.tz_convert(tz)
                    merged = pd.merge(df_signals, final_data[['date','Signal_ID','logtime']], 
                                      on='date', how='left')
                    return final_data, merged.to_dict('records')
                else:
                    return final_data, []
            else:
                return final_data, []
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return None, []


# ==============================
# Higher-level driver
# ==============================
def find_price_action_entry(last_trading_day):
    """
    For each ticker in selected_stocks:
      - process_ticker(...) => returns new combined main_indicators + signals
      - gather signals in a single DataFrame
    """
    tz = india_tz
    # prepare intraday ranges
    from_date_intra = tz.localize(datetime.combine(last_trading_day, datetime_time(9,15)))
    to_date_intra   = tz.localize(datetime.combine(last_trading_day, datetime_time(15,30)))
    now_ = datetime.now(tz)
    if now_.date()==last_trading_day and now_.time()<datetime_time(15,30):
        to_date_intra = now_

    def wrapper(tkr):
        return process_ticker(tkr, from_date_intra, to_date_intra, last_trading_day, tz)

    entries_list = []
    with ThreadPoolExecutor(max_workers=2) as exe:
        futures = {exe.submit(wrapper, t):t for t in selected_stocks}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                r = fut.result()
                if r:
                    _, signals_ = r
                    if signals_:
                        entries_list.extend(signals_)
            except Exception as e:
                logging.error(f"[{ticker}] error: {e}")

    if not entries_list:
        return pd.DataFrame()

    df_out = pd.DataFrame(entries_list)
    if not df_out.empty:
        df_out = normalize_time(df_out, tz=tz)
        df_out.sort_values('date', inplace=True)
    return df_out


def create_papertrade_file(entries_file, papertrade_file, last_trading_day):
    """
    Takes minimal entries from 'entries_file' => earliest per ticker => add to papertrade.
    """
    tz = india_tz
    if not os.path.exists(entries_file):
        print(f"No entries file {entries_file}, skipping.")
        return

    ent = pd.read_csv(entries_file)
    if 'date' not in ent.columns:
        print(f"No 'date' in entries. Cannot proceed.")
        return

    ent = normalize_time(ent, tz=tz)
    ent = ent.sort_values('date').reset_index(drop=True)
    # filter to after 9:20 if you want:
    start_ = tz.localize(datetime.combine(last_trading_day, datetime_time(9,20)))
    filtered = ent[(ent['date']>=start_)&(ent['date'].dt.date==last_trading_day)]
    if filtered.empty:
        print(f"No valid entries after 9:20 for {last_trading_day}.")
        return

    earliest = filtered.groupby('Ticker').first().reset_index()
    # add target price etc. 
    # ... your logic ...
    # now unify with existing 'papertrade_file'
    if os.path.exists(papertrade_file):
        old_pap = pd.read_csv(papertrade_file)
        old_pap = normalize_time(old_pap, tz=tz)
        combined = pd.concat([old_pap, earliest], ignore_index=True)
        combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
        combined.sort_values('date', inplace=True)
        combined.to_csv(papertrade_file, index=False)
    else:
        earliest.to_csv(papertrade_file, index=False)

    print(f"Papertrade saved => {papertrade_file}")


def run():
    """
    driver:
      1) get last trading day
      2) find new entries
      3) store them in "entries file"
      4) create papertrade
    """
    last_td = get_last_trading_day()
    new_ents = find_price_action_entry(last_td)
    if new_ents.empty:
        print(f"No new signals for {last_td}")
        return

    # write to price_action_entries
    dtstr = last_td.strftime('%Y-%m-%d')
    entries_file = f"price_action_entries_15min_{dtstr}.csv"
    if os.path.exists(entries_file):
        old_e = load_and_normalize_csv(entries_file, tz=india_tz)
        combined = pd.concat([old_e, new_ents], ignore_index=True)
        combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
        combined.sort_values('date', inplace=True)
        combined.to_csv(entries_file, index=False)
    else:
        new_ents.to_csv(entries_file, index=False)

    # create papertrade
    papertrade_file = f"papertrade_{dtstr}.csv"
    create_papertrade_file(entries_file, papertrade_file, last_td)


# The rest of your scheduling logic...


if __name__=="__main__":
    run()
