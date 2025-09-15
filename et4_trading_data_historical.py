# -*- coding: utf-8 -*-
"""
Modified script for historical 15-minute data over specific dates,
calculating indicators and storing results in 'main_indicators_historical'.
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
india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for date/time operations

# IMPORTANT: Provide or import your filtered list of stocks here:
from et4_filtered_stocks_market_cap import selected_stocks
# from et4_filtered_stocks import selected_stocks

CACHE_DIR = "data_cache"
HIST_INDICATORS_DIR = "main_indicators_historical"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(HIST_INDICATORS_DIR, exist_ok=True)

api_semaphore = threading.Semaphore(2)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Base level to avoid extremely verbose logs

file_handler = logging.FileHandler("logs\\trading_data_historical.log")
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

# Change working directory if needed:
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


class APICallError(Exception):
    """Custom Exception for repeated API call failures."""
    pass

def setup_kite_session():
    """Reads stored credentials and returns an authenticated KiteConnect instance."""
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()
        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        kite_session = KiteConnect(api_key=key_secret[0])
        kite_session.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        print("Kite Connect session initialized successfully.")
        return kite_session
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

kite = setup_kite_session()


def refresh_instrument_list():
    """Refreshes the instrument tokens map from the NSE instrument dump."""
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        shares_tokens_updated = dict(
            zip(instrument_df['tradingsymbol'], instrument_df['instrument_token'])
        )
        logging.info("Instrument list refreshed successfully.")
        return shares_tokens_updated
    except Exception as e:
        logging.error(f"Error refreshing instrument list: {e}")
        # If refresh fails, we continue with old tokens in shares_tokens (defined later)
        return None


def normalize_time(df, timezone='Asia/Kolkata'):
    """Ensures 'date' column is in the specified timezone."""
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)  # Remove invalid date rows

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("Could not convert 'date' column to datetime.")

    # If not tz-aware, assume UTC, then convert to local
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(timezone)

    return df


def load_and_normalize_csv(file_path, expected_cols=None, timezone='Asia/Kolkata'):
    """Loads a CSV (if exists), normalizes the date column, ensures expected cols, returns DataFrame."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, timezone=timezone)
        df = df.sort_values(by='date').reset_index(drop=True)
    else:
        # If no 'date' in existing, fill missing columns
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


def get_tokens_for_stocks_with_retry(stocks, retries=3, delay=5):
    """Retries fetching tokens for given stocks if there's a failure."""
    for attempt in range(1, retries + 1):
        try:
            return get_tokens_for_stocks(stocks)
        except Exception as e:
            logging.error(f"Attempt {attempt} to fetch tokens failed: {e}")
            if attempt < retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Exiting token fetch.")
                raise


def get_tokens_for_stocks(stocks):
    """Fetches instrument tokens for the given list of tradingsymbols from Kite."""
    try:
        logging.info("Fetching tokens for provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][
            ['tradingsymbol','instrument_token']
        ]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        raise


shares_tokens_master = refresh_instrument_list()
if shares_tokens_master is None:
    # fallback to direct fetch with retry
    shares_tokens_master = get_tokens_for_stocks_with_retry(selected_stocks)
else:
    # If we got a new map, we still ensure we have tokens for the selected_stocks
    # We can merge or just re-fetch for selected stocks:
    existing_symbols = set(shares_tokens_master.keys())
    needed_symbols = set(selected_stocks)
    if not needed_symbols.issubset(existing_symbols):
        # Some tokens missing from the refresher, so re-fetch for those
        missing_symbols = needed_symbols - existing_symbols
        partial_tokens = get_tokens_for_stocks_with_retry(list(missing_symbols))
        shares_tokens_master.update(partial_tokens)


def make_api_call_with_retry(kite_obj, token, from_date, to_date, interval,
                             max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Makes a historical_data API call with exponential backoff for rate-limiting
    and connection issues.
    """
    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            logging.info(f"API call attempt {attempt+1} for token {token}.")
            data = kite_obj.historical_data(token, from_date, to_date, interval)
            if not data:
                logging.warning(f"No data returned for token={token} in attempt {attempt+1}, retrying...")
                time.sleep(delay)
                attempt += 1
                delay *= backoff_factor
                continue
            logging.info(f"Data fetched for token={token}, attempt {attempt+1}.")
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

    raise APICallError(f"Failed to fetch data for token={token} after {max_retries} retries.")


def cache_file_path(ticker, data_type):
    """Constructs a cache file path for a given ticker and data type."""
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}.csv")


def is_cache_fresh(cache_file, freshness_threshold=timedelta(minutes=15)):
    """Checks if the cache file is within the given freshness threshold."""
    if not os.path.exists(cache_file):
        return False
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file), pytz.UTC)
    current_time = datetime.now(pytz.UTC)
    return (current_time - cache_mtime) < freshness_threshold


def cache_update(ticker, data_type, fetch_function,
                 freshness_threshold=None, bypass_cache=False):
    """
    Generic caching layer. If `bypass_cache` is True, fetches fresh data and merges
    with existing. Otherwise, tries to use the cache if it's fresh.
    """
    cache_path = cache_file_path(ticker, data_type)
    if freshness_threshold is None:
        default_thresholds = {"historical": timedelta(hours=10),
                              "intraday": timedelta(minutes=15)}
        freshness_threshold = default_thresholds.get(data_type, timedelta(minutes=15))

    # Use cache if it's fresh and bypass_cache is False
    if not bypass_cache and is_cache_fresh(cache_path, freshness_threshold):
        try:
            logging.info(f"Loading cached {data_type} data for {ticker}.")
            df = pd.read_csv(cache_path, parse_dates=['date'])
            df = normalize_time(df)
            return df
        except Exception:
            logging.warning("Could not load from cache, fetching fresh data...")

    logging.info(f"Fetching fresh {data_type} data for {ticker}.")
    fresh_data = fetch_function(ticker)
    if fresh_data is None or fresh_data.empty:
        logging.warning(f"No fresh {data_type} data for {ticker}. Returning empty.")
        return pd.DataFrame()

    # Merge with existing cache if it exists
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, parse_dates=['date'])
        existing = normalize_time(existing)
        combined = pd.concat([existing, fresh_data]).drop_duplicates(subset='date')
        combined.sort_values('date', inplace=True)
        combined.to_csv(cache_path, index=False)
        return combined
    else:
        fresh_data.to_csv(cache_path, index=False)
        return fresh_data


def fetch_intraday_data_with_historical(ticker, token, from_dt, to_dt, interval="15minute"):
    """
    Fetch a single day's intraday data + a bit of prior historical (10 days)
    for context, combine, then recalc indicators for the latest rows.
    """
    def fetch_intraday(_):
        df = make_api_call_with_retry(
            kite, token,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            interval
        )
        return normalize_time(df) if not df.empty else df

    def fetch_historical(_):
        # For example, we fetch 10 days prior to the 'from_dt'
        from_date_hist = from_dt - timedelta(days=10)
        to_date_hist = from_dt - timedelta(minutes=1)
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

    return combined


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
    req_cols = {'high', 'low', 'close'}
    if not req_cols.issubset(d.columns):
        raise ValueError(f"Missing required columns for ADX: {req_cols - set(d.columns)}")

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

    # Smooth further with Wilder's method
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


def recalc_indicators_for_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculates all indicators over the entire DataFrame (15min candles).
    Returns a new DataFrame with the indicator columns added.
    """
    if df.empty:
        return df

    # We'll work on a copy
    out = df.copy().reset_index(drop=True)

    try:
        # RSI
        out['RSI'] = calculate_rsi(out['close'], timeperiod=14)
        # ATR
        out['ATR'] = calculate_atr(out)
        # MACD
        macd_df = calculate_macd(out, fast=12, slow=26, signal=9)
        out[['MACD', 'Signal_Line', 'Histogram']] = macd_df
        # Bollinger Bands
        boll_df = calculate_bollinger_bands(out, period=20, up=2, dn=2)
        out[['Upper_Band','Lower_Band']] = boll_df
        # Stochastic
        stoch_k = calculate_stochastic(out, fastk=14, slowk=3)
        out['Stochastic'] = stoch_k
        # VWAP
        #  (for 15min data, typically you'd do partial sums, but here we do a simple cumulative approach)
        out['VWAP'] = (out['close'] * out['volume']).cumsum() / out['volume'].cumsum()
        # 20 SMA
        out['20_SMA'] = out['close'].rolling(window=20, min_periods=1).mean()
        # 5-candle recent high/low
        out['Recent_High'] = out['high'].rolling(window=5, min_periods=1).max()
        out['Recent_Low']  = out['low'].rolling(window=5, min_periods=1).min()
        # ADX
        adx_series = calculate_adx(out, period=14)
        out['ADX'] = adx_series

    except Exception as e:
        logging.error(f"Indicator calculation error: {e}")
        logging.error(traceback.format_exc())
        raise

    # forward/back fill if needed
    out = out.ffill()
    out = out.bfill()
    return out


def build_signal_id(ticker, row):
    """You can define a unique ID combining ticker and the ISO datetime of the row."""
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"


def process_ticker_for_date(ticker, day_date):
    """
    Core function to fetch 15min data for a single 'day_date' (9:15–15:30),
    recalc indicators, append results to the historical indicators CSV.
    Returns the full day's data (with indicators) for that ticker.
    """
    try:
        token = shares_tokens_master.get(ticker)
        if not token:
            logging.warning(f"[{ticker}] No instrument token found.")
            return pd.DataFrame()

        from_date_intra = india_tz.localize(datetime.combine(day_date, datetime_time(9, 15)))
        to_date_intra   = india_tz.localize(datetime.combine(day_date, datetime_time(15, 30)))

        # 1) Fetch that day's combined data
        combined_data = fetch_intraday_data_with_historical(
            ticker, token, from_date_intra, to_date_intra, interval="15minute"
        )
        if combined_data.empty:
            logging.warning(f"[{ticker}] No data returned for date={day_date}")
            return pd.DataFrame()

        # 2) Recalculate indicators across all those bars
        full_df = recalc_indicators_for_data(combined_data)
        if full_df.empty:
            return full_df

        # 3) Add "Signal_ID" and default "Entry Signal" = "No"
        if 'Signal_ID' not in full_df.columns:
            full_df['Signal_ID'] = full_df.apply(lambda r: build_signal_id(ticker, r), axis=1)
        else:
            mask_missing = full_df['Signal_ID'].isna() | (full_df['Signal_ID'] == "")
            full_df.loc[mask_missing, 'Signal_ID'] = full_df.loc[mask_missing].apply(
                lambda r: build_signal_id(ticker, r), axis=1
            )

        if 'Entry Signal' not in full_df.columns:
            full_df['Entry Signal'] = 'No'
        else:
            # force them all to "No" for safety
            full_df['Entry Signal'] = 'No'

        # 4) Set a logtime for each row
        full_df['logtime'] = datetime.now(india_tz).isoformat()

        # 5) Load existing historical CSV (if exists), merge
        hist_path = os.path.join(HIST_INDICATORS_DIR, f"{ticker}_main_indicators_historical.csv")
        if os.path.exists(hist_path):
            existing = pd.read_csv(hist_path, parse_dates=['date'])
            existing = normalize_time(existing, timezone='Asia/Kolkata')
        else:
            existing = pd.DataFrame()

        # Combine and remove duplicates
        combined = pd.concat([existing, full_df], ignore_index=True)
        # Deduplicate on 'Signal_ID'
        if 'Signal_ID' in combined.columns:
            combined.drop_duplicates(subset=['Signal_ID'], keep='first', inplace=True)
        combined.sort_values('date', inplace=True)

        # 6) Save final
        combined.to_csv(hist_path, index=False)
        logging.info(f"[{ticker}] Updated historical for date={day_date} => {hist_path}")

        return full_df

    except Exception as e:
        logging.error(f"[{ticker}] Error processing for date={day_date}: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()


def initialize_calc_flag_in_csvs(indicators_dir="main_indicators_historical"):
    """
    Example helper to ensure a 'CalcFlag' column is present in your CSVs.
    Not strictly required, but sometimes used for post-processing.
    """
    import glob
    import os
    import traceback

    pattern = os.path.join(indicators_dir, "*_main_indicators_historical.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            if "CalcFlag" not in df.columns:
                df["CalcFlag"] = "No"
                logging.info(f"Added 'CalcFlag' column to {file_path} with default 'No'.")
            else:
                # fill any blank or NaN as "No"
                df["CalcFlag"] = df["CalcFlag"].fillna("")
                df.loc[df["CalcFlag"].eq(""), "CalcFlag"] = "No"

            df.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Error adding CalcFlag to {file_path}: {e}")
            logging.error(traceback.format_exc())


def signal_handler(sig, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

# Register the handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ===================================
# MAIN: Run Historical for Each Date
# ===================================
def run_historical(dates_list):
    """
    Main entry function to iterate over a list of YYYY-MM-DD strings,
    fetch 15-minute data, calculate indicators, and store final outputs
    in 'main_indicators_historical' directory.
    """
    # Ensure the directory exists
    os.makedirs(HIST_INDICATORS_DIR, exist_ok=True)

    for dt_str in dates_list:
        try:
            day_date = datetime.strptime(dt_str, '%Y-%m-%d').date()
        except ValueError:
            logging.error(f"Invalid date string: {dt_str}")
            continue

        logging.info(f"Processing date: {day_date}")
        print(f"\n=== Processing date: {day_date} ===")

        # We'll process all stocks in parallel
        def wrapper(tkr):
            return process_ticker_for_date(tkr, day_date)

        # Store resulting data frames in a dict, if needed
        processed_data_dict = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_ticker = {executor.submit(wrapper, t): t for t in selected_stocks}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df_result = future.result()
                    processed_data_dict[ticker] = df_result
                except Exception as e:
                    logging.error(f"Error in future for {ticker} on {day_date}: {e}")

        # Optionally do something with processed_data_dict
        print(f"Finished date: {day_date}, updated {len(processed_data_dict)} tickers.")


# =========================
# Example usage of run_historical
# =========================
if __name__ == "__main__":
    # Suppose we want to fetch for these two historical dates:
    historical_dates = ["2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01"]

    # Call the function:
    run_historical(historical_dates)

    # (Optional) Initialize CalcFlag columns in newly created CSVs, if desired:
    initialize_calc_flag_in_csvs(indicators_dir=HIST_INDICATORS_DIR)

    print("\nAll historical data processing completed.\n")
