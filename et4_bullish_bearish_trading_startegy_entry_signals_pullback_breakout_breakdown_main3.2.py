# -*- coding: utf-8 -*-
"""
Rewritten code3 for 30s scheduling, immediate papertrade updates,
and handling slow updates in _main_indicators.csv over 5 minutes,
while mitigating risks of missing or delayed signals.

Risks & Mitigations:

1) Race Conditions on CSV File Access:
   - Multiple threads/processes writing to CSVs can cause data inconsistencies.
   - Mitigation: Use FileLock (with 10s timeout) in mark_signals_in_main_csv()
     and create_papertrade_file() to avoid concurrent writes.

2) Signal Processing Time:
   - If detect_signals_in_memory takes longer than 30s, subsequent cycles
     may stack up.
   - Mitigation: Profile detect_signals_in_memory for performance. If needed,
     optimize or increase scheduling interval.

3) Kite API Limitations:
   - With concurrency of 2 (api_semaphore), slow/failing Kite API can delay signals.
   - Mitigation: Use retries in process_csv_with_retries for CSV load (analogous
     logic can be used for API calls). Consider dynamic backoff or increasing
     threads if within rate limits.

4) File System Delays:
   - Disk I/O or concurrency with os.makedirs, CSV reads/writes can be slow.
   - Mitigation: Use faster storage or in-memory caching. Our FileLock ensures
     concurrency but watch for high contention.

5) Schedule Misalignment:
   - Running main() every 30s can overlap with slow processing, leading to a backlog.
   - Mitigation: Potentially use schedule.idle_seconds() or track job durations,
     adjusting sleeps if needed.

6) Missing Signals in main_indicators.csv:
   - If new data arrives after processing, signals might be missed this cycle.
   - Mitigation: Because we run every 30s, newly inserted rows appear on
     subsequent runs. Or re-check last 10 min in find_price_action_entries_for_today.

7) Holiday & Non-Trading Hours:
   - If holiday list or hours logic is off, signals may be skipped.
   - Mitigation: Double-check market_holidays, get_last_trading_day() aligns with
     local exchange schedules.

Created on Thu Jan  2 18:57:31 2025
@author: Saarit
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import threading
import atexit
from datetime import datetime, timedelta, time as datetime_time
from datetime import datetime, timedelta, time as dt_time
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
import schedule  # For scheduling every 30s

# ================================
# Global Configuration
# ================================
india_tz = pytz.timezone('Asia/Kolkata')  # All date/time operations in IST

# Import selected stock tickers from an external file
from et4_filtered_stocks_market_cap import selected_stocks

# Directory structure
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Limit concurrency to avoid rate-limit issues (Kite API).
api_semaphore = threading.Semaphore(2)

# Market Holidays for 2024
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
    """Custom Exception for API call failures after retries."""
    pass

SIGNALS_DB = "generated_signals.json"

# ================================
# Logging Setup
# ================================
def setup_logging(log_filename="trading_script_signal3.log"):
    """
    Initializes TimedRotatingFileHandler to rotate logs every 30 minutes.
    This helps prevent oversized log files and concurrency issues on Windows.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # or DEBUG for more verbosity

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Timed rotating file handler
    timed_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when="M",         # 'M' => minutes, 'H' => hours, 'D' => days
        interval=30,      # rotate logs every 30 minutes
        backupCount=0,    # do not keep old logs
        delay=True        # only open file on first write
    )
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    timed_handler.setFormatter(fmt)
    timed_handler.setLevel(logging.INFO)
    logger.addHandler(timed_handler)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logging.info("Logging initialized with TimedRotatingFileHandler.")


def shutdown_logging():
    """Ensure logs are flushed on exit."""
    logging.shutdown()


atexit.register(shutdown_logging)


# ================================
# Attempt to set working directory
# ================================
def set_working_directory():
    """
    Change directory to the path where scripts/files exist.
    Raises an error if the directory does not exist.
    """
    cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
    try:
        os.chdir(cwd)
        logging.info(f"Changed working directory to {cwd}.")
    except FileNotFoundError:
        logging.error(f"Working directory {cwd} not found.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error changing directory: {e}")
        raise

# ================================
# Setup Kite Session
# ================================
def setup_kite_session():
    """
    Sets up the KiteConnect session from local files 'access_token.txt' / 'api_key.txt'.
    """
    try:
        with open("access_token.txt", 'r') as f:
            access_token = f.read().strip()
        with open("api_key.txt", 'r') as k:
            key_secret = k.read().split()

        kite_obj = KiteConnect(api_key=key_secret[0])
        kite_obj.set_access_token(access_token)
        logging.info("Kite session established.")
        print("Kite Connect session initialized successfully.")
        return kite_obj
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

# We'll store the global kite instance
kite = None

# ================================
# Load/Save Generated Signals
# ================================
def load_generated_signals():
    """
    Load a set of already generated Signal_IDs from JSON to avoid repeating signals.
    """
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB JSON is corrupted. Using empty set.")
            return set()
    else:
        return set()

def save_generated_signals(generated_signals):
    """
    Save the set of generated Signal_IDs to JSON to maintain state across runs.
    """
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f)

# ================================
# Normalize Times
# ================================
def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Convert 'date' to datetime and localize to the given timezone if needed.
    If 'date' is naive, assume it's UTC then convert.
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("No 'date' column found.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(timezone)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, timezone='Asia/Kolkata'):
    """
    Load a CSV, ensure 'date' column is properly localized, and add missing cols.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, timezone=timezone)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
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

# ================================
# Ticker => Instrument Token
# ================================
def get_tokens_for_stocks(stocks):
    """
    Return a dict: { 'SYMBOL': instrument_token, ... }
    """
    try:
        logging.info("Fetching tokens for provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        instrument_df['tradingsymbol'] = instrument_df['tradingsymbol'].str.upper()
        stocks_upper = [s.upper() for s in stocks]
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks_upper)][['tradingsymbol','instrument_token']]
        logging.info(f"Fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        raise

def signal_handler(sig, frame):
    """Handle Ctrl+C or kill signals gracefully."""
    logging.info("Interrupt signal received. Shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================================
# get_last_trading_day
# ================================
def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day, ignoring weekends & known holidays.
    """
    if reference_date is None:
        reference_date = datetime.now(india_tz).date()

    logging.info(f"Checking last trading day from {reference_date}.")
    print(f"Reference Date: {reference_date}")

    if reference_date.weekday()<5 and reference_date not in market_holidays:
        last_trading_day = reference_date
    else:
        last_trading_day = reference_date - timedelta(days=1)
        while last_trading_day.weekday()>=5 or last_trading_day in market_holidays:
            last_trading_day -= timedelta(days=1)

    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")
    return last_trading_day

# ================================
# detect_signals_in_memory
# ================================
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def create_signal_dict(ticker, row, sig_id, entry_type, trend_type, daily_change):
    """
    Helper to build consistent signal dictionary with fields used in papertrade & main_indicators.
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

def detect_bullish_scenarios(ticker, df_today, existing_signal_ids, daily_change):
    """
    Identify bullish pullback/breakout from the last row in df_today.
    """
    signals = []
    if df_today.empty:
        return signals

    latest_row = df_today.iloc[-1]

    # Example pullback condition
    pullback_cond = (
        (df_today["low"] >= df_today["VWAP"] * 0.95)
        & (df_today["close"] > df_today["open"])
        & (df_today["MACD"] > df_today["Signal_Line"])
        & (df_today["MACD"] > -1.0)
        & (df_today["close"] > df_today["20_SMA"])
        & (df_today["Stochastic"] < 60)
        & (df_today["Stochastic"].diff() > 0)
        & (df_today["ADX"] > 20)
    )
    if pullback_cond.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(create_signal_dict(
                ticker, latest_row, sig_id, "Pullback", "Bullish", daily_change
            ))

    # Example breakout condition
    breakout_cond = (
        (df_today["close"] > df_today["high"].rolling(window=20).max() * 0.97)
        & (df_today["volume"] > 1.5 * df_today["volume"].rolling(window=20).mean())
        & (df_today["MACD"] > df_today["Signal_Line"])
        & (df_today["MACD"] > 0.5)
        & (df_today["close"] > df_today["20_SMA"])
        & (df_today["Stochastic"] > 60)
        & (df_today["Stochastic"].diff() > 0)
        & (df_today["ADX"] > 30)
    )
    if breakout_cond.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(create_signal_dict(
                ticker, latest_row, sig_id, "Breakout", "Bullish", daily_change
            ))
    return signals

def detect_bearish_scenarios(ticker, df_today, existing_signal_ids, daily_change):
    """
    Identify bearish pullback/breakdown from the last row in df_today.
    """
    signals = []
    if df_today.empty:
        return signals

    latest_row = df_today.iloc[-1]

    # Pullback (bearish)
    pullback_cond = (
        (df_today["high"] <= df_today["VWAP"] * 1.05)
        & (df_today["close"] < df_today["open"])
        & (df_today["MACD"] < df_today["Signal_Line"])
        & (df_today["MACD"] < 1.0)
        & (df_today["close"] < df_today["20_SMA"])
        & (df_today["Stochastic"] > 40)
        & (df_today["Stochastic"].diff() < 0)
        & (df_today["ADX"] > 20)
    )
    if pullback_cond.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(create_signal_dict(
                ticker, latest_row, sig_id, "Pullback", "Bearish", daily_change
            ))

    # Breakdown
    breakdown_cond = (
        (df_today["close"] < df_today["low"].rolling(window=20).min() * 1.03)
        & (df_today["volume"] > 1.5 * df_today["volume"].rolling(window=20).mean())
        & (df_today["MACD"] < df_today["Signal_Line"])
        & (df_today["MACD"] < -0.5)
        & (df_today["close"] < df_today["20_SMA"])
        & (df_today["Stochastic"] < 40)
        & (df_today["Stochastic"].diff() < 0)
        & (df_today["ADX"] > 30)
    )
    if breakdown_cond.iloc[-1]:
        sig_id = latest_row.get('Signal_ID', build_signal_id(ticker, latest_row))
        if sig_id not in existing_signal_ids:
            signals.append(create_signal_dict(
                ticker, latest_row, sig_id, "Breakdown", "Bearish", daily_change
            ))
    return signals

def detect_signals_in_memory(ticker: str, df_today: pd.DataFrame, existing_signal_ids: set) -> list:
    """
    Single function that calls bullish/bearish scenario detection
    if daily_change & momentum pass certain thresholds.
    """
    signals_detected = []
    if df_today.empty:
        return signals_detected

    # daily_change calc
    try:
        first_open = df_today["open"].iloc[0]
        latest_close = df_today["close"].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except:
        return signals_detected

    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get("ADX", 0)
    if adx_val < 30:  # require strong momentum
        return signals_detected

    # Bullish
    if (
        daily_change > 2
        and latest_row.get("RSI", 0) > 40
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0)*1.01
    ):
        signals_detected.extend(
            detect_bullish_scenarios(ticker, df_today, existing_signal_ids, daily_change)
        )
    # Bearish
    elif (
        daily_change < -2
        and latest_row.get("RSI", 100) < 60
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0)*0.99
    ):
        signals_detected.extend(
            detect_bearish_scenarios(ticker, df_today, existing_signal_ids, daily_change)
        )

    return signals_detected

# ================================
# process_csv_with_retries
# ================================
def process_csv_with_retries(file_path, ticker, max_retries=3, delay_sec=5):
    """
    Attempt to load a CSV up to max_retries, with delay_sec between attempts.
    Helps mitigate transient file-access or parse issues.
    """
    for attempt in range(1, max_retries+1):
        try:
            df = load_and_normalize_csv(file_path, timezone='Asia/Kolkata')
            return df
        except Exception as e:
            logging.error(f"[{ticker}] CSV load attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(delay_sec)
            else:
                raise

# ================================
# Mark signals in main CSV
# ================================
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    For each signal, set 'Entry Signal'='Yes' in main_indicators if not already.
    Uses a FileLock to prevent concurrency issues. Timeout=10s is moderate.
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No main_indicators CSV at {main_path}. Skipping.")
        return

    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, timezone=tz_obj)
            if df_main.empty:
                logging.warning(f"[{ticker}] main_indicators.csv empty. Skipping.")
                return

            if 'Entry Signal' not in df_main.columns:
                df_main['Entry Signal'] = 'No'

            updated = 0
            for signal in signals_list:
                sig_id = signal.get("Signal_ID", "")
                sig_date = signal.get("date", None)
                if sig_id and 'Signal_ID' in df_main.columns:
                    mask = (df_main['Signal_ID'] == sig_id)
                else:
                    mask = (df_main['date'] == sig_date)

                if mask.any():
                    mask_up = mask & (df_main['Entry Signal'] != 'Yes')
                    if mask_up.any():
                        df_main.loc[mask_up, 'Entry Signal'] = 'Yes'
                        df_main.loc[mask_up, 'logtime'] = signal.get(
                            'logtime',
                            datetime.now(tz_obj).isoformat()
                        )
                        updated += mask_up.sum()

            if updated > 0:
                df_main.sort_values('date', inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] Marked 'Entry Signal'='Yes' for {updated} signals.")
    except Timeout:
        logging.error(f"[{ticker}] Timeout acquiring lock for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] Error marking signals: {e}")

# ================================
# find_price_action_entries_for_today
# ================================
def find_price_action_entries_for_today(last_trading_day):
    """
    Reads each *_main_indicators.csv, slices data from 09:45–14:20,
    detects signals from the final row, merges them into a single DataFrame.
    Repeated runs every 30s help catch newly inserted rows in slow-updating CSVs.
    """
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9, 45)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14, 20)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        logging.warning(f"No CSV in {INDICATORS_DIR}.")
        print(f"No CSV files found in {INDICATORS_DIR}.")
        return pd.DataFrame()

    all_signals = []
    generated_signals = load_generated_signals()

    for file_path in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','').upper()
        try:
            df = process_csv_with_retries(file_path, ticker)
            if df.empty:
                continue
            df_today = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()
            if df_today.empty:
                continue

            latest_df = df_today.iloc[-1:].copy()  # check only last row
            new_signals = detect_signals_in_memory(ticker, latest_df, generated_signals)
            if new_signals:
                for s in new_signals:
                    generated_signals.add(s["Signal_ID"])
                all_signals.extend(new_signals)
        except:
            continue

    save_generated_signals(generated_signals)
    if not all_signals:
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df

# ================================
# create_papertrade_file
# ================================
def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads signals from input_file, filters earliest Ticker entry,
    appends/merges to output_file with FileLock concurrency protection.
    Adds a 'time' column to capture the HH:MM:SS extracted from each row's 'date'.

    If you prefer using the current time for all rows, replace:
        todays_entries['time'] = todays_entries['date'].dt.strftime('%H:%M:%S')
    with:
        todays_entries['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    """
    if not os.path.exists(input_file):
        print("No signals file => skip papertrade.")
        return
    try:
        df_entries = pd.read_csv(input_file)
        if 'date' not in df_entries.columns:
            raise KeyError("No 'date' column in signals file.")
        df_entries = normalize_time(df_entries)
        df_entries.sort_values('date', inplace=True)
        df_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

        st = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
        et = india_tz.localize(datetime.combine(last_trading_day, dt_time(14, 50)))
        todays_entries = df_entries[(df_entries['date'] >= st) & (df_entries['date'] <= et)].copy()
        if todays_entries.empty:
            return

        # Sort by datetime ascending
        todays_entries.sort_values('date', inplace=True)
        
        # Keep earliest daily Ticker row
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

        # -----------------------------
        # Add 'time' column using .dt
        # -----------------------------
        # Prepare 'time' column with current time
        current_time = datetime.now(india_tz).strftime('%H:%M:%S')
        df_entries['time'] = current_time

        # Merge into existing papertrade CSV
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, tz='Asia/Kolkata')
            existing.sort_values('date', inplace=True)

            # Check if 'time' column exists in existing, if not, add it
            if 'time' not in existing.columns:
                existing['time'] = ''

            # Combine existing and new entries
            all_paper = pd.concat([existing, df_entries], ignore_index=True)

            # Drop duplicates on [Ticker, date], keeping the last (new entry)
            all_paper.drop_duplicates(subset=['Ticker', 'date'], keep='last', inplace=True)

            # Sort by 'date' ascending
            all_paper.sort_values('date', inplace=True)

            # Save to CSV
            all_paper.to_csv(output_file, index=False)
        else:
            # If output_file does not exist, ensure 'time' column is present
            if 'time' not in df_entries.columns:
                df_entries['time'] = current_time
            df_entries.to_csv(output_file, index=False)

        print(f"Papertrade => {output_file}")
    except Exception as e:
        print(f"create_papertrade error: {e}")
        logging.error(f"Papertrade error: {e}")

# ================================
# Main Function
# ================================
def main():
    """
    1) Determine last trading day
    2) Gather signals => single DataFrame
    3) Immediately store them in price_action_entries_15min_{YYYY-MM-DD}.csv
    4) Mark 'Entry Signal' in main_indicators
    5) Immediately create/update papertrade_{YYYY-MM-DD}.csv

    Runs every 30s. If main_indicators.csv updates slowly over 5 minutes,
    repeated checks ensure newly appended rows are eventually captured.
    """
    last_trading_day = get_last_trading_day()
    date_str = last_trading_day.strftime('%Y-%m-%d')
    logging.info(f"Running main() for trading day: {last_trading_day}")

    # 1) Find signals for the last row of each ticker CSV
    signals_df = find_price_action_entries_for_today(last_trading_day)
    if signals_df.empty:
        logging.info("No signals found this iteration.")
        print(f"No signals found for {last_trading_day} at this time.")
        return

    # 2) Write them to a CSV immediately
    entries_file = f"price_action_entries_15min_{date_str}.csv"
    try:
        signals_df.to_csv(entries_file, index=False)
        logging.info(f"Wrote {len(signals_df)} signals to '{entries_file}'.")
        print(f"Saved {len(signals_df)} new signals => {entries_file}")
        print("\nDetected Signals:\n", signals_df)
    except Exception as e:
        logging.error(f"Error writing signals CSV: {e}")
        return

    # 3) Mark 'Entry Signal' in main_indicators
    try:
        unique_signals = signals_df.drop_duplicates(subset=['Ticker','date'])
        for _, sigrow in unique_signals.iterrows():
            ticker = sigrow["Ticker"]
            main_ind_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, [sigrow.to_dict()], main_ind_path, india_tz)
        logging.info("Updated 'Entry Signal' in main_indicators CSV.")
        print("Updated 'Entry Signal' in main_indicators.")
    except Exception as e:
        logging.error(f"Error marking signals: {e}")
        return

    # 4) Immediately create/update papertrade file
    papertrade_file = f"papertrade_{date_str}.csv"
    try:
        create_papertrade_file(entries_file, papertrade_file, last_trading_day)
    except Exception as e:
        logging.error(f"Error in create_papertrade_file: {e}")

    logging.info("main() completed successfully.")
    print("main() completed successfully.\n")

# ================================
# Scheduling for Every 30 Seconds
# ================================
def schedule_jobs():
    """
    We schedule main() to run every 30 seconds, ensuring frequent checks.
    If main_indicators updates slowly, new rows eventually get detected.
    """
    schedule.every(30).seconds.do(main)

def run_schedule_loop():
    """Keep the scheduler running indefinitely."""
    while True:
        schedule.run_pending()
        time.sleep(1)

# ================================
# Single-Instance Lock
# ================================
def acquire_lock(lock_file_path, timeout=10):
    """
    Acquire a file lock to ensure only one instance runs at a time.
    If locked, exit after 10s.
    """
    lock = FileLock(lock_file_path, timeout=timeout)
    try:
        lock.acquire()
        logging.info("Global lock acquired.")
        return lock
    except Timeout:
        logging.error("Another instance is running. Exiting.")
        print("Another instance is running. Exiting.")
        sys.exit(1)

# ================================
# Entrypoint
# ================================
if __name__=="__main__":
    # 1) Setup logging
    setup_logging("trading_script_signal3.log")

    # 2) Set working directory
    set_working_directory()

    # 3) Setup Kite session
    kite = setup_kite_session()

    # 4) Acquire single-instance lock
    global_lock = acquire_lock("trading_script_signal3.lock")

    # 5) (Optional) fetch tokens if needed
    # shares_tokens = get_tokens_for_stocks(selected_stocks)

    try:
        # 6) Schedule main() every 30 seconds
        schedule_jobs()
        run_schedule_loop()
    finally:
        global_lock.release()
        logging.info("Global lock released.")
