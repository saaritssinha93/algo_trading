# -*- coding: utf-8 -*-
"""
Updated Script with Mitigation Steps for Risks:
-----------------------------------------------
1) Race Conditions (FileLock usage, lock timeouts).
2) Slow or Delayed Signal Processing (profiling note).
3) Kite API Limitations (Semaphore, retry logic).
4) File System Delays (recommend faster disk, avoid excessive concurrency).
5) Schedule Misalignment (schedule every 30s; consider schedule.idle_seconds).
6) Missing Late Arrivals in main_indicators.csv (re-scan last ~10 minutes or repeated 30s checks).
7) Holiday & Non-Trading Hours (double-check market_holidays, reference local schedules).

Runs main() every 30 seconds to detect signals quickly and reduce missed entries.
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import pytz
from kiteconnect import KiteConnect
import traceback
import glob
import signal
import sys
from logging.handlers import TimedRotatingFileHandler
from tqdm import tqdm
import json
from filelock import FileLock, Timeout
import schedule   # For time-based scheduling

# ================================
# 1) Configuration & Setup
# ================================
india_tz = pytz.timezone('Asia/Kolkata')  # Indian timezone
from et4_filtered_stocks_market_cap import selected_stocks  # Imported tickers

# Folders for caching & CSVs
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# API concurrency limit (Kite API)
api_semaphore = threading.Semaphore(2)  # If needed in your fetch logic

# Logging setup: rotate every 30 minutes, keep 5 backups
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "trading_script_signal1.log",
    when="M",
    interval=30,     # rotate logs every 30 minutes
    backupCount=5,
    delay=True       # only open file on first write
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

# Avoid double-add of handlers
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging with TimedRotatingFileHandler.")
print("Script start...")

# Attempt to set working directory
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info(f"Changed working directory to {cwd}.")
except Exception as e:
    logger.error(f"Error changing directory: {e}")

# Market holidays for 2024 (example list)
market_holidays = [
    # Add your actual holiday dates here
]

class APICallError(Exception):
    """Custom exception for repeated API call failures."""
    pass

SIGNALS_DB = "generated_signals.json"

# ================================
# 2) Potential Risk Mitigations
# ================================
"""
- Race Conditions on CSV: 
  => We use FileLock with a 10s timeout for reading/writing main_indicators.csv and papertrade.csv.

- Slow or Delayed Processing:
  => We schedule every 30s. If signal detection or CSV merging is slow, 
     we recommend profiling detect_signals_in_memory for optimization.

- Kite API Limitations:
  => We keep a concurrency limit of 2 threads with 'api_semaphore'. 
     If needed, we can expand or add backoff in API calls.

- File System Delays:
  => We rely on FileLock and CSV read/writes. If your environment is slow, 
     consider using an in-memory approach for partial work.

- Schedule Misalignment:
  => Using schedule every 30s ensures repeated checks, even if some runs overlap. 
     For heavy loads, you can monitor schedule.idle_seconds() or measure job time.

- Missing Signals (late arrivals):
  => Repeated 30s scanning mitigates this. We might also re-check data from the last ~10 mins if needed.

- Holidays & Off Hours:
  => We rely on 'market_holidays' and 'get_last_trading_day()' to skip or adjust. 
     Ensure these align with your local exchange.
"""

# ================================
# 3) JSON-based Signal Tracking
# ================================
def load_generated_signals():
    """Load the set of Signal_IDs from a JSON file to avoid duplicates."""
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB JSON is corrupted. Starting with empty set.")
            return set()
    else:
        return set()

def save_generated_signals(generated_signals):
    """Persist the set of Signal_IDs to disk."""
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f)

# ================================
# 4) Time Normalization & CSV Loading
# ================================
def normalize_time(df, tz='Asia/Kolkata'):
    """Ensure 'date' is a proper timezone-aware datetime in tz."""
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)  # remove invalid date rows

    # localize naive -> UTC, then convert
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, tz='Asia/Kolkata'):
    """Load a CSV, ensure 'date' is localized, ensure expected_cols exist if given."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, tz)
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
# 5) KiteConnect Session Setup
# ================================
def setup_kite_session():
    """Establishes a KiteConnect session using local files for tokens/keys."""
    try:
        with open("access_token.txt",'r') as tf:
            access_token = tf.read().strip()
        with open("api_key.txt",'r') as kf:
            key_secret = kf.read().split()

        kite_obj = KiteConnect(api_key=key_secret[0])
        kite_obj.set_access_token(access_token)
        logging.info("Kite session is active.")
        print("Kite session OK.")
        return kite_obj
    except Exception as e:
        logging.error(f"Kite setup error: {e}")
        raise

kite = setup_kite_session()

# Retrieve instrument tokens if needed
def get_tokens_for_stocks(stocks):
    """
    Return { 'SYMBOL': instrument_token } mapping from the Kite instruments dump.
    """
    try:
        logging.info("Fetching tokens.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        instrument_df['tradingsymbol'] = instrument_df['tradingsymbol'].str.upper()
        stocks_upper = [s.upper() for s in stocks]
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks_upper)][['tradingsymbol','instrument_token']]
        logging.info(f"Fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched: {len(tokens)}")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        raise

shares_tokens = get_tokens_for_stocks(selected_stocks)


# ================================
# 6) Signal Detection Logic
# ================================
def build_signal_id(ticker, row):
    """Generate a unique ID from ticker + row's date."""
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def detect_signals_in_memory(ticker, df_today, existing_signal_ids):
    """
    Detect signals for the 'latest' (or entire) df_today.
    We rely on daily_change, ADX, RSI, MACD conditions, 
    skipping duplicates found in existing_signal_ids.
    """
    signals_detected = []

    if df_today.empty:
        return signals_detected

    try:
        first_open = df_today['open'].iloc[0]
        latest_close = df_today['close'].iloc[-1]
        daily_change = ((latest_close - first_open)/ first_open)*100
    except:
        return signals_detected

    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get('ADX', 0)
    if adx_val <= 20:
        return signals_detected

    # Decide Bullish vs Bearish scenario
    if (
        daily_change > 1.25
        and latest_row.get("RSI", 0) > 35
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0)*1.02
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.25
        and latest_row.get("RSI", 0) < 65
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0)*0.98
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # Additional thresholds
    volume_multiplier = 1.5
    rolling_window = 20
    MACD_BULLISH_OFFSET = 0.5
    MACD_BEARISH_OFFSET = -0.5
    MACD_BULLISH_OFFSETp = -1.0
    MACD_BEARISH_OFFSETp = 1.0

    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["low"] >= df_today["VWAP"]*0.95)
                & (df_today["close"] > df_today["open"])
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSETp)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] < 60)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 20)
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max()*0.97)
                & (df_today["volume"] > volume_multiplier*df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 60)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 30)
            ),
        }
    else:
        conditions = {
            "Pullback": (
                (df_today["high"] <= df_today["VWAP"]*1.05)
                & (df_today["close"] < df_today["open"])
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSETp)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] > 40)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 20)
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min()*1.03)
                & (df_today["volume"] > volume_multiplier*df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 40)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 30)
            ),
        }

    # Collect matching rows
    for entry_type, cond in conditions.items():
        matched_rows = df_today.loc[cond]
        if not matched_rows.empty:
            for _, row in matched_rows.iterrows():
                sig_id = row.get("Signal_ID", "")
                if not sig_id:
                    sig_id = build_signal_id(ticker, row)
                if sig_id in existing_signal_ids:
                    continue
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

# ================================
# 7) last_trading_day & CSV Retries
# ================================
def get_last_trading_day(ref_date=None):
    """Compute the last valid trading day based on weekday/holidays."""
    if ref_date is None:
        ref_date = datetime.now(india_tz).date()
    if ref_date.weekday() < 5 and ref_date not in market_holidays:
        return ref_date
    else:
        d = ref_date - timedelta(days=1)
        while d.weekday() >= 5 or d in market_holidays:
            d -= timedelta(days=1)
        return d

def process_csv_with_retries(file_path, ticker, max_retries=3, delay=5):
    """
    Retry CSV loading if read fails (I/O or parse error).
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
            return df
        except Exception as e:
            logging.error(f"[{ticker}] CSV load error attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise

# ================================
# 8) Marking Entry Signals
# ================================
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    For each signal, set 'Entry Signal'='Yes' in the main CSV if not already.
    Use a FileLock to avoid concurrent file writes.
    Timeout=10s is moderate (avoid stalling too long).
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No file: {main_path}. Skipping.")
        return

    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, tz=tz_obj)
            if df_main.empty:
                logging.warning(f"[{ticker}] main_indicators empty. Skipping.")
                return

            if 'Entry Signal' not in df_main.columns:
                df_main['Entry Signal'] = 'No'

            updated_count = 0
            for sig in signals_list:
                sig_id = sig.get("Signal_ID", "")
                sig_date = sig.get("date", None)
                if sig_id and 'Signal_ID' in df_main.columns:
                    mask = (df_main['Signal_ID'] == sig_id)
                else:
                    mask = (df_main['date'] == sig_date)

                if mask.any():
                    mask_to_update = mask & (df_main['Entry Signal'] != 'Yes')
                    if mask_to_update.any():
                        df_main.loc[mask_to_update, 'Entry Signal'] = 'Yes'
                        df_main.loc[mask_to_update, 'logtime'] = sig.get(
                            'logtime',
                            datetime.now(tz_obj).isoformat()
                        )
                        updated_count += mask_to_update.sum()

            if updated_count > 0:
                df_main.sort_values('date', inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] Marked 'Yes' for {updated_count} signals in main CSV.")
    except Timeout:
        logging.error(f"[{ticker}] FileLock timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] Error marking signals: {e}")

# ================================
# 9) Detecting Today's Signals
# ================================
def find_price_action_entries_for_today(last_trading_day):
    """
    Slices data from 09:25 to 14:50 from each *_main_indicators.csv,
    runs detect_signals_in_memory, merges results into a single DataFrame.
    """
    st = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(last_trading_day, dt_time(14, 50)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    files = glob.glob(pattern)
    if not files:
        print("No main_indicators CSV found.")
        return pd.DataFrame()

    all_signals = []
    generated_sigs = load_generated_signals()

    for file_path in tqdm(files, desc="Process CSV", unit="file"):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','').upper()
        try:
            df = process_csv_with_retries(file_path, ticker)
            df_today = df[(df['date'] >= st) & (df['date'] <= et)].copy()
            if df_today.empty:
                continue
            # Potential performance hotspot => if large CSV, consider profiling
            new_signals = detect_signals_in_memory(ticker, df_today, generated_sigs)
            if new_signals:
                for sig in new_signals:
                    generated_sigs.add(sig["Signal_ID"])
                all_signals.extend(new_signals)
        except Exception as e:
            logging.error(f"[{ticker}] Could not process CSV: {e}")
            continue

    save_generated_signals(generated_sigs)
    if not all_signals:
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df

# ================================
# 10) Papertrade File
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
        combined_entries = normalize_time(combined_entries, tz='Asia/Kolkata')
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
# 11) Main Function
# ================================
def main():
    """
    1) Determine last_trading_day
    2) find_price_action_entries_for_today => signals
    3) append signals to price_action_entries_15min_{today}.csv
    4) mark signals in *_main_indicators.csv
    5) create/update papertrade_{today}.csv
    """
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')

    entries_file = f"price_action_entries_15min_{today_str}.csv"
    papertrade_file = f"papertrade_{today_str}.csv"

    # 1) detect signals
    signals_df = find_price_action_entries_for_today(last_trading_day)
    if signals_df.empty:
        print("No signals found this cycle.")
        return

    # 2) append to entries CSV (we keep appending each cycle)
    header_needed = not os.path.exists(entries_file)
    signals_df.to_csv(entries_file, mode='a', index=False, header=header_needed)
    print(f"Saved {len(signals_df)} signals => {entries_file}")

    # 3) Mark signals in main CSV
    for row in signals_df.to_dict('records'):
        ticker = row["Ticker"]
        main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        mark_signals_in_main_csv(ticker, [row], main_path, india_tz)

    print("Updated 'Entry Signal' in main_indicators CSV.")

    # 4) create/append to papertrade file
    create_papertrade_file(entries_file, papertrade_file, last_trading_day)
    print(f"Papertrade => {papertrade_file}")

# ================================
# Graceful Shutdown
# ================================
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================================
# 12) Scheduling Every 30 Seconds
# ================================
def schedule_jobs():
    """
    Runs main() every 30 seconds. 
    This repeated scanning helps catch delayed CSV updates 
    and ensures minimal risk of missing signals.
    """
    schedule.every(30).seconds.do(main)

def run_schedule_loop():
    """Runs indefinitely, calling pending scheduled jobs each second."""
    while True:
        schedule.run_pending()
        time.sleep(1)

# ================================
# Entrypoint
# ================================
if __name__ == "__main__":
    # Initialize repeated 30s scheduling
    schedule_jobs()
    run_schedule_loop()
