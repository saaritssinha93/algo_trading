# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:29:18 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:29:58 2025

@author: Saarit
"""
# -*- coding: utf-8 -*-
"""
Solution B: Keep original detection data for papertrade CSV, 
only check CalcFlag='Yes' from main CSV.
"""

import os
import logging
import sys
import glob
import json
import time
import schedule
import pytz
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from filelock import FileLock, Timeout
from tqdm import tqdm
from kiteconnect import KiteConnect
from logging.handlers import TimedRotatingFileHandler
import signal


###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
# We import a list of stocks from another file
#from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks import selected_stocks

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signal1.log", when="M", interval=30,
    backupCount=5, delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging with TimedRotatingFileHandler.")
print("Script start...")

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

market_holidays = []
SIGNALS_DB = "generated_signals.json"



from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SignalFileChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        
    def on_modified(self, event):
        # Check if the modified file is a relevant CSV
        if not event.is_directory and event.src_path.endswith("_main_indicators.csv"):
            print(f"Detected change in {event.src_path}, triggering signal processing.")
            self.callback()


def start_file_observer(path_to_watch, callback):
    event_handler = SignalFileChangeHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()
    return observer


def on_signal_file_change():
    try:
        main()  # or a lighter version if needed
    except Exception as e:
        logging.error(f"Error during file change processing: {e}")


###########################################################
# 2) JSON-based Signal Tracking
###########################################################
def load_generated_signals():
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
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f, indent=2)

###########################################################
# 3) Time Normalization & CSV Loading
###########################################################
def normalize_time(df, tz='Asia/Kolkata'):
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    # If date is naive, localize to UTC first
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, tz='Asia/Kolkata'):
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
            df = df[expected_cols]
    return df

###########################################################
# 4) last_trading_day logic
###########################################################
def get_last_trading_day(ref_date=None):
    if ref_date is None:
        ref_date = datetime.now(india_tz).date()
    if ref_date.weekday() < 5 and ref_date not in market_holidays:
        return ref_date
    else:
        d = ref_date - timedelta(days=1)
        while d.weekday() >= 5 or d in market_holidays:
            d -= timedelta(days=1)
        return d

###########################################################
# 5) detect_signals_in_memory
#    Example detection (with daily_change, etc.)
###########################################################
def detect_signals_in_memory(ticker, df_today, existing_signal_ids):
    """
    Detect signals for the 'latest' (or entire) df_today.
    Relies on daily_change, ADX, RSI, MACD conditions,
    skipping duplicates found in existing_signal_ids.
    """
    signals_detected = []

    if df_today.empty:
        return signals_detected

    try:
        first_open = df_today['open'].iloc[0]
        latest_close = df_today['close'].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except Exception as e:
        logging.error(f"Error calculating daily_change for {ticker}: {e}")
        return signals_detected

    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get('ADX', 0)
    if adx_val <= 20:  # Maintained ADX threshold
        return signals_detected

    # Decide Bullish vs Bearish scenario
    if (
        daily_change > 1.0
        and latest_row.get("RSI", 0) > 30
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.02
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.0
        and latest_row.get("RSI", 0) < 70
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.98
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # Relaxed Additional Thresholds
    volume_multiplier = 1.5  # Reduced from 1.5
    rolling_window = 10     # Reduced from 20
    MACD_BULLISH_OFFSET = 0 # Slightly increased to allow more signals
    MACD_BEARISH_OFFSET = 0 # Slightly decreased to allow more signals
    MACD_BULLISH_OFFSETp = -2 # Reduced from -1.0
    MACD_BEARISH_OFFSETp = 2  # Reduced from 1.0

    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["low"] >= df_today["VWAP"] * 0.98)
                & (df_today["close"] > df_today["open"])
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSETp)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] < 60)  # Maintained or slightly adjusted
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 20)
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.98)  # Relaxed from 0.97
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 60)  # Relaxed from 60
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 40)
            ),
        }
    else:
        conditions = {
            "Pullback": (
                (df_today["high"] <= df_today["VWAP"] * 1.02)
                & (df_today["close"] < df_today["open"])
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSETp)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] > 40)  # Increased from 40
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 20)
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.02)  # Relaxed from 1.03
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 40)  # Increased from 40
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 40)
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


def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

###########################################################
# 6) Marking signals in main CSV => CalcFlag='Yes'
###########################################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    Updates 'main_indicators' CSV for a given ticker:
      - Ensures 'Signal_ID', 'Entry Signal', 'CalcFlag', 'logtime' exist.
      - Matches rows by 'Signal_ID', sets 'Entry Signal'='Yes', 'CalcFlag'='Yes'.
      - If 'logtime' is empty, fills it.
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return

    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, tz=tz_obj)
            if df_main.empty:
                logging.warning(f"[{ticker}] => main_indicators empty. Skipping.")
                return

            for col in ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]:
                if col not in df_main.columns:
                    df_main[col] = ""

            updated_count = 0

            for sig in signals_list:
                sig_id = sig.get("Signal_ID", "")
                mask = (df_main["Signal_ID"] == sig_id)
                if mask.any():
                    df_main.loc[mask, "Entry Signal"] = "Yes"
                    df_main.loc[mask, "CalcFlag"] = "Yes"
                    for idx in df_main.loc[mask].index:
                        if not df_main.at[idx, "logtime"]:
                            df_main.at[idx, "logtime"] = sig.get(
                                "logtime", datetime.now(tz_obj).isoformat()
                            )
                    updated_count += mask.sum()

            if updated_count > 0:
                df_main.sort_values("date", inplace=True)
                df_main.to_csv(main_path, index=False)
                print(f"[{ticker}] => Set CalcFlag='Yes' for {updated_count} row(s).")
            else:
                print(f"[{ticker}] => Could not find matching Signal_ID to update.")
    except Timeout:
        logging.error(f"[{ticker}] => FileLock timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] => Error marking signals: {e}")


###########################################################
# 7) find_price_action_entries_for_today
###########################################################
def find_price_action_entries_for_today(last_trading_day):
    st = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(last_trading_day, dt_time(15, 5)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    all_signals = []
    existing_ids = load_generated_signals()

    for file_path in tqdm(files, desc="Process CSV", unit="file"):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','').upper()
        df = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
        df_today = df[(df['date'] >= st) & (df['date'] <= et)].copy()
        new_signals = detect_signals_in_memory(ticker, df_today, existing_ids)
        if new_signals:
            for sig in new_signals:
                existing_ids.add(sig["Signal_ID"])
            all_signals.extend(new_signals)

    save_generated_signals(existing_ids)
    if not all_signals:
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df

###########################################################
# 8) create_papertrade_df => final CSV
###########################################################
def create_papertrade_df(df_entries_yes, output_file, last_trading_day):
    """
    Takes a DataFrame of signals (all CalcFlag='Yes'),
    filters them by intraday time, keeps earliest Ticker row,
    computes target/quantity, then appends to papertrade_*.csv.
    """
    if df_entries_yes.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return

    # 1) Normalize date, sort
    df_entries_yes = normalize_time(df_entries_yes, tz='Asia/Kolkata')
    df_entries_yes.sort_values('date', inplace=True)
    df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

    # 2) Filter [09:25 - 17:30] if you want
    start_dt = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    end_dt   = india_tz.localize(datetime.combine(last_trading_day, dt_time(15, 5)))
    mask = (df_entries_yes['date'] >= start_dt) & (df_entries_yes['date'] <= end_dt)
    df_entries_yes = df_entries_yes.loc[mask].copy()
    if df_entries_yes.empty:
        print("[Papertrade] No rows in [09:25 - 17:30].")
        return

    # 3) Keep earliest row per Ticker
    df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

    # 4) Ensure columns for 'Target Price', 'Quantity', 'Total value'
    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries_yes.columns:
            df_entries_yes[col] = ""

    # Example logic for bullish/bearish
    for idx, row in df_entries_yes.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price > 0:
            if trend == "Bullish":
                target = price * 1.01
            elif trend == "Bearish":
                target = price * 0.99
            else:
                target = price

            qty = int(20000 / price)
            total_val = qty * price

            df_entries_yes.at[idx, "Target Price"] = round(target, 2)
            df_entries_yes.at[idx, "Quantity"] = qty
            df_entries_yes.at[idx, "Total value"] = round(total_val, 2)

    # 5) If 'time' missing, fill it
    if 'time' not in df_entries_yes.columns:
        df_entries_yes['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries_yes['time'].isna()) | (df_entries_yes['time'] == "")
        if blank_mask.any():
            df_entries_yes.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')

    # 6) Append to papertrade with FileLock
    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, tz='Asia/Kolkata')
            combined = pd.concat([existing, df_entries_yes], ignore_index=True)
            # Drop duplicates if needed
            combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file}")
        else:
            df_entries_yes.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries_yes)} rows.")


###########################################################
# 9) main() => Use Original Signals + Check Only CalcFlag
###########################################################
def main():
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')
    papertrade_file = f"papertrade_{today_str}.csv"

    # 1) Detect signals => returns new signals (calcFlag="No" by default)
    raw_signals_df = find_price_action_entries_for_today(last_trading_day)
    if raw_signals_df.empty:
        print("No signals found this cycle.")
        return

    # 2) Mark signals in the main CSV right away
    for signal_row in raw_signals_df.to_dict("records"):
        ticker = signal_row["Ticker"]
        main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        # This sets 'CalcFlag'='Yes' in the CSV
        mark_signals_in_main_csv(ticker, [signal_row], main_csv_path, india_tz)

    # 3) Now that they're guaranteed to be "Yes", 
    #    directly go to papertrade (skip re-check).
    create_papertrade_df(raw_signals_df, papertrade_file, last_trading_day)
    print(f"Papertrade => {papertrade_file}")


###########################################################
# 10) Graceful Shutdown, Scheduling
###########################################################
###########################################################
# 10) Graceful Shutdown, Scheduling
###########################################################
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_cycle():
    """
    Runs the main() function every 15 seconds for 5 minutes.
    """
    logging.info("Starting process_cycle.")
    print("Starting process_cycle.")
    end_time = time.time() + 300  # 5 minutes
    while time.time() < end_time:
        try:
            main()
        except Exception as e:
            logging.error(f"Error in main during process_cycle: {e}")
            logging.error(traceback.format_exc())
        time.sleep(15)
    logging.info("Ending process_cycle.")
    print("Ending process_cycle.")

def schedule_jobs():
    """
    Schedules the process_cycle to run every 15 minutes starting at 09:30 AM.
    """
    """
    Schedules the process_cycle at every 15-minute mark from 09:30 to 15:30.
    Total scheduled times: 25 (inclusive).
    """
    start_time = datetime.strptime("09:30", "%H:%M")
    interval_minutes = 15
    total_cycles = 25  # From 09:30 to 15:30 inclusive

    for i in range(total_cycles):
        scheduled_time = (start_time + timedelta(minutes=interval_minutes * i)).time().strftime("%H:%M")
        schedule.every().day.at(scheduled_time).do(process_cycle)
        logging.info(f"Scheduled process_cycle at {scheduled_time} every day.")
        print(f"Scheduled process_cycle at {scheduled_time} every day.")

def run_schedule_loop():
    """
    Runs the scheduled jobs.
    """
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Start file observer for immediate processing on changes
    observer = start_file_observer(INDICATORS_DIR, on_signal_file_change)
    
    # Existing scheduling setup
    schedule_jobs()
    run_schedule_loop()




'''
import os
import glob
import pandas as pd
from filelock import FileLock, Timeout
import logging
import traceback

def reset_calcflag_in_main_indicators(indicators_dir="main_indicators"):
    """
    For every *_main_indicators.csv in 'indicators_dir',
    set 'CalcFlag' = 'No' for all rows, ignoring existing values.
    Uses a FileLock to avoid concurrency issues.
    """

    pattern = os.path.join(indicators_dir, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No main_indicator CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        # Create a .lock path for concurrency control
        lock_path = file_path + ".lock"
        lock = FileLock(lock_path, timeout=10)

        try:
            with lock:
                # Read CSV
                df = pd.read_csv(file_path)
                # Ensure 'CalcFlag' column exists
                if "CalcFlag" not in df.columns:
                    df["CalcFlag"] = ""
                # Set every row's CalcFlag to "No"
                df["CalcFlag"] = "No"

                # Overwrite the file
                df.to_csv(file_path, index=False)
                logging.info(f"Set CalcFlag='No' for all rows in {file_path}.")
                print(f"Set CalcFlag='No' for all rows in {os.path.basename(file_path)}.")

        except Timeout:
            logging.error(f"File lock timeout for {file_path}.")
        except Exception as e:
            logging.error(f"Error resetting CalcFlag in {file_path}: {e}")
            logging.error(traceback.format_exc())



reset_calcflag_in_main_indicators()





import os
import glob
import pandas as pd
from filelock import FileLock, Timeout
import logging
import traceback

def reset_entry_signal_in_main_indicators(indicators_dir="main_indicators"):
    """
    For every *_main_indicators.csv in 'indicators_dir',
    set 'Entry Signal' = 'No' for all rows, ignoring existing values.
    Uses a FileLock to avoid concurrency issues.
    """

    pattern = os.path.join(indicators_dir, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No main_indicator CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        # Create a .lock path for concurrency control
        lock_path = file_path + ".lock"
        lock = FileLock(lock_path, timeout=10)

        try:
            with lock:
                # Read CSV
                df = pd.read_csv(file_path)
                # Ensure 'Entry Signal' column exists
                if "Entry Signal" not in df.columns:
                    df["Entry Signal"] = ""
                # Set every row's Entry Signal to "No"
                df["Entry Signal"] = "No"

                # Overwrite the file
                df.to_csv(file_path, index=False)
                logging.info(f"Set Entry Signal='No' for all rows in {file_path}.")
                print(f"Set Entry Signal='No' for all rows in {os.path.basename(file_path)}.")

        except Timeout:
            logging.error(f"File lock timeout for {file_path}.")
        except Exception as e:
            logging.error(f"Error resetting Entry Signal in {file_path}: {e}")
            logging.error(traceback.format_exc())

reset_entry_signal_in_main_indicators()


'''
