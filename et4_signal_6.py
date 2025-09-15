# -*- coding: utf-8 -*-
"""
Event-Driven, Row-by-Row Change Tracking Example

Watch 'main_indicators' folder for changes to *_main_indicators.csv.
When changed, compare old vs. new row-by-row; process only changed/new rows.

@author: Saarit (modified example)
"""

import os
import logging
import sys
import glob
import json
import time
import pytz
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from filelock import FileLock, Timeout
from kiteconnect import KiteConnect
from logging.handlers import TimedRotatingFileHandler
import signal

# For Watchdog file event
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

##############################################
# Global Config
##############################################
india_tz = pytz.timezone('Asia/Kolkata')
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

# Store the "previous snapshot" of each ticker's DF so we can do row-by-row comparisons
previous_data = {}  # dict[ticker] = DataFrame

##############################################
# JSON-based Signal Tracking
##############################################
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

##############################################
# Time Normalization & CSV Loading
##############################################
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

##############################################
# Trading Day Logic
##############################################
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

##############################################
# Row-by-Row Comparison Logic
##############################################
def get_changed_or_new_rows(old_df, new_df):
    """
    Returns the rows from new_df that are new or changed
    compared to old_df (based on all columns).
    
    We'll assume 'date' is a unique key and possibly also 'Signal_ID'.
    Adjust as needed for your data structure.
    """
    if old_df.empty:
        # If there's no old data, everything in new_df is "new"
        return new_df.copy()

    # Merge on 'date' and (optionally) other columns that form a unique row ID
    # so we can detect new or changed rows. We'll do a left merge from new_df to old_df.
    # We'll suffix old columns with '_old' for comparison.
    merge_on = ['date']  # or also add 'Signal_ID' if that is a unique key
    common_cols = list(set(old_df.columns).intersection(set(new_df.columns)))

    # We'll do a left merge: keep all rows from new_df
    merged = pd.merge(
        new_df, 
        old_df[common_cols], 
        on=merge_on, 
        how='left', 
        suffixes=('', '_old')
    )

    changed_rows = []
    for idx, row in merged.iterrows():
        # If 'Signal_ID_old' is NaN, that means it's a brand new row
        if pd.isna(row.get('Signal_ID_old', None)) and 'Signal_ID' in row:
            changed_rows.append(idx)
        else:
            # Check if any columns differ
            # We'll compare each col in common_cols
            for col in common_cols:
                old_col = f"{col}_old"
                if old_col in merged.columns:
                    new_val = row[col]
                    old_val = row[old_col]
                    # If they differ (and ignoring minor float diffs if needed), then it's changed
                    if pd.isna(old_val) and not pd.isna(new_val):
                        changed_rows.append(idx)
                        break
                    if not pd.isna(old_val) and not pd.isna(new_val):
                        if old_val != new_val:
                            changed_rows.append(idx)
                            break
                    # If one is NaN but not the other, that's also a difference
                    if (pd.isna(old_val) and not pd.isna(new_val)) or (not pd.isna(old_val) and pd.isna(new_val)):
                        changed_rows.append(idx)
                        break

    if not changed_rows:
        return pd.DataFrame()

    diff_df = merged.loc[changed_rows, new_df.columns]  # keep only the "new" columns
    diff_df.reset_index(drop=True, inplace=True)
    return diff_df

##############################################
# Signal Detection (unchanged)
##############################################
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def detect_signals_in_memory(ticker, df_today, existing_signal_ids):
    """
    Original detection logic, runs on the subset DF you pass in.
    If you pass only changed rows, it will only check those.
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
    if adx_val <= 25:  
        return signals_detected

    # Decide Bullish vs Bearish
    if (
        daily_change > 1.25
        and latest_row.get("RSI", 0) > 30
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.02
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.25
        and latest_row.get("RSI", 0) < 70
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.98
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # Example thresholds
    macd_extreme_upper_limit = 50
    macd_extreme_lower_limit = -50
    volume_multiplier = 1.5
    rolling_window = 10
    MACD_BULLISH_OFFSET = 1
    MACD_BEARISH_OFFSET = -1
    MACD_BULLISH_OFFSETp = 0
    MACD_BEARISH_OFFSETp = 0

    def is_macd_histogram_increasing(df, window=3, tolerance=0.0):
        if "MACD_histogram" in df.columns and len(df["MACD_histogram"]) >= window:
            recent_vals = df["MACD_histogram"].iloc[-window:].values
            return (recent_vals[-1] - recent_vals[0]) > tolerance
        return False

    def is_macd_histogram_decreasing(df, window=3, tolerance=0.0):
        if "MACD_histogram" in df.columns and len(df["MACD_histogram"]) >= window:
            recent_vals = df["MACD_histogram"].iloc[-window:].values
            return (recent_vals[-1] - recent_vals[0]) < -tolerance
        return False

    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["low"] >= df_today["VWAP"] * 0.98)
                & (df_today["close"] > df_today["open"])
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSETp)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] < 50)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 25)
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.99)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["MACD"] < macd_extreme_upper_limit)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 50)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 35)
                & is_macd_histogram_increasing(df_today)
            ),
        }
    else:  # Bearish
        conditions = {
            "Pullback": (
                (df_today["high"] <= df_today["VWAP"] * 1.02)
                & (df_today["close"] < df_today["open"])
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSETp)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] > 50)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 25)
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.01)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["MACD"] > macd_extreme_lower_limit)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 50)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 35)
                & is_macd_histogram_decreasing(df_today)
            ),
        }

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

##############################################
# Marking signals => CalcFlag='Yes'
##############################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return

    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, expected_cols=["Signal_ID"], tz='Asia/Kolkata')
            if df_main.empty:
                logging.warning(f"[{ticker}] => main_indicators empty. Skipping.")
                return

            required_cols = ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]
            for col in required_cols:
                if col not in df_main.columns:
                    df_main[col] = ""

            # Extract Signal_IDs
            signal_ids = [sig["Signal_ID"] for sig in signals_list if "Signal_ID" in sig]
            if not signal_ids:
                logging.info(f"[{ticker}] => No valid Signal_IDs to update.")
                return

            mask = df_main["Signal_ID"].isin(signal_ids)
            rows_to_update = df_main[mask]
            if not rows_to_update.empty:
                df_main.loc[mask, ["Entry Signal", "CalcFlag"]] = "Yes"
                current_time_iso = datetime.now(tz_obj).isoformat()
                df_main.loc[mask & (df_main["logtime"] == ""), "logtime"] = current_time_iso

                df_main.sort_values("date", inplace=True)
                df_main.to_csv(main_path, index=False)
                updated_count = mask.sum()
                logging.info(f"[{ticker}] => Set CalcFlag='Yes' for {updated_count} row(s).")
            else:
                logging.info(f"[{ticker}] => No matching Signal_IDs found to update.")
    except Timeout:
        logging.error(f"[{ticker}] => FileLock timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] => Error marking signals: {e}")

##############################################
# Papertrade DF logic
##############################################
def create_papertrade_df(df_entries_yes, output_file, last_trading_day):
    if df_entries_yes.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return

    df_entries_yes = normalize_time(df_entries_yes, tz='Asia/Kolkata')
    df_entries_yes.sort_values('date', inplace=True)
    df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

    start_dt = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    end_dt   = india_tz.localize(datetime.combine(last_trading_day, dt_time(14, 5)))
    mask = (df_entries_yes['date'] >= start_dt) & (df_entries_yes['date'] <= end_dt)
    df_entries_yes = df_entries_yes.loc[mask].copy()
    if df_entries_yes.empty:
        print("[Papertrade] No rows in [09:25 - 17:30].")
        return

    df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries_yes.columns:
            df_entries_yes[col] = ""

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

    if 'time' not in df_entries_yes.columns:
        df_entries_yes['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries_yes['time'].isna()) | (df_entries_yes['time'] == "")
        if blank_mask.any():
            df_entries_yes.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')

    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, tz='Asia/Kolkata')
            combined = pd.concat([existing, df_entries_yes], ignore_index=True)
            combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file}")
        else:
            df_entries_yes.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries_yes)} rows.")

##############################################
# main()
##############################################
def main_for_ticker(ticker, new_df):
    """
    This function is called whenever we detect row-by-row changes
    in that ticker's CSV. We run detection only on the changed rows.
    """
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')
    papertrade_file = f"papertrade_{today_str}.csv"

    # Existing signals
    existing_ids = load_generated_signals()

    # 1) Detect signals for changed rows
    new_signals = detect_signals_in_memory(ticker, new_df, existing_ids)
    if not new_signals:
        logging.info(f"[{ticker}] => No new signals in changed rows.")
        return

    # Update the JSON so we don't repeat signals
    for sig in new_signals:
        existing_ids.add(sig["Signal_ID"])
    save_generated_signals(existing_ids)

    # 2) Mark signals in main CSV
    main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    mark_signals_in_main_csv(ticker, new_signals, main_csv_path, india_tz)

    # 3) Papertrade
    signals_df = pd.DataFrame(new_signals)
    create_papertrade_df(signals_df, papertrade_file, last_trading_day)
    logging.info(f"[{ticker}] => Processed {len(new_signals)} new signals.")


##############################################
# Watchdog + Row-by-Row Code
##############################################
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class CSVUpdateHandler(FileSystemEventHandler):
    """
    Watchdog handler that detects CSV changes,
    compares new vs. old row-by-row, and calls main_for_ticker(...)
    only with changed rows.
    """
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith("_main_indicators.csv"):
            logging.info(f"Detected change in: {event.src_path}")
            self.process_csv_change(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith("_main_indicators.csv"):
            logging.info(f"Detected new file: {event.src_path}")
            self.process_csv_change(event.src_path)

    def process_csv_change(self, file_path):
        # figure out ticker from file name
        ticker = os.path.basename(file_path).replace('_main_indicators.csv', '').upper()

        # load new DF
        new_df = load_and_normalize_csv(file_path, tz='Asia/Kolkata')

        if ticker not in previous_data:
            # no old snapshot => entire new_df is "new"
            logging.info(f"[{ticker}] => No old snapshot. Processing all rows.")
            changed_df = new_df
        else:
            # compare with old snapshot
            old_df = previous_data[ticker]
            changed_df = get_changed_or_new_rows(old_df, new_df)
            logging.info(f"[{ticker}] => Detected {len(changed_df)} changed/new rows.")

        if not changed_df.empty:
            # run detection on changed_df
            main_for_ticker(ticker, changed_df)

        # update snapshot
        previous_data[ticker] = new_df.copy()


def run_event_driven_loop():
    """
    Start the watchdog observer on INDICATORS_DIR.
    We rely on row-by-row difference logic whenever
    files are created or modified.
    """
    event_handler = CSVUpdateHandler()
    observer = Observer()
    observer.schedule(event_handler, INDICATORS_DIR, recursive=False)
    observer.start()
    print(f"Watching for CSV changes in: {INDICATORS_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run_event_driven_loop()
