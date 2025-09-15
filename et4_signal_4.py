# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:02 2025

@author: Saarit

Solution B: Keep original detection data for papertrade CSV, 
only check CalcFlag='Yes' from main CSV.

### CHANGES ###
1) daily_change now uses previous day's close, not today's first open.
2) We ensure only the *first* signal for each ticker goes into papertrade CSV each day.
3) The 'time' column is the actual detection time (when we write the entry).
   The scheduling frequency (e.g., every 15s for 5 min) ensures timely capture.
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
# from et4_filtered_stocks_market_cap import selected_stocks
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
SIGNALS_DB = "generated_signals4.json"

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
###########################################################
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def detect_signals_in_memory(ticker, df_today, existing_signal_ids, prev_day_close=None):
    """
    Detect signals for the 'latest' (or entire) df_today.
    daily_change is from the *previous day's close* (if provided).

    This version is relaxed to capture more signals/stocks.
    """
    signals_detected = []

    if df_today.empty:
        return signals_detected

    try:
        latest_close = df_today['close'].iloc[-1]
        # daily_change from previous day's close
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except Exception as e:
        logging.error(f"Error computing daily_change for {ticker}: {e}")
        return signals_detected

    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get('ADX', 0)
    # Relax ADX threshold from 25 -> 20 for more captures
    if adx_val <= 25:
        return signals_detected

    # Decide Bullish vs Bearish scenario
    # Relax daily_change threshold from +/-1.25% -> +/-1.0%
    # Relax RSI from 30/70 -> 20/80
    # Relax VWAP requirement from +/-2% -> +/-1%
    if (
        daily_change > 1.0
        and latest_row.get("RSI", 0) > 20
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.01
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.0
        and latest_row.get("RSI", 0) < 80
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.99
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # Extra thresholds
    # Keep them for reference, but can also be softened if desired
    macd_bullish_threshold = 50
    macd_bearish_threshold = -50
    macd_extreme_upper_limit = 50
    macd_extreme_lower_limit = -50

    # Slightly reduce volume multiplier from 1.5 -> 1.25
    volume_multiplier = 1.25
    # Shorten rolling_window from 10 -> 8
    rolling_window = 12
    # Loosen MACD offsets
    MACD_BULLISH_OFFSET = 1
    MACD_BEARISH_OFFSET = -1
    MACD_BULLISH_OFFSETp = -0.5
    MACD_BEARISH_OFFSETp = 0.5
    '''
    def is_macd_histogram_increasing(df, window=2, tolerance=-0.02):
        """Slightly smaller window to detect earlier moves."""
        if "MACD_histogram" in df.columns and len(df["MACD_histogram"]) >= window:
            recent_vals = df["MACD_histogram"].iloc[-window:].values
            return (recent_vals[-1] - recent_vals[0]) > tolerance
        return False

    def is_macd_histogram_decreasing(df, window=2, tolerance=0.02):
        """Slightly smaller window to detect earlier moves."""
        if "MACD_histogram" in df.columns and len(df["MACD_histogram"]) >= window:
            recent_vals = df["MACD_histogram"].iloc[-window:].values
            return (recent_vals[-1] - recent_vals[0]) < -tolerance
        return False
    '''
    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                # Relax close vs. open from strictly above open to (close >= open * 0.995)
                (df_today["close"] >= df_today["open"] * 0.995)
                # Relax VWAP ratio from 0.98 -> 0.97
                & (df_today["low"] >= df_today["VWAP"] * 0.98)
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSETp)
                & (df_today["close"] > df_today["20_SMA"])
                # Relax the Stochastic < 50 -> < 60
                & (df_today["Stochastic"] < 50)
                & (df_today["Stochastic"].diff() > 0)
                # ADX from 20 -> 20 is unchanged here
                & (df_today["ADX"] > 30)
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.998)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                # Loosen from 1 -> 0.5
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["MACD"] < macd_extreme_upper_limit)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 70)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 40)
                #& is_macd_histogram_increasing(df_today)
            ),
        }
    else:  # Bearish scenario
        conditions = {
            "Pullback": (
                # Relax close < open from strictly below open to (close <= open * 1.005)
                (df_today["close"] <= df_today["open"] * 1.005)
                # Relax VWAP ratio from 1.02 -> 1.03
                & (df_today["high"] <= df_today["VWAP"] * 1.02)
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSETp)
                & (df_today["close"] < df_today["20_SMA"])
                # Relax the Stochastic > 50 -> > 40
                & (df_today["Stochastic"] > 50)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 30)  # was 25
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.002)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["MACD"] > macd_extreme_lower_limit)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 30)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 40)
                #& is_macd_histogram_decreasing(df_today)
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


###########################################################
# 6) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    Updates 'main_indicators' CSV for a given ticker:
      - Sets 'Entry Signal'='Yes' and 'CalcFlag'='Yes' for all matching Signal_IDs.
    """
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

            # Ensure required columns exist
            required_cols = ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]
            for col in required_cols:
                if col not in df_main.columns:
                    df_main[col] = ""

            # Extract Signal_IDs from signals_list
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

###########################################################
# 7) find_price_action_entries_for_today
###########################################################
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_price_action_entries_for_today(last_trading_day):
    """
    Detects and collects all new trading signals for the given trading day,
    across all *_main_indicators.csv.
    """
    st = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(last_trading_day, dt_time(15, 30)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()  # To synchronize access to shared resources

    def find_prev_day_close_for_ticker(df_full, trading_day):
        """
        Find the last close from the day BEFORE `trading_day`.
        Return None if not found.
        """
        prev_day = trading_day - timedelta(days=1)
        # All rows strictly before that trading_day date
        df_before = df_full[df_full['date'].dt.date < trading_day]
        if df_before.empty:
            return None
        # We'll take the last row from the previous day
        return df_before.iloc[-1]['close']

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv', '').upper()
        df_full = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
        if df_full.empty:
            return []

        # Get prev_day_close
        prev_day_close = find_prev_day_close_for_ticker(df_full, last_trading_day)

        # Only analyze the last_trading_day portion
        df_today = df_full[(df_full['date'] >= st) & (df_full['date'] <= et)].copy()

        new_signals = detect_signals_in_memory(
            ticker,
            df_today,
            existing_signal_ids=existing_ids,
            prev_day_close=prev_day_close
        )
        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
                return new_signals
        return []

    # Use ThreadPoolExecutor for I/O-bound operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing CSVs"):
            file = future_to_file[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logging.info("No new signals detected today.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    logging.info(f"Total new signals detected: {len(signals_df)}")
    return signals_df

###########################################################
# 8) create_papertrade_df => final CSV
###########################################################
def create_papertrade_df(df_entries_yes, output_file, last_trading_day):
    """
    Takes a DataFrame of signals (all CalcFlag='Yes'),
    ensures only *first* Ticker per day is appended,
    then writes to the daily papertrade CSV with minimal duplication.
    """
    if df_entries_yes.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return

    # 1) Normalize date, sort
    df_entries_yes = normalize_time(df_entries_yes, tz='Asia/Kolkata')
    df_entries_yes.sort_values('date', inplace=True)

    # 2) Filter [09:25 - 17:30] if you want
    start_dt = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    end_dt = india_tz.localize(datetime.combine(last_trading_day, dt_time(17, 30)))
    mask = (df_entries_yes['date'] >= start_dt) & (df_entries_yes['date'] <= end_dt)
    df_entries_yes = df_entries_yes.loc[mask].copy()
    if df_entries_yes.empty:
        print("[Papertrade] No rows in [09:25 - 17:30].")
        return

    # 3) Ensure columns for 'Target Price', 'Quantity', 'Total value'
    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries_yes.columns:
            df_entries_yes[col] = ""

    # 4) Minimal example of target & quantity
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

    # 5) Set the 'time' column to now => actual insertion/detection time
    #    so you know at what clock time we appended this row.
    if 'time' not in df_entries_yes.columns:
        df_entries_yes['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries_yes['time'].isna()) | (df_entries_yes['time'] == "")
        if blank_mask.any():
            df_entries_yes.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')

    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        # 6) If file exists, read it, then only append new Tickers
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, tz='Asia/Kolkata')

            # Build a set of (Ticker, date_of_signal) already in the CSV
            # We want only 1 entry per ticker per day, earliest is kept
            existing_key_set = set()
            for _, erow in existing.iterrows():
                # day_of_signal: you can store erow['date'].date() or str
                existing_key_set.add( (erow['Ticker'], erow['date'].date()) )

            new_rows = []
            for idx, row in df_entries_yes.iterrows():
                ticker = row['Ticker']
                row_date_day = row['date'].date()
                key = (ticker, row_date_day)
                if key not in existing_key_set:
                    new_rows.append(row)

            if not new_rows:
                print("[Papertrade] No brand-new rows to append. All existing.")
                return

            new_df = pd.DataFrame(new_rows, columns=df_entries_yes.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")

        else:
            # If the file does not exist, we still only keep the earliest row per Ticker
            # from df_entries_yes in case there are duplicates in the same run
            # Keep earliest
            df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
            df_entries_yes.sort_values('date', inplace=True)
            df_entries_yes.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries_yes)} rows.")

###########################################################
# 9) main() => Use Original Signals + Check Only CalcFlag
###########################################################
def main():
    """
    Orchestrates the detection of signals, updates CalcFlag, and creates paper trade entries.
    """
    try:
        last_trading_day = get_last_trading_day()
        today_str = last_trading_day.strftime('%Y-%m-%d')
        papertrade_file = f"papertrade4_{today_str}.csv"

        # 1) Detect signals
        raw_signals_df = find_price_action_entries_for_today(last_trading_day)
        if raw_signals_df.empty:
            logging.info("No signals found this cycle.")
            return

        # 2) Mark signals in the main CSV
        signals_grouped = raw_signals_df.groupby('Ticker')
        for ticker, group in signals_grouped:
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, group.to_dict('records'), main_csv_path, india_tz)

        # 3) Append to papertrade CSV
        create_papertrade_df(raw_signals_df, papertrade_file, last_trading_day)
        logging.info(f"Papertrade entries written to {papertrade_file}.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.error(traceback.format_exc())

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
    This ensures that if a signal appears at 10:00, we pick it up
    (in normal usage) well before 10:05, so we don't miss it.
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
    Schedules the process_cycle at every 15-minute mark from 09:30 to 15:30.
    """
    start_time = datetime.strptime("09:30", "%H:%M")
    interval_minutes = 15
    total_cycles = 25  # from 09:30 to 15:30 inclusive

    for i in range(total_cycles):
        scheduled_time = (start_time + timedelta(minutes=interval_minutes * i)).time().strftime("%H:%M")
        schedule.every().day.at(scheduled_time).do(process_cycle)
        logging.info(f"Scheduled process_cycle at {scheduled_time} every day.")
        print(f"Scheduled process_cycle at {scheduled_time} every day.")

def run_schedule_loop():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_jobs()
    run_schedule_loop()
