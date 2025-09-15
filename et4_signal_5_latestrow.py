# -*- coding: utf-8 -*-
"""
Filename: code2_revised.py

Created on Wed Jan  8 15:02:02 2025
@author: Saarit

Solution B with improvements:
1) Introduce a CONFIG dictionary for concurrency, intervals, file paths, etc.
2) Use more granular logging levels (DEBUG, INFO, WARNING, ERROR).
3) Provide a safe_invoke decorator for consistent try/except logging.
4) Use parameterized concurrency from CONFIG (ThreadPoolExecutor max_workers).
5) Minor naming, scheduling, and consistency improvements.

Core logic remains intact: 
 - We detect signals from main CSV files,
 - We mark them as CalcFlag='Yes',
 - We create papertrade entries for the first (earliest) signal of each Ticker/day.

"""

import os
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
import logging
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

###########################################################
# 0) CONFIGURATION & SETUP
###########################################################

# You can move these to a config file or environment variables if desired
CONFIG = {
    "CONCURRENCY_LEVEL": 4,    # e.g., how many threads for parallel processing
    "INTERVAL_MINUTES": 15,
    "TOTAL_CYCLES": 25,        # from 09:30 to 15:30 inclusive
    "POLL_INTERVAL_SEC": 15,   # how often to call main() within process_cycle
    "POLL_DURATION_SEC": 300,  # total = 5 minutes
    "CONCURRENCY_LEVEL": 4,
    "LOG_LEVEL": "INFO",
    # We'll just keep these for reference, but won't do the 15-min schedule:
    "MARKET_START": "09:30",
    "MARKET_END": "15:30",
    "RUN_SLEEP_SEC": 15  # how often we call main() in continuous mode
}

india_tz = pytz.timezone("Asia/Kolkata")

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Example holiday list
market_holidays = []

# JSON-based tracking of generated signals
SIGNALS_DB = "generated_signals.json"

# If you prefer, import your stock list from a config file or a module
from et4_filtered_stocks import selected_stocks

###########################################################
# Logging Configuration
###########################################################
logger = logging.getLogger(__name__)
logger.setLevel(CONFIG["LOG_LEVEL"].upper())

# Create a rotating file handler
file_handler = TimedRotatingFileHandler(
    "logs\\signal_latestrow_1.log", when="M", interval=30,
    backupCount=5, delay=True
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create a console handler (optional)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(file_formatter)

# Attach handlers if not present
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Initialized logging with level=%s", CONFIG["LOG_LEVEL"])
print("Script start with improved logging and concurrency...")

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info("Working directory changed to %s", cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

###########################################################
# 0.1) A Safe-Invoke Decorator for Error Handling
###########################################################
def safe_invoke(func):
    """
    Decorator to wrap function calls in a try/except,
    log exceptions consistently, and optionally re-raise.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.error("Exception in %s: %s", func.__name__, str(exc))
            logger.debug(traceback.format_exc())
            # re-raise or return None; here we choose to re-raise:
            raise
    return wrapper

###########################################################
# 1) JSON-based Signal Tracking
###########################################################
def load_generated_signals():
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logger.error("Signals DB JSON is corrupted. Starting with empty set.")
            return set()
    else:
        return set()

@safe_invoke
def save_generated_signals(generated_signals):
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f, indent=2)
    logger.info("Saved generated signals to %s with %d entries.", SIGNALS_DB, len(generated_signals))

###########################################################
# 2) Time Normalization & CSV Loading
###########################################################
@safe_invoke
def normalize_time(df, tz='Asia/Kolkata'):
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    return df

@safe_invoke
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
# 3) last_trading_day logic
###########################################################
@safe_invoke
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


def get_latest_15min_interval(now=None, tz=india_tz):
    """
    Returns the most recent 15-minute floored datetime in the given timezone.
    E.g., if now=10:05, returns 10:00 exactly.
    """
    if now is None:
        now = datetime.now(tz)
    else:
        # ensure it's timezone-aware
        if now.tzinfo is None:
            now = tz.localize(now)
        else:
            now = now.astimezone(tz)

    minute_block = (now.minute // 15) * 15
    floored = now.replace(minute=minute_block, second=0, microsecond=0)
    return floored

###########################################################
# 4) Signal Detection
###########################################################
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

@safe_invoke
def detect_signals_in_memory(ticker, df_today, existing_signal_ids, prev_day_close=None):
    """
    Relaxed detection:
      - daily_change from previous day close
      - ADX > 20
      - RSI checks
      - Price vs. VWAP checks
      - 'Bullish' if daily_change > +1.0%, 'Bearish' if < -1.0%
    """
    signals_detected = []
    if df_today.empty:
        return signals_detected

    # 1) Compute daily_change
    try:
        latest_close = df_today['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except Exception as e:
        logger.error(f"[{ticker}] Error computing daily_change: {e}")
        return signals_detected

    # 2) Fetch latest row, check ADX
    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get('ADX', 0)

    # Relax the ADX threshold from 25 down to 20
    if adx_val <= 20:
        return signals_detected

    # 3) Basic Trend Detection
    # Lower the daily_change threshold from +/-1.25% to +/-1.0% 
    # (so we get more signals)
    if (
        daily_change > 1.0
        and latest_row.get("RSI", 0) > 30
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.01
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.0
        and latest_row.get("RSI", 0) < 70
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.99
    ):
        trend_type = "Bearish"
    else:
        # If it doesn't fit either environment, skip
        return signals_detected

    # 4) Additional thresholds, relaxed
    # Previously:
    # macd_bullish_threshold = 50  # etc.
    # We can keep them if you want advanced checks. 
    # Or skip them entirely for a simpler approach.
    volume_multiplier = 1.2      # relaxed from 1.5
    rolling_window = 5           # smaller window than 10
    bullish_offset = 0.5         # was 1
    bearish_offset = -0.5        # was -1
    stoch_bullish_limit = 60     # was 50
    stoch_bearish_limit = 40     # was 50

    # 5) Define convenience check for MACD histogram direction
    def is_macd_histogram_increasing(df, window=2, tolerance=0.0):
        if "MACD_histogram" in df.columns and len(df["MACD_histogram"]) >= window:
            recent_vals = df["MACD_histogram"].iloc[-window:].values
            return (recent_vals[-1] - recent_vals[0]) > tolerance
        return False

    def is_macd_histogram_decreasing(df, window=2, tolerance=0.0):
        if "MACD_histogram" in df.columns and len(df["MACD_histogram"]) >= window:
            recent_vals = df["MACD_histogram"].iloc[-window:].values
            return (recent_vals[-1] - recent_vals[0]) < -tolerance
        return False

    # 6) Build scenario conditions
    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["close"] > df_today["open"])              # price rising
                & (df_today["MACD"] > df_today["Signal_Line"])      
                & (df_today["MACD"] > bullish_offset) 
                & (df_today["Stochastic"] < stoch_bullish_limit)
                & (df_today["Stochastic"].diff() > 0)
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.99)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > bullish_offset)
                & is_macd_histogram_increasing(df_today, window=2, tolerance=0.01)
            ),
        }
    else:  # Bearish
        conditions = {
            "Pullback": (
                (df_today["close"] < df_today["open"])             # price falling
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < bearish_offset)
                & (df_today["Stochastic"] > stoch_bearish_limit)
                & (df_today["Stochastic"].diff() < 0)
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.01)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < bearish_offset)
                & is_macd_histogram_decreasing(df_today, window=2, tolerance=0.01)
            ),
        }

    # 7) Check which conditions pass
    for entry_label, cond in conditions.items():
        matched_rows = df_today.loc[cond]
        if not matched_rows.empty:
            # We found at least one row that meets the condition
            for _, row in matched_rows.iterrows():
                sig_id = row.get("Signal_ID", "")
                if not sig_id:
                    sig_id = build_signal_id(ticker, row)

                # Only record new signals if not seen before
                if sig_id not in existing_signal_ids:
                    signals_detected.append({
                        "Ticker": ticker,
                        "date": row["date"],
                        "Entry Type": entry_label,
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
# 5) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
@safe_invoke
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    if not os.path.exists(main_path):
        logger.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return

    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)

    with lock:
        df_main = load_and_normalize_csv(main_path, expected_cols=["Signal_ID"], tz='Asia/Kolkata')
        if df_main.empty:
            logger.warning(f"[{ticker}] => main_indicators empty. Skipping.")
            return

        required_cols = ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]
        for col in required_cols:
            if col not in df_main.columns:
                df_main[col] = ""

        # Extract Signal_IDs
        signal_ids = [sig["Signal_ID"] for sig in signals_list if "Signal_ID" in sig]
        if not signal_ids:
            logger.info(f"[{ticker}] => No valid Signal_IDs to update.")
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
            logger.info(f"[{ticker}] => Set CalcFlag='Yes' for {updated_count} row(s).")
        else:
            logger.info(f"[{ticker}] => No matching Signal_IDs found to update.")

###########################################################
# 6) find_price_action_entries_for_today
###########################################################
def find_price_action_entries_for_today():
    """
    Detect signals only for the current 15-minute timestamp
    across all *_main_indicators.csv.
    """
    # 1) Determine the most recent 15-min timestamp
    current_15min = get_latest_15min_interval()  # e.g., 10:00, 10:15, etc.

    # Optionally, you can log or print this
    logger.info(f"[find_price_action_entries_for_today] Checking for candle at {current_15min} ...")
    
    # --- NEW LINE: Print the interval to console ---
    print(f"Analyzing 15-min interval: {current_15min.strftime('%H:%M')}")

    # 2) Locate main_indicators CSV files
    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    files = glob.glob(pattern)
    if not files:
        logger.info("No indicator CSV files found.")
        return pd.DataFrame()

    # 3) Load existing signals from SIGNALS_DB to avoid duplicates
    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def find_prev_day_close_for_ticker(df_full, reference_day):
        """
        Same as before: returns the last close from *before* reference_day.
        If none found, returns None.
        """
        df_before = df_full[df_full['date'].dt.date < reference_day]
        if df_before.empty:
            return None
        return df_before.iloc[-1]['close']

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv', '').upper()
        df_full = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
        if df_full.empty:
            return []

        # 4) For 'prev_day_close', assume "today" is current_15min.date()
        #    (You could also use get_last_trading_day() if you prefer.)
        prev_close = find_prev_day_close_for_ticker(df_full, current_15min.date())

        # 5) Filter the DataFrame to only the row(s) for exactly the current 15-min stamp
        #    i.e. those rows whose 'date' == current_15min
        df_current_bar = df_full[df_full['date'] == current_15min].copy()
        if df_current_bar.empty:
            # No row for that exact time => no signals
            logger.debug(f"[{ticker}] No row for {current_15min}. Skipping.")
            return []

        # 6) Detect signals in that single 15-min bar
        new_signals = detect_signals_in_memory(
            ticker,
            df_current_bar,
            existing_signal_ids=existing_ids,
            prev_day_close=prev_close
        )
        if new_signals:
            # Acquire lock so we can safely update existing_ids from multiple threads
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
            return new_signals
        return []

    # 7) Parallel processing
    with ThreadPoolExecutor(max_workers=CONFIG["CONCURRENCY_LEVEL"]) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing CSVs"):
            file_path = future_to_file[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                logger.debug(traceback.format_exc())

    # 8) Save updated existing_ids to SIGNALS_DB
    save_generated_signals(existing_ids)

    if not all_signals:
        logger.info(f"No new signals detected for candle {current_15min}.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    logger.info(f"Total new signals detected for {current_15min}: {len(signals_df)}")
    return signals_df


###########################################################
# 7) create_papertrade_df => final CSV
###########################################################
@safe_invoke
def create_papertrade_df(df_entries_yes, output_file, last_trading_day):
    if df_entries_yes.empty:
        logger.info("[Papertrade] DataFrame is empty => no rows to add.")
        return

    # 1) Normalize and sort
    df_entries_yes = normalize_time(df_entries_yes, tz='Asia/Kolkata')
    df_entries_yes.sort_values('date', inplace=True)

    # 2) Filter [9:25 - 17:30]
    start_dt = india_tz.localize(datetime.combine(last_trading_day, dt_time(9, 25)))
    end_dt = india_tz.localize(datetime.combine(last_trading_day, dt_time(17, 30)))
    mask = (df_entries_yes['date'] >= start_dt) & (df_entries_yes['date'] <= end_dt)
    df_entries_yes = df_entries_yes.loc[mask].copy()
    if df_entries_yes.empty:
        logger.info("[Papertrade] No rows in [09:25 - 17:30].")
        return

    # 3) Compute Target Price, Quantity, Total Value
    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries_yes.columns:
            df_entries_yes[col] = ""

    for idx, row in df_entries_yes.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price <= 0:
            continue
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

    # 4) Set 'time' column to now if missing
    if 'time' not in df_entries_yes.columns:
        df_entries_yes['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries_yes['time'].isna()) | (df_entries_yes['time'] == "")
        if blank_mask.any():
            df_entries_yes.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')

    # 5) In-memory de-dup so only earliest row per (Ticker, Day) in this run
    df_entries_yes['DayPart'] = df_entries_yes['date'].dt.date
    df_entries_yes.drop_duplicates(subset=['Ticker', 'DayPart'], keep='first', inplace=True)
    df_entries_yes.drop(columns=['DayPart'], inplace=True)

    # 6) Now append only truly new (Ticker, day) rows to papertrade CSV
    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, tz='Asia/Kolkata')

            # Build a set of (Ticker, day) from existing
            existing_key_set = set()
            for _, erow in existing.iterrows():
                existing_key_set.add((erow['Ticker'], erow['date'].date()))

            # Collect new rows that are not in existing
            new_rows = []
            for idx, row in df_entries_yes.iterrows():
                day_key = (row['Ticker'], row['date'].date())
                if day_key not in existing_key_set:
                    new_rows.append(row)
                    # optionally add to existing_key_set here if you want 
                    # to skip further duplicates within the same run

            if not new_rows:
                logger.info("[Papertrade] No new unique ticker/day signals to append.")
                return

            new_df = pd.DataFrame(new_rows, columns=df_entries_yes.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)

            # *Optionally*, if you want to ensure no duplicates by date-time,
            # you can do something like:
            # combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)

            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            logger.info(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")

        else:
            # If there's no existing file, we still keep earliest row per Ticker
            df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
            df_entries_yes.sort_values('date', inplace=True)
            df_entries_yes.to_csv(output_file, index=False)
            logger.info(f"[Papertrade] Created => {output_file} with {len(df_entries_yes)} rows.")


###########################################################
# 8) Main Orchestration
###########################################################
@safe_invoke
@safe_invoke
def main():
    """
    1) Pull signals for the single latest 15-min candle from each CSV
    2) Mark them in main CSV
    3) Append to papertrade CSV
    """
    # If you prefer a last_trading_day for naming output files, keep it:
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')
    papertrade_file = f"papertrade_{today_str}.csv"

    # 1) We no longer pass 'last_trading_day' to the function, because
    #    it now uses the actual system time to find the current 15min bar.
    raw_signals_df = find_price_action_entries_for_today()
    if raw_signals_df.empty:
        logger.info("No signals found this cycle.")
        return

    # 2) Mark signals in the main CSV
    signals_grouped = raw_signals_df.groupby('Ticker')
    for ticker, group in signals_grouped:
        main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        mark_signals_in_main_csv(ticker, group.to_dict('records'), main_csv_path, india_tz)

    # 3) Append to papertrade CSV
    create_papertrade_df(raw_signals_df, papertrade_file, last_trading_day)
    logger.info(f"Papertrade entries written/updated to {papertrade_file}.")


###########################################################
# 9) Graceful Shutdown, Scheduling
###########################################################
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@safe_invoke
def run_intraday_cycle():
    """
    Runs main() every POLL_INTERVAL_SEC for POLL_DURATION_SEC
    """
    logger.info("Starting run_intraday_cycle.")
    end_time = time.time() + CONFIG["POLL_DURATION_SEC"]
    while time.time() < end_time:
        main()
        time.sleep(CONFIG["POLL_INTERVAL_SEC"])
    logger.info("Ending run_intraday_cycle.")

@safe_invoke
def schedule_jobs():
    """
    Schedules run_intraday_cycle() at every 15-min block
    from MARKET_START to MARKET_END.
    """
    start_dt = datetime.strptime(CONFIG["MARKET_START"], "%H:%M")
    interval_minutes = CONFIG["INTERVAL_MINUTES"]
    total_cycles = CONFIG["TOTAL_CYCLES"]

    for i in range(total_cycles):
        scheduled_time = (start_dt + timedelta(minutes=interval_minutes * i)).time().strftime("%H:%M")
        schedule.every().day.at(scheduled_time).do(run_intraday_cycle)
        logger.info(f"Scheduled run_intraday_cycle at {scheduled_time} daily.")
        print(f"Scheduled run_intraday_cycle at {scheduled_time} daily.")

def run_schedule_loop():
    while True:
        schedule.run_pending()
        time.sleep(1)

###########################################################
# 10) Continuous Run from 9:30 to 15:30 every 15 seconds
###########################################################
def run_continuous():
    """
    Continuously runs `main()` every 15 seconds from 9:30 to 15:30 local time.
    """
    logger.info("Starting continuous run...")
    print("Starting continuous run...")

    tz = india_tz
    now = datetime.now(tz)
    today = now.date()

    # 1) Build today's 9:30 start
    market_start = tz.localize(datetime.combine(today, dt_time(hour=9, minute=30)))
    # 2) Build today's 15:30 end
    market_end = tz.localize(datetime.combine(today, dt_time(hour=15, minute=30)))

    # If we start earlier than 9:30, sleep until 9:30
    if now < market_start:
        sleep_seconds = (market_start - now).total_seconds()
        logger.info(f"Sleeping until 9:30. {sleep_seconds} seconds left.")
        time.sleep(sleep_seconds)

    # Now run main() every 15 seconds until 15:30
    while True:
        now = datetime.now(tz)
        if now > market_end:
            logger.info("Reached 15:30, exiting run.")
            print("Reached 15:30, stopping continuous run.")
            break

        # Call the main routine
        main()

        # Sleep 15 seconds
        time.sleep(CONFIG["RUN_SLEEP_SEC"])

    logger.info("Continuous run ended.")

###########################################################
# Script Entry
###########################################################
if __name__ == "__main__":
    run_continuous()
