# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:20:34 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:02 2025

@author: Saarit

Solution B (modified for multiple user-specified back-dates):
- Keep original detection data for papertrade CSV, only check CalcFlag='Yes' from main CSV.
- daily_change uses previous day's close as the reference.
- Only the *first* signal per ticker goes into papertrade CSV each day.
- 'time' column is the actual detection time.
- Allows you to run for a list of chosen dates (past or present) in one go.
"""

import os
import logging
import sys
import glob
import json
import pytz
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from filelock import FileLock, Timeout
from tqdm import tqdm
from logging.handlers import TimedRotatingFileHandler
import signal

###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks import selected_stocks  # Example import for your stock list

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators_historical"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signal1.log",
    when="M",
    interval=30,
    backupCount=5,
    delay=True
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

# If needed, change to your working directory:
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

# Removed any holiday/weekend logic; no more market_holidays or get_last_trading_day
SIGNALS_DB = "generated_signals.json"

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
# 4) detect_signals_in_memory
#    With the relaxed breakout/breakdown conditions
###########################################################
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def is_bullish_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    curr_open = df["open"]
    curr_close = df["close"]
    return (
        (curr_close > curr_open) &
        (prev_close < prev_open) &
        (curr_open < prev_close) &
        (curr_close > prev_open)
    )

def is_bearish_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    curr_open = df["open"]
    curr_close = df["close"]
    return (
        (curr_close < curr_open) &
        (prev_close > prev_open) &
        (curr_open > prev_close) &
        (curr_close < prev_open)
    )

def robust_prev_day_close_for_ticker(ticker, date_of_interest):
    """Load entire CSV for ticker, find previous day's final bar, return that close."""
    main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker.lower()}_main_indicators_historical.csv")
    if not os.path.exists(main_csv_path):
        return None

    try:
        df_full = load_and_normalize_csv(main_csv_path, tz='Asia/Kolkata')
        if df_full.empty:
            return None

        day_date = date_of_interest.date()
        df_before = df_full[df_full['date'].dt.date < day_date]
        if df_before.empty:
            return None
        last_close = df_before.iloc[-1]['close']
        return float(last_close)

    except Exception as e:
        logging.error(f"{ticker} => robust_prev_day_close_for_ticker failed: {e}")
        return None

def detect_signals_in_memory(
    ticker,
    df_for_rolling,
    df_for_detection,
    existing_signal_ids,
    prev_day_close=None
):
    """
    1) df_for_rolling: a bigger dataset (e.g. last 20+ bars) so rolling(10) is accurate
    2) df_for_detection: only the final portion (last 4 bars) for new signals
    3) We do rolling computations (like rolling high/low) on df_for_rolling
       but only "detect signals" in the last 4 bars.
    """
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    # (A) Handling prev_day_close if not supplied
    final_bar_date = df_for_detection['date'].iloc[-1]
    if not prev_day_close:
        prev_day_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

    # (B) Combine frames for rolling context
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)

    # (C) daily_change based on final detection bar
    try:
        latest_close = df_for_detection['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except:
        daily_change = 0.0

    combined = combined.copy()

    # (D) Precompute Price Action columns
    combined["Bullish_Engulfing"] = is_bullish_engulfing(combined)
    combined["Bearish_Engulfing"] = is_bearish_engulfing(combined)

    combined["Bullish_PriceAction"] = (
        (combined["high"] > combined["high"].shift(1) * 1.002)
        & (combined["low"] > combined["low"].shift(1) * 1.002)
        & ((combined["close"] - combined["open"]) > 0.001 * combined["open"])
        & ((combined["high"] - combined["close"]) < 0.2 * (combined["high"] - combined["low"]))
    )
    combined["Bearish_PriceAction"] = (
        (combined["high"] < combined["high"].shift(1) * 0.998)
        & (combined["low"] < combined["low"].shift(1) * 0.998)
        & ((combined["open"] - combined["close"]) > 0.001 * combined["open"])
        & ((combined["close"] - combined["low"]) < 0.2 * (combined["high"] - combined["low"]))
    )

    # Pullback requires referencing shift(1); handle if cidx==0 or missing
    combined["Bullish_PullbackPriceAction"] = (
        (combined["close"] > combined["open"])
        & (combined["low"] >= combined["20_SMA"] * 0.99)
        & (combined["low"] <= combined["20_SMA"] * 1.01)
    )
    combined["Bearish_PullbackPriceAction"] = (
        (combined["close"] < combined["open"])
        & (combined["high"] <= combined["20_SMA"] * 1.01)
        & (combined["high"] >= combined["20_SMA"] * 0.99)
    )

    # (E) Rolling columns (with min_periods=10 to avoid NaN)
    rolling_window = 10
    combined["rolling_high_10"] = (
        combined["high"].rolling(rolling_window, min_periods=rolling_window).max()
    )
    combined["rolling_low_10"]  = (
        combined["low"].rolling(rolling_window, min_periods=rolling_window).min()
    )
    combined["rolling_vol_10"]  = (
        combined["volume"].rolling(rolling_window, min_periods=rolling_window).mean()
    )

    # (F) Stochastic difference
    combined["StochDiff"] = combined["Stochastic"].diff()

    combined.reset_index(drop=True, inplace=True)

    # (G) MACD crossing logic
    def is_macd_above(row_idx):
        if row_idx < 0 or row_idx >= len(combined):
            return False
        macd_val    = combined.loc[row_idx, "MACD"]
        signal_val  = combined.loc[row_idx, "Signal_Line"]
        if pd.isna(macd_val) or pd.isna(signal_val):
            return False
        return macd_val > signal_val

    def macd_cross_up(row_idx):
        if row_idx <= 0 or row_idx >= len(combined):
            return False
        macd_now    = combined.loc[row_idx, "MACD"]
        signal_now  = combined.loc[row_idx, "Signal_Line"]
        macd_prev   = combined.loc[row_idx-1, "MACD"]
        signal_prev = combined.loc[row_idx-1, "Signal_Line"]
        if (
            pd.isna(macd_now) or pd.isna(signal_now)
            or pd.isna(macd_prev) or pd.isna(signal_prev)
        ):
            return False
        return (macd_prev <= signal_prev) and (macd_now > signal_now)

    def get_trend_type(row):
        rsi_val = row.get("RSI", 0)
        cls_val = row.get("close", 0)
        vwap_val = row.get("VWAP", 0)

        if (rsi_val > 20) and (cls_val > vwap_val * 1.01):
            return "Bullish"
        elif (rsi_val < 80) and (cls_val < vwap_val * 0.99):
            return "Bearish"
        return None

    def valid_adx(val):
        if pd.isna(val):
            return False
        return (val > 25)

    volume_multiplier = 1.5
    macd_extreme_upper_limit = 50
    macd_extreme_lower_limit = -50
    MACD_BULLISH_OFFSET = 3
    MACD_BULLISH_OFFSETp = 0
    MACD_BEARISH_OFFSET = -3
    MACD_BEARISH_OFFSETp = 0

    # (H) Loop over detection bars only
    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect["date"]
        match = combined[combined["date"] == dt]
        if match.empty:
            continue
        cidx = match.index[0]

        adx_val = combined.loc[cidx, "ADX"]
        if not valid_adx(adx_val):
            continue

        row_trend = get_trend_type(combined.loc[cidx])
        if not row_trend:
            continue

        # Pullback checks
        bullish_pullback_check = False
        if cidx > 0:
            if (
                not pd.isna(combined.loc[cidx-1, "close"])
                and not pd.isna(combined.loc[cidx-1, "open"])
                and (combined.loc[cidx-1, "close"] < combined.loc[cidx-1, "open"])
            ):
                bullish_pullback_check = True

        bearish_pullback_check = False
        if cidx > 0:
            if (
                not pd.isna(combined.loc[cidx-1, "close"])
                and not pd.isna(combined.loc[cidx-1, "open"])
                and (combined.loc[cidx-1, "close"] > combined.loc[cidx-1, "open"])
            ):
                bearish_pullback_check = True

        macd_above_signal_now = is_macd_above(cidx)

        bullish_conditions = {
            "Pullback": (
                (combined.loc[cidx, "close"] >= combined.loc[cidx, "open"])
                and (daily_change > 0.5)
                and (combined.loc[cidx, "low"] >= combined.loc[cidx, "VWAP"] * 0.98)
                and macd_above_signal_now
                and (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSETp)
                and (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] < 45)
                and (combined.loc[cidx, "StochDiff"] > 0)
                and (adx_val > 25)
                and bullish_pullback_check
            ),
            "Breakout": (
                (combined.loc[cidx, "close"] > (combined.loc[cidx, "rolling_high_10"] * 0.995))
                and (10 > daily_change > 1.5)
                and (combined.loc[cidx, "volume"] > volume_multiplier * combined.loc[cidx, "rolling_vol_10"])
                and macd_above_signal_now
                and (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSET)
                and (combined.loc[cidx, "MACD"] < macd_extreme_upper_limit)
                and (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] > 70)
                and (combined.loc[cidx, "StochDiff"] > 0)
                and (adx_val > 35 and adx_val < 45)
                and (combined.loc[cidx, "Bullish_PriceAction"])
            ),
        }

        bearish_conditions = {
            "Pullback": (
                (combined.loc[cidx, "close"] <= combined.loc[cidx, "open"])
                and (daily_change < -0.5)
                and (combined.loc[cidx, "high"] <= combined.loc[cidx, "VWAP"] * 1.02)
                and not macd_above_signal_now
                and (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSETp)
                and (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] > 55)
                and (combined.loc[cidx, "StochDiff"] < 0)
                and (adx_val > 25)
                and bearish_pullback_check
            ),
            "Breakdown": (
                (combined.loc[cidx, "close"] < (combined.loc[cidx, "rolling_low_10"] * 1.005))
                and (-10 < daily_change < -1.5)
                and (combined.loc[cidx, "volume"] > volume_multiplier * combined.loc[cidx, "rolling_vol_10"])
                and not macd_above_signal_now
                and (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSET)
                and (combined.loc[cidx, "MACD"] > macd_extreme_lower_limit)
                and (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] < 30)
                and (combined.loc[cidx, "StochDiff"] < 0)
                and (adx_val > 35 and adx_val < 45)
                and (combined.loc[cidx, "Bearish_PriceAction"])
            ),
        }

        chosen_dict = bullish_conditions if (row_trend == "Bullish") else bearish_conditions

        for entry_type, condition_bool in chosen_dict.items():
            if condition_bool:
                sig_id = row_detect.get("Signal_ID")
                if not sig_id:
                    dt_str = row_detect["date"].isoformat()
                    sig_id = f"{ticker}-{dt_str}"
                if sig_id in existing_signal_ids:
                    continue

                signals_detected.append({
                    "Ticker": ticker,
                    "date": row_detect["date"],
                    "Entry Type": entry_type,
                    "Trend Type": row_trend,
                    "Price": round(row_detect.get("close", 0), 2),
                    "Daily Change %": round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    "ADX": round(adx_val, 2),
                    "MACD": row_detect.get("MACD", None),
                    "Signal_ID": sig_id,
                    "logtime": row_detect.get("logtime", datetime.now(india_tz).isoformat()),
                    "Entry Signal": "Yes"
                })

    return signals_detected

###########################################################
# 5) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    Updates 'main_indicators_historical' CSV for a given ticker:
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
                logging.warning(f"[{ticker}] => main_indicators_historical empty. Skipping.")
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
# 6) find_price_action_entries_for_date
###########################################################
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_price_action_entries_for_date(target_date):
    """
    Detects and collects all new trading signals for the given date,
    across all *_main_indicators_historical.csv.
    Intraday timeframe considered is 09:25 to 14:05 (example).
    """
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(14, 46)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators_historical.csv')
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def find_prev_day_close_for_ticker(df_full, trading_day):
        """
        Find the last close from the day BEFORE `trading_day`.
        Return None if not found.
        """
        prev_day = trading_day - timedelta(days=1)
        df_before = df_full[df_full['date'].dt.date < trading_day]
        if df_before.empty:
            return None
        return df_before.iloc[-1]['close']

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace('_main_indicators_historical.csv', '').upper()
        df_full = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
        if df_full.empty:
            return []

        # Get prev_day_close
        prev_day_close = find_prev_day_close_for_ticker(df_full, target_date)

        # Only analyze the portion for our target_date
        df_today = df_full[(df_full['date'] >= st) & (df_full['date'] <= et)].copy()

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,   # entire dataset for correct rolling(10)
            df_for_detection=df_today,# final portion for new signals
            existing_signal_ids=existing_ids,
            prev_day_close=prev_day_close
        )

        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
                return new_signals
        return []

    # ThreadPoolExecutor for parallel reading & detection
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Processing {target_date}"):
            file = future_to_file[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logging.info(f"No new signals detected for {target_date}.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    logging.info(f"Total new signals detected for {target_date}: {len(signals_df)}")
    return signals_df

###########################################################
# 7) create_papertrade_df => final CSV
###########################################################
def create_papertrade_df(df_entries_yes, output_file, target_date):
    """
    Takes a DataFrame of signals, ensures only *first* Ticker per day is appended,
    then writes to the daily papertrade CSV with minimal duplication.
    """
    if df_entries_yes.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return

    df_entries_yes = normalize_time(df_entries_yes, tz='Asia/Kolkata')
    df_entries_yes.sort_values('date', inplace=True)

    # Example intraday timeframe for final filter
    start_dt = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    end_dt = india_tz.localize(datetime.combine(target_date, dt_time(17, 30)))
    mask = (df_entries_yes['date'] >= start_dt) & (df_entries_yes['date'] <= end_dt)
    df_entries_yes = df_entries_yes.loc[mask].copy()
    if df_entries_yes.empty:
        print("[Papertrade] No rows in [09:25 - 17:30].")
        return

    # Ensure columns for 'Target Price', 'Quantity', 'Total value'
    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries_yes.columns:
            df_entries_yes[col] = ""

    # Simple example for target & quantity
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

    # Set the 'time' column to the current detection time if missing
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

            existing_key_set = set()
            for _, erow in existing.iterrows():
                existing_key_set.add((erow['Ticker'], erow['date'].date()))

            new_rows = []
            for _, row in df_entries_yes.iterrows():
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
            combined.drop_duplicates(subset=['Ticker', 'date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")
        else:
            df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
            df_entries_yes.sort_values('date', inplace=True)
            df_entries_yes.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries_yes)} rows.")

###########################################################
# 8) run_for_date(date_str) -- no weekend/holiday skip
###########################################################
def run_for_date(date_str):
    """
    Runs the detection & CSV updates for one specific date (YYYY-MM-DD),
    with no weekend/holiday skipping.
    """
    try:
        # Convert user-specified string to a date
        requested_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # Force EXACT date instead of skipping weekends/holidays
        last_trading_day = requested_date

        date_label = last_trading_day.strftime('%Y-%m-%d')
        papertrade_file = f"papertrade_{date_label}.csv"

        # 1) Detect signals
        raw_signals_df = find_price_action_entries_for_date(last_trading_day)
        if raw_signals_df.empty:
            logging.info(f"No signals found for {date_label}.")
            return

        # 2) Mark signals in main CSV
        signals_grouped = raw_signals_df.groupby('Ticker')
        for ticker, group in signals_grouped:
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_historical.csv")
            mark_signals_in_main_csv(ticker, group.to_dict('records'), main_csv_path, india_tz)

        # 3) Append to papertrade CSV
        create_papertrade_df(raw_signals_df, papertrade_file, last_trading_day)
        logging.info(f"Papertrade entries written to {papertrade_file}.")

    except Exception as e:
        logging.error(f"Error in run_for_date({date_str}): {e}")
        logging.error(traceback.format_exc())

###########################################################
# 9) Graceful Shutdown Handler
###########################################################
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

###########################################################
# Entry Point (Multiple Dates)
###########################################################
if __name__ == "__main__":
    # If multiple dates are provided on the command line, we'll iterate over them.
    # Example usage:
    #   python script.py 2025-01-14 2025-01-15 2025-01-16
    date_args = sys.argv[1:]
    if not date_args:
        # If no args given, define a list of back-dates here:
        date_args = ["2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01"]
        #date_args = ["2025-02-01"]

    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
