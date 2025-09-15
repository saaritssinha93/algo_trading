# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 06:31:56 2025

@author: Saarit

Script to read only the *last 4 rows* from each *_main_indicators.csv
(but after 9:30 AM candle on that day), detect signals,
mark CalcFlag='Yes', and immediately update a papertrade CSV
for quick entries, while still computing rolling(10) over a
bigger data subset (including older bars) to avoid missing breakouts.
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
import signal
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from filelock import FileLock, Timeout
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import TimedRotatingFileHandler

###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
INDICATORS_DIR = "main_indicators"
SIGNALS_DB = "generated_signals.json"
market_holidays = []  # If needed, populate

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signal6.log", when="M", interval=30, backupCount=5, delay=True
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

###########################################################
# 2) JSON-based Signal Tracking
###########################################################
def load_generated_signals():
    """Load any previously saved signal IDs from the JSON file."""
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
    """Save updated signal IDs to the JSON file."""
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f, indent=2)

###########################################################
# 3) Time Normalization & CSV Loading
###########################################################
def normalize_time(df, tz='Asia/Kolkata'):
    """Convert 'date' column to DateTime in a consistent timezone."""
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    if df['date'].dt.tz is None:
        # If no timezone, assume UTC
        df['date'] = df['date'].dt.tz_localize('UTC')

    df['date'] = df['date'].dt.tz_convert(tz)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, tz='Asia/Kolkata'):
    """Load CSV and normalize 'date' column to Asia/Kolkata."""
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
    """Returns the most recent trading day on/before ref_date."""
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
    main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker.lower()}_main_indicators.csv")
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

    # ------------------------------------------------------
    # (A) Handling prev_day_close if not supplied
    # ------------------------------------------------------
    final_bar_date = df_for_detection['date'].iloc[-1]
    if not prev_day_close:
        prev_day_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

    # ------------------------------------------------------
    # (B) Combine frames for rolling context
    # ------------------------------------------------------
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)

    # ------------------------------------------------------
    # (C) daily_change based on final detection bar
    # ------------------------------------------------------
    try:
        latest_close = df_for_detection['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except:
        daily_change = 0.0

    # Make a copy to avoid SettingWithCopy issues
    combined = combined.copy()

    # ------------------------------------------------------
    # (D) Precompute Price Action columns
    # ------------------------------------------------------
    combined["Bullish_Engulfing"] = is_bullish_engulfing(combined)
    combined["Bearish_Engulfing"] = is_bearish_engulfing(combined)

    combined["Bullish_PriceAction"] = (
        (combined["high"] > combined["high"].shift(1) * 1.0015)
        & (combined["low"] > combined["low"].shift(1) * 1.0015)
        & ((combined["close"] - combined["open"]) > 0.001 * combined["open"])
        & ((combined["high"] - combined["close"]) < 0.1 * (combined["high"] - combined["low"]))
    )
    combined["Bearish_PriceAction"] = (
        (combined["high"] < combined["high"].shift(1) * 0.9985)
        & (combined["low"] < combined["low"].shift(1) * 0.9985)
        & ((combined["open"] - combined["close"]) > 0.001 * combined["open"])
        & ((combined["close"] - combined["low"]) < 0.1 * (combined["high"] - combined["low"]))
    )

    # Pullback requires referencing shift(1); handle if cidx==0 or missing
    combined["Bullish_PullbackPriceAction"] = (
        (combined["close"] > combined["open"])
        & (combined["low"] >= combined["20_SMA"] * 0.99)
        & (combined["low"] <= combined["20_SMA"] * 1.01)
        # We’ll later check if cidx>0 to ensure shift(1) is valid
    )
    combined["Bearish_PullbackPriceAction"] = (
        (combined["close"] < combined["open"])
        & (combined["high"] <= combined["20_SMA"] * 1.01)
        & (combined["high"] >= combined["20_SMA"] * 0.99)
        # Also handle cidx>0
    )

    # ------------------------------------------------------
    # (E) Rolling columns (with min_periods=10 to avoid NaN)
    # ------------------------------------------------------
    rolling_window = 10
    # If fewer than 10 bars exist at that point, we get NaN => handle or skip
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

    # ------------------------------------------------------
    # (G) MACD crossing logic - define small helper if needed
    # ------------------------------------------------------
    def is_macd_above(row_idx):
        """Check if MACD is above Signal line at row_idx."""
        # If row_idx < 0 or row_idx >= len(combined), handle safely
        if row_idx < 0 or row_idx >= len(combined):
            return False
        macd_val    = combined.loc[row_idx, "MACD"]
        signal_val  = combined.loc[row_idx, "Signal_Line"]
        if pd.isna(macd_val) or pd.isna(signal_val):
            return False
        return macd_val > signal_val

    def macd_cross_up(row_idx):
        """
        True if MACD *just* crossed above the signal line:
        e.g. was below or equal last bar, now above.
        """
        if row_idx <= 0 or row_idx >= len(combined):
            return False
        macd_now    = combined.loc[row_idx, "MACD"]
        signal_now  = combined.loc[row_idx, "Signal_Line"]
        macd_prev   = combined.loc[row_idx-1, "MACD"]
        signal_prev = combined.loc[row_idx-1, "Signal_Line"]
        if pd.isna(macd_now) or pd.isna(signal_now) or pd.isna(macd_prev) or pd.isna(signal_prev):
            return False
        return (macd_prev <= signal_prev) and (macd_now > signal_now)

    # Helper: determine "Bullish" vs. "Bearish" from RSI & VWAP
    def get_trend_type(row):
        # If RSI or close or VWAP are missing => safe fallback
        rsi_val = row.get("RSI", 0)
        cls_val = row.get("close", 0)
        vwap_val = row.get("VWAP", 0)

        if (rsi_val > 20) and (cls_val > vwap_val * 1.01):
            return "Bullish"
        elif (rsi_val < 80) and (cls_val < vwap_val * 0.99):
            return "Bearish"
        return None

    # Tweak ADX gating or add skip if ADX is NaN
    def valid_adx(val):
        # If ADX is missing => skip
        if pd.isna(val):
            return False
        # If you want to skip if ADX < 25 => trending logic
        return (val > 25)

    volume_multiplier = 1.5
    macd_extreme_upper_limit = 50
    macd_extreme_lower_limit = -50
    MACD_BULLISH_OFFSET = 3      # e.g. want MACD > 3
    MACD_BULLISH_OFFSETp = 0     # e.g. want MACD > 0
    MACD_BEARISH_OFFSET = -3
    MACD_BEARISH_OFFSETp = 0

    # ------------------------------------------------------
    # (H) Loop over detection bars only
    # ------------------------------------------------------
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

        # For the "Pullback" shift(1) checks, ensure cidx>0
        # e.g. 'Bullish_PullbackPriceAction' => we only confirm it if cidx>0 and no NaN
        bullish_pullback_check = False
        if cidx > 0 and not pd.isna(combined.loc[cidx-1, "close"]) and not pd.isna(combined.loc[cidx-1, "open"]):
            # Now we can read the (combined["close"].shift(1) < combined["open"].shift(1)) part
            if combined.loc[cidx-1, "close"] < combined.loc[cidx-1, "open"]:
                # Then 'Bullish_PullbackPriceAction' condition is satisfied from a "previous bar was bearish" perspective
                bullish_pullback_check = True

        # Similar for bearish
        bearish_pullback_check = False
        if cidx > 0 and not pd.isna(combined.loc[cidx-1, "close"]) and not pd.isna(combined.loc[cidx-1, "open"]):
            if combined.loc[cidx-1, "close"] > combined.loc[cidx-1, "open"]:
                bearish_pullback_check = True

        # MACD conditions (just an example: we might want crossing or just above)
        macd_above_signal_now = is_macd_above(cidx)
        # or crossing_up_now = macd_cross_up(cidx)

        # Build condition dict
        # Use daily_change from earlier
        # For "Pullback", we rely on the 'bullish_pullback_check' we set
        # For "Breakout", we rely on e.g. macd_above_signal_now or crossing_up_now
        bullish_conditions = {
            "Pullback": (
                (combined.loc[cidx, "close"] >= combined.loc[cidx, "open"])
                and (daily_change > 0.5)
                and (combined.loc[cidx, "low"] >= combined.loc[cidx, "VWAP"] * 0.98)
                and macd_above_signal_now
                and (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSETp)
                and (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] < 60)
                and (combined.loc[cidx, "StochDiff"] > 0)  # Stochastic rising
                and bullish_pullback_check
                # ADX validated above
            ),
            "Breakout": (
                (combined.loc[cidx, "close"] > (combined.loc[cidx, "rolling_high_10"] * 0.995))
                and (10 > daily_change > 1.5)
                and (combined.loc[cidx, "volume"] > volume_multiplier * combined.loc[cidx, "rolling_vol_10"])
                and macd_above_signal_now
                and (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSET)
                and (combined.loc[cidx, "MACD"] < macd_extreme_upper_limit)
                and (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] > 60)
                and (combined.loc[cidx, "StochDiff"] > 0)
                and (adx_val > 30 and adx_val < 45)
                and (combined.loc[cidx, "Bullish_PriceAction"])
            ),
        }

        bearish_conditions = {
            "Pullback": (
                (combined.loc[cidx, "close"] <= combined.loc[cidx, "open"])
                and (daily_change < -0.5)
                and (combined.loc[cidx, "high"] <= combined.loc[cidx, "VWAP"] * 1.02)
                and not macd_above_signal_now  # or check crossing below
                and (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSETp)
                and (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] > 40)
                and (combined.loc[cidx, "StochDiff"] < 0)
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
                and (combined.loc[cidx, "Stochastic"] < 40)
                and (combined.loc[cidx, "StochDiff"] < 0)
                and (adx_val > 30 and adx_val < 45)
                and (combined.loc[cidx, "Bearish_PriceAction"])
            ),
        }

        chosen_dict = bullish_conditions if (row_trend == "Bullish") else bearish_conditions

        for entry_type, condition_bool in chosen_dict.items():
            if condition_bool:
                sig_id = row_detect.get("Signal_ID")
                if not sig_id:
                    # Generate new one
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
# 6) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
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
# 7) Immediate append to papertrade for each ticker
###########################################################
def append_signals_to_papertrade(signals_list, output_file, trading_day):
    if not signals_list:
        return

    df_entries = pd.DataFrame(signals_list)
    if df_entries.empty:
        return

    df_entries = normalize_time(df_entries, tz='Asia/Kolkata')
    df_entries.sort_values('date', inplace=True)

    start_dt = india_tz.localize(datetime.combine(trading_day, datetime.min.time()))
    end_dt = india_tz.localize(datetime.combine(trading_day, datetime.max.time()))
    mask = (df_entries['date'] >= start_dt) & (df_entries['date'] <= end_dt)
    df_entries = df_entries.loc[mask].copy()
    if df_entries.empty:
        return

    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries.columns:
            df_entries[col] = ""

    for idx, row in df_entries.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price > 0:
            if trend == "Bullish":
                target = price * 1.01
            elif trend == "Bearish":
                target = price * 0.99
            else:
                target = price
            qty = int(20000 / price) if price != 0 else 1
            total_val = qty * price

            df_entries.at[idx, "Target Price"] = round(target, 2)
            df_entries.at[idx, "Quantity"] = qty
            df_entries.at[idx, "Total value"] = round(total_val, 2)

    if 'time' not in df_entries.columns:
        df_entries['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries['time'].isna()) | (df_entries['time'] == "")
        if blank_mask.any():
            df_entries.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')

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
            for _, row in df_entries.iterrows():
                key = (row['Ticker'], row['date'].date())
                if key not in existing_key_set:
                    new_rows.append(row)

            if not new_rows:
                return

            new_df = pd.DataFrame(new_rows, columns=df_entries.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")

        else:
            df_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
            df_entries.sort_values('date', inplace=True)
            df_entries.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries)} rows.")

###########################################################
# 8) find_price_action_entries_for_today
###########################################################
def find_price_action_entries_for_today(trading_day, papertrade_file):
    """
    1. Load full CSV (df_full)
    2. Keep only rows after 9:30 AM => that is 'df_today'
    3. df_for_rolling => last 20 or 30 bars from the entire df_full
    4. df_for_detection => last 4 bars from df_today
    5. pass BOTH to detect_signals_in_memory
    6. append to papertrade CSV, mark signals in main CSV
    """
    start_dt = india_tz.localize(datetime.combine(trading_day, dt_time(9, 30)))
    now_ist = datetime.now(india_tz)
    if now_ist.date() == trading_day:
        end_dt = now_ist
    else:
        end_dt = india_tz.localize(datetime.combine(trading_day, dt_time(15, 30)))

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    analysis_info = {}
    lock = threading.Lock()

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()
        main_csv_path = file_path

        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df_full.empty:
            with lock:
                analysis_info[ticker] = {'rows_after_930': 0, 'rows_used': 0, 'reason': "File empty"}
            return []

        # 1) Filter only rows from 9:30 to end_dt => "df_today"
        df_today = df_full[(df_full["date"] >= start_dt) & (df_full["date"] <= end_dt)].copy()
        rows_after_930 = len(df_today)
        if df_today.empty:
            with lock:
                analysis_info[ticker] = {'rows_after_930': 0, 'rows_used': 0, 'reason': "No data after 9:30"}
            return []

        # 2) We want the last 20-30 bars from the entire df_full => "df_for_rolling"
        df_for_rolling = df_full.tail(30).copy()

        # 3) last 4 bars from df_today => detection
        df_for_detection = df_today.tail(4).copy()
        rows_used = len(df_for_detection)

        # 4) detect signals => pass BOTH DataFrames
        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_for_rolling,
            df_for_detection=df_for_detection,  # <-- THE KEY FIX
            existing_signal_ids=existing_ids
        )

        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])

            # Optionally mark signals in main CSV
            # mark_signals_in_main_csv(ticker, new_signals, main_csv_path, india_tz)

            # Immediately append to papertrade
            append_signals_to_papertrade(new_signals, papertrade_file, trading_day)

        with lock:
            analysis_info[ticker] = {
                'rows_after_930': rows_after_930,
                'rows_used': rows_used,
                'reason': "" if new_signals else "No new signals"
            }

        return new_signals

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing CSVs"):
            file = future_to_file[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    save_generated_signals(existing_ids)

    print("\n=== ANALYSIS SUMMARY ===")
    for t in sorted(analysis_info.keys()):
        info = analysis_info[t]
        if info['rows_used'] == 0:
            print(f"[NOT ANALYZED] {t} => reason: {info['reason']}")
        else:
            print(f"[ANALYZED] {t} => used last {info['rows_used']} of {info['rows_after_930']} rows after 9:30, reason: {info['reason']}")

    if not all_signals:
        logging.info("No new signals detected.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values("date", inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    logging.info(f"Total new signals detected: {len(signals_df)}")
    return signals_df

###########################################################
# 9) main()
###########################################################
def main():
    """Runs the signal detection & immediate papertrade updates once."""
    try:
        last_trading_day = get_last_trading_day()
        today_str = last_trading_day.strftime('%Y-%m-%d')
        papertrade_file = f"papertrade_{today_str}.csv"

        raw_signals_df = find_price_action_entries_for_today(last_trading_day, papertrade_file)
        if raw_signals_df.empty:
            logging.info("No signals found this cycle.")
            return

        logging.info("All signals processed & appended immediately.")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.error(traceback.format_exc())

###########################################################
# 10) Continuous run every 20 seconds
###########################################################
def signal_handler(sig, frame):
    print("\nInterrupt received, shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_continuous_every_20s():
    while True:
        try:
            main()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            logging.error(traceback.format_exc())
        time.sleep(20)

if __name__ == "__main__":
    run_continuous_every_20s()
