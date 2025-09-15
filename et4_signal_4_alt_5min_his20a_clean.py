# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:02 2025

Author: Saarit

Clean version:
- Detects intraday signals using Daily_Change + Intra_Change streak logic.
- Uses session VWAP (resets daily) and a few momentum/participation checks.
- Writes all detected signals per day to ENTRIES_DIR and appends the first signal
  per ticker to the papertrade CSV for that date.
- Threaded scanning across *_main_indicators.csv files.

Assumptions on input CSV columns:
- Must include: date, open, high, low, close, volume
- Also expected: Daily_Change, Stoch_%K, Stoch_%D, ADX, RSI
- Optional inputs: Intra_Streak_Bars, Intra_Threshold, Intra_Threshold_L, Intra_Threshold_H
"""

import os
import sys
import glob
import json
import signal
import pytz
import logging
import traceback
import threading
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
from tqdm import tqdm
from filelock import FileLock, Timeout
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------
# 1) Configuration & setup
# ---------------------------------------------
INDICATORS_DIR = "main_indicators_5min"
SIGNALS_DB     = "generated_signals_historical5minv7_relaxed.json"
ENTRIES_DIR    = "main_indicators_history_entries_5minv7_relaxed"  # where day's detected signals are saved
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(ENTRIES_DIR, exist_ok=True)

india_tz = pytz.timezone("Asia/Kolkata")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = TimedRotatingFileHandler(
    "logs\\signalnewn5min3.log", when="M", interval=30, backupCount=5, delay=True
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

# Optional: change working directory if needed
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")


# ---------------------------------------------
# 2) Small utilities: signal DB, time parsing, CSV loading
# ---------------------------------------------
def load_generated_signals() -> set:
    """Load the set of Signal_IDs we've already produced, to avoid duplicates."""
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, "r") as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB JSON is corrupted. Starting with empty set.")
    return set()


def save_generated_signals(generated_signals: set) -> None:
    """Persist the set of Signal_IDs."""
    with open(SIGNALS_DB, "w") as f:
        json.dump(list(generated_signals), f, indent=2)


def normalize_time(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """
    Ensure 'date' is timezone-aware in IST and sorted ascending.
    If naive, assume it's already local (IST) and localize directly.
    """
    d = df.copy()
    if "date" not in d.columns:
        raise KeyError("DataFrame missing 'date' column.")

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d.dropna(subset=["date"], inplace=True)

    if d["date"].dt.tz is None:
        d["date"] = d["date"].dt.tz_localize(tz)
    else:
        d["date"] = d["date"].dt.tz_convert(tz)

    d.sort_values("date", inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


def load_and_normalize_csv(file_path: str, expected_cols=None, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Read CSV; if it has 'date', normalize to IST; else ensure expected columns exist."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df = normalize_time(df, tz)
    else:
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[expected_cols]
    return df


# ---------------------------------------------
# 3) Indicators (only what's needed): Session VWAP
# ---------------------------------------------
def calculate_session_vwap(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vol_col: str = "volume",
    tz: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """
    Compute session VWAP (resets each trading day in IST).
    Safe if 'date' is already tz-aware; we just group by date(). 
    """
    d = df.copy()
    dt = pd.to_datetime(d["date"], errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(tz)
    else:
        dt = dt.dt.tz_convert(tz)

    d["_session"] = dt.dt.date
    tp = (d[high_col] + d[low_col] + d[price_col]) / 3.0
    tp_vol = tp * d[vol_col]

    d["_cum_tp_vol"] = tp_vol.groupby(d["_session"]).cumsum()
    d["_cum_vol"] = d[vol_col].groupby(d["_session"]).cumsum()
    d["VWAP"] = d["_cum_tp_vol"] / d["_cum_vol"]

    d.drop(columns=["_session", "_cum_tp_vol", "_cum_vol"], inplace=True)
    return d


# ---------------------------------------------
# 4) Signal detection
# ---------------------------------------------
def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set,
) -> list:
    """
    Detect signals using daily move + intraday streak conditions with momentum & VWAP checks.

    Bullish (example):
      - Daily_Change > +2%
      - 'n' consecutive bars each with Intra_Change above threshold (or within a positive band)
      - ADX trending up (last 2 bars), ADX moderately strong
      - RSI rising (last 3 bars), RSI in [50, 75)
      - Stoch_%D > 50 and rising; Stoch bull cross
      - Price > VWAP by ~0.5%
      - Volume >= 2.5x rolling(10) volume

    Bearish is symmetric with signs inverted.
    """

    signals = []
    if df_for_detection.empty:
        return signals

    DEFAULT_STREAK = int(globals().get("INTRA_STREAK_BARS", 1))
    DEFAULT_THRESH = float(globals().get("INTRA_THRESHOLD", 0.0))

    # ---- Combine, sort, and build session key (IST) ----
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined.dropna(subset=["date"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    dt_series = combined["date"]
    try:
        ist = (
            dt_series.dt.tz_localize("Asia/Kolkata")
            if getattr(dt_series.dt, "tz", None) is None
            else dt_series.dt.tz_convert("Asia/Kolkata")
        )
    except Exception:
        ist = pd.to_datetime(dt_series, errors="coerce").dt.tz_localize("Asia/Kolkata")
    combined["_date_only"] = ist.dt.date

    # ---- Ensure VWAP, and minimal rolling stats we actually use ----
    if "VWAP" not in combined.columns:
        combined = calculate_session_vwap(combined)

    combined["rolling_vol_10"] = combined["volume"].rolling(10, min_periods=10).mean()
    combined["StochDiff"] = combined["Stoch_%D"].diff()

    # ---- Build Intra_Change if missing (intraday % change within the same IST session) ----
    if ("Intra_Change" not in combined.columns) or combined["Intra_Change"].isna().all():
        combined["Intra_Change"] = combined.groupby("_date_only")["close"].pct_change().mul(100.0)
    intra_by_idx = combined["Intra_Change"]

    # ---- Helpers used in conditions ----
    def find_cidx(ts):
        """Find index of the bar at or immediately prior to timestamp 'ts' (IST), same session."""
        t = pd.Timestamp(ts)
        # Align ts to IST and to combined['date'] tz
        if (combined["date"].dt.tz is not None):
            t = t.tz_localize("Asia/Kolkata") if t.tzinfo is None else t.tz_convert("Asia/Kolkata")
        else:
            t = t.tz_localize("Asia/Kolkata") if t.tzinfo is None else t.tz_convert("Asia/Kolkata")
            t = t.tz_localize(None)

        exact = combined.index[combined["date"] == t]
        if len(exact):
            return int(exact[0])

        day = t.date()
        day_mask = combined["_date_only"] == day
        if not day_mask.any():
            return None
        prior = combined.index[day_mask & (combined["date"] <= t)]
        if len(prior):
            return int(prior[-1])
        return int(combined.index[day_mask][0])

    def is_adx_increasing(df_, cix, bars_ago=1) -> bool:
        """Strictly increasing ADX over the last 'bars_ago'+1 samples."""
        si = cix - bars_ago
        if si < 0:
            return False
        vals = df_.loc[si:cix, "ADX"].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        return all(vals[j] < vals[j + 1] for j in range(len(vals) - 1))

    def is_rsi_increasing(df_, cix, bars_ago=1) -> bool:
        si = cix - bars_ago
        if si < 0:
            return False
        vals = df_.loc[si:cix, "RSI"].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        return all(vals[j] < vals[j + 1] for j in range(len(vals) - 1))

    def is_rsi_decreasing(df_, cix, bars_ago=1) -> bool:
        si = cix - bars_ago
        if si < 0:
            return False
        vals = df_.loc[si:cix, "RSI"].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        return all(vals[j] > vals[j + 1] for j in range(len(vals) - 1))

    def has_intra_streak(cidx: int, n: int, thresh, bullish: bool) -> bool:
        """
        Check that the last 'n' bars (including cidx) meet either:
          - single threshold 't':  bull: > +t, bear: < -t
          - band (low, high):     bull: in [+low, +high], bear: in [-high, -low]
        Evaluated only within the same session day.
        """
        if n <= 0 or cidx is None or cidx - (n - 1) < 0:
            return False

        start = cidx - (n - 1)
        cur_day = combined["_date_only"].iloc[cidx]
        window_days = combined["_date_only"].iloc[start : cidx + 1]
        if window_days.nunique() != 1 or window_days.iloc[0] != cur_day:
            return False

        seg = intra_by_idx.iloc[start : cidx + 1]
        if seg.isna().any():
            return False

        # Parse threshold(s)
        try:
            if isinstance(thresh, (tuple, list)) and len(thresh) == 2:
                lo = float(thresh[0])
                hi = float(thresh[1])
                lo, hi = sorted((abs(lo), abs(hi)))
                if bullish:
                    return ((seg >= lo) & (seg <= hi)).all()
                return ((seg <= -lo) & (seg >= -hi)).all()
            else:
                t = abs(float(thresh))
                return (seg > t).all() if bullish else (seg < -t).all()
        except Exception:
            return False

    # Simple stochastic crosses with tolerance (default = 1 point)
    STOCH_TOL = float(globals().get("STOCH_TOL", 1.0))

    def stoch_bull_cross(df_, idx, tol=STOCH_TOL) -> bool:
        if idx < 1:
            return False
        k0, d0 = df_.at[idx, "Stoch_%K"], df_.at[idx, "Stoch_%D"]
        k1 = df_.at[idx - 1, "Stoch_%K"]
        if any(pd.isna(x) for x in (k0, d0, k1)):
            return False
        return (k0 >= d0 - tol) and (k0 > k1)

    def stoch_bear_cross(df_, idx, tol=STOCH_TOL) -> bool:
        if idx < 1:
            return False
        k0, d0 = df_.at[idx, "Stoch_%K"], df_.at[idx, "Stoch_%D"]
        k1 = df_.at[idx - 1, "Stoch_%K"]
        if any(pd.isna(x) for x in (k0, d0, k1)):
            return False
        return (k0 <= d0 + tol) and (k0 < k1)

    # ---- Evaluate each row in today's detection slice ----
    for _, row_detect in df_for_detection.iterrows():
        dt = row_detect.get("date", None)
        if pd.isna(dt):
            continue

        daily_change = row_detect.get("Daily_Change", np.nan)
        if pd.isna(daily_change):
            continue

        # streak bars (N)
        n = row_detect.get("Intra_Streak_Bars", DEFAULT_STREAK)
        try:
            n = int(n) if pd.notna(n) else DEFAULT_STREAK
        except Exception:
            n = DEFAULT_STREAK

        # threshold or band from row → globals → default
        thresh = None
        tL, tH = row_detect.get("Intra_Threshold_L", None), row_detect.get("Intra_Threshold_H", None)
        if tL is not None and tH is not None and not (pd.isna(tL) or pd.isna(tH)):
            try:
                thresh = (float(tL), float(tH))
            except Exception:
                thresh = None
        if thresh is None:
            gL, gH = globals().get("INTRA_THRESHOLD_L", None), globals().get("INTRA_THRESHOLD_H", None)
            if gL is not None and gH is not None:
                try:
                    thresh = (float(gL), float(gH))
                except Exception:
                    thresh = None
        if thresh is None:
            tv = row_detect.get("Intra_Threshold", None)
            try:
                thresh = float(tv) if (tv is not None and not pd.isna(tv)) else DEFAULT_THRESH
            except Exception:
                thresh = DEFAULT_THRESH

        cidx = find_cidx(dt)
        if cidx is None:
            continue

        # Read current-bar features
        price     = combined.loc[cidx, "close"]
        vol       = combined.loc[cidx, "volume"]
        roll_vol  = combined.loc[cidx, "rolling_vol_10"]
        stoch_d   = combined.loc[cidx, "Stoch_%D"]
        stoch_diff= combined.loc[cidx, "StochDiff"]
        adx_val   = combined.loc[cidx, "ADX"]
        rsi       = combined.loc[cidx, "RSI"]
        vwap      = combined.loc[cidx, "VWAP"]

        # ---- Bullish condition ----
        if (
            daily_change > 2
            and has_intra_streak(cidx, n, thresh, bullish=True)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and 30 < adx_val < 45
            and is_rsi_increasing(combined, cidx, bars_ago=2)
            and 50 < rsi < 75
            and stoch_d > 50
            and stoch_diff > 0
            and stoch_bull_cross(combined, cidx)
            and pd.notna(vwap) and price > 1.005 * vwap
            and vol >= 2.5 * roll_vol
        ):
            signal_id = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_DCHANGE_INTRASTREAK"
            if signal_id not in existing_signal_ids:
                signals.append({
                    "Ticker": ticker,
                    "date": dt,
                    "Entry Type": "DailyChange+IntraStreak",
                    "Trend Type": "Bullish",
                    "Price": round(float(price), 2),
                    "Daily Change %": round(float(daily_change), 2),
                    "Signal_ID": signal_id,
                    "logtime": row_detect.get("logtime", datetime.now().isoformat()),
                    "Entry Signal": "Yes",
                })
                existing_signal_ids.add(signal_id)

        # ---- Bearish condition ----
        if (
            daily_change < -2
            and has_intra_streak(cidx, n, thresh, bullish=False)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and 30 < adx_val < 45
            and is_rsi_decreasing(combined, cidx, bars_ago=2)
            and 25 < rsi < 50
            and stoch_d < 50
            and stoch_diff < 0
            and stoch_bear_cross(combined, cidx)
            and pd.notna(vwap) and price < 0.995 * vwap
            and vol >= 2.5 * roll_vol
        ):
            signal_id = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_DCHANGE_INTRASTREAK"
            if signal_id not in existing_signal_ids:
                signals.append({
                    "Ticker": ticker,
                    "date": dt,
                    "Entry Type": "DailyChange+IntraStreak",
                    "Trend Type": "Bearish",
                    "Price": round(float(price), 2),
                    "Daily Change %": round(float(daily_change), 2),
                    "Signal_ID": signal_id,
                    "logtime": row_detect.get("logtime", datetime.now().isoformat()),
                    "Entry Signal": "Yes",
                })
                existing_signal_ids.add(signal_id)

    return signals


# ---------------------------------------------
# 5) Mark signals in main CSV => CalcFlag='Yes'
# ---------------------------------------------
def mark_signals_in_main_csv(ticker: str, signals_list: list, main_path: str, tz_obj) -> None:
    """
    For rows in main CSV whose Signal_ID is in 'signals_list', set:
      Entry Signal = Yes, CalcFlag = Yes, and fill logtime if empty.
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return

    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, expected_cols=["Signal_ID"], tz="Asia/Kolkata")
            if df_main.empty:
                logging.warning(f"[{ticker}] => main_indicators empty. Skipping.")
                return

            # Ensure required columns exist
            for col in ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]:
                if col not in df_main.columns:
                    df_main[col] = ""

            # Extract Signal_IDs we want to mark
            signal_ids = [sig["Signal_ID"] for sig in signals_list if "Signal_ID" in sig]
            if not signal_ids:
                logging.info(f"[{ticker}] => No valid Signal_IDs to update.")
                return

            mask = df_main["Signal_ID"].isin(signal_ids)
            if mask.any():
                df_main.loc[mask, ["Entry Signal", "CalcFlag"]] = "Yes"
                current_time_iso = datetime.now(tz_obj).isoformat()
                df_main.loc[mask & (df_main["logtime"] == ""), "logtime"] = current_time_iso

                df_main.sort_values("date", inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] => Set CalcFlag='Yes' for {mask.sum()} row(s).")
            else:
                logging.info(f"[{ticker}] => No matching Signal_IDs found to update.")
    except Timeout:
        logging.error(f"[{ticker}] => FileLock timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] => Error marking signals: {e}")


# ---------------------------------------------
# 6) Detect signals for one trading date across all CSVs
# ---------------------------------------------
def find_price_action_entries_for_date(target_date: datetime.date) -> pd.DataFrame:
    """
    Collect new trading signals for the given date across all *_main_indicators.csv files.
    Intraday window considered: 09:25 to 14:04 IST (example).
    """
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(14, 4)))

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def process_file(file_path: str) -> list:
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()
        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df_full.empty:
            return []

        # Focus on today's intraday slice
        df_today = df_full[(df_full["date"] >= st) & (df_full["date"] <= et)].copy()

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,      # full data for rolling stats
            df_for_detection=df_today,   # only today's slice for detection
            existing_signal_ids=existing_ids,
        )
        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
            return new_signals
        return []

    # Parallel processing of files
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {target_date}"):
            try:
                signals = fut.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {futures[fut]}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logging.info(f"No new signals detected for {target_date}.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals).sort_values("date").reset_index(drop=True)
    logging.info(f"Total new signals detected for {target_date}: {len(signals_df)}")
    return signals_df


# ---------------------------------------------
# 7) Build papertrade CSV for a date (first signal per ticker)
# ---------------------------------------------
def create_papertrade_df(df_signals: pd.DataFrame, output_file: str, target_date) -> None:
    """
    Append only the first signal per ticker for the target_date to 'output_file'.
    Adds simple 'Target Price', 'Quantity', 'Total value' columns if missing.
    """
    if df_signals.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return

    df = normalize_time(df_signals, tz="Asia/Kolkata").sort_values("date")

    start_dt = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    end_dt   = india_tz.localize(datetime.combine(target_date, dt_time(15, 30)))
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()

    if df.empty:
        print("[Papertrade] No rows in [09:25 - 15:30].")
        return

    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df.columns:
            df[col] = ""

    # Simple target/size illustration; adjust to your risk sizing if needed
    for idx, row in df.iterrows():
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

        df.at[idx, "Target Price"] = round(target, 2)
        df.at[idx, "Quantity"] = qty
        df.at[idx, "Total value"] = round(total_val, 2)

    if "time" not in df.columns:
        df["time"] = datetime.now(india_tz).strftime("%H:%M:%S")
    else:
        blank_mask = (df["time"].isna()) | (df["time"] == "")
        if blank_mask.any():
            df.loc[blank_mask, "time"] = datetime.now(india_tz).strftime("%H:%M:%S")

    # Append first signal per ticker (per day) with a file lock
    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)

    with lock:
        if os.path.exists(output_file):
            existing = normalize_time(pd.read_csv(output_file), tz="Asia/Kolkata")
            existing_keys = set((erow["Ticker"], erow["date"].date()) for _, erow in existing.iterrows())

            new_rows = []
            for _, row in df.iterrows():
                key = (row["Ticker"], row["date"].date())
                if key not in existing_keys:
                    new_rows.append(row)

            if not new_rows:
                print("[Papertrade] No brand-new rows to append. All existing.")
                return

            new_df = pd.DataFrame(new_rows, columns=df.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["Ticker", "date"], keep="first", inplace=True)
            combined.sort_values("date", inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")
        else:
            df.drop_duplicates(subset=["Ticker"], keep="first", inplace=True)
            df.sort_values("date", inplace=True)
            df.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df)} rows.")


# ---------------------------------------------
# 8) Orchestrator for a single date
# ---------------------------------------------
def run_for_date(date_str: str) -> None:
    """
    Run detection + updates for one date (YYYY-MM-DD).
    - Writes all signals to ENTRIES_DIR/{date}_entries.csv
    - Marks matched rows in main CSVs as CalcFlag='Yes'
    - Appends first signal per ticker to papertrade CSV
    """
    try:
        target_day = datetime.strptime(date_str, "%Y-%m-%d").date()
        date_label = target_day.strftime("%Y-%m-%d")

        entries_file    = os.path.join(ENTRIES_DIR, f"{date_label}_entries.csv")
        papertrade_file = f"papertrade_5min_v7_relaxed_{date_label}.csv"

        # 1) Detect signals across files
        signals_df = find_price_action_entries_for_date(target_day)
        if signals_df.empty:
            logging.info(f"No signals found for {date_label}.")
            pd.DataFrame().to_csv(entries_file, index=False)  # write empty for completeness
            return

        # 2) Save all signals for the day
        signals_df.to_csv(entries_file, index=False)
        print(f"[ENTRIES] Wrote {len(signals_df)} signals to {entries_file}")

        # 3) Mark signals in the respective *_main_indicators.csv files
        for ticker, group in signals_df.groupby("Ticker"):
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, group.to_dict("records"), main_csv_path, india_tz)

        # 4) Append first-per-ticker to papertrade CSV
        create_papertrade_df(signals_df, papertrade_file, target_day)
        logging.info(f"Papertrade entries written to {papertrade_file}.")
    except Exception as e:
        logging.error(f"Error in run_for_date({date_str}): {e}")
        logging.error(traceback.format_exc())


# ---------------------------------------------
# 9) Graceful shutdown handler
# ---------------------------------------------
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ---------------------------------------------
# 10) Entry Point (Multiple Dates)
# ---------------------------------------------
if __name__ == "__main__":
    date_args = sys.argv[1:]
    if not date_args:
        # Backfill calendar (example)
        date_args = [
            "2025-09-04"
        ]

    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
