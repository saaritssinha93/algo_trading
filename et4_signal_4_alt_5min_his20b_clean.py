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
INDICATORS_DIR = "main_indicators_july_5min"
SIGNALS_DB     = "generated_signals_historical5minv2a.json"
ENTRIES_DIR    = "main_indicators_history_entries_5minv2a"  # where day's detected signals are saved
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(ENTRIES_DIR, exist_ok=True)

india_tz = pytz.timezone("Asia/Kolkata")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = TimedRotatingFileHandler(
    "logs\\signalnewn5minv2a.log", when="M", interval=30, backupCount=5, delay=True
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
    fitted_csv_path: str = "fitted_indicator_ranges_by_ticker.csv",
    volrel_window: int = 10,
    volrel_min_periods: int = 10,     # consider lowering to 5 if early-bar signals are desirable
    debug: bool = False,
) -> list:
    """
    Signal detector driven by per-ticker winners-only fitted ranges loaded from CSV.

    Fixes vs prior:
      - Case/whitespace insensitive ticker join.
      - Ignores constraints with NaN thresholds or missing data columns.
      - Accepts precomputed rolling_vol_10 or rolling_vol10; else computes.
      - Optional debug counters for quick diagnosis.
    """

    import ast
    import numpy as np
    import pandas as pd
    from datetime import datetime

    signals = []
    if df_for_detection.empty:
        return signals

    # -----------------------------
    # Load per-ticker constraints
    # -----------------------------
    def _parse_range_cell(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return (np.nan, np.nan)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            try:
                return (float(x[0]), float(x[1]))
            except Exception:
                return (np.nan, np.nan)
        s = str(x).strip()
        if not s:
            return (np.nan, np.nan)
        try:
            lo, hi = ast.literal_eval(s)
            return (float(lo), float(hi))
        except Exception:
            s = s.strip("[]")
            parts = [p for p in s.split(",") if p.strip() != ""]
            if len(parts) == 2:
                try:
                    return (float(parts[0]), float(parts[1]))
                except Exception:
                    pass
            return (np.nan, np.nan)

    def _to_float(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return np.nan
        try:
            return float(x)
        except Exception:
            try:
                return float(str(x).strip())
            except Exception:
                return np.nan

    try:
        cfg_df = pd.read_csv(fitted_csv_path)
    except Exception as e:
        print(f"[detect] Could not read '{fitted_csv_path}': {e}")
        return signals

    # Normalize ticker matching (case/space insensitive)
    tkey = str(ticker).strip().upper()
    if "Ticker" not in cfg_df.columns:
        return signals
    cfg_df = cfg_df.copy()
    cfg_df["__TickerKey"] = cfg_df["Ticker"].astype(str).str.strip().str.upper()
    cfg_df = cfg_df[cfg_df["__TickerKey"] == tkey]
    if cfg_df.empty:
        # No config => nothing to enforce
        if debug:
            print(f"[detect:{ticker}] no per-ticker rows in fitted CSV.")
        return signals

    # Build raw side->constraints (we'll clean after we know available columns)
    sides_raw = {}
    for _, row in cfg_df.iterrows():
        side = str(row.get("Side", "")).strip()
        if side not in ("Bullish", "Bearish"):
            continue
        constraints = {}
        for col in row.index:
            if col in ("Ticker", "__TickerKey", "Side", "Samples_wins", "Quantiles_used"):
                continue
            if col.endswith("_range"):
                constraints[col] = _parse_range_cell(row[col])
            elif col.endswith("_min") or col.endswith("_max"):
                constraints[col] = _to_float(row[col])
        if constraints:
            sides_raw[side] = constraints

    if not sides_raw:
        if debug:
            print(f"[detect:{ticker}] no constraints present after parse.")
        return signals

    # --------------------------------
    # Merge & compute derived features
    # --------------------------------
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined.dropna(subset=["date"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["date_only"] = combined["date"].dt.date

    # Ensure VWAP
    if "VWAP" not in combined.columns:
        combined = calculate_session_vwap(combined)

    # Rolling volume → VolRel (accept precomputed names)
    if "rolling_vol_10" in combined.columns:
        rv = combined["rolling_vol_10"]
    elif "rolling_vol10" in combined.columns:
        rv = combined["rolling_vol10"]
        combined["rolling_vol_10"] = rv
    else:
        rv = combined["volume"].rolling(volrel_window, min_periods=volrel_min_periods).mean()
        combined["rolling_vol_10"] = rv

    eps = 1e-9
    combined["StochD"] = combined.get("Stoch_%D", np.nan)
    combined["StochK"] = combined.get("Stoch_%K", np.nan)
    combined["VWAP_Dist"] = combined["close"] / combined["VWAP"] - 1.0
    combined["EMA50_Dist"]  = combined["close"] / combined["EMA_50"]  - 1.0 if "EMA_50"  in combined.columns else np.nan
    combined["EMA200_Dist"] = combined["close"] / combined["EMA_200"] - 1.0 if "EMA_200" in combined.columns else np.nan
    combined["SMA20_Dist"]  = combined["close"] / combined["20_SMA"]  - 1.0 if "20_SMA"  in combined.columns else np.nan
    combined["VolRel"] = combined["volume"] / (combined["rolling_vol_10"] + eps)
    combined["ATR_Pct"] = (combined["ATR"] / (combined["close"] + eps)) if "ATR" in combined.columns else np.nan

    if {"Upper_Band","Lower_Band"}.issubset(combined.columns):
        bbw = (combined["Upper_Band"] - combined["Lower_Band"])
        combined["BandWidth"] = bbw / (combined["close"] + eps)
        combined["BBP"] = (combined["close"] - combined["Lower_Band"]) / (bbw + eps)
    else:
        combined["BandWidth"] = np.nan
        combined["BBP"] = np.nan

    if "OBV" in combined.columns:
        obv_prev = combined["OBV"].shift(10)
        combined["OBV_Slope10"] = (combined["OBV"] - obv_prev) / (obv_prev.abs() + eps)
    else:
        combined["OBV_Slope10"] = np.nan

    # Base-name → column-name map
    colmap = {
        "RSI": "RSI",
        "ADX": "ADX",
        "CCI": "CCI",
        "MFI": "MFI",
        "BBP": "BBP",
        "BandWidth": "BandWidth",
        "ATR_Pct": "ATR_Pct",
        "StochD": "StochD",
        "StochK": "StochK",
        "MACD_Hist": "MACD_Hist",
        "VWAP_Dist": "VWAP_Dist",
        "Daily_Change": "Daily_Change",
        "EMA50_Dist": "EMA50_Dist",
        "EMA200_Dist": "EMA200_Dist",
        "SMA20_Dist": "SMA20_Dist",
        "OBV_Slope10": "OBV_Slope10",
        "VolRel": "VolRel",
    }
    available_cols = set(combined.columns)

    # Clean constraints: drop any with NaN thresholds or missing data columns
    def _clean_constraints(raw: dict) -> dict:
        cleaned = {}
        for k, v in raw.items():
            if k.endswith("_range"):
                base = k[:-6]
                col = colmap.get(base, base)
                if col not in available_cols:
                    continue
                lo, hi = v if isinstance(v, (list, tuple)) and len(v) == 2 else (np.nan, np.nan)
                if np.isnan(lo) or np.isnan(hi):
                    continue
                cleaned[k] = (float(lo), float(hi))
            elif k.endswith("_min") or k.endswith("_max"):
                base = k[:-4]
                col = colmap.get(base, base)
                if col not in available_cols:
                    continue
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                cleaned[k] = float(v)
        return cleaned

    sides = {side: _clean_constraints(raw) for side, raw in sides_raw.items()}

    if all(len(cfg) == 0 for cfg in sides.values()):
        if debug:
            print(f"[detect:{ticker}] All constraints dropped (NaN or missing columns).")
        return signals

    # Helper: find index of detection row
    def find_cidx(ts):
        ts = pd.to_datetime(ts)
        exact = combined.index[combined["date"] == ts]
        if len(exact):
            return int(exact[0])
        d = ts.date()
        mask = (combined["date_only"] == d) & (combined["date"] <= ts)
        if mask.any():
            return int(combined.index[mask][-1])
        return None

    # Constraint evaluation
    def _check_side_constraints(row: pd.Series, side_constraints: dict) -> bool:
        # ranges
        for k, rng in side_constraints.items():
            if k.endswith("_range"):
                base = k[:-6]
                col = colmap.get(base, base)
                x = row.get(col, np.nan)
                lo, hi = rng
                if pd.isna(x) or not (lo <= x <= hi):
                    return False
        # mins
        for k, thr in side_constraints.items():
            if k.endswith("_min"):
                base = k[:-4]
                col = colmap.get(base, base)
                x = row.get(col, np.nan)
                if pd.isna(x) or not (x >= thr):
                    return False
        # max
        for k, thr in side_constraints.items():
            if k.endswith("_max"):
                base = k[:-4]
                col = colmap.get(base, base)
                x = row.get(col, np.nan)
                if pd.isna(x) or not (x <= thr):
                    return False
        return True

    # Optional debug counters
    fail_counts = {"Bullish": 0, "Bearish": 0}
    checked_counts = {"Bullish": 0, "Bearish": 0}

    # Evaluate detection slice
    for _, row_det in df_for_detection.iterrows():
        dt = row_det.get("date", None)
        if pd.isna(dt):
            continue

        cidx = find_cidx(dt)
        if cidx is None:
            continue

        r = combined.loc[cidx]

        bull_cfg = sides.get("Bullish")
        if bull_cfg:
            checked_counts["Bullish"] += 1
            if _check_side_constraints(r, bull_cfg):
                sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_CSV_FITTED"
                if sid not in existing_signal_ids:
                    signals.append({
                        "Ticker": ticker,
                        "date": dt,
                        "Entry Type": "CSVFittedFilter",
                        "Trend Type": "Bullish",
                        "Price": round(float(r.get("close", np.nan)), 2),
                        "Signal_ID": sid,
                        "logtime": row_det.get("logtime", datetime.now().isoformat()),
                        "Entry Signal": "Yes",
                    })
                    existing_signal_ids.add(sid)
            else:
                fail_counts["Bullish"] += 1

        bear_cfg = sides.get("Bearish")
        if bear_cfg:
            checked_counts["Bearish"] += 1
            if _check_side_constraints(r, bear_cfg):
                sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_CSV_FITTED"
                if sid not in existing_signal_ids:
                    signals.append({
                        "Ticker": ticker,
                        "date": dt,
                        "Entry Type": "CSVFittedFilter",
                        "Trend Type": "Bearish",
                        "Price": round(float(r.get("close", np.nan)), 2),
                        "Signal_ID": sid,
                        "logtime": row_det.get("logtime", datetime.now().isoformat()),
                        "Entry Signal": "Yes",
                    })
                    existing_signal_ids.add(sid)
            else:
                fail_counts["Bearish"] += 1

    if debug:
        for side in ("Bullish", "Bearish"):
            tot = checked_counts[side]
            if tot:
                print(f"[detect:{ticker}] {side} checked={tot}, failed={fail_counts[side]}, passed={tot - fail_counts[side]}")
            else:
                if side in sides and len(sides[side]):
                    print(f"[detect:{ticker}] {side} had constraints but 0 rows checked (time window?).")

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
    Intraday window considered: 09:25 to 15:10 IST (align with training window).
    """
    # OPTIONAL: align runtime window to training end time (15:10)
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(15, 10)))  # was 14:04

    # Tunables for detector (you can move these to your global config section)
    DETECTOR_VOLREL_WINDOW = 10
    DETECTOR_VOLREL_MIN_PERIODS = 10   # set to 5 if you want earlier-bar signals
    DETECTOR_DEBUG = False             # flip True for one-day debugging

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def process_file(file_path: str) -> list:
        # keep .upper() if you want, detector now matches case-insensitively anyway
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()

        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df_full.empty:
            return []

        # Focus on today's intraday slice
        df_today = df_full[(df_full["date"] >= st) & (df_full["date"] <= et)].copy()
        if df_today.empty:
            return []

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,       # full history for rolling stats
            df_for_detection=df_today,    # today's slice for detection
            existing_signal_ids=existing_ids,
            fitted_csv_path="fitted_indicator_ranges_by_ticker.csv",
            volrel_window=DETECTOR_VOLREL_WINDOW,
            volrel_min_periods=DETECTOR_VOLREL_MIN_PERIODS,
            debug=DETECTOR_DEBUG,
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
        papertrade_file = f"papertrade_5min_v2a_{date_label}.csv"

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
            # August 2024
            "2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09",
            "2024-08-12","2024-08-13","2024-08-14","2024-08-16",
            "2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23",
            "2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30",

            # September 2024
            "2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06",
            "2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13",
            "2024-09-16","2024-09-17","2024-09-18","2024-09-19","2024-09-20",
            "2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27",
            "2024-09-30",

            # October 2024 (Oct 2 holiday)
            "2024-10-01","2024-10-03","2024-10-04","2024-10-07","2024-10-08",
            "2024-10-09","2024-10-10","2024-10-11","2024-10-14","2024-10-15",
            "2024-10-16","2024-10-17","2024-10-18","2024-10-21","2024-10-22",
            "2024-10-23","2024-10-24","2024-10-25","2024-10-28","2024-10-29",
            "2024-10-30","2024-10-31",

            # November 2024 (Nov 1 & Nov 15 holidays)
            "2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08",
            "2024-11-11","2024-11-12","2024-11-13","2024-11-14",
            "2024-11-18","2024-11-19","2024-11-20","2024-11-21","2024-11-22",
            "2024-11-25","2024-11-26","2024-11-27","2024-11-28","2024-11-29",

            # December 2024 (Dec 25 holiday)
            "2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06",
            "2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13",
            "2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20",
            "2024-12-23","2024-12-24","2024-12-26","2024-12-27","2024-12-30",
            "2024-12-31",

            # January 2025
            "2025-01-01","2025-01-02","2025-01-03",
            "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
            "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
            "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
            # 2025-01-26 is Republic Day (Sunday+holiday)
            "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

            # February 2025
            "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
            "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
            "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
            "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

            # March 2025
            "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
            "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
            # 2025-03-14 is Holi (holiday)
            "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
            "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
            "2025-03-31",

            # April 2025
            "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
            "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
            # 2025-04-11 is Eid-ul-Fitr (holiday)
            "2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18",
            "2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25",
            "2025-04-28","2025-04-29","2025-04-30",

            # May 2025
            "2025-05-02",
            "2025-05-05","2025-05-06","2025-05-07","2025-05-08","2025-05-09",
            "2025-05-12","2025-05-13","2025-05-14","2025-05-15","2025-05-16",
            "2025-05-19","2025-05-20","2025-05-21","2025-05-22","2025-05-23",
            "2025-05-26","2025-05-27","2025-05-28","2025-05-29","2025-05-30",

            # June 2025
            "2025-06-02","2025-06-03","2025-06-04","2025-06-05","2025-06-06",
            "2025-06-09","2025-06-10","2025-06-11","2025-06-12","2025-06-13",
            "2025-06-16","2025-06-17","2025-06-18","2025-06-19","2025-06-20",
            "2025-06-23","2025-06-24","2025-06-25","2025-06-26","2025-06-27",
            "2025-06-30",

            # July 2025
            "2025-07-01","2025-07-02","2025-07-03","2025-07-04",
            "2025-07-07","2025-07-08","2025-07-09","2025-07-10"
        ]

    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
