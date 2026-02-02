# -*- coding: utf-8 -*-
"""
Evaluate historical positional signals using 1H data
and check whether a ±5% move was achieved after entry.

For each signal (row) in the signals file:
    • Read corresponding 1H data for that ticker
    • Starting from AFTER the signal bar, scan forward:
        - LONG:  target = entry_price * 1.05, check high >= target
        - SHORT: target = entry_price * 0.95, check low  <= target
    • Record:
        - hit_5pct (True/False)
        - time_to_5pct_days (trading days, assuming TRADING_HOURS_PER_DAY hours per day)
        - max_favorable_pct
        - max_adverse_pct

Then:
    • Print 5% stats for ALL signals
    • Separately print 5% stats for NON-RECENT signals
      (all rows except the last N_RECENT_SKIP, ordered by signal_time_ist)

Inputs:
    • Latest signals file: signals/positional_signals_history_*.csv
    • 1H data per ticker: main_indicators_1h/<TICKER>_main_indicators_1h.csv

Output:
    • signals/positional_signals_5pct_eval_<YYYYMMDD_HHMM>_IST.csv  (ALL rows)
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_1H    = "main_indicators_1h"
SIG_DIR   = "signals"
OUT_DIR   = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# Approx trading hours per day (used to convert hours -> trading days)
TRADING_HOURS_PER_DAY = 6.25

# How many most-recent signals to treat as "recent"
N_RECENT_SKIP = 100

# If you want to explicitly set the signals file, put its path here.
# If left as None, the script will pick the latest
# "positional_signals_history_*.csv" under SIG_DIR.
SIGNALS_FILE = None  # e.g. "signals/positional_signals_history_20251101_1030_IST.csv"


# ----------------------- HELPERS -----------------------
def _safe_read_1h(path: str) -> pd.DataFrame:
    """Read 1H CSV, parse 'date' as tz-aware IST, sort."""
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(IST)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading 1H data {path}: {e}")
        return pd.DataFrame()


def _safe_read_signals(path: str) -> pd.DataFrame:
    """Read signals CSV, parse 'signal_time_ist' as tz-aware IST, sort by time."""
    try:
        df = pd.read_csv(path)
        df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"], utc=True, errors="coerce")
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_convert(IST)
        df = df.dropna(subset=["signal_time_ist"]).sort_values("signal_time_ist").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading signals file {path}: {e}")
        return pd.DataFrame()


def _find_latest_signals_file() -> str | None:
    """Pick the newest positional_signals_history_*.csv in SIG_DIR."""
    pattern = os.path.join(SIG_DIR, "positional_signals_history_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    # sort by modified time; take latest
    files.sort(key=os.path.getmtime)
    return files[-1]


def evaluate_signal_on_1h(row: pd.Series, df_1h: pd.DataFrame) -> dict:
    """
    Given one signal row and the 1H dataframe for that ticker,
    compute:
        - hit_5pct (bool)
        - time_to_5pct_days (trading days)
        - max_favorable_pct
        - max_adverse_pct
    """
    side = row["signal_side"]
    entry_time = row["signal_time_ist"]
    entry_price = float(row["entry_price"])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return {
            "hit_5pct": False,
            "time_to_5pct_days": np.nan,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }

    # Use only bars AFTER the signal bar;
    # if you prefer to include the signal bar itself, change > to >=
    future = df_1h[df_1h["date"] > entry_time].copy()
    if future.empty:
        return {
            "hit_5pct": False,
            "time_to_5pct_days": np.nan,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }

    if side.upper() == "LONG":
        target_price = entry_price * 1.05
        # target hit: any future high >= target_price
        hit_mask = future["high"] >= target_price

        # Favorable: how far above entry we went (%)
        max_fav = ((future["high"] - entry_price) / entry_price * 100.0).max()
        # Adverse: how far below entry we dipped (%; typically negative)
        max_adv = ((future["low"] - entry_price) / entry_price * 100.0).min()

    else:  # SHORT
        target_price = entry_price * 0.95
        # target hit: any future low <= target_price
        hit_mask = future["low"] <= target_price

        # For short: favorable = move down (entry - low) in %
        max_fav = ((entry_price - future["low"]) / entry_price * 100.0).max()
        # Adverse: move up against us (high - entry) as negative
        max_adv = ((entry_price - future["high"]) / entry_price * 100.0).min()

    if hit_mask.any():
        first_hit_idx = hit_mask[hit_mask].index[0]
        hit_time = future.loc[first_hit_idx, "date"]
        dt_hours = (hit_time - entry_time).total_seconds() / 3600.0
        # convert to trading days using TRADING_HOURS_PER_DAY
        dt_days = dt_hours / TRADING_HOURS_PER_DAY
        hit_flag = True
        time_to_target_days = dt_days
    else:
        hit_flag = False
        time_to_target_days = np.nan

    return {
        "hit_5pct": hit_flag,
        "time_to_5pct_days": time_to_target_days,
        "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
        "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
    }


# ----------------------- MAIN -----------------------
def main():
    # 1) Locate signals file
    global SIGNALS_FILE
    if SIGNALS_FILE is None:
        SIGNALS_FILE = _find_latest_signals_file()

    if SIGNALS_FILE is None or not os.path.exists(SIGNALS_FILE):
        print("No signals file found. Set SIGNALS_FILE or generate historical signals first.")
        return

    print(f"Using signals file: {SIGNALS_FILE}")
    sig_df = _safe_read_signals(SIGNALS_FILE)
    if sig_df.empty:
        print("Signals file is empty or invalid.")
        return

    total_signals = len(sig_df)
    print(f"Total signals in file: {total_signals}")

    # 2) Cache 1H data per ticker to avoid re-reading
    tickers = sig_df["ticker"].unique().tolist()
    print(f"Found {len(tickers)} tickers in signals file.")

    one_hour_data_cache: dict[str, pd.DataFrame] = {}
    results = []

    for t in tickers:
        path_1h = os.path.join(DIR_1H, f"{t}_main_indicators_1h.csv")
        df_1h = _safe_read_1h(path_1h)
        if df_1h.empty:
            print(f"- No 1H data for {t}, skipping its signals.")
            one_hour_data_cache[t] = pd.DataFrame()
        else:
            one_hour_data_cache[t] = df_1h

    # 3) Evaluate each signal row (ALL rows)
    for idx, row in sig_df.iterrows():
        ticker = row["ticker"]
        df_1h = one_hour_data_cache.get(ticker, pd.DataFrame())
        if df_1h.empty:
            # Skip if no data
            new_info = {
                "hit_5pct": False,
                "time_to_5pct_days": np.nan,
                "max_favorable_pct": np.nan,
                "max_adverse_pct": np.nan,
            }
        else:
            new_info = evaluate_signal_on_1h(row, df_1h)

        # Merge original signal row + new evaluation fields
        merged_row = row.to_dict()
        merged_row.update(new_info)
        results.append(merged_row)

    if not results:
        print("No signal evaluations produced.")
        return

    out_df = pd.DataFrame(results)

    # 4) Save evaluated signals (ALL rows)
    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"positional_signals_5pct_eval_{ts}_IST.csv")
    out_df.sort_values(["signal_time_ist", "ticker", "signal_side"], inplace=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved 5% evaluation for ALL signals to: {out_path}")

    # ---------- SUMMARY 1: ALL ROWS ----------
    hits_all = out_df["hit_5pct"].sum()
    total_all = len(out_df)
    hit_pct_all = (hits_all / total_all * 100.0) if total_all > 0 else 0.0
    print(f"\n[ALL SIGNALS] Total evaluated: {total_all}, 5% target hit: {hits_all} ({hit_pct_all:.1f}%)")

    # ---------- SUMMARY 2: NON-RECENT ROWS (excluding last N_RECENT_SKIP) ----------
    if total_all > N_RECENT_SKIP:
        out_df_non_recent = out_df.iloc[:-N_RECENT_SKIP].copy()
        total_non_recent = len(out_df_non_recent)
        hits_non_recent = out_df_non_recent["hit_5pct"].sum()
        hit_pct_non_recent = (hits_non_recent / total_non_recent * 100.0) if total_non_recent > 0 else 0.0

        print(f"\n[NON-RECENT SUBSET] (excluding last {N_RECENT_SKIP} signals)")
        print(f"Signals evaluated (non-recent): {total_non_recent}, 5% target hit: {hits_non_recent} ({hit_pct_non_recent:.1f}%)")
    else:
        print(f"\n[NON-RECENT SUBSET] Not enough rows to exclude last {N_RECENT_SKIP} signals; subset not computed.")

    # ---------- PREVIEW ----------
    print("\nLast few evaluated signals (ALL):")
    cols_preview = [
        "signal_time_ist",
        "ticker",
        "signal_side",
        "entry_price",
        "hit_5pct",
        "time_to_5pct_days",
        "max_favorable_pct",
        "max_adverse_pct",
    ]
    print(out_df[cols_preview].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
