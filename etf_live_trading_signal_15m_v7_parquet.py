# -*- coding: utf-8 -*-
"""
LIVE ETF 15‑MIN SIGNAL CHECKER (NO TICKER LOCK) — scheduled runner

What this does
--------------
- During market hours, every 15 minutes at:
      09:15:30, 09:30:30, ..., 15:30:30 IST
  it runs TWO checks per slot:
      - first at the slot time
      - second 20 seconds later

- For EACH ticker, it checks ONLY the latest 15‑minute row in your *15m indicator parquet*,
  merges daily+weekly context (no weekly lookahead), and evaluates the same LONG mask
  you use in your v7 parquet backtest.

- It writes per‑run parquet outputs (so we don't "rewrite" a giant parquet on every run):
    out_live_checks/YYYYMMDD/checks_HHMMSS_A.parquet     (all tickers, incl. signal_triggered False)
    out_live_signals/YYYYMMDD/signals_HHMMSS_A.parquet   (only signal_triggered True)

Each output row contains:
- ALL merged columns (15m + daily + weekly + helper features)
- ticker, bar_time_ist, checked_at_ist, run_tag, signal_triggered, entry_price

How to run
----------
1) Put this script in your algo_trading folder.
2) Ensure these directories exist (or change CONFIG below):
      etf_indicators_15min_pq/
      etf_indicators_daily_pq/
      etf_indicators_weekly_pq/
   with files named like:
      <TICKER>_etf_indicators_15min.parquet
      <TICKER>_etf_indicators_daily.parquet
      <TICKER>_etf_indicators_weekly.parquet

3) Start this script at ~09:10 IST:
      python live_etf_15m_signal_scheduler.py

Notes
-----
- "No ticker lock": we do NOT suppress multiple signals from the same ticker.
- This script DOES NOT place orders. It only generates/checks signals and logs them.
- If your 15m parquet isn’t updated yet for the new candle, the second check (t+20s)
  gives it another chance without scanning full history.
"""

from __future__ import annotations

import os
import glob
import time
from dataclasses import dataclass
from datetime import datetime, date, time as dtime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# Optional (for faster parquet tail reads)
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


# =============================================================================
# CONFIG
# =============================================================================

IST = pytz.timezone("Asia/Kolkata")

# Indicator parquet directories (match your v7 parquet scripts)
DIR_15M = "etf_indicators_15min_pq"
DIR_D   = "etf_indicators_daily_pq"
DIR_W   = "etf_indicators_weekly_pq"

# Output folders (per-run parquet files)
OUT_CHECKS_DIR  = "out_live_checks"
OUT_SIGNALS_DIR = "out_live_signals"

# File endings (match your naming)
END_15M = "_etf_indicators_15min.parquet"
END_D   = "_etf_indicators_daily.parquet"
END_W   = "_etf_indicators_weekly.parquet"

# Scheduling
START_TIME = dtime(9, 15, 10)   # 09:15:30 IST
END_TIME   = dtime(15, 30, 10)  # 15:30:30 IST
SLOT_MINS  = 15
SECOND_RUN_GAP_SECONDS = 15
SESSION_END_TIME = dtime(15, 40, 0)   # hard stop for the script (exit)

# Concurrency / performance
MAX_WORKERS = max(4, (os.cpu_count() or 4) // 2)   # adjust if you want
INTRADAY_TAIL_ROWS = 250  # enough for rolling(5), plus some buffer

# Signal output columns
SIGNAL_SIDE = "LONG"


# =============================================================================
# HELPERS
# =============================================================================

def _now_ist() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(IST))


SPECIAL_OPEN_DAYS = {
    date(2026, 2, 1),  # Budget special trading Sunday
}

def is_trading_day(day: date) -> bool:
    if day in SPECIAL_OPEN_DAYS:
        return True
    return day.weekday() < 5


def list_tickers_from_dir(directory: str, ending: str) -> List[str]:
    pattern = os.path.join(directory, f"*{ending}")
    files = glob.glob(pattern)
    tickers: List[str] = []
    for f in files:
        base = os.path.basename(f)
        if base.endswith(ending):
            t = base[: -len(ending)]
            if t:
                tickers.append(t)
    return sorted(set(tickers))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def parquet_tail(path: str, n: int) -> pd.DataFrame:
    """
    Try to read only the last row-groups of a parquet (fast), fallback to full read.
    Works best if your parquet is written in multiple row groups over time.
    """
    if pq is None:
        return pd.read_parquet(path)

    try:
        pf = pq.ParquetFile(path)
        num_rg = pf.num_row_groups
        if num_rg <= 0:
            return pf.read().to_pandas()

        # Read from the end until we have >= n rows
        dfs: List[pd.DataFrame] = []
        rows = 0
        for rg in range(num_rg - 1, -1, -1):
            t = pf.read_row_group(rg).to_pandas()
            dfs.append(t)
            rows += len(t)
            if rows >= n:
                break
        df = pd.concat(reversed(dfs), ignore_index=True)
        if len(df) > n:
            df = df.tail(n).reset_index(drop=True)
        return df
    except Exception:
        return pd.read_parquet(path)


def read_tf_parquet(path: str, tz=IST, tail_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Read parquet, parse 'date' as UTC then convert to IST, sort by date.
    If tail_rows is provided, attempt to read only last rows.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = parquet_tail(path, tail_rows) if tail_rows else pd.read_parquet(path)
        if "date" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(tz)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def suffix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)


# =============================================================================
# FEATURES + MERGE (same shape as your v7 parquet logic)
# =============================================================================

def add_intraday_features(df_i: pd.DataFrame) -> pd.DataFrame:
    """Rolling/shift helpers (15m). Keep minimal + consistent with backtest."""
    if df_i.empty:
        return df_i

    # These names assume your indicator columns are already present (ATR, RSI, MACD_Hist, ADX, Stoch_%K, etc.)
    df_i["ATR_I_5bar_mean"] = df_i.get("ATR_I", np.nan).rolling(5).mean()
    df_i["RSI_I_prev"] = df_i.get("RSI_I", np.nan).shift(1)
    df_i["MACD_Hist_I_prev"] = df_i.get("MACD_Hist_I", np.nan).shift(1)
    df_i["Recent_High_5_I"] = df_i.get("high_I", np.nan).rolling(5).max()
    df_i["ADX_I_5bar_mean"] = df_i.get("ADX_I", np.nan).rolling(5).mean()
    df_i["Stoch_%K_I_prev"] = df_i.get("Stoch_%K_I", np.nan).shift(1)
    return df_i


def merge_daily_and_weekly_onto_intraday(df_i: pd.DataFrame, df_d: pd.DataFrame, df_w: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Daily + Weekly context onto each intraday row using as-of logic.
    Weekly lookahead guard: shift weekly bar by +1s.
    """
    if df_i.empty or df_d.empty or df_w.empty:
        return pd.DataFrame()

    df_id = pd.merge_asof(
        df_i.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward",
    )

    df_w_shift = df_w.sort_values("date").copy()
    df_w_shift["date"] += pd.Timedelta(seconds=1)

    df_idw = pd.merge_asof(
        df_id.sort_values("date"),
        df_w_shift,
        on="date",
        direction="backward",
    )
    return df_idw


def compute_long_signal_mask(df: pd.DataFrame) -> pd.Series:
    """
    LONG mask (ported from your v7 parquet 15m script).
    This expects suffixed columns: _W, _D, _I.
    """
    req = [
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W", "Daily_Change_prev_D",
        "close_I", "EMA_200_I", "EMA_50_I", "RSI_I", "MACD_Hist_I", "Stoch_%K_I", "Stoch_%D_I",
        "ATR_I", "ATR_I_5bar_mean", "ADX_I", "ADX_I_5bar_mean", "VWAP_I", "Recent_High_5_I"
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return pd.Series(False, index=df.index)

    # ---- Weekly ----
    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 40.0) &
        (df["ADX_W"] > 10.0)
    )
    weekly_mild = (
        (df["close_W"] > df["EMA_200_W"] * 1.01) &
        (df["RSI_W"].between(40.0, 68.0)) &
        (df["ADX_W"] > 9.0)
    )
    weekly_not_extreme = df["RSI_W"] < 88.0
    weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme

    # ---- Daily (prev day move) ----
    daily_prev = df["Daily_Change_prev_D"]
    daily_ok = (daily_prev >= 2) & (daily_prev <= 9.0)

    # ---- Intraday (15m) ----
    intraday_trend = (
        (df["close_I"] >= df["EMA_200_I"] * 0.995) &
        (df["close_I"] >= df["EMA_50_I"] * 0.995)
    )

    rsi_ok = (
        (df["RSI_I"] > 49.0) |
        ((df["RSI_I"] > 46.0) & (df["RSI_I"] > df["RSI_I_prev"]))
    )

    macd_ok = (
        (df["MACD_Hist_I"] > -0.055) &
        (df["MACD_Hist_I"] >= df["MACD_Hist_I_prev"] * 0.75)
    )

    stoch_ok = (
        (df["Stoch_%K_I"] >= df["Stoch_%D_I"] * 0.98) &
        (df["Stoch_%K_I"] >= 18.0)
    )

    momentum_ok = rsi_ok & macd_ok & stoch_ok

    intraday_vol_ok = (
        (df["ATR_I"] >= df["ATR_I_5bar_mean"] * 0.92) &
        ((df["ADX_I"] >= 10.0) | (df["ADX_I"] >= df["ADX_I_5bar_mean"]))
    )

    price_above_vwap = df["close_I"] >= df["VWAP_I"]

    breakout_strict = df["close_I"] >= df["Recent_High_5_I"]
    breakout_near   = df["close_I"] >= df["Recent_High_5_I"] * 0.99

    intra_chg = df.get("Intra_Change_I", 0.0)
    intra_chg = pd.to_numeric(intra_chg, errors="coerce").fillna(0.0)
    strong_up = intra_chg >= 0.05

    breakout_ok = price_above_vwap & (
        breakout_strict |
        (breakout_near & strong_up & (df["RSI_I"] >= 47.0))
    )

    intraday_ok = intraday_trend & momentum_ok & intraday_vol_ok & breakout_ok

    return weekly_ok & daily_ok & intraday_ok


# =============================================================================
# CACHING (speed)
# =============================================================================

@dataclass
class CacheItem:
    mtime: float
    df: pd.DataFrame


class ParquetCache:
    def __init__(self):
        self._cache: Dict[str, CacheItem] = {}

    def get(self, path: str, *, tail_rows: Optional[int] = None) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()

        mtime = os.path.getmtime(path)
        item = self._cache.get(path)
        if item is not None and item.mtime == mtime:
            return item.df

        df = read_tf_parquet(path, tz=IST, tail_rows=tail_rows)
        self._cache[path] = CacheItem(mtime=mtime, df=df)
        return df


CACHE = ParquetCache()


# =============================================================================
# PER-TICKER CHECK (latest row only)
# =============================================================================

def check_latest_for_ticker(ticker: str, checked_at: pd.Timestamp, run_tag: str) -> Optional[Dict]:
    path_i = os.path.join(DIR_15M, f"{ticker}{END_15M}")
    path_d = os.path.join(DIR_D,   f"{ticker}{END_D}")
    path_w = os.path.join(DIR_W,   f"{ticker}{END_W}")

    # Intraday: tail only
    df_i = CACHE.get(path_i, tail_rows=INTRADAY_TAIL_ROWS)
    df_d = CACHE.get(path_d, tail_rows=None)
    df_w = CACHE.get(path_w, tail_rows=None)

    if df_i.empty or df_d.empty or df_w.empty:
        return None

    # Daily: previous day's Daily_Change
    if "Daily_Change" in df_d.columns:
        df_d = df_d.copy()
        df_d["Daily_Change_prev"] = pd.to_numeric(df_d["Daily_Change"], errors="coerce").shift(1)
    else:
        df_d = df_d.copy()
        df_d["Daily_Change_prev"] = np.nan

    # Suffix columns (I/D/W)
    df_i = suffix_columns(df_i, "_I")
    df_d = suffix_columns(df_d, "_D")
    df_w = suffix_columns(df_w, "_W")

    # Intraday helper features
    df_i = add_intraday_features(df_i)

    # Merge context
    df_idw = merge_daily_and_weekly_onto_intraday(df_i, df_d, df_w)
    if df_idw.empty:
        return None

    # Take ONLY the latest row
    last_row = df_idw.iloc[[-1]].copy()

    # Minimal dropna to avoid noisy false signals
    last_row = last_row.dropna(subset=["close_I", "close_W", "Daily_Change_prev_D"], how="any")
    if last_row.empty:
        return None

    mask = bool(compute_long_signal_mask(last_row).iloc[0])

    r = last_row.iloc[0]
    bar_time = r["date"]
    entry_price = safe_float(r.get("close_I", np.nan))

    # Build record with all columns + meta
    rec = r.to_dict()
    rec.update({
        "ticker": ticker,
        "signal_side": SIGNAL_SIDE,
        "signal_triggered": mask,
        "entry_price": entry_price,
        "bar_time_ist": bar_time,
        "checked_at_ist": checked_at,
        "run_tag": run_tag,
    })
    return rec


# =============================================================================
# OUTPUT (per-run parquet files)
# =============================================================================

def write_run_outputs(records: List[Dict], checked_at: pd.Timestamp, run_tag: str) -> None:
    if not records:
        return

    df = pd.DataFrame(records)

    # ensure datetime types
    for c in ["bar_time_ist", "checked_at_ist"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    day_folder = checked_at.strftime("%Y%m%d")
    tstamp = checked_at.strftime("%H%M%S")

    ensure_dir(os.path.join(OUT_CHECKS_DIR, day_folder))
    ensure_dir(os.path.join(OUT_SIGNALS_DIR, day_folder))

    checks_path = os.path.join(OUT_CHECKS_DIR, day_folder, f"checks_{tstamp}_{run_tag}.parquet")
    df.to_parquet(checks_path, index=False)

    sig_df = df[df.get("signal_triggered", False)].copy()
    if not sig_df.empty:
        signals_path = os.path.join(OUT_SIGNALS_DIR, day_folder, f"signals_{tstamp}_{run_tag}.parquet")
        sig_df.to_parquet(signals_path, index=False)


# =============================================================================
# SCHEDULER
# =============================================================================

def build_today_slots(day: date) -> List[datetime]:
    """List of base slot datetimes for today (IST)."""
    start_dt = IST.localize(datetime.combine(day, START_TIME))
    end_dt = IST.localize(datetime.combine(day, END_TIME))

    slots: List[datetime] = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur)
        cur = cur + timedelta(minutes=SLOT_MINS)
    return slots


def sleep_until(target: datetime) -> None:
    """Sleep until target (IST aware)."""
    while True:
        now = datetime.now(IST)
        delta = (target - now).total_seconds()
        if delta <= 0:
            return
        time.sleep(min(delta, 1.0))


def run_one_check(tickers: List[str], run_tag: str) -> None:
    checked_at = _now_ist()

    # parallel per ticker
    records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(check_latest_for_ticker, t, checked_at, run_tag): t for t in tickers}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    write_run_outputs(records, checked_at, run_tag)

    # Small console summary
    sigs = sum(1 for r in records if bool(r.get("signal_triggered", False)))
    print(f"[{checked_at.strftime('%Y-%m-%d %H:%M:%S%z')}] run_tag={run_tag} "
          f"tickers_checked={len(records)} signals={sigs}")


def main() -> None:
    # Discover tickers by files present in DIR_15M (only those with 15m parquet)
    tickers = list_tickers_from_dir(DIR_15M, END_15M)
    if not tickers:
        print(f"[ERROR] No tickers found in {DIR_15M} with ending {END_15M}")
        return

    print(f"[INFO] Loaded {len(tickers)} tickers from {DIR_15M}.")
    print(f"[INFO] Schedule: every {SLOT_MINS}m from {START_TIME} to {END_TIME}, two runs each slot (+{SECOND_RUN_GAP_SECONDS}s).")
    print(f"[INFO] Outputs: {OUT_CHECKS_DIR}/ and {OUT_SIGNALS_DIR}/")

    last_day: Optional[date] = None

    while True:
        now = datetime.now(IST)
        today = now.date()

        # ---------------- HARD STOP AT 15:40 ----------------
        if now.time() >= SESSION_END_TIME:
            print(f"[INFO] Session ended at {SESSION_END_TIME}. Exiting.")
            return
        # ----------------------------------------------------

        if last_day != today:
            last_day = today
            print(f"\n[INFO] New day detected: {today}. Trading day={is_trading_day(today)}")

        if not is_trading_day(today):
            print("[INFO] Non-trading day. Exiting.")
            return

        slots = build_today_slots(today)

        # Find next slot >= now
        next_slot = None
        for s in slots:
            if s >= now:
                next_slot = s
                break

        if next_slot is None:
            # No more slots for today → wait until 15:40 then exit
            stop_time = IST.localize(datetime.combine(today, SESSION_END_TIME))
            print(f"[INFO] Market slots completed. Waiting until {stop_time} then exiting.")
            sleep_until(stop_time)
            print(f"[INFO] Session ended at {SESSION_END_TIME}. Exiting.")
            return

        # Wait for next slot
        sleep_until(next_slot)

        # Run first check
        run_one_check(tickers, run_tag="A")

        # Run second check 20 seconds later (same slot)
        time.sleep(SECOND_RUN_GAP_SECONDS)
        run_one_check(tickers, run_tag="B")



if __name__ == "__main__":
    main()
