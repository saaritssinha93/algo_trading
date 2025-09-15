# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 16:57:22 2025

@author: Saarit

Combined runner:
1) Normalize Quantity & Total value in papertrade files.
2) Run batched backtests (v5) with SL/TP logic and skip-if-results-exist.
3) Build per-day summaries and pretty console table.
4) Repeat every 5 minutes (configurable).

Author: Saarit
"""

import os
import csv
import glob
import time
import logging
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import pytz

# ---- Optional dependency (file locks) ----
try:
    from filelock import FileLock, Timeout
except Exception:
    class FileLock:
        def __init__(self, filename, timeout=10): self.filename = filename
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Timeout(Exception): pass

# ---- Zerodha Kite ----
try:
    from kiteconnect import KiteConnect
except Exception as e:
    raise ImportError("kiteconnect is required. pip install kiteconnect") from e


# =========================
# CONFIG (paths, dates, scheduling)
# =========================
CWD = r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo"
os.chdir(CWD)

# Input/Output file prefixes
PAPERTRADE_PREFIX     = "papertrade_5min_v5_"
RESULT_PREFIX         = "result_papertradesl_v5_"      # per-trade rows
RESULT_FINAL_PREFIX   = "result_finalsl_5min_v5_"      # per-day net % summary

# Backtest control
SKIP_IF_RESULTS_EXIST     = True    # skip whole date if either result already exists
PURGE_RESULTS_ON_FIRST_RUN = True   # delete old results once on the very first cycle only
PURGE_PATTERNS = [
    f"{RESULT_PREFIX}*.csv",
    f"{RESULT_FINAL_PREFIX}*.csv",
]

# Market session (IST)
IST = pytz.timezone("Asia/Kolkata")
SESSION_START = dt_time(9, 25)
SESSION_END   = dt_time(15, 15)

# Engine params
BATCH_SIZE   = 25
MAX_WORKERS  = 8
API_SLEEP_BETWEEN_BATCHES = 0.4   # seconds between batches
INTERVAL = "minute"               # historical candle interval for backtest

# Trade params
BULL_TARGET_PCT = 0.05
BEAR_TARGET_PCT = 0.042

# NEW: separate SL for long vs short
BULL_STOP_LOSS_PCT = 0.035   # e.g., 3% below buy for bullish trades
BEAR_STOP_LOSS_PCT = 0.03   # e.g., 3% above sell for bearish trades


# Summary params
MARGIN_MULTIPLIER = 5

# Scheduler
RUN_EVERY_5_MIN = True
SLEEP_SECONDS   = 5 * 60   # 5 minutes

# Logging
logging.basicConfig(
    filename="trading_log_batched_v5.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

# Calendar (same list you’ve been using)
LAST_TRADING_DAYS = [
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

    # December 2024
    "2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06",
    "2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13",
    "2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20",
    "2024-12-23","2024-12-24","2024-12-26","2024-12-27","2024-12-30","2024-12-31",

    # January 2025
    "2025-01-01","2025-01-02","2025-01-03",
    "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
    "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
    "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
    "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

    # February 2025
    "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
    "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
    "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
    "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

    # March 2025
    "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
    "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
    "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
    "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28","2025-03-31",

    # April 2025
    "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
    "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
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
    "2025-06-23","2025-06-24","2025-06-25","2025-06-26","2025-06-27","2025-06-30",

    # July 2025
    "2025-07-01","2025-07-02","2025-07-03","2025-07-04","2025-07-07","2025-07-08","2025-07-09","2025-07-10",
    "2025-07-11","2025-07-14","2025-07-15","2025-07-16","2025-07-17","2025-07-18","2025-07-21","2025-07-22",
    "2025-07-23","2025-07-24","2025-07-25","2025-07-28","2025-07-29","2025-07-30","2025-07-31",

    # August 2025
    "2025-08-01","2025-08-04","2025-08-05","2025-08-06","2025-08-07","2025-08-08","2025-08-11","2025-08-12",
    "2025-08-13","2025-08-14","2025-08-18","2025-08-19","2025-08-20","2025-08-21","2025-08-22","2025-08-25",
    "2025-08-26","2025-08-28","2025-08-29"

]


# =========================
# SMALL HELPERS
# =========================
def _ensure_ist(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)

def _session_end_for(ts: pd.Timestamp) -> pd.Timestamp:
    d = _ensure_ist(ts).date()
    return IST.localize(datetime.combine(d, SESSION_END))

def _price_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == "price":
            return c
    return None

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the actual column name matching any candidate (case/underscore/space-insensitive)."""
    norm = {str(c).strip().lower().replace("_"," ").replace("-"," "): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace("_"," ").replace("-"," ")
        if key in norm:
            return norm[key]
    return None


# =========================
# 1) PAPERTRADE NORMALIZER (qty & total value)
# =========================
def normalize_papertrade_qty_totalvalue(file_path: str) -> bool:
    """
    Ensures 'Quantity' (or 'Qty') and 'Total value' columns exist/are recalculated from Price.
    qty = int(100000 / Price) if Price > 0 else 0
    total = qty * Price
    Returns True if updated/saved; False if skipped.
    """
    if not os.path.exists(file_path):
        return False

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.warning(f"[PT-NORM] read fail {file_path}: {e}")
        return False

    if df.empty:
        return False

    price_col = _price_col(df)
    if price_col is None:
        logging.info(f"[PT-NORM] no Price column: {file_path}")
        return False

    # Coerce price
    price = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)

    # Compute
    qty = np.where(price > 0, np.floor(100000.0 / price).astype(int), 0)
    total_val = qty * price

    # Quantity column name (prefer existing)
    qty_col = _find_col(df, ["quantity", "qty"]) or "Quantity"
    if qty_col in df.columns:
        df[qty_col] = qty
    else:
        insert_at = list(df.columns).index(price_col) + 1
        df.insert(insert_at, qty_col, qty)

    # Total value (respect your lowercase v usage)
    tv_col = _find_col(df, ["total value", "total_value", "total Value", "Total value", "Total Value"]) or "Total value"
    if tv_col in df.columns:
        df[tv_col] = total_val.round(2)
    else:
        insert_at = list(df.columns).index(qty_col) + 1
        df.insert(insert_at, tv_col, total_val.round(2))

    try:
        df.to_csv(file_path, index=False)
        logging.info(f"[PT-NORM] updated {file_path} (rows={len(df)})")
        return True
    except Exception as e:
        logging.warning(f"[PT-NORM] write fail {file_path}: {e}")
        return False


def normalize_all_papertrades(dates: List[str]) -> None:
    count = 0
    for d in dates:
        path = f"{PAPERTRADE_PREFIX}{d}.csv"
        if os.path.exists(path):
            if normalize_papertrade_qty_totalvalue(path):
                count += 1
    logging.info(f"[PT-NORM] normalized {count} papertrade file(s).")


# =========================
# 2) KITE SESSION & DATA
# =========================
def setup_kite_session() -> KiteConnect:
    try:
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()
        with open("api_key.txt", "r") as f:
            key, *_ = f.read().split()

        kite = KiteConnect(api_key=key)
        kite.set_access_token(access_token)
        logging.info("Kite session established.")
        return kite
    except Exception as e:
        logging.error(f"Failed to establish Kite session: {e}")
        raise

def fetch_instruments(kite: KiteConnect) -> pd.DataFrame:
    try:
        inst = kite.instruments("NSE")
        df = pd.DataFrame(inst)
        logging.info("Fetched NSE instruments.")
        return df
    except Exception as e:
        logging.error(f"Error fetching instruments: {e}")
        raise

def instrument_lookup(instrument_df: pd.DataFrame, symbol: str) -> int:
    try:
        return int(instrument_df.loc[instrument_df["tradingsymbol"] == symbol, "instrument_token"].values[0])
    except Exception:
        logging.warning(f"Instrument not found for {symbol}")
        return -1


# =========================
# 3) BACKTEST (vectorized long/short)
# =========================
def _first_true_idx(mask: np.ndarray) -> Optional[int]:
    if mask is None or len(mask) == 0:
        return None
    return int(np.argmax(mask)) if mask.any() else None

def fetch_historical_price(
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
    ticker: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    token = instrument_lookup(instrument_df, ticker)
    if token == -1:
        return pd.DataFrame()

    try:
        from_str = _ensure_ist(pd.Timestamp(start_time)).strftime("%Y-%m-%d %H:%M:%S")
        to_str   = _ensure_ist(pd.Timestamp(end_time)).strftime("%Y-%m-%d %H:%M:%S")
        data = kite.historical_data(token, from_date=from_str, to_date=to_str, interval=interval)
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df["date"] = df["date"].apply(_ensure_ist)
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logging.warning(f"[{ticker}] historical_data error: {e}")
        return pd.DataFrame()

def simulate_long(
    ticker: str,
    trade_time: pd.Timestamp,
    buy_price: float,
    qty: int,
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    start_t = _ensure_ist(pd.Timestamp(trade_time))
    end_t   = _session_end_for(start_t)
    df = fetch_historical_price(kite, instrument_df, ticker, start_t, end_t, INTERVAL)
    if df.empty:
        logging.info(f"[{ticker}] No hist data.")
        return None

    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    times = df["date"].to_numpy()

    target_px = buy_price * (1 + BULL_TARGET_PCT)
    stop_px   = buy_price * (1 - BULL_STOP_LOSS_PCT)   # ⬅️ separate bull SL

    tp_idx = _first_true_idx(high >= target_px)
    sl_idx = _first_true_idx(low  <= stop_px)

    if tp_idx is None and sl_idx is None:
        exit_idx, exit_px, reason = len(df) - 1, float(close[-1]), "EOD"
    elif tp_idx is None:
        exit_idx, exit_px, reason = sl_idx, float(stop_px), "SL"
    elif sl_idx is None:
        exit_idx, exit_px, reason = tp_idx, float(target_px), "TP"
    else:
        if sl_idx <= tp_idx:
            exit_idx, exit_px, reason = sl_idx, float(stop_px), "SL"
        else:
            exit_idx, exit_px, reason = tp_idx, float(target_px), "TP"

    exit_time = pd.Timestamp(times[exit_idx])
    total_bought = float(qty * buy_price)
    total_sold   = float(qty * exit_px)
    pnl          = total_sold - total_bought
    pct          = (pnl / total_bought) * 100.0 if total_bought else 0.0

    return {
        "Ticker": ticker,
        "date": exit_time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "Shares Sold": qty,
        "Total Buy": round(total_bought, 2),
        "Total Sold": round(total_sold, 2),
        "Percentage Profit/Loss": f"{pct:.2f}%",
        "Trend Type": "Bullish",
        "ExitReason": reason,
    }


def simulate_short(
    ticker: str,
    trade_time: pd.Timestamp,
    sell_price: float,
    qty: int,
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    start_t = _ensure_ist(pd.Timestamp(trade_time))
    end_t   = _session_end_for(start_t)
    df = fetch_historical_price(kite, instrument_df, ticker, start_t, end_t, INTERVAL)
    if df.empty:
        logging.info(f"[{ticker}] No hist data.")
        return None

    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    times = df["date"].to_numpy()

    target_px = sell_price * (1 - BEAR_TARGET_PCT)
    stop_px   = sell_price * (1 + BEAR_STOP_LOSS_PCT)   # ⬅️ separate bear SL

    tp_idx = _first_true_idx(low  <= target_px)
    sl_idx = _first_true_idx(high >= stop_px)

    if tp_idx is None and sl_idx is None:
        exit_idx, exit_px, reason = len(df) - 1, float(close[-1]), "EOD"
    elif tp_idx is None:
        exit_idx, exit_px, reason = sl_idx, float(stop_px), "SL"
    elif sl_idx is None:
        exit_idx, exit_px, reason = tp_idx, float(target_px), "TP"
    else:
        if sl_idx <= tp_idx:
            exit_idx, exit_px, reason = sl_idx, float(stop_px), "SL"
        else:
            exit_idx, exit_px, reason = tp_idx, float(target_px), "TP"

    exit_time = pd.Timestamp(times[exit_idx])
    total_sold   = float(qty * sell_price)
    total_bought = float(qty * exit_px)
    pnl          = total_sold - total_bought
    pct          = (pnl / total_sold) * 100.0 if total_sold else 0.0

    return {
        "Ticker": ticker,
        "date": exit_time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "Shares Bought Back": qty,
        "Total Sold": round(total_sold, 2),
        "Total Bought": round(total_bought, 2),
        "Percentage Profit/Loss": f"{pct:.2f}%",
        "Trend Type": "Bearish",
        "ExitReason": reason,
    }



# =========================
# 4) IO HELPERS & DAILY NET %
# =========================
def locked_append_rows(csv_path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    lock_path = csv_path + ".lock"
    try:
        with FileLock(lock_path, timeout=15):
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                for r in rows:
                    writer.writerow(r)
    except Timeout:
        logging.error(f"Timeout acquiring lock for {csv_path}. Skipping append of {len(rows)} rows.")
    except Exception as e:
        logging.error(f"Error appending to {csv_path}: {e}")

def calculate_net_percentage_for_day(trading_day: str) -> Optional[float]:
    file_path = f"{RESULT_PREFIX}{trading_day}.csv"
    out_file  = f"{RESULT_FINAL_PREFIX}{trading_day}.csv"
    if not os.path.exists(file_path):
        logging.info(f"No results for {trading_day} to summarize.")
        return None

    df = pd.read_csv(file_path)
    if df.empty or "Percentage Profit/Loss" not in df.columns:
        logging.info(f"No rows in {file_path} to summarize.")
        return None

    df["Percentage Profit/Loss"] = df["Percentage Profit/Loss"].astype(str).str.rstrip("%").astype(float)
    results = []
    total_pct = 0.0
    n_tickers = 0

    for ticker, g in df.groupby("Ticker"):
        g = g.sort_values("date")
        if len(g) > 1:
            pct = (g.iloc[0]["Percentage Profit/Loss"] * 2 + g.iloc[1]["Percentage Profit/Loss"]) / 2.0
        else:
            pct = g.iloc[0]["Percentage Profit/Loss"]
        results.append({"Ticker": ticker, "Net Percentage": round(pct, 4)})
        total_pct += pct
        n_tickers += 1

    overall = round(total_pct / n_tickers, 4) if n_tickers else 0.0
    results.append({"Ticker": "Overall", "Net Percentage": overall})

    pd.DataFrame(results).to_csv(out_file, index=False)
    logging.info(f"[{trading_day}] Net Percentage: {overall}% (saved to {out_file})")
    return overall


# =========================
# 5) BATCH ENGINE
# =========================
def chunked(df: pd.DataFrame, size: int) -> List[pd.DataFrame]:
    n = len(df)
    if n == 0:
        return []
    return [df.iloc[i : i + size].copy() for i in range(0, n, size)]

def _normalize_trade_row(row: pd.Series) -> Tuple[str, float, int, pd.Timestamp, str]:
    ticker = str(row.get("Ticker", "")).strip().upper()
    price  = pd.to_numeric(row.get("Price", np.nan), errors="coerce")
    qty    = pd.to_numeric(row.get(_find_col(pd.DataFrame([row]), ["quantity", "qty"]) or "Quantity", np.nan), errors="coerce")
    trend  = str(row.get("Trend Type", "")).strip().lower()

    t_raw = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(t_raw):
        raise ValueError("Invalid trade timestamp")
    trade_time = _ensure_ist(t_raw)

    if (pd.isna(qty) or qty <= 0) and not pd.isna(price) and price > 0:
        qty = int(100000 / float(price))
    if pd.isna(price) or price <= 0 or qty <= 0:
        raise ValueError("Invalid price/quantity")
    return ticker, float(price), int(qty), trade_time, trend

def process_batch(batch_df: pd.DataFrame, kite: KiteConnect, instrument_df: pd.DataFrame, trading_day: str) -> None:
    if batch_df.empty:
        return
    result_path = f"{RESULT_PREFIX}{trading_day}.csv"
    headers = [
        "Ticker", "date",
        "Shares Sold", "Total Buy", "Total Sold",
        "Shares Bought Back", "Total Bought",
        "Percentage Profit/Loss", "Trend Type", "ExitReason"
    ]
    rows_out: List[Dict[str, Any]] = []

    def worker(row: pd.Series) -> Optional[Dict[str, Any]]:
        try:
            ticker, px, qty, ttime, trend = _normalize_trade_row(row)
            if trend == "bullish":
                return simulate_long(ticker, ttime, px, qty, kite, instrument_df)
            elif trend == "bearish":
                return simulate_short(ticker, ttime, px, qty, kite, instrument_df)
            else:
                return None
        except Exception as e:
            logging.warning(f"Row skipped due to error: {e}")
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker, r) for _, r in batch_df.iterrows()]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                rows_out.append(res)

    if rows_out:
        locked_append_rows(result_path, headers, rows_out)

def run_backtest_for_day(trading_day: str, kite: KiteConnect, instrument_df: pd.DataFrame) -> None:
    per_trade_path = f"{RESULT_PREFIX}{trading_day}.csv"
    final_path     = f"{RESULT_FINAL_PREFIX}{trading_day}.csv"

    if SKIP_IF_RESULTS_EXIST and (os.path.exists(per_trade_path) or os.path.exists(final_path)):
        logging.info(f"[{trading_day}] Results already exist. Skipping.")
        return

    file_path = f"{PAPERTRADE_PREFIX}{trading_day}.csv"
    if not os.path.exists(file_path):
        logging.info(f"[{trading_day}] No papertrade file: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"[{trading_day}] Failed reading {file_path}: {e}")
        return

    req = {"Ticker", "date", "Trend Type", "Price"}
    if not req.issubset(df.columns):
        logging.warning(f"[{trading_day}] Missing required columns. Skipping.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["date_ist"] = df["date"].apply(_ensure_ist)
    d0 = IST.localize(datetime.combine(datetime.strptime(trading_day, "%Y-%m-%d").date(), SESSION_START))
    d1 = IST.localize(datetime.combine(datetime.strptime(trading_day, "%Y-%m-%d").date(), SESSION_END))
    df = df[(df["date_ist"] >= d0) & (df["date_ist"] <= d1)].copy()
    if df.empty:
        logging.info(f"[{trading_day}] No trades in session window.")
        return

    batches = chunked(df, BATCH_SIZE)
    logging.info(f"[{trading_day}] {len(df)} trades → {len(batches)} batches (size={BATCH_SIZE})")

    for i, bdf in enumerate(batches, start=1):
        logging.info(f"[{trading_day}] Batch {i}/{len(batches)}: {len(bdf)} rows")
        process_batch(bdf, kite, instrument_df, trading_day)
        if i < len(batches) and API_SLEEP_BETWEEN_BATCHES > 0:
            time.sleep(API_SLEEP_BETWEEN_BATCHES)

    calculate_net_percentage_for_day(trading_day)


# =========================
# 6) PURGE
# =========================
def purge_previous_results(patterns: List[str]) -> None:
    total_deleted = 0
    total_lock_deleted = 0
    for pat in patterns:
        for fp in glob.glob(os.path.join(CWD, pat)):
            try:
                os.remove(fp); total_deleted += 1
                logging.info(f"Deleted file: {fp}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.warning(f"Could not delete {fp}: {e}")
            lock_fp = fp + ".lock"
            if os.path.exists(lock_fp):
                try:
                    os.remove(lock_fp); total_lock_deleted += 1
                    logging.info(f"Deleted lock: {lock_fp}")
                except Exception as e:
                    logging.warning(f"Could not delete lock {lock_fp}: {e}")
    logging.info(f"Purge complete. Deleted {total_deleted} result files and {total_lock_deleted} lock files.")


# =========================
# 7) SUMMARY (code2 merged; tolerant to 'Total value'/'Total Value')
# =========================
def _read_result_df(date: str) -> Optional[pd.DataFrame]:
    filename = f"{RESULT_FINAL_PREFIX}{date}.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found for date {date}.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        return None

    if 'Ticker' not in df.columns or 'Net Percentage' not in df.columns:
        print(f"Error: required columns missing in {filename}")
        return None

    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    df['Net Percentage'] = pd.to_numeric(df['Net Percentage'], errors='coerce')
    df = df[df['Ticker'].str.lower() != 'overall'].copy()
    df = df.drop_duplicates(subset=['Ticker'], keep='first')
    return df

def _read_papertrade_df(date: str) -> Optional[pd.DataFrame]:
    filename = f"{PAPERTRADE_PREFIX}{date}.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found for date {date}.")
        return None
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None

    need_ticker  = _find_col(df, ["ticker"])
    need_trend   = _find_col(df, ["trend type","trend"])
    need_total   = _find_col(df, ["total value","total_value","Total value","Total Value"])

    missing = []
    if not need_ticker: missing.append("Ticker")
    if not need_trend:  missing.append("Trend Type")
    if not need_total:  missing.append("Total value")
    if missing:
        print(f"Error: Missing columns {missing} in {filename}")
        return None

    df[need_ticker] = df[need_ticker].astype(str).str.strip()
    df[need_trend]  = df[need_trend].astype(str).str.strip().str.title()
    df[need_total]  = pd.to_numeric(df[need_total], errors='coerce').fillna(0.0)
    df = df.rename(columns={need_ticker:"Ticker", need_trend:"Trend Type", need_total:"Total value"})
    df = df.drop_duplicates(subset=['Ticker'], keep='first')
    return df

def _join_for_date(date: str) -> Optional[pd.DataFrame]:
    pt = _read_papertrade_df(date)
    rs = _read_result_df(date)
    if pt is None or rs is None:
        return None
    merged = pd.merge(
        pt[['Ticker', 'Trend Type', 'Total value']],
        rs[['Ticker', 'Net Percentage']],
        on='Ticker',
        how='left'
    )
    merged['Net Percentage'] = merged['Net Percentage'].fillna(0.0)
    merged['PL'] = (merged['Net Percentage'] / 100.0) * merged['Total value']
    return merged

def sum_net_percentage_for_date(date: str) -> Optional[float]:
    merged = _join_for_date(date)
    if merged is None:
        return None
    return float(merged[merged['Ticker'].str.lower()!='overall']['Net Percentage'].sum())

def sum_total_value_for_date(date: str) -> float:
    pt = _read_papertrade_df(date)
    if pt is None:
        return 0.0
    return float(pt['Total value'].sum())

def count_tickers_for_date(date: str) -> int:
    pt = _read_papertrade_df(date)
    if pt is None:
        return 0
    return int(pt['Ticker'].nunique())

def _compute_pl_breakout(merged: pd.DataFrame, margin_mult: float):
    pl_total = float(merged['PL'].sum())
    pl_margins_total = pl_total * margin_mult

    bull = merged[merged['Trend Type'].str.lower() == 'bullish']
    bear = merged[merged['Trend Type'].str.lower() == 'bearish']

    pl_bull = float(bull['PL'].sum()) if not bull.empty else 0.0
    pl_bear = float(bear['PL'].sum()) if not bear.empty else 0.0

    return {
        'PL_total': pl_total,
        'PL_margins_total': pl_margins_total,
        'PL_bull': pl_bull,
        'PL_bear': pl_bear,
        'PL_margins_bull': pl_bull * margin_mult,
        'PL_margins_bear': pl_bear * margin_mult,
    }

def process_multiple_dates(
    dates: List[str],
    margin_mult: float = MARGIN_MULTIPLIER,
    output_csv_path: str = "summary_5min_resultsv5.csv",
    pretty_csv_path: Optional[str] = "summary_5min_results_prettyv5.csv",
    print_console: bool = True
) -> pd.DataFrame:

    rows = []
    for date in dates:
        merged = _join_for_date(date)
        if merged is None or merged.empty:
            rows.append({
                "Date": date,
                "Total Value": 0.0,
                "Total Net %": 0.0,
                "Avg %": 0.0,
                "TC": 0,
                # NEW: bull/bear %
                "Total Net % (Bu)": 0.0,
                "Avg % (Bu)": 0.0,
                "TC (Bu)": 0,
                "Total Net % (Be)": 0.0,
                "Avg % (Be)": 0.0,
                "TC (Be)": 0,
                # Existing P/L sections
                "P/L": 0.0,
                "P/L (M)": 0.0,
                "P/L (Bu)": 0.0,
                "P/L (M) (Bu)": 0.0,
                "P/L (Be)": 0.0,
                "P/L (M) (Be)": 0.0,
            })
            continue

        total_value = float(merged['Total value'].sum())

        # Overall
        mask_overall = merged['Ticker'].astype(str).str.strip().str.casefold().eq('overall')
        total_net_percentage = pd.to_numeric(
            merged.loc[~mask_overall, 'Net Percentage'], errors='coerce'
        ).fillna(0).sum()
        ticker_count = merged.loc[~mask_overall, 'Ticker'].nunique()
        avg_percentage = (total_net_percentage / ticker_count) if ticker_count else 0.0

        # ---- NEW: Bull/Bear splits (by Trend Type from papertrade) ----
        tt = merged['Trend Type'].astype(str).str.strip().str.casefold()
        bull_mask = tt.eq('bullish') & (~mask_overall)
        bear_mask = tt.eq('bearish') & (~mask_overall)

        total_net_pct_bull = pd.to_numeric(
            merged.loc[bull_mask, 'Net Percentage'], errors='coerce'
        ).fillna(0).sum()
        total_net_pct_bear = pd.to_numeric(
            merged.loc[bear_mask, 'Net Percentage'], errors='coerce'
        ).fillna(0).sum()

        ticker_count_bull = merged.loc[bull_mask, 'Ticker'].nunique()
        ticker_count_bear = merged.loc[bear_mask, 'Ticker'].nunique()

        avg_pct_bull = (total_net_pct_bull / ticker_count_bull) if ticker_count_bull else 0.0
        avg_pct_bear = (total_net_pct_bear / ticker_count_bear) if ticker_count_bear else 0.0

        # P/L breakdowns (existing)
        pl_parts = _compute_pl_breakout(merged, margin_mult)

        rows.append({
            "Date": date,
            "Total Value": total_value,
            "Total Net %": total_net_percentage,
            "Avg %": avg_percentage,
            "TC": ticker_count,
            # NEW: Bull/Bear %
            "Total Net % (Bu)": total_net_pct_bull,
            "Avg % (Bu)": avg_pct_bull,
            "TC (Bu)": ticker_count_bull,
            "Total Net % (Be)": total_net_pct_bear,
            "Avg % (Be)": avg_pct_bear,
            "TC (Be)": ticker_count_bear,
            # Existing P/L sections
            "P/L": pl_parts['PL_total'],
            "P/L (M)": pl_parts['PL_margins_total'],
            "P/L (Bu)": pl_parts['PL_bull'],
            "P/L (M) (Bu)": pl_parts['PL_margins_bull'],
            "P/L (Be)": pl_parts['PL_bear'],
            "P/L (M) (Be)": pl_parts['PL_margins_bear'],
        })

    df_results = pd.DataFrame(rows)

    # Totals row
    totals_row = {"Date": "TOTAL"}
    sum_cols = [
        "Total Value", "Total Net %", "TC",
        "Total Net % (Bu)", "TC (Bu)",
        "Total Net % (Be)", "TC (Be)",
        "P/L", "P/L (M)",
        "P/L (Bu)", "P/L (M) (Bu)",
        "P/L (Be)", "P/L (M) (Be)"
    ]
    for col in sum_cols:
        totals_row[col] = df_results[col].sum()

    # Weighted averages for TOTALS (overall + bull + bear)
    totals_row["Avg %"] = (
        (totals_row["Total Net %"] / totals_row["TC"])
        if totals_row.get("TC", 0) else 0.0
    )
    totals_row["Avg % (Bu)"] = (
        (totals_row["Total Net % (Bu)"] / totals_row["TC (Bu)"])
        if totals_row.get("TC (Bu)", 0) else 0.0
    )
    totals_row["Avg % (Be)"] = (
        (totals_row["Total Net % (Be)"] / totals_row["TC (Be)"])
        if totals_row.get("TC (Be)", 0) else 0.0
    )

    df_results = pd.concat([df_results, pd.DataFrame([totals_row])], ignore_index=True)

    # Write numeric CSV
    df_results.to_csv(output_csv_path, index=False)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved numeric summary to: {output_csv_path}")

    # Pretty copy (for console + optional pretty CSV)
    df_pretty = df_results.copy()

    _f2 = lambda x: (
        f"{float(x):,.2f}"
        if (pd.notna(x) and isinstance(x, (int, float, np.floating)))
        else x
    )
    _fi = lambda x: (
        f"{int(x):,d}"
        if (pd.notna(x) and isinstance(x, (int, float, np.floating)) and float(x).is_integer())
        else x
    )

    float_cols = [
        "Total Value","Total Net %","Avg %",
        "Total Net % (Bu)","Avg % (Bu)",
        "Total Net % (Be)","Avg % (Be)",
        "P/L","P/L (M)",
        "P/L (Bu)","P/L (M) (Bu)",
        "P/L (Be)","P/L (M) (Be)"
    ]
    for c in float_cols:
        if c in df_pretty.columns:
            df_pretty[c] = df_pretty[c].apply(_f2)

    int_cols = ["TC", "TC (Bu)", "TC (Be)"]
    for c in int_cols:
        if c in df_pretty.columns:
            df_pretty[c] = df_pretty[c].apply(_fi)

    if print_console:
        print("\n===== Summary (Pretty) =====")
        print(df_pretty.to_string(index=False))

        df_days = df_results[df_results["Date"] != "TOTAL"].copy()
        pos_days = int((df_days["P/L (M)"] > 0).sum())
        neg_days = int((df_days["P/L (M)"] < 0).sum())
        flat_days = int((df_days["P/L (M)"] == 0).sum())
        total_days = len(df_days)
        print("\n----- Day Outcome Counts (by P/L (M)) -----")
        print(f"Positive days : {pos_days}")
        print(f"Negative days : {neg_days}")
        print(f"Flat days     : {flat_days}")
        if total_days:
            win_rate = (pos_days / total_days) * 100.0
            print(f"Win rate      : {win_rate:.2f}%  ({pos_days}/{total_days})")
        ratio_str = f"{(pos_days / neg_days):.2f}" if neg_days > 0 else ("∞" if pos_days > 0 else "0.00")
        print(f"Positive : Negative (ratio) = {pos_days}:{neg_days} ({ratio_str})")

        total_pl_margins_all = float(df_days["P/L (M)"].sum())
        total_total_value_all = float(df_days["Total Value"].sum())
        pct_total_pl = (total_pl_margins_all / total_total_value_all * 100.0) if total_total_value_all else 0.0
        print(f"% total_P/L : {pct_total_pl:.2f}%  ( {total_pl_margins_all:,.2f} / {total_total_value_all:,.2f} )")

    if pretty_csv_path:
        df_pretty.to_csv(pretty_csv_path, index=False)
        print(f"Saved pretty-formatted summary to: {pretty_csv_path}")

    return df_results



# =========================
# 8) ONE-CYCLE RUNNER (normalize → backtest → summaries)
# =========================
_FIRST_RUN_DONE = False

def run_cycle():
    global _FIRST_RUN_DONE

    # Purge only once at start (if enabled)
    if PURGE_RESULTS_ON_FIRST_RUN and not _FIRST_RUN_DONE:
        logging.info("Purging previous results before first execution…")
        purge_previous_results(PURGE_PATTERNS)

    # Normalize papertrade files (Quantity & Total value)
    normalize_all_papertrades(LAST_TRADING_DAYS)

    # Backtest
    logging.info("Starting batched backtest (v5)…")
    kite = setup_kite_session()
    instrument_df = fetch_instruments(kite)

    for d in LAST_TRADING_DAYS:
        logging.info(f"\n=== Processing {d} ===")
        try:
            run_backtest_for_day(d, kite, instrument_df)
        except Exception as e:
            logging.error(f"Fatal error on {d}: {e}")

    # Summaries
    process_multiple_dates(
        LAST_TRADING_DAYS,
        margin_mult=MARGIN_MULTIPLIER,
        output_csv_path="summary_5min_resultsv5.csv",
        pretty_csv_path="summary_5min_results_prettyv5.csv",
        print_console=True
    )

    _FIRST_RUN_DONE = True
    logging.info("Cycle complete.")


# =========================
# 9) MAIN (scheduler)
# =========================
if __name__ == "__main__":
    try:
        if RUN_EVERY_5_MIN:
            print("Starting 5-minute scheduler. Press Ctrl+C to stop.")
            while True:
                print("\n" + "=" * 80)
                print(f"Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                run_cycle()
                print("=" * 80)
                print(f"Next run in {SLEEP_SECONDS//60} minutes...")
                time.sleep(SLEEP_SECONDS)
        else:
            run_cycle()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        logging.shutdown()
