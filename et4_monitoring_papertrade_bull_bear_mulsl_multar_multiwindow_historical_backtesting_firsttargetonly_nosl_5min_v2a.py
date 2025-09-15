# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:26:33 2024
Rewritten on request for robust, batched backtests with 100+ entries (NO SL).

Author: Saarit

What changed:
- Headless backtest (no Tkinter).
- Batch execution (default 25) with capped concurrency (default 8).
- Vectorized exit logic on 1-min bars (Target or EOD only; NO SL).
- Thread-safe, locked appends to result files.
- Solid timezone handling and CSV hygiene.
"""

import os
import csv
import time
import logging
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import pytz

try:
    from filelock import FileLock, Timeout
except Exception:
    # Minimal fallback if filelock isn't available
    class FileLock:
        def __init__(self, filename, timeout=10): self.filename = filename
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Timeout(Exception): pass

from kiteconnect import KiteConnect

# =========================
# CONFIG
# =========================
CWD = r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo"
os.chdir(CWD)

RESULT_PREFIX  = "result_papertradesl_v3_"        # per-day detailed results
RESULT_FINAL_PREFIX = "result_finalsl_5min_v3_"   # per-day summary

# Backtest session window (IST)
IST = pytz.timezone("Asia/Kolkata")
SESSION_START = dt_time(9, 25)
SESSION_END   = dt_time(15, 15)

# Execution control
BATCH_SIZE   = 25          # entries per batch
MAX_WORKERS  = 8           # threads per batch
PAUSE_BETWEEN_BATCHES = 0.4  # seconds

# Trade params (1-minute timeframe); NO SL — only TP or EOD
INTERVAL = "minute"
BULL_TARGET_PCT = 0.01     # +2% TP for longs
BEAR_TARGET_PCT = 0.01     # -2% TP for shorts

# Logging
logging.basicConfig(
    filename="trading_log_batched.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

# =========================
# RESULT SCHEMA
# (kept compact/unique headers for DictWriter)
# =========================
RESULT_COLUMNS = [
    "Ticker","date",
    "Shares Sold","Total Buy","Total Sold",
    "Shares Bought Back","Total Bought",
    "Percentage Profit/Loss","Trend Type","ExitReason"
]

def _ensure_result_file(path: str):
    """Create results file with the schema if missing."""
    if not os.path.isfile(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(RESULT_COLUMNS)

def _pct_str(x: float) -> str:
    return f"{x:.2f}%"

# =========================
# SESSION / INSTRUMENTS
# =========================
def setup_kite_session() -> KiteConnect:
    """Establish a Kite Connect session from local tokens."""
    try:
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()
        with open("api_key.txt", "r") as f:
            key = f.read().split()[0]

        kite = KiteConnect(api_key=key)
        kite.set_access_token(access_token)
        logging.info("Kite session established.")
        return kite
    except Exception as e:
        logging.error(f"Failed to establish Kite session: {e}")
        raise

def fetch_instruments(kite: KiteConnect) -> pd.DataFrame:
    """Fetch NSE instruments once; cache in memory."""
    try:
        inst = kite.instruments("NSE")
        df = pd.DataFrame(inst)
        logging.info("Fetched NSE instruments.")
        return df
    except Exception as e:
        logging.error(f"Error fetching instruments: {e}")
        raise

def instrument_lookup(instrument_df: pd.DataFrame, symbol: str) -> int:
    """Look up instrument token; return -1 on failure."""
    try:
        return int(instrument_df.loc[instrument_df["tradingsymbol"] == symbol, "instrument_token"].values[0])
    except Exception:
        logging.warning(f"Instrument not found for {symbol}")
        return -1

# =========================
# TIME HELPERS
# =========================
def _ensure_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Ensure a timestamp is IST tz-aware."""
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)

def _session_end_for(ts: pd.Timestamp) -> pd.Timestamp:
    """Return session end datetime for ts's date in IST."""
    d = _ensure_ist(ts).date()
    return IST.localize(datetime.combine(d, SESSION_END))

# =========================
# DATA
# =========================
def fetch_historical_price(
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
    ticker: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Fetch historical candle data [start_time, end_time] (1-minute bars).
    Returns empty DF on failure.
    """
    token = instrument_lookup(instrument_df, ticker)
    if token == -1:
        return pd.DataFrame()

    try:
        from_str = _ensure_ist(pd.Timestamp(start_time)).strftime("%Y-%m-%d %H:%M:%S")
        to_str   = _ensure_ist(pd.Timestamp(end_time)).strftime("%Y-%m-%d %H:%M:%S")
        data = kite.historical_data(
            token,
            from_date=from_str,
            to_date=to_str,
            interval=interval,
        )
        df = pd.DataFrame(data)
        if df.empty:
            return df
        # Normalize to IST
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df["date"] = df["date"].apply(_ensure_ist)
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logging.warning(f"[{ticker}] historical_data error: {e}")
        return pd.DataFrame()

# =========================
# SIMULATION (Target or EOD only)
# =========================
def _first_true_idx(mask: np.ndarray) -> Optional[int]:
    """Return index of first True, else None."""
    if not mask.any():
        return None
    return int(np.argmax(mask))  # first True

def simulate_long(
    ticker: str,
    trade_time: pd.Timestamp,
    buy_price: float,
    qty: int,
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Long trade (NO SL):
      - Exit at +BULL_TARGET_PCT on close, else EOD close.
    """
    start_t = _ensure_ist(pd.Timestamp(trade_time))
    end_t   = _session_end_for(start_t)

    df = fetch_historical_price(kite, instrument_df, ticker, start_t, end_t, INTERVAL)
    if df.empty:
        logging.info(f"[{ticker}] No hist data.")
        return None

    close = df["close"].to_numpy(dtype=float)
    times = df["date"].to_numpy()

    target_px = buy_price * (1 + BULL_TARGET_PCT)

    hit_target_idx = _first_true_idx(close >= target_px)

    if hit_target_idx is None:
        exit_idx = len(df) - 1
        hit_type = "EOD"
    else:
        exit_idx = hit_target_idx
        hit_type = "TP"

    exit_time  = pd.Timestamp(times[exit_idx])
    exit_price = float(close[exit_idx])

    total_bought = float(qty * buy_price)
    total_sold   = float(qty * exit_price)
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
        "ExitReason": hit_type,
    }

def simulate_short(
    ticker: str,
    trade_time: pd.Timestamp,
    sell_price: float,
    qty: int,
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Short trade (NO SL):
      - Exit at -BEAR_TARGET_PCT on close, else EOD close.
    """
    start_t = _ensure_ist(pd.Timestamp(trade_time))
    end_t   = _session_end_for(start_t)

    df = fetch_historical_price(kite, instrument_df, ticker, start_t, end_t, INTERVAL)
    if df.empty:
        logging.info(f"[{ticker}] No hist data.")
        return None

    close = df["close"].to_numpy(dtype=float)
    times = df["date"].to_numpy()

    target_px = sell_price * (1 - BEAR_TARGET_PCT)

    hit_target_idx = _first_true_idx(close <= target_px)

    if hit_target_idx is None:
        exit_idx = len(df) - 1
        hit_type = "EOD"
    else:
        exit_idx = hit_target_idx
        hit_type = "TP"

    exit_time  = pd.Timestamp(times[exit_idx])
    exit_price = float(close[exit_idx])

    total_sold   = float(qty * sell_price)
    total_bought = float(qty * exit_price)
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
        "ExitReason": hit_type,
    }

# =========================
# IO HELPERS
# =========================
def locked_append_rows(csv_path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    """Append rows to CSV with a file lock; create file with header if missing."""
    if not rows:
        return
    _ensure_result_file(csv_path)
    lock_path = csv_path + ".lock"
    try:
        with FileLock(lock_path, timeout=15):
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                # header already written by _ensure_result_file
                for r in rows:
                    writer.writerow(r)
    except Timeout:
        logging.error(f"Timeout acquiring lock for {csv_path}. Skipping append of {len(rows)} rows.")
    except Exception as e:
        logging.error(f"Error appending to {csv_path}: {e}")

def calculate_net_percentage_for_day(trading_day: str) -> Optional[float]:
    """
    Compute per-day net percentage across tickers.
    Rule: if a ticker has 2 rows, (first*2 + second)/2, else single row.
    """
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
# BATCH ENGINE
# =========================
def chunked(df: pd.DataFrame, size: int) -> List[pd.DataFrame]:
    """Yield DataFrame chunks of roughly 'size' rows."""
    n = len(df)
    if n == 0:
        return []
    return [df.iloc[i : i + size].copy() for i in range(0, n, size)]

def _normalize_trade_row(row: pd.Series) -> Tuple[str, float, int, pd.Timestamp, str]:
    """
    Extract (ticker, price, quantity, trade_time_IST, trend).
    Ensures quantity/price numeric; infers qty if missing (₹20k/price).
    """
    ticker = str(row.get("Ticker", "")).strip().upper()
    price  = pd.to_numeric(row.get("Price", np.nan), errors="coerce")
    qty    = pd.to_numeric(row.get("Quantity", np.nan), errors="coerce")
    trend  = str(row.get("Trend Type", "")).strip().lower()

    # Date normalize
    t_raw = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(t_raw):
        raise ValueError("Invalid trade timestamp")
    trade_time = _ensure_ist(t_raw)

    # Quantity rule
    if (pd.isna(qty) or qty <= 0) and not pd.isna(price) and price > 0:
        qty = int(20000 / float(price))  # same sizing rule as papertrade builder

    if pd.isna(price) or price <= 0 or qty <= 0:
        raise ValueError("Invalid price/quantity")

    return ticker, float(price), int(qty), trade_time, trend

def process_batch(
    batch_df: pd.DataFrame,
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
    trading_day: str,
) -> None:
    """
    Run a batch (<= BATCH_SIZE) concurrently with at most MAX_WORKERS threads.
    Results are appended once per batch (file-locked).
    """
    if batch_df.empty:
        return

    result_path = f"{RESULT_PREFIX}{trading_day}.csv"
    _ensure_result_file(result_path)

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
        locked_append_rows(result_path, RESULT_COLUMNS, rows_out)

def run_backtest_for_day(
    trading_day: str,
    kite: KiteConnect,
    instrument_df: pd.DataFrame,
) -> None:
    """
    Load papertrade file for the day and process in batches.
    """
    file_path = f"papertrade_5min_v3_{trading_day}.csv"
    if not os.path.exists(file_path):
        logging.info(f"[{trading_day}] No papertrade file found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"[{trading_day}] Failed reading {file_path}: {e}")
        return

    req = {"Ticker", "date", "Trend Type", "Price"}
    if not req.issubset(df.columns):
        logging.warning(f"[{trading_day}] Missing required columns in {file_path}. Skipping.")
        return

    # Keep only relevant intraday window (safeguard)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["date_ist"] = df["date"].apply(_ensure_ist)

    d0 = IST.localize(datetime.strptime(trading_day, "%Y-%m-%d"))
    d0 = d0.replace(hour=SESSION_START.hour, minute=SESSION_START.minute, second=0, microsecond=0)
    d1 = IST.localize(datetime.strptime(trading_day, "%Y-%m-%d"))
    d1 = d1.replace(hour=SESSION_END.hour, minute=SESSION_END.minute, second=0, microsecond=0)

    df = df[(df["date_ist"] >= d0) & (df["date_ist"] <= d1)].copy()
    if df.empty:
        logging.info(f"[{trading_day}] No trades in session window.")
        return

    # Process in batches
    batches = chunked(df, BATCH_SIZE)
    logging.info(f"[{trading_day}] {len(df)} trades → {len(batches)} batches (size={BATCH_SIZE}, max_workers={MAX_WORKERS})")

    for i, bdf in enumerate(batches, start=1):
        logging.info(f"[{trading_day}] Batch {i}/{len(batches)}: {len(bdf)} rows")
        process_batch(bdf, kite, instrument_df, trading_day)
        if i < len(batches) and PAUSE_BETWEEN_BATCHES > 0:
            time.sleep(PAUSE_BETWEEN_BATCHES)

    # Summarize per-day net %
    calculate_net_percentage_for_day(trading_day)

# =========================
# MAIN
# =========================
def main():
    # Adjust your calendar here
    last_trading_days = [
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
    ]

    logging.info("Starting batched backtest (NO SL)…")
    kite = setup_kite_session()
    instrument_df = fetch_instruments(kite)

    for d in last_trading_days:
        logging.info(f"\n=== Processing {d} ===")
        try:
            run_backtest_for_day(d, kite, instrument_df)
        except Exception as e:
            logging.error(f"Fatal error on {d}: {e}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
    logging.shutdown()
