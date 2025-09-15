# -*- coding: utf-8 -*-
"""
Fast 5-minute fetcher with incremental delta updates, indicator calc, and append-only CSV.

Key speed/storage optimizations:
- Incremental intraday fetch (from last written timestamp + 1 interval) per ticker.
- Historical window (7–10 days) fetched once per day and cached.
- Intraday cache freshness 5m; avoids duplicate API pulls.
- Append-only CSV writes with header-once and last-line duplicate guard via small state file.
- Vectorized indicators (OBV, ADX via Wilder EWM); recompute on last ~200 rows only.
- Concurrency: ThreadPoolExecutor (workers=min(8, N)) + API semaphore (3) + jittered retries.

Robust state (Windows/OneDrive safe):
- State directory defaults to local temp (avoid sync locks), override with ET4_STATE_DIR.
- FileLock serializes cross-thread/process access.
- os.replace() wrapped with retries; final safe fallback write.

Resilient API layer (this rewrite):
- Floor the end time to the previous 5-minute boundary (avoid forming candle empties).
- Soft-fail after retries (no exception): return empty DataFrame and cool down the token briefly.
- When API returns empty, fall back to cache (stale > crash), continue loop quietly.

Schedule:
- Exact 5-minute boundaries from 09:15:00 to 15:19:00 IST.
- Two quick passes each tick, then sleep until the next boundary.
"""

import os
import io
import csv
import sys
import glob
import json
import time
import math
import signal
import random
import logging
import traceback
import threading
import tempfile
import uuid
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import HTTPError
from filelock import FileLock, Timeout

from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException, KiteException

# ================================
# Config
# ================================
india_tz = pytz.timezone("Asia/Kolkata")

# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks_MIS import selected_stocks  # <-- adjust to your list

CACHE_DIR = "data_cache_5min"
INDICATORS_DIR = "main_indicators_5min"
LOGS_DIR = "logs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Robust state (default temp dir to avoid OneDrive locks; override via ENV) ---
STATE_DIR = os.getenv("ET4_STATE_DIR", os.path.join(tempfile.gettempdir(), "et4_state_5min"))
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "last_written.json")
STATE_LOCK_PATH = os.path.join(STATE_DIR, "last_written.json.lock")
STATE_FILE_LOCK = FileLock(STATE_LOCK_PATH, timeout=10)
_state_lock = threading.Lock()  # in-process guard

# Rate limits / concurrency
API_SEMAPHORE = threading.Semaphore(3)  # tune if you still see rate limits
MAX_WORKERS = 8

# Intervals
INTERVAL = "5minute"
INTRADAY_FRESHNESS = timedelta(minutes=5)
HIST_FRESHNESS = timedelta(hours=8)
HIST_DAYS = 9  # fetch this many previous days once; enough for indicators

# Market window
SESSION_START = dt_time(9, 15, 0)
SESSION_END_CAP = dt_time(15, 30, 0)

# Resilient fetch knobs
FAIL_COOLDOWN_SECS = 180  # after repeated failures, skip token for this long
_FAIL_UNTIL: Dict[int, float] = {}  # token -> epoch until which we skip API

# Logging
logger = logging.getLogger("fast_fetch_5m")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOGS_DIR, "fast_fetch_5m.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    fh.setLevel(logging.INFO); ch.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(ch)

print("Logging initialized and script execution started.")

# Market holidays (example; extend as needed)
market_holidays = {
    datetime(2024, 1, 26).date(), datetime(2024, 3, 8).date(),  datetime(2024, 3, 25).date(),
    datetime(2024, 3, 29).date(), datetime(2024, 4, 11).date(), datetime(2024, 4, 17).date(),
    datetime(2024, 5, 1).date(),  datetime(2024, 6, 17).date(), datetime(2024, 7, 17).date(),
    datetime(2024, 8, 15).date(), datetime(2024, 10, 2).date(), datetime(2024, 11, 1).date(),
    datetime(2024, 11, 15).date(), datetime(2024, 11, 20).date(), datetime(2024, 12, 25).date(),
}

# ================================
# Session / tokens
# ================================
def setup_kite_session() -> KiteConnect:
    with open("access_token.txt", "r") as f:
        access_token = f.read().strip()
    with open("api_key.txt", "r") as f:
        key = f.read().split()[0]
    kite = KiteConnect(api_key=key)
    kite.set_access_token(access_token)
    logger.info("Kite session established.")
    return kite

kite = setup_kite_session()

_tokens_lock = threading.Lock()
SHARES_TOKENS: Dict[str, int] = {}

def refresh_tokens_if_needed() -> None:
    global SHARES_TOKENS
    with _tokens_lock:
        if SHARES_TOKENS:
            return
        inst = kite.instruments("NSE")
        df = pd.DataFrame(inst)
        mapping = dict(zip(df["tradingsymbol"].astype(str), df["instrument_token"].astype(int)))
        SHARES_TOKENS = {s: mapping.get(s, None) for s in selected_stocks}

refresh_tokens_if_needed()

# ================================
# Tiny utils
# ================================
def tz_aware(ts: pd.Series, tz="Asia/Kolkata") -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    s = s.dropna()
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        s = s.dt.tz_convert(tz)
    return s

def now_ist() -> datetime:
    return datetime.now(india_tz)

def market_window_for(d: datetime.date) -> Tuple[datetime, datetime]:
    st = india_tz.localize(datetime.combine(d, SESSION_START))
    et_cap = india_tz.localize(datetime.combine(d, SESSION_END_CAP))
    return st, min(now_ist(), et_cap)

def cache_path(ticker: str, kind: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{kind}.csv")

def is_fresh(path: str, thr: timedelta) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path), pytz.UTC)
    return (datetime.now(pytz.UTC) - mtime) < thr

def floor_to_5m(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    return dt - timedelta(minutes=(dt.minute % 5))

# --- Robust state helpers (FileLock + retrying replace) ---
def _safe_json_load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    except Exception:
        return {}

def read_last_written_state() -> Dict[str, str]:
    with STATE_FILE_LOCK:
        return _safe_json_load(STATE_FILE)

def write_last_written_state(state: Dict[str, str]) -> None:
    tmp = f"{STATE_FILE}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    with STATE_FILE_LOCK:
        for attempt in range(6):
            try:
                os.replace(tmp, STATE_FILE)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        # final fallback write-in-place
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

def set_last_written(ticker: str, iso_ts: str) -> None:
    with _state_lock:
        st = read_last_written_state()
        cur = st.get(ticker)
        if (not cur) or (iso_ts > cur):
            st[ticker] = iso_ts
            write_last_written_state(st)

def get_last_written(ticker: str) -> Optional[datetime]:
    with _state_lock:
        st = read_last_written_state()
        iso = st.get(ticker)
    if not iso:
        return None
    try:
        return pd.to_datetime(iso).to_pydatetime()
    except Exception:
        return None

def csv_append_row(path: str, row: Dict) -> None:
    """Append one row with a per-file lock to avoid concurrent writers."""
    lock = FileLock(path + ".lock", timeout=10)
    with lock:
        header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if header_needed:
                w.writeheader()
            w.writerow(row)

def fast_read_last_line_date(path: str) -> Optional[datetime]:
    """Read last non-empty line's 'date' quickly; returns tz-naive datetime or None."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            line = b""
            while pos > 0:
                pos -= 1
                f.seek(pos, os.SEEK_SET)
                ch = f.read(1)
                if ch == b"\n":
                    if line.strip():
                        break
                    else:
                        line = b""
                else:
                    line = ch + line
            s = line.decode("utf-8").strip()
            if not s or s.startswith("date,"):
                return None
            # try CSV parse
            try:
                fields = next(csv.DictReader(io.StringIO(s)))
                sdt = fields.get("date", None)
            except Exception:
                sdt = None
            if not sdt:
                sdt = s.split(",")[0]
            return pd.to_datetime(sdt).tz_localize(None)
    except Exception:
        return None

# ================================
# API with retries/backoff (SOFT-FAIL + COOLDOWN)
# ================================
def call_historical(token: int, from_dt: datetime, to_dt: datetime, interval: str) -> pd.DataFrame:
    """
    Kite call with jittered backoff + soft fail + per-token cooldown.
    Returns empty DataFrame on failure instead of raising.
    """
    # cooldown gate
    now_epoch = time.time()
    until = _FAIL_UNTIL.get(token, 0.0)
    if now_epoch < until:
        logger.warning(f"[{token}] in cooldown for {int(until - now_epoch)}s; skipping API call.")
        return pd.DataFrame()

    max_retries = 4
    delay = 0.6
    for attempt in range(1, max_retries + 1):
        try:
            with API_SEMAPHORE:
                data = kite.historical_data(
                    token,
                    from_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    to_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    interval
                )
            if not data:
                raise RuntimeError("Empty response")
            df = pd.DataFrame(data)
            if df.empty:
                raise RuntimeError("Empty frame")
            df["date"] = tz_aware(df["date"], "Asia/Kolkata")
            return df
        except (NetworkException, KiteException, HTTPError, RuntimeError) as e:
            sleep_s = delay * (1.6 ** (attempt - 1)) + random.uniform(0, 0.3)
            logger.warning(f"[{token}] attempt {attempt}/{max_retries} failed: {e} -> sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)

    # soft-fail: set cooldown, return empty
    _FAIL_UNTIL[token] = time.time() + FAIL_COOLDOWN_SECS
    logger.error(f"[{token}] soft-failed after {max_retries} attempts; cooldown {FAIL_COOLDOWN_SECS}s.")
    return pd.DataFrame()

# ================================
# Indicators (optimized)
# ================================
def calc_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()

def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def calc_bb(close: pd.Series, period=20, up=2.0, dn=2.0) -> Tuple[pd.Series, pd.Series]:
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper = sma + up * std
    lower = sma - dn * std
    return upper, lower

def calc_stochastic(df: pd.DataFrame, k=14, d=3) -> pd.Series:
    low_min = df["low"].rolling(k, min_periods=1).min()
    high_max = df["high"].rolling(k, min_periods=1).max()
    percent_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    return percent_k.rolling(d, min_periods=1).mean()

def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/n, adjust=False, min_periods=n).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/n, adjust=False, min_periods=n).mean() / (atr + 1e-12)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    adx = dx.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    return adx

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(n, min_periods=1).mean()
    mad = tp.rolling(n, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * (mad + 1e-12))

def calc_mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    delta_tp = tp.diff()
    pos = mf.where(delta_tp > 0, 0.0)
    neg = mf.where(delta_tp < 0, 0.0).abs()
    pos_sum = pos.rolling(n, min_periods=1).sum()
    neg_sum = neg.rolling(n, min_periods=1).sum()
    mr = pos_sum / (neg_sum + 1e-12)
    return 100.0 - (100.0 / (1.0 + mr))

def calc_obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()

def calc_psar(df: pd.DataFrame, initial_af=0.02, step_af=0.02, max_af=0.2) -> pd.Series:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    psar = np.zeros(n)
    bull = True
    af = initial_af
    ep = high[0]
    psar[0] = low[0]
    for i in range(1, n):
        prev = psar[i-1]
        if bull:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], low[i-1], low[i])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = initial_af
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], high[i-1], high[i])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = initial_af
        if bull and high[i] > ep:
            ep = high[i]; af = min(max_af, af + step_af)
        if (not bull) and low[i] < ep:
            ep = low[i]; af = min(max_af, af + step_af)
    return pd.Series(psar, index=df.index)

def calc_ichimoku(df: pd.DataFrame, conv=9, base=26, span_b=52, disp=26) -> pd.DataFrame:
    high = df["high"]; low = df["low"]
    tenkan = (high.rolling(conv, min_periods=1).max() + low.rolling(conv, min_periods=1).min()) / 2.0
    kijun = (high.rolling(base, min_periods=1).max() + low.rolling(base, min_periods=1).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(disp)
    span_bv = ((high.rolling(span_b, min_periods=1).max() + low.rolling(span_b, min_periods=1).min()) / 2.0).shift(disp)
    chikou = df["close"].shift(-disp)
    return pd.DataFrame({
        "Tenkan_Sen": tenkan, "Kijun_Sen": kijun,
        "Senkou_Span_A": span_a, "Senkou_Span_B": span_bv, "Chikou_Span": chikou
    })

def calc_williams_r(df: pd.DataFrame, n=14) -> pd.Series:
    hh = df["high"].rolling(n, min_periods=1).max()
    ll = df["low"].rolling(n, min_periods=1).min()
    return -100.0 * (hh - df["close"]) / (hh - ll + 1e-12)

def calc_roc(close: pd.Series, n=12) -> pd.Series:
    return (close.diff(n) / close.shift(n)) * 100.0

def calc_bb_width(close: pd.Series, period=20, up=2.0, dn=2.0) -> pd.Series:
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper = sma + up * std
    lower = sma - dn * std
    return upper - lower

# ================================
# Indicator recomputation (last window only)
# ================================
def recompute_last_window(df: pd.DataFrame, window_rows: int = 200) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    w = df.tail(window_rows).copy()

    rsi = calc_rsi(w["close"])
    atr = calc_atr(w)
    macd, sig, hist = calc_macd(w["close"])
    ub, lb = calc_bb(w["close"])
    stoch = calc_stochastic(w)
    adx = calc_adx(w)

    ema50 = calc_ema(w["close"], 50)
    ema200 = calc_ema(w["close"], 200)
    cci = calc_cci(w)
    mfi = calc_mfi(w)
    obv = calc_obv(w)
    psar = calc_psar(w)
    ichi = calc_ichimoku(w)
    willr = calc_williams_r(w)
    roc = calc_roc(w["close"])
    bb_w = calc_bb_width(w["close"])

    cols = {
        "RSI": rsi, "ATR": atr, "MACD": macd, "Signal_Line": sig, "Histogram": hist,
        "Upper_Band": ub, "Lower_Band": lb, "Stochastic": stoch, "ADX": adx,
        "EMA_50": ema50, "EMA_200": ema200, "CCI": cci, "MFI": mfi,
        "OBV": obv, "Parabolic_SAR": psar, "Williams_%R": willr,
        "ROC": roc, "BB_Width": bb_w
    }
    for c, s in cols.items():
        w[c] = s
    for c in ichi.columns:
        w[c] = ichi[c]

    w = w.ffill().bfill()
    return w.tail(1).copy()

# ================================
# Fetch logic (incremental + safe end boundary + soft-fallbacks)
# ================================
def fetch_or_cache_historical(ticker: str, token: int, start_dt: datetime) -> pd.DataFrame:
    """Fetch a fixed historical window ending at start_dt - 1 min, cache once per ~8h. Fall back to cache on failure."""
    path = cache_path(ticker, "historical")
    if is_fresh(path, HIST_FRESHNESS):
        try:
            df = pd.read_csv(path)
            df["date"] = tz_aware(df["date"], "Asia/Kolkata")
            return df
        except Exception:
            pass

    from_dt = (start_dt - timedelta(days=HIST_DAYS)).replace(second=0, microsecond=0)
    to_dt = (start_dt - timedelta(minutes=1)).replace(second=0, microsecond=0)
    df = call_historical(token, from_dt, to_dt, INTERVAL)

    if df.empty:
        # fall back to any existing cache (stale is better than nothing)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df["date"] = tz_aware(df["date"], "Asia/Kolkata")
                logger.warning(f"[{ticker}] using stale historical cache.")
                return df
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    df.to_csv(path, index=False)
    return df

def fetch_intraday_delta(ticker: str, token: int, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """
    Fetch only the missing intraday bars.
    Safe end boundary: floor to previous full 5-min (avoid forming candle empties).
    Soft-fallback to cache on failures.
    """
    path = cache_path(ticker, "intraday")

    # Safe end boundary
    safe_to = floor_to_5m(to_dt)

    # Starting point
    start_dt = max(from_dt, safe_to - timedelta(hours=6))  # guard: don't pull huge spans
    lw = get_last_written(ticker)
    if lw is not None:
        start_dt = max(start_dt, (lw + timedelta(minutes=5)))

    # Consider intraday cache tail if fresh
    if is_fresh(path, INTRADAY_FRESHNESS):
        try:
            c = pd.read_csv(path)
            if not c.empty and "date" in c.columns:
                c["date"] = tz_aware(c["date"], "Asia/Kolkata")
                last_dt = c["date"].max().to_pydatetime()
                start_dt = max(start_dt, (last_dt + timedelta(minutes=5)))
        except Exception:
            pass

    start_dt = floor_to_5m(start_dt)
    if start_dt >= safe_to:
        # nothing new; return current cache if any
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "date" in df.columns:
                    df["date"] = tz_aware(df["date"], "Asia/Kolkata")
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    df_new = call_historical(token, start_dt, safe_to, INTERVAL)

    # If API failed/empty, keep existing cache silently
    if df_new.empty:
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                old["date"] = tz_aware(old["date"], "Asia/Kolkata")
                return old
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    # Merge with cache
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            old["date"] = tz_aware(old["date"], "Asia/Kolkata")
        except Exception:
            old = pd.DataFrame()
        combined = pd.concat([old, df_new], ignore_index=True)
    else:
        combined = df_new

    combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined

# ================================
# Per-ticker processing
# ================================
def build_signal_id(ticker: str, iso_ts: str) -> str:
    return f"{ticker}-{iso_ts}"

def compute_daily_change(latest_close: float, prev_close: Optional[float]) -> float:
    if prev_close is None or prev_close == 0:
        return float("nan")
    return (latest_close - prev_close) / prev_close * 100.0

def process_ticker(ticker: str, from_dt: datetime, to_dt: datetime) -> Optional[Dict]:
    token = SHARES_TOKENS.get(ticker)
    if not token:
        logger.warning(f"[{ticker}] missing token")
        return None

    try:
        hist = fetch_or_cache_historical(ticker, token, from_dt)
        intra = fetch_intraday_delta(ticker, token, from_dt, to_dt)
        if hist.empty and intra.empty:
            return None

        df = pd.concat([hist, intra], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
        if df.empty:
            return None

        latest_row = recompute_last_window(df, window_rows=200)
        if latest_row.empty:
            return None

        latest_row = latest_row.iloc[0]
        latest_ts = latest_row["date"]
        if pd.isna(latest_ts):
            return None

        pre = df[df["date"] < from_dt]
        prev_close = float(pre.iloc[-1]["close"]) if not pre.empty else None
        daily_change = compute_daily_change(float(latest_row["close"]), prev_close)

        iso_ts = pd.to_datetime(latest_ts).isoformat()
        signal_id = build_signal_id(ticker, iso_ts)

        out = {
            "date": iso_ts,
            "open": float(latest_row.get("open", np.nan)),
            "high": float(latest_row.get("high", np.nan)),
            "low": float(latest_row.get("low", np.nan)),
            "close": float(latest_row.get("close", np.nan)),
            "volume": float(latest_row.get("volume", np.nan)),
            "RSI": float(latest_row.get("RSI", np.nan)),
            "ATR": float(latest_row.get("ATR", np.nan)),
            "MACD": float(latest_row.get("MACD", np.nan)),
            "Signal_Line": float(latest_row.get("Signal_Line", np.nan)),
            "Histogram": float(latest_row.get("Histogram", np.nan)),
            "Upper_Band": float(latest_row.get("Upper_Band", np.nan)),
            "Lower_Band": float(latest_row.get("Lower_Band", np.nan)),
            "Stochastic": float(latest_row.get("Stochastic", np.nan)),
            "ADX": float(latest_row.get("ADX", np.nan)),
            "EMA_50": float(latest_row.get("EMA_50", np.nan)),
            "EMA_200": float(latest_row.get("EMA_200", np.nan)),
            "CCI": float(latest_row.get("CCI", np.nan)),
            "MFI": float(latest_row.get("MFI", np.nan)),
            "OBV": float(latest_row.get("OBV", np.nan)),
            "Parabolic_SAR": float(latest_row.get("Parabolic_SAR", np.nan)),
            "Tenkan_Sen": float(latest_row.get("Tenkan_Sen", np.nan)),
            "Kijun_Sen": float(latest_row.get("Kijun_Sen", np.nan)),
            "Senkou_Span_A": float(latest_row.get("Senkou_Span_A", np.nan)),
            "Senkou_Span_B": float(latest_row.get("Senkou_Span_B", np.nan)),
            "Chikou_Span": float(latest_row.get("Chikou_Span", np.nan)),
            "Williams_%R": float(latest_row.get("Williams_%R", np.nan)),
            "ROC": float(latest_row.get("ROC", np.nan)),
            "BB_Width": float(latest_row.get("BB_Width", np.nan)),
            "Daily_Change": float(daily_change),
            "Entry Signal": "No",
            "logtime": now_ist().isoformat(),
            "Signal_ID": signal_id
        }

        main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")

        last_written = get_last_written(ticker)
        if last_written and pd.to_datetime(iso_ts) <= last_written:
            logger.info(f"[{ticker}] up-to-date (last_written={last_written.isoformat()})")
            return out

        tail_dt = fast_read_last_line_date(main_path)
        if tail_dt is not None and pd.to_datetime(iso_ts).tz_localize(None) <= tail_dt:
            logger.info(f"[{ticker}] duplicate/older vs last CSV row; skip append")
            return out

        csv_append_row(main_path, out)
        set_last_written(ticker, iso_ts)
        logger.info(f"[{ticker}] appended {iso_ts}")
        return out

    except Exception as e:
        logger.error(f"[{ticker}] error: {e}")
        # keep trace at debug to avoid noisy logs
        logger.debug(traceback.format_exc())
        return None

# ================================
# Validation helpers
# ================================
def ensure_entry_signal_column():
    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    for fp in glob.glob(pattern):
        try:
            lock = FileLock(fp + ".lock", timeout=5)
            with lock:
                df = pd.read_csv(fp)
                if "Entry Signal" not in df.columns:
                    df["Entry Signal"] = "No"
                    df.to_csv(fp, index=False)
        except Timeout:
            logger.warning(f"Lock timeout ensuring Entry Signal in {fp}")
        except Exception as e:
            logger.warning(f"Failed to ensure Entry Signal in {fp}: {e}")

# ================================
# Trading-day helpers
# ================================
def last_trading_day(reference: Optional[datetime.date] = None) -> datetime.date:
    if reference is None:
        reference = now_ist().date()
    if reference.weekday() < 5 and reference not in market_holidays:
        return reference
    d = reference - timedelta(days=1)
    while d.weekday() >= 5 or d in market_holidays:
        d -= timedelta(days=1)
    return d

def prepare_time_range(d: datetime.date) -> Tuple[datetime, datetime]:
    st, et = market_window_for(d)
    return st, et

# ================================
# One pass & scheduler
# ================================
def run_once():
    d = last_trading_day()
    fr, to = prepare_time_range(d)
    logger.info(f"Run window: {fr} -> {to}")

    def worker(tkr: str):
        return process_ticker(tkr, fr, to)

    ensure_entry_signal_column()

    results = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(selected_stocks)))) as ex:
        futs = {ex.submit(worker, t): t for t in selected_stocks}
        for fut in as_completed(futs):
            tkr = futs[fut]
            try:
                out = fut.result()
                if out is not None:
                    results[tkr] = out
            except Exception as e:
                logger.error(f"[{tkr}] worker error: {e}")
    if results:
        print(f"Updated {len(results)} tickers.")
    else:
        print("No updates this tick.")

def _align_to_next_5m(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    if dt.minute % 5 == 0:
        return dt
    return dt + timedelta(minutes=(5 - dt.minute % 5))

def run_5m_cycle() -> dict:
    """Run two quick passes and measure timings."""
    t0 = time.perf_counter()
    t_first0 = t0
    run_once()
    t_first = time.perf_counter() - t_first0

    time.sleep(4)  # your short settle delay

    t_second0 = time.perf_counter()
    run_once()
    t_second = time.perf_counter() - t_second0

    total = time.perf_counter() - t0
    logger.info(f"5m-cycle timings: total={total:.2f}s | first_run={t_first:.2f}s | second_run={t_second:.2f}s")
    return {"total_sec": total, "first_run_sec": t_first, "second_run_sec": t_second}


def main_periodic():
    today = now_ist().date()
    start = india_tz.localize(datetime.combine(today, SESSION_START))
    end = india_tz.localize(datetime.combine(today, SESSION_END_CAP))
    nxt = max(start, _align_to_next_5m(now_ist()))

    while True:
        cur = now_ist()
        if cur > end:
            print("Reached end of session; exiting.")
            break
        if cur < nxt:
            time.sleep((nxt - cur).total_seconds())

        # Two quick iterations to catch late bars
        metrics = run_5m_cycle()
        # Optional: warn if a cycle is getting too close to the next boundary
        if metrics["total_sec"] > 240:
            logger.warning(f"Slow cycle ({metrics['total_sec']:.1f}s) – may bump next 5m boundary.")


        nxt = nxt + timedelta(minutes=5)



# ================================
# Signals
# ================================
def _sig_handler(sig, frame):
    print("Interrupt received. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

if __name__ == "__main__":
    main_periodic()
    # run_once()  # single-shot debug
