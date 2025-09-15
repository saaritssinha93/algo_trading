# -*- coding: utf-8 -*-
"""
Ultra-fast LIVE 5-minute fetcher (KiteTicker) with indicators/naming aligned to et4_trading*.

IMPORTANT: As requested, **no changes** to how data is fetched or stored (streaming, de-dup,
state files, CSV append). Only the **indicator set and column names** have been harmonized
with the et4_trading historical script, while keeping all original live logic.

Key naming alignments vs historical:
- Provide **MACD_Hist** (alias of Histogram) and **MACD_Signal** (alias of Signal_Line)
- Provide **Stoch_%K** and **Stoch_%D** (slow stochastic)
- Provide **20_SMA**, **VWAP**, **Recent_High**, **Recent_Low**
- Provide **Intra_Change**, **Prev_Day_Close**, and **Daily_Change** (% vs previous trading day's final close)
- Keep existing extras (Ichimoku, PSAR, Williams_%R, ROC, BB_Width), and also output **BandWidth**
  (normalized Bollinger width) for downstream consumers.

Everything else — session windowing, bucket end-time stamping, cache/files, locks, dedup —
remains exactly as before.
"""

import os
import io
import csv
import sys
import glob
import json
import time
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
from filelock import FileLock, Timeout

from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import NetworkException, KiteException

# ================================
# Config
# ================================
india_tz = pytz.timezone("Asia/Kolkata")

# Your list of tradingsymbols
# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks_MIS import selected_stocks  # list[str]

LOGS_DIR = "logs"
CACHE_DIR = "bars_cache_5min"          # unified bars cache (rolling window)
INDICATORS_DIR = "main_indicators_5min"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Robust state (prevents duplicates across restarts)
STATE_DIR = os.getenv("ET4_STATE_DIR", os.path.join(tempfile.gettempdir(), "et4_state_5min"))
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "last_written.json")
STATE_LOCK_PATH = os.path.join(STATE_DIR, "last_written.json.lock")
STATE_FILE_LOCK = FileLock(STATE_LOCK_PATH, timeout=10)
_state_lock = threading.Lock()

# Intervals / windows
INTERVAL = "5minute"
REQUIRED_BARS = 240                    # enough for EMA200 etc.
CACHE_TARGET = REQUIRED_BARS + 20      # small cushion

# Market window
SESSION_START = dt_time(9, 15, 0)
SESSION_END_CAP = dt_time(15, 29, 0)

# Live ticker mode
TICK_MODE = KiteTicker.MODE_QUOTE

# If a tick’s last_trade_time is very stale we still bucket by wall-clock,
# but log a debug if older than this many seconds:
STALE_TRADE_WARN_SEC = 180

# Logging
logger = logging.getLogger("live_fetch_5m")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOGS_DIR, "live_fetch_5m.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    fh.setLevel(logging.INFO); ch.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(ch)

print("Logging initialized and script execution started (LIVE).")

# 2024 holidays (extend as needed)
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

def read_api_creds():
    with open("api_key.txt", "r", encoding="utf-8") as f:
        key = f.read().split()[0].strip()
    with open("access_token.txt", "r", encoding="utf-8") as f:
        access_token = f.read().strip()
    return key, access_token


def setup_kite_session() -> KiteConnect:
    key, access_token = read_api_creds()
    kite = KiteConnect(api_key=key)
    kite.set_access_token(access_token)
    logger.info("Kite session established.")
    return kite


kite = setup_kite_session()

_tokens_lock = threading.Lock()
SYMBOL_TO_TOKEN: Dict[str, int] = {}
TOKEN_TO_SYMBOL: Dict[int, str] = {}


def refresh_tokens_if_needed() -> None:
    global SYMBOL_TO_TOKEN, TOKEN_TO_SYMBOL
    with _tokens_lock:
        if SYMBOL_TO_TOKEN and TOKEN_TO_SYMBOL:
            return
        inst = kite.instruments("NSE")
        df = pd.DataFrame(inst)
        mp = dict(zip(df["tradingsymbol"].astype(str), df["instrument_token"].astype(int)))
        SYMBOL_TO_TOKEN = {s: mp.get(s) for s in selected_stocks}
        TOKEN_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_TOKEN.items() if v is not None}


refresh_tokens_if_needed()

# ================================
# Time helpers & state
# ================================

def series_to_ist(s: pd.Series) -> pd.Series:
    """Coerce any date-like series to tz-aware IST Timestamps."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(india_tz)


def now_ist() -> datetime:
    return datetime.now(india_tz)


def floor_to_5m(dt: datetime) -> datetime:
    """Floor to 5-minute boundary (keeps tz)."""
    dt = dt.replace(second=0, microsecond=0)
    return dt - timedelta(minutes=(dt.minute % 5))


def cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_bars.csv")


def indicators_path(ticker: str) -> str:
    return os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")


def _safe_json_load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
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
        try:
            os.replace(tmp, STATE_FILE)
        finally:
            try: os.remove(tmp)
            except Exception: pass


def set_last_written(ticker: str, iso_ts: str) -> None:
    with _state_lock:
        st = read_last_written_state()
        cur = st.get(ticker)
        if (not cur) or (iso_ts > cur):
            st[ticker] = iso_ts
            write_last_written_state(st)


def get_last_written(ticker: str) -> Optional[datetime]:
    """Return UTC-naive datetime for simple comparisons."""
    with _state_lock:
        st = read_last_written_state()
        iso = st.get(ticker)
    if not iso:
        return None
    try:
        ts = pd.to_datetime(iso)
        if ts.tzinfo is None:
            return ts.to_pydatetime()
        return ts.tz_convert(pytz.UTC).tz_localize(None).to_pydatetime()
    except Exception:
        return None


def _fast_read_last_line_date(path: str) -> Optional[datetime]:
    """Read CSV last line's 'date' (returns UTC-naive datetime for easy compare)."""
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
            try:
                fields = next(csv.DictReader(io.StringIO(s)))
                sdt = fields.get("date", None)
            except Exception:
                sdt = None
            if not sdt:
                sdt = s.split(",")[0]
            ts = pd.to_datetime(sdt)
            if ts.tzinfo is None:
                return ts.to_pydatetime()
            return ts.tz_convert(pytz.UTC).tz_localize(None).to_pydatetime()
    except Exception:
        return None


def csv_append_row(path: str, row: Dict) -> None:
    lock = FileLock(path + ".lock", timeout=10)
    with lock:
        header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if header_needed:
                w.writeheader()
            w.writerow(row)

# ================================
# Indicators (vectorized)
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


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def calc_bb(close: pd.Series, period=20, up=2.0, dn=2.0):
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return sma + up*std, sma - dn*std


def _ma(x: pd.Series, window: int, kind: str = "sma") -> pd.Series:
    if kind == "ema":
        return x.ewm(span=window, adjust=False).mean()
    return x.rolling(window, min_periods=window).mean()


def calc_stochastic_fast(df: pd.DataFrame, k=14, d=3):
    low_min  = df["low"].rolling(k, min_periods=k).min()
    high_max = df["high"].rolling(k, min_periods=k).max()
    rng = (high_max - low_min)
    percent_k = pd.Series(0.0, index=df.index)
    valid = rng > 0
    percent_k.loc[valid] = 100.0 * (df["close"].loc[valid] - low_min.loc[valid]) / rng.loc[valid]
    percent_k = percent_k.clip(0.0, 100.0)
    percent_d = percent_k.rolling(d, min_periods=d).mean().clip(0.0, 100.0)
    return percent_k, percent_d


def calc_stochastic_slow(df: pd.DataFrame, k=14, k_smooth=3, d=3, ma_kind="sma"):
    k_fast, _ = calc_stochastic_fast(df, k=k, d=1)
    k_slow = _ma(k_fast, k_smooth, kind=ma_kind).clip(0.0, 100.0)
    d_out = _ma(k_slow, d, kind=ma_kind).clip(0.0, 100.0)
    return k_slow, d_out


def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/n, adjust=False, min_periods=n).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/n, adjust=False, min_periods=n).mean() / (atr + 1e-12)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    return dx.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * (mad + 1e-12))


def calc_mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    delta_tp = tp.diff()
    pos = mf.where(delta_tp > 0, 0.0)
    neg = mf.where(delta_tp < 0, 0.0).abs()
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    mr = pos_sum / (neg_sum + 1e-12)
    return 100.0 - (100.0 / (1.0 + mr))


def calc_obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()


def calc_psar(df: pd.DataFrame, initial_af=0.02, step_af=0.02, max_af=0.2) -> pd.Series:
    high = df["high"].to_numpy(); low = df["low"].to_numpy()
    n = len(df)
    psar = np.zeros(n); bull = True; af = initial_af; ep = high[0]; psar[0] = low[0]
    for i in range(1, n):
        prev = psar[i-1]
        if bull:
            psar[i] = min(prev + af * (ep - prev), low[i-1], low[i])
            if low[i] < psar[i]:
                bull = False; psar[i] = ep; ep = low[i]; af = initial_af
        else:
            psar[i] = max(prev + af * (ep - prev), high[i-1], high[i])
            if high[i] > psar[i]:
                bull = True; psar[i] = ep; ep = high[i]; af = initial_af
        if bull and high[i] > ep: ep = high[i]; af = min(max_af, af + step_af)
        if (not bull) and low[i] < ep: ep = low[i]; af = min(max_af, af + step_af)
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
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = sma + up * std; lower = sma - dn * std
    return upper - lower


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    # cumulative vwap (session-agnostic, matches historical script semantics)
    return (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-12)

# ================================
# Indicator recomputation (tail)
# ================================

def ensure_dt_ist(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = series_to_ist(df["date"])
    return df


def recompute_tail_features(df: pd.DataFrame, window_rows: int = 200) -> pd.DataFrame:
    if df.empty:
        return df
    df = ensure_dt_ist(df).sort_values("date").reset_index(drop=True)
    w = df.tail(window_rows).copy()

    # Core indicators (keep originals)
    rsi = calc_rsi(w["close"])                          # RSI
    atr = calc_atr(w)                                     # ATR
    macd, sig, hist = calc_macd(w["close"])              # MACD, Signal, Hist
    ub, lb = calc_bb(w["close"])                         # Upper/Lower bands
    adx = calc_adx(w)                                     # ADX
    ema50 = calc_ema(w["close"], 50)                     # EMA_50
    ema200 = calc_ema(w["close"], 200)                   # EMA_200
    cci = calc_cci(w)                                     # CCI
    mfi = calc_mfi(w)                                     # MFI
    obv = calc_obv(w)                                     # OBV
    psar = calc_psar(w)                                   # PSAR
    ichi = calc_ichimoku(w)                               # Ichimoku
    willr = calc_williams_r(w)                            # Williams %R
    roc = calc_roc(w["close"])                           # ROC
    bb_w = calc_bb_width(w["close"])                     # BB_Width (absolute)

    # Historical-aligned additions / names
    sma20 = w["close"].rolling(20, min_periods=20).mean() # 20_SMA
    vwap = calc_vwap(w)                                   # VWAP
    k_slow, d_slow = calc_stochastic_slow(w, k=14, k_smooth=3, d=3, ma_kind="sma")  # Stoch_%K/%D
    recent_high = w["high"].rolling(5, min_periods=5).max()
    recent_low  = w["low"].rolling(5, min_periods=5).min()

    # Intra_Change (by IST day) & Daily_Change vs prev trading day's final close
    w["date_only"] = w["date"].dt.tz_convert(india_tz).dt.date
    intra_change = (
        w.groupby("date_only")["close"].pct_change().mul(100.0)
    )
    last_close_per_day = w.groupby("date_only", sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)
    prev_day_map = w["date_only"].map(prev_day_last_close)
    daily_change = (w["close"] - prev_day_map) / (prev_day_map + 1e-12) * 100.0

    # Assign
    w["RSI"] = rsi
    w["ATR"] = atr
    w["MACD"] = macd
    w["MACD_Signal"] = sig          # <- historical name
    w["Signal_Line"] = sig           # alias retained
    w["MACD_Hist"] = hist           # <- historical name
    w["Histogram"] = hist            # alias retained
    w["Upper_Band"] = ub
    w["Lower_Band"] = lb
    w["BandWidth"] = (ub - lb) / (w["close"] + 1e-12)  # normalized width for downstream
    w["ADX"] = adx
    w["EMA_50"] = ema50
    w["EMA_200"] = ema200
    w["20_SMA"] = sma20
    w["VWAP"] = vwap
    w["CCI"] = cci
    w["MFI"] = mfi
    w["OBV"] = obv
    w["Parabolic_SAR"] = psar
    for c in ichi.columns:
        w[c] = ichi[c]
    w["Williams_%R"] = willr
    w["ROC"] = roc
    w["BB_Width"] = bb_w
    w["Stoch_%K"] = k_slow
    w["Stoch_%D"] = d_slow
    # Legacy single stochastic for compatibility
    w["Stochastic"] = d_slow

    w["Recent_High"] = recent_high
    w["Recent_Low"] = recent_low

    w["Intra_Change"] = intra_change
    w["Prev_Day_Close"] = prev_day_map
    w["Daily_Change"] = daily_change

    w = w.ffill().bfill()
    return w.tail(1).copy()

# ================================
# Cold-start seeding (optional)
# ================================

def cold_start_seed_one(ticker: str, token: int) -> None:
    try:
        end = floor_to_5m(now_ist())
        start = end - timedelta(minutes=(REQUIRED_BARS + 10) * 5)
        data = kite.historical_data(
            token,
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
            INTERVAL
        )
        if not data:
            return
        df = pd.DataFrame(data)
        if df.empty:
            return
        df["date"] = series_to_ist(df["date"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        if len(df) > CACHE_TARGET:
            df = df.tail(CACHE_TARGET).copy()
        df.to_csv(cache_path(ticker), index=False)
        logger.info(f"[{ticker}] cold-start cache={len(df)} bars")
    except Exception as e:
        logger.warning(f"[{ticker}] cold-start failed: {e}")


def cold_start_seed_all():
    for s in selected_stocks:
        token = SYMBOL_TO_TOKEN.get(s)
        if token is None:
            continue
        p = cache_path(s)
        need = True
        if os.path.exists(p):
            try:
                d = pd.read_csv(p)
                need = len(d) < int(0.5 * REQUIRED_BARS)
            except Exception:
                need = True
        if need:
            cold_start_seed_one(s, token)

# ================================
# Live aggregator (tick → 5m bars)
# ================================
class LiveAgg:
    """
    Aggregates quote ticks into 5m candles keyed by wall-clock IST.
    Prevents duplicate writes via last_written_bucket + in-flight guard.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.bar_start: Dict[int, datetime] = {}           # current open bucket start (IST)
        self.open: Dict[int, float] = {}
        self.high: Dict[int, float] = {}
        self.low: Dict[int, float] = {}
        self.close: Dict[int, float] = {}
        self.day_vol_anchor: Dict[int, int] = {}           # volume at bucket open
        self.last_day_volume: Dict[int, int] = {}          # latest cumulative day vol
        self.last_written_bucket: Dict[int, datetime] = {} # last finalized bucket (IST start)
        self.finalizing_now = set()                        # (token, start_ts) in-flight

    def _finalize_and_write(self, token: int, end_boundary: datetime):
        """
        Finalize the bar that started at bar_start[token] and ended at end_boundary.
        We **write the bar END time** (end_boundary) to CSV, so 10:55 is stamped at 10:55.
        """
        tkr = TOKEN_TO_SYMBOL.get(token)
        if not tkr:
            return

        with self.lock:
            start_ts = self.bar_start.get(token)
            if start_ts is None:
                return
            if start_ts >= end_boundary:
                return

            # De-dup guard: if already finalized this bucket, skip
            if self.last_written_bucket.get(token) == start_ts:
                return
            if (token, start_ts) in self.finalizing_now:
                return
            # Mark in-flight & last-written BEFORE I/O to prevent races
            self.finalizing_now.add((token, start_ts))
            self.last_written_bucket[token] = start_ts

            o = self.open.get(token)
            h = self.high.get(token)
            l = self.low.get(token)
            c = self.close.get(token)
            dv0 = self.day_vol_anchor.get(token, 0)
            dv1 = self.last_day_volume.get(token, 0)
            vol = max(0, int(dv1) - int(dv0))

        try:
            # Safety: if OHLC incomplete, advance to next bucket and bail
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [o, h, l, c]):
                with self.lock:
                    self.bar_start[token] = end_boundary
                    self.day_vol_anchor[token] = dv1
                return

            # Merge into cache (keeps date as IST start for internal calc)
            p_cache = cache_path(tkr)
            if os.path.exists(p_cache):
                df = pd.read_csv(p_cache)
                if "date" in df.columns:
                    df["date"] = series_to_ist(df["date"])
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()

            new_row = {
                "date": start_ts,  # cache uses bucket START (internal)
                "open": float(o), "high": float(h), "low": float(l), "close": float(c),
                "volume": float(vol)
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df["date"] = series_to_ist(df["date"])
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            if len(df) > CACHE_TARGET:
                df = df.tail(CACHE_TARGET).copy()
            df.to_csv(p_cache, index=False)

            # Recompute tail indicators & write one row to main CSV
            last_feat = recompute_tail_features(df, window_rows=200)
            if last_feat.empty:
                return

            lr = last_feat.iloc[0]

            # === IMPORTANT: stamp the CSV row with the BUCKET END time ===
            bar_end_ist = end_boundary  # tz-aware IST end timestamp (e.g., 10:55)
            if isinstance(bar_end_ist, pd.Timestamp):
                out_iso = bar_end_ist.isoformat()
            else:
                out_iso = pd.to_datetime(bar_end_ist).isoformat()

            # Build UTC-naive for dedup comparisons
            be_ts = pd.to_datetime(bar_end_ist)
            if be_ts.tzinfo is None:
                out_naive_utc = be_ts.to_pydatetime()
            else:
                out_naive_utc = be_ts.tz_convert("UTC").tz_localize(None).to_pydatetime()

            # Build output row with aligned names (retain backwards-compatible aliases too)
            out = {
                "date": out_iso,   # <= end time shown in CSV (e.g., 10:55)
                "open": float(lr.get("open", o)),
                "high": float(lr.get("high", h)),
                "low":  float(lr.get("low", l)),
                "close": float(lr.get("close", c)),
                "volume": float(lr.get("volume", vol)),

                # Core indicators
                "RSI": float(lr.get("RSI", np.nan)),
                "ATR": float(lr.get("ATR", np.nan)),
                "MACD": float(lr.get("MACD", np.nan)),
                "MACD_Signal": float(lr.get("MACD_Signal", np.nan)),  # historical name
                "MACD_Hist": float(lr.get("MACD_Hist", np.nan)),      # historical name
                "Signal_Line": float(lr.get("Signal_Line", np.nan)),  # alias kept
                "Histogram": float(lr.get("Histogram", np.nan)),      # alias kept
                "Upper_Band": float(lr.get("Upper_Band", np.nan)),
                "Lower_Band": float(lr.get("Lower_Band", np.nan)),
                "BandWidth": float(lr.get("BandWidth", np.nan)),
                "ADX": float(lr.get("ADX", np.nan)),
                "EMA_50": float(lr.get("EMA_50", np.nan)),
                "EMA_200": float(lr.get("EMA_200", np.nan)),
                "20_SMA": float(lr.get("20_SMA", np.nan)),
                "VWAP": float(lr.get("VWAP", np.nan)),
                "CCI": float(lr.get("CCI", np.nan)),
                "MFI": float(lr.get("MFI", np.nan)),
                "OBV": float(lr.get("OBV", np.nan)),
                "Parabolic_SAR": float(lr.get("Parabolic_SAR", np.nan)),
                "Tenkan_Sen": float(lr.get("Tenkan_Sen", np.nan)),
                "Kijun_Sen": float(lr.get("Kijun_Sen", np.nan)),
                "Senkou_Span_A": float(lr.get("Senkou_Span_A", np.nan)),
                "Senkou_Span_B": float(lr.get("Senkou_Span_B", np.nan)),
                "Chikou_Span": float(lr.get("Chikou_Span", np.nan)),
                "Williams_%R": float(lr.get("Williams_%R", np.nan)),
                "ROC": float(lr.get("ROC", np.nan)),
                "BB_Width": float(lr.get("BB_Width", np.nan)),

                # Stochastics (historical names)
                "Stoch_%K": float(lr.get("Stoch_%K", np.nan)),
                "Stoch_%D": float(lr.get("Stoch_%D", np.nan)),
                "Stochastic": float(lr.get("Stochastic", np.nan)),  # legacy alias

                # Extras to match historical
                "Recent_High": float(lr.get("Recent_High", np.nan)),
                "Recent_Low": float(lr.get("Recent_Low", np.nan)),
                "Intra_Change": float(lr.get("Intra_Change", np.nan)),
                "Prev_Day_Close": float(lr.get("Prev_Day_Close", np.nan)) if not pd.isna(lr.get("Prev_Day_Close", np.nan)) else np.nan,
                "Daily_Change": float(lr.get("Daily_Change", np.nan)),

                # Housekeeping
                "Entry Signal": "No",
                "logtime": now_ist().isoformat(),
                "Signal_ID": f"{tkr}-{out_iso}",
            }

            main_path = indicators_path(tkr)

            # Idempotency (state + file tail)
            lw = get_last_written(tkr)  # UTC-naive
            if lw is not None and out_naive_utc <= lw:
                pass
            else:
                tail_dt = _fast_read_last_line_date(main_path)  # UTC-naive
                if (tail_dt is None) or (out_naive_utc > tail_dt):
                    csv_append_row(main_path, out)
                    set_last_written(tkr, out_iso)
                    logger.info(f"[{tkr}] wrote 5m bar end={bar_end_ist.time()} (start={start_ts.time()}, vol={vol})")

        except Exception as e:
            logger.error(f"[{tkr}] finalize/write error: {e}")
        finally:
            # Advance to next bucket and reset for the new bar
            with self.lock:
                self.bar_start[token] = end_boundary
                self.day_vol_anchor[token] = self.last_day_volume.get(token, dv1)
                # new bar opens at previous close (until fresh tick)
                self.open[token] = self.close.get(token, c)
                self.high[token] = self.open[token]
                self.low[token]  = self.open[token]
                self.finalizing_now.discard((token, start_ts))

    def on_tick(self, tick: dict):
        token = tick.get("instrument_token")
        if token not in TOKEN_TO_SYMBOL:
            return

        # Bucket by wall-clock IST to avoid stale LTT issues
        now_ts = now_ist()
        bucket = floor_to_5m(now_ts)

        # Extract price & cumulative day volume
        price = float(tick.get("last_price") or tick.get("last_traded_price") or tick.get("ltp") or 0.0)
        day_vol = int(
            tick.get("volume_traded", 0)
            or tick.get("volume", 0)
            or tick.get("last_quantity", 0)
        )

        # (Optional) stale LTT debug
        ltt = tick.get("timestamp") or tick.get("last_trade_time")
        if ltt:
            try:
                ltt_ist = pd.to_datetime(ltt, utc=True).tz_convert(india_tz)
                if (now_ts - ltt_ist.to_pydatetime()).total_seconds() > STALE_TRADE_WARN_SEC:
                    logger.debug(f"[{TOKEN_TO_SYMBOL[token]}] stale LTT ~{int((now_ts - ltt_ist.to_pydatetime()).total_seconds())}s")
            except Exception:
                pass

        with self.lock:
            cur_bucket = self.bar_start.get(token)

        # New bucket started? Finalize previous, then start this one
        if (cur_bucket is None) or (bucket > cur_bucket):
            if cur_bucket is not None:
                self._finalize_and_write(token, bucket)
            with self.lock:
                self.bar_start[token] = bucket
                # initialize OHLC for new bucket
                if price > 0:
                    self.open[token] = price
                    self.high[token] = price
                    self.low[token]  = price
                    self.close[token] = price
                else:
                    prev_close = self.close.get(token, np.nan)
                    self.open[token] = prev_close
                    self.high[token] = prev_close
                    self.low[token]  = prev_close
                    self.close[token] = prev_close
                self.day_vol_anchor[token] = day_vol
                self.last_day_volume[token] = day_vol
            return

        # Same bucket: update OHLC + last_day_volume
        with self.lock:
            if price > 0:
                if token not in self.open or np.isnan(self.open[token]):
                    self.open[token] = price
                    self.high[token] = price
                    self.low[token]  = price
                self.close[token] = price
                self.high[token] = max(self.high[token], price)
                self.low[token]  = min(self.low[token], price)
            self.last_day_volume[token] = max(self.last_day_volume.get(token, 0), day_vol)

    def flush_all_if_needed(self):
        """
        Finalize any buckets that have not been closed yet at the current 5m boundary.
        """
        boundary = floor_to_5m(now_ist())
        with self.lock:
            tokens = list(self.bar_start.keys())

        for tok in tokens:
            with self.lock:
                cur = self.bar_start.get(tok)
            if (cur is not None) and (cur < boundary):
                self._finalize_and_write(tok, boundary)

# ================================
# CSV maintenance
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

def is_trading_day(d: datetime.date) -> bool:
    return (d.weekday() < 5) and (d not in market_holidays)


def wait_until(t: datetime):
    while True:
        now = now_ist()
        if now >= t:
            break
        time.sleep(min(0.5, (t - now).total_seconds()))

# ================================
# Streaming main
# ================================

def main_stream_live():
    today = now_ist().date()
    if not is_trading_day(today):
        print("Non-trading day. Exiting.")
        return

    start_dt = india_tz.localize(datetime.combine(today, SESSION_START))
    end_dt = india_tz.localize(datetime.combine(today, SESSION_END_CAP))

    ensure_entry_signal_column()
    # Seed minimal history so indicators stabilize
    cold_start_seed_all()

    api_key, access_token = read_api_creds()
    tokens = [t for t in (SYMBOL_TO_TOKEN.get(s) for s in selected_stocks) if t]

    if not tokens:
        print("No instrument tokens found. Exiting.")
        return

    agg = LiveAgg()
    ticker = KiteTicker(api_key, access_token)

    def on_connect(ws, response):
        try:
            ws.subscribe(tokens)
            ws.set_mode(TICK_MODE, tokens)
            logger.info(f"Ticker connected. Subscribed {len(tokens)} tokens.")
        except Exception as e:
            logger.error(f"on_connect error: {e}")

    def on_ticks(ws, ticks):
        try:
            for t in (ticks or []):
                tick = {
                    "instrument_token": t.get("instrument_token"),
                    "last_price": t.get("last_price"),
                    "timestamp": t.get("timestamp") or t.get("last_trade_time"),
                    "volume_traded": t.get("volume_traded") or t.get("volume") or t.get("last_quantity"),
                    "last_traded_price": t.get("last_traded_price"),
                    "ltp": t.get("last_price"),
                }
                agg.on_tick(tick)
        except Exception as e:
            logger.error(f"on_ticks error: {e}")

    def on_error(ws, code, reason):
        logger.error(f"Ticker error: code={code}, reason={reason}")

    def on_close(ws, code, reason):
        logger.warning(f"Ticker closed: code={code}, reason={reason}")

    ticker.on_connect = on_connect
    ticker.on_ticks = on_ticks
    ticker.on_error = on_error
    ticker.on_close = on_close

    # Align to the next boundary (or session start)
    first_boundary = max(start_dt, floor_to_5m(now_ist()))
    wait_until(first_boundary)

    ticker.connect(threaded=True)
    logger.info("Streaming started (threaded).")

    try:
        next_flush = floor_to_5m(now_ist()) + timedelta(minutes=5)
        while now_ist() <= end_dt:
            time.sleep(0.8)
            now = now_ist()
            # flush exactly on each 5m boundary (and only once)
            if now >= next_flush:
                agg.flush_all_if_needed()
                next_flush += timedelta(minutes=5)
    finally:
        try:
            ticker.close()
        except Exception:
            pass
        logger.info("Streaming stopped.")

# ================================
# Signals / entry point
# ================================

def _sig_handler(sig, frame):
    print("Interrupt received. Exiting.")
    sys.exit(0)


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

if __name__ == "__main__":
    main_stream_live()
