# -*- coding: utf-8 -*-
"""
Live 5-minute candle builder using Zerodha KiteTicker (WebSocket).
- Builds candles ONLY from live ticks (no historical pulls).
- Seals exactly on 5m boundaries: 09:15, 09:20, ...; session 09:15–15:30 IST.
- Appends one row per completed bar to CSV under main_indicators_5m_ws/.
- Recomputes indicators on last ~200 bars for speed.
- Names & fields aligned to your 5m pipeline (MACD_Signal, MACD_Hist, Stoch_%K/%D, etc.)

Requirements:
    pip install kiteconnect pandas numpy pytz

Prepare:
    access_token.txt  -> access token string
    api_key.txt       -> API key string on first line
    et4_filtered_stocks_MIS.py -> defines `selected_stocks` (set/list of NSE symbols)

Notes:
- Run during market hours (09:15–15:30 IST) to receive ticks.
- If you get zero ticks, the access token is almost always expired. This script now fails fast with a clear log.
"""

import os
import sys
import csv
import time
import json
import math
import queue
import signal
import random
import logging
import threading
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import KiteException

# ================================
# Basic config
# ================================
india_tz = pytz.timezone("Asia/Kolkata")

# symbols list
try:
    from et4_filtered_stocks_MIS import selected_stocks
except Exception:
    selected_stocks = []

# dirs
CACHE_DIR = "data_cache_5m_ws"
INDICATORS_DIR = "main_indicators_5m_ws"
LOGS_DIR = "logs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# logging
logger = logging.getLogger("ws_5m_builder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOGS_DIR, "ws_5m_builder.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    fh.setLevel(logging.INFO); ch.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(ch)

print("Logging initialized and script execution started.")

# session window
SESSION_START = dt_time(9, 15, 0)   # IST
SESSION_END   = dt_time(15, 30, 0)  # IST

# subscribe chunking to avoid large single subscribe
SUB_CHUNK = 400

# ================================
# Helpers
# ================================
def now_ist() -> datetime:
    return datetime.now(india_tz)

def floor_to_5m(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    return dt - timedelta(minutes=(dt.minute % 5))

def next_5m_boundary(dt: datetime) -> datetime:
    f = floor_to_5m(dt)
    if dt == f:
        return dt
    return f + timedelta(minutes=5)

def in_session(dt: datetime) -> bool:
    # Compare naive times to avoid tz-aware vs tz-naive TypeError
    t = dt.astimezone(india_tz).time().replace(tzinfo=None)
    return SESSION_START <= t <= SESSION_END

def csv_path_for(symbol: str) -> str:
    return os.path.join(INDICATORS_DIR, f"{symbol}_main_indicators.csv")

def tz_aware(series_or_scalar, tz="Asia/Kolkata"):
    if isinstance(series_or_scalar, pd.Series):
        s = pd.to_datetime(series_or_scalar, errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(tz)
        else:
            s = s.dt.tz_convert(tz)
        return s
    else:
        d = pd.to_datetime(series_or_scalar, errors="coerce")
        if d is pd.NaT:
            return None
        if d.tzinfo is None:
            return india_tz.localize(d)
        return d.astimezone(india_tz)

def setup_ws_reconnect(kws, logger):
    """
    Make WebSocket auto-reconnect work across kiteconnect versions.
    Returns "builtin" when the SDK handles reconnect, else "manual".
    """
    if hasattr(kws, "enable_reconnect"):
        try:
            kws.enable_reconnect(reconnect=True, reconnect_interval=5, reconnect_tries=50)
        except TypeError:
            kws.enable_reconnect(True, 5, 50)
        logger.info("Auto-reconnect enabled via enable_reconnect().")
        return "builtin"
    if hasattr(kws, "set_reconnect"):
        try:
            kws.set_reconnect(max_retries=50, retry_interval=5)
            logger.info("Auto-reconnect enabled via set_reconnect().")
            return "builtin"
        except Exception:
            pass
    logger.warning("Auto-reconnect API not available in this kiteconnect version; using manual reconnect loop.")
    return "manual"

# ================================
# Indicators (aligned to your 5m pipeline)
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

def calc_bb(close: pd.Series, period=20, k_up=2.0, k_dn=2.0):
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper = sma + k_up*std
    lower = sma - k_dn*std
    return sma, upper, lower

def calc_stoch_kd(df: pd.DataFrame, k=14, d=3):
    low_min = df["low"].rolling(k, min_periods=1).min()
    high_max = df["high"].rolling(k, min_periods=1).max()
    percent_k = 100.0 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    percent_d = percent_k.rolling(d, min_periods=1).mean()
    return percent_k, percent_d

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
    return dx.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()

def calc_ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

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

def calc_session_vwap_last(df: pd.DataFrame) -> float:
    """VWAP of the current day up to the last row."""
    if df.empty:
        return float("nan")
    d = df.copy()
    d["date_only"] = d["date"].dt.date
    last_day = d.iloc[-1]["date_only"]
    day_df = d[d["date_only"] == last_day]
    tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
    vwap = (tp * day_df["volume"]).sum() / max(1.0, day_df["volume"].sum())
    return float(vwap)

# ================================
# Session / tokens
# ================================
def setup_kite_session() -> Tuple[KiteConnect, KiteTicker, str, str]:
    if not os.path.exists("access_token.txt") or not os.path.exists("api_key.txt"):
        logger.error("Missing access_token.txt or api_key.txt")
        sys.exit(1)
    with open("access_token.txt", "r") as f:
        access_token = f.read().strip()
    with open("api_key.txt", "r") as f:
        api_key = f.read().split()[0]
    if not access_token or not api_key:
        logger.error("Empty API key or access token.")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Fail fast if token is bad/expired
    try:
        kite.profile()
    except Exception as e:
        logger.error(f"Access token invalid/expired. Generate a fresh token. Error: {e}")
        sys.exit(1)

    kws = KiteTicker(api_key, access_token)
    logger.info("Kite HTTP + WebSocket sessions initialized.")
    return kite, kws, api_key, access_token

kite, kws, _API_KEY, _ACCESS_TOKEN = setup_kite_session()

def get_instrument_mapping(symbols: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    try:
        ins = pd.DataFrame(kite.instruments("NSE"))
    except Exception as e:
        logger.error(f"Failed to fetch instruments dump: {e}")
        sys.exit(1)

    if ins.empty:
        logger.error("Empty instruments dump for NSE.")
        sys.exit(1)

    ins["tradingsymbol"] = ins["tradingsymbol"].astype(str)
    sym2tok = {}
    for s in symbols:
        s_up = str(s).strip().upper()
        r = ins.loc[ins["tradingsymbol"] == s_up]
        if r.empty:
            logger.warning(f"[{s_up}] token not found in NSE instruments.")
            continue
        sym2tok[s_up] = int(r.iloc[0]["instrument_token"])
    tok2sym = {v: k for k, v in sym2tok.items()}
    return sym2tok, tok2sym

selected_list = [str(s).strip().upper() for s in (list(selected_stocks) if selected_stocks else [])]
if not selected_list:
    logger.error("selected_stocks is empty or could not be imported.")
    sys.exit(1)

SYM2TOK, TOK2SYM = get_instrument_mapping(selected_list)

# ================================
# Candle builder state
# ================================
class CandleState:
    __slots__ = ("bin_start", "open", "high", "low", "close",
                 "start_cum_vol", "last_cum_vol", "prev_day_close")

    def __init__(self, bin_start: datetime, ltp: float, cum_vol: Optional[int], prev_day_close: Optional[float]):
        self.bin_start = bin_start
        self.open = ltp
        self.high = ltp
        self.low  = ltp
        self.close = ltp
        self.start_cum_vol = int(cum_vol or 0)
        self.last_cum_vol  = int(cum_vol or 0)
        self.prev_day_close = prev_day_close  # used for Daily_Change

    def update(self, ltp: float, cum_vol: Optional[int]):
        self.close = ltp
        if ltp > self.high: self.high = ltp
        if ltp < self.low:  self.low  = ltp
        if cum_vol is not None:
            self.last_cum_vol = int(cum_vol)

# per-token live state
state_lock = threading.Lock()
live_state: Dict[int, CandleState] = {}        # token -> current 5m state
last_day_seen: Dict[int, datetime.date] = {}   # reset guards by day
prev_day_close_map: Dict[int, float] = {}      # last day's close from tick OHLC
mem_df: Dict[str, pd.DataFrame] = {}           # small rolling df per symbol

# ================================
# Append helpers
# ================================
# A wide header covering detector/trainer expected columns
HEADERS = [
    "date","open","high","low","close","volume",
    "RSI","ATR","EMA_50","EMA_200","20_SMA","VWAP","CCI","MFI","OBV",
    "MACD","MACD_Signal","MACD_Hist","Upper_Band","Lower_Band",
    "Stoch_%K","Stoch_%D","ADX",
    "BBP","BandWidth","ATR_Pct","OBV_Slope10",
    "Daily_Change","Intra_Change",
    "Entry Signal","logtime","Signal_ID"
]

def append_csv(symbol: str, row: Dict):
    path = csv_path_for(symbol)
    header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
    # Ensure consistent column order and presence
    safe_row = {k: row.get(k, "") for k in HEADERS}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        if header_needed: w.writeheader()
        w.writerow(safe_row)

def ensure_mem_df(symbol: str):
    if symbol not in mem_df:
        mem_df[symbol] = pd.DataFrame(columns=["date","open","high","low","close","volume"])

def load_existing_csv_to_memory(symbol: str, max_rows: int = 500):
    path = csv_path_for(symbol)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = tz_aware(df["date"], "Asia/Kolkata")
                df = df.sort_values("date").tail(max_rows).reset_index(drop=True)
                mem_df[symbol] = df[["date","open","high","low","close","volume"]].copy()
        except Exception as e:
            logger.warning(f"[{symbol}] failed to load existing CSV: {e}")

for s in selected_list:
    load_existing_csv_to_memory(s)

# ================================
# Tick handlers
# ================================
_TICK_COUNTER = 0

def _ensure_current_bin_start(now: datetime) -> datetime:
    # ensure start at >= 09:15
    st = india_tz.localize(datetime.combine(now.date(), SESSION_START))
    bs = floor_to_5m(now)
    return max(bs, st)

def on_ticks(ws, ticks):
    global _TICK_COUNTER
    _TICK_COUNTER += len(ticks)
    if _TICK_COUNTER and _TICK_COUNTER % 1000 == 0:
        logger.info(f"Ticks received so far: {_TICK_COUNTER}")

    cur = now_ist()
    bin_start = _ensure_current_bin_start(cur)

    with state_lock:
        for tk in ticks:
            token = tk.get("instrument_token")
            if token not in TOK2SYM:
                continue
            symbol = TOK2SYM[token]

            ltp = tk.get("last_price")
            if ltp is None:
                continue

            # QUOTE mode cumulative day volume
            cum_vol = tk.get("volume_traded") or tk.get("volume")
            ohlc = tk.get("ohlc") or {}
            pclose = ohlc.get("close")
            if pclose is not None:
                prev_day_close_map[token] = float(pclose)

            # reset guard at day change
            day = cur.date()
            if last_day_seen.get(token) != day:
                last_day_seen[token] = day
                live_state.pop(token, None)

            st = live_state.get(token)
            if st is None or st.bin_start != bin_start:
                # new bin state
                st = CandleState(bin_start, float(ltp), cum_vol, prev_day_close_map.get(token))
                live_state[token] = st
            else:
                st.update(float(ltp), cum_vol)

def on_connect(ws, response):
    logger.info("WebSocket connected; subscribing tokens.")
    tokens = list(TOK2SYM.keys())
    if not tokens:
        logger.error("No tokens to subscribe.")
        return
    # chunked subscribe
    for i in range(0, len(tokens), SUB_CHUNK):
        batch = tokens[i:i+SUB_CHUNK]
        ws.subscribe(batch)
        try:
            ws.set_mode(KiteTicker.MODE_QUOTE, batch)
        except Exception:
            # older builds expose mode on instance
            ws.set_mode(ws.MODE_QUOTE, batch)
        time.sleep(0.05)
    logger.info(f"Subscribed {len(tokens)} tokens; mode=QUOTE")

def on_close(ws, code, reason):
    logger.warning(f"WebSocket closed: code={code} reason={reason}")

def on_error(ws, code, reason):
    logger.error(f"WebSocket error: code={code} reason={reason}")

def on_noreconnect(ws):
    logger.error("WebSocket: no more reconnect attempts; exiting.")
    os._exit(1)

def on_reconnect(ws, attempt_count):
    logger.warning(f"WebSocket reconnecting... attempt {attempt_count}")

# ================================
# Indicator recompute (last row only)
# ================================
def recompute_last(df: pd.DataFrame, tail_rows: int = 200) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    w = df.tail(tail_rows).copy()

    # Moving/derived
    ema50  = calc_ema(w["close"], 50)
    ema200 = calc_ema(w["close"], 200)
    sma20, ub, lb = calc_bb(w["close"], period=20, k_up=2.0, k_dn=2.0)
    rsi = calc_rsi(w["close"])
    atr = calc_atr(w)
    macd, macd_sig, macd_hist = calc_macd(w["close"])
    stoch_k, stoch_d = calc_stoch_kd(w)
    adx = calc_adx(w)
    cci = calc_cci(w)
    mfi = calc_mfi(w)
    obv = calc_obv(w)

    # Derived bands and pct
    eps = 1e-12
    bbw = (ub - lb)
    bandwidth = bbw / (w["close"] + eps)
    bbp = (w["close"] - lb) / (bbw + eps)
    atr_pct = atr / (w["close"] + eps)

    # OBV slope over 10 bars
    obv_slope10 = (obv - obv.shift(10)) / (obv.shift(10).abs() + eps)

    # Session VWAP (per day)
    vwap_last = calc_session_vwap_last(w)

    # pack last row
    last = w.iloc[-1].copy()
    out = pd.Series({
        "RSI": float(rsi.iloc[-1]),
        "ATR": float(atr.iloc[-1]),
        "EMA_50": float(ema50.iloc[-1]),
        "EMA_200": float(ema200.iloc[-1]),
        "20_SMA": float(sma20.iloc[-1]),
        "Upper_Band": float(ub.iloc[-1]),
        "Lower_Band": float(lb.iloc[-1]),
        "MACD": float(macd.iloc[-1]),
        "MACD_Signal": float(macd_sig.iloc[-1]),
        "MACD_Hist": float(macd_hist.iloc[-1]),
        "Stoch_%K": float(stoch_k.iloc[-1]),
        "Stoch_%D": float(stoch_d.iloc[-1]),
        "ADX": float(adx.iloc[-1]),
        "CCI": float(cci.iloc[-1]),
        "MFI": float(mfi.iloc[-1]),
        "OBV": float(obv.iloc[-1]),
        "VWAP": float(vwap_last),
        "BandWidth": float(bandwidth.iloc[-1]),
        "BBP": float(bbp.iloc[-1]),
        "ATR_Pct": float(atr_pct.iloc[-1]),
        "OBV_Slope10": float(obv_slope10.iloc[-1]),
    })
    return out

# ================================
# Seal & write bars every 5 minutes
# ================================
stop_event = threading.Event()

def compute_daily_change(latest_close: float, previous_day_close: Optional[float]) -> float:
    if not previous_day_close or previous_day_close == 0:
        return float("nan")
    return (latest_close - previous_day_close) / previous_day_close * 100.0

def seal_and_write_completed_bin():
    """Seal any bin whose end time has passed current 5m boundary and write CSV row."""
    cur = now_ist()
    current_boundary = floor_to_5m(cur)  # start of current bin

    to_seal: List[Tuple[int, CandleState]] = []
    with state_lock:
        for token, st in list(live_state.items()):
            # a bin [bin_start, bin_start+5m) is complete if current_boundary >= bin_start+5m
            if current_boundary >= (st.bin_start + timedelta(minutes=5)):
                to_seal.append((token, st))
                live_state.pop(token, None)  # remove; next bin created on next tick

    # Process seals outside lock
    for token, st in to_seal:
        symbol = TOK2SYM.get(token)
        if not symbol:
            continue

        # bar volume = delta of cumulative day volume
        vol = max(0, st.last_cum_vol - st.start_cum_vol)

        # update in-memory df (limited to last ~520 rows)
        ensure_mem_df(symbol)
        df = mem_df[symbol]
        df = pd.concat([df, pd.DataFrame([{
            "date": st.bin_start,
            "open": st.open, "high": st.high, "low": st.low, "close": st.close, "volume": vol
        }])], ignore_index=True)
        df = df.sort_values("date").tail(520).reset_index(drop=True)

        # Indicators (last row only)
        ind = recompute_last(df, tail_rows=200)

        # Daily_Change vs previous day's close (from ticks)
        prev_close = st.prev_day_close if st.prev_day_close is not None else prev_day_close_map.get(token)
        daily_change = compute_daily_change(st.close, prev_close)

        # Intra_Change vs previous bar close
        if len(df) >= 2 and df.iloc[-2]["close"]:
            intra_change = (st.close - float(df.iloc[-2]["close"])) / float(df.iloc[-2]["close"]) * 100.0
        else:
            intra_change = float("nan")

        # compose final row
        bar_time_iso = st.bin_start.isoformat()
        row = {
            "date": bar_time_iso,
            "open": st.open, "high": st.high, "low": st.low, "close": st.close, "volume": vol,
            "Daily_Change": float(daily_change),
            "Intra_Change": float(intra_change),
            "Entry Signal": "No",
            "logtime": now_ist().isoformat(),
            "Signal_ID": f"{symbol}-{bar_time_iso}"
        }
        # merge indicators
        for k, v in ind.items():
            row[k] = float(v) if pd.notna(v) else ""

        # persist
        append_csv(symbol, row)
        mem_df[symbol] = df  # save back
        logger.info(f"[{symbol}] sealed {bar_time_iso}  O:{st.open} H:{st.high} L:{st.low} C:{st.close} V:{vol}")

def scheduler_thread():
    """Seals bars right at 5m boundaries. Exits after the 15:30 boundary (last bar 15:25–15:29 sealed)."""
    # Wait until session start
    while not stop_event.is_set():
        now = now_ist()
        if now.astimezone(india_tz).time().replace(tzinfo=None) < SESSION_START:
            time.sleep(0.5)
            continue
        break

    while not stop_event.is_set():
        now = now_ist()
        nxt = next_5m_boundary(now)
        sleep_s = (nxt - now).total_seconds()
        if sleep_s > 0:
            # sleep in small chunks to be responsive to stop_event
            time.sleep(min(sleep_s, 1.0))
            continue

        # At (or just after) boundary -> seal previous bin
        seal_and_write_completed_bin()

        # Stop after market close boundary
        if now.astimezone(india_tz).time().replace(tzinfo=None) >= SESSION_END:
            logger.info("Reached session end boundary; stopping scheduler.")
            stop_event.set()
            break

        time.sleep(0.5)  # tiny drift control

# ================================
# Main
# ================================
def run():
    # WebSocket callbacks
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.on_noreconnect = on_noreconnect
    kws.on_reconnect = on_reconnect

    # Try to enable SDK-level reconnect when available
    reconnect_mode = setup_ws_reconnect(kws, logger)

    # Start the 5-minute boundary scheduler
    th = threading.Thread(target=scheduler_thread, daemon=True)
    th.start()

    # Start / loop the socket
    try:
        if reconnect_mode == "builtin":
            kws.connect(threaded=False, disable_ssl_verification=False)
        else:
            # Manual reconnect loop
            while not stop_event.is_set():
                try:
                    kws.connect(threaded=False, disable_ssl_verification=False)
                except Exception as e:
                    logger.warning(f"WebSocket ended with error: {e}; retrying in 5s")
                    time.sleep(5)
                    continue
                if not stop_event.is_set():
                    logger.warning("WebSocket disconnected; reconnecting in 5s")
                    time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_event.set()
        try:
            th.join(timeout=5)
        except Exception:
            pass

def _sig_handler(sig, frame):
    print("Interrupt received. Exiting.")
    stop_event.set()
    try:
        kws.close()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

if __name__ == "__main__":
    if not SYM2TOK:
        logger.error("No tradingsymbol → token mapping found. Check your selected_stocks and instruments.")
        sys.exit(1)

    logger.info(f"Streaming {len(SYM2TOK)} NSE instruments in QUOTE mode.")
    run()
