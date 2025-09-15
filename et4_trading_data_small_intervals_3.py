# live_2min_collector_stable_retry.py
# ------------------------------------------------------------------
# • Polls kite.quote once a second in bulk mode (500 tokens / call)
# • Rolls 2‑minute candles; keeps a 14‑calendar‑day warm‑up so that
#       Daily_Change | Change_2min | RSI‑14 | ADX‑14 | MFI‑14 | OBV | Force_Index
#   are never NaN
# • Force_Index now uses **true 2‑minute volume** (Δ of cumulative volume)
# • NEW: Automatic retries (exponential back‑off) when Zerodha returns
#   HTTP‑429 "Too many requests" or kiteconnect.NetworkException.
# ------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import csv
import time
import signal
import logging
import threading
import random
from datetime import datetime, timedelta, time as dtime, date
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kex

from et4_filtered_stocks_MIS import selected_stocks  # <- your user list

# ───────── BASIC CONFIG ─────────
INDIA_TZ = pytz.timezone("Asia/Kolkata")
OUT_DIR = "main_indicators_smlitvl"
BAR_MIN = 2                 # candle size in minutes
BATCH_SZ = 500              # max tokens per kite.quote() call
QPS = 10                    # self‑throttle (calls / second)

# ⬇️ NEW: retry tuning ---------------------------------------------------------
MAX_RETRIES = 5             # retry this many times on 429 / NetworkException
BASE_BACKOFF = 0.5          # first wait (sec); doubles each retry 0.5,1,2,4…
# -----------------------------------------------------------------------------

HOLIDAYS = {
    date(2025, 1, 26), date(2025, 3, 14), date(2025, 4, 11),
    date(2025, 8, 15), date(2025, 10, 2), date(2025, 11, 5),
    date(2025, 12, 25),
}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ───────── LOGGING ─────────
log = logging.getLogger("collector")
log.setLevel(logging.INFO)
log.propagate = False
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
log.addHandler(logging.FileHandler("logs/collect_2min_live.log", encoding="utf-8"))
_cons = logging.StreamHandler(sys.stdout)
_cons.setFormatter(_fmt)
log.addHandler(_cons)

# ───────── KITE SESSION ─────────

def kite_session() -> KiteConnect:
    """Return an authenticated KiteConnect client (access‑token file required)."""
    with open("api_key.txt") as f:
        api_key, *_ = f.read().split()
    with open("access_token.txt") as f:
        access = f.read().strip()
    kc = KiteConnect(api_key=api_key)
    kc.set_access_token(access)
    log.info("Kite session ready.")
    return kc


kite = kite_session()

TOKENS: Dict[str, int] = {
    d["tradingsymbol"]: d["instrument_token"]
    for d in kite.instruments("NSE")
    if d["tradingsymbol"] in selected_stocks
}
TOKEN2SYM = {v: k for k, v in TOKENS.items()}
TOKENS_LIST: List[int] = list(TOKENS.values())

# ───────── RATE BUCKET ─────────

class Bucket:
    """Simple leaky‑bucket to cap requests to *cap*/second."""

    def __init__(self, cap: int):
        self.cap = cap
        self.t = cap
        self.lock = threading.Lock()
        self.ts = time.perf_counter()

    def take(self):
        while True:
            with self.lock:
                if time.perf_counter() - self.ts >= 1:
                    self.t = self.cap
                    self.ts = time.perf_counter()
                if self.t:
                    self.t -= 1
                    return
            time.sleep(0.01)


bucket = Bucket(QPS)

# ───────── RETRY HELPERS ─────────

def _sleep_backoff(attempt: int) -> None:
    jitter = random.uniform(0, 0.25)  # avoid thundering‑herd
    time.sleep(BASE_BACKOFF * (2 ** attempt) + jitter)


def safe_quote(batch: List[int]) -> dict:
    """kite.quote wrapper with retries. Returns {{}} on total failure."""

    for attempt in range(MAX_RETRIES + 1):
        bucket.take()
        try:
            return kite.quote(batch)
        except kex.NetworkException as e:
            if "Too many requests" in str(e):
                log.warning(
                    "quote(): 429 – retry %s/%s after back‑off", attempt + 1, MAX_RETRIES
                )
                _sleep_backoff(attempt)
                continue
            log.error("quote(): network error %s", e)
            return {}
        except Exception as e:  # pragma: no cover
            log.error("quote(): unexpected error %s", e)
            return {}

    log.error("quote(): exceeded retries – returning empty dict")
    return {}


def safe_historical(
    tok: int, start: str, end: str, interval: str = "2minute"
) -> list:
    """kite.historical_data wrapper with retries. Returns [] on failure."""

    for attempt in range(MAX_RETRIES + 1):
        bucket.take()
        try:
            return kite.historical_data(tok, start, end, interval)
        except kex.NetworkException as e:
            if "Too many requests" in str(e):
                log.warning(
                    "historical(%s): 429 – retry %s/%s", TOKEN2SYM.get(tok, tok), attempt + 1, MAX_RETRIES
                )
                _sleep_backoff(attempt)
                continue
            log.error("historical(%s): network error %s", TOKEN2SYM.get(tok, tok), e)
            return []
        except Exception as e:  # pragma: no cover
            log.error("historical(%s): unexpected error %s", TOKEN2SYM.get(tok, tok), e)
            return []

    log.error("historical(%s): exceeded retries – returning empty list", TOKEN2SYM.get(tok, tok))
    return []


# ───────── TIME HELPERS ─────────

def floor_minute(ts: datetime, m: int = BAR_MIN) -> datetime:
    """Floor *ts* to nearest *m*-minute boundary."""
    return ts - timedelta(minutes=ts.minute % m, seconds=ts.second, microseconds=ts.microsecond)


def last_trade_day(d: date) -> date:
    while d.weekday() >= 5 or d in HOLIDAYS:
        d -= timedelta(days=1)
    return d


# ───────── TA FUNCTIONS ─────────


def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx14(df: pd.DataFrame) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    plus = (hi.diff()).where((hi.diff() > lo.diff()) & (hi.diff() > 0), 0.0)
    minus = (lo.diff()).where((lo.diff() > hi.diff()) & (lo.diff() > 0), 0.0)
    tr = pd.concat([(hi - lo).abs(), (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    pdi = 100 * plus.rolling(14).sum() / atr
    mdi = 100 * minus.rolling(14).sum() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.rolling(14).mean()


def mfi14(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0).abs()
    mfr = pos.rolling(14).sum() / neg.rolling(14).sum()
    return 100 - 100 / (1 + mfr)


def obv(df: pd.DataFrame) -> pd.Series:
    return (df["volume"] * np.sign(df["close"].diff()).fillna(0)).cumsum()


def force(df: pd.DataFrame) -> pd.Series:
    return df["volume"] * df["close"].diff()


# ───────── WARM‑UP HISTORY (14 calendar days) ─────────

def preload_history(tok: int, days: int = 14) -> pd.DataFrame:
    start = INDIA_TZ.localize(
        datetime.combine(last_trade_day(datetime.now(INDIA_TZ).date() - timedelta(days=days)), dtime(9, 15))
    )
    end = datetime.now(INDIA_TZ)

    data = safe_historical(
        tok,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        "2minute",
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    dt = pd.to_datetime(df["date"])
    df["date"] = (
        dt.tz_localize("UTC").dt.tz_convert(INDIA_TZ) if dt.dt.tz is None else dt.dt.tz_convert(INDIA_TZ)
    )
    return df


# ───────── STATE ─────────

class Slot:
    __slots__ = ("bar", "prev_close", "df", "warm_done", "last_cum_vol")

    def __init__(self):
        self.bar = None               # current building bar
        self.prev_close = None        # previous bar close (for Daily_Change)
        self.df = pd.DataFrame()      # full history
        self.warm_done = False
        self.last_cum_vol = 0         # last cumulative volume


def csv_path(sym: str) -> str:
    return os.path.join(OUT_DIR, f"{sym}_smlitvl.csv")


def write_bar(
    sym: str,
    end: datetime,
    b: dict,
    daily: float,
    chg2m: float | None,
    ind: dict,
) -> None:
    hdr = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Daily_Change",
        "Change_2min",
        "RSI_14",
        "ADX_14",
        "MFI_14",
        "OBV",
        "Force_Index",
    ]

    row = [
        end.isoformat(),
        b["open"],
        b["high"],
        b["low"],
        b["close"],
        b["volume"],
        round(daily, 4),
        "" if chg2m is None else round(chg2m, 4),
        ind["RSI_14"],
        ind["ADX_14"],
        ind["MFI_14"],
        ind["OBV"],
        ind["Force"],
    ]

    p = csv_path(sym)
    with locks[p]:
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(hdr)
            w.writerow(row)


state: Dict[int, Slot] = defaultdict(Slot)
locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


# ───────── PREVIOUS‑DAY CLOSE ─────────

def prev_close(tok: int, bar_time: datetime) -> float:
    ref = last_trade_day(bar_time.date() - timedelta(days=1))
    end = INDIA_TZ.localize(datetime.combine(ref, dtime(15, 30)))
    start = end - timedelta(minutes=15)
    data = safe_historical(
        tok,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        "15minute",
    )
    return data[-1]["close"] if data else np.nan


# ───────── BAR FLUSH ─────────

def flush_bar(tok: int, allow_net: bool = True) -> None:
    slot = state[tok]
    b = slot.bar
    if not b:
        return

    st = b["bar_time"]
    if st.minute % BAR_MIN:
        return  # incomplete bar
    sym = TOKEN2SYM[tok]
    end = st + timedelta(minutes=BAR_MIN)

    if allow_net and not slot.warm_done:
        slot.df = preload_history(tok)
        slot.warm_done = True

    if allow_net and slot.prev_close is None:
        slot.prev_close = prev_close(tok, st)

    # append finished bar to history
    slot.df = pd.concat(
        [
            slot.df,
            pd.DataFrame(
                [{"date": end, "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"], "volume": b["volume"]}]
            ),
        ],
        ignore_index=True,
    )

    df = slot.df
    df["RSI_14"] = rsi14(df["close"])
    df["ADX_14"] = adx14(df)
    df["MFI_14"] = mfi14(df)
    df["OBV"] = obv(df)
    df["Force"] = force(df)

    ind = df.iloc[-1][["RSI_14", "ADX_14", "MFI_14", "OBV", "Force"]].to_dict()

    daily = (
        np.nan
        if slot.prev_close is None or np.isnan(slot.prev_close)
        else (b["close"] - slot.prev_close) / slot.prev_close * 100
    )
    chg2m = (
        None if slot.prev_close is None else (b["close"] - slot.prev_close) / slot.prev_close * 100
    )

    write_bar(sym, end, b, daily, chg2m, ind)
    slot.prev_close = b["close"]
    slot.bar = None


# ───────── QUOTE POLLING ─────────

def fetch_quotes(batch: List[int]) -> dict:
    return safe_quote(batch)


def poll_once() -> None:
    key = floor_minute(datetime.now(INDIA_TZ))

    for i in range(0, len(TOKENS_LIST), BATCH_SZ):
        batch = TOKENS_LIST[i : i + BATCH_SZ]
        quotes = fetch_quotes(batch)

        for tok in batch:
            q = quotes.get(str(tok), {})
            ltp = q.get("last_price")
            cumv = q.get("volume") or 0  # cumulative volume from exchange
            if ltp is None:
                continue

            slot = state[tok]
            dvol = max(cumv - slot.last_cum_vol, 0)  # Δvolume since previous tick
            slot.last_cum_vol = cumv

            bar = slot.bar
            if bar is None or bar["bar_time"] != key:  # new bar
                if bar:
                    flush_bar(tok)
                bar = {
                    "open": ltp,
                    "high": ltp,
                    "low": ltp,
                    "close": ltp,
                    "volume": 0,
                    "bar_time": key,
                }
                slot.bar = bar

            # update current bar
            bar["high"] = max(bar["high"], ltp)
            bar["low"] = min(bar["low"], ltp)
            bar["close"] = ltp
            bar["volume"] += dvol  # true 2‑minute volume


# ───────── CLEAN SHUTDOWN ─────────
exit_flag = False


def on_signal(*_):
    global exit_flag
    exit_flag = True


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


# ───────── MARKET HOURS CHECK ─────────

def market_open() -> bool:
    now = datetime.now(INDIA_TZ)
    return now.replace(hour=9, minute=15, second=0) <= now <= now.replace(
        hour=15, minute=30, second=0
    )


log.info("live 2‑minute collector – %d symbols", len(selected_stocks))
heartbeat = time.time()

while not exit_flag:
    if market_open():
        poll_once()
        if time.time() - heartbeat >= 10:
            log.info("heartbeat …")
            heartbeat = time.time()
        time.sleep(1)
    else:
        time.sleep(5)

# graceful exit (no new network calls)
for tok in list(state):
    flush_bar(tok, allow_net=False)

logging.shutdown()
sys.exit(0)
