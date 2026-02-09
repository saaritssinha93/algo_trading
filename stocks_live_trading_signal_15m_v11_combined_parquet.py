# -*- coding: utf-8 -*-
"""
ALGO-SM1 (Stocks only) — LIVE 15m Signal Scanner (AVWAP v11 combined: LONG + SHORT)

This file is intentionally modeled after the structure of:
- etf_live_trading_signal_15m_v7_parquet.py  (slot scheduler + per-run parquet outputs)

But the SIGNAL LOGIC is derived from the user's ALGO-SM1 AVWAP v11 intraday scripts:
- algosm1_trading_signal_profit_analysis_eq_15m_v11_intraday_parquet.py          (SHORT)
- algosm1_trading_signal_profit_analysis_eq_15m_v11_long_intraday_parquet.py     (LONG)
- algosm1_trading_signal_profit_analysis_eq_15m_v11_combined_parquet.py          (combined reference)

What this script does
---------------------
1) Runs in 15-minute "slots" (09:15 -> 15:30 IST), with an optional second run shortly after each slot.
2) For each ticker parquet in stocks_indicators_15min_eq:
   - Loads recent rows
   - Filters today's session rows
   - Computes intraday AVWAP (anchored to today's first candle)
   - Evaluates whether the *latest completed candle* triggers a LONG and/or SHORT entry
     using v11-style impulse + pullback/bounce + AVWAP rejection + trend filters.
3) Writes **two** Parquet outputs per run (like the ETF live file concept):
   - out_live_checks_15m/YYYYMMDD/checks_<timestamp>_<A|B>.parquet (all tickers, both sides, with diagnostics)
   - out_live_signals_15m/YYYYMMDD/signals_<timestamp>_<A|B>.parquet (only triggered signals)
4) Maintains a simple JSON state file to avoid duplicate signals:
   - Per ticker/day/side cap = 1 (configurable)

Notes
-----
- This is a live *signal* file (no trade simulation / exits).
- It assumes your 15m parquet indicators are already being updated.
  If you want, you can set UPDATE_15M_BEFORE_CHECK=True to call your core updater once per slot.

IMPORTANT: This is stocks-only and uses:
    import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly as core

"""

from __future__ import annotations

import os
import glob
import json
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# --------------------
# Core updater module
# --------------------
import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly as core

# =============================================================================
# TIMEZONE + DIRECTORIES
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

DIR_15M = "stocks_indicators_15min_eq"
END_15M = "_stocks_indicators_15min.parquet"

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CHECKS_DIR = ROOT / "out_live_checks_15m"
OUT_SIGNALS_DIR = ROOT / "out_live_signals_15m"
OUT_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
OUT_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIR = ROOT / "logs"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "avwap_live_state_v11.json"

PARQUET_ENGINE = "pyarrow"

# =============================================================================
# SCHEDULER CONFIG (modeled after etf_live_trading_signal_15m_v7_parquet.py)
# =============================================================================
START_TIME = dtime(9, 15)          # first slot at 09:15
END_TIME = dtime(15, 30)           # last slot at 15:30
SECOND_RUN_GAP_SECONDS = 18        # optional B-run after each slot
ENABLE_SECOND_RUN = True

# Safety hard stop (after market)
HARD_STOP_TIME = dtime(15, 40)

# Only scan if enough data is present & candles left for management (optional)
MIN_BARS_LEFT = 4  # used when you want to ensure enough time left (intraday); kept for parity

# Update flag
UPDATE_15M_BEFORE_CHECK = False    # set True if you want to run core.run_mode("15min") before scanning

# How many rows to load per ticker (tail). 250 rows ~= > 2 days of 15m bars.
TAIL_ROWS = 260

# =============================================================================
# COMMON SESSION FILTER (data)
# =============================================================================
SESSION_START = dtime(9, 15, 0)
SESSION_END = dtime(14, 30, 0)

# =============================================================================
# V11 SHORT PARAMETERS (tunable / keep separate from LONG)
# =============================================================================
SHORT_STOP_PCT = 0.01
SHORT_TARGET_PCT = 0.0065

SHORT_ADX_MIN = 20.0
SHORT_ADX_SLOPE_MIN = 1.0

SHORT_RSI_MAX = 60.0
SHORT_STOCHK_MAX = 80.0

# Strict trend (Option A)
SHORT_REQUIRE_EMA_TREND = True
SHORT_REQUIRE_AVWAP_BELOW = True

# Time windows (Option E) - short
SHORT_USE_TIME_WINDOWS = True
SHORT_SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 30, 0)),
    (dtime(13, 0, 0), dtime(14, 30, 0)),
]

# Impulse (short)
SHORT_MOD_RED_MIN_ATR = 0.40
SHORT_MOD_RED_MAX_ATR = 1.00
SHORT_HUGE_RED_MIN_ATR = 1.50
SHORT_HUGE_RED_MIN_RANGE_ATR = 2.00
SHORT_CLOSE_NEAR_LOW_MAX = 0.25

SHORT_SMALL_GREEN_MAX_ATR = 0.20

# Entry buffer
BUFFER_ABS = 0.05
BUFFER_PCT = 0.0002

# AVWAP rejection (Option B-like)
SHORT_AVWAP_REJ_ENABLED = True
SHORT_AVWAP_REJ_TOUCH = True               # allow "touch" definition
SHORT_AVWAP_REJ_CONSEC_CLOSES = 2          # close stays below AVWAP for N bars after touch
SHORT_AVWAP_REJ_DIST_ATR_MULT = 0.25       # require AVWAP - close >= mult*ATR at entry
SHORT_AVWAP_REJ_MODE = "any"               # any: touch OR consec; touch_only / consec_only

# Throttles
SHORT_CAP_PER_TICKER_PER_DAY = 1

# =============================================================================
# V11 LONG PARAMETERS (separate)
# =============================================================================
LONG_STOP_PCT = 0.01
LONG_TARGET_PCT = 0.0065

LONG_ADX_MIN = 20.0
LONG_ADX_SLOPE_MIN = 1.0

LONG_RSI_MIN = 40.0                 # long bias (example; adjust to your v11 long file)
LONG_STOCHK_MIN = 20.0              # avoid extreme oversold-only; keep configurable

# Strict trend (symmetric)
LONG_REQUIRE_EMA_TREND = True        # EMA20 > EMA50 and close > EMA20
LONG_REQUIRE_AVWAP_ABOVE = True      # close > AVWAP

# Time windows - long
LONG_USE_TIME_WINDOWS = True
LONG_SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 30, 0)),
    (dtime(13, 0, 0), dtime(14, 30, 0)),
]

# Impulse (long)
LONG_MOD_GREEN_MIN_ATR = 0.40
LONG_MOD_GREEN_MAX_ATR = 1.00
LONG_HUGE_GREEN_MIN_ATR = 1.50
LONG_HUGE_GREEN_MIN_RANGE_ATR = 2.00
LONG_CLOSE_NEAR_HIGH_MAX = 0.25

LONG_SMALL_RED_MAX_ATR = 0.20

# AVWAP rejection (long) - fail to break below and reclaim, symmetric definition
LONG_AVWAP_REJ_ENABLED = True
LONG_AVWAP_REJ_TOUCH = True
LONG_AVWAP_REJ_CONSEC_CLOSES = 2          # close stays ABOVE AVWAP for N bars after touch
LONG_AVWAP_REJ_DIST_ATR_MULT = 0.25       # require close - AVWAP >= mult*ATR at entry
LONG_AVWAP_REJ_MODE = "any"

LONG_CAP_PER_TICKER_PER_DAY = 1

# =============================================================================
# TOP-N filter (optional)
# =============================================================================
USE_TOPN_PER_RUN = False       # in live, usually false; enable if you want to only act on best N per run
TOPN_PER_RUN = 30

# =============================================================================
# PYARROW REQUIREMENT
# =============================================================================
def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError("Parquet support requires 'pyarrow' (pip install pyarrow).") from e


# =============================================================================
# UTILITIES
# =============================================================================
def now_ist() -> datetime:
    return datetime.now(IST)


def _buffer(price: float) -> float:
    return max(float(BUFFER_ABS), float(price) * float(BUFFER_PCT))


def in_session(ts: pd.Timestamp) -> bool:
    t = ts.tz_convert(IST).time()
    return (t >= SESSION_START) and (t <= SESSION_END)


def _in_windows(ts: pd.Timestamp, windows: List[Tuple[dtime, dtime]], enabled: bool) -> bool:
    if not enabled:
        return True
    t = ts.tz_convert(IST).time()
    for a, b in windows:
        if a <= t <= b:
            return True
    return False


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def _twice_increasing(df: pd.DataFrame, idx: int, col: str) -> bool:
    if idx < 2 or col not in df.columns:
        return False
    a = _safe_float(df.at[idx, col])
    b = _safe_float(df.at[idx - 1, col])
    c = _safe_float(df.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a > b > c)


def _twice_decreasing(df: pd.DataFrame, idx: int, col: str) -> bool:
    if idx < 2 or col not in df.columns:
        return False
    a = _safe_float(df.at[idx, col])
    b = _safe_float(df.at[idx - 1, col])
    c = _safe_float(df.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a < b < c)


def _slope_ok(df: pd.DataFrame, idx: int, col: str, min_slope: float, direction: str = "up") -> bool:
    """
    direction="up":   col[idx] - col[idx-2] >= min_slope
    direction="down": col[idx-2] - col[idx] >= min_slope
    """
    if idx < 2 or col not in df.columns:
        return False
    a = _safe_float(df.at[idx, col])
    c = _safe_float(df.at[idx - 2, col])
    if not (np.isfinite(a) and np.isfinite(c)):
        return False
    if direction == "up":
        return (a - c) >= float(min_slope)
    return (c - a) >= float(min_slope)


# =============================================================================
# INDICATORS (fallback computations, only if columns missing)
# =============================================================================
def ensure_ema(close: pd.Series, span: int) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    return close.ewm(span=span, adjust=False).mean()


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(14).mean()


def compute_rsi14(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stoch_14_3(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    ll = low.rolling(14).min()
    hh = high.rolling(14).max()
    denom = (hh - ll).replace(0, np.nan)

    k = 100.0 * (close - ll) / denom
    d = k.rolling(3).mean()
    return k, d


def compute_adx14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)

    plus_di = 100.0 * (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    )
    minus_di = 100.0 * (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / 14, adjust=False).mean()


def compute_day_avwap(df_day: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df_day["high"], errors="coerce")
    low = pd.to_numeric(df_day["low"], errors="coerce")
    close = pd.to_numeric(df_day["close"], errors="coerce")
    vol = pd.to_numeric(df_day.get("volume", 0.0), errors="coerce").fillna(0.0)

    tp = (high + low + close) / 3.0
    pv = tp * vol

    cum_pv = pv.cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    return cum_pv / cum_v


# =============================================================================
# IO (fast tail read)
# =============================================================================
def read_parquet_tail(path: str, n: int = 250) -> pd.DataFrame:
    """
    Efficient-ish tail for parquet:
    - If pyarrow dataset is available, try row-group trimming.
    - Fallback: pandas read then tail (slower).
    """
    _require_pyarrow()
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        num_row_groups = pf.num_row_groups
        if num_row_groups <= 1:
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
        else:
            rows = 0
            groups = []
            rg = num_row_groups - 1
            while rg >= 0 and rows < n:
                tbl = pf.read_row_group(rg)
                groups.append(tbl)
                rows += tbl.num_rows
                rg -= 1
            import pyarrow as pa

            tbl_all = pa.concat_tables(list(reversed(groups)))
            df = tbl_all.to_pandas()
            if len(df) > n:
                df = df.tail(n).reset_index(drop=True)
        return df
    except Exception:
        df = pd.read_parquet(path, engine=PARQUET_ENGINE)
        return df.tail(n).reset_index(drop=True)


def list_tickers_15m() -> List[str]:
    pattern = os.path.join(DIR_15M, f"*{END_15M}")
    files = glob.glob(pattern)
    out: List[str] = []
    for f in files:
        base = os.path.basename(f)
        if base.endswith(END_15M):
            out.append(base[:-len(END_15M)].upper())
    return sorted(set(out))


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df

    dt = pd.to_datetime(df["date"], errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    dt = dt.dt.tz_convert(IST)
    df = df.copy()
    df["date"] = dt
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# =============================================================================
# STATE (duplicate prevention)
# =============================================================================
def _load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"last_signal": {}}  # key: "TICKER|SIDE" -> "YYYY-MM-DD"
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"last_signal": {}}


def _save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_FILE)


def _state_key(ticker: str, side: str) -> str:
    return f"{ticker.upper()}|{side.upper()}"


def allow_signal_today(state: Dict[str, Any], ticker: str, side: str, today: str, cap_per_day: int) -> bool:
    """
    Returns True if we have not already emitted cap_per_day signals for ticker/side/today.
    Implementation: track count in state["count"][today][key] (created lazily).
    """
    cap = int(cap_per_day)
    if cap <= 0:
        return True

    state.setdefault("count", {})
    state["count"].setdefault(today, {})
    key = _state_key(ticker, side)
    n = int(state["count"][today].get(key, 0))
    return n < cap


def mark_signal(state: Dict[str, Any], ticker: str, side: str, today: str) -> None:
    state.setdefault("count", {})
    state["count"].setdefault(today, {})
    key = _state_key(ticker, side)
    state["count"][today][key] = int(state["count"][today].get(key, 0)) + 1
    state.setdefault("last_signal", {})
    state["last_signal"][key] = today


# =============================================================================
# IMPULSE CLASSIFIERS (short/long)
# =============================================================================
def classify_red_impulse(row: pd.Series) -> str:
    """
    SHORT impulse (red candle).
    """
    o = _safe_float(row["open"])
    c = _safe_float(row["close"])
    h = _safe_float(row["high"])
    l = _safe_float(row["low"])
    atr = _safe_float(row["ATR15"])

    if not np.isfinite(atr) or atr <= 0:
        return ""
    if not (c < o):
        return ""

    body = abs(c - o)
    rng = (h - l) if (h >= l) else np.nan
    if not np.isfinite(rng) or rng <= 0:
        return ""

    close_near_low = ((c - l) / rng) <= SHORT_CLOSE_NEAR_LOW_MAX

    if (body >= SHORT_HUGE_RED_MIN_ATR * atr) or (rng >= SHORT_HUGE_RED_MIN_RANGE_ATR * atr):
        return "HUGE"

    if (body >= SHORT_MOD_RED_MIN_ATR * atr) and (body <= SHORT_MOD_RED_MAX_ATR * atr) and close_near_low:
        return "MODERATE"

    return ""


def classify_green_impulse(row: pd.Series) -> str:
    """
    LONG impulse (green candle), symmetric to short.
    """
    o = _safe_float(row["open"])
    c = _safe_float(row["close"])
    h = _safe_float(row["high"])
    l = _safe_float(row["low"])
    atr = _safe_float(row["ATR15"])

    if not np.isfinite(atr) or atr <= 0:
        return ""
    if not (c > o):
        return ""

    body = abs(c - o)
    rng = (h - l) if (h >= l) else np.nan
    if not np.isfinite(rng) or rng <= 0:
        return ""

    close_near_high = ((h - c) / rng) <= LONG_CLOSE_NEAR_HIGH_MAX

    if (body >= LONG_HUGE_GREEN_MIN_ATR * atr) or (rng >= LONG_HUGE_GREEN_MIN_RANGE_ATR * atr):
        return "HUGE"

    if (body >= LONG_MOD_GREEN_MIN_ATR * atr) and (body <= LONG_MOD_GREEN_MAX_ATR * atr) and close_near_high:
        return "MODERATE"

    return ""


# =============================================================================
# AVWAP REJECTION (Option B - quality boost)
# =============================================================================
def _avwap_rejection_short(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> Tuple[bool, Dict[str, Any]]:
    """
    After impulse, require "rejection" of AVWAP (fail to reclaim) and distance at entry.
    - touch: high >= AVWAP sometime between impulse+1..entry, but close remains < AVWAP on that bar
    - consec: last N closes before entry are < AVWAP
    - dist: at entry, AVWAP - close >= mult * ATR (avoid mid-zone chop)
    """
    dbg: Dict[str, Any] = {"rej_ok": False, "rej_touch": False, "rej_consec": False, "rej_dist": False}
    if not SHORT_AVWAP_REJ_ENABLED:
        dbg["rej_ok"] = True
        return True, dbg

    if "AVWAP" not in df_day.columns:
        return False, dbg

    start = impulse_idx + 1
    end = entry_idx  # inclusive range for "touch" checks (entry candle included)
    if start >= len(df_day) or entry_idx <= impulse_idx:
        return False, dbg

    seg = df_day.iloc[start : end + 1].copy()
    av = pd.to_numeric(seg["AVWAP"], errors="coerce")
    hi = pd.to_numeric(seg["high"], errors="coerce")
    cl = pd.to_numeric(seg["close"], errors="coerce")

    touch_ok = False
    if SHORT_AVWAP_REJ_TOUCH:
        # touch: high >= AVWAP and close < AVWAP on at least one candle
        touch_ok = bool(((hi >= av) & (cl < av)).fillna(False).any())

    consec_ok = False
    n = int(SHORT_AVWAP_REJ_CONSEC_CLOSES)
    if n > 0 and len(seg) >= n:
        consec_ok = bool((cl.tail(n) < av.tail(n)).fillna(False).all())

    mode = str(SHORT_AVWAP_REJ_MODE).lower().strip()
    if mode == "touch_only":
        rej_struct = touch_ok
    elif mode == "consec_only":
        rej_struct = consec_ok
    else:
        rej_struct = touch_ok or consec_ok

    # distance at entry (use entry close, not low)
    entry_close = _safe_float(df_day.at[entry_idx, "close"])
    entry_avwap = _safe_float(df_day.at[entry_idx, "AVWAP"])
    entry_atr = _safe_float(df_day.at[entry_idx, "ATR15"])
    dist_ok = np.isfinite(entry_close) and np.isfinite(entry_avwap) and np.isfinite(entry_atr) and entry_atr > 0 and ((entry_avwap - entry_close) >= SHORT_AVWAP_REJ_DIST_ATR_MULT * entry_atr)

    dbg.update({"rej_touch": touch_ok, "rej_consec": consec_ok, "rej_dist": dist_ok})
    dbg["rej_ok"] = bool(rej_struct and dist_ok)
    return dbg["rej_ok"], dbg


def _avwap_rejection_long(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> Tuple[bool, Dict[str, Any]]:
    """
    Symmetric for LONG:
    - touch: low <= AVWAP and close > AVWAP at least once between impulse+1..entry
    - consec: last N closes before entry are > AVWAP
    - dist: at entry, close - AVWAP >= mult * ATR
    """
    dbg: Dict[str, Any] = {"rej_ok": False, "rej_touch": False, "rej_consec": False, "rej_dist": False}
    if not LONG_AVWAP_REJ_ENABLED:
        dbg["rej_ok"] = True
        return True, dbg

    if "AVWAP" not in df_day.columns:
        return False, dbg

    start = impulse_idx + 1
    end = entry_idx
    if start >= len(df_day) or entry_idx <= impulse_idx:
        return False, dbg

    seg = df_day.iloc[start : end + 1].copy()
    av = pd.to_numeric(seg["AVWAP"], errors="coerce")
    lo = pd.to_numeric(seg["low"], errors="coerce")
    cl = pd.to_numeric(seg["close"], errors="coerce")

    touch_ok = False
    if LONG_AVWAP_REJ_TOUCH:
        touch_ok = bool(((lo <= av) & (cl > av)).fillna(False).any())

    consec_ok = False
    n = int(LONG_AVWAP_REJ_CONSEC_CLOSES)
    if n > 0 and len(seg) >= n:
        consec_ok = bool((cl.tail(n) > av.tail(n)).fillna(False).all())

    mode = str(LONG_AVWAP_REJ_MODE).lower().strip()
    if mode == "touch_only":
        rej_struct = touch_ok
    elif mode == "consec_only":
        rej_struct = consec_ok
    else:
        rej_struct = touch_ok or consec_ok

    entry_close = _safe_float(df_day.at[entry_idx, "close"])
    entry_avwap = _safe_float(df_day.at[entry_idx, "AVWAP"])
    entry_atr = _safe_float(df_day.at[entry_idx, "ATR15"])
    dist_ok = np.isfinite(entry_close) and np.isfinite(entry_avwap) and np.isfinite(entry_atr) and entry_atr > 0 and ((entry_close - entry_avwap) >= LONG_AVWAP_REJ_DIST_ATR_MULT * entry_atr)

    dbg.update({"rej_touch": touch_ok, "rej_consec": consec_ok, "rej_dist": dist_ok})
    dbg["rej_ok"] = bool(rej_struct and dist_ok)
    return dbg["rej_ok"], dbg


# =============================================================================
# LIVE ENTRY CHECKS (based on v11 intraday pattern logic)
# =============================================================================
@dataclass
class LiveSignal:
    ticker: str
    side: str  # "SHORT" / "LONG"
    bar_time_ist: pd.Timestamp
    setup: str
    entry_price: float
    sl_price: float
    target_price: float
    score: float
    diagnostics: Dict[str, Any]


def _prepare_today_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make today's df_day with required columns:
    - ATR15, EMA20, EMA50, RSI15, STOCHK15, STOCHD15, ADX15, AVWAP
    """
    df = normalize_dates(df)
    if df.empty:
        return df

    # session filter
    df = df[df["date"].apply(in_session)].copy()
    if df.empty:
        return df

    today = now_ist().date()
    df["day"] = df["date"].dt.tz_convert(IST).dt.date
    df_day = df[df["day"] == today].copy()
    if df_day.empty:
        return df_day

    df_day = df_day.sort_values("date").reset_index(drop=True)

    # Ensure numeric OHLC
    for c in ["open", "high", "low", "close"]:
        df_day[c] = pd.to_numeric(df_day[c], errors="coerce")

    # ATR
    if "ATR" in df_day.columns:
        df_day["ATR15"] = pd.to_numeric(df_day["ATR"], errors="coerce")
    else:
        df_day["ATR15"] = compute_atr14(df_day)

    # EMA20/50
    if "EMA_20" in df_day.columns:
        df_day["EMA20"] = pd.to_numeric(df_day["EMA_20"], errors="coerce")
    else:
        df_day["EMA20"] = ensure_ema(df_day["close"], 20)

    if "EMA_50" in df_day.columns:
        df_day["EMA50"] = pd.to_numeric(df_day["EMA_50"], errors="coerce")
    else:
        df_day["EMA50"] = ensure_ema(df_day["close"], 50)

    # RSI
    if "RSI" in df_day.columns:
        df_day["RSI15"] = pd.to_numeric(df_day["RSI"], errors="coerce")
    else:
        df_day["RSI15"] = compute_rsi14(df_day["close"])

    # Stoch
    if "Stoch_%K" in df_day.columns:
        df_day["STOCHK15"] = pd.to_numeric(df_day["Stoch_%K"], errors="coerce")
        df_day["STOCHD15"] = pd.to_numeric(df_day.get("Stoch_%D", np.nan), errors="coerce")
    else:
        k, d = compute_stoch_14_3(df_day)
        df_day["STOCHK15"] = k
        df_day["STOCHD15"] = d

    # ADX
    if "ADX" in df_day.columns:
        df_day["ADX15"] = pd.to_numeric(df_day["ADX"], errors="coerce")
    else:
        df_day["ADX15"] = compute_adx14(df_day)

    # AVWAP anchored to day start
    df_day["AVWAP"] = compute_day_avwap(df_day)

    return df_day


def _score_signal_short(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> float:
    """
    Simple ranking score used for optional Top-N:
    - larger ADX slope
    - larger AVWAP distance at entry (in ATR units)
    - larger impulse body in ATR units
    """
    adx = _safe_float(df_day.at[entry_idx, "ADX15"])
    adx_prev2 = _safe_float(df_day.at[max(0, entry_idx - 2), "ADX15"])
    adx_slope = (adx - adx_prev2) if (np.isfinite(adx) and np.isfinite(adx_prev2)) else 0.0

    av = _safe_float(df_day.at[entry_idx, "AVWAP"])
    cl = _safe_float(df_day.at[entry_idx, "close"])
    atr = _safe_float(df_day.at[entry_idx, "ATR15"])
    dist_atr = ((av - cl) / atr) if (np.isfinite(av) and np.isfinite(cl) and np.isfinite(atr) and atr > 0) else 0.0

    # impulse body in ATR
    o = _safe_float(df_day.at[impulse_idx, "open"])
    c = _safe_float(df_day.at[impulse_idx, "close"])
    body_atr = (abs(c - o) / atr) if (np.isfinite(o) and np.isfinite(c) and np.isfinite(atr) and atr > 0) else 0.0

    return float(adx_slope + 1.5 * dist_atr + 0.7 * body_atr)


def _score_signal_long(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> float:
    adx = _safe_float(df_day.at[entry_idx, "ADX15"])
    adx_prev2 = _safe_float(df_day.at[max(0, entry_idx - 2), "ADX15"])
    adx_slope = (adx - adx_prev2) if (np.isfinite(adx) and np.isfinite(adx_prev2)) else 0.0

    av = _safe_float(df_day.at[entry_idx, "AVWAP"])
    cl = _safe_float(df_day.at[entry_idx, "close"])
    atr = _safe_float(df_day.at[entry_idx, "ATR15"])
    dist_atr = ((cl - av) / atr) if (np.isfinite(av) and np.isfinite(cl) and np.isfinite(atr) and atr > 0) else 0.0

    o = _safe_float(df_day.at[impulse_idx, "open"])
    c = _safe_float(df_day.at[impulse_idx, "close"])
    body_atr = (abs(c - o) / atr) if (np.isfinite(o) and np.isfinite(c) and np.isfinite(atr) and atr > 0) else 0.0

    return float(adx_slope + 1.5 * dist_atr + 0.7 * body_atr)


def _check_common_filters_short(df_day: pd.DataFrame, i: int) -> Tuple[bool, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}

    adx = _safe_float(df_day.at[i, "ADX15"])
    rsi = _safe_float(df_day.at[i, "RSI15"])
    k = _safe_float(df_day.at[i, "STOCHK15"])
    d = _safe_float(df_day.at[i, "STOCHD15"])

    adx_ok = (
        np.isfinite(adx)
        and adx >= SHORT_ADX_MIN
        and _twice_increasing(df_day, i, "ADX15")
        and _slope_ok(df_day, i, "ADX15", SHORT_ADX_SLOPE_MIN, direction="up")
    )
    rsi_ok = np.isfinite(rsi) and (rsi <= SHORT_RSI_MAX) and _twice_decreasing(df_day, i, "RSI15")
    stoch_ok = np.isfinite(k) and np.isfinite(d) and (k <= SHORT_STOCHK_MAX) and (k < d) and _twice_decreasing(df_day, i, "STOCHK15")

    dbg.update({"adx": adx, "rsi": rsi, "k": k, "d": d, "adx_ok": adx_ok, "rsi_ok": rsi_ok, "stoch_ok": stoch_ok})

    if not (adx_ok and rsi_ok and stoch_ok):
        return False, dbg

    # Strict trend: EMA20 < EMA50 and close < EMA20 and close < AVWAP
    close1 = _safe_float(df_day.at[i, "close"])
    ema20 = _safe_float(df_day.at[i, "EMA20"])
    ema50 = _safe_float(df_day.at[i, "EMA50"])
    av = _safe_float(df_day.at[i, "AVWAP"])

    ema_ok = True
    av_ok = True

    if SHORT_REQUIRE_EMA_TREND:
        ema_ok = np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(close1) and (ema20 < ema50) and (close1 < ema20)

    if SHORT_REQUIRE_AVWAP_BELOW:
        av_ok = np.isfinite(av) and np.isfinite(close1) and (close1 < av)

    dbg.update({"close": close1, "ema20": ema20, "ema50": ema50, "avwap": av, "ema_trend_ok": ema_ok, "avwap_ok": av_ok})

    return bool(ema_ok and av_ok), dbg


def _check_common_filters_long(df_day: pd.DataFrame, i: int) -> Tuple[bool, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}

    adx = _safe_float(df_day.at[i, "ADX15"])
    rsi = _safe_float(df_day.at[i, "RSI15"])
    k = _safe_float(df_day.at[i, "STOCHK15"])
    d = _safe_float(df_day.at[i, "STOCHD15"])

    adx_ok = (
        np.isfinite(adx)
        and adx >= LONG_ADX_MIN
        and _twice_increasing(df_day, i, "ADX15")
        and _slope_ok(df_day, i, "ADX15", LONG_ADX_SLOPE_MIN, direction="up")
    )

    # LONG: RSI twice increasing, Stoch bullish (K > D) and K twice increasing
    rsi_ok = np.isfinite(rsi) and (rsi >= LONG_RSI_MIN) and _twice_increasing(df_day, i, "RSI15")
    stoch_ok = np.isfinite(k) and np.isfinite(d) and (k >= LONG_STOCHK_MIN) and (k > d) and _twice_increasing(df_day, i, "STOCHK15")

    dbg.update({"adx": adx, "rsi": rsi, "k": k, "d": d, "adx_ok": adx_ok, "rsi_ok": rsi_ok, "stoch_ok": stoch_ok})

    if not (adx_ok and rsi_ok and stoch_ok):
        return False, dbg

    close1 = _safe_float(df_day.at[i, "close"])
    ema20 = _safe_float(df_day.at[i, "EMA20"])
    ema50 = _safe_float(df_day.at[i, "EMA50"])
    av = _safe_float(df_day.at[i, "AVWAP"])

    ema_ok = True
    av_ok = True

    if LONG_REQUIRE_EMA_TREND:
        ema_ok = np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(close1) and (ema20 > ema50) and (close1 > ema20)

    if LONG_REQUIRE_AVWAP_ABOVE:
        av_ok = np.isfinite(av) and np.isfinite(close1) and (close1 > av)

    dbg.update({"close": close1, "ema20": ema20, "ema50": ema50, "avwap": av, "ema_trend_ok": ema_ok, "avwap_ok": av_ok})

    return bool(ema_ok and av_ok), dbg


def _latest_entry_signals_for_ticker(ticker: str, df_day: pd.DataFrame, state: Dict[str, Any]) -> Tuple[List[LiveSignal], List[Dict[str, Any]]]:
    """
    Returns (signals, checks_rows)
    - signals: list of LiveSignal for SHORT/LONG if latest bar triggers entry
    - checks_rows: diagnostic rows for both sides (even if not triggered)
    """
    signals: List[LiveSignal] = []
    checks: List[Dict[str, Any]] = []

    if df_day.empty or len(df_day) < 7:
        return signals, checks

    entry_idx = len(df_day) - 1
    entry_ts = df_day.at[entry_idx, "date"]
    today_str = str(entry_ts.tz_convert(IST).date())

    # Ensure we have enough bars left constraint (rough: based on time remaining to SESSION_END)
    # For live, this is optional; keep light.
    # If you want, uncomment and enforce.
    # bars_left = len(df_day) - entry_idx - 1  # always 0 in live latest-bar check
    # (kept for parity; not used)

    # -------------------------
    # SHORT side checks
    # -------------------------
    short_window_ok = _in_windows(entry_ts, SHORT_SIGNAL_WINDOWS, SHORT_USE_TIME_WINDOWS)
    short_allowed = allow_signal_today(state, ticker, "SHORT", today_str, SHORT_CAP_PER_TICKER_PER_DAY)
    short_triggered = False
    short_setup = ""
    short_entry_price = np.nan
    short_diag: Dict[str, Any] = {"side": "SHORT", "window_ok": short_window_ok, "cap_ok": short_allowed}

    if short_window_ok and short_allowed:
        # check patterns around latest candle, limited search
        # Candidate impulse indices around entry: entry-1 (MOD break), entry-2 (MOD pullback), and small range for HUGE
        candidates_i = list(range(max(2, entry_idx - 6), entry_idx))
        for i in candidates_i:
            impulse_type = classify_red_impulse(df_day.iloc[i])
            if impulse_type == "":
                continue

            # impulse must be in allowed window too (as in backtests)
            if not _in_windows(df_day.at[i, "date"], SHORT_SIGNAL_WINDOWS, SHORT_USE_TIME_WINDOWS):
                continue

            common_ok, common_dbg = _check_common_filters_short(df_day, i)
            if not common_ok:
                continue

            # MODERATE patterns
            if impulse_type == "MODERATE":
                # Option 1: break impulse low on next candle (i+1)
                if i + 1 == entry_idx:
                    low1 = _safe_float(df_day.at[i, "low"])
                    buf = _buffer(low1)
                    trigger = low1 - buf
                    low_entry = _safe_float(df_day.at[entry_idx, "low"])
                    close_entry = _safe_float(df_day.at[entry_idx, "close"])

                    close_confirm_ok = True  # if you later add "close-confirm", enforce close_entry < trigger
                    if close_confirm_ok and np.isfinite(close_entry) and np.isfinite(trigger):
                        close_confirm_ok = (close_entry < trigger)

                    if np.isfinite(low_entry) and np.isfinite(trigger) and (low_entry < trigger) and close_confirm_ok:
                        rej_ok, rej_dbg = _avwap_rejection_short(df_day, i, entry_idx)
                        if not rej_ok:
                            continue

                        short_triggered = True
                        short_setup = "A_MOD_BREAK_C1_LOW"
                        short_entry_price = float(trigger)

                        score = _score_signal_short(df_day, i, entry_idx)
                        sl = short_entry_price * (1.0 + SHORT_STOP_PCT)
                        tgt = short_entry_price * (1.0 - SHORT_TARGET_PCT)

                        diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger, "low_entry": low_entry, "close_entry": close_entry}
                        signals.append(LiveSignal(ticker=ticker, side="SHORT", bar_time_ist=entry_ts, setup=short_setup, entry_price=short_entry_price, sl_price=sl, target_price=tgt, score=score, diagnostics=diag))
                        break

                # Option 2: impulse i, pullback i+1 (small green below AVWAP), entry i+2
                if i + 2 == entry_idx:
                    c2 = df_day.iloc[i + 1]
                    c2o, c2c = _safe_float(c2["open"]), _safe_float(c2["close"])
                    c2_body = abs(c2c - c2o)
                    c2_atr = _safe_float(c2.get("ATR15", df_day.at[i, "ATR15"]))
                    c2_av = _safe_float(c2.get("AVWAP", np.nan))

                    c2_small_green = (np.isfinite(c2c) and np.isfinite(c2o) and c2c > c2o and np.isfinite(c2_atr) and c2_atr > 0 and (c2_body <= SHORT_SMALL_GREEN_MAX_ATR * c2_atr))
                    c2_below_avwap = np.isfinite(c2_av) and np.isfinite(c2c) and (c2c < c2_av)

                    if c2_small_green and c2_below_avwap:
                        low2 = _safe_float(c2["low"])
                        buf = _buffer(low2)
                        trigger = low2 - buf

                        low_entry = _safe_float(df_day.at[entry_idx, "low"])
                        close_entry = _safe_float(df_day.at[entry_idx, "close"])

                        close_confirm_ok = True
                        if close_confirm_ok and np.isfinite(close_entry) and np.isfinite(trigger):
                            close_confirm_ok = (close_entry < trigger)

                        if np.isfinite(low_entry) and np.isfinite(trigger) and (low_entry < trigger) and close_confirm_ok:
                            rej_ok, rej_dbg = _avwap_rejection_short(df_day, i, entry_idx)
                            if not rej_ok:
                                continue

                            short_triggered = True
                            short_setup = "A_PULLBACK_C2_THEN_BREAK_C2_LOW"
                            short_entry_price = float(trigger)

                            score = _score_signal_short(df_day, i, entry_idx)
                            sl = short_entry_price * (1.0 + SHORT_STOP_PCT)
                            tgt = short_entry_price * (1.0 - SHORT_TARGET_PCT)

                            diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger, "low_entry": low_entry, "close_entry": close_entry, "c2_small_green": c2_small_green, "c2_below_avwap": c2_below_avwap}
                            signals.append(LiveSignal(ticker=ticker, side="SHORT", bar_time_ist=entry_ts, setup=short_setup, entry_price=short_entry_price, sl_price=sl, target_price=tgt, score=score, diagnostics=diag))
                            break

            # HUGE pattern (failed bounce) - allow last candle as breakdown after a small-bounce window
            if impulse_type == "HUGE":
                bounce_end = min(i + 3, entry_idx - 1)  # bounce must finish before entry
                if bounce_end <= i:
                    continue
                bounce = df_day.iloc[i + 1 : bounce_end + 1].copy()
                if bounce.empty:
                    continue

                # bounce must have some small green candle(s)
                bounce_atr = pd.to_numeric(bounce.get("ATR15", np.nan), errors="coerce").fillna(_safe_float(df_day.at[i, "ATR15"]))
                bounce_body = (pd.to_numeric(bounce["close"], errors="coerce") - pd.to_numeric(bounce["open"], errors="coerce")).abs()
                bounce_green = pd.to_numeric(bounce["close"], errors="coerce") > pd.to_numeric(bounce["open"], errors="coerce")
                bounce_small = bounce_body <= (SHORT_SMALL_GREEN_MAX_ATR * bounce_atr)

                if not bool((bounce_green & bounce_small).fillna(False).any()):
                    continue

                # fail criteria: all bounce closes below AVWAP or highs below mid-body
                mid_body = (_safe_float(df_day.at[i, "open"]) + _safe_float(df_day.at[i, "close"])) / 2.0
                closes = pd.to_numeric(bounce["close"], errors="coerce")
                avwaps = pd.to_numeric(bounce["AVWAP"], errors="coerce")
                fail_avwap = bool((closes < avwaps).fillna(False).all())

                highs = pd.to_numeric(bounce["high"], errors="coerce")
                fail_mid = bool((highs < mid_body).fillna(False).all())

                if not (fail_avwap or fail_mid):
                    continue

                bounce_low = float(pd.to_numeric(bounce["low"], errors="coerce").min())
                buf = _buffer(bounce_low)
                trigger = bounce_low - buf

                # Entry on latest candle: low breaks bounce_low-buffer and price still below AVWAP
                low_entry = _safe_float(df_day.at[entry_idx, "low"])
                close_entry = _safe_float(df_day.at[entry_idx, "close"])
                av_entry = _safe_float(df_day.at[entry_idx, "AVWAP"])

                if np.isfinite(av_entry) and np.isfinite(close_entry) and (close_entry >= av_entry):
                    continue

                close_confirm_ok = True
                if close_confirm_ok and np.isfinite(close_entry) and np.isfinite(trigger):
                    close_confirm_ok = (close_entry < trigger)

                if np.isfinite(low_entry) and np.isfinite(trigger) and (low_entry < trigger) and close_confirm_ok:
                    rej_ok, rej_dbg = _avwap_rejection_short(df_day, i, entry_idx)
                    if not rej_ok:
                        continue

                    short_triggered = True
                    short_setup = "B_HUGE_RED_FAILED_BOUNCE"
                    short_entry_price = float(trigger)

                    score = _score_signal_short(df_day, i, entry_idx)
                    sl = short_entry_price * (1.0 + SHORT_STOP_PCT)
                    tgt = short_entry_price * (1.0 - SHORT_TARGET_PCT)

                    diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger, "low_entry": low_entry, "close_entry": close_entry, "fail_avwap": fail_avwap, "fail_mid": fail_mid}
                    signals.append(LiveSignal(ticker=ticker, side="SHORT", bar_time_ist=entry_ts, setup=short_setup, entry_price=short_entry_price, sl_price=sl, target_price=tgt, score=score, diagnostics=diag))
                    break

    # Diagnostic row for SHORT (even if no signal)
    short_diag.update({
        "signal": bool(short_triggered),
        "setup": short_setup,
        "entry_price": float(short_entry_price) if np.isfinite(short_entry_price) else np.nan,
    })
    checks.append({"ticker": ticker, "side": "SHORT", "bar_time_ist": entry_ts, **short_diag})

    # -------------------------
    # LONG side checks
    # -------------------------
    long_window_ok = _in_windows(entry_ts, LONG_SIGNAL_WINDOWS, LONG_USE_TIME_WINDOWS)
    long_allowed = allow_signal_today(state, ticker, "LONG", today_str, LONG_CAP_PER_TICKER_PER_DAY)
    long_triggered = False
    long_setup = ""
    long_entry_price = np.nan
    long_diag: Dict[str, Any] = {"side": "LONG", "window_ok": long_window_ok, "cap_ok": long_allowed}

    if long_window_ok and long_allowed:
        candidates_i = list(range(max(2, entry_idx - 6), entry_idx))
        for i in candidates_i:
            impulse_type = classify_green_impulse(df_day.iloc[i])
            if impulse_type == "":
                continue

            if not _in_windows(df_day.at[i, "date"], LONG_SIGNAL_WINDOWS, LONG_USE_TIME_WINDOWS):
                continue

            common_ok, common_dbg = _check_common_filters_long(df_day, i)
            if not common_ok:
                continue

            if impulse_type == "MODERATE":
                # Option 1: break impulse high on next candle (i+1)
                if i + 1 == entry_idx:
                    high1 = _safe_float(df_day.at[i, "high"])
                    buf = _buffer(high1)
                    trigger = high1 + buf
                    high_entry = _safe_float(df_day.at[entry_idx, "high"])
                    close_entry = _safe_float(df_day.at[entry_idx, "close"])

                    close_confirm_ok = True
                    if close_confirm_ok and np.isfinite(close_entry) and np.isfinite(trigger):
                        close_confirm_ok = (close_entry > trigger)

                    if np.isfinite(high_entry) and np.isfinite(trigger) and (high_entry > trigger) and close_confirm_ok:
                        rej_ok, rej_dbg = _avwap_rejection_long(df_day, i, entry_idx)
                        if not rej_ok:
                            continue

                        long_triggered = True
                        long_setup = "A_MOD_BREAK_C1_HIGH"
                        long_entry_price = float(trigger)

                        score = _score_signal_long(df_day, i, entry_idx)
                        sl = long_entry_price * (1.0 - LONG_STOP_PCT)
                        tgt = long_entry_price * (1.0 + LONG_TARGET_PCT)

                        diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger, "high_entry": high_entry, "close_entry": close_entry}
                        signals.append(LiveSignal(ticker=ticker, side="LONG", bar_time_ist=entry_ts, setup=long_setup, entry_price=long_entry_price, sl_price=sl, target_price=tgt, score=score, diagnostics=diag))
                        break

                # Option 2: small red pullback C2 + above AVWAP, then break C2 high on C3
                if i + 2 == entry_idx:
                    c2 = df_day.iloc[i + 1]
                    c2o, c2c = _safe_float(c2["open"]), _safe_float(c2["close"])
                    c2_body = abs(c2c - c2o)
                    c2_atr = _safe_float(c2.get("ATR15", df_day.at[i, "ATR15"]))
                    c2_av = _safe_float(c2.get("AVWAP", np.nan))

                    c2_small_red = (np.isfinite(c2c) and np.isfinite(c2o) and c2c < c2o and np.isfinite(c2_atr) and c2_atr > 0 and (c2_body <= LONG_SMALL_RED_MAX_ATR * c2_atr))
                    c2_above_avwap = np.isfinite(c2_av) and np.isfinite(c2c) and (c2c > c2_av)

                    if c2_small_red and c2_above_avwap:
                        high2 = _safe_float(c2["high"])
                        buf = _buffer(high2)
                        trigger = high2 + buf

                        high_entry = _safe_float(df_day.at[entry_idx, "high"])
                        close_entry = _safe_float(df_day.at[entry_idx, "close"])

                        close_confirm_ok = True
                        if close_confirm_ok and np.isfinite(close_entry) and np.isfinite(trigger):
                            close_confirm_ok = (close_entry > trigger)

                        if np.isfinite(high_entry) and np.isfinite(trigger) and (high_entry > trigger) and close_confirm_ok:
                            rej_ok, rej_dbg = _avwap_rejection_long(df_day, i, entry_idx)
                            if not rej_ok:
                                continue

                            long_triggered = True
                            long_setup = "A_PULLBACK_C2_THEN_BREAK_C2_HIGH"
                            long_entry_price = float(trigger)

                            score = _score_signal_long(df_day, i, entry_idx)
                            sl = long_entry_price * (1.0 - LONG_STOP_PCT)
                            tgt = long_entry_price * (1.0 + LONG_TARGET_PCT)

                            diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger, "high_entry": high_entry, "close_entry": close_entry, "c2_small_red": c2_small_red, "c2_above_avwap": c2_above_avwap}
                            signals.append(LiveSignal(ticker=ticker, side="LONG", bar_time_ist=entry_ts, setup=long_setup, entry_price=long_entry_price, sl_price=sl, target_price=tgt, score=score, diagnostics=diag))
                            break

            if impulse_type == "HUGE":
                # Huge green failed retrace (symmetric to short huge failed bounce)
                bounce_end = min(i + 3, entry_idx - 1)
                if bounce_end <= i:
                    continue
                bounce = df_day.iloc[i + 1 : bounce_end + 1].copy()
                if bounce.empty:
                    continue

                bounce_atr = pd.to_numeric(bounce.get("ATR15", np.nan), errors="coerce").fillna(_safe_float(df_day.at[i, "ATR15"]))
                bounce_body = (pd.to_numeric(bounce["close"], errors="coerce") - pd.to_numeric(bounce["open"], errors="coerce")).abs()
                bounce_red = pd.to_numeric(bounce["close"], errors="coerce") < pd.to_numeric(bounce["open"], errors="coerce")
                bounce_small = bounce_body <= (LONG_SMALL_RED_MAX_ATR * bounce_atr)

                if not bool((bounce_red & bounce_small).fillna(False).any()):
                    continue

                mid_body = (_safe_float(df_day.at[i, "open"]) + _safe_float(df_day.at[i, "close"])) / 2.0
                closes = pd.to_numeric(bounce["close"], errors="coerce")
                avwaps = pd.to_numeric(bounce["AVWAP"], errors="coerce")
                fail_avwap = bool((closes > avwaps).fillna(False).all())  # all closes above AVWAP
                lows = pd.to_numeric(bounce["low"], errors="coerce")
                fail_mid = bool((lows > mid_body).fillna(False).all())     # all lows above mid-body

                if not (fail_avwap or fail_mid):
                    continue

                bounce_high = float(pd.to_numeric(bounce["high"], errors="coerce").max())
                buf = _buffer(bounce_high)
                trigger = bounce_high + buf

                high_entry = _safe_float(df_day.at[entry_idx, "high"])
                close_entry = _safe_float(df_day.at[entry_idx, "close"])
                av_entry = _safe_float(df_day.at[entry_idx, "AVWAP"])

                if np.isfinite(av_entry) and np.isfinite(close_entry) and (close_entry <= av_entry):
                    continue

                close_confirm_ok = True
                if close_confirm_ok and np.isfinite(close_entry) and np.isfinite(trigger):
                    close_confirm_ok = (close_entry > trigger)

                if np.isfinite(high_entry) and np.isfinite(trigger) and (high_entry > trigger) and close_confirm_ok:
                    rej_ok, rej_dbg = _avwap_rejection_long(df_day, i, entry_idx)
                    if not rej_ok:
                        continue

                    long_triggered = True
                    long_setup = "B_HUGE_GREEN_FAILED_RETRACE"
                    long_entry_price = float(trigger)

                    score = _score_signal_long(df_day, i, entry_idx)
                    sl = long_entry_price * (1.0 - LONG_STOP_PCT)
                    tgt = long_entry_price * (1.0 + LONG_TARGET_PCT)

                    diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger, "high_entry": high_entry, "close_entry": close_entry, "fail_avwap": fail_avwap, "fail_mid": fail_mid}
                    signals.append(LiveSignal(ticker=ticker, side="LONG", bar_time_ist=entry_ts, setup=long_setup, entry_price=long_entry_price, sl_price=sl, target_price=tgt, score=score, diagnostics=diag))
                    break

    long_diag.update({
        "signal": bool(long_triggered),
        "setup": long_setup,
        "entry_price": float(long_entry_price) if np.isfinite(long_entry_price) else np.nan,
    })
    checks.append({"ticker": ticker, "side": "LONG", "bar_time_ist": entry_ts, **long_diag})

    # If we emitted a signal, mark state
    for s in signals:
        mark_signal(state, s.ticker, s.side, today_str)

    return signals, checks


# =============================================================================
# TRADING DAY HELPERS (holiday handling)
# =============================================================================
def _read_holidays_safe() -> set:
    try:
        return set(core._read_holidays(core.HOLIDAYS_FILE_DEFAULT))
    except Exception:
        return set()


def _read_special_open_days_safe() -> set:
    # Some codebases have this; keep safe.
    fn = getattr(core, "_read_special_trading_days", None)
    if fn is None:
        return set()
    try:
        return set(fn(getattr(core, "SPECIAL_TRADING_DAYS_FILE_DEFAULT", "special_trading_days.txt")))
    except Exception:
        return set()


def is_trading_day_safe(d: datetime.date, holidays: set, special_open: set) -> bool:
    """
    Try to call core._is_trading_day with (d, holidays, special_open) if supported,
    else fallback to (d, holidays), else simple weekday check.
    """
    fn = getattr(core, "_is_trading_day", None)
    if fn is None:
        return d.weekday() < 5 and (d not in holidays)

    try:
        # Try 3-arg version
        return bool(fn(d, holidays, special_open))
    except TypeError:
        try:
            return bool(fn(d, holidays))
        except Exception:
            return d.weekday() < 5 and (d not in holidays)
    except Exception:
        return d.weekday() < 5 and (d not in holidays)


# =============================================================================
# OPTIONAL UPDATER
# =============================================================================
def run_update_15m_once(holidays: set) -> None:
    core.run_mode(
        mode="15min",
        context="live",
        max_workers=4,
        force_today_daily=False,
        skip_if_fresh=True,
        intraday_ts="end",
        holidays=holidays,
        refresh_tokens=False,
        report_dir=str(REPORTS_DIR),
        print_missing_rows=False,
        print_missing_rows_max=200,
    )


# =============================================================================
# SLOT SCHEDULER
# =============================================================================
def _next_slot_after(now: datetime) -> datetime:
    """
    Returns the next 15-min slot datetime (IST) between START_TIME and END_TIME (inclusive).
    If now before START_TIME -> returns today START_TIME.
    If after END_TIME -> returns tomorrow START_TIME.
    """
    now = now.astimezone(IST)
    today = now.date()
    start_dt = IST.localize(datetime.combine(today, START_TIME))
    end_dt = IST.localize(datetime.combine(today, END_TIME))

    if now <= start_dt:
        return start_dt

    if now > end_dt:
        # tomorrow start
        tomorrow = today + timedelta(days=1)
        return IST.localize(datetime.combine(tomorrow, START_TIME))

    # Round up to next multiple of 15 minutes
    minute = (now.minute // 15) * 15
    slot = now.replace(minute=minute, second=0, microsecond=0)
    if slot < now:
        slot += timedelta(minutes=15)

    # clamp
    if slot < start_dt:
        slot = start_dt
    if slot > end_dt:
        # tomorrow start
        tomorrow = today + timedelta(days=1)
        slot = IST.localize(datetime.combine(tomorrow, START_TIME))
    return slot


def _sleep_until(dt: datetime) -> None:
    now = now_ist()
    delta = (dt - now).total_seconds()
    if delta > 0:
        time.sleep(delta)


# =============================================================================
# RUN ONE SCAN (A or B run)
# =============================================================================
def run_one_scan(run_tag: str = "A") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (checks_df, signals_df) for this scan.
    """
    tickers = list_tickers_15m()
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    state = _load_state()

    all_checks: List[Dict[str, Any]] = []
    all_signals: List[Dict[str, Any]] = []

    for idx, t in enumerate(tickers, start=1):
        path = os.path.join(DIR_15M, f"{t}{END_15M}")
        try:
            df_raw = read_parquet_tail(path, n=TAIL_ROWS)
            df_raw = normalize_dates(df_raw)
            df_day = _prepare_today_df(df_raw)
            if df_day.empty:
                # still emit a minimal checks row so you can see "no data"
                all_checks.append({"ticker": t, "side": "SHORT", "bar_time_ist": pd.NaT, "no_data": True})
                all_checks.append({"ticker": t, "side": "LONG", "bar_time_ist": pd.NaT, "no_data": True})
                continue

            signals, checks = _latest_entry_signals_for_ticker(t, df_day, state)
            all_checks.extend(checks)

            for s in signals:
                all_signals.append({
                    "ticker": s.ticker,
                    "side": s.side,
                    "bar_time_ist": s.bar_time_ist,
                    "setup": s.setup,
                    "entry_price": s.entry_price,
                    "sl_price": s.sl_price,
                    "target_price": s.target_price,
                    "score": s.score,
                    # store diagnostics as JSON string to stay parquet-friendly
                    "diagnostics_json": json.dumps(s.diagnostics, default=str),
                })
        except Exception as e:
            all_checks.append({"ticker": t, "side": "SHORT", "bar_time_ist": pd.NaT, "error": str(e)})
            all_checks.append({"ticker": t, "side": "LONG", "bar_time_ist": pd.NaT, "error": str(e)})

        if idx % 100 == 0:
            print(f"  scanned {idx}/{len(tickers)} | signals_so_far={len(all_signals)}")

    # Persist state after scanning
    _save_state(state)

    checks_df = pd.DataFrame(all_checks)
    signals_df = pd.DataFrame(all_signals)

    # Optional Top-N per run (by score) - applied per side
    if USE_TOPN_PER_RUN and (not signals_df.empty):
        keep = []
        for side in ["SHORT", "LONG"]:
            df_side = signals_df[signals_df["side"] == side].copy()
            df_side = df_side.sort_values(["score"], ascending=False).head(int(TOPN_PER_RUN))
            keep.append(df_side)
        signals_df = pd.concat(keep, ignore_index=True) if keep else signals_df

    # Save outputs
    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    day_folder = now_ist().strftime("%Y%m%d")

    out_checks_day = OUT_CHECKS_DIR / day_folder
    out_signals_day = OUT_SIGNALS_DIR / day_folder
    out_checks_day.mkdir(parents=True, exist_ok=True)
    out_signals_day.mkdir(parents=True, exist_ok=True)

    checks_path = out_checks_day / f"checks_{ts}_{run_tag}.parquet"
    signals_path = out_signals_day / f"signals_{ts}_{run_tag}.parquet"

    _require_pyarrow()
    checks_df.to_parquet(checks_path, index=False, engine=PARQUET_ENGINE)
    signals_df.to_parquet(signals_path, index=False, engine=PARQUET_ENGINE)

    print(f"[SAVED] {checks_path}")
    print(f"[SAVED] {signals_path}")
    print(f"[RUN ] done | checks={len(checks_df)} rows | signals={len(signals_df)} rows")

    return checks_df, signals_df


# =============================================================================
# MAIN LOOP
# =============================================================================
def main() -> None:
    print("[LIVE] ALGO-SM1 AVWAP v11 combined LIVE scanner (15m)")
    print(f"[INFO] DIR_15M={DIR_15M} | tickers={len(list_tickers_15m())}")
    print(f"[INFO] SHORT windows: {', '.join([f'{a.strftime('%H:%M')}-{b.strftime('%H:%M')}' for a,b in SHORT_SIGNAL_WINDOWS]) if SHORT_USE_TIME_WINDOWS else 'OFF'}")
    print(f"[INFO] LONG  windows: {', '.join([f'{a.strftime('%H:%M')}-{b.strftime('%H:%M')}' for a,b in LONG_SIGNAL_WINDOWS]) if LONG_USE_TIME_WINDOWS else 'OFF'}")
    print(f"[INFO] SHORT AVWAP rejection: enabled={SHORT_AVWAP_REJ_ENABLED}, mode={SHORT_AVWAP_REJ_MODE}, touch={SHORT_AVWAP_REJ_TOUCH}, consec>={SHORT_AVWAP_REJ_CONSEC_CLOSES}, dist={SHORT_AVWAP_REJ_DIST_ATR_MULT}*ATR")
    print(f"[INFO] LONG  AVWAP rejection: enabled={LONG_AVWAP_REJ_ENABLED}, mode={LONG_AVWAP_REJ_MODE}, touch={LONG_AVWAP_REJ_TOUCH}, consec>={LONG_AVWAP_REJ_CONSEC_CLOSES}, dist={LONG_AVWAP_REJ_DIST_ATR_MULT}*ATR")
    print(f"[INFO] per-ticker/day cap: SHORT={SHORT_CAP_PER_TICKER_PER_DAY}, LONG={LONG_CAP_PER_TICKER_PER_DAY}")
    print(f"[INFO] UPDATE_15M_BEFORE_CHECK={UPDATE_15M_BEFORE_CHECK} | ENABLE_SECOND_RUN={ENABLE_SECOND_RUN} (gap={SECOND_RUN_GAP_SECONDS}s)")

    holidays = _read_holidays_safe()
    special_open = _read_special_open_days_safe()

    while True:
        now = now_ist()
        # Hard stop for the day
        if now.time() >= HARD_STOP_TIME:
            print("[STOP] Hard-stop reached for today. Exiting.")
            return

        # Only run on trading days
        if not is_trading_day_safe(now.date(), holidays, special_open):
            nxt = _next_slot_after(now + timedelta(days=1))
            print(f"[SKIP] Not a trading day ({now.date()}). Sleeping until {nxt}.")
            _sleep_until(nxt)
            continue

        slot = _next_slot_after(now)
        if slot.date() != now.date():
            # moved to tomorrow start
            print(f"[WAIT] Next slot is tomorrow {slot}. Sleeping.")
            _sleep_until(slot)
            continue

        # Wait until slot time
        if now < slot:
            print(f"[WAIT] Sleeping until slot {slot.strftime('%Y-%m-%d %H:%M:%S%z')}")
            _sleep_until(slot)

        # If after end time, jump to tomorrow
        now = now_ist()
        if now.time() > END_TIME:
            nxt = _next_slot_after(now + timedelta(days=1))
            print(f"[DONE] Past END_TIME. Sleeping until {nxt}.")
            _sleep_until(nxt)
            continue

        # Optional updater once per slot
        if UPDATE_15M_BEFORE_CHECK:
            try:
                print(f"[UPD ] Running core 15m update at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
                run_update_15m_once(holidays)
            except Exception as e:
                print(f"[WARN] Update failed: {e!r}")

        # Run A
        print(f"[RUN ] Slot {slot.strftime('%H:%M')} | scan A")
        run_one_scan(run_tag="A")

        # Optional run B
        if ENABLE_SECOND_RUN:
            time.sleep(float(SECOND_RUN_GAP_SECONDS))
            print(f"[RUN ] Slot {slot.strftime('%H:%M')} | scan B")
            run_one_scan(run_tag="B")

        # Sleep a bit so we don't immediately re-run same slot if system clock jitter
        time.sleep(1.0)


if __name__ == "__main__":
    main()

