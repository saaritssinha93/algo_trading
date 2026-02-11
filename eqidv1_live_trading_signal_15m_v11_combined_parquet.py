# -*- coding: utf-8 -*-
"""
EQIDV1 — LIVE 15m Signal Scanner (AVWAP v11 combined: LONG + SHORT)
====================================================================

Adapted from stocks_live_trading_signal_15m_v11_combined_parquet.py, but wired
to the eqidv1 backtesting ecosystem with **refined parameters** from:
    backtesting/eqidv1/avwap_v11_refactored/avwap_common.py

Key refinements vs the original live scanner:
- Tighter SL: 0.75% (was 1.0%)
- Better R:R: SHORT TGT=1.2%, LONG TGT=1.5%
- Stricter ADX: min=25, slope_min=1.25 (SHORT) / 0.80 (LONG)
- Stricter RSI: SHORT max=55, LONG min=45
- Stricter Stoch: SHORT max=75, LONG min=25
- NEW: Volume filter — impulse bar volume >= 1.2x SMA(20) volume
- NEW: ATR% volatility filter — ATR/close >= 0.20%
- Close-confirm required on entry candle by default

Data: reads from stocks_indicators_15min_eq/ (same parquet directory).
Core: backtesting/eqidv1/trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py
"""

from __future__ import annotations

import os
import sys
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

# =============================================================================
# Wire eqidv1 core into sys.path
# =============================================================================
_ROOT = Path(__file__).resolve().parent
_EQIDV1 = _ROOT / "backtesting" / "eqidv1"
if str(_EQIDV1) not in sys.path:
    sys.path.insert(0, str(_EQIDV1))

import trading_data_continous_run_historical_alltf_v3_parquet_stocksonly as core  # noqa: E402


# =============================================================================
# TIMEZONE + DIRECTORIES
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

DIR_15M = "stocks_indicators_15min_eq"
END_15M = "_stocks_indicators_15min.parquet"

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports" / "eqidv1_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CHECKS_DIR = ROOT / "out_eqidv1_live_checks_15m"
OUT_SIGNALS_DIR = ROOT / "out_eqidv1_live_signals_15m"
OUT_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
OUT_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIR = ROOT / "logs"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "eqidv1_avwap_live_state_v11.json"

PARQUET_ENGINE = "pyarrow"

# =============================================================================
# SCHEDULER CONFIG
# =============================================================================
START_TIME = dtime(9, 15)
END_TIME = dtime(15, 30)
SECOND_RUN_GAP_SECONDS = 18
ENABLE_SECOND_RUN = True
HARD_STOP_TIME = dtime(15, 40)

# Update flag: set True to call eqidv1 core.run_mode("15min") before each scan
UPDATE_15M_BEFORE_CHECK = False

# How many tail rows to load per ticker parquet
TAIL_ROWS = 260

# =============================================================================
# SESSION FILTER
# =============================================================================
SESSION_START = dtime(9, 15, 0)
SESSION_END = dtime(14, 30, 0)

# =============================================================================
# V11 SHORT PARAMETERS — refined from avwap_common.default_short_config()
# =============================================================================
SHORT_STOP_PCT = 0.0075          # tighter SL: 0.75% (was 1.0%)
SHORT_TARGET_PCT = 0.0120        # better R:R: 1.2% (was 0.65%)

SHORT_ADX_MIN = 25.0             # stricter (was 20)
SHORT_ADX_SLOPE_MIN = 1.25       # stricter (was 1.0)

SHORT_RSI_MAX = 55.0             # stricter (was 60)
SHORT_STOCHK_MAX = 75.0          # stricter (was 80)

SHORT_REQUIRE_EMA_TREND = True
SHORT_REQUIRE_AVWAP_BELOW = True

SHORT_USE_TIME_WINDOWS = True
SHORT_SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 30, 0)),
    (dtime(13, 0, 0), dtime(14, 30, 0)),
]

# Impulse thresholds
SHORT_MOD_RED_MIN_ATR = 0.45     # slightly tighter (was 0.40)
SHORT_MOD_RED_MAX_ATR = 1.00
SHORT_HUGE_RED_MIN_ATR = 1.60    # slightly tighter (was 1.50)
SHORT_HUGE_RED_MIN_RANGE_ATR = 2.00
SHORT_CLOSE_NEAR_LOW_MAX = 0.25

SHORT_SMALL_GREEN_MAX_ATR = 0.20

# Entry buffer
BUFFER_ABS = 0.05
BUFFER_PCT = 0.0002

# AVWAP rejection
SHORT_AVWAP_REJ_ENABLED = True
SHORT_AVWAP_REJ_TOUCH = True
SHORT_AVWAP_REJ_CONSEC_CLOSES = 2
SHORT_AVWAP_REJ_DIST_ATR_MULT = 0.25
SHORT_AVWAP_REJ_MODE = "any"

SHORT_CAP_PER_TICKER_PER_DAY = 1

# =============================================================================
# V11 LONG PARAMETERS — refined from avwap_common.default_long_config()
# =============================================================================
LONG_STOP_PCT = 0.0075           # tighter SL: 0.75% (was 1.0%)
LONG_TARGET_PCT = 0.0150         # better TGT: 1.5% (was 0.65%)

LONG_ADX_MIN = 25.0              # stricter (was 20)
LONG_ADX_SLOPE_MIN = 0.80        # relaxed vs short (was 1.0)

LONG_RSI_MIN = 45.0              # stricter (was 40)
LONG_STOCHK_MIN = 25.0           # stricter (was 20)

LONG_REQUIRE_EMA_TREND = True
LONG_REQUIRE_AVWAP_ABOVE = True

LONG_USE_TIME_WINDOWS = True
LONG_SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 30, 0)),
    (dtime(13, 0, 0), dtime(14, 30, 0)),
]

# Impulse thresholds
LONG_MOD_GREEN_MIN_ATR = 0.30    # relaxed for LONG (was 0.40)
LONG_MOD_GREEN_MAX_ATR = 1.00
LONG_HUGE_GREEN_MIN_ATR = 1.60   # slightly tighter (was 1.50)
LONG_HUGE_GREEN_MIN_RANGE_ATR = 2.00
LONG_CLOSE_NEAR_HIGH_MAX = 0.25

LONG_SMALL_RED_MAX_ATR = 0.20

# AVWAP rejection
LONG_AVWAP_REJ_ENABLED = True
LONG_AVWAP_REJ_TOUCH = True
LONG_AVWAP_REJ_CONSEC_CLOSES = 2
LONG_AVWAP_REJ_DIST_ATR_MULT = 0.25
LONG_AVWAP_REJ_MODE = "any"

LONG_CAP_PER_TICKER_PER_DAY = 1

# =============================================================================
# NEW: QUALITY FILTERS (from avwap_common refactored config)
# =============================================================================
# Volume filter: impulse bar volume must be >= ratio * SMA(period) of volume
USE_VOLUME_FILTER = True
VOLUME_SMA_PERIOD = 20
VOLUME_MIN_RATIO = 1.2

# ATR% volatility filter: ATR/close must be >= threshold
USE_ATR_PCT_FILTER = True
ATR_PCT_MIN = 0.0020  # 0.20%

# Close-confirm: entry candle close must confirm the breakout direction
REQUIRE_CLOSE_CONFIRM = True

# Top-N filter (optional; usually off in live)
USE_TOPN_PER_RUN = False
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
        return float(x)
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
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
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
    return 100 - (100 / (1 + rs))


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
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr)
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr)
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
    _require_pyarrow()
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

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
            out.append(base[: -len(END_15M)].upper())
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
        return {"last_signal": {}}
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
# NEW: VOLUME + ATR% QUALITY FILTERS
# =============================================================================
def _volume_filter_ok(df_day: pd.DataFrame, bar_idx: int) -> bool:
    """Check if the impulse bar's volume is >= VOLUME_MIN_RATIO * SMA(VOLUME_SMA_PERIOD)."""
    if not USE_VOLUME_FILTER:
        return True
    if "volume" not in df_day.columns:
        return True  # pass if no volume data
    vol = _safe_float(df_day.at[bar_idx, "volume"])
    if not np.isfinite(vol) or vol <= 0:
        return False
    # Compute volume SMA over the available preceding bars
    start = max(0, bar_idx - VOLUME_SMA_PERIOD + 1)
    vol_window = pd.to_numeric(df_day.loc[start:bar_idx, "volume"], errors="coerce").dropna()
    if len(vol_window) < 3:
        return True  # not enough data, pass
    avg_vol = vol_window.mean()
    if avg_vol <= 0:
        return True
    return vol >= (VOLUME_MIN_RATIO * avg_vol)


def _atr_pct_filter_ok(df_day: pd.DataFrame, bar_idx: int) -> bool:
    """Check if ATR/close >= ATR_PCT_MIN (volatility filter)."""
    if not USE_ATR_PCT_FILTER:
        return True
    atr = _safe_float(df_day.at[bar_idx, "ATR15"])
    close = _safe_float(df_day.at[bar_idx, "close"])
    if not (np.isfinite(atr) and np.isfinite(close) and close > 0):
        return False
    return (atr / close) >= ATR_PCT_MIN


# =============================================================================
# IMPULSE CLASSIFIERS
# =============================================================================
def classify_red_impulse(row: pd.Series) -> str:
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
# AVWAP REJECTION (Option B)
# =============================================================================
def _avwap_rejection_short(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> Tuple[bool, Dict[str, Any]]:
    dbg: Dict[str, Any] = {"rej_ok": False, "rej_touch": False, "rej_consec": False, "rej_dist": False}
    if not SHORT_AVWAP_REJ_ENABLED:
        dbg["rej_ok"] = True
        return True, dbg

    if "AVWAP" not in df_day.columns:
        return False, dbg

    start = impulse_idx + 1
    end = entry_idx
    if start >= len(df_day) or entry_idx <= impulse_idx:
        return False, dbg

    seg = df_day.iloc[start: end + 1].copy()
    av = pd.to_numeric(seg["AVWAP"], errors="coerce")
    hi = pd.to_numeric(seg["high"], errors="coerce")
    cl = pd.to_numeric(seg["close"], errors="coerce")

    touch_ok = False
    if SHORT_AVWAP_REJ_TOUCH:
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

    entry_close = _safe_float(df_day.at[entry_idx, "close"])
    entry_avwap = _safe_float(df_day.at[entry_idx, "AVWAP"])
    entry_atr = _safe_float(df_day.at[entry_idx, "ATR15"])
    dist_ok = (
        np.isfinite(entry_close) and np.isfinite(entry_avwap) and np.isfinite(entry_atr)
        and entry_atr > 0 and ((entry_avwap - entry_close) >= SHORT_AVWAP_REJ_DIST_ATR_MULT * entry_atr)
    )

    dbg.update({"rej_touch": touch_ok, "rej_consec": consec_ok, "rej_dist": dist_ok})
    dbg["rej_ok"] = bool(rej_struct and dist_ok)
    return dbg["rej_ok"], dbg


def _avwap_rejection_long(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> Tuple[bool, Dict[str, Any]]:
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

    seg = df_day.iloc[start: end + 1].copy()
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
    dist_ok = (
        np.isfinite(entry_close) and np.isfinite(entry_avwap) and np.isfinite(entry_atr)
        and entry_atr > 0 and ((entry_close - entry_avwap) >= LONG_AVWAP_REJ_DIST_ATR_MULT * entry_atr)
    )

    dbg.update({"rej_touch": touch_ok, "rej_consec": consec_ok, "rej_dist": dist_ok})
    dbg["rej_ok"] = bool(rej_struct and dist_ok)
    return dbg["rej_ok"], dbg


# =============================================================================
# LIVE ENTRY CHECKS
# =============================================================================
@dataclass
class LiveSignal:
    ticker: str
    side: str
    bar_time_ist: pd.Timestamp
    setup: str
    entry_price: float
    sl_price: float
    target_price: float
    score: float
    diagnostics: Dict[str, Any]


def _prepare_today_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build today's df_day with required indicator columns."""
    df = normalize_dates(df)
    if df.empty:
        return df

    df = df[df["date"].apply(in_session)].copy()
    if df.empty:
        return df

    today = now_ist().date()
    df["day"] = df["date"].dt.tz_convert(IST).dt.date
    df_day = df[df["day"] == today].copy()
    if df_day.empty:
        return df_day

    df_day = df_day.sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        df_day[c] = pd.to_numeric(df_day[c], errors="coerce")

    # ATR
    if "ATR" in df_day.columns:
        df_day["ATR15"] = pd.to_numeric(df_day["ATR"], errors="coerce")
    else:
        df_day["ATR15"] = compute_atr14(df_day)

    # EMAs
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

    # Stochastic
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

    # AVWAP
    df_day["AVWAP"] = compute_day_avwap(df_day)

    return df_day


def _score_signal_short(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> float:
    adx = _safe_float(df_day.at[entry_idx, "ADX15"])
    adx_prev2 = _safe_float(df_day.at[max(0, entry_idx - 2), "ADX15"])
    adx_slope = (adx - adx_prev2) if (np.isfinite(adx) and np.isfinite(adx_prev2)) else 0.0

    av = _safe_float(df_day.at[entry_idx, "AVWAP"])
    cl = _safe_float(df_day.at[entry_idx, "close"])
    atr = _safe_float(df_day.at[entry_idx, "ATR15"])
    dist_atr = ((av - cl) / atr) if (np.isfinite(av) and np.isfinite(cl) and np.isfinite(atr) and atr > 0) else 0.0

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
        np.isfinite(adx) and adx >= SHORT_ADX_MIN
        and _twice_increasing(df_day, i, "ADX15")
        and _slope_ok(df_day, i, "ADX15", SHORT_ADX_SLOPE_MIN, direction="up")
    )
    rsi_ok = np.isfinite(rsi) and (rsi <= SHORT_RSI_MAX) and _twice_decreasing(df_day, i, "RSI15")
    stoch_ok = np.isfinite(k) and np.isfinite(d) and (k <= SHORT_STOCHK_MAX) and (k < d) and _twice_decreasing(df_day, i, "STOCHK15")

    # NEW: volume + ATR% quality filters on impulse bar
    vol_ok = _volume_filter_ok(df_day, i)
    atr_pct_ok = _atr_pct_filter_ok(df_day, i)

    dbg.update({
        "adx": adx, "rsi": rsi, "k": k, "d": d,
        "adx_ok": adx_ok, "rsi_ok": rsi_ok, "stoch_ok": stoch_ok,
        "vol_ok": vol_ok, "atr_pct_ok": atr_pct_ok,
    })

    if not (adx_ok and rsi_ok and stoch_ok and vol_ok and atr_pct_ok):
        return False, dbg

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
        np.isfinite(adx) and adx >= LONG_ADX_MIN
        and _twice_increasing(df_day, i, "ADX15")
        and _slope_ok(df_day, i, "ADX15", LONG_ADX_SLOPE_MIN, direction="up")
    )
    rsi_ok = np.isfinite(rsi) and (rsi >= LONG_RSI_MIN) and _twice_increasing(df_day, i, "RSI15")
    stoch_ok = np.isfinite(k) and np.isfinite(d) and (k >= LONG_STOCHK_MIN) and (k > d) and _twice_increasing(df_day, i, "STOCHK15")

    # NEW: volume + ATR% quality filters on impulse bar
    vol_ok = _volume_filter_ok(df_day, i)
    atr_pct_ok = _atr_pct_filter_ok(df_day, i)

    dbg.update({
        "adx": adx, "rsi": rsi, "k": k, "d": d,
        "adx_ok": adx_ok, "rsi_ok": rsi_ok, "stoch_ok": stoch_ok,
        "vol_ok": vol_ok, "atr_pct_ok": atr_pct_ok,
    })

    if not (adx_ok and rsi_ok and stoch_ok and vol_ok and atr_pct_ok):
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


# =============================================================================
# MAIN SIGNAL DETECTION (per-ticker, both sides)
# =============================================================================
def _latest_entry_signals_for_ticker(
    ticker: str, df_day: pd.DataFrame, state: Dict[str, Any]
) -> Tuple[List[LiveSignal], List[Dict[str, Any]]]:
    signals: List[LiveSignal] = []
    checks: List[Dict[str, Any]] = []

    if df_day.empty or len(df_day) < 7:
        return signals, checks

    entry_idx = len(df_day) - 1
    entry_ts = df_day.at[entry_idx, "date"]
    today_str = str(entry_ts.tz_convert(IST).date())

    # ---- SHORT side ----
    short_window_ok = _in_windows(entry_ts, SHORT_SIGNAL_WINDOWS, SHORT_USE_TIME_WINDOWS)
    short_allowed = allow_signal_today(state, ticker, "SHORT", today_str, SHORT_CAP_PER_TICKER_PER_DAY)
    short_triggered = False
    short_setup = ""
    short_entry_price = np.nan
    short_diag: Dict[str, Any] = {"side": "SHORT", "window_ok": short_window_ok, "cap_ok": short_allowed}

    if short_window_ok and short_allowed:
        candidates_i = list(range(max(2, entry_idx - 6), entry_idx))
        for i in candidates_i:
            impulse_type = classify_red_impulse(df_day.iloc[i])
            if impulse_type == "":
                continue
            if not _in_windows(df_day.at[i, "date"], SHORT_SIGNAL_WINDOWS, SHORT_USE_TIME_WINDOWS):
                continue

            common_ok, common_dbg = _check_common_filters_short(df_day, i)
            if not common_ok:
                continue

            # MODERATE: break impulse low on next candle
            if impulse_type == "MODERATE" and i + 1 == entry_idx:
                low1 = _safe_float(df_day.at[i, "low"])
                buf = _buffer(low1)
                trigger = low1 - buf
                low_entry = _safe_float(df_day.at[entry_idx, "low"])
                close_entry = _safe_float(df_day.at[entry_idx, "close"])

                close_confirm_ok = (not REQUIRE_CLOSE_CONFIRM) or (np.isfinite(close_entry) and np.isfinite(trigger) and close_entry < trigger)

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
                    diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger}
                    signals.append(LiveSignal(ticker, "SHORT", entry_ts, short_setup, short_entry_price, sl, tgt, score, diag))
                    break

            # MODERATE: pullback + break C2 low
            if impulse_type == "MODERATE" and i + 2 == entry_idx:
                c2 = df_day.iloc[i + 1]
                c2o, c2c = _safe_float(c2["open"]), _safe_float(c2["close"])
                c2_body = abs(c2c - c2o)
                c2_atr = _safe_float(c2.get("ATR15", df_day.at[i, "ATR15"]))
                c2_av = _safe_float(c2.get("AVWAP", np.nan))

                c2_small_green = (np.isfinite(c2c) and np.isfinite(c2o) and c2c > c2o
                                  and np.isfinite(c2_atr) and c2_atr > 0 and (c2_body <= SHORT_SMALL_GREEN_MAX_ATR * c2_atr))
                c2_below_avwap = np.isfinite(c2_av) and np.isfinite(c2c) and (c2c < c2_av)

                if c2_small_green and c2_below_avwap:
                    low2 = _safe_float(c2["low"])
                    buf = _buffer(low2)
                    trigger = low2 - buf
                    low_entry = _safe_float(df_day.at[entry_idx, "low"])
                    close_entry = _safe_float(df_day.at[entry_idx, "close"])

                    close_confirm_ok = (not REQUIRE_CLOSE_CONFIRM) or (np.isfinite(close_entry) and np.isfinite(trigger) and close_entry < trigger)

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
                        diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger}
                        signals.append(LiveSignal(ticker, "SHORT", entry_ts, short_setup, short_entry_price, sl, tgt, score, diag))
                        break

            # HUGE: failed bounce breakdown
            if impulse_type == "HUGE":
                bounce_end = min(i + 3, entry_idx - 1)
                if bounce_end <= i:
                    continue
                bounce = df_day.iloc[i + 1: bounce_end + 1].copy()
                if bounce.empty:
                    continue

                bounce_atr = pd.to_numeric(bounce.get("ATR15", np.nan), errors="coerce").fillna(_safe_float(df_day.at[i, "ATR15"]))
                bounce_body = (pd.to_numeric(bounce["close"], errors="coerce") - pd.to_numeric(bounce["open"], errors="coerce")).abs()
                bounce_green = pd.to_numeric(bounce["close"], errors="coerce") > pd.to_numeric(bounce["open"], errors="coerce")
                bounce_small = bounce_body <= (SHORT_SMALL_GREEN_MAX_ATR * bounce_atr)

                if not bool((bounce_green & bounce_small).fillna(False).any()):
                    continue

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

                low_entry = _safe_float(df_day.at[entry_idx, "low"])
                close_entry = _safe_float(df_day.at[entry_idx, "close"])
                av_entry = _safe_float(df_day.at[entry_idx, "AVWAP"])

                if np.isfinite(av_entry) and np.isfinite(close_entry) and (close_entry >= av_entry):
                    continue

                close_confirm_ok = (not REQUIRE_CLOSE_CONFIRM) or (np.isfinite(close_entry) and np.isfinite(trigger) and close_entry < trigger)

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
                    diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger}
                    signals.append(LiveSignal(ticker, "SHORT", entry_ts, short_setup, short_entry_price, sl, tgt, score, diag))
                    break

    short_diag.update({"signal": bool(short_triggered), "setup": short_setup,
                       "entry_price": float(short_entry_price) if np.isfinite(short_entry_price) else np.nan})
    checks.append({"ticker": ticker, "side": "SHORT", "bar_time_ist": entry_ts, **short_diag})

    # ---- LONG side ----
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

            # MODERATE: break impulse high on next candle
            if impulse_type == "MODERATE" and i + 1 == entry_idx:
                high1 = _safe_float(df_day.at[i, "high"])
                buf = _buffer(high1)
                trigger = high1 + buf
                high_entry = _safe_float(df_day.at[entry_idx, "high"])
                close_entry = _safe_float(df_day.at[entry_idx, "close"])

                close_confirm_ok = (not REQUIRE_CLOSE_CONFIRM) or (np.isfinite(close_entry) and np.isfinite(trigger) and close_entry > trigger)

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
                    diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger}
                    signals.append(LiveSignal(ticker, "LONG", entry_ts, long_setup, long_entry_price, sl, tgt, score, diag))
                    break

            # MODERATE: small red pullback + break C2 high
            if impulse_type == "MODERATE" and i + 2 == entry_idx:
                c2 = df_day.iloc[i + 1]
                c2o, c2c = _safe_float(c2["open"]), _safe_float(c2["close"])
                c2_body = abs(c2c - c2o)
                c2_atr = _safe_float(c2.get("ATR15", df_day.at[i, "ATR15"]))
                c2_av = _safe_float(c2.get("AVWAP", np.nan))

                c2_small_red = (np.isfinite(c2c) and np.isfinite(c2o) and c2c < c2o
                                and np.isfinite(c2_atr) and c2_atr > 0 and (c2_body <= LONG_SMALL_RED_MAX_ATR * c2_atr))
                c2_above_avwap = np.isfinite(c2_av) and np.isfinite(c2c) and (c2c > c2_av)

                if c2_small_red and c2_above_avwap:
                    high2 = _safe_float(c2["high"])
                    buf = _buffer(high2)
                    trigger = high2 + buf
                    high_entry = _safe_float(df_day.at[entry_idx, "high"])
                    close_entry = _safe_float(df_day.at[entry_idx, "close"])

                    close_confirm_ok = (not REQUIRE_CLOSE_CONFIRM) or (np.isfinite(close_entry) and np.isfinite(trigger) and close_entry > trigger)

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
                        diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger}
                        signals.append(LiveSignal(ticker, "LONG", entry_ts, long_setup, long_entry_price, sl, tgt, score, diag))
                        break

            # HUGE: failed retrace breakout
            if impulse_type == "HUGE":
                bounce_end = min(i + 3, entry_idx - 1)
                if bounce_end <= i:
                    continue
                bounce = df_day.iloc[i + 1: bounce_end + 1].copy()
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
                fail_avwap = bool((closes > avwaps).fillna(False).all())
                lows = pd.to_numeric(bounce["low"], errors="coerce")
                fail_mid = bool((lows > mid_body).fillna(False).all())

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

                close_confirm_ok = (not REQUIRE_CLOSE_CONFIRM) or (np.isfinite(close_entry) and np.isfinite(trigger) and close_entry > trigger)

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
                    diag = {"impulse_idx": i, "impulse_type": impulse_type, **common_dbg, **rej_dbg, "trigger": trigger}
                    signals.append(LiveSignal(ticker, "LONG", entry_ts, long_setup, long_entry_price, sl, tgt, score, diag))
                    break

    long_diag.update({"signal": bool(long_triggered), "setup": long_setup,
                      "entry_price": float(long_entry_price) if np.isfinite(long_entry_price) else np.nan})
    checks.append({"ticker": ticker, "side": "LONG", "bar_time_ist": entry_ts, **long_diag})

    for s in signals:
        mark_signal(state, s.ticker, s.side, today_str)

    return signals, checks


# =============================================================================
# TRADING DAY HELPERS
# =============================================================================
def _read_holidays_safe() -> set:
    try:
        return set(core._read_holidays(core.HOLIDAYS_FILE_DEFAULT))
    except Exception:
        return set()


def is_trading_day_safe(d, holidays: set) -> bool:
    fn = getattr(core, "_is_trading_day", None)
    if fn is None:
        return d.weekday() < 5 and (d not in holidays)
    try:
        return bool(fn(d, holidays))
    except Exception:
        return d.weekday() < 5 and (d not in holidays)


# =============================================================================
# OPTIONAL UPDATER
# =============================================================================
def run_update_15m_once(holidays: set) -> None:
    core.run_mode(
        mode="15min",
        max_workers=4,
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
    now = now.astimezone(IST)
    today = now.date()
    start_dt = IST.localize(datetime.combine(today, START_TIME))
    end_dt = IST.localize(datetime.combine(today, END_TIME))

    if now <= start_dt:
        return start_dt

    if now > end_dt:
        tomorrow = today + timedelta(days=1)
        return IST.localize(datetime.combine(tomorrow, START_TIME))

    minute = (now.minute // 15) * 15
    slot = now.replace(minute=minute, second=0, microsecond=0)
    if slot < now:
        slot += timedelta(minutes=15)

    if slot < start_dt:
        slot = start_dt
    if slot > end_dt:
        tomorrow = today + timedelta(days=1)
        slot = IST.localize(datetime.combine(tomorrow, START_TIME))
    return slot


def _sleep_until(dt: datetime) -> None:
    now = now_ist()
    delta = (dt - now).total_seconds()
    if delta > 0:
        time.sleep(delta)


# =============================================================================
# RUN ONE SCAN
# =============================================================================
def run_one_scan(run_tag: str = "A") -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                all_checks.append({"ticker": t, "side": "SHORT", "bar_time_ist": pd.NaT, "no_data": True})
                all_checks.append({"ticker": t, "side": "LONG", "bar_time_ist": pd.NaT, "no_data": True})
                continue

            signals, checks_rows = _latest_entry_signals_for_ticker(t, df_day, state)
            all_checks.extend(checks_rows)

            for s in signals:
                all_signals.append({
                    "ticker": s.ticker, "side": s.side, "bar_time_ist": s.bar_time_ist,
                    "setup": s.setup, "entry_price": s.entry_price,
                    "sl_price": s.sl_price, "target_price": s.target_price,
                    "score": s.score,
                    "diagnostics_json": json.dumps(s.diagnostics, default=str),
                })
        except Exception as e:
            all_checks.append({"ticker": t, "side": "SHORT", "bar_time_ist": pd.NaT, "error": str(e)})
            all_checks.append({"ticker": t, "side": "LONG", "bar_time_ist": pd.NaT, "error": str(e)})

        if idx % 100 == 0:
            print(f"  scanned {idx}/{len(tickers)} | signals_so_far={len(all_signals)}")

    _save_state(state)

    checks_df = pd.DataFrame(all_checks)
    signals_df = pd.DataFrame(all_signals)

    if USE_TOPN_PER_RUN and (not signals_df.empty):
        keep = []
        for side in ["SHORT", "LONG"]:
            df_side = signals_df[signals_df["side"] == side].copy()
            df_side = df_side.sort_values(["score"], ascending=False).head(int(TOPN_PER_RUN))
            keep.append(df_side)
        signals_df = pd.concat(keep, ignore_index=True) if keep else signals_df

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
    print("[LIVE] EQIDV1 AVWAP v11 combined LIVE scanner (15m)")
    print(f"[INFO] DIR_15M={DIR_15M} | tickers={len(list_tickers_15m())}")
    print(f"[INFO] SHORT: SL={SHORT_STOP_PCT*100:.2f}%, TGT={SHORT_TARGET_PCT*100:.2f}%, ADX>={SHORT_ADX_MIN}, RSI<={SHORT_RSI_MAX}, StochK<={SHORT_STOCHK_MAX}")
    print(f"[INFO] LONG : SL={LONG_STOP_PCT*100:.2f}%, TGT={LONG_TARGET_PCT*100:.2f}%, ADX>={LONG_ADX_MIN}, RSI>={LONG_RSI_MIN}, StochK>={LONG_STOCHK_MIN}")
    print(f"[INFO] Volume filter: {USE_VOLUME_FILTER} (ratio>={VOLUME_MIN_RATIO}, SMA={VOLUME_SMA_PERIOD})")
    print(f"[INFO] ATR% filter: {USE_ATR_PCT_FILTER} (min={ATR_PCT_MIN*100:.2f}%)")
    print(f"[INFO] Close-confirm: {REQUIRE_CLOSE_CONFIRM}")
    print(f"[INFO] UPDATE_15M_BEFORE_CHECK={UPDATE_15M_BEFORE_CHECK} | ENABLE_SECOND_RUN={ENABLE_SECOND_RUN} (gap={SECOND_RUN_GAP_SECONDS}s)")

    holidays = _read_holidays_safe()

    while True:
        now = now_ist()
        if now.time() >= HARD_STOP_TIME:
            print("[STOP] Hard-stop reached for today. Exiting.")
            return

        if not is_trading_day_safe(now.date(), holidays):
            nxt = _next_slot_after(now + timedelta(days=1))
            print(f"[SKIP] Not a trading day ({now.date()}). Sleeping until {nxt}.")
            _sleep_until(nxt)
            continue

        slot = _next_slot_after(now)
        if slot.date() != now.date():
            print(f"[WAIT] Next slot is tomorrow {slot}. Sleeping.")
            _sleep_until(slot)
            continue

        if now < slot:
            print(f"[WAIT] Sleeping until slot {slot.strftime('%Y-%m-%d %H:%M:%S%z')}")
            _sleep_until(slot)

        now = now_ist()
        if now.time() > END_TIME:
            nxt = _next_slot_after(now + timedelta(days=1))
            print(f"[DONE] Past END_TIME. Sleeping until {nxt}.")
            _sleep_until(nxt)
            continue

        if UPDATE_15M_BEFORE_CHECK:
            try:
                print(f"[UPD ] Running eqidv1 core 15m update at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
                run_update_15m_once(holidays)
            except Exception as e:
                print(f"[WARN] Update failed: {e!r}")

        print(f"[RUN ] Slot {slot.strftime('%H:%M')} | scan A")
        run_one_scan(run_tag="A")

        if ENABLE_SECOND_RUN:
            time.sleep(float(SECOND_RUN_GAP_SECONDS))
            print(f"[RUN ] Slot {slot.strftime('%H:%M')} | scan B")
            run_one_scan(run_tag="B")

        time.sleep(1.0)


if __name__ == "__main__":
    main()
