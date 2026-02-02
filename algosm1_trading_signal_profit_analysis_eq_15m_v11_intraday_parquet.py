# -*- coding: utf-8 -*-
"""
15m SHORT STRATEGY (Anchored VWAP) — Scan ALL DAYS present in data (row-by-row)

CURRENT VERSION: A + B + E + (Quality upgrades to reduce trades / improve hit-rate)
---------------------------------------------------------------------------------
Option A (Stricter trend filter)
Option E (Time-of-day filter)
Option B (AVWAP “rejection” rule + distance from AVWAP)

NEW QUALITY UPGRADES (defaults tuned for *fewer, higher-quality* trades):
Q1) Per-ticker per-day cap:
    - MAX_TRADES_PER_TICKER_PER_DAY (default 1)

Q2) "Close-confirmed" breakdown entries (reduces wick-fakeouts / SLs):
    - REQUIRE_ENTRY_CLOSE_CONFIRM = True
    - Entry candle must CLOSE below trigger level (not just make a low)

Q3) Avoid late entries that tend to become EOD exits:
    - MIN_BARS_LEFT_AFTER_ENTRY (default 4 x 15m = 1 hour)

Q4) Breakeven protection (reduces full SL hits after partial move):
    - ENABLE_BREAKEVEN = True
    - When trade moves in favor by BE_TRIGGER_PCT, move SL to entry (+BE_PAD_PCT)

Q5) Optional post-filter: take only Top-N trades per day by a quality score
    (helps reduce total trades/day when scanning 1000+ tickers):
    - ENABLE_TOPN_PER_DAY = True
    - TOPN_PER_DAY = 30

Outputs:
- reports/avwap_short_trades_ALL_DAYS_<timestamp>.csv  (final, post-filtered if enabled)

Notes:
- This is a scanner/backtest style script (not live execution).
- Conservative fill assumption is kept: if a candle hits both SL and Target, SL is assumed first.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz

# Optional: if you still want to run your 15m updater once before scan
from kiteconnect import KiteConnect
import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly as core


# =============================================================================
# CONFIG
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

# 15m parquet dir
DIR_15M = "stocks_indicators_15min_eq"
END_15M = "_stocks_indicators_15min.parquet"

# Optional updater
UPDATE_15M_FIRST = False  # set True if you want update once before scan

# -------------------------
# STRICT risk rules
# -------------------------
STOP_PCT = 0.0100
# Smaller target generally increases target-hit rate on intraday 15m shorts.
TARGET_PCT = 0.01

# -------------------------
# Impulse definitions
# -------------------------
MOD_RED_MIN_ATR = 0.45
MOD_RED_MAX_ATR = 1.00
HUGE_RED_MIN_ATR = 1.60
HUGE_RED_MIN_RANGE_ATR = 2.00
CLOSE_NEAR_LOW_MAX = 0.25

# Pullback + Bounce candle size
SMALL_GREEN_MAX_ATR = 0.20

# Entry buffer
BUFFER_ABS = 0.05          # absolute
BUFFER_PCT = 0.0002        # 0.02%

# Session bounds (data filter)
SESSION_START = dtime(9, 15, 0)
SESSION_END   = dtime(14, 30, 0)

# -------------------------------------------------------------------------
# OPTION E: Time-of-day filter (ONLY trade in these windows)
# Default tightened to avoid very open + late noise.
# -------------------------------------------------------------------------
USE_TIME_WINDOWS = True
SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 30, 0)),
    (dtime(13, 00, 0), dtime(14, 30, 0)),
]

# -------------------------------------------------------------------------
# OPTION A: Stricter trend filter
# -------------------------------------------------------------------------
ADX_MIN = 25.0
ADX_SLOPE_MIN = 1.25  # require ADX[i] - ADX[i-2] >= this
RSI_MAX_SHORT = 55.0
STOCHK_MAX = 75.0

# Extra (optional) volatility sanity: skip very low ATR% names (reduces chop)
USE_ATR_PCT_FILTER = True
ATR_PCT_MIN = 0.0020  # ATR/close must be >= 0.20% (tune 0.15%–0.35%)

# -------------------------------------------------------------------------
# OPTION B: AVWAP rejection + distance rules
# -------------------------------------------------------------------------
REQUIRE_AVWAP_REJECTION = True

# Rejection evidence:
#   - touch rule: any candle (between impulse and entry) has high >= AVWAP and close < AVWAP
AVWAP_REJECT_TOUCH = True

#   - consecutive closes rule: need N consecutive closes below AVWAP between impulse and entry
AVWAP_REJECT_MIN_CONSEC_CLOSES = 2  # 2->3 reduces random breakdowns

# How to combine touch + consec evidence: "any" or "both"
AVWAP_REJECT_MODE = "any"  # set "both" for very strict filtering

# Distance from AVWAP at entry (avoid mid-zone chop)
AVWAP_DIST_ATR_MULT = 0.25  # 0.25->0.35 cuts mid-zone trades

# -------------------------------------------------------------------------
# QUALITY UPGRADES
# -------------------------------------------------------------------------
MAX_TRADES_PER_TICKER_PER_DAY = 1

REQUIRE_ENTRY_CLOSE_CONFIRM = True
# if True: entry candle CLOSE must be below trigger (not only LOW)
# (reduces SLs from wick-breakouts)

MIN_BARS_LEFT_AFTER_ENTRY = 4  # 4*15m=60 minutes left in session after entry

ENABLE_BREAKEVEN = True
BE_TRIGGER_PCT = 0.0040  # once price moves ~0.4% in favor, protect
BE_PAD_PCT = 0.0001      # SL moved to entry*(1+pad) for shorts

# Post-filter to reduce daily trade count across 1000+ tickers:
ENABLE_TOPN_PER_DAY = True
TOPN_PER_DAY = 30

SAVE_RAW_BEFORE_TOPN = False  # if True, also writes a *_RAW.csv with all trades before top-N

PARQUET_ENGINE = "pyarrow"

# Paths relative to this file
ROOT = Path(__file__).resolve().parent
API_KEY_FILE = ROOT / "api_key.txt"
ACCESS_TOKEN_FILE = ROOT / "access_token.txt"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# AUTH FIX (only used if UPDATE_15M_FIRST=True)
# =============================================================================
def read_first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise ValueError(f"{path} is empty")


def setup_kite_session_fixed() -> KiteConnect:
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE).split()[0]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


core.setup_kite_session = setup_kite_session_fixed


def now_ist() -> datetime:
    return datetime.now(IST)


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
# IO HELPERS
# =============================================================================
def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError("Parquet support requires 'pyarrow' (pip install pyarrow).") from e


def read_15m_parquet(path: str) -> pd.DataFrame:
    _require_pyarrow()
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    if "date" not in df.columns:
        return pd.DataFrame()

    dt = pd.to_datetime(df["date"], errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    dt = dt.dt.tz_convert(IST)

    df = df.copy()
    df["date"] = dt
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def list_tickers_15m() -> list[str]:
    pattern = os.path.join(DIR_15M, f"*{END_15M}")
    files = glob.glob(pattern)
    out = []
    for f in files:
        base = os.path.basename(f)
        if base.endswith(END_15M):
            out.append(base[:-len(END_15M)].upper())
    return sorted(set(out))


def in_session(ts: pd.Timestamp) -> bool:
    t = ts.tz_convert(IST).time()
    return (t >= SESSION_START) and (t <= SESSION_END)


def in_signal_window(ts: pd.Timestamp) -> bool:
    if not USE_TIME_WINDOWS:
        return True
    t = ts.tz_convert(IST).time()
    for a, b in SIGNAL_WINDOWS:
        if (t >= a) and (t <= b):
            return True
    return False


def _buffer(price: float) -> float:
    return max(float(BUFFER_ABS), float(price) * float(BUFFER_PCT))


# =============================================================================
# INDICATORS
# =============================================================================
def ensure_ema(df: pd.DataFrame, span: int, col_close: str = "close") -> pd.Series:
    close = pd.to_numeric(df[col_close], errors="coerce")
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


def compute_rsi14(df: pd.DataFrame, col_close: str = "close") -> pd.Series:
    close = pd.to_numeric(df[col_close], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stoch_14_3(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return adx


def compute_day_avwap(df_day: pd.DataFrame) -> pd.Series:
    """Anchored VWAP for a single day (anchored at day start candle)."""
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
# STRATEGY
# =============================================================================
@dataclass
class Trade:
    trade_date: str
    ticker: str
    setup: str
    impulse_type: str
    signal_time_ist: pd.Timestamp
    entry_time_ist: pd.Timestamp
    entry_price: float
    sl_price: float
    target_price: float
    exit_time_ist: pd.Timestamp
    exit_price: float
    outcome: str  # TARGET / SL / BE / EOD
    pnl_pct: float

    # diagnostics / scoring
    adx_signal: float
    rsi_signal: float
    stochk_signal: float
    avwap_dist_atr_signal: float
    ema20_gap_atr_signal: float
    atr_pct_signal: float
    quality_score: float


def classify_red_impulse(row: pd.Series) -> str:
    o = float(row["open"])
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    atr = float(row["ATR15"])

    if not np.isfinite(atr) or atr <= 0:
        return ""
    if not (c < o):
        return ""

    body = abs(c - o)
    rng = (h - l) if (h >= l) else np.nan
    if not np.isfinite(rng) or rng <= 0:
        return ""

    close_near_low = ((c - l) / rng) <= CLOSE_NEAR_LOW_MAX

    if (body >= HUGE_RED_MIN_ATR * atr) or (rng >= HUGE_RED_MIN_RANGE_ATR * atr):
        return "HUGE"

    if (body >= MOD_RED_MIN_ATR * atr) and (body <= MOD_RED_MAX_ATR * atr) and close_near_low:
        return "MODERATE"

    return ""


def _twice_increasing(df_day: pd.DataFrame, idx: int, col: str) -> bool:
    if idx < 2:
        return False
    a = float(df_day.at[idx, col]) if col in df_day.columns else np.nan
    b = float(df_day.at[idx - 1, col]) if col in df_day.columns else np.nan
    c = float(df_day.at[idx - 2, col]) if col in df_day.columns else np.nan
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a > b > c)


def _twice_reducing(df_day: pd.DataFrame, idx: int, col: str) -> bool:
    if idx < 2:
        return False
    a = float(df_day.at[idx, col]) if col in df_day.columns else np.nan
    b = float(df_day.at[idx - 1, col]) if col in df_day.columns else np.nan
    c = float(df_day.at[idx - 2, col]) if col in df_day.columns else np.nan
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a < b < c)


def _adx_slope_ok(df_day: pd.DataFrame, idx: int, col: str, min_slope: float) -> bool:
    if idx < 2 or col not in df_day.columns:
        return False
    a = float(df_day.at[idx, col])
    c = float(df_day.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(c) and ((a - c) >= float(min_slope))


# =============================================================================
# OPTION B helpers
# =============================================================================
def _max_consecutive_true(flags: np.ndarray) -> int:
    best = 0
    run = 0
    for x in flags:
        if bool(x):
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def _avwap_rejection_pass(df_day: pd.DataFrame, impulse_idx: int, entry_idx: int) -> bool:
    """
    Require AVWAP rejection evidence between impulse and entry.

    Evidence:
      - TOUCH: any candle has high >= AVWAP and close < AVWAP
      - CONSEC: at least N consecutive closes below AVWAP

    Combine with AVWAP_REJECT_MODE:
      - "any":  touch OR consec
      - "both": touch AND consec
    """
    if not REQUIRE_AVWAP_REJECTION:
        return True
    if "AVWAP" not in df_day.columns:
        return False

    i0 = max(0, int(impulse_idx))
    e0 = max(0, int(entry_idx))
    if e0 <= i0:
        return False

    win_a = df_day.iloc[i0 + 1 : e0 + 1]  # bounce/transition
    if win_a.empty:
        return False

    touch_ok = False
    if AVWAP_REJECT_TOUCH:
        hi = pd.to_numeric(win_a["high"], errors="coerce")
        cl = pd.to_numeric(win_a["close"], errors="coerce")
        av = pd.to_numeric(win_a["AVWAP"], errors="coerce")
        touch_ok = bool(((hi >= av) & (cl < av)).fillna(False).any())

    consec_ok = False
    n = int(AVWAP_REJECT_MIN_CONSEC_CLOSES)
    if n > 0:
        win_b = df_day.iloc[i0 : e0 + 1]  # include impulse candle
        clb = pd.to_numeric(win_b["close"], errors="coerce")
        avb = pd.to_numeric(win_b["AVWAP"], errors="coerce")
        below = ((clb < avb) & np.isfinite(clb) & np.isfinite(avb)).to_numpy(dtype=bool)
        consec_ok = _max_consecutive_true(below) >= n

    mode = str(AVWAP_REJECT_MODE).strip().lower()
    if mode == "both":
        return bool(touch_ok and consec_ok)
    return bool(touch_ok or consec_ok)


def _avwap_distance_pass(df_day: pd.DataFrame, idx: int) -> bool:
    """
    Require distance from AVWAP at entry candle:
      (AVWAP - close) >= AVWAP_DIST_ATR_MULT * ATR15
    """
    mult = float(AVWAP_DIST_ATR_MULT or 0.0)
    if mult <= 0:
        return True

    if "AVWAP" not in df_day.columns or "ATR15" not in df_day.columns or "close" not in df_day.columns:
        return False

    av = float(df_day.at[idx, "AVWAP"]) if np.isfinite(df_day.at[idx, "AVWAP"]) else np.nan
    cl = float(df_day.at[idx, "close"]) if np.isfinite(df_day.at[idx, "close"]) else np.nan
    atr = float(df_day.at[idx, "ATR15"]) if np.isfinite(df_day.at[idx, "ATR15"]) else np.nan

    if not (np.isfinite(av) and np.isfinite(cl) and np.isfinite(atr) and atr > 0):
        return False

    return (av - cl) >= (mult * atr)


# =============================================================================
# Execution simulation (with optional breakeven)
# =============================================================================
def simulate_exit_short_within_day(df_day: pd.DataFrame, entry_idx: int, entry_price: float) -> tuple[int, pd.Timestamp, float, str]:
    """
    Walk forward within the same day until TARGET / SL / BE / EOD.

    Conservative:
      - If a candle hits both SL and Target, assume SL first.
    """
    sl0 = entry_price * (1.0 + STOP_PCT)
    tgt = entry_price * (1.0 - TARGET_PCT)

    sl = sl0
    be_armed = False
    be_trigger = entry_price * (1.0 - BE_TRIGGER_PCT)
    be_sl = entry_price * (1.0 + BE_PAD_PCT)

    for k in range(entry_idx + 1, len(df_day)):
        hi = float(df_day.at[k, "high"])
        lo = float(df_day.at[k, "low"])
        ts = df_day.at[k, "date"]

        # Arm breakeven if moved in favor enough (for short: price goes DOWN)
        if ENABLE_BREAKEVEN and (not be_armed) and np.isfinite(lo) and (lo <= be_trigger):
            be_armed = True
            sl = min(sl, be_sl)  # move stop down toward entry

        hit_sl = np.isfinite(hi) and (hi >= sl)
        hit_tg = np.isfinite(lo) and (lo <= tgt)

        if hit_sl and hit_tg:
            outcome = "BE" if (be_armed and abs(sl - be_sl) <= 1e-9) else "SL"
            return k, ts, float(sl), outcome
        if hit_sl:
            outcome = "BE" if (be_armed and abs(sl - be_sl) <= 1e-9) else "SL"
            return k, ts, float(sl), outcome
        if hit_tg:
            return k, ts, float(tgt), "TARGET"

    last = len(df_day) - 1
    ts = df_day.at[last, "date"]
    px = float(df_day.at[last, "close"])
    return last, ts, px, "EOD"


# =============================================================================
# Quality score (used for TOPN_PER_DAY)
# =============================================================================
def compute_quality_score(adx: float, avwap_dist_atr: float, ema_gap_atr: float, impulse: str) -> float:
    """
    Simple bounded score.
    Higher is "better".
    """
    adx_n = np.clip((adx - 20.0) / 30.0, 0.0, 1.0)  # 20->50 maps to 0->1
    av_n = np.clip(avwap_dist_atr / 2.0, 0.0, 1.0)  # 0->2 ATR maps to 0->1
    ema_n = np.clip(ema_gap_atr / 2.0, 0.0, 1.0)

    imp = 1.0 if impulse == "HUGE" else 0.6

    score = (0.45 * adx_n) + (0.35 * av_n) + (0.10 * ema_n) + (0.10 * imp)
    return float(score)


# =============================================================================
# SCAN ONE DAY
# =============================================================================
def scan_one_day_row_by_row(ticker: str, df_day: pd.DataFrame, day_str: str) -> list[Trade]:
    if len(df_day) < 7:
        return []

    trades: list[Trade] = []
    i = 2

    while i < len(df_day) - 3:
        if (MAX_TRADES_PER_TICKER_PER_DAY is not None) and (len(trades) >= int(MAX_TRADES_PER_TICKER_PER_DAY)):
            break

        c1 = df_day.iloc[i]
        ts1 = c1["date"]

        # OPTION E
        if not in_signal_window(ts1):
            i += 1
            continue

        impulse = classify_red_impulse(c1)
        if impulse == "":
            i += 1
            continue

        # Indicators at signal
        adx1 = float(df_day.at[i, "ADX15"]) if "ADX15" in df_day.columns else np.nan
        rsi1 = float(df_day.at[i, "RSI15"]) if "RSI15" in df_day.columns else np.nan
        k1   = float(df_day.at[i, "STOCHK15"]) if "STOCHK15" in df_day.columns else np.nan
        d1   = float(df_day.at[i, "STOCHD15"]) if "STOCHD15" in df_day.columns else np.nan

        # ATR sanity
        atr1 = float(c1["ATR15"]) if np.isfinite(c1.get("ATR15", np.nan)) else np.nan
        close1 = float(c1["close"])
        if not (np.isfinite(atr1) and atr1 > 0 and np.isfinite(close1) and close1 > 0):
            i += 1
            continue

        atr_pct = atr1 / close1
        if USE_ATR_PCT_FILTER and (atr_pct < float(ATR_PCT_MIN)):
            i += 1
            continue

        # OPTION A indicators
        adx_ok = (
            np.isfinite(adx1)
            and (adx1 >= ADX_MIN)
            and _twice_increasing(df_day, i, "ADX15")
            and _adx_slope_ok(df_day, i, "ADX15", ADX_SLOPE_MIN)
        )
        rsi_ok = np.isfinite(rsi1) and (rsi1 <= RSI_MAX_SHORT) and _twice_reducing(df_day, i, "RSI15")
        stoch_ok = (
            np.isfinite(k1) and np.isfinite(d1)
            and (k1 <= STOCHK_MAX)
            and (k1 < d1)
            and _twice_reducing(df_day, i, "STOCHK15")
        )
        if not (adx_ok and rsi_ok and stoch_ok):
            i += 1
            continue

        # Strict EMA + AVWAP
        avwap1 = float(c1["AVWAP"]) if np.isfinite(c1.get("AVWAP", np.nan)) else np.nan
        ema20_1 = float(c1["EMA20"]) if np.isfinite(c1.get("EMA20", np.nan)) else np.nan
        ema50_1 = float(c1["EMA50"]) if np.isfinite(c1.get("EMA50", np.nan)) else np.nan
        if not (np.isfinite(avwap1) and np.isfinite(ema20_1) and np.isfinite(ema50_1)):
            i += 1
            continue
        strict_ema_ok = (ema20_1 < ema50_1) and (close1 < ema20_1)
        strict_avwap_ok = close1 < avwap1
        if not (strict_ema_ok and strict_avwap_ok):
            i += 1
            continue

        # Diagnostic distances at signal
        avwap_dist_atr_signal = (avwap1 - close1) / atr1
        ema_gap_atr_signal = (ema20_1 - close1) / atr1

        quality_score = compute_quality_score(
            adx=adx1,
            avwap_dist_atr=avwap_dist_atr_signal,
            ema_gap_atr=ema_gap_atr_signal,
            impulse=impulse
        )

        signal_time = ts1
        low1 = float(c1["low"])
        open1 = float(c1["open"])
        close1 = float(c1["close"])

        def _bars_left_ok(entry_idx: int) -> bool:
            left = (len(df_day) - 1) - int(entry_idx)
            return left >= int(MIN_BARS_LEFT_AFTER_ENTRY)

        def _entry_close_confirm_ok(entry_idx: int, trigger_level: float) -> bool:
            if not REQUIRE_ENTRY_CLOSE_CONFIRM:
                return True
            cl = float(df_day.at[entry_idx, "close"])
            return np.isfinite(cl) and (cl < float(trigger_level))

        # -------------------------
        # SETUP A (MODERATE)
        # -------------------------
        if impulse == "MODERATE":
            c2 = df_day.iloc[i + 1]
            buf1 = _buffer(low1)
            trigger1 = float(low1 - buf1)

            # Option 1: break C1 low
            if float(c2["low"]) < trigger1:
                entry_idx = i + 1
                entry_ts = df_day.at[entry_idx, "date"]

                if not in_signal_window(entry_ts):
                    i += 1
                    continue
                if not _bars_left_ok(entry_idx):
                    i += 1
                    continue
                if not _entry_close_confirm_ok(entry_idx, trigger1):
                    i += 1
                    continue

                if not _avwap_rejection_pass(df_day, i, entry_idx):
                    i += 1
                    continue
                if not _avwap_distance_pass(df_day, entry_idx):
                    i += 1
                    continue

                entry_price = trigger1

                exit_idx, exit_time, exit_price, outcome = simulate_exit_short_within_day(df_day, entry_idx, entry_price)
                pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                trades.append(Trade(
                    trade_date=day_str,
                    ticker=ticker,
                    setup="A_MOD_BREAK_C1_LOW",
                    impulse_type=impulse,
                    signal_time_ist=signal_time,
                    entry_time_ist=entry_ts,
                    entry_price=entry_price,
                    sl_price=entry_price * (1.0 + STOP_PCT),
                    target_price=entry_price * (1.0 - TARGET_PCT),
                    exit_time_ist=exit_time,
                    exit_price=exit_price,
                    outcome=outcome,
                    pnl_pct=float(pnl_pct),
                    adx_signal=float(adx1),
                    rsi_signal=float(rsi1),
                    stochk_signal=float(k1),
                    avwap_dist_atr_signal=float(avwap_dist_atr_signal),
                    ema20_gap_atr_signal=float(ema_gap_atr_signal),
                    atr_pct_signal=float(atr_pct),
                    quality_score=float(quality_score),
                ))
                i = exit_idx + 1
                continue

            # Option 2: small green pullback C2 + below AVWAP, then break C2 low on C3
            c2o, c2c = float(c2["open"]), float(c2["close"])
            c2_body = abs(c2c - c2o)
            c2_atr = float(c2.get("ATR15", atr1)) if np.isfinite(c2.get("ATR15", atr1)) else atr1
            c2_avwap = float(c2.get("AVWAP", np.nan)) if np.isfinite(c2.get("AVWAP", np.nan)) else np.nan

            c2_small_green = (c2c > c2o) and np.isfinite(c2_atr) and (c2_body <= SMALL_GREEN_MAX_ATR * c2_atr)
            c2_below_avwap = np.isfinite(c2_avwap) and (c2c < c2_avwap)

            if c2_small_green and c2_below_avwap:
                c3 = df_day.iloc[i + 2]
                low2 = float(c2["low"])
                buf2 = _buffer(low2)
                trigger2 = float(low2 - buf2)

                if float(c3["low"]) < trigger2:
                    entry_idx = i + 2
                    entry_ts = df_day.at[entry_idx, "date"]

                    if not in_signal_window(entry_ts):
                        i += 1
                        continue
                    if not _bars_left_ok(entry_idx):
                        i += 1
                        continue
                    if not _entry_close_confirm_ok(entry_idx, trigger2):
                        i += 1
                        continue

                    if not _avwap_rejection_pass(df_day, i, entry_idx):
                        i += 1
                        continue
                    if not _avwap_distance_pass(df_day, entry_idx):
                        i += 1
                        continue

                    entry_price = trigger2

                    exit_idx, exit_time, exit_price, outcome = simulate_exit_short_within_day(df_day, entry_idx, entry_price)
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                    trades.append(Trade(
                        trade_date=day_str,
                        ticker=ticker,
                        setup="A_PULLBACK_C2_THEN_BREAK_C2_LOW",
                        impulse_type=impulse,
                        signal_time_ist=signal_time,
                        entry_time_ist=entry_ts,
                        entry_price=entry_price,
                        sl_price=entry_price * (1.0 + STOP_PCT),
                        target_price=entry_price * (1.0 - TARGET_PCT),
                        exit_time_ist=exit_time,
                        exit_price=exit_price,
                        outcome=outcome,
                        pnl_pct=float(pnl_pct),
                        adx_signal=float(adx1),
                        rsi_signal=float(rsi1),
                        stochk_signal=float(k1),
                        avwap_dist_atr_signal=float(avwap_dist_atr_signal),
                        ema20_gap_atr_signal=float(ema_gap_atr_signal),
                        atr_pct_signal=float(atr_pct),
                        quality_score=float(quality_score),
                    ))
                    i = exit_idx + 1
                    continue

            i += 1
            continue

        # -------------------------
        # SETUP B (HUGE)
        # -------------------------
        if impulse == "HUGE":
            bounce_end = min(i + 3, len(df_day) - 1)
            bounce = df_day.iloc[i + 1 : bounce_end + 1].copy()
            if bounce.empty:
                i += 1
                continue

            closes = pd.to_numeric(bounce["close"], errors="coerce")
            avwaps = pd.to_numeric(bounce["AVWAP"], errors="coerce")
            highs  = pd.to_numeric(bounce["high"], errors="coerce")

            touch_fail = bool(((highs >= avwaps) & (closes < avwaps)).fillna(False).any()) if AVWAP_REJECT_TOUCH else True
            if REQUIRE_AVWAP_REJECTION and (not touch_fail):
                i += 1
                continue

            bounce_atr = pd.to_numeric(bounce.get("ATR15", atr1), errors="coerce").fillna(atr1)
            bounce_body = (pd.to_numeric(bounce["close"], errors="coerce") - pd.to_numeric(bounce["open"], errors="coerce")).abs()
            bounce_green = pd.to_numeric(bounce["close"], errors="coerce") > pd.to_numeric(bounce["open"], errors="coerce")
            bounce_small = bounce_body <= (SMALL_GREEN_MAX_ATR * bounce_atr)
            if not bool((bounce_green & bounce_small).any()):
                i += 1
                continue

            bounce_low = float(pd.to_numeric(bounce["low"], errors="coerce").min())
            if not np.isfinite(bounce_low):
                i += 1
                continue

            start_j = bounce_end + 1
            entered = False
            buf = _buffer(bounce_low)
            trigger_b = float(bounce_low - buf)

            for j in range(start_j, len(df_day)):
                tsj = df_day.at[j, "date"]
                if not in_signal_window(tsj):
                    continue
                if not _bars_left_ok(j):
                    continue

                closej = float(df_day.at[j, "close"])
                avwapj = float(df_day.at[j, "AVWAP"]) if np.isfinite(df_day.at[j, "AVWAP"]) else np.nan
                if np.isfinite(avwapj) and (closej >= avwapj):
                    break

                if float(df_day.at[j, "low"]) < trigger_b:
                    if not _entry_close_confirm_ok(j, trigger_b):
                        continue
                    if not _avwap_distance_pass(df_day, j):
                        continue
                    if not _avwap_rejection_pass(df_day, i, j):
                        continue

                    entry_idx = j
                    entry_price = trigger_b

                    exit_idx, exit_time, exit_price, outcome = simulate_exit_short_within_day(df_day, entry_idx, entry_price)
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                    trades.append(Trade(
                        trade_date=day_str,
                        ticker=ticker,
                        setup="B_HUGE_RED_FAILED_BOUNCE",
                        impulse_type=impulse,
                        signal_time_ist=signal_time,
                        entry_time_ist=tsj,
                        entry_price=entry_price,
                        sl_price=entry_price * (1.0 + STOP_PCT),
                        target_price=entry_price * (1.0 - TARGET_PCT),
                        exit_time_ist=exit_time,
                        exit_price=exit_price,
                        outcome=outcome,
                        pnl_pct=float(pnl_pct),
                        adx_signal=float(adx1),
                        rsi_signal=float(rsi1),
                        stochk_signal=float(k1),
                        avwap_dist_atr_signal=float(avwap_dist_atr_signal),
                        ema20_gap_atr_signal=float(ema_gap_atr_signal),
                        atr_pct_signal=float(atr_pct),
                        quality_score=float(quality_score),
                    ))
                    i = exit_idx + 1
                    entered = True
                    break

            if entered:
                continue

            i += 1
            continue

        i += 1

    return trades


# =============================================================================
# SCAN ALL DAYS FOR TICKER
# =============================================================================
def scan_all_days_for_ticker(ticker: str, df_full: pd.DataFrame) -> list[Trade]:
    if df_full.empty:
        return []

    df = df_full.copy()
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            return []

    df = df[df["date"].apply(in_session)].copy()
    if df.empty:
        return []

    df = df.sort_values("date").reset_index(drop=True)

    if "ATR" in df.columns:
        df["ATR15"] = pd.to_numeric(df["ATR"], errors="coerce")
    else:
        df["ATR15"] = compute_atr14(df)

    if "EMA_20" in df.columns:
        df["EMA20"] = pd.to_numeric(df["EMA_20"], errors="coerce")
    else:
        df["EMA20"] = ensure_ema(df, 20)

    if "EMA_50" in df.columns:
        df["EMA50"] = pd.to_numeric(df["EMA_50"], errors="coerce")
    else:
        df["EMA50"] = ensure_ema(df, 50)

    if "RSI" in df.columns:
        df["RSI15"] = pd.to_numeric(df["RSI"], errors="coerce")
    else:
        df["RSI15"] = compute_rsi14(df)

    if "Stoch_%K" in df.columns:
        df["STOCHK15"] = pd.to_numeric(df["Stoch_%K"], errors="coerce")
        df["STOCHD15"] = pd.to_numeric(df.get("Stoch_%D", np.nan), errors="coerce")
    else:
        k, d = compute_stoch_14_3(df)
        df["STOCHK15"] = k
        df["STOCHD15"] = d

    if "ADX" in df.columns:
        df["ADX15"] = pd.to_numeric(df["ADX"], errors="coerce")
    else:
        df["ADX15"] = compute_adx14(df)

    df["day"] = df["date"].dt.tz_convert(IST).dt.date

    all_trades: list[Trade] = []
    for day_val, df_day in df.groupby("day", sort=True):
        df_day = df_day.copy().reset_index(drop=True)
        if len(df_day) < 7:
            continue
        df_day["AVWAP"] = compute_day_avwap(df_day)
        trades = scan_one_day_row_by_row(ticker, df_day, str(day_val))
        if trades:
            all_trades.extend(trades)

    return all_trades


# =============================================================================
# SUMMARY
# =============================================================================
def summarize_trades(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "days_target_hit": 0,
            "trades_target_hit": 0,
            "trades_sl_hit": 0,
            "trades_be_hit": 0,
            "avg_pnl_pct": 0.0,
            "sum_pnl_pct": 0.0,
        }

    df = df.copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")

    days_target_hit = df.loc[df["outcome"].eq("TARGET"), "trade_date"].nunique()
    trades_target_hit = int(df["outcome"].eq("TARGET").sum())
    trades_sl_hit = int(df["outcome"].eq("SL").sum())
    trades_be_hit = int(df["outcome"].eq("BE").sum())

    avg_pnl_pct = float(df["pnl_pct"].mean(skipna=True))
    sum_pnl_pct = float(df["pnl_pct"].sum(skipna=True))

    return {
        "days_target_hit": int(days_target_hit),
        "trades_target_hit": int(trades_target_hit),
        "trades_sl_hit": int(trades_sl_hit),
        "trades_be_hit": int(trades_be_hit),
        "avg_pnl_pct": avg_pnl_pct,
        "sum_pnl_pct": sum_pnl_pct,
    }


def print_summary(stats: dict, df: pd.DataFrame) -> None:
    total_trades = len(df)
    total_days = df["trade_date"].nunique() if total_trades else 0
    sl_trades = int(df["outcome"].eq("SL").sum()) if total_trades else 0
    be_trades = int(df["outcome"].eq("BE").sum()) if total_trades else 0
    eod_trades = int(df["outcome"].eq("EOD").sum()) if total_trades else 0

    print("\n================ SUMMARY =================")
    print(f"Total trades                 : {total_trades}")
    print(f"Unique trade days            : {total_days}")
    print(f"Days TARGET was hit          : {stats['days_target_hit']}")
    print(f"Trades that hit TARGET       : {stats['trades_target_hit']}")
    print(f"Trades that hit SL           : {sl_trades}")
    print(f"Breakeven exits (BE)         : {be_trades}")
    print(f"EOD exits (no SL/Target)     : {eod_trades}")
    print(f"Average % PnL (per trade)    : {stats['avg_pnl_pct']:.4f}%")
    print(f"Sum of all % PnL (all trades): {stats['sum_pnl_pct']:.4f}%")
    if total_trades:
        print(f"Target hit-rate              : {100.0*stats['trades_target_hit']/total_trades:.2f}%")
        print(f"SL hit-rate                  : {100.0*sl_trades/total_trades:.2f}%")
        print(f"EOD rate                     : {100.0*eod_trades/total_trades:.2f}%")
    print("=========================================\n")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("[RUN] AVWAP SHORT scan — ALL DAYS in data (15m) | strict SL/Target")
    print("[INFO] Option A: ADX twice↑ + slope | RSI twice↓ | Stoch bearish + K twice↓")
    print("[INFO] Strict trend: EMA20<EMA50 AND close<EMA20 AND close<AVWAP")
    if USE_TIME_WINDOWS:
        print("[INFO] Time windows enabled:", ", ".join([f"{a.strftime('%H:%M')}-{b.strftime('%H:%M')}" for a, b in SIGNAL_WINDOWS]))
    if REQUIRE_AVWAP_REJECTION:
        print(f"[INFO] AVWAP rejection: mode={AVWAP_REJECT_MODE}, touch={AVWAP_REJECT_TOUCH}, consec>={AVWAP_REJECT_MIN_CONSEC_CLOSES}, dist={AVWAP_DIST_ATR_MULT}*ATR")
    print(f"[INFO] Per-ticker/day cap={MAX_TRADES_PER_TICKER_PER_DAY}, close-confirm={REQUIRE_ENTRY_CLOSE_CONFIRM}, min-bars-left={MIN_BARS_LEFT_AFTER_ENTRY}, breakeven={ENABLE_BREAKEVEN}")
    if ENABLE_TOPN_PER_DAY:
        print(f"[INFO] Top-N per day enabled: TOPN_PER_DAY={TOPN_PER_DAY}")

    if UPDATE_15M_FIRST:
        try:
            holidays = core._read_holidays(core.HOLIDAYS_FILE_DEFAULT)
        except Exception:
            holidays = set()
        print(f"[STEP] Updating 15m parquet once at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
        run_update_15m_once(holidays)

    tickers = list_tickers_15m()
    print(f"[STEP] Tickers found: {len(tickers)}")

    all_trades: list[Trade] = []
    for k, t in enumerate(tickers, start=1):
        path = os.path.join(DIR_15M, f"{t}{END_15M}")
        df = read_15m_parquet(path)
        if df.empty:
            continue

        trades = scan_all_days_for_ticker(t, df)
        if trades:
            all_trades.extend(trades)

        if k % 50 == 0:
            print(f"  scanned {k}/{len(tickers)} | trades_so_far={len(all_trades)}")

    if not all_trades:
        print("[DONE] No trades found across available days.")
        return

    out_raw = pd.DataFrame([x.__dict__ for x in all_trades])

    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    if SAVE_RAW_BEFORE_TOPN:
        raw_csv = REPORTS_DIR / f"avwap_short_trades_ALL_DAYS_{ts}_RAW.csv"
        out_raw.sort_values(["trade_date", "ticker", "entry_time_ist"]).to_csv(raw_csv, index=False)
        print(f"[FILE SAVED] {raw_csv}")

    out = out_raw.copy()

    if ENABLE_TOPN_PER_DAY and ("quality_score" in out.columns):
        out = out.sort_values(["trade_date", "quality_score", "ticker", "entry_time_ist"], ascending=[True, False, True, True])
        out = out.groupby("trade_date", sort=False).head(int(TOPN_PER_DAY)).reset_index(drop=True)

    front = [
        "trade_date", "ticker", "setup", "impulse_type",
        "signal_time_ist", "entry_time_ist",
        "entry_price", "sl_price", "target_price",
        "exit_time_ist", "exit_price", "outcome", "pnl_pct",
        "quality_score", "adx_signal", "rsi_signal", "stochk_signal",
        "avwap_dist_atr_signal", "ema20_gap_atr_signal", "atr_pct_signal",
    ]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    out = out.sort_values(["trade_date", "quality_score", "ticker", "entry_time_ist"], ascending=[True, False, True, True]).reset_index(drop=True)

    out_csv = REPORTS_DIR / f"avwap_short_trades_ALL_DAYS_{ts}.csv"
    out.to_csv(out_csv, index=False)

    stats = summarize_trades(out)
    print_summary(stats, out)

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 220)
    print("=============== SAMPLE TRADES (first 50) ===============")
    print(out.head(50)[["trade_date","ticker","setup","impulse_type","quality_score","entry_price","sl_price","target_price","exit_price","outcome","pnl_pct"]].to_string(index=False))

    print(f"\n[FILE SAVED] {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
