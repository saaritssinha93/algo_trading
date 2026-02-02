# -*- coding: utf-8 -*-
"""
15m LONG STRATEGY (Anchored VWAP) — Scan ALL DAYS present in data (row-by-row)

CURRENT VERSION: A + B + E + (Quality upgrades to reduce trades / improve hit-rate)
---------------------------------------------------------------------------------

OPTION A (Stricter trend filter - LONG):
- ADX must be twice-INCREASING + slope min
- RSI must be twice-INCREASING + above RSI_MIN_LONG
- Stochastic bullish + %K twice-INCREASING
- STRICT EMA alignment: EMA20 > EMA50 AND close > EMA20
- STRICT AVWAP: close > AVWAP

OPTION B (AVWAP "support / rejection-below" rule + distance):
Goal: Don’t long just because price is above AVWAP; long when it FAILS to break below / holds support.

Evidence between impulse and entry:
- touch-support rule: any candle has low <= AVWAP and close > AVWAP
- consecutive closes rule: need N consecutive closes ABOVE AVWAP
Combine via AVWAP_SUPPORT_MODE: "any" or "both"
Distance rule at entry (avoid mid-zone chop):
    (close_at_entry_candle - AVWAP) >= AVWAP_DIST_ATR_MULT * ATR

OPTION E (Time-of-day filter):
- Only allow signals/entries in selected windows (default):
    09:15–11:30 and 13:00–14:30

QUALITY UPGRADES:
Q1) Per-ticker per-day cap (default 1)
Q2) Close-confirmed breakout entries (reduces fakeouts)
Q3) Avoid late entries (min bars left)
Q4) Breakeven protection (reduces full SL hits after partial move)
Q5) Optional Top-N trades per day by quality score

Trade rules:
- Stop Loss: STOP_PCT below entry (long)
- Target:    TARGET_PCT above entry (long)
- If neither hit by last candle -> exit at day last close (EOD)
- Conservative: if a candle hits both SL and Target, assume SL first.

Outputs:
- reports/avwap_long_trades_ALL_DAYS_<timestamp>.csv  (final post-filtered if enabled)
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

# Optional: run your 15m updater once before scan (kept same structure)
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
UPDATE_15M_FIRST = False  # True if you want update once before scan

# -------------------------
# STRICT risk rules (LONG)
# -------------------------
STOP_PCT = 0.0100
TARGET_PCT = 0.0200

# -------------------------
# Impulse definitions (GREEN)
# -------------------------
MOD_GREEN_MIN_ATR = 0.30
MOD_GREEN_MAX_ATR = 1.00
HUGE_GREEN_MIN_ATR = 1.60
HUGE_GREEN_MIN_RANGE_ATR = 2.00
CLOSE_NEAR_HIGH_MAX = 0.25   # (high-close)/range <= 0.25

# Pullback candle size (small RED)
SMALL_RED_MAX_ATR = 0.20

# Entry buffer
BUFFER_ABS = 0.05
BUFFER_PCT = 0.0002

# Session bounds (data filter)
SESSION_START = dtime(9, 15, 0)
SESSION_END   = dtime(14, 30, 0)

# -------------------------------------------------------------------------
# OPTION E: Time-of-day filter (ONLY trade in these windows)
# -------------------------------------------------------------------------
USE_TIME_WINDOWS = True
SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 30, 0)),
    (dtime(13, 0, 0), dtime(14, 30, 0)),
]

# -------------------------------------------------------------------------
# OPTION A: Stricter trend filter (LONG)
# -------------------------------------------------------------------------
ADX_MIN = 25.0
ADX_SLOPE_MIN = 1.25

RSI_MIN_LONG = 45.0
STOCHK_MIN = 25.0
STOCHK_MAX = 95.0   # optional cap (avoid extreme overbought entries)

# Extra (optional) volatility sanity: skip very low ATR% names (reduces chop)
USE_ATR_PCT_FILTER = True
ATR_PCT_MIN = 0.0020  # ATR/close must be >= 0.20%

# -------------------------------------------------------------------------
# OPTION B: AVWAP support + distance rules (LONG)
# -------------------------------------------------------------------------
REQUIRE_AVWAP_SUPPORT = True

# Support evidence:
#   - touch rule: any candle (between impulse and entry) has low <= AVWAP and close > AVWAP
AVWAP_SUPPORT_TOUCH = True

#   - consecutive closes rule: need N consecutive closes ABOVE AVWAP between impulse and entry
AVWAP_SUPPORT_MIN_CONSEC_CLOSES = 2

# Combine touch + consec evidence: "any" or "both"
AVWAP_SUPPORT_MODE = "any"

# Distance from AVWAP at entry (avoid mid-zone chop)
AVWAP_DIST_ATR_MULT = 0.25

# -------------------------------------------------------------------------
# QUALITY UPGRADES
# -------------------------------------------------------------------------
MAX_TRADES_PER_TICKER_PER_DAY = 1

REQUIRE_ENTRY_CLOSE_CONFIRM = True
# if True: entry candle CLOSE must be above trigger (not only HIGH)

MIN_BARS_LEFT_AFTER_ENTRY = 4  # 4*15m = 60 minutes left after entry

ENABLE_BREAKEVEN = True
BE_TRIGGER_PCT = 0.0040   # once price moves +0.4% in favor
BE_PAD_PCT = 0.0001       # move SL to entry*(1+pad) for longs (tiny locked profit)

ENABLE_TOPN_PER_DAY = True
TOPN_PER_DAY = 30
SAVE_RAW_BEFORE_TOPN = False

# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
API_KEY_FILE = ROOT / "api_key.txt"
ACCESS_TOKEN_FILE = ROOT / "access_token.txt"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_ENGINE = "pyarrow"


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
    quality_score: float


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


def classify_green_impulse(row: pd.Series) -> str:
    o = float(row["open"])
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    atr = float(row["ATR15"])

    if not np.isfinite(atr) or atr <= 0:
        return ""
    if not (c > o):
        return ""

    body = abs(c - o)
    rng = (h - l) if (h >= l) else np.nan
    if not np.isfinite(rng) or rng <= 0:
        return ""

    close_near_high = ((h - c) / rng) <= CLOSE_NEAR_HIGH_MAX

    if (body >= HUGE_GREEN_MIN_ATR * atr) or (rng >= HUGE_GREEN_MIN_RANGE_ATR * atr):
        return "HUGE"

    if (body >= MOD_GREEN_MIN_ATR * atr) and (body <= MOD_GREEN_MAX_ATR * atr) and close_near_high:
        return "MODERATE"

    return ""


def _count_max_consec(condition: pd.Series) -> int:
    """Maximum consecutive Trues in a boolean series."""
    best = 0
    cur = 0
    for v in condition.fillna(False).astype(bool).tolist():
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def avwap_support_ok(
    df_day: pd.DataFrame,
    impulse_idx: int,
    entry_idx: int,
    atr_entry: float,
) -> tuple[bool, float]:
    """
    OPTION B (LONG): require AVWAP support evidence between impulse and entry,
    plus distance from AVWAP at entry.

    Returns: (ok, avwap_dist_atr)
    """
    if not REQUIRE_AVWAP_SUPPORT:
        return True, 0.0

    if entry_idx <= impulse_idx:
        return False, 0.0

    seg = df_day.iloc[impulse_idx + 1 : entry_idx + 1].copy()
    if seg.empty:
        return False, 0.0

    av = pd.to_numeric(seg["AVWAP"], errors="coerce")
    lo = pd.to_numeric(seg["low"], errors="coerce")
    cl = pd.to_numeric(seg["close"], errors="coerce")

    # Evidence 1: touch support (low <= AVWAP and close > AVWAP)
    touch_ok = False
    if AVWAP_SUPPORT_TOUCH:
        touch_ok = bool(((lo <= av) & (cl > av)).fillna(False).any())

    # Evidence 2: consecutive closes above AVWAP
    consec_ok = False
    if AVWAP_SUPPORT_MIN_CONSEC_CLOSES and AVWAP_SUPPORT_MIN_CONSEC_CLOSES > 0:
        consec_ok = _count_max_consec((cl > av)) >= int(AVWAP_SUPPORT_MIN_CONSEC_CLOSES)

    if AVWAP_SUPPORT_MODE.lower() == "both":
        evidence_ok = touch_ok and consec_ok
    else:
        evidence_ok = touch_ok or consec_ok

    # Distance from AVWAP at entry candle (use entry candle CLOSE vs AVWAP)
    entry_close = float(df_day.at[entry_idx, "close"])
    entry_avwap = float(df_day.at[entry_idx, "AVWAP"])
    if (not np.isfinite(entry_close)) or (not np.isfinite(entry_avwap)) or (not np.isfinite(atr_entry)) or atr_entry <= 0:
        return False, 0.0

    avwap_dist = entry_close - entry_avwap
    avwap_dist_atr = avwap_dist / atr_entry
    dist_ok = avwap_dist >= (AVWAP_DIST_ATR_MULT * atr_entry)

    return bool(evidence_ok and dist_ok), float(avwap_dist_atr)


def simulate_exit_long_within_day(
    df_day: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
) -> tuple[int, pd.Timestamp, float, str]:
    """
    Walk forward within the same day until target/SL/BE/EOD.
    Conservative: if candle hits both SL and Target, assume SL/BE first.
    """
    sl = entry_price * (1.0 - STOP_PCT)
    tgt = entry_price * (1.0 + TARGET_PCT)

    sl_curr = float(sl)
    be_armed = False
    be_level = entry_price * (1.0 + BE_PAD_PCT)

    for k in range(entry_idx + 1, len(df_day)):
        hi = float(df_day.at[k, "high"])
        lo = float(df_day.at[k, "low"])
        ts = df_day.at[k, "date"]

        # Arm breakeven if moved enough in favor
        if ENABLE_BREAKEVEN and (not be_armed) and np.isfinite(hi) and (hi >= entry_price * (1.0 + BE_TRIGGER_PCT)):
            be_armed = True
            sl_curr = max(sl_curr, be_level)  # lock tiny profit

        hit_sl = np.isfinite(lo) and (lo <= sl_curr)
        hit_tg = np.isfinite(hi) and (hi >= tgt)

        if hit_sl and hit_tg:
            # Conservative: assume stop first
            if be_armed and sl_curr >= entry_price:
                return k, ts, float(sl_curr), "BE"
            return k, ts, float(sl_curr), "SL"
        if hit_sl:
            if be_armed and sl_curr >= entry_price:
                return k, ts, float(sl_curr), "BE"
            return k, ts, float(sl_curr), "SL"
        if hit_tg:
            return k, ts, float(tgt), "TARGET"

    last = len(df_day) - 1
    ts = df_day.at[last, "date"]
    px = float(df_day.at[last, "close"])
    return last, ts, px, "EOD"


def quality_score_long(
    adx: float,
    adx_slope2: float,
    avwap_dist_atr: float,
    ema_gap_atr: float,
    impulse_type: str,
) -> float:
    """
    Higher score = better candidate.
    """
    imp_bonus = 0.25 if impulse_type == "HUGE" else 0.0
    score = (
        0.04 * float(adx) +
        0.20 * float(adx_slope2) +
        1.20 * float(avwap_dist_atr) +
        0.80 * float(ema_gap_atr) +
        imp_bonus
    )
    return float(score)


def scan_one_day_row_by_row(ticker: str, df_day: pd.DataFrame, day_str: str) -> list[Trade]:
    """
    Row-by-row scan INSIDE one day.
    One trade at a time (sequential, no overlap).
    Also enforces per-ticker/day cap.
    """
    if len(df_day) < 7:
        return []

    trades: list[Trade] = []
    i = 2

    while i < len(df_day) - 3:
        if len(trades) >= int(MAX_TRADES_PER_TICKER_PER_DAY):
            break

        c1 = df_day.iloc[i]

        # Option E: signal must be in allowed windows
        if not in_signal_window(c1["date"]):
            i += 1
            continue

        impulse = classify_green_impulse(c1)
        if impulse == "":
            i += 1
            continue

        # -------------------------
        # ADX/RSI/STOCH checks (Option A)
        # -------------------------
        adx1 = float(df_day.at[i, "ADX15"]) if "ADX15" in df_day.columns else np.nan
        rsi1 = float(df_day.at[i, "RSI15"]) if "RSI15" in df_day.columns else np.nan
        k1   = float(df_day.at[i, "STOCHK15"]) if "STOCHK15" in df_day.columns else np.nan
        d1   = float(df_day.at[i, "STOCHD15"]) if "STOCHD15" in df_day.columns else np.nan

        adx_ok = (
            np.isfinite(adx1)
            and (adx1 >= ADX_MIN)
            and _twice_increasing(df_day, i, "ADX15")
            and _adx_slope_ok(df_day, i, "ADX15", ADX_SLOPE_MIN)
        )

        rsi_ok = (
            np.isfinite(rsi1)
            and (rsi1 >= RSI_MIN_LONG)
            and _twice_increasing(df_day, i, "RSI15")
        )

        stoch_ok = (
            np.isfinite(k1) and np.isfinite(d1)
            and (k1 >= STOCHK_MIN)
            and (k1 <= STOCHK_MAX)
            and (k1 > d1)
            and _twice_increasing(df_day, i, "STOCHK15")
        )

        if not (adx_ok and rsi_ok and stoch_ok):
            i += 1
            continue

        # -------------------------
        # OPTION A: STRICT EMA alignment + close above AVWAP
        # EMA20 > EMA50 AND close > EMA20 AND close > AVWAP
        # -------------------------
        close1 = float(c1["close"])
        avwap1 = float(c1["AVWAP"]) if np.isfinite(c1["AVWAP"]) else np.nan
        ema20_1 = float(c1["EMA20"]) if np.isfinite(c1["EMA20"]) else np.nan
        ema50_1 = float(c1["EMA50"]) if np.isfinite(c1["EMA50"]) else np.nan

        strict_ema_ok = (
            np.isfinite(ema20_1) and np.isfinite(ema50_1)
            and (ema20_1 > ema50_1)
            and (close1 > ema20_1)
        )
        strict_avwap_ok = np.isfinite(avwap1) and (close1 > avwap1)

        if not (strict_ema_ok and strict_avwap_ok):
            i += 1
            continue

        atr1 = float(c1["ATR15"]) if np.isfinite(c1["ATR15"]) else np.nan
        if not np.isfinite(atr1) or atr1 <= 0:
            i += 1
            continue

        # optional ATR% filter
        if USE_ATR_PCT_FILTER:
            if close1 <= 0 or (atr1 / close1) < float(ATR_PCT_MIN):
                i += 1
                continue

        signal_time = c1["date"]
        high1 = float(c1["high"])
        open1 = float(c1["open"])

        # =========================
        # SETUP A (MODERATE impulse)
        # =========================
        if impulse == "MODERATE":
            c2 = df_day.iloc[i + 1]
            buf1 = _buffer(high1)

            # ---- Option 1: break C1 high on C2
            trigger = float(high1 + buf1)

            if float(c2["high"]) > trigger:
                entry_idx = i + 1
                entry_ts = df_day.at[entry_idx, "date"]
                if not in_signal_window(entry_ts):
                    i += 1
                    continue

                # Close-confirm breakout
                if REQUIRE_ENTRY_CLOSE_CONFIRM and not (float(c2["close"]) > trigger):
                    i += 1
                    continue

                # Late-entry filter
                bars_left = (len(df_day) - 1) - entry_idx
                if bars_left < int(MIN_BARS_LEFT_AFTER_ENTRY):
                    i += 1
                    continue

                # AVWAP support rule (Option B)
                atr_entry = float(df_day.at[entry_idx, "ATR15"])
                ok_support, avwap_dist_atr = avwap_support_ok(df_day, i, entry_idx, atr_entry)
                if not ok_support:
                    i += 1
                    continue

                entry_price = trigger
                exit_idx, exit_time, exit_price, outcome = simulate_exit_long_within_day(
                    df_day, entry_idx, entry_price
                )
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0

                adx_slope2 = float(df_day.at[i, "ADX15"] - df_day.at[i - 2, "ADX15"]) if i >= 2 else 0.0
                ema_gap_atr = (close1 - ema20_1) / atr1 if atr1 > 0 else 0.0
                qscore = quality_score_long(adx1, adx_slope2, avwap_dist_atr, ema_gap_atr, impulse)

                trades.append(
                    Trade(
                        trade_date=day_str,
                        ticker=ticker,
                        setup="A_MOD_BREAK_C1_HIGH",
                        impulse_type=impulse,
                        signal_time_ist=signal_time,
                        entry_time_ist=entry_ts,
                        entry_price=float(entry_price),
                        sl_price=float(entry_price * (1.0 - STOP_PCT)),
                        target_price=float(entry_price * (1.0 + TARGET_PCT)),
                        exit_time_ist=exit_time,
                        exit_price=float(exit_price),
                        outcome=outcome,
                        pnl_pct=float(pnl_pct),
                        quality_score=float(qscore),
                    )
                )
                i = exit_idx + 1
                continue

            # ---- Option 2: small red pullback C2 (above AVWAP), then break C2 high on C3
            c2o, c2c = float(c2["open"]), float(c2["close"])
            c2_body = abs(c2c - c2o)
            c2_atr = float(c2["ATR15"]) if np.isfinite(c2["ATR15"]) else atr1
            c2_avwap = float(c2["AVWAP"]) if np.isfinite(c2["AVWAP"]) else np.nan

            c2_small_red = (c2c < c2o) and np.isfinite(c2_atr) and (c2_body <= SMALL_RED_MAX_ATR * c2_atr)
            c2_above_avwap = np.isfinite(c2_avwap) and (c2c > c2_avwap)

            if c2_small_red and c2_above_avwap:
                c3 = df_day.iloc[i + 2]
                high2 = float(c2["high"])
                buf2 = _buffer(high2)
                trigger2 = float(high2 + buf2)

                if float(c3["high"]) > trigger2:
                    entry_idx = i + 2
                    entry_ts = df_day.at[entry_idx, "date"]
                    if not in_signal_window(entry_ts):
                        i += 1
                        continue

                    if REQUIRE_ENTRY_CLOSE_CONFIRM and not (float(c3["close"]) > trigger2):
                        i += 1
                        continue

                    bars_left = (len(df_day) - 1) - entry_idx
                    if bars_left < int(MIN_BARS_LEFT_AFTER_ENTRY):
                        i += 1
                        continue

                    atr_entry = float(df_day.at[entry_idx, "ATR15"])
                    ok_support, avwap_dist_atr = avwap_support_ok(df_day, i, entry_idx, atr_entry)
                    if not ok_support:
                        i += 1
                        continue

                    entry_price = trigger2
                    exit_idx, exit_time, exit_price, outcome = simulate_exit_long_within_day(
                        df_day, entry_idx, entry_price
                    )
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0

                    adx_slope2 = float(df_day.at[i, "ADX15"] - df_day.at[i - 2, "ADX15"]) if i >= 2 else 0.0
                    ema_gap_atr = (close1 - ema20_1) / atr1 if atr1 > 0 else 0.0
                    qscore = quality_score_long(adx1, adx_slope2, avwap_dist_atr, ema_gap_atr, impulse)

                    trades.append(
                        Trade(
                            trade_date=day_str,
                            ticker=ticker,
                            setup="A_PULLBACK_C2_THEN_BREAK_C2_HIGH",
                            impulse_type=impulse,
                            signal_time_ist=signal_time,
                            entry_time_ist=entry_ts,
                            entry_price=float(entry_price),
                            sl_price=float(entry_price * (1.0 - STOP_PCT)),
                            target_price=float(entry_price * (1.0 + TARGET_PCT)),
                            exit_time_ist=exit_time,
                            exit_price=float(exit_price),
                            outcome=outcome,
                            pnl_pct=float(pnl_pct),
                            quality_score=float(qscore),
                        )
                    )
                    i = exit_idx + 1
                    continue

            i += 1
            continue

        # =========================
        # SETUP B (HUGE impulse)
        # =========================
        if impulse == "HUGE":
            pull_end = min(i + 3, len(df_day) - 1)
            pull = df_day.iloc[i + 1 : pull_end + 1].copy()
            if pull.empty:
                i += 1
                continue

            mid_body = (open1 + close1) / 2.0

            pull_atr = pd.to_numeric(pull["ATR15"], errors="coerce").fillna(atr1)
            pull_body = (pd.to_numeric(pull["close"], errors="coerce") - pd.to_numeric(pull["open"], errors="coerce")).abs()
            pull_red = pd.to_numeric(pull["close"], errors="coerce") < pd.to_numeric(pull["open"], errors="coerce")
            pull_small = pull_body <= (SMALL_RED_MAX_ATR * pull_atr)

            # need at least one small red pullback candle
            if not bool((pull_red & pull_small).any()):
                i += 1
                continue

            # Structure hold: prefer that pullback lows stay above impulse mid-body OR closes stay above AVWAP
            lows = pd.to_numeric(pull["low"], errors="coerce")
            closes = pd.to_numeric(pull["close"], errors="coerce")
            avwaps = pd.to_numeric(pull["AVWAP"], errors="coerce")

            hold_mid = bool((lows > mid_body).fillna(False).all())
            hold_avwap = bool((closes > avwaps).fillna(False).all())

            if not (hold_mid or hold_avwap):
                i += 1
                continue

            pull_high = float(pd.to_numeric(pull["high"], errors="coerce").max())
            if not np.isfinite(pull_high):
                i += 1
                continue

            # breakout after pullback window
            start_j = pull_end + 1
            entered = False
            trigger = float(pull_high + _buffer(pull_high))

            for j in range(start_j, len(df_day)):
                tsj = df_day.at[j, "date"]
                if not in_signal_window(tsj):
                    continue

                # if price collapses below AVWAP before breakout, invalidate
                closej = float(df_day.at[j, "close"])
                avwapj = float(df_day.at[j, "AVWAP"]) if np.isfinite(df_day.at[j, "AVWAP"]) else np.nan
                if np.isfinite(avwapj) and closej <= avwapj:
                    break

                if float(df_day.at[j, "high"]) > trigger:
                    if REQUIRE_ENTRY_CLOSE_CONFIRM and not (float(df_day.at[j, "close"]) > trigger):
                        continue

                    bars_left = (len(df_day) - 1) - j
                    if bars_left < int(MIN_BARS_LEFT_AFTER_ENTRY):
                        continue

                    atr_entry = float(df_day.at[j, "ATR15"])
                    ok_support, avwap_dist_atr = avwap_support_ok(df_day, i, j, atr_entry)
                    if not ok_support:
                        continue

                    entry_idx = j
                    entry_price = trigger

                    exit_idx, exit_time, exit_price, outcome = simulate_exit_long_within_day(
                        df_day, entry_idx, entry_price
                    )
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0

                    adx_slope2 = float(df_day.at[i, "ADX15"] - df_day.at[i - 2, "ADX15"]) if i >= 2 else 0.0
                    ema_gap_atr = (close1 - ema20_1) / atr1 if atr1 > 0 else 0.0
                    qscore = quality_score_long(adx1, adx_slope2, avwap_dist_atr, ema_gap_atr, impulse)

                    trades.append(
                        Trade(
                            trade_date=day_str,
                            ticker=ticker,
                            setup="B_HUGE_GREEN_PULLBACK_HOLD_THEN_BREAK",
                            impulse_type=impulse,
                            signal_time_ist=signal_time,
                            entry_time_ist=tsj,
                            entry_price=float(entry_price),
                            sl_price=float(entry_price * (1.0 - STOP_PCT)),
                            target_price=float(entry_price * (1.0 + TARGET_PCT)),
                            exit_time_ist=exit_time,
                            exit_price=float(exit_price),
                            outcome=outcome,
                            pnl_pct=float(pnl_pct),
                            quality_score=float(qscore),
                        )
                    )
                    i = exit_idx + 1
                    entered = True
                    break

            if entered:
                continue

            i += 1
            continue

        i += 1

    return trades


def scan_all_days_for_ticker(ticker: str, df_full: pd.DataFrame) -> list[Trade]:
    """
    Prepare indicators on full series, then loop day-by-day and scan row-by-row.
    """
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

    # ATR/EMA on full timeline
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

    # RSI / STOCH / ADX (use existing columns if present, else compute)
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
# SUMMARY STATS
# =============================================================================
def summarize_trades(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "days_target_hit": 0,
            "trades_target_hit": 0,
            "avg_pnl_pct": 0.0,
            "sum_pnl_pct": 0.0,
            "target_hit_rate": 0.0,
            "sl_hit_rate": 0.0,
            "be_rate": 0.0,
            "eod_rate": 0.0,
        }

    df = df.copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")

    total = len(df)
    days_target_hit = df.loc[df["outcome"].eq("TARGET"), "trade_date"].nunique()
    trades_target_hit = int(df["outcome"].eq("TARGET").sum())
    sl_trades = int(df["outcome"].eq("SL").sum())
    be_trades = int(df["outcome"].eq("BE").sum())
    eod_trades = int(df["outcome"].eq("EOD").sum())

    avg_pnl_pct = float(df["pnl_pct"].mean(skipna=True))
    sum_pnl_pct = float(df["pnl_pct"].sum(skipna=True))

    return {
        "days_target_hit": int(days_target_hit),
        "trades_target_hit": int(trades_target_hit),
        "avg_pnl_pct": avg_pnl_pct,
        "sum_pnl_pct": sum_pnl_pct,
        "target_hit_rate": 100.0 * trades_target_hit / total if total else 0.0,
        "sl_hit_rate": 100.0 * sl_trades / total if total else 0.0,
        "be_rate": 100.0 * be_trades / total if total else 0.0,
        "eod_rate": 100.0 * eod_trades / total if total else 0.0,
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
    print(f"Target hit-rate              : {stats['target_hit_rate']:.2f}%")
    print(f"SL hit-rate                  : {stats['sl_hit_rate']:.2f}%")
    print(f"BE rate                      : {stats['be_rate']:.2f}%")
    print(f"EOD rate                     : {stats['eod_rate']:.2f}%")
    print("=========================================\n")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("[RUN] AVWAP LONG scan — ALL DAYS in data (15m) | strict SL/Target")
    print("[INFO] Option A: ADX twice↑ + slope | RSI twice↑ | Stoch bullish + K twice↑")
    print("[INFO] Strict trend: EMA20>EMA50 AND close>EMA20 AND close>AVWAP")
    if USE_TIME_WINDOWS:
        print("[INFO] Time windows enabled:", ", ".join([f"{a.strftime('%H:%M')}-{b.strftime('%H:%M')}" for a, b in SIGNAL_WINDOWS]))
    if REQUIRE_AVWAP_SUPPORT:
        print(f"[INFO] AVWAP support: mode={AVWAP_SUPPORT_MODE}, touch={AVWAP_SUPPORT_TOUCH}, consec>={AVWAP_SUPPORT_MIN_CONSEC_CLOSES}, dist={AVWAP_DIST_ATR_MULT}*ATR")
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
        raw_csv = REPORTS_DIR / f"avwap_long_trades_ALL_DAYS_{ts}_RAW.csv"
        out_raw.sort_values(["trade_date", "ticker", "entry_time_ist"]).to_csv(raw_csv, index=False)
        print(f"[FILE SAVED] {raw_csv}")

    out = out_raw.copy()

    # Top-N per day post-filter
    if ENABLE_TOPN_PER_DAY and ("quality_score" in out.columns):
        out = out.sort_values(["trade_date", "quality_score", "ticker", "entry_time_ist"], ascending=[True, False, True, True])
        out = out.groupby("trade_date", as_index=False).head(int(TOPN_PER_DAY)).reset_index(drop=True)

    # Column order
    front = [
        "trade_date",
        "ticker",
        "setup",
        "impulse_type",
        "signal_time_ist",
        "entry_time_ist",
        "entry_price",
        "sl_price",
        "target_price",
        "exit_time_ist",
        "exit_price",
        "outcome",
        "pnl_pct",
        "quality_score",
    ]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    out = out.sort_values(["trade_date", "ticker", "entry_time_ist"]).reset_index(drop=True)

    out_csv = REPORTS_DIR / f"avwap_long_trades_ALL_DAYS_{ts}.csv"
    out.to_csv(out_csv, index=False)

    stats = summarize_trades(out)
    print_summary(stats, out)

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 220)
    print("=============== SAMPLE TRADES (first 50) ===============")
    print(out.head(50)[["trade_date","ticker","setup","entry_price","sl_price","target_price","exit_price","outcome","pnl_pct","quality_score"]].to_string(index=False))

    print(f"\n[FILE SAVED] {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
