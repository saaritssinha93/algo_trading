# -*- coding: utf-8 -*-
"""
15m SHORT STRATEGY (Anchored VWAP) — Scan ALL DAYS present in data (row-by-row)

UPDATED (as requested): Option A + Option E
------------------------------------------------
OPTION A (Stricter trend filter):
1) ADX_MIN raised (default 20)
2) ADX must be twice-INCREASING AND have minimum slope:
      ADX[i] > ADX[i-1] > ADX[i-2]  AND  (ADX[i] - ADX[i-2]) >= ADX_SLOPE_MIN
3) EMA alignment is STRICT:
      EMA20 < EMA50  AND  close < EMA20
4) Also require close < AVWAP (strict; increases selectivity)

OPTION E (Time-of-day filter):
- Only allow signals/entries in selected windows (default):
    09:30–11:00 and 13:30–14:15
- This avoids midday chop and reduces low-quality trades.

Adds SUMMARY STATS:
- Days with at least one TARGET
- Trades that hit TARGET
- Avg pnl_pct
- Sum pnl_pct

Trade rules:
- STRICT Stop Loss: STOP_PCT above entry (short)
- STRICT Target:    TARGET_PCT below entry (short)
- If neither hit by last candle of that day -> exit at day last close (EOD)

CORRECTION (kept):
- ADX must be "twice INCREASING" at impulse candle:
      ADX[i] > ADX[i-1] > ADX[i-2]
- RSI "twice REDUCING":
      RSI[i] < RSI[i-1] < RSI[i-2]
- Stochastic bearish + "twice REDUCING" %K:
      K[i] < D[i]  AND  K[i] < K[i-1] < K[i-2]

Outputs:
- reports/avwap_short_trades_ALL_DAYS_<timestamp>.csv
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

# STRICT risk rules
STOP_PCT = 0.007
TARGET_PCT = 0.0065

# Impulse definitions
MOD_RED_MIN_ATR = 0.40
MOD_RED_MAX_ATR = 1.00
HUGE_RED_MIN_ATR = 1.50
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
# -------------------------------------------------------------------------
USE_TIME_WINDOWS = True
SIGNAL_WINDOWS = [
    (dtime(9, 15, 0), dtime(11, 0, 0)),
    (dtime(13, 30, 0), dtime(14, 30, 0)),
]

# --- Filters ---
# -------------------------------------------------------------------------
# OPTION A: Stricter trend filter
# -------------------------------------------------------------------------
ADX_MIN = 20.0          # raised from 10 -> 20 (more selective)
ADX_SLOPE_MIN = 1.0     # require ADX[i] - ADX[i-2] >= this

RSI_MAX_SHORT = 60.0
STOCHK_MAX = 80.0

# Paths relative to this file
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
    # If naive -> assume UTC; else keep tz
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
    """OPTION E: True if timestamp is inside ANY allowed signal window."""
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
    signal_time_ist: pd.Timestamp
    entry_time_ist: pd.Timestamp
    entry_price: float
    sl_price: float
    target_price: float
    exit_time_ist: pd.Timestamp
    exit_price: float
    outcome: str  # TARGET / SL / EOD
    pnl_pct: float


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


def simulate_exit_short_within_day(
    df_day: pd.DataFrame, entry_idx: int, entry_price: float
) -> tuple[int, pd.Timestamp, float, str]:
    """
    Walk forward within the same day until target/SL/EOD.
    Conservative: if candle hits both, assume SL first.
    """
    sl = entry_price * (1.0 + STOP_PCT)
    tgt = entry_price * (1.0 - TARGET_PCT)

    for k in range(entry_idx + 1, len(df_day)):
        hi = float(df_day.at[k, "high"])
        lo = float(df_day.at[k, "low"])
        ts = df_day.at[k, "date"]

        hit_sl = np.isfinite(hi) and (hi >= sl)
        hit_tg = np.isfinite(lo) and (lo <= tgt)

        if hit_sl and hit_tg:
            return k, ts, float(sl), "SL"
        if hit_sl:
            return k, ts, float(sl), "SL"
        if hit_tg:
            return k, ts, float(tgt), "TARGET"

    last = len(df_day) - 1
    ts = df_day.at[last, "date"]
    px = float(df_day.at[last, "close"])
    return last, ts, px, "EOD"


def _twice_increasing(df_day: pd.DataFrame, idx: int, col: str) -> bool:
    """True if col[idx] > col[idx-1] > col[idx-2] and all finite."""
    if idx < 2:
        return False
    a = float(df_day.at[idx, col]) if col in df_day.columns else np.nan
    b = float(df_day.at[idx - 1, col]) if col in df_day.columns else np.nan
    c = float(df_day.at[idx - 2, col]) if col in df_day.columns else np.nan
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a > b > c)


def _twice_reducing(df_day: pd.DataFrame, idx: int, col: str) -> bool:
    """True if col[idx] < col[idx-1] < col[idx-2] and all finite."""
    if idx < 2:
        return False
    a = float(df_day.at[idx, col]) if col in df_day.columns else np.nan
    b = float(df_day.at[idx - 1, col]) if col in df_day.columns else np.nan
    c = float(df_day.at[idx - 2, col]) if col in df_day.columns else np.nan
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a < b < c)


def _adx_slope_ok(df_day: pd.DataFrame, idx: int, col: str, min_slope: float) -> bool:
    """True if (col[idx] - col[idx-2]) >= min_slope and both finite."""
    if idx < 2 or col not in df_day.columns:
        return False
    a = float(df_day.at[idx, col])
    c = float(df_day.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(c) and ((a - c) >= float(min_slope))


def scan_one_day_row_by_row(ticker: str, df_day: pd.DataFrame, day_str: str) -> list[Trade]:
    """
    Row-by-row scan INSIDE one day.
    One trade at a time (sequential, no overlap).
    """
    if len(df_day) < 7:  # need at least i-2 and lookahead to i+3
        return []

    trades: list[Trade] = []
    i = 2  # start at 2 so we can test "twice" conditions

    while i < len(df_day) - 3:
        c1 = df_day.iloc[i]

        # OPTION E: signal must be in allowed time windows
        if not in_signal_window(c1["date"]):
            i += 1
            continue

        impulse = classify_red_impulse(c1)
        if impulse == "":
            i += 1
            continue

        # -------------------------
        # ADX/RSI/STOCH checks (Option A strengthens ADX)
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

        # -------------------------
        # OPTION A: STRICT EMA alignment + close below AVWAP
        # EMA20 < EMA50 AND close < EMA20 AND close < AVWAP
        # -------------------------
        close1 = float(c1["close"])
        avwap1 = float(c1["AVWAP"]) if np.isfinite(c1["AVWAP"]) else np.nan
        ema20_1 = float(c1["EMA20"]) if np.isfinite(c1["EMA20"]) else np.nan
        ema50_1 = float(c1["EMA50"]) if np.isfinite(c1["EMA50"]) else np.nan

        strict_ema_ok = (
            np.isfinite(ema20_1) and np.isfinite(ema50_1)
            and (ema20_1 < ema50_1)
            and (close1 < ema20_1)
        )
        strict_avwap_ok = np.isfinite(avwap1) and (close1 < avwap1)

        if not (strict_ema_ok and strict_avwap_ok):
            i += 1
            continue

        signal_time = c1["date"]
        low1 = float(c1["low"])
        open1 = float(c1["open"])
        close1 = float(c1["close"])
        atr1 = float(c1["ATR15"]) if np.isfinite(c1["ATR15"]) else np.nan
        if not np.isfinite(atr1) or atr1 <= 0:
            i += 1
            continue

        # -------------------------
        # SETUP A (MODERATE)
        # -------------------------
        if impulse == "MODERATE":
            c2 = df_day.iloc[i + 1]
            buf1 = _buffer(low1)

            # Option 1: break C1 low (entry must be inside allowed windows)
            if float(c2["low"]) < (low1 - buf1):
                entry_idx = i + 1
                entry_ts = df_day.at[entry_idx, "date"]
                if not in_signal_window(entry_ts):
                    i += 1
                    continue

                entry_price = float(low1 - buf1)

                exit_idx, exit_time, exit_price, outcome = simulate_exit_short_within_day(
                    df_day, entry_idx, entry_price
                )
                pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                trades.append(
                    Trade(
                        trade_date=day_str,
                        ticker=ticker,
                        setup="A_MOD_BREAK_C1_LOW",
                        signal_time_ist=signal_time,
                        entry_time_ist=entry_ts,
                        entry_price=entry_price,
                        sl_price=entry_price * (1.0 + STOP_PCT),
                        target_price=entry_price * (1.0 - TARGET_PCT),
                        exit_time_ist=exit_time,
                        exit_price=exit_price,
                        outcome=outcome,
                        pnl_pct=float(pnl_pct),
                    )
                )
                i = exit_idx + 1
                continue

            # Option 2: small green pullback C2 + below AVWAP, then break C2 low on C3
            c2o, c2c = float(c2["open"]), float(c2["close"])
            c2_body = abs(c2c - c2o)
            c2_atr = float(c2["ATR15"]) if np.isfinite(c2["ATR15"]) else atr1
            c2_avwap = float(c2["AVWAP"]) if np.isfinite(c2["AVWAP"]) else np.nan

            c2_small_green = (c2c > c2o) and np.isfinite(c2_atr) and (c2_body <= SMALL_GREEN_MAX_ATR * c2_atr)
            c2_below_avwap = np.isfinite(c2_avwap) and (c2c < c2_avwap)

            if c2_small_green and c2_below_avwap:
                c3 = df_day.iloc[i + 2]
                low2 = float(c2["low"])
                buf2 = _buffer(low2)

                if float(c3["low"]) < (low2 - buf2):
                    entry_idx = i + 2
                    entry_ts = df_day.at[entry_idx, "date"]
                    if not in_signal_window(entry_ts):
                        i += 1
                        continue

                    entry_price = float(low2 - buf2)

                    exit_idx, exit_time, exit_price, outcome = simulate_exit_short_within_day(
                        df_day, entry_idx, entry_price
                    )
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                    trades.append(
                        Trade(
                            trade_date=day_str,
                            ticker=ticker,
                            setup="A_PULLBACK_C2_THEN_BREAK_C2_LOW",
                            signal_time_ist=signal_time,
                            entry_time_ist=entry_ts,
                            entry_price=entry_price,
                            sl_price=entry_price * (1.0 + STOP_PCT),
                            target_price=entry_price * (1.0 - TARGET_PCT),
                            exit_time_ist=exit_time,
                            exit_price=exit_price,
                            outcome=outcome,
                            pnl_pct=float(pnl_pct),
                        )
                    )
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

            mid_body = (open1 + close1) / 2.0

            bounce_atr = pd.to_numeric(bounce["ATR15"], errors="coerce").fillna(atr1)
            bounce_body = (
                pd.to_numeric(bounce["close"], errors="coerce")
                - pd.to_numeric(bounce["open"], errors="coerce")
            ).abs()
            bounce_green = pd.to_numeric(bounce["close"], errors="coerce") > pd.to_numeric(
                bounce["open"], errors="coerce"
            )
            bounce_small = bounce_body <= (SMALL_GREEN_MAX_ATR * bounce_atr)

            if not bool((bounce_green & bounce_small).any()):
                i += 1
                continue

            closes = pd.to_numeric(bounce["close"], errors="coerce")
            avwaps = pd.to_numeric(bounce["AVWAP"], errors="coerce")
            fail_avwap = bool((closes < avwaps).fillna(False).all())

            highs = pd.to_numeric(bounce["high"], errors="coerce")
            fail_mid = bool((highs < mid_body).fillna(False).all())

            if not (fail_avwap or fail_mid):
                i += 1
                continue

            bounce_low = float(pd.to_numeric(bounce["low"], errors="coerce").min())
            if not np.isfinite(bounce_low):
                i += 1
                continue

            # breakdown after bounce window
            start_j = bounce_end + 1
            entered = False
            for j in range(start_j, len(df_day)):
                tsj = df_day.at[j, "date"]

                # OPTION E: entry must be in allowed time windows
                if not in_signal_window(tsj):
                    continue

                closej = float(df_day.at[j, "close"])
                avwapj = float(df_day.at[j, "AVWAP"]) if np.isfinite(df_day.at[j, "AVWAP"]) else np.nan
                if np.isfinite(avwapj) and closej >= avwapj:
                    break  # reclaimed AVWAP -> no failed-bounce short

                buf = _buffer(bounce_low)
                if float(df_day.at[j, "low"]) < (bounce_low - buf):
                    entry_idx = j
                    entry_price = float(bounce_low - buf)

                    exit_idx, exit_time, exit_price, outcome = simulate_exit_short_within_day(
                        df_day, entry_idx, entry_price
                    )
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0

                    trades.append(
                        Trade(
                            trade_date=day_str,
                            ticker=ticker,
                            setup="B_HUGE_RED_FAILED_BOUNCE",
                            signal_time_ist=signal_time,
                            entry_time_ist=tsj,
                            entry_price=entry_price,
                            sl_price=entry_price * (1.0 + STOP_PCT),
                            target_price=entry_price * (1.0 - TARGET_PCT),
                            exit_time_ist=exit_time,
                            exit_price=exit_price,
                            outcome=outcome,
                            pnl_pct=float(pnl_pct),
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
        }

    df = df.copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")

    days_target_hit = df.loc[df["outcome"].eq("TARGET"), "trade_date"].nunique()
    trades_target_hit = int(df["outcome"].eq("TARGET").sum())

    avg_pnl_pct = float(df["pnl_pct"].mean(skipna=True))
    sum_pnl_pct = float(df["pnl_pct"].sum(skipna=True))

    return {
        "days_target_hit": int(days_target_hit),
        "trades_target_hit": int(trades_target_hit),
        "avg_pnl_pct": avg_pnl_pct,
        "sum_pnl_pct": sum_pnl_pct,
    }


def print_summary(stats: dict, df: pd.DataFrame) -> None:
    total_trades = len(df)
    total_days = df["trade_date"].nunique() if total_trades else 0
    sl_trades = int(df["outcome"].eq("SL").sum()) if total_trades else 0
    eod_trades = int(df["outcome"].eq("EOD").sum()) if total_trades else 0

    print("\n================ SUMMARY =================")
    print(f"Total trades                 : {total_trades}")
    print(f"Unique trade days            : {total_days}")
    print(f"Days TARGET was hit          : {stats['days_target_hit']}")
    print(f"Trades that hit TARGET       : {stats['trades_target_hit']}")
    print(f"Trades that hit SL           : {sl_trades}")
    print(f"EOD exits (no SL/Target)     : {eod_trades}")
    print(f"Average % PnL (per trade)    : {stats['avg_pnl_pct']:.4f}%")
    print(f"Sum of all % PnL (all trades): {stats['sum_pnl_pct']:.4f}%")
    print("=========================================\n")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("[RUN] AVWAP SHORT scan — ALL DAYS in data (15m) | strict SL/Target")
    print("[INFO] Filters: ADX twice-INCREASING + ADX slope + RSI twice-DECREASING + Stoch bearish & %K twice-DECREASING")
    print("[INFO] Strict trend: EMA20<EMA50 AND close<EMA20 AND close<AVWAP")
    if USE_TIME_WINDOWS:
        print("[INFO] Time windows enabled:", ", ".join([f"{a.strftime('%H:%M')}-{b.strftime('%H:%M')}" for a, b in SIGNAL_WINDOWS]))

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

    out = pd.DataFrame([x.__dict__ for x in all_trades])

    # ticker as SECOND column
    front = [
        "trade_date",
        "ticker",
        "setup",
        "signal_time_ist",
        "entry_time_ist",
        "entry_price",
        "sl_price",
        "target_price",
        "exit_time_ist",
        "exit_price",
        "outcome",
        "pnl_pct",
    ]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    out = out.sort_values(["trade_date", "ticker", "entry_time_ist"]).reset_index(drop=True)

    # Save CSV
    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    out_csv = REPORTS_DIR / f"avwap_short_trades_ALL_DAYS_{ts}.csv"
    out.to_csv(out_csv, index=False)

    # Compute + print stats
    stats = summarize_trades(out)
    print_summary(stats, out)

    # Sample
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 220)
    print("=============== SAMPLE TRADES (first 50) ===============")
    print(out.head(50)[["trade_date","ticker","setup","entry_price","sl_price","target_price","exit_price","outcome","pnl_pct"]].to_string(index=False))

    print(f"\n[FILE SAVED] {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
