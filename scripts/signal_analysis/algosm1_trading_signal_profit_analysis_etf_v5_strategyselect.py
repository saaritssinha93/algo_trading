# -*- coding: utf-8 -*-
"""
Multi-timeframe positional LONG backtest framework with selectable strategies.

Core flow
---------
1) Generate multi-timeframe positional LONG signals using a chosen strategy:
    • Weekly + Daily (and sometimes Hourly) logic.
    • ENTRY TIMING: every 1H bar where that strategy's conditions are true
      becomes a LONG entry.

2) Evaluate those LONG signals using 1H data to see if a +TARGET_PCT% move
   was achieved after entry:
       - LONG  → +TARGET_PCT% target  (high >= (1 + TARGET_PCT/100) * entry_price)

3) Capital model:
    • You invest CAPITAL_PER_SIGNAL_RS in each signal you TAKE.
    • Total capital pool is limited to MAX_CAPITAL_RS.
      - If available capital < CAPITAL_PER_SIGNAL_RS at a new signal,
        that signal is SKIPPED.
      - Once some open trades exit and free capital, you can start taking
        new signals again.

Weekly rule (NO SAME-DATE LOOKAHEAD)
------------------------------------
    • Weekly bars are stored at week-end (e.g. Friday close).
    • For any intraday bar on that same calendar date, we must still use
      the *previous* weekly bar, not the current week’s bar.
    • Implementation:
        - Before merging, shift weekly 'date' forward by a tiny delta
          (e.g. +1 second).
        - Then do pd.merge_asof(..., direction="backward").
        - This ensures all hourly bars on the weekly bar’s calendar date
          still see the previous week's weekly bar.

Daily rule using **previous day's** Daily_Change (for some strategies)
----------------------------------------------------------------------
    • For an hourly bar on day D, we look at Daily_Change of day D-1.
    • Implemented via Daily_Change_prev in daily TF, then carried into
      merged dataframe as Daily_Change_prev_D.

Outputs
-------
    • signals/multi_tf_signals_<YYYYMMDD_HHMM>_IST.csv
    • signals/multi_tf_signals_etf_daily_counts_<YYYYMMDD_HHMM>_IST.csv
    • signals/multi_tf_signals_<TARGET_INT>pct_eval_<YYYYMMDD_HHMM>_IST.csv

Strategies
----------
Base strategies:
    1) trend_pullback
    2) weekly_breakout
    3) weekly_squeeze
    4) insidebar_breakout
    5) hh_hl_retest
    6) golden_cross

Custom strategies:
    7) custom1
    8) custom2
    9) custom3
   10) custom4
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_etf_1h  = "main_indicators_etf_1h"
DIR_D       = "main_indicators_etf_daily"
DIR_W       = "main_indicators_etf_weekly"
OUT_DIR     = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Target configuration (single source of truth) ----
TARGET_PCT: float = 10  # <-- change this once (e.g. 4.0, 5.0, 7.0, 10.0)
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)  # used in names, assume integer percent
TARGET_LABEL: str = f"{TARGET_INT}%"  # for printing

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 50000.0   # fixed rupee amount per signal
MAX_CAPITAL_RS: float = 5000000.0        # total capital pool

# Dynamic column names based on TARGET_PCT
HIT_COL: str = f"hit_{TARGET_INT}pct"
TIME_TD_COL: str = f"time_to_{TARGET_INT}pct_days"
TIME_CAL_COL: str = f"days_to_{TARGET_INT}pct_calendar"

# Eval filename suffix based on TARGET_PCT
EVAL_SUFFIX: str = f"{TARGET_INT}pct_eval"

# For outcome evaluation (LONG only)
TRADING_HOURS_PER_DAY = 6.25
N_RECENT_SKIP = 1  # how many most-recent LONG signals to treat as "recent"


# ----------------------- HELPERS (COMMON) -----------------------
def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """
    Read a timeframe CSV, parse 'date' as tz-aware IST, sort.
    """
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(tz)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()


def _prepare_tf_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Take a dataframe with a 'date' column and indicator columns and
    append a suffix (e.g. '_H', '_D', '_W') to all columns except 'date'.
    """
    if df.empty:
        return df
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)


def _list_tickers_from_dir(directory: str, ending: str) -> set[str]:
    """
    List tickers in a directory by stripping a known suffix from filenames.
    Example:
        directory = main_indicators_etf_1h
        ending    = "_main_indicators_etf_1h.csv"
    """
    pattern = os.path.join(directory, f"*{ending}")
    files = glob.glob(pattern)
    tickers = set()
    for f in files:
        base = os.path.basename(f)
        if base.endswith(ending):
            ticker = base[: -len(ending)]
            if ticker:
                tickers.add(ticker)
    return tickers


# ======================================================================
#                 COMMON MTF LOADER (1H + DAILY + WEEKLY)
# ======================================================================
def _load_mtf_context(ticker: str) -> pd.DataFrame:
    """
    Load 1H, Daily, Weekly CSVs for a ticker, suffix columns
    (_H, _D, _W) and merge Daily + Weekly onto 1H via as-of joins.

    Returns:
        df_hdw: 1H-based dataframe with *_H, *_D, *_W columns.
                If anything fails or data is missing, returns empty DF.
    """
    path_etf_1h = os.path.join(DIR_etf_1h, f"{ticker}_main_indicators_etf_1h.csv")
    path_d      = os.path.join(DIR_D,      f"{ticker}_main_indicators_etf_daily.csv")
    path_w      = os.path.join(DIR_W,      f"{ticker}_main_indicators_etf_weekly.csv")

    df_h = _safe_read_tf(path_etf_1h)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return pd.DataFrame()

    # Daily extras: previous-day Daily_Change (for context if needed)
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # Suffix all TFs
    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Merge Daily onto 1H (backward as-of)
    df_hd = pd.merge_asof(
        df_h.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward",
    )

    # Shift Weekly dates by +1s to avoid same-date lookahead, then merge
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] += pd.Timedelta(seconds=1)

    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_sorted,
        on="date",
        direction="backward",
    )

    return df_hdw


# ======================================================================
# STRATEGY 1:
#   Weekly Trend + Daily Pullback & Resume (Classic Trend-Follow)
# ======================================================================
def build_signals_trend_pullback(ticker: str) -> list[dict]:
    """
    STRATEGY 1: Weekly Trend + Daily Pullback & Resume

    WEEKLY:
        • Strong uptrend:
            - close_W > EMA_50_W > EMA_200_W
            - RSI_W > 50
            - ADX_W > 18

    DAILY:
        • Prior pullback then bullish resumption:
            - Daily close_D around 20_SMA_D / EMA_50_D (mild pullback zone)
            - Bullish candle (close_D > open_D)
            - Close in upper part of day's range (strong close)
            - RSI_D > 50
            - MACD_Hist_D rising vs previous day

    ENTRY:
        • Every 1H bar where weekly_ok & daily_ok are true
          becomes a LONG signal (entry_price = close_H).
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W",
        "close_D", "open_D", "high_D", "low_D",
        "20_SMA_D", "EMA_50_D", "RSI_D", "MACD_Hist_D",
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 1.")
        return []

    # -------- WEEKLY FILTER --------
    weekly_ok = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 50.0) &
        (df_hdw["ADX_W"] > 18.0)
    )

    # -------- DAILY FILTER (PULLBACK + RESUME) --------
    macd_prev = df_hdw["MACD_Hist_D"].shift(1)
    macd_rising = df_hdw["MACD_Hist_D"] >= macd_prev

    day_range = (df_hdw["high_D"] - df_hdw["low_D"]).replace(0, np.nan)
    range_pos = (df_hdw["close_D"] - df_hdw["low_D"]) / day_range  # 0 at low, 1 at high
    strong_close = range_pos >= 0.6

    pullback_zone = (
        (df_hdw["close_D"] >= df_hdw["20_SMA_D"] * 0.98) &
        (df_hdw["close_D"] <= df_hdw["EMA_50_D"]  * 1.02)
    )

    daily_ok = (
        (df_hdw["close_D"] > df_hdw["open_D"]) &
        pullback_zone &
        strong_close &
        (df_hdw["RSI_D"] > 50.0) &
        macd_rising
    )

    long_mask = weekly_ok & daily_ok

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,  # no hourly filters used
        })
    return signals


# ======================================================================
# STRATEGY 2:
#   Weekly Breakout + Daily Confirmation (Momentum Continuation)
# ======================================================================
def build_signals_weekly_breakout_daily_confirm(ticker: str) -> list[dict]:
    """
    STRATEGY 2: Weekly Breakout + Daily Confirmation

    WEEKLY:
        • Breakout above Recent_High_W
        • RSI_W > 55

    DAILY:
        • Daily close_D confirms breakout:
            - close_D > Recent_High_W (no failed breakout)
            - close_D > 20_SMA_D and > EMA_50_D
            - RSI_D > 55
            - MACD_Hist_D > 0

    ENTRY:
        • Every 1H bar where weekly_ok & daily_ok are true.
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        "close_W", "Recent_High_W", "RSI_W",
        "close_D", "EMA_50_D", "20_SMA_D", "RSI_D", "MACD_Hist_D",
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 2.")
        return []

    weekly_ok = (
        (df_hdw["close_W"] > df_hdw["Recent_High_W"]) &
        (df_hdw["RSI_W"] > 55.0)
    )

    daily_ok = (
        (df_hdw["close_D"] > df_hdw["Recent_High_W"]) &
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &
        (df_hdw["close_D"] > df_hdw["EMA_50_D"]) &
        (df_hdw["RSI_D"] > 55.0) &
        (df_hdw["MACD_Hist_D"] > 0.0)
    )

    long_mask = weekly_ok & daily_ok

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
        })
    return signals


# ======================================================================
# STRATEGY 3:
#   Weekly Squeeze + Daily Vol Expansion (TTM-style)
# ======================================================================
def build_signals_weekly_squeeze_daily_expansion(ticker: str) -> list[dict]:
    """
    STRATEGY 3: Weekly Volatility Squeeze + Daily Expansion

    WEEKLY:
        • Volatility compression / "squeeze":
            - Bollinger band width small (Upper_Band_W - Lower_Band_W) / close_W
        • Momentum bias:
            - RSI_W > 50
            - MACD_Hist_W rising

    DAILY:
        • Expansion:
            - Daily close_D near or above Upper_Band_D
            - ATR_D rising vs previous few days
            - RSI_D > 55
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        "close_W", "Upper_Band_W", "Lower_Band_W", "RSI_W", "MACD_Hist_W",
        "close_D", "Upper_Band_D", "ATR_D", "RSI_D",
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 3.")
        return []

    bb_width_W = (df_hdw["Upper_Band_W"] - df_hdw["Lower_Band_W"]) / df_hdw["close_W"].replace(0, np.nan)
    bb_squeeze = bb_width_W <= 0.12

    macd_prev_W = df_hdw["MACD_Hist_W"].shift(1)
    macd_rising_W = df_hdw["MACD_Hist_W"] >= macd_prev_W

    weekly_ok = (
        bb_squeeze &
        (df_hdw["RSI_W"] > 50.0) &
        macd_rising_W
    )

    atr_prev1 = df_hdw["ATR_D"].shift(1)
    atr_prev2 = df_hdw["ATR_D"].shift(2)
    atr_rising = (df_hdw["ATR_D"] >= atr_prev1) & (atr_prev1 >= atr_prev2)

    daily_ok = (
        (df_hdw["close_D"] >= df_hdw["Upper_Band_D"] * 0.995) &
        atr_rising &
        (df_hdw["RSI_D"] > 55.0)
    )

    long_mask = weekly_ok & daily_ok

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
        })
    return signals


# ======================================================================
# STRATEGY 4:
#   Weekly Trend + Daily Inside Bar Breakout
# ======================================================================
def build_signals_weekly_trend_insidebar_breakout(ticker: str) -> list[dict]:
    """
    STRATEGY 4: Weekly Trend + Daily Inside Bar Breakout

    WEEKLY:
        • 20_SMA_W > EMA_50_W > EMA_200_W (20_SMA_W ~ EMA_20_W proxy)
        • ADX_W > 20
        • RSI_W > 50

    DAILY:
        • Inside bar then breakout:
            - (high_D < high_D_prev) & (low_D > low_D_prev)
            - close_D > high_D_prev
        • Confirmation:
            - close_D > 20_SMA_D
            - Stoch_%K_D > Stoch_%D_D
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W", "20_SMA_W",
        "high_D", "low_D", "close_D", "20_SMA_D", "Stoch_%K_D", "Stoch_%D_D",
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 4.")
        return []

    weekly_ok = (
        (df_hdw["20_SMA_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["ADX_W"] > 20.0) &
        (df_hdw["RSI_W"] > 50.0)
    )

    high_prev = df_hdw["high_D"].shift(1)
    low_prev  = df_hdw["low_D"].shift(1)

    inside_bar = (
        (df_hdw["high_D"] < high_prev) &
        (df_hdw["low_D"]  > low_prev)
    )

    breakout = df_hdw["close_D"] > high_prev
    stoch_cross = df_hdw["Stoch_%K_D"] > df_hdw["Stoch_%D_D"]

    daily_ok = (
        inside_bar &
        breakout &
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &
        stoch_cross
    )

    long_mask = weekly_ok & daily_ok

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
        })
    return signals


# ======================================================================
# STRATEGY 5:
#   Weekly Higher Highs + Daily Retest
# ======================================================================
def build_signals_weekly_hh_hl_retest(ticker: str) -> list[dict]:
    """
    STRATEGY 5: Weekly Higher Highs + Daily Retest

    WEEKLY:
        • Approximate HH/HL behaviour:
            - close_W > 20_SMA_W
            - RSI_W > 55
            - ADX_W > 15

    DAILY:
        • Retest of breakout zone (Recent_High_W proxy):
            - low_D <= Recent_High_W (retest)
            - close_D > Recent_High_W (reclaim)
            - close_D > open_D (bullish candle)
            - MACD_Hist_D rising vs previous day
            - MFI_D > 50
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        "close_W", "20_SMA_W", "RSI_W", "ADX_W", "Recent_High_W",
        "open_D", "low_D", "close_D", "MACD_Hist_D", "MFI_D",
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 5.")
        return []

    weekly_ok = (
        (df_hdw["close_W"] > df_hdw["20_SMA_W"]) &
        (df_hdw["RSI_W"] > 55.0) &
        (df_hdw["ADX_W"] > 15.0)
    )

    macd_prev = df_hdw["MACD_Hist_D"].shift(1)
    macd_rising = df_hdw["MACD_Hist_D"] >= macd_prev

    daily_ok = (
        (df_hdw["low_D"] <= df_hdw["Recent_High_W"]) &
        (df_hdw["close_D"] > df_hdw["Recent_High_W"]) &
        (df_hdw["close_D"] > df_hdw["open_D"]) &
        macd_rising &
        (df_hdw["MFI_D"] > 50.0)
    )

    long_mask = weekly_ok & daily_ok

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
        })
    return signals


# ======================================================================
# STRATEGY 6:
#   Weekly Golden Cross + Daily EMA(ish) Bounce
# ======================================================================
def build_signals_weekly_golden_cross_ema_bounce(ticker: str) -> list[dict]:
    """
    STRATEGY 6: Weekly Golden Cross + Daily EMA-Bounce Style Entry

    WEEKLY:
        • Golden cross regime:
            - EMA_50_W > EMA_200_W
            - RSI_W > 50

    DAILY:
        • Pullback to MA and bounce:
            - low_D <= 20_SMA_D or EMA_50_D (touch / undercut)
            - close_D > 20_SMA_D
            - close_D > open_D (bullish candle)
            - RSI_D > 50
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        "EMA_50_W", "EMA_200_W", "RSI_W",
        "low_D", "close_D", "open_D", "20_SMA_D", "EMA_50_D", "RSI_D",
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 6.")
        return []

    weekly_ok = (
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 50.0)
    )

    touch_ma = (
        (df_hdw["low_D"] <= df_hdw["20_SMA_D"]) |
        (df_hdw["low_D"] <= df_hdw["EMA_50_D"])
    )

    daily_ok = (
        touch_ma &
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &
        (df_hdw["close_D"] > df_hdw["open_D"]) &
        (df_hdw["RSI_D"] > 50.0)
    )

    long_mask = weekly_ok & daily_ok

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
        })
    return signals


# ======================================================================
# CUSTOM STRATEGY 1:
#   Rich weekly regime + relaxed previous-day daily filter (no hourly)
# ======================================================================
def build_signals_custom1(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 1

    Quality-focused MTF strategy using only WEEKLY + previous-day DAILY:

    WEEKLY (regime filter – slightly stricter & richer):
        Uses weekly:
            - Trend structure: close vs EMA_50_W / EMA_200_W
            - Momentum: RSI_W, MACD_Hist_W
            - Trend strength: ADX_W
            - Money flow: MFI_W
            - Volatility regime: ATR_W as % of price
            - Price location: vs 20_SMA_W / Bands / Recent_High_W
            - Stochastic: %K_W vs %D_W

    DAILY (setup filter – slightly relaxed):
        Uses previous day's daily context:
            - Daily_Change_prev_D positive band
            - Trend vs EMAs (softened)
            - RSI_D 'healthy bullish' with wider bounds
            - MACD_Hist_D can be mildly negative
            - MFI_D, ADX_D floors relaxed
            - ATR% regime widened
            - Bollinger position, Recent_High_D, price-action & Stoch filters

    ENTRY:
        • Every 1H bar where WEEKLY and DAILY are true (no hourly indicators).
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    # Need Weekly regime columns + all Daily columns used in filters.
    required_cols = [
        # Weekly regime (trend, momentum, mf, vol, structure)
        "close_W", "open_W", "high_W", "low_W",
        "EMA_50_W", "EMA_200_W",
        "RSI_W", "ADX_W", "MACD_Hist_W", "MFI_W", "ATR_W",
        "20_SMA_W", "Upper_Band_W", "Lower_Band_W",
        "Recent_High_W", "Stoch_%K_W", "Stoch_%D_W",
        # Daily regime / setup
        "close_D", "open_D", "high_D", "low_D",
        "EMA_50_D", "EMA_200_D",
        "RSI_D", "MACD_Hist_D", "MFI_D", "ADX_D", "ATR_D",
        "20_SMA_D", "Upper_Band_D", "Lower_Band_D",
        "Recent_High_D", "Stoch_%K_D", "Stoch_%D_D",
        "Daily_Change_prev_D",
        # Entry price
        "close_H",
    ]
    if not set(required_cols).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Custom Strategy 1.")
        return []

    df_hdw = df_hdw.dropna(subset=required_cols)
    if df_hdw.empty:
        return []

    signals: list[dict] = []

    # --- Basic weekly trend block (strong / mild) ---
    weekly_strong = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"] * 1.005) &   # clearly above EMA_50_W
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 42.0) &                           # slightly higher floor
        (df_hdw["ADX_W"] > 12.0)                             # stronger trend
    )

    weekly_mild = (
        (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.015) &      # more clearly above EMA_200_W
        (df_hdw["close_W"] >= df_hdw["EMA_50_W"] * 0.99) &       # not far below EMA_50_W
        (df_hdw["RSI_W"].between(42.0, 66.0)) &                  # tighter RSI band
        (df_hdw["ADX_W"] > 10.0)                                 # slightly stronger floor
    )

    # 1) Weekly MACD histogram: avoid clearly bearish regimes
    weekly_macd_ok = df_hdw["MACD_Hist_W"] >= -0.02

    # 2) Weekly MFI: confirm participation
    weekly_mfi_ok = df_hdw["MFI_W"] > 45.0

    # 3) Weekly ATR%: volatility regime
    close_W = df_hdw["close_W"]
    atr_pct_W = (df_hdw["ATR_W"] / close_W.replace(0, np.nan)) * 100.0
    weekly_atr_ok = (atr_pct_W >= 0.4) & (atr_pct_W <= 10.0)

    # 4) Weekly Bollinger / 20-SMA structure
    mid_W   = df_hdw["20_SMA_W"]
    upper_W = df_hdw["Upper_Band_W"]
    lower_W = df_hdw["Lower_Band_W"]

    weekly_bb_pos_ok = (
        (close_W >= mid_W * 0.995) &
        (close_W <= upper_W * 1.03) &
        (close_W >= lower_W * 0.99)
    )

    # 5) Weekly position vs Recent_High_W
    recent_high_W = df_hdw["Recent_High_W"]
    weekly_recent_high_ok = close_W >= (recent_high_W * 0.94)

    # 6) Weekly Stochastic
    stoch_k_W = df_hdw["Stoch_%K_W"]
    stoch_d_W = df_hdw["Stoch_%D_W"]
    weekly_stoch_ok = (
        (stoch_k_W >= stoch_d_W * 0.98) &
        (stoch_k_W >= 35.0) &
        (stoch_k_W <= 88.0)
    )

    # 7) Weekly RSI extreme cutoff
    weekly_not_extreme = df_hdw["RSI_W"] < 85.0

    weekly_ok = (
        (weekly_strong | weekly_mild) &
        weekly_macd_ok &
        weekly_mfi_ok &
        weekly_atr_ok &
        weekly_bb_pos_ok &
        weekly_recent_high_ok &
        weekly_stoch_ok &
        weekly_not_extreme
    )

    # ---------- DAILY SETUP FILTER (LONG) – RELAXED ----------
    daily_prev = df_hdw["Daily_Change_prev_D"]
    daily_move_ok = (daily_prev >= 1.5) & (daily_prev <= 10.0)

    daily_trend_ok = (
        (df_hdw["close_D"] >= df_hdw["EMA_50_D"] * 0.995) &
        (df_hdw["EMA_50_D"] >= df_hdw["EMA_200_D"] * 0.995)
    )

    daily_rsi_ok = df_hdw["RSI_D"].between(42.0, 72.0)
    daily_macd_ok = df_hdw["MACD_Hist_D"] >= -0.02
    daily_mfi_ok = df_hdw["MFI_D"] > 45.0
    daily_adx_ok = df_hdw["ADX_D"] > 10.0

    close_D = df_hdw["close_D"]
    atr_pct_D = (df_hdw["ATR_D"] / close_D.replace(0, np.nan)) * 100.0
    atr_vol_ok = (atr_pct_D >= 0.3) & (atr_pct_D <= 7.0)

    mid_D   = df_hdw["20_SMA_D"]
    upper_D = df_hdw["Upper_Band_D"]
    lower_D = df_hdw["Lower_Band_D"]

    bb_pos_ok = (
        (close_D >= mid_D * 0.99) &
        (close_D <= upper_D * 1.04) &
        (close_D >= lower_D * 0.99)
    )

    recent_high_D = df_hdw["Recent_High_D"]
    recent_high_ok = close_D >= (recent_high_D * 0.95)

    high_D = df_hdw["high_D"]
    low_D  = df_hdw["low_D"]
    day_range = (high_D - low_D).replace(0, np.nan)
    range_pos = (close_D - low_D) / day_range
    pa_upper_close_ok = range_pos >= 0.55

    stoch_k_D = df_hdw["Stoch_%K_D"]
    stoch_d_D = df_hdw["Stoch_%D_D"]
    stoch_daily_ok = (
        (stoch_k_D >= stoch_d_D * 0.98) &
        (stoch_k_D >= 35.0) &
        (stoch_k_D <= 90.0)
    )

    daily_ok = (
        daily_move_ok &
        daily_trend_ok &
        daily_rsi_ok &
        daily_macd_ok &
        daily_mfi_ok &
        daily_adx_ok &
        atr_vol_ok &
        bb_pos_ok &
        recent_high_ok &
        pa_upper_close_ok &
        stoch_daily_ok
    )

    long_mask = weekly_ok & daily_ok

    for _, r in df_hdw[long_mask].iterrows():
        signals.append(
            {
                "signal_time_ist": r["date"],
                "ticker": ticker,
                "signal_side": "LONG",
                "entry_price": float(r["close_H"]),  # 1H close used as entry
                "weekly_ok": True,
                "daily_ok": True,
                "hourly_ok": True,  # no hourly filters used
            }
        )

    return signals


# ======================================================================
# CUSTOM STRATEGY 2:
#   Relaxed WEEKLY + prev-day DAILY + *looser* HOURLY trigger
# ======================================================================
def build_signals_custom2(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 2

    For a single ticker:
        • Read 1H, Daily, Weekly indicator CSVs
        • Compute previous-day Daily_Change on daily TF
        • Merge them into a single 1H-based dataframe
        • Apply relaxed Weekly filter + even looser Hourly trigger
        • Return a list of LONG signal dictionaries
    """
    path_etf_1h = os.path.join(DIR_etf_1h, f"{ticker}_main_indicators_etf_1h.csv")
    path_d      = os.path.join(DIR_D,      f"{ticker}_main_indicators_etf_daily.csv")
    path_w      = os.path.join(DIR_W,      f"{ticker}_main_indicators_etf_weekly.csv")

    df_h = _safe_read_tf(path_etf_1h)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # --- Daily: compute previous-day Daily_Change before suffixing ---
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # --- Prepare timeframe-suffixed dataframes ---
    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # --- Hourly extras ---
    if "ATR_H" in df_h.columns:
        df_h["ATR_H_5bar_mean"] = df_h["ATR_H"].rolling(5, min_periods=5).mean()
    else:
        df_h["ATR_H_5bar_mean"] = np.nan

    if "RSI_H" in df_h.columns:
        df_h["RSI_H_prev"] = df_h["RSI_H"].shift(1)
    else:
        df_h["RSI_H_prev"] = np.nan

    if "MACD_Hist_H" in df_h.columns:
        df_h["MACD_Hist_H_prev"] = df_h["MACD_Hist_H"].shift(1)
    else:
        df_h["MACD_Hist_H_prev"] = np.nan

    if "high_H" in df_h.columns:
        df_h["Recent_High_5_H"] = df_h["high_H"].rolling(5, min_periods=5).max()
    else:
        df_h["Recent_High_5_H"] = np.nan

    if "ADX_H" in df_h.columns:
        df_h["ADX_H_5bar_mean"] = df_h["ADX_H"].rolling(5, min_periods=5).mean()
    else:
        df_h["ADX_H_5bar_mean"] = np.nan

    if "Stoch_%K_H" in df_h.columns:
        df_h["Stoch_%K_H_prev"] = df_h["Stoch_%K_H"].shift(1)
    else:
        df_h["Stoch_%K_H_prev"] = np.nan

    # --- Merge Daily + Weekly onto Hourly using as-of (backward) joins --->
    df_h_sorted = df_h.sort_values("date")
    df_d_sorted = df_d.sort_values("date")

    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(seconds=1)

    df_hd = pd.merge_asof(
        df_h_sorted,
        df_d_sorted,
        on="date",
        direction="backward"
    )

    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_sorted,
        on="date",
        direction="backward"
    )

    df_hdw = df_hdw.dropna(
        subset=[
            "close_W",
            "EMA_50_W",
            "EMA_200_W",
            "RSI_W",
            "ADX_W",
            "Daily_Change_prev_D",
        ]
    ).reset_index(drop=True)

    if df_hdw.empty:
        print(f"- After MTF merge, no usable rows for {ticker}.")
        return []

    signals: list[dict] = []

    # ========== WEEKLY FILTER (RELAXED) ==========
    weekly_strong_up = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 46.0) &
        (df_hdw["ADX_W"] > 11.0)
    )

    weekly_mild_up = (
        (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.01) &
        (df_hdw["RSI_W"].between(44.0, 65.0)) &
        (df_hdw["ADX_W"] > 9.0)
    )

    weekly_long_ok = weekly_strong_up | weekly_mild_up

    # ========== DAILY FILTER ==========
    daily_change_prev = df_hdw["Daily_Change_prev_D"]
    daily_long_ok = daily_change_prev >= 2.0

    # ========== HOURLY TRIGGER (RELAXED FURTHER) ==========

    hourly_trend_ok = (
        (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &
        (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)
    )

    rsi_trend_ok = (
        (df_hdw["RSI_H"] > 49.0) |
        ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
    )

    macd_trend_ok = (
        (df_hdw["MACD_Hist_H"] > -0.06) &
        (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.85)
    )

    stoch_trend_ok = df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98

    hourly_momentum_ok = rsi_trend_ok & macd_trend_ok & stoch_trend_ok

    hourly_vol_ok = (
        (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.92) &
        (
            (df_hdw["ADX_H"] >= 11.0) |
            (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"])
        )
    )

    price_above_vwap = df_hdw["close_H"] >= df_hdw["VWAP_H"]

    breakout_strict = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]
    breakout_near   = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.995

    intra_change_h = df_hdw.get("Intra_Change_H", pd.Series(0.0, index=df_hdw.index)).fillna(0.0)
    strong_up_move = intra_change_h >= 0.2

    hourly_breakout_ok = price_above_vwap & (
        breakout_strict |
        (breakout_near & strong_up_move)
    )

    hourly_long_ok = (
        hourly_trend_ok &
        hourly_momentum_ok &
        hourly_vol_ok &
        hourly_breakout_ok
    )

    long_mask = weekly_long_ok & daily_long_ok & hourly_long_ok

    for _, r in df_hdw[long_mask].iterrows():
        signals.append(
            {
                "signal_time_ist": r["date"],
                "ticker": ticker,
                "signal_side": "LONG",
                "entry_price": float(r["close_H"]),
                "weekly_ok": True,
                "daily_ok": True,
                "hourly_ok": True,
            }
        )

    return signals


# ======================================================================
# CUSTOM STRATEGY 3:
#   Mildly-strict relaxed MTF with special D-1 daily merge
# ======================================================================
def build_signals_custom3(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 3

    For a single ticker:
        • Read 1H, Daily, Weekly indicator CSVs
        • Merge them into a single 1H-based dataframe (Daily shifted by +1 day
          so each intraday bar on D sees daily context from D-1).
        • Apply the mildly-strict relaxed multi-timeframe LONG strategy
        • Return a list of LONG signal dictionaries
    """
    path_etf_1h = os.path.join(DIR_etf_1h, f"{ticker}_main_indicators_etf_1h.csv")
    path_d  = os.path.join(DIR_D,  f"{ticker}_main_indicators_etf_daily.csv")
    path_w  = os.path.join(DIR_W,  f"{ticker}_main_indicators_etf_weekly.csv")

    df_h = _safe_read_tf(path_etf_1h)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # --- Prepare timeframe-suffixed dataframes ---
    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # --- Daily extras: 5-day ATR mean, 10-bar Recent High (from daily high) ---
    if "ATR_D" in df_d.columns:
        df_d["ATR_D_5d_mean"] = df_d["ATR_D"].rolling(5, min_periods=5).mean()
    else:
        df_d["ATR_D_5d_mean"] = np.nan

    if "high_D" in df_d.columns:
        df_d["Recent_High_10_D"] = df_d["high_D"].rolling(10, min_periods=10).max()
    else:
        df_d["Recent_High_10_D"] = np.nan

    # --- Hourly extras: 5-bar ATR mean, helpers, 5-bar Recent High ---
    if "ATR_H" in df_h.columns:
        df_h["ATR_H_5bar_mean"] = df_h["ATR_H"].rolling(5, min_periods=5).mean()
    else:
        df_h["ATR_H_5bar_mean"] = np.nan

    if "RSI_H" in df_h.columns:
        df_h["RSI_H_prev"] = df_h["RSI_H"].shift(1)
    else:
        df_h["RSI_H_prev"] = np.nan

    if "MACD_Hist_H" in df_h.columns:
        df_h["MACD_Hist_H_prev"] = df_h["MACD_Hist_H"].shift(1)
    else:
        df_h["MACD_Hist_H_prev"] = np.nan

    if "high_H" in df_h.columns:
        df_h["Recent_High_5_H"] = df_h["high_H"].rolling(5, min_periods=5).max()
    else:
        df_h["Recent_High_5_H"] = np.nan

    # --- Merge Daily + Weekly onto Hourly using as-of (backward) joins ---
    df_h_sorted = df_h.sort_values("date")

    # DAILY: shift 'date' forward by +1 day for merge so that
    # hourly bars on calendar day D see daily data from D-1.
    df_d_sorted = df_d.sort_values("date").copy()
    df_d_for_merge = df_d_sorted.copy()
    df_d_for_merge["date"] = df_d_for_merge["date"] + pd.Timedelta(days=1)

    # WEEKLY: shift 'date' forward by +1 second for merge so that
    # hourly bars on the same calendar date as a weekly bar still
    # use the previous week's bar (no same-date weekly lookahead).
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_for_merge = df_w_sorted.copy()
    df_w_for_merge["date"] = df_w_for_merge["date"] + pd.Timedelta(seconds=1)

    # Merge daily into hourly (using shifted daily dates)
    df_hd = pd.merge_asof(
        df_h_sorted,
        df_d_for_merge,
        on="date",
        direction="backward"
    )

    # Merge weekly into (hourly+daily) (using shifted weekly dates)
    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_for_merge,
        on="date",
        direction="backward"
    )

    # Drop rows where we don't yet have valid Daily/Weekly context
    df_hdw = df_hdw.dropna(
        subset=[
            "close_W",
            "EMA_50_W",
            "EMA_200_W",
            "RSI_W",
            "ADX_W",
            "close_D",
            "EMA_50_D",
            "RSI_D",
            "MACD_Hist_D",
            "ATR_D",
            "ATR_D_5d_mean",
            "Recent_High_10_D",
        ]
    ).reset_index(drop=True)

    if df_hdw.empty:
        print(f"- After MTF merge, no usable rows for {ticker}.")
        return []

    signals: list[dict] = []

    # WEEKLY FILTER (trend, mildly strict)
    weekly_ok = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 48.0) &
        (df_hdw["ADX_W"] > 15.0)
    )

    # DAILY FILTER (setup, mildly strict)
    daily_close_vs_high = df_hdw["close_D"] >= (df_hdw["Recent_High_10_D"] * 0.985)
    daily_atr_expansion = df_hdw["ATR_D"] >= (df_hdw["ATR_D_5d_mean"] * 0.93)

    daily_ok = (
        (df_hdw["close_D"] > df_hdw["EMA_50_D"]) &
        (df_hdw["RSI_D"] > 48.0) &
        (df_hdw["MACD_Hist_D"] >= 0.0) &
        daily_atr_expansion &
        daily_close_vs_high
    )

    # HOURLY TRIGGER (entry timing, mildly strict)
    hourly_atr_expansion = df_hdw["ATR_H"] >= (df_hdw["ATR_H_5bar_mean"] * 0.93)
    hourly_recent_high_break = df_hdw["close_H"] >= (df_hdw["Recent_High_5_H"] * 0.998)

    hourly_ok = (
        (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
        (df_hdw["RSI_H"] > 49.0) &
        (df_hdw["MACD_Hist_H"] > 0.0) &
        hourly_atr_expansion &
        hourly_recent_high_break
    )

    final_mask = weekly_ok & daily_ok & hourly_ok

    sig_rows = df_hdw[final_mask].copy()
    if sig_rows.empty:
        return []

    for _, r in sig_rows.iterrows():
        signals.append(
            {
                "signal_time_ist": r["date"],
                "ticker": ticker,
                "signal_side": "LONG",
                "entry_price": float(r["close_H"]),
                "weekly_ok": True,
                "daily_ok": True,
                "hourly_ok": True,
            }
        )

    return signals


# ======================================================================
# CUSTOM STRATEGY 4:
#   Slightly-loosened quality-focused MTF (small relaxations)
# ======================================================================
def build_signals_custom4(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 4

    Slightly-loosened quality-focused MTF strategy.
    Only minimal relaxations added to increase signal count
    while keeping overall P&L quality stable.
    """
    path_etf_1h = os.path.join(DIR_etf_1h, f"{ticker}_main_indicators_etf_1h.csv")
    path_d      = os.path.join(DIR_D,      f"{ticker}_main_indicators_etf_daily.csv")
    path_w      = os.path.join(DIR_W,      f"{ticker}_main_indicators_etf_weekly.csv")

    df_h = _safe_read_tf(path_etf_1h)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # --- Daily: compute previous-day Daily_Change ---
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # --- Hourly extras ---
    if "ATR_H" in df_h.columns:
        df_h["ATR_H_5bar_mean"] = df_h["ATR_H"].rolling(5, min_periods=5).mean()
    else:
        df_h["ATR_H_5bar_mean"] = np.nan

    if "RSI_H" in df_h.columns:
        df_h["RSI_H_prev"] = df_h["RSI_H"].shift(1)
    else:
        df_h["RSI_H_prev"] = np.nan

    if "MACD_Hist_H" in df_h.columns:
        df_h["MACD_Hist_H_prev"] = df_h["MACD_Hist_H"].shift(1)
    else:
        df_h["MACD_Hist_H_prev"] = np.nan

    if "high_H" in df_h.columns:
        df_h["Recent_High_5_H"] = df_h["high_H"].rolling(5, min_periods=5).max()
    else:
        df_h["Recent_High_5_H"] = np.nan

    if "ADX_H" in df_h.columns:
        df_h["ADX_H_5bar_mean"] = df_h["ADX_H"].rolling(5, min_periods=5).mean()
    else:
        df_h["ADX_H_5bar_mean"] = np.nan

    if "Stoch_%K_H" in df_h.columns:
        df_h["Stoch_%K_H_prev"] = df_h["Stoch_%K_H"].shift(1)
    else:
        df_h["Stoch_%K_H_prev"] = np.nan

    # Merge D + W onto H
    df_hd = pd.merge_asof(
        df_h.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward"
    )

    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] += pd.Timedelta(seconds=1)

    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_sorted,
        on="date",
        direction="backward"
    )

    df_hdw = df_hdw.dropna(subset=[
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W", "Daily_Change_prev_D"
    ])

    if df_hdw.empty:
        return []

    signals: list[dict] = []

    # -------------------- WEEKLY (very lightly relaxed) --------------------
    weekly_strong = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 40.0) &
        (df_hdw["ADX_W"] > 10.0)
    )

    weekly_mild = (
        (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.01) &
        (df_hdw["RSI_W"].between(40.0, 68.0)) &
        (df_hdw["ADX_W"] > 9.0)
    )

    weekly_not_extreme = df_hdw["RSI_W"] < 88.0  # was 85 → slightly looser

    weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme

    # -------------------- DAILY (slightly expanded range) --------------------
    daily_prev = df_hdw["Daily_Change_prev_D"]
    daily_ok = (daily_prev >= 2.0) & (daily_prev <= 9.0)  # was 10% cap

    # -------------------- HOURLY (tiny relaxations) --------------------
    hourly_trend = (
        (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &
        (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)
    )

    rsi_ok = (
        (df_hdw["RSI_H"] > 49.0) |
        ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
    )

    macd_ok = (
        (df_hdw["MACD_Hist_H"] > -0.055) &      # was -0.045
        (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.75)
    )

    stoch_ok = (
        (df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98) &
        (df_hdw["Stoch_%K_H"] >= 18.0)           # was 20
    )

    momentum_ok = rsi_ok & macd_ok & stoch_ok

    hourly_vol_ok = (
        (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.92) &
        ((df_hdw["ADX_H"] >= 10.0) | (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"]))
    )

    price_above_vwap = df_hdw["close_H"] >= df_hdw["VWAP_H"]

    breakout_strict = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]
    breakout_near   = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.99

    intra_chg = df_hdw.get("Intra_Change_H", pd.Series(0.0, index=df_hdw.index)).fillna(0.0)
    strong_up = intra_chg >= 0.05

    breakout_ok = price_above_vwap & (
        breakout_strict |
        (breakout_near & strong_up & (df_hdw["RSI_H"] >= 47.0))  # was 48 → now slightly relaxed
    )

    hourly_ok = hourly_trend & momentum_ok & hourly_vol_ok & breakout_ok

    long_mask = weekly_ok & daily_ok & hourly_ok

    for _, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
        })

    return signals


# ======================================================================
# STRATEGY DISPATCHER
# ======================================================================
def build_signals_for_ticker(ticker: str, strategy: str = "trend_pullback") -> list[dict]:
    """
    Dispatch to one of the strategy-specific builders.

    strategy options (case-insensitive):
        "trend_pullback"
        "weekly_breakout"
        "weekly_squeeze"
        "insidebar_breakout"
        "hh_hl_retest"
        "golden_cross"
        "custom1"
        "custom2"
        "custom3"
        "custom4"
    """
    s = strategy.lower()

    if s == "trend_pullback":
        return build_signals_trend_pullback(ticker)
    elif s == "weekly_breakout":
        return build_signals_weekly_breakout_daily_confirm(ticker)
    elif s == "weekly_squeeze":
        return build_signals_weekly_squeeze_daily_expansion(ticker)
    elif s == "insidebar_breakout":
        return build_signals_weekly_trend_insidebar_breakout(ticker)
    elif s == "hh_hl_retest":
        return build_signals_weekly_hh_hl_retest(ticker)
    elif s == "golden_cross":
        return build_signals_weekly_golden_cross_ema_bounce(ticker)
    elif s == "custom1":
        return build_signals_custom1(ticker)
    elif s == "custom2":
        return build_signals_custom2(ticker)
    elif s == "custom3":
        return build_signals_custom3(ticker)
    elif s == "custom4":
        return build_signals_custom4(ticker)
    else:
        print(f"! Unknown strategy '{strategy}', defaulting to 'trend_pullback'.")
        return build_signals_trend_pullback(ticker)


# ======================================================================
# USER INPUT FOR STRATEGY
# ======================================================================
def choose_strategy_from_input() -> str:
    """
    Simple interactive menu to choose which strategy to run.

    Returns:
        strategy key string compatible with build_signals_for_ticker.
    """
    options = {
        "1": "trend_pullback",
        "2": "weekly_breakout",
        "3": "weekly_squeeze",
        "4": "insidebar_breakout",
        "5": "hh_hl_retest",
        "6": "golden_cross",
        "7": "custom1",
        "8": "custom2",
        "9": "custom3",
        "10": "custom4",
    }

    print("\n=== Select Strategy for LONG Signal Generation ===")
    print(" 1) trend_pullback        - Weekly trend + daily pullback & resume")
    print(" 2) weekly_breakout       - Weekly breakout + daily confirmation")
    print(" 3) weekly_squeeze        - Weekly squeeze + daily expansion")
    print(" 4) insidebar_breakout    - Weekly trend + daily inside bar breakout")
    print(" 5) hh_hl_retest          - Weekly HH/HL + daily retest")
    print(" 6) golden_cross          - Weekly golden cross + daily EMA bounce")
    print(" 7) custom1               - Rich weekly regime + relaxed prev-day daily")
    print(" 8) custom2               - Relaxed weekly + looser hourly trigger")
    print(" 9) custom3               - Mildly-strict relaxed MTF (D-1 merge)")
    print("10) custom4               - Slightly loosened quality-focused MTF")
    print("--------------------------------------------------")
    raw = input("Enter choice (1-10 or strategy name): ").strip().lower()

    if raw in options:
        strategy = options[raw]
    else:
        strategy = raw if raw else "trend_pullback"

    print(f"\n>> Using strategy: {strategy}\n")
    return strategy


# ----------------------- HELPERS FOR TARGET EVAL -----------------------
def _safe_read_etf_1h(path: str) -> pd.DataFrame:
    """Read 1H CSV, parse 'date' as tz-aware IST, sort."""
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(IST)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading 1H data {path}: {e}")
        return pd.DataFrame()


def evaluate_signal_on_etf_1h_long_only(row: pd.Series, df_etf_1h: pd.DataFrame) -> dict:
    """
    Evaluate +TARGET_PCT% outcome for LONG signals.
    """
    entry_time = row["signal_time_ist"]
    entry_price = float(row["entry_price"])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return {
            HIT_COL: False,
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "pnl_pct": np.nan,
            "pnl_rs": np.nan,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }

    future = df_etf_1h[df_etf_1h["date"] > entry_time].copy()
    if future.empty:
        return {
            HIT_COL: False,
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": entry_time,
            "exit_price": entry_price,
            "pnl_pct": 0.0,
            "pnl_rs": 0.0,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }

    target_price = entry_price * TARGET_MULT
    hit_mask = future["high"] >= target_price

    max_fav = ((future["high"] - entry_price) / entry_price * 100.0).max()
    max_adv = ((future["low"] - entry_price) / entry_price * 100.0).min()

    if hit_mask.any():
        first_hit_idx = hit_mask[hit_mask].index[0]
        hit_time = future.loc[first_hit_idx, "date"]

        exit_time = hit_time
        exit_price = target_price

        dt_hours = (hit_time - entry_time).total_seconds() / 3600.0
        dt_days = dt_hours / TRADING_HOURS_PER_DAY
        cal_days = (hit_time.date() - entry_time.date()).days

        hit_flag = True
        time_to_target_days = dt_days
        days_to_target_calendar = cal_days
    else:
        last_row = future.iloc[-1]
        exit_time = last_row["date"]
        exit_price = float(last_row["close"]) if "close" in last_row else entry_price

        hit_flag = False
        time_to_target_days = np.nan
        days_to_target_calendar = np.nan

    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    pnl_rs = CAPITAL_PER_SIGNAL_RS * (pnl_pct / 100.0)

    return {
        HIT_COL: hit_flag,
        TIME_TD_COL: time_to_target_days,
        TIME_CAL_COL: days_to_target_calendar,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "pnl_pct": float(pnl_pct),
        "pnl_rs": float(pnl_rs),
        "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
        "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
    }


# ----------------------- CAPITAL-CONSTRAINED PORTFOLIO SIM -----------------------
def apply_capital_constraint(out_df: pd.DataFrame) -> dict:
    """
    Apply a portfolio-level capital constraint:

        - Start with MAX_CAPITAL_RS of cash.
        - For each signal in chronological order:
            * Release any open trades whose exit_time <= entry_time.
            * If available_capital >= CAPITAL_PER_SIGNAL_RS -> take trade.
              Else -> skip trade.
    """
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    available_capital = MAX_CAPITAL_RS
    open_trades = []  # list of dicts: {"exit_time": ts, "value": float}

    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        # 1) Release trades that have exited before/at this entry_time
        still_open = []
        for trade in open_trades:
            exit_time = trade["exit_time"]
            if pd.isna(exit_time) or exit_time <= entry_time:
                available_capital += trade["value"]
            else:
                still_open.append(trade)
        open_trades = still_open

        # 2) Try to take this trade
        if available_capital >= CAPITAL_PER_SIGNAL_RS:
            df.at[idx, "taken"] = True
            available_capital -= CAPITAL_PER_SIGNAL_RS

            exit_time = row["exit_time"]
            if pd.isna(exit_time):
                exit_time = entry_time

            final_value = row["final_value_per_signal"]
            open_trades.append(
                {
                    "exit_time": exit_time,
                    "value": final_value,
                }
            )
        else:
            df.at[idx, "taken"] = False

    # 3) Release remaining open trades
    for trade in open_trades:
        available_capital += trade["value"]

    df["invested_amount_constrained"] = np.where(df["taken"], CAPITAL_PER_SIGNAL_RS, 0.0)
    df["final_value_constrained"] = np.where(df["taken"], df["final_value_per_signal"], 0.0)

    total_signals = len(df)
    signals_taken = int(df["taken"].sum())
    gross_invested = df["invested_amount_constrained"].sum()
    final_portfolio_value = available_capital
    net_pnl_rs = final_portfolio_value - MAX_CAPITAL_RS
    net_pnl_pct = (net_pnl_rs / MAX_CAPITAL_RS * 100.0) if MAX_CAPITAL_RS > 0 else 0.0

    summary = {
        "total_signals": total_signals,
        "signals_taken": signals_taken,
        "gross_invested": gross_invested,
        "final_portfolio_value": final_portfolio_value,
        "net_pnl_rs": net_pnl_rs,
        "net_pnl_pct": net_pnl_pct,
    }

    return df, summary


# ----------------------- MAIN -----------------------
def main(strategy: str):
    # --------- PART 1: GENERATE LONG SIGNALS ---------
    tickers_etf_1h = _list_tickers_from_dir(DIR_etf_1h, "_main_indicators_etf_1h.csv")
    tickers_d      = _list_tickers_from_dir(DIR_D,      "_main_indicators_etf_daily.csv")
    tickers_w      = _list_tickers_from_dir(DIR_W,      "_main_indicators_etf_weekly.csv")

    tickers = tickers_etf_1h & tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across 1H, Daily, and Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with all 3 timeframes.")
    print(f"Running strategy: {strategy}\n")

    all_signals: list[dict] = []

    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Processing {ticker} ...")
        try:
            ticker_signals = build_signals_for_ticker(ticker, strategy=strategy)
            all_signals.extend(ticker_signals)
            print(f"  -> {len(ticker_signals)} LONG signals for {ticker}")
        except Exception as e:
            print(f"! Error while processing {ticker}: {e}")

    if not all_signals:
        print("No LONG signals generated.")
        return

    sig_df = pd.DataFrame(all_signals)
    sig_df.sort_values(["signal_time_ist", "ticker"], inplace=True)

    # Add itr (1-based index) for each signal
    sig_df["itr"] = np.arange(1, len(sig_df) + 1)

    # Extract calendar date (IST) for per-day counts
    sig_df["signal_time_ist"] = sig_df["signal_time_ist"].dt.tz_convert(IST)
    sig_df["signal_date"] = sig_df["signal_time_ist"].dt.date

    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")

    # Save raw LONG signals with itr + signal_date
    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_{ts}_IST.csv")
    sig_df.to_csv(signals_path, index=False)

    print(f"\nSaved {len(sig_df)} multi-timeframe LONG signals to: {signals_path}")
    print("\nLast few LONG signals:")
    print(sig_df.tail(20).to_string(index=False))

    # --------- PART 1b: DAILY COUNTS (LONG ONLY) ---------
    print("\n--- Computing per-day LONG signal counts ---")

    daily_counts = (
        sig_df
        .groupby("signal_date")
        .size()
        .reset_index(name="LONG_signals")
    )

    daily_counts.rename(columns={"signal_date": "date"}, inplace=True)

    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_etf_daily_counts_{ts}_IST.csv")
    daily_counts.to_csv(daily_counts_path, index=False)

    print(f"\nSaved daily LONG signal counts to: {daily_counts_path}")
    print("\nDaily counts preview:")
    print(daily_counts.tail(20).to_string(index=False))

    # --------- PART 2: EVALUATE TARGET_PCT OUTCOME (LONG ONLY) ---------
    print(f"\n--- Evaluating +{TARGET_LABEL} outcome on LONG signals ---")
    print(f"Assuming capital per signal : ₹{CAPITAL_PER_SIGNAL_RS:,.2f}")

    total_signals = len(sig_df)
    print(f"Total LONG signals in file : {total_signals}")

    tickers_in_signals = sig_df["ticker"].unique().tolist()
    print(f"Found {len(tickers_in_signals)} tickers in generated LONG signals.")

    one_hour_data_cache: dict[str, pd.DataFrame] = {}
    results = []

    for t in tickers_in_signals:
        path_etf_1h = os.path.join(DIR_etf_1h, f"{t}_main_indicators_etf_1h.csv")
        df_etf_1h = _safe_read_etf_1h(path_etf_1h)
        if df_etf_1h.empty:
            print(f"- No 1H data for {t}, skipping its signals in evaluation.")
            one_hour_data_cache[t] = pd.DataFrame()
        else:
            one_hour_data_cache[t] = df_etf_1h

    for _, row in sig_df.iterrows():
        ticker = row["ticker"]
        df_etf_1h = one_hour_data_cache.get(ticker, pd.DataFrame())
        if df_etf_1h.empty:
            new_info = {
                HIT_COL: False,
                TIME_TD_COL: np.nan,
                TIME_CAL_COL: np.nan,
                "exit_time": row["signal_time_ist"],
                "exit_price": row["entry_price"],
                "pnl_pct": 0.0,
                "pnl_rs": 0.0,
                "max_favorable_pct": np.nan,
                "max_adverse_pct": np.nan,
            }
        else:
            new_info = evaluate_signal_on_etf_1h_long_only(row, df_etf_1h)

        merged_row = row.to_dict()
        merged_row.update(new_info)
        results.append(merged_row)

    if not results:
        print("No LONG signal evaluations produced.")
        return

    out_df = pd.DataFrame(results)
    out_df.sort_values(["signal_time_ist", "ticker"], inplace=True)

    # Per-signal invested and final value (UNCONSTRAINED)
    out_df["invested_amount"] = CAPITAL_PER_SIGNAL_RS
    out_df["final_value_per_signal"] = out_df["invested_amount"] * (1.0 + out_df["pnl_pct"] / 100.0)

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_{EVAL_SUFFIX}_{ts}_IST.csv")
    out_df.to_csv(eval_path, index=False)

    print(f"\nSaved {TARGET_LABEL} evaluation (LONG only) to: {eval_path}")

    total_long = len(out_df)

    # ---------- SUMMARY 1: ALL LONG ROWS (UNCONSTRAINED) ----------
    hits_long_all = out_df[HIT_COL].sum()
    hit_pct_long_all = (hits_long_all / total_long * 100.0) if total_long > 0 else 0.0
    print(
        f"\n[LONG SIGNALS - ALL] (unconstrained) Total evaluated: {total_long}, "
        f"{TARGET_LABEL} target hit: {hits_long_all} ({hit_pct_long_all:.1f}%)"
    )

    # ---------- SUMMARY 2: NON-RECENT LONG ROWS ----------
    if total_long > N_RECENT_SKIP:
        long_non_recent = out_df.iloc[:-N_RECENT_SKIP].copy()
        total_long_non_recent = len(long_non_recent)
        hits_long_non_recent = long_non_recent[HIT_COL].sum()
        hit_pct_long_non_recent = (
            hits_long_non_recent / total_long_non_recent * 100.0
            if total_long_non_recent > 0 else 0.0
        )

        print(f"\n[LONG SIGNALS - NON-RECENT] (unconstrained, excluding last {N_RECENT_SKIP} LONG signals)")
        print(
            f"Signals evaluated (non-recent LONG): {total_long_non_recent}, "
            f"{TARGET_LABEL} target hit: {hits_long_non_recent} ({hit_pct_long_non_recent:.1f}%)"
        )
    else:
        print(
            f"\n[LONG SIGNALS - NON-RECENT] Not enough LONG rows to exclude last "
            f"{N_RECENT_SKIP} signals; subset not computed."
        )

    # ---------- CAPITAL-CONSTRAINED SIMULATION ----------
    print("\n--- CAPITAL-CONSTRAINED PORTFOLIO SIMULATION ---")
    print(f"Max capital pool           : ₹{MAX_CAPITAL_RS:,.2f}")
    print(f"Capital per signal (fixed) : ₹{CAPITAL_PER_SIGNAL_RS:,.2f}")

    out_df_constrained, cap_summary = apply_capital_constraint(out_df)

    total_invested_constrained = cap_summary["gross_invested"]
    final_portfolio_value = cap_summary["final_portfolio_value"]
    net_pnl_rs = cap_summary["net_pnl_rs"]
    net_pnl_pct = cap_summary["net_pnl_pct"]

    print(f"Signals generated (total)  : {cap_summary['total_signals']}")
    print(f"Signals actually taken     : {cap_summary['signals_taken']}")
    print(f"Gross capital deployed     : ₹{total_invested_constrained:,.2f}")
    print(f"Final portfolio value      : ₹{final_portfolio_value:,.2f}")
    print(f"Net portfolio P&L          : ₹{net_pnl_rs:,.2f} ({net_pnl_pct:.2f}%)")

    # Overwrite eval file with constrained columns as well
    out_df_constrained.to_csv(eval_path, index=False)
    print(f"\nRe-saved evaluation with constrained-capital columns to: {eval_path}")

    # ---------- PREVIEW ----------
    print("\nLast few evaluated LONG signals (with capital constraint columns):")
    cols_preview = [
        "itr",
        "signal_time_ist",
        "signal_date",
        "ticker",
        "signal_side",
        "entry_price",
        "exit_time",
        "exit_price",
        "pnl_pct",
        "pnl_rs",
        HIT_COL,
        TIME_TD_COL,
        TIME_CAL_COL,
        "max_favorable_pct",
        "max_adverse_pct",
        "taken",
        "invested_amount_constrained",
        "final_value_constrained",
    ]
    print(out_df_constrained[cols_preview].tail(20).to_string(index=False))


if __name__ == "__main__":
    selected_strategy = choose_strategy_from_input()
    main(selected_strategy)
