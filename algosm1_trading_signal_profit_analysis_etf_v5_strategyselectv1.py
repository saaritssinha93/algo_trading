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
TARGET_PCT: float = 8  # <-- change this once (e.g. 4.0, 5.0, 7.0, 10.0)
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)  # used in names, assume integer percent
TARGET_LABEL: str = f"{TARGET_INT}%"  # for printing

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 50000.0   # fixed rupee amount per signal
MAX_CAPITAL_RS: float = 2000000.0        # total capital pool

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
#   Weekly Trend + Daily Pullback & Resume (slightly stricter)
# ======================================================================
def build_signals_trend_pullback(ticker: str) -> list[dict]:
    """
    STRATEGY 1: Weekly Trend + Daily Pullback & Resume (slightly stricter)

    WEEKLY:
        • Strong uptrend:
            - close_W > EMA_50_W > EMA_200_W
            - RSI_W > 52
            - ADX_W > 19
        • Prefer price fairly close to Recent_High_W (if available).

    DAILY:
        • Pullback then resumption:
            - close_D in a tighter 20_SMA_D / EMA_50_D pullback band
            - Bullish candle (close_D > open_D)
            - Close in upper ~35% of day's range
            - RSI_D > 52
            - MACD_Hist_D rising vs previous day
        • If Daily_Change_prev_D exists:
            - 0.7%–7.5% previous-day move.

    HOURLY (if available):
        • close_H > VWAP_H
        • RSI_H > 50

    ENTRY:
        • Every 1H bar where weekly_ok & daily_ok & hourly_ok are true.
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    base_required = [
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W",
        "close_D", "open_D", "high_D", "low_D",
        "20_SMA_D", "EMA_50_D", "RSI_D", "MACD_Hist_D",
        "close_H",
    ]
    if not set(base_required).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 1.")
        return []

    # -------- WEEKLY FILTER (slightly stricter) --------
    weekly_core_ok = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 52.0) &          # was 50
        (df_hdw["ADX_W"] > 19.0)            # was 18
    )

    if "Recent_High_W" in df_hdw.columns:
        weekly_loc_ok = df_hdw["close_W"] >= df_hdw["Recent_High_W"] * 0.93  # was 0.90
    else:
        weekly_loc_ok = True

    weekly_final_ok = weekly_core_ok & weekly_loc_ok

    # -------- DAILY FILTER (PULLBACK + RESUME) --------
    macd_prev = df_hdw["MACD_Hist_D"].shift(1)
    macd_rising = df_hdw["MACD_Hist_D"] >= macd_prev

    day_range = (df_hdw["high_D"] - df_hdw["low_D"]).replace(0, np.nan)
    range_pos = (df_hdw["close_D"] - df_hdw["low_D"]) / day_range
    strong_close = range_pos >= 0.65  # was 0.60

    # Slightly tighter pullback zone
    pullback_zone = (
        (df_hdw["close_D"] >= df_hdw["20_SMA_D"] * 0.985) &  # was 0.97
        (df_hdw["close_D"] <= df_hdw["EMA_50_D"] * 1.02)     # was 1.03
    )

    daily_core_ok = (
        (df_hdw["close_D"] > df_hdw["open_D"]) &
        pullback_zone &
        strong_close &
        (df_hdw["RSI_D"] > 52.0) &           # was 50
        macd_rising
    )

    # Optional Daily_Change_prev_D band (marginally tighter)
    if "Daily_Change_prev_D" in df_hdw.columns:
        dc_prev = df_hdw["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 0.7) & (dc_prev <= 7.5)   # was 0.5–8.0
    else:
        daily_move_ok = True

    daily_final_ok = daily_core_ok & daily_move_ok

    # -------- HOURLY GATE (if RSI_H / VWAP_H exist) --------
    if "VWAP_H" in df_hdw.columns and "RSI_H" in df_hdw.columns:
        hourly_mask = (
            (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
            (df_hdw["RSI_H"] > 50.0)          # was 48
        )
    else:
        # all True series aligned with df_hdw
        hourly_mask = pd.Series(True, index=df_hdw.index)

    long_mask = weekly_final_ok & daily_final_ok & hourly_mask

    signals: list[dict] = []
    for idx, r in df_hdw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": bool(weekly_final_ok.loc[idx]),
            "daily_ok": bool(daily_final_ok.loc[idx]),
            "hourly_ok": bool(hourly_mask.loc[idx]),
        })

    return signals



# ======================================================================
# STRATEGY 2:
#   Weekly Breakout + Daily Confirmation (relaxed so it actually fires)
# ======================================================================
def build_signals_weekly_breakout_daily_confirm(ticker: str) -> list[dict]:
    """
    STRATEGY 2: Weekly Breakout + Daily Confirmation

    WEEKLY:
        • Breakout regime:
            - close_W >= Recent_High_W * 1.005  (0.5%+ breakout)
            - RSI_W > 52
            - ADX_W > 12

    DAILY:
        • Confirmation:
            - close_D >= Recent_High_W * 0.995  (no obvious failed breakout)
            - close_D > 20_SMA_D and > EMA_50_D
            - RSI_D > 52
            - MACD_Hist_D > -0.02
        • If Daily_Change_prev_D exists:
            - 1%–10% previous-day move.

    HOURLY (optional):
        • close_H > VWAP_H  (if VWAP_H exists)

    ENTRY:
        • Every 1H bar where weekly_ok & daily_ok & hourly_ok are true.
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required = [
        "close_W", "Recent_High_W", "RSI_W", "ADX_W",
        "close_D", "EMA_50_D", "20_SMA_D", "RSI_D", "MACD_Hist_D",
        "close_H",
    ]
    if not set(required).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 2.")
        return []

    # WEEKLY BREAKOUT REGIME
    weekly_ok = (
        (df_hdw["close_W"] >= df_hdw["Recent_High_W"] * 1.005) &
        (df_hdw["RSI_W"] > 52.0) &
        (df_hdw["ADX_W"] > 12.0)
    )

    # DAILY CONFIRMATION
    daily_core_ok = (
        (df_hdw["close_D"] >= df_hdw["Recent_High_W"] * 0.995) &
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &
        (df_hdw["close_D"] > df_hdw["EMA_50_D"]) &
        (df_hdw["RSI_D"] > 52.0) &
        (df_hdw["MACD_Hist_D"] > -0.02)
    )

    if "Daily_Change_prev_D" in df_hdw.columns:
        dc_prev = df_hdw["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 1.0) & (dc_prev <= 10.0)
    else:
        daily_move_ok = True

    daily_ok = daily_core_ok & daily_move_ok

    # SIMPLE HOURLY FILTER
    if "VWAP_H" in df_hdw.columns:
        hourly_ok = df_hdw["close_H"] > df_hdw["VWAP_H"]
    else:
        hourly_ok = True

    long_mask = weekly_ok & daily_ok & hourly_ok

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
#   Weekly Squeeze + Daily Vol Expansion (made stricter to cut junk)
# ======================================================================
def build_signals_weekly_squeeze_daily_expansion(ticker: str) -> list[dict]:
    """
    STRATEGY 3: Weekly Volatility Squeeze + Daily Expansion

    WEEKLY:
        • Volatility compression:
            - BB width <= 10% of price
        • Momentum bias:
            - RSI_W > 48
            - MACD_Hist_W rising vs prev
            - ADX_W between 8 and 25.

    DAILY:
        • Expansion:
            - close_D >= Upper_Band_D * 1.0
            - ATR_D rising vs last 2 days
            - RSI_D > 58
        • If Daily_Change_prev_D exists: 1.5%–12% band.

    HOURLY (if available):
        • close_H > VWAP_H
        • RSI_H > 50

    ENTRY:
        • Every 1H bar where weekly_ok & daily_ok & hourly_ok are true.
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required = [
        "close_W", "Upper_Band_W", "Lower_Band_W", "RSI_W", "MACD_Hist_W", "ADX_W",
        "close_D", "Upper_Band_D", "ATR_D", "RSI_D",
        "close_H",
    ]
    if not set(required).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 3.")
        return []

    # WEEKLY SQUEEZE
    bb_width_W = (df_hdw["Upper_Band_W"] - df_hdw["Lower_Band_W"]) / df_hdw["close_W"].replace(0, np.nan)
    bb_squeeze = bb_width_W <= 0.10

    macd_prev_W = df_hdw["MACD_Hist_W"].shift(1)
    macd_rising_W = df_hdw["MACD_Hist_W"] >= macd_prev_W

    weekly_ok = (
        bb_squeeze &
        (df_hdw["RSI_W"] > 48.0) &
        (df_hdw["ADX_W"].between(8.0, 25.0)) &
        macd_rising_W
    )

    # DAILY EXPANSION
    atr_prev1 = df_hdw["ATR_D"].shift(1)
    atr_prev2 = df_hdw["ATR_D"].shift(2)
    atr_rising = (df_hdw["ATR_D"] >= atr_prev1) & (atr_prev1 >= atr_prev2)

    daily_core_ok = (
        (df_hdw["close_D"] >= df_hdw["Upper_Band_D"] * 1.0) &
        atr_rising &
        (df_hdw["RSI_D"] > 58.0)
    )

    if "Daily_Change_prev_D" in df_hdw.columns:
        dc_prev = df_hdw["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 1.5) & (dc_prev <= 12.0)
    else:
        daily_move_ok = True

    daily_ok = daily_core_ok & daily_move_ok

    # HOURLY GATE
    if "VWAP_H" in df_hdw.columns and "RSI_H" in df_hdw.columns:
        hourly_ok = (df_hdw["close_H"] > df_hdw["VWAP_H"]) & (df_hdw["RSI_H"] > 50.0)
    else:
        hourly_ok = True

    long_mask = weekly_ok & daily_ok & hourly_ok

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
#   Weekly Trend + Daily Inside Bar Breakout (relaxed so it fires)
# ======================================================================
def build_signals_weekly_trend_insidebar_breakout(ticker: str) -> list[dict]:
    """
    STRATEGY 4: Weekly Trend + Daily Inside Bar Breakout (slightly stricter)

    WEEKLY:
        • 20_SMA_W > EMA_50_W > EMA_200_W
        • ADX_W > 20
        • RSI_W > 50

    DAILY:
        • Inside bar then breakout:
            - (high_D <= high_D_prev * 1.001) & (low_D >= low_D_prev * 0.999)
              (small tolerance around a true inside bar)
            - close_D >= high_D_prev    (true breakout, no discount)
        • Confirmation:
            - close_D > 20_SMA_D
            - close in upper 55% of the day's range
            - RSI_D > 50
            - Stoch_%K_D > Stoch_%D_D

    ENTRY:
        • No extra hourly filters (pure daily breakout timing).
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required = [
        # Weekly
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W", "20_SMA_W",
        # Daily
        "high_D", "low_D", "close_D", "open_D", "20_SMA_D",
        "Stoch_%K_D", "Stoch_%D_D", "RSI_D",
        # Entry price
        "close_H",
    ]
    if not set(required).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 4.")
        return []

    # -------------------- WEEKLY FILTER --------------------
    weekly_ok = (
        (df_hdw["20_SMA_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["ADX_W"] > 20.0) &
        (df_hdw["RSI_W"] > 50.0)
    )

    # -------------------- DAILY FILTER --------------------
    high_prev = df_hdw["high_D"].shift(1)
    low_prev  = df_hdw["low_D"].shift(1)

    # Inside bar with a tiny tolerance
    inside_bar = (
        (df_hdw["high_D"] <= high_prev * 1.001) &
        (df_hdw["low_D"]  >= low_prev * 0.999)
    )

    # True breakout of the mother bar high
    breakout = df_hdw["close_D"] >= high_prev

    stoch_cross = df_hdw["Stoch_%K_D"] > df_hdw["Stoch_%D_D"]

    # Position of close within the daily range
    day_range = (df_hdw["high_D"] - df_hdw["low_D"]).replace(0, np.nan)
    range_pos = (df_hdw["close_D"] - df_hdw["low_D"]) / day_range  # 0 = at low, 1 = at high
    strong_close = range_pos >= 0.55

    daily_ok = (
        inside_bar &
        breakout &
        strong_close &
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &
        (df_hdw["RSI_D"] > 50.0) &
        stoch_cross
    )

    # -------------------- FINAL MASK --------------------
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
            "hourly_ok": True,  # no dedicated hourly filters
        })
    return signals



# ======================================================================
# STRATEGY 5:
#   Weekly Higher Highs + Daily Retest (slightly relaxed)
# ======================================================================
def build_signals_weekly_hh_hl_retest(ticker: str) -> list[dict]:
    """
    STRATEGY 5: Weekly Higher Highs + Daily Retest (slightly stricter)

    WEEKLY:
        • Approx HH/HL behaviour:
            - close_W > 20_SMA_W * 1.01   (price clearly above weekly mean)
            - RSI_W > 57
            - ADX_W > 17

    DAILY:
        • Retest of breakout zone (Recent_High_W proxy):
            - low_D <= Recent_High_W * 1.005   (tighter retest band)
            - close_D >= Recent_High_W * 0.998 (tighter reclaim)
            - close_D > open_D (bullish candle)
            - close_D > 20_SMA_D
            - MACD_Hist_D > 0.0
            - MFI_D > 50
            - Close in upper 55% of the day's range

    HOURLY (optional, if columns exist):
        • close_H > VWAP_H
        • RSI_H > 50
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required = [
        # Weekly
        "close_W", "20_SMA_W", "RSI_W", "ADX_W", "Recent_High_W",
        # Daily
        "open_D", "low_D", "high_D", "close_D", "MACD_Hist_D",
        "MFI_D", "20_SMA_D",
        # Hourly entry price
        "close_H",
    ]
    if not set(required).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 5.")
        return []

    # -------------------- WEEKLY FILTER --------------------
    weekly_ok = (
        (df_hdw["close_W"] > df_hdw["20_SMA_W"] * 1.01) &  # clearly above weekly mean
        (df_hdw["RSI_W"] > 57.0) &                        # slightly stronger momentum
        (df_hdw["ADX_W"] > 17.0)                          # slightly stronger trend
    )

    # -------------------- DAILY FILTER --------------------
    recent_high_W = df_hdw["Recent_High_W"]

    # Tighter retest band and reclaim
    retest_ok = df_hdw["low_D"] <= recent_high_W * 1.005
    reclaim_ok = df_hdw["close_D"] >= recent_high_W * 0.998

    # Stronger daily structure
    day_range = (df_hdw["high_D"] - df_hdw["low_D"]).replace(0, np.nan)
    range_pos = (df_hdw["close_D"] - df_hdw["low_D"]) / day_range
    strong_close = range_pos >= 0.55   # slightly stricter than very loose

    daily_ok = (
        retest_ok &
        reclaim_ok &
        (df_hdw["close_D"] > df_hdw["open_D"]) &   # bullish candle
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) & # above daily mean
        (df_hdw["MACD_Hist_D"] > 0.0) &            # positive momentum
        (df_hdw["MFI_D"] > 50.0) &                 # stronger participation
        strong_close
    )

    # -------------------- HOURLY FILTER (OPTIONAL) --------------------
    if "VWAP_H" in df_hdw.columns and "RSI_H" in df_hdw.columns:
        hourly_ok = (
            (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
            (df_hdw["RSI_H"] > 50.0)
        )
    else:
        hourly_ok = True

    # -------------------- FINAL MASK --------------------
    long_mask = weekly_ok & daily_ok & hourly_ok

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


#strategy 6
def build_signals_weekly_golden_cross_ema_bounce(ticker: str) -> list[dict]:
    """
    STRATEGY 6 (refined slightly stricter):
        Weekly Golden Cross + Daily EMA-Bounce with small quality improvements.
    """

    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required = [
        "EMA_50_W", "EMA_200_W", "RSI_W",
        "low_D", "close_D", "open_D", "20_SMA_D", "EMA_50_D",
        "RSI_D", "MACD_Hist_D",
        "close_H",
    ]
    if not set(required).issubset(df_hdw.columns):
        print(f"- Skipping {ticker}: missing columns for Strategy 6.")
        return []

    # ==========================
    # WEEKLY (slightly stricter)
    # ==========================
    weekly_ok = (
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &     # Golden cross
        (df_hdw["RSI_W"] > 25.0)                          # was 50
    )

    # ==========================
    # DAILY (lightly refined)
    # ==========================

    # Price must tap 20SMA_D or EMA50_D (pullback)
    touch_ma = (
        (df_hdw["low_D"] <= df_hdw["20_SMA_D"]) |
        (df_hdw["low_D"] <= df_hdw["EMA_50_D"])
    )

    # Very small increases in strictness
    daily_core_ok = (
        touch_ma &
        (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &      # reclaim 20 SMA
        (df_hdw["close_D"] > df_hdw["EMA_50_D"]) &       # NEW: must reclaim EMA50 as well
        (df_hdw["close_D"] > df_hdw["open_D"]) &         # bullish close
        (df_hdw["RSI_D"].between(30.0, 70.0)) &          # slightly narrowed
        (df_hdw["MACD_Hist_D"] > 1)                 # slightly stricter MACD
    )

    # Previous-day momentum constraint
    if "Daily_Change_prev_D" in df_hdw.columns:
        dc_prev = df_hdw["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 0.0) & (dc_prev <= 7.0)   # was 8%
    else:
        daily_move_ok = True

    daily_ok = daily_core_ok & daily_move_ok

    # ==========================
    # HOURLY FILTER (lightly refined)
    # ==========================
    if "VWAP_H" in df_hdw.columns and "RSI_H" in df_hdw.columns:
        hourly_ok = (
            (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
            (df_hdw["RSI_H"] > 25.0)                     # slightly raised from 48
        )
    else:
        hourly_ok = True

    # ==========================
    # FINAL SIGNAL MASK
    # ==========================
    long_mask = weekly_ok & daily_ok & hourly_ok

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
#   Rich weekly regime + relaxed prev-day daily (no hourly)
# ======================================================================
def build_signals_custom1(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 1

    Quality-focused MTF strategy using only WEEKLY + previous-day DAILY.

    WEEKLY:
        • Trend structure, momentum, MFI, ATR%, BB position, Recent_High, Stoch.

    DAILY:
        • Previous-day Daily_Change_prev_D slightly relaxed:
            - 1%–7% band.
        • Trend vs EMAs (soft), RSI_D 42–72, MACD_Hist_D >= -0.02,
          MFI_D > 45, ADX_D > 10, ATR% 0.3–7%, BB position, Recent_High_D,
          upper-range close, Stoch_%K_D vs %D_D.

    ENTRY:
        • Every 1H bar where weekly_ok & daily_ok are true.
    """
    df_hdw = _load_mtf_context(ticker)
    if df_hdw.empty:
        return []

    required_cols = [
        # Weekly
        "close_W", "open_W", "high_W", "low_W",
        "EMA_50_W", "EMA_200_W",
        "RSI_W", "ADX_W", "MACD_Hist_W", "MFI_W", "ATR_W",
        "20_SMA_W", "Upper_Band_W", "Lower_Band_W",
        "Recent_High_W", "Stoch_%K_W", "Stoch_%D_W",
        # Daily
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

    # --- WEEKLY REGIME ---
    weekly_strong = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"] * 1.005) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 42.0) &
        (df_hdw["ADX_W"] > 12.0)
    )

    weekly_mild = (
        (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.015) &
        (df_hdw["close_W"] >= df_hdw["EMA_50_W"] * 0.99) &
        (df_hdw["RSI_W"].between(42.0, 66.0)) &
        (df_hdw["ADX_W"] > 10.0)
    )

    weekly_macd_ok = df_hdw["MACD_Hist_W"] >= -0.02
    weekly_mfi_ok = df_hdw["MFI_W"] > 45.0

    close_W = df_hdw["close_W"]
    atr_pct_W = (df_hdw["ATR_W"] / close_W.replace(0, np.nan)) * 100.0
    weekly_atr_ok = (atr_pct_W >= 0.4) & (atr_pct_W <= 10.0)

    mid_W   = df_hdw["20_SMA_W"]
    upper_W = df_hdw["Upper_Band_W"]
    lower_W = df_hdw["Lower_Band_W"]

    weekly_bb_pos_ok = (
        (close_W >= mid_W * 0.995) &
        (close_W <= upper_W * 1.03) &
        (close_W >= lower_W * 0.99)
    )

    recent_high_W = df_hdw["Recent_High_W"]
    weekly_recent_high_ok = close_W >= (recent_high_W * 0.94)

    stoch_k_W = df_hdw["Stoch_%K_W"]
    stoch_d_W = df_hdw["Stoch_%D_W"]
    weekly_stoch_ok = (
        (stoch_k_W >= stoch_d_W * 0.98) &
        (stoch_k_W >= 35.0) &
        (stoch_k_W <= 88.0)
    )

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

    # --- DAILY SETUP (relaxed) ---
    daily_prev = df_hdw["Daily_Change_prev_D"]
    daily_move_ok = (daily_prev >= 1.0) & (daily_prev <= 7.0)

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

    signals: list[dict] = []
    for _, r in df_hdw[long_mask].iterrows():
        signals.append(
            {
                "signal_time_ist": r["date"],
                "ticker": ticker,
                "signal_side": "LONG",
                "entry_price": float(r["close_H"]),
                "weekly_ok": True,
                "daily_ok": True,
                "hourly_ok": True,  # no hourly filters used
            }
        )

    return signals


# ======================================================================
# CUSTOM STRATEGY 2:
#   Relaxed WEEKLY + prev-day DAILY + looser HOURLY trigger
# ======================================================================
def build_signals_custom2(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 2

    Relaxed weekly trend + prev-day Daily_Change band + looser but
    still directional hourly trigger.
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

    # Daily prev change
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Hourly extras
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

    # Merge D+W onto H
    df_h_sorted = df_h.sort_values("date")
    df_d_sorted = df_d.sort_values("date")
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(seconds=1)

    df_hd = pd.merge_asof(df_h_sorted, df_d_sorted, on="date", direction="backward")
    df_hdw = pd.merge_asof(df_hd.sort_values("date"), df_w_sorted, on="date", direction="backward")

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
    signals: list[dict] = []

    # WEEKLY FILTER (relaxed)
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

    # DAILY FILTER (prev-day move band: 1–8%)
    daily_change_prev = df_hdw["Daily_Change_prev_D"]
    daily_long_ok = (daily_change_prev >= 1.0) & (daily_change_prev <= 8.0)

    # HOURLY TRIGGER (relaxed but directional)
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
#   Mildly-strict relaxed MTF with D-1 daily merge (now stricter to cut count)
# ======================================================================
def build_signals_custom3(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 3

    Originally produced *too many* signals with low 10% hit.
    Now made stricter on daily/hourly momentum to improve quality.
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

    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Daily extras
    if "ATR_D" in df_d.columns:
        df_d["ATR_D_5d_mean"] = df_d["ATR_D"].rolling(5, min_periods=5).mean()
    else:
        df_d["ATR_D_5d_mean"] = np.nan

    if "high_D" in df_d.columns:
        df_d["Recent_High_10_D"] = df_d["high_D"].rolling(10, min_periods=10).max()
    else:
        df_d["Recent_High_10_D"] = np.nan

    # Hourly extras
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

    df_h_sorted = df_h.sort_values("date")

    # Daily D-1 merge
    df_d_sorted = df_d.sort_values("date").copy()
    df_d_for_merge = df_d_sorted.copy()
    df_d_for_merge["date"] = df_d_for_merge["date"] + pd.Timedelta(days=1)

    # Weekly no same-date lookahead
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_for_merge = df_w_sorted.copy()
    df_w_for_merge["date"] = df_w_for_merge["date"] + pd.Timedelta(seconds=1)

    df_hd = pd.merge_asof(df_h_sorted, df_d_for_merge, on="date", direction="backward")
    df_hdw = pd.merge_asof(df_hd.sort_values("date"), df_w_for_merge, on="date", direction="backward")

    df_hdw = df_hdw.dropna(
        subset=[
            "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W",
            "close_D", "EMA_50_D", "RSI_D", "MACD_Hist_D",
            "ATR_D", "ATR_D_5d_mean", "Recent_High_10_D",
            "ATR_H", "ATR_H_5bar_mean", "VWAP_H", "RSI_H", "MACD_Hist_H",
        ]
    ).reset_index(drop=True)

    if df_hdw.empty:
        print(f"- After MTF merge, no usable rows for {ticker}.")
        return []

    signals: list[dict] = []

    # WEEKLY FILTER (mildly strict)
    weekly_ok = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 50.0) &
        (df_hdw["ADX_W"] > 15.0)
    )

    # DAILY FILTER (stricter now)
    daily_close_vs_high = df_hdw["close_D"] >= (df_hdw["Recent_High_10_D"] * 0.99)
    daily_atr_expansion = df_hdw["ATR_D"] >= (df_hdw["ATR_D_5d_mean"] * 0.97)

    daily_ok = (
        (df_hdw["close_D"] > df_hdw["EMA_50_D"]) &
        (df_hdw["RSI_D"] > 52.0) &
        (df_hdw["MACD_Hist_D"] >= 0.0) &
        daily_atr_expansion &
        daily_close_vs_high
    )

    # HOURLY TRIGGER (stricter)
    hourly_atr_expansion = df_hdw["ATR_H"] >= (df_hdw["ATR_H_5bar_mean"] * 1.0)
    hourly_recent_high_break = df_hdw["close_H"] >= (df_hdw["Recent_High_5_H"] * 0.999)

    hourly_ok = (
        (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
        (df_hdw["RSI_H"] > 50.0) &
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
#   Slightly-loosened quality-focused MTF (tiny tweaks)
# ======================================================================
def build_signals_custom4(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 4

    This already gave very good hit-rate & P&L.
    Only tiny tweaks: slightly narrower Daily_Change_prev band
    to avoid extreme 1-day moves.
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

    # Daily: compute previous-day Daily_Change
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Hourly extras
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

    # WEEKLY (very lightly relaxed)
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

    weekly_not_extreme = df_hdw["RSI_W"] < 88.0

    weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme

    # DAILY (slightly narrowed band 2–8%)
    daily_prev = df_hdw["Daily_Change_prev_D"]
    daily_ok = (daily_prev >= 2.0) & (daily_prev <= 8.0)

    # HOURLY (tiny relaxations vs original strict logic)
    hourly_trend = (
        (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &
        (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)
    )

    rsi_ok = (
        (df_hdw["RSI_H"] > 49.0) |
        ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
    )

    macd_ok = (
        (df_hdw["MACD_Hist_H"] > -0.055) &
        (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.75)
    )

    stoch_ok = (
        (df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98) &
        (df_hdw["Stoch_%K_H"] >= 18.0)
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
        (breakout_near & strong_up & (df_hdw["RSI_H"] >= 47.0))
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



#strategy11


def build_signals_strategy11(ticker: str) -> list[dict]:
    """
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

    # --- Merge Daily + Weekly onto Hourly using as-of (backward) joins ---

    df_h_sorted = df_h.sort_values("date")
    df_d_sorted = df_d.sort_values("date")

    # Weekly no-same-date-lookahead trick
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(seconds=1)

    # Merge daily into hourly
    df_hd = pd.merge_asof(
        df_h_sorted,
        df_d_sorted,
        on="date",
        direction="backward"
    )

    # Merge weekly into (hourly+daily)
    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_sorted,
        on="date",
        direction="backward"
    )

    # Require weekly context and previous-day Daily_Change_prev_D
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

    # ----------------- STRATEGY CONDITIONS (LONG ONLY) -----------------

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

    # ========== DAILY FILTER (same as before) ==========
    daily_change_prev = df_hdw["Daily_Change_prev_D"]
    daily_long_ok = daily_change_prev >= 2.0

    # ========== HOURLY TRIGGER (RELAXED FURTHER) ==========

    # Trend: price roughly around EMAs, but not in clear downtrend
    hourly_trend_ok = (
        (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &    # around / above 200 EMA
        (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)       # can dip ~0.5% under EMA50
    )

    # Momentum:
    #   - RSI: moderate, either > 49 OR (> 46 and rising)
    rsi_trend_ok = (
        (df_hdw["RSI_H"] > 49.0) |
        ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
    )

    #   - MACD: allow a bit more negative but not collapsing
    macd_trend_ok = (
        (df_hdw["MACD_Hist_H"] > -0.06) &  # allow small negative
        (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.85)
    )

    #   - Stochastic: fast can be close to slow (not strictly above)
    stoch_trend_ok = df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98

    hourly_momentum_ok = rsi_trend_ok & macd_trend_ok & stoch_trend_ok

    # Volatility & trend strength:
    #   ATR slightly expanded OR ADX decent / improving
    hourly_vol_ok = (
        (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.92) &
        (
            (df_hdw["ADX_H"] >= 11.0) |
            (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"])   # ADX rising vs its mean
        )
    )

    # Breakout & location:
    #   - Above VWAP
    #   - Either:
    #       * clean 5-bar breakout, OR
    #       * within 0.5% of 5-bar high if current hour candle is strong
    price_above_vwap = df_hdw["close_H"] >= df_hdw["VWAP_H"]

    breakout_strict = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]
    breakout_near   = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.995  # up to 0.5% below

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

    # FINAL MASK (LONG ONLY)
    long_mask = weekly_long_ok & daily_long_ok & hourly_long_ok

    long_rows = df_hdw[long_mask].copy()

    # Build LONG signals
    for _, r in long_rows.iterrows():
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
# CUSTOM STRATEGY 6:
#   Union of 5 families on a single rich MTF context (1H + Daily + Weekly)
# ======================================================================
def build_signals_custom6(ticker: str) -> list[dict]:
    """
    CUSTOM STRATEGY 6

    Combines 5 strategy families into ONE on a single MTF dataframe:

        1) trend_pullback                    (Weekly Trend + Daily Pullback & Resume)
        2) weekly_golden_cross_ema_bounce   (Weekly golden cross + EMA bounce)
        3) custom2                           (Relaxed weekly + D-1 band + looser hourly)
        4) custom4                           (Slightly-loosened quality-focused MTF)
        5) strategy11                        (Relaxed weekly + even looser hourly trigger)

    Flow:
        • Load 1H, Daily, Weekly CSVs.
        • Compute Daily_Change_prev on Daily.
        • Suffix with _H / _D / _W.
        • Add hourly extras: ATR_H_5bar_mean, RSI_H_prev, MACD_Hist_H_prev,
          Recent_High_5_H, ADX_H_5bar_mean, Stoch_%K_H_prev.
        • Merge Daily + Weekly into Hourly via as-of joins (no same-date weekly lookahead).
        • For each strategy family, build its boolean mask (if required columns exist).
        • Final LONG mask = OR of all masks.
        • Emit LONG signals with `sub_strategy` tag listing which families fired.
    """
    # ------------------------------------------------------------------
    # 1) Load and prepare MTF data (shared for all sub-strategies)
    # ------------------------------------------------------------------
    path_etf_1h = os.path.join(DIR_etf_1h, f"{ticker}_main_indicators_etf_1h.csv")
    path_d      = os.path.join(DIR_D,      f"{ticker}_main_indicators_etf_daily.csv")
    path_w      = os.path.join(DIR_W,      f"{ticker}_main_indicators_etf_weekly.csv")

    df_h = _safe_read_tf(path_etf_1h)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # Daily: compute previous-day Daily_Change BEFORE suffixing
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # Suffix all TFs
    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Hourly extras (on suffixed 1H)
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

    # Merge Daily + Weekly onto Hourly (backward as-of; weekly no same-date lookahead)
    df_h_sorted = df_h.sort_values("date")
    df_d_sorted = df_d.sort_values("date")

    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(seconds=1)

    df_hd = pd.merge_asof(df_h_sorted, df_d_sorted, on="date", direction="backward")
    df_hdw = pd.merge_asof(df_hd.sort_values("date"), df_w_sorted, on="date", direction="backward")

    if df_hdw.empty:
        print(f"- After MTF merge, no usable rows for {ticker}.")
        return []

    cols = set(df_hdw.columns)

    def _false_series() -> pd.Series:
        return pd.Series(False, index=df_hdw.index)

    # Masks for each family
    mask_trend_pullback  = _false_series()
    mask_golden_cross    = _false_series()
    mask_custom2         = _false_series()
    mask_custom4         = _false_series()
    mask_strategy11      = _false_series()

    # ------------------------------------------------------------------
    # 2) STRATEGY 1 – Weekly Trend + Daily Pullback & Resume
    # ------------------------------------------------------------------
    required_s1 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "close_D","open_D","high_D","low_D",
        "20_SMA_D","EMA_50_D","RSI_D","MACD_Hist_D",
        "close_H"
    }
    if required_s1.issubset(cols):
        weekly_core_ok = (
            (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
            (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
            (df_hdw["RSI_W"] > 52.0) &
            (df_hdw["ADX_W"] > 19.0)
        )

        if "Recent_High_W" in cols:
            weekly_loc_ok = df_hdw["close_W"] >= df_hdw["Recent_High_W"] * 0.93
        else:
            weekly_loc_ok = True

        weekly_final_ok = weekly_core_ok & weekly_loc_ok

        macd_prev_D = df_hdw["MACD_Hist_D"].shift(1)
        macd_rising_D = df_hdw["MACD_Hist_D"] >= macd_prev_D

        day_range = (df_hdw["high_D"] - df_hdw["low_D"]).replace(0,np.nan)
        range_pos = (df_hdw["close_D"] - df_hdw["low_D"]) / day_range
        strong_close = range_pos >= 0.65

        pullback_zone = (
            (df_hdw["close_D"] >= df_hdw["20_SMA_D"] * 0.985) &
            (df_hdw["close_D"] <= df_hdw["EMA_50_D"] * 1.02)
        )

        daily_core_ok = (
            (df_hdw["close_D"] > df_hdw["open_D"]) &
            pullback_zone &
            strong_close &
            (df_hdw["RSI_D"] > 52.0) &
            macd_rising_D
        )

        if "Daily_Change_prev_D" in cols:
            dc_prev = df_hdw["Daily_Change_prev_D"]
            daily_move_ok = (dc_prev >= 0.7) & (dc_prev <= 7.5)
        else:
            daily_move_ok = True

        daily_final_ok = daily_core_ok & daily_move_ok

        if "VWAP_H" in cols and "RSI_H" in cols:
            hourly_ok_s1 = (
                (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
                (df_hdw["RSI_H"] > 50.0)
            )
        else:
            hourly_ok_s1 = True

        mask_trend_pullback = weekly_final_ok & daily_final_ok & hourly_ok_s1

    # ------------------------------------------------------------------
    # 3) STRATEGY 6 – Weekly Golden Cross + Daily EMA Bounce
    # ------------------------------------------------------------------
    required_gc = {
        "EMA_50_W","EMA_200_W","RSI_W",
        "low_D","close_D","open_D","20_SMA_D","EMA_50_D",
        "RSI_D","MACD_Hist_D",
        "close_H"
    }
    if required_gc.issubset(cols):
        weekly_ok_gc = (
            (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
            (df_hdw["RSI_W"] > 25.0)
        )

        touch_ma = (
            (df_hdw["low_D"] <= df_hdw["20_SMA_D"]) |
            (df_hdw["low_D"] <= df_hdw["EMA_50_D"])
        )

        daily_core_gc = (
            touch_ma &
            (df_hdw["close_D"] > df_hdw["20_SMA_D"]) &
            (df_hdw["close_D"] > df_hdw["EMA_50_D"]) &
            (df_hdw["close_D"] > df_hdw["open_D"]) &
            (df_hdw["RSI_D"].between(30.0, 70.0)) &
            (df_hdw["MACD_Hist_D"] > 1)
        )

        if "Daily_Change_prev_D" in cols:
            dc_prev_gc = df_hdw["Daily_Change_prev_D"]
            daily_move_gc = (dc_prev_gc >= 0.0) & (dc_prev_gc <= 7.0)
        else:
            daily_move_gc = True

        daily_ok_gc = daily_core_gc & daily_move_gc

        if "VWAP_H" in cols and "RSI_H" in cols:
            hourly_ok_gc = (
                (df_hdw["close_H"] > df_hdw["VWAP_H"]) &
                (df_hdw["RSI_H"] > 25.0)
            )
        else:
            hourly_ok_gc = True

        mask_golden_cross = weekly_ok_gc & daily_ok_gc & hourly_ok_gc

    # ------------------------------------------------------------------
    # 4) CUSTOM STRATEGY 2 – Relaxed WEEKLY + banded DAILY + looser HOURLY
    # ------------------------------------------------------------------
    required_c2 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "Daily_Change_prev_D",
        "close_H","EMA_200_H","EMA_50_H",
        "RSI_H","RSI_H_prev",
        "MACD_Hist_H","MACD_Hist_H_prev",
        "Stoch_%K_H","Stoch_%D_H",
        "ATR_H","ATR_H_5bar_mean",
        "ADX_H","ADX_H_5bar_mean",
        "VWAP_H","Recent_High_5_H"
    }
    if required_c2.issubset(cols):
        weekly_strong_up_c2 = (
            (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
            (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
            (df_hdw["RSI_W"] > 46.0) &
            (df_hdw["ADX_W"] > 11.0)
        )
        weekly_mild_up_c2 = (
            (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.01) &
            (df_hdw["RSI_W"].between(44.0, 65.0)) &
            (df_hdw["ADX_W"] > 9.0)
        )
        weekly_long_ok_c2 = weekly_strong_up_c2 | weekly_mild_up_c2

        dprev_c2 = df_hdw["Daily_Change_prev_D"]
        daily_long_ok_c2 = (dprev_c2 >= 1.0) & (dprev_c2 <= 8.0)

        hourly_trend_ok_c2 = (
            (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &
            (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)
        )

        rsi_trend_ok_c2 = (
            (df_hdw["RSI_H"] > 49.0) |
            ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
        )

        macd_trend_ok_c2 = (
            (df_hdw["MACD_Hist_H"] > -0.06) &
            (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.85)
        )

        stoch_trend_ok_c2 = df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98

        hourly_momentum_ok_c2 = rsi_trend_ok_c2 & macd_trend_ok_c2 & stoch_trend_ok_c2

        hourly_vol_ok_c2 = (
            (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.92) &
            (
                (df_hdw["ADX_H"] >= 11.0) |
                (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"])
            )
        )

        price_above_vwap_c2 = df_hdw["close_H"] >= df_hdw["VWAP_H"]
        breakout_strict_c2 = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]
        breakout_near_c2   = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.995

        intra_change_h_c2 = df_hdw.get("Intra_Change_H", pd.Series(0.0, index=df_hdw.index)).fillna(0.0)
        strong_up_move_c2 = intra_change_h_c2 >= 0.2

        hourly_breakout_ok_c2 = price_above_vwap_c2 & (
            breakout_strict_c2 |
            (breakout_near_c2 & strong_up_move_c2)
        )

        hourly_long_ok_c2 = (
            hourly_trend_ok_c2 &
            hourly_momentum_ok_c2 &
            hourly_vol_ok_c2 &
            hourly_breakout_ok_c2
        )

        mask_custom2 = weekly_long_ok_c2 & daily_long_ok_c2 & hourly_long_ok_c2

    # ------------------------------------------------------------------
    # 5) CUSTOM STRATEGY 4 – Quality-focused MTF
    # ------------------------------------------------------------------
    required_c4 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W","Daily_Change_prev_D",
        "close_H","EMA_200_H","EMA_50_H",
        "RSI_H","RSI_H_prev",
        "MACD_Hist_H","MACD_Hist_H_prev",
        "Stoch_%K_H","Stoch_%D_H","Stoch_%K_H_prev",
        "ATR_H","ATR_H_5bar_mean",
        "ADX_H","ADX_H_5bar_mean",
        "VWAP_H","Recent_High_5_H"
    }
    if required_c4.issubset(cols):
        weekly_strong_c4 = (
            (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
            (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
            (df_hdw["RSI_W"] > 40.0) &
            (df_hdw["ADX_W"] > 10.0)
        )
        weekly_mild_c4 = (
            (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.01) &
            (df_hdw["RSI_W"].between(40.0, 68.0)) &
            (df_hdw["ADX_W"] > 9.0)
        )
        weekly_not_extreme_c4 = df_hdw["RSI_W"] < 88.0
        weekly_ok_c4 = (weekly_strong_c4 | weekly_mild_c4) & weekly_not_extreme_c4

        dprev_c4 = df_hdw["Daily_Change_prev_D"]
        daily_ok_c4 = (dprev_c4 >= 2.0) & (dprev_c4 <= 8.0)

        hourly_trend_c4 = (
            (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &
            (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)
        )

        rsi_ok_c4 = (
            (df_hdw["RSI_H"] > 49.0) |
            ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
        )

        macd_ok_c4 = (
            (df_hdw["MACD_Hist_H"] > -0.055) &
            (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.75)
        )

        stoch_ok_c4 = (
            (df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98) &
            (df_hdw["Stoch_%K_H"] >= 18.0)
        )

        momentum_ok_c4 = rsi_ok_c4 & macd_ok_c4 & stoch_ok_c4

        hourly_vol_ok_c4 = (
            (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.92) &
            ((df_hdw["ADX_H"] >= 10.0) | (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"]))
        )

        price_above_vwap_c4 = df_hdw["close_H"] >= df_hdw["VWAP_H"]
        breakout_strict_c4 = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]
        breakout_near_c4   = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.99

        intra_chg_c4 = df_hdw.get("Intra_Change_H", pd.Series(0.0, index=df_hdw.index)).fillna(0.0)
        strong_up_c4 = intra_chg_c4 >= 0.05

        breakout_ok_c4 = price_above_vwap_c4 & (
            breakout_strict_c4 |
            (breakout_near_c4 & strong_up_c4 & (df_hdw["RSI_H"] >= 47.0))
        )

        hourly_ok_c4 = hourly_trend_c4 & momentum_ok_c4 & hourly_vol_ok_c4 & breakout_ok_c4

        mask_custom4 = weekly_ok_c4 & daily_ok_c4 & hourly_ok_c4

    # ------------------------------------------------------------------
    # 6) STRATEGY 11 – Relaxed Weekly + even looser Hourly
    # ------------------------------------------------------------------
    required_s11 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "Daily_Change_prev_D",
        "close_H","EMA_200_H","EMA_50_H",
        "RSI_H","RSI_H_prev",
        "MACD_Hist_H","MACD_Hist_H_prev",
        "Stoch_%K_H","Stoch_%D_H",
        "ATR_H","ATR_H_5bar_mean",
        "ADX_H","ADX_H_5bar_mean",
        "VWAP_H","Recent_High_5_H"
    }
    if required_s11.issubset(cols):
        weekly_strong_up_11 = (
            (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
            (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
            (df_hdw["RSI_W"] > 46.0) &
            (df_hdw["ADX_W"] > 11.0)
        )
        weekly_mild_up_11 = (
            (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.01) &
            (df_hdw["RSI_W"].between(44.0, 65.0)) &
            (df_hdw["ADX_W"] > 9.0)
        )
        weekly_long_ok_11 = weekly_strong_up_11 | weekly_mild_up_11

        dprev_11 = df_hdw["Daily_Change_prev_D"]
        daily_long_ok_11 = dprev_11 >= 2.0

        hourly_trend_ok_11 = (
            (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 0.995) &
            (df_hdw["close_H"] >= df_hdw["EMA_50_H"] * 0.995)
        )

        rsi_trend_ok_11 = (
            (df_hdw["RSI_H"] > 49.0) |
            ((df_hdw["RSI_H"] > 46.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))
        )

        macd_trend_ok_11 = (
            (df_hdw["MACD_Hist_H"] > -0.06) &
            (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.85)
        )

        stoch_trend_ok_11 = df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"] * 0.98

        hourly_momentum_ok_11 = rsi_trend_ok_11 & macd_trend_ok_11 & stoch_trend_ok_11

        hourly_vol_ok_11 = (
            (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.92) &
            (
                (df_hdw["ADX_H"] >= 11.0) |
                (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"])
            )
        )

        price_above_vwap_11 = df_hdw["close_H"] >= df_hdw["VWAP_H"]
        breakout_strict_11 = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]
        breakout_near_11   = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.995

        intra_change_h_11 = df_hdw.get("Intra_Change_H", pd.Series(0.0, index=df_hdw.index)).fillna(0.0)
        strong_up_move_11 = intra_change_h_11 >= 0.2

        hourly_breakout_ok_11 = price_above_vwap_11 & (
            breakout_strict_11 |
            (breakout_near_11 & strong_up_move_11)
        )

        hourly_long_ok_11 = (
            hourly_trend_ok_11 &
            hourly_momentum_ok_11 &
            hourly_vol_ok_11 &
            hourly_breakout_ok_11
        )

        mask_strategy11 = weekly_long_ok_11 & daily_long_ok_11 & hourly_long_ok_11

    # ------------------------------------------------------------------
    # FINAL COMBINED MASK – weighted votes (cores count double)
    # ------------------------------------------------------------------
    votes_weighted = (
        1 * mask_trend_pullback.astype(int) +
        1 * mask_custom4.astype(int)       +
        2 * mask_golden_cross.astype(int)  +
        1 * mask_custom2.astype(int)       +
        3 * mask_strategy11.astype(int)
    )

    # threshold: 3 is moderately strict, 4 is stricter
    THRESH = 3
    combined_mask = votes_weighted >= THRESH

    if not combined_mask.any():
        return []

    signals: list[dict] = []
    for idx, r in df_hdw[combined_mask].iterrows():
        tags = []
        if mask_trend_pullback.loc[idx]:
            tags.append("trend_pullback")
        if mask_golden_cross.loc[idx]:
            tags.append("golden_cross")
        if mask_custom2.loc[idx]:
            tags.append("custom2")
        if mask_custom4.loc[idx]:
            tags.append("custom4")
        if mask_strategy11.loc[idx]:
            tags.append("strategy11")

        sub_strat = "+".join(tags) if tags else "custom6"

        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "weekly_ok": True,
            "daily_ok": True,
            "hourly_ok": True,
            "sub_strategy": sub_strat,
            "sub_strategy_votes": int(votes_weighted.loc[idx]),
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
    elif s == "golden_cross":
        return build_signals_weekly_golden_cross_ema_bounce(ticker)
    elif s == "custom2":
        return build_signals_custom2(ticker)
    elif s == "custom4":
        return build_signals_custom4(ticker)
    elif s == "custom5":
        return build_signals_strategy11(ticker)
    elif s == "custom6":
        return build_signals_custom6(ticker)
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
        "6": "golden_cross",
        "8": "custom2",
        "10": "custom4",
        "11": "custom5",
        "12": "custom6",
    }

    print("\n=== Select Strategy for LONG Signal Generation ===")
    print(" 1) trend_pullback        - Weekly trend + daily pullback & resume")
    print(" 6) golden_cross          - Weekly golden cross + daily EMA bounce")
    print(" 8) custom2               - Relaxed weekly + looser hourly trigger")
    print(" 10) custom4               - Slightly loosened quality-focused MTF")
    print(" 11) custom5               ")
    print(" 12) custom6               ")
    print("--------------------------------------------------")
    raw = input("Enter choice (1-12 or strategy name): ").strip().lower()

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
