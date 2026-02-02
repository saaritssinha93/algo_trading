# -*- coding: utf-8 -*-
"""
ALGO-SM1 ETF Multi‑TF LONG signal generator + evaluator + portfolio simulator (beginner-friendly).

UPDATED (as requested):
1) Use 15-minute (15m) intraday data instead of 1H.
2) UNREALISED (OPEN) TRADES:
   - If a TAKEN trade does NOT hit the target within available intraday data,
     we DO NOT force-close it at the last close.
     Instead it's marked as UNREALIZED_OPEN with MTM metrics:
       exit_time  = NaT
       exit_price = NaN
       mtm_time / mtm_price / mtm_pnl_pct / mtm_pnl_rs / mtm_value computed on last available close.
3) ONE POSITION PER TICKER:
   - Once a trade is TAKEN for a ticker, we do not consider further signals for that ticker
     until it is closed. If it is UNREALIZED_OPEN at the end, it stays open (ticker remains locked).
4) SUPPRESS SIGNALS DURING OPEN POSITIONS:
   - Signals that occur while the ticker is already open are removed from output CSVs.
5) FIX FOR "SIGNALS START LATE" (e.g. 2025-05-13):
   - This typically happens because weekly EMA_200 is NaN until ~200 weeks of data exist.
   - We add a SAFE fallback:
        EMA_LONG_W = EMA_200_W, but if it's NaN, try EMA_100_W, then EMA_50_W (if available).
     This allows signals earlier when weekly EMA_200 is not yet available.
   - You can disable fallback by setting ALLOW_WEEKLY_EMA_FALLBACK=False.

ASSUMPTIONS
- CSV 'date' column is UTC (or parseable as UTC). We convert to IST.
- Intraday CSV contains OHLC columns: open, high, low, close (at least high/low/close for evaluation).
- Indicator columns exist as in your pipeline (EMA_50, EMA_200, RSI, ADX, MACD_Hist, Stoch_%K, Stoch_%D, ATR, VWAP, Recent_High, Intra_Change, etc.)

OUTPUT (OUT_DIR)
- multi_tf_signals_<timestamp>_IST.csv
- multi_tf_signals_etf_daily_counts_<timestamp>_IST.csv
- multi_tf_signals_<TARGET>pct_eval_<timestamp>_IST.csv

Printed at end:
- Portfolio summary (realised + unrealised)
- TAKEN TRADES (closed + open)
- UNREALISED OPEN TRADES (detailed MTM)
"""

from __future__ import annotations

import os
import glob
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz


# =============================================================================
# CONFIG
# =============================================================================

# Intraday TF configuration (15-minute)
INTRA_TF_LABEL = "15m"
DIR_INTRA = "etf_indicators_15min"
INTRA_FILE_ENDING = "_etf_indicators_15min.csv"
INTRA_SUFFIX = "_M"   # intraday columns become close_M, RSI_M, etc.

# Higher TF directories
DIR_D  = "etf_indicators_daily"
DIR_W  = "etf_indicators_weekly"
D_FILE_ENDING = "_etf_indicators_daily.csv"
W_FILE_ENDING = "_etf_indicators_weekly.csv"

# Outputs
OUT_DIR = "signals"
os.makedirs(OUT_DIR, exist_ok=True)

# Timezone
IST = pytz.timezone("Asia/Kolkata")

# Target configuration
TARGET_PCT: float = 6.0
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)
TARGET_LABEL: str = f"{TARGET_INT}%"

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 200_000.0
MAX_CAPITAL_RS: float = 1100_000.0

# Position constraints
ONE_POSITION_PER_TICKER: bool = True

# Weekly EMA fallback (to avoid late start due to EMA_200 warmup)
ALLOW_WEEKLY_EMA_FALLBACK: bool = True

# A "far future" timestamp used internally to keep a ticker locked for an unrealised trade
FAR_FUTURE_LOCK = pd.Timestamp("2099-12-31 23:59:59", tz=IST)

# Derived columns
HIT_COL: str      = f"hit_{TARGET_INT}pct"
TIME_TD_COL: str  = f"time_to_{TARGET_INT}pct_days"
TIME_CAL_COL: str = f"days_to_{TARGET_INT}pct_calendar"
EVAL_SUFFIX: str  = f"{TARGET_INT}pct_eval"

# For time-to-target in “trading days” (approx)
TRADING_HOURS_PER_DAY: float = 6.25


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def read_tf_csv(path: str, tz=IST) -> pd.DataFrame:
    """Read a timeframe CSV and return DataFrame sorted by 'date' (tz-aware IST)."""
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(tz)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()


def suffix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Rename all columns except 'date' by adding a suffix."""
    if df.empty:
        return df
    return df.rename(columns={c: f"{c}{suffix}" for c in df.columns if c != "date"})


def list_tickers(directory: str, ending: str) -> set[str]:
    """List tickers by stripping a known filename ending."""
    pattern = os.path.join(directory, f"*{ending}")
    files = glob.glob(pattern)

    tickers: set[str] = set()
    for f in files:
        base = os.path.basename(f)
        if base.endswith(ending):
            t = base[: -len(ending)]
            if t:
                tickers.add(t)
    return tickers


def fmt_rs(x: float) -> str:
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "₹NA"


def fmt_dt(x) -> str:
    try:
        x = pd.to_datetime(x, errors="coerce")
        return x.strftime("%Y-%m-%d %H:%M:%S%z") if pd.notna(x) else "NA"
    except Exception:
        return "NA"


# =============================================================================
# SIGNAL GENERATION (Weekly + Daily(prev) + Intraday trigger)
# =============================================================================

def add_intraday_features(df_i: pd.DataFrame) -> pd.DataFrame:
    """Add rolling/shift features used by the intraday timing trigger."""
    if df_i.empty:
        return df_i

    df_i[f"ATR{INTRA_SUFFIX}_5bar_mean"] = df_i.get(f"ATR{INTRA_SUFFIX}", np.nan).rolling(5).mean()
    df_i[f"RSI{INTRA_SUFFIX}_prev"] = df_i.get(f"RSI{INTRA_SUFFIX}", np.nan).shift(1)
    df_i[f"MACD_Hist{INTRA_SUFFIX}_prev"] = df_i.get(f"MACD_Hist{INTRA_SUFFIX}", np.nan).shift(1)
    df_i[f"Recent_High_5{INTRA_SUFFIX}"] = df_i.get(f"high{INTRA_SUFFIX}", np.nan).rolling(5).max()
    df_i[f"ADX{INTRA_SUFFIX}_5bar_mean"] = df_i.get(f"ADX{INTRA_SUFFIX}", np.nan).rolling(5).mean()
    df_i[f"Stoch_%K{INTRA_SUFFIX}_prev"] = df_i.get(f"Stoch_%K{INTRA_SUFFIX}", np.nan).shift(1)

    return df_i


def merge_daily_and_weekly_onto_intraday(df_i: pd.DataFrame, df_d: pd.DataFrame, df_w: pd.DataFrame) -> pd.DataFrame:
    """Merge Daily + Weekly context onto each intraday row using asof backward merge."""
    if df_i.empty or df_d.empty or df_w.empty:
        return pd.DataFrame()

    df_id = pd.merge_asof(
        df_i.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward",
    )

    df_w_shift = df_w.sort_values("date").copy()
    df_w_shift["date"] += pd.Timedelta(seconds=1)

    df_idw = pd.merge_asof(
        df_id.sort_values("date"),
        df_w_shift,
        on="date",
        direction="backward",
    )

    return df_idw


def make_weekly_ema_long(df: pd.DataFrame) -> pd.Series:
    """Build EMA_LONG_W = EMA_200_W with optional fallback to EMA_100_W then EMA_50_W."""
    ema200 = df.get("EMA_200_W", pd.Series(np.nan, index=df.index))
    if not ALLOW_WEEKLY_EMA_FALLBACK:
        return ema200

    ema_long = ema200.copy()
    if "EMA_100_W" in df.columns:
        ema_long = ema_long.fillna(df["EMA_100_W"])
    if "EMA_50_W" in df.columns:
        ema_long = ema_long.fillna(df["EMA_50_W"])
    return ema_long


def compute_long_signal_mask(df: pd.DataFrame) -> pd.Series:
    """Compute boolean mask for LONG signals using your logic adapted to 15m suffix."""
    if "EMA_LONG_W" not in df.columns:
        df["EMA_LONG_W"] = make_weekly_ema_long(df)

    req = [
        "close_W", "EMA_50_W", "EMA_LONG_W", "RSI_W", "ADX_W", "Daily_Change_prev_D",
        f"close{INTRA_SUFFIX}", f"EMA_200{INTRA_SUFFIX}", f"EMA_50{INTRA_SUFFIX}", f"RSI{INTRA_SUFFIX}",
        f"MACD_Hist{INTRA_SUFFIX}", f"Stoch_%K{INTRA_SUFFIX}", f"Stoch_%D{INTRA_SUFFIX}",
        f"ATR{INTRA_SUFFIX}", f"ATR{INTRA_SUFFIX}_5bar_mean",
        f"ADX{INTRA_SUFFIX}", f"ADX{INTRA_SUFFIX}_5bar_mean",
        f"VWAP{INTRA_SUFFIX}", f"Recent_High_5{INTRA_SUFFIX}",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return pd.Series(False, index=df.index)

    # WEEKLY
    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_LONG_W"]) &
        (df["RSI_W"] > 40.0) &
        (df["ADX_W"] > 10.0)
    )
    weekly_mild = (
        (df["close_W"] > df["EMA_LONG_W"] * 1.01) &
        (df["RSI_W"].between(40.0, 68.0)) &
        (df["ADX_W"] > 9.0)
    )
    weekly_ok = (weekly_strong | weekly_mild) & (df["RSI_W"] < 88.0)

    # DAILY prev day move
    daily_prev = df["Daily_Change_prev_D"]
    daily_ok = (daily_prev >= 2.0) & (daily_prev <= 9.0)

    # INTRADAY
    close_i = df[f"close{INTRA_SUFFIX}"]
    ema200_i = df[f"EMA_200{INTRA_SUFFIX}"]
    ema50_i = df[f"EMA_50{INTRA_SUFFIX}"]

    rsi_i = df[f"RSI{INTRA_SUFFIX}"]
    rsi_prev = df.get(f"RSI{INTRA_SUFFIX}_prev", np.nan)

    macd_hist = df[f"MACD_Hist{INTRA_SUFFIX}"]
    macd_prev = df.get(f"MACD_Hist{INTRA_SUFFIX}_prev", np.nan)

    stoch_k = df[f"Stoch_%K{INTRA_SUFFIX}"]
    stoch_d = df[f"Stoch_%D{INTRA_SUFFIX}"]

    atr = df[f"ATR{INTRA_SUFFIX}"]
    atr_mean = df[f"ATR{INTRA_SUFFIX}_5bar_mean"]

    adx = df[f"ADX{INTRA_SUFFIX}"]
    adx_mean = df[f"ADX{INTRA_SUFFIX}_5bar_mean"]

    vwap = df[f"VWAP{INTRA_SUFFIX}"]
    recent_high_5 = df[f"Recent_High_5{INTRA_SUFFIX}"]

    trend_ok = (close_i >= ema200_i * 0.995) & (close_i >= ema50_i * 0.995)

    rsi_ok = (rsi_i > 49.0) | ((rsi_i > 46.0) & (rsi_i > rsi_prev))
    macd_ok = (macd_hist > -0.055) & (macd_hist >= macd_prev * 0.75)
    stoch_ok = (stoch_k >= stoch_d * 0.98) & (stoch_k >= 18.0)
    momentum_ok = rsi_ok & macd_ok & stoch_ok

    vol_ok = (atr >= atr_mean * 0.92) & ((adx >= 10.0) | (adx >= adx_mean))

    price_above_vwap = close_i >= vwap
    breakout_strict = close_i >= recent_high_5
    breakout_near = close_i >= recent_high_5 * 0.99

    intra_chg = df.get(f"Intra_Change{INTRA_SUFFIX}", 0.0)
    intra_chg = pd.to_numeric(intra_chg, errors="coerce").fillna(0.0)
    strong_up = intra_chg >= 0.05

    breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up & (rsi_i >= 47.0)))

    intraday_ok = trend_ok & momentum_ok & vol_ok & breakout_ok

    return weekly_ok & daily_ok & intraday_ok


def build_signals_for_ticker(ticker: str, debug_first: bool = False) -> List[dict]:
    """Generate candidate LONG signals for one ticker using 15m + daily + weekly."""
    path_i = os.path.join(DIR_INTRA, f"{ticker}{INTRA_FILE_ENDING}")
    path_d = os.path.join(DIR_D, f"{ticker}{D_FILE_ENDING}")
    path_w = os.path.join(DIR_W, f"{ticker}{W_FILE_ENDING}")

    df_i_raw = read_tf_csv(path_i)
    df_d_raw = read_tf_csv(path_d)
    df_w_raw = read_tf_csv(path_w)

    if df_i_raw.empty or df_d_raw.empty or df_w_raw.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # Daily prev
    if "Daily_Change" in df_d_raw.columns:
        df_d_raw["Daily_Change_prev"] = df_d_raw["Daily_Change"].shift(1)
    else:
        df_d_raw["Daily_Change_prev"] = np.nan

    # Suffix columns
    df_i = suffix_columns(df_i_raw, INTRA_SUFFIX)
    df_d = suffix_columns(df_d_raw, "_D")
    df_w = suffix_columns(df_w_raw, "_W")

    df_i = add_intraday_features(df_i)

    df_idw = merge_daily_and_weekly_onto_intraday(df_i, df_d, df_w)
    if df_idw.empty:
        return []

    df_idw["EMA_LONG_W"] = make_weekly_ema_long(df_idw)

    # Drop rows missing key context (this gates earliest possible signals)
    df_idw = df_idw.dropna(subset=[
        "close_W", "EMA_50_W", "EMA_LONG_W", "RSI_W", "ADX_W",
        "Daily_Change_prev_D",
        f"close{INTRA_SUFFIX}",
    ])
    if df_idw.empty:
        return []

    if debug_first:
        first_ok = df_idw["date"].min()
        first_ema200 = df_idw.loc[df_idw.get("EMA_200_W", pd.Series(index=df_idw.index)).notna(), "date"].min() \
            if "EMA_200_W" in df_idw.columns else pd.NaT
        print(f"[DEBUG {ticker}] intraday range: {df_i_raw['date'].min()} -> {df_i_raw['date'].max()}")
        print(f"[DEBUG {ticker}] daily range   : {df_d_raw['date'].min()} -> {df_d_raw['date'].max()}")
        print(f"[DEBUG {ticker}] weekly range  : {df_w_raw['date'].min()} -> {df_w_raw['date'].max()}")
        print(f"[DEBUG {ticker}] first row after required non-NaN merge: {first_ok}")
        if pd.notna(first_ema200):
            print(f"[DEBUG {ticker}] first non-NaN EMA_200_W (if present): {first_ema200}")
        print(f"[DEBUG {ticker}] weekly EMA fallback enabled: {ALLOW_WEEKLY_EMA_FALLBACK}")

    long_mask = compute_long_signal_mask(df_idw)

    signals: List[dict] = []
    for _, r in df_idw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r[f"close{INTRA_SUFFIX}"]),
            "weekly_ok": True,
            "daily_ok": True,
            "intraday_ok": True,
        })
    return signals


# =============================================================================
# SIGNAL EVALUATION (exit_time, MTM for unrealised)
# =============================================================================

def read_intra_ohlc(path: str) -> pd.DataFrame:
    """Read intraday OHLC (needs date, high, low, close)."""
    return read_tf_csv(path, tz=IST)


def evaluate_long_signal(row: pd.Series, df_intra: pd.DataFrame) -> dict:
    """Evaluate one LONG signal on intraday OHLC."""
    entry_time = row["signal_time_ist"]
    entry_price = float(row["entry_price"])

    base = {
        HIT_COL: False,
        TIME_TD_COL: np.nan,
        TIME_CAL_COL: np.nan,
        "exit_time": pd.NaT,
        "exit_price": np.nan,
        "trade_status": "UNREALIZED_OPEN",
        "is_unrealized": True,
        "mtm_time": pd.NaT,
        "mtm_price": np.nan,
        "mtm_pnl_pct": np.nan,
        "mtm_pnl_rs": np.nan,
        "mtm_value": np.nan,
        "pnl_pct": np.nan,
        "pnl_rs": np.nan,
        "max_favorable_pct": np.nan,
        "max_adverse_pct": np.nan,
    }

    if not np.isfinite(entry_price) or entry_price <= 0:
        return base

    if df_intra.empty or "date" not in df_intra.columns:
        base.update({
            "mtm_time": entry_time,
            "mtm_price": entry_price,
            "mtm_pnl_pct": 0.0,
            "mtm_pnl_rs": 0.0,
            "mtm_value": float(CAPITAL_PER_SIGNAL_RS),
            "pnl_pct": 0.0,
            "pnl_rs": 0.0,
        })
        return base

    future = df_intra[df_intra["date"] > entry_time].copy()
    if future.empty:
        base.update({
            "mtm_time": entry_time,
            "mtm_price": entry_price,
            "mtm_pnl_pct": 0.0,
            "mtm_pnl_rs": 0.0,
            "mtm_value": float(CAPITAL_PER_SIGNAL_RS),
            "pnl_pct": 0.0,
            "pnl_rs": 0.0,
        })
        return base

    for c in ["high", "low", "close"]:
        if c not in future.columns:
            base.update({
                "mtm_time": entry_time,
                "mtm_price": entry_price,
                "mtm_pnl_pct": 0.0,
                "mtm_pnl_rs": 0.0,
                "mtm_value": float(CAPITAL_PER_SIGNAL_RS),
                "pnl_pct": 0.0,
                "pnl_rs": 0.0,
            })
            return base

    target_price = entry_price * TARGET_MULT
    hit_mask = future["high"] >= target_price

    max_fav = ((future["high"] - entry_price) / entry_price * 100.0).max()
    max_adv = ((future["low"] - entry_price) / entry_price * 100.0).min()

    if bool(hit_mask.any()):
        first_hit_idx = hit_mask[hit_mask].index[0]
        hit_time = future.loc[first_hit_idx, "date"]

        exit_time = hit_time
        exit_price = float(target_price)

        dt_hours = (hit_time - entry_time).total_seconds() / 3600.0
        time_to_target_days = dt_hours / TRADING_HOURS_PER_DAY
        days_to_target_calendar = (hit_time.date() - entry_time.date()).days

        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        pnl_rs = float(CAPITAL_PER_SIGNAL_RS) * (pnl_pct / 100.0)
        value = float(CAPITAL_PER_SIGNAL_RS) * (1.0 + pnl_pct / 100.0)

        return {
            HIT_COL: True,
            TIME_TD_COL: float(time_to_target_days),
            TIME_CAL_COL: float(days_to_target_calendar),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "trade_status": "REALIZED_TARGET",
            "is_unrealized": False,
            "mtm_time": exit_time,
            "mtm_price": exit_price,
            "mtm_pnl_pct": float(pnl_pct),
            "mtm_pnl_rs": float(pnl_rs),
            "mtm_value": float(value),
            "pnl_pct": float(pnl_pct),
            "pnl_rs": float(pnl_rs),
            "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
            "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
        }

    # UNREALIZED: MTM at last close
    last_row = future.iloc[-1]
    mtm_time = last_row["date"]
    mtm_price = float(last_row["close"])

    mtm_pnl_pct = (mtm_price - entry_price) / entry_price * 100.0
    mtm_pnl_rs = float(CAPITAL_PER_SIGNAL_RS) * (mtm_pnl_pct / 100.0)
    mtm_value = float(CAPITAL_PER_SIGNAL_RS) * (1.0 + mtm_pnl_pct / 100.0)

    base.update({
        "mtm_time": mtm_time,
        "mtm_price": mtm_price,
        "mtm_pnl_pct": float(mtm_pnl_pct),
        "mtm_pnl_rs": float(mtm_pnl_rs),
        "mtm_value": float(mtm_value),
        "pnl_pct": float(mtm_pnl_pct),
        "pnl_rs": float(mtm_pnl_rs),
        "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
        "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
    })
    return base


# =============================================================================
# PORTFOLIO SIMULATION (capital constraint + one position per ticker)
# =============================================================================

def apply_portfolio_rules(evaluated_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Capital + ticker lock simulation."""
    df = evaluated_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()
    cash = float(MAX_CAPITAL_RS)

    open_trades: List[dict] = []
    open_by_ticker: Dict[str, pd.Timestamp] = {}

    df["taken"] = False
    df["skip_reason"] = ""

    for i, row in df.iterrows():
        t = str(row["ticker"])
        signal_time = row["signal_time_ist"]

        # release realized
        still_open = []
        open_by_ticker_new: Dict[str, pd.Timestamp] = {}
        for tr in open_trades:
            rel_time = tr["release_time"]
            if rel_time <= signal_time:
                cash += float(tr["return_value"])
            else:
                still_open.append(tr)
                open_by_ticker_new[str(tr["ticker"])] = rel_time
        open_trades = still_open
        open_by_ticker = open_by_ticker_new

        # ticker lock
        if ONE_POSITION_PER_TICKER and (t in open_by_ticker) and (open_by_ticker[t] > signal_time):
            df.at[i, "taken"] = False
            df.at[i, "skip_reason"] = "TICKER_OPEN"
            continue

        # capital constraint
        if cash < float(CAPITAL_PER_SIGNAL_RS):
            df.at[i, "taken"] = False
            df.at[i, "skip_reason"] = "NO_CAPITAL"
            continue

        # take
        df.at[i, "taken"] = True
        df.at[i, "skip_reason"] = ""
        cash -= float(CAPITAL_PER_SIGNAL_RS)

        is_unreal = bool(row.get("is_unrealized", False))
        mtm_value = float(row.get("mtm_value", float(CAPITAL_PER_SIGNAL_RS)))

        if is_unreal:
            release_time = FAR_FUTURE_LOCK
        else:
            exit_time = row.get("exit_time", pd.NaT)
            if pd.isna(exit_time):
                is_unreal = True
                release_time = FAR_FUTURE_LOCK
            else:
                release_time = exit_time

        open_trades.append({
            "ticker": t,
            "release_time": release_time,
            "return_value": mtm_value,
            "is_unrealized": is_unreal,
        })
        open_by_ticker[t] = release_time

    # end
    open_positions_value = 0.0
    for tr in open_trades:
        if tr["is_unrealized"]:
            open_positions_value += float(tr["return_value"])
        else:
            cash += float(tr["return_value"])

    final_equity = cash + open_positions_value
    net_pnl_rs = final_equity - float(MAX_CAPITAL_RS)
    net_pnl_pct = (net_pnl_rs / float(MAX_CAPITAL_RS) * 100.0) if MAX_CAPITAL_RS > 0 else 0.0

    df["invested_amount_constrained"] = np.where(df["taken"], float(CAPITAL_PER_SIGNAL_RS), 0.0)
    df["value_constrained"] = np.where(df["taken"], df.get("mtm_value", np.nan), 0.0)
    df["final_value_constrained"] = df["value_constrained"]

    taken = df[df["taken"]].copy()
    unreal_taken = taken[taken.get("is_unrealized", False)].copy()
    real_taken = taken[~taken.get("is_unrealized", False)].copy()

    realised_pnl_rs = float(real_taken.get("pnl_rs", pd.Series(dtype=float)).sum()) if not real_taken.empty else 0.0
    unrealised_pnl_rs = float(unreal_taken.get("mtm_pnl_rs", pd.Series(dtype=float)).sum()) if not unreal_taken.empty else 0.0

    summary = {
        "total_signals": int(len(df)),
        "signals_taken": int(len(taken)),
        "signals_blocked_ticker_open": int((df["skip_reason"] == "TICKER_OPEN").sum()),
        "signals_skipped_no_capital": int((df["skip_reason"] == "NO_CAPITAL").sum()),
        "realised_trades": int(len(real_taken)),
        "unrealised_trades": int(len(unreal_taken)),
        "realised_pnl_rs": float(realised_pnl_rs),
        "unrealised_pnl_rs": float(unrealised_pnl_rs),
        "cash_end": float(cash),
        "open_positions_value": float(open_positions_value),
        "final_equity": float(final_equity),
        "net_pnl_rs": float(net_pnl_rs),
        "net_pnl_pct": float(net_pnl_pct),
    }
    return df, summary


def suppress_signals_during_open_positions(df_with_rules: pd.DataFrame) -> pd.DataFrame:
    """Remove signals blocked because ticker was already open."""
    if df_with_rules.empty or "skip_reason" not in df_with_rules.columns:
        return df_with_rules
    return df_with_rules[df_with_rules["skip_reason"] != "TICKER_OPEN"].copy()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    tickers_i = list_tickers(DIR_INTRA, INTRA_FILE_ENDING)
    tickers_d = list_tickers(DIR_D, D_FILE_ENDING)
    tickers_w = list_tickers(DIR_W, W_FILE_ENDING)

    tickers = tickers_i & tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across 15m, Daily, Weekly directories.")
        return

    print(f"[INFO] Found {len(tickers)} tickers with all 3 timeframes.")

    # Generate signals
    all_signals: List[dict] = []
    for idx, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{idx}/{len(tickers)}] Generating signals: {ticker}")
        try:
            sigs = build_signals_for_ticker(ticker, debug_first=(idx == 1))
            all_signals.extend(sigs)
            print(f"    -> {len(sigs)} signals")
        except Exception as e:
            print(f"! Error while processing {ticker}: {e}")

    if not all_signals:
        print("No LONG signals generated.")
        return

    sig_df = pd.DataFrame(all_signals).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    sig_df["itr"] = np.arange(1, len(sig_df) + 1)
    sig_df["signal_date"] = sig_df["signal_time_ist"].dt.tz_convert(IST).dt.date

    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")

    # Evaluate signals
    print(f"\n[INFO] Evaluating +{TARGET_LABEL} outcome on signals (LONG only) using {INTRA_TF_LABEL} OHLC.")
    tickers_in_signals = sig_df["ticker"].unique().tolist()
    intra_cache: Dict[str, pd.DataFrame] = {}

    for t in tickers_in_signals:
        path = os.path.join(DIR_INTRA, f"{t}{INTRA_FILE_ENDING}")
        df_intra = read_intra_ohlc(path)
        if df_intra.empty:
            print(f"- No {INTRA_TF_LABEL} data for {t}, its signals will be MTM flat at entry.")
        intra_cache[t] = df_intra

    evaluated_rows: List[dict] = []
    for _, row in sig_df.iterrows():
        t = row["ticker"]
        df_intra = intra_cache.get(t, pd.DataFrame())
        eval_info = evaluate_long_signal(row, df_intra) if not df_intra.empty else {
            HIT_COL: False,
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "trade_status": "UNREALIZED_OPEN",
            "is_unrealized": True,
            "mtm_time": row["signal_time_ist"],
            "mtm_price": row["entry_price"],
            "mtm_pnl_pct": 0.0,
            "mtm_pnl_rs": 0.0,
            "mtm_value": float(CAPITAL_PER_SIGNAL_RS),
            "pnl_pct": 0.0,
            "pnl_rs": 0.0,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }
        merged = row.to_dict()
        merged.update(eval_info)
        evaluated_rows.append(merged)

    out_df = pd.DataFrame(evaluated_rows).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    out_df["invested_amount"] = float(CAPITAL_PER_SIGNAL_RS)
    out_df["final_value_per_signal"] = out_df.get("mtm_value", np.nan)

    # Portfolio simulation
    print("\n[INFO] Portfolio simulation (cash + ticker lock + unrealised MTM).")
    print(f"  Max capital pool           : {fmt_rs(MAX_CAPITAL_RS)}")
    print(f"  Capital per trade          : {fmt_rs(CAPITAL_PER_SIGNAL_RS)}")
    print(f"  One position per ticker    : {ONE_POSITION_PER_TICKER}")
    print(f"  Weekly EMA fallback        : {ALLOW_WEEKLY_EMA_FALLBACK}")

    out_df_ruled, summary = apply_portfolio_rules(out_df)

    print("\n[SUMMARY BEFORE SUPPRESSION]")
    print(f"  Candidate signals (all)    : {summary['total_signals']}")
    print(f"  Taken trades               : {summary['signals_taken']}")
    print(f"  Blocked (ticker open)      : {summary['signals_blocked_ticker_open']}")
    print(f"  Skipped (no capital)       : {summary['signals_skipped_no_capital']}")
    print(f"  Realised trades            : {summary['realised_trades']}  | P&L: {fmt_rs(summary['realised_pnl_rs'])}")
    print(f"  Unrealised trades          : {summary['unrealised_trades']} | MTM P&L: {fmt_rs(summary['unrealised_pnl_rs'])}")
    print(f"  Cash (end)                 : {fmt_rs(summary['cash_end'])}")
    print(f"  Open positions value (MTM) : {fmt_rs(summary['open_positions_value'])}")
    print(f"  Final equity               : {fmt_rs(summary['final_equity'])}")
    print(f"  Net P&L                    : {fmt_rs(summary['net_pnl_rs'])} ({summary['net_pnl_pct']:.2f}%)")

    # Suppress blocked signals
    out_df_final = suppress_signals_during_open_positions(out_df_ruled)
    out_df_final = out_df_final.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    out_df_final["itr"] = np.arange(1, len(out_df_final) + 1)
    out_df_final["signal_date"] = pd.to_datetime(out_df_final["signal_time_ist"], errors="coerce").dt.tz_convert(IST).dt.date

    print("\n[SUMMARY AFTER SUPPRESSION]")
    print(f"  Signals kept (output)      : {len(out_df_final)}  (suppressed TICKER_OPEN rows removed)")

    # Save outputs
    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_{ts}_IST.csv")
    out_df_final[[
        "itr", "signal_time_ist", "signal_date", "ticker", "signal_side", "entry_price",
        "weekly_ok", "daily_ok", "intraday_ok"
    ]].to_csv(signals_path, index=False)

    daily_counts = (
        out_df_final.groupby("signal_date").size().reset_index(name="LONG_signals").rename(columns={"signal_date": "date"})
    )
    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_etf_daily_counts_{ts}_IST.csv")
    daily_counts.to_csv(daily_counts_path, index=False)

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_{EVAL_SUFFIX}_{ts}_IST.csv")
    out_df_final.to_csv(eval_path, index=False)

    print("\n[FILES SAVED]")
    print(f"  Signals         : {signals_path}")
    print(f"  Daily counts    : {daily_counts_path}")
    print(f"  Evaluation      : {eval_path}")

    # Prints
    taken_df = out_df_final[out_df_final["taken"]].copy().sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    print("\n================ TAKEN TRADES (REAL ENTRIES) ================")
    print(f"Taken trades: {len(taken_df)} / {len(out_df_final)} signals kept")

    if taken_df.empty:
        print("No trades were taken under the current rules.")
        return

    cols_taken = [
        "itr", "signal_time_ist", "signal_date", "ticker", "entry_price",
        "trade_status", "exit_time", "exit_price",
        "mtm_time", "mtm_price", "mtm_pnl_pct", "mtm_pnl_rs", "mtm_value",
        "max_favorable_pct", "max_adverse_pct",
        "invested_amount_constrained", "value_constrained",
    ]
    cols_taken = [c for c in cols_taken if c in taken_df.columns]
    disp = taken_df[cols_taken].copy()
    for c in ["signal_time_ist", "exit_time", "mtm_time"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(fmt_dt)

    max_print = 200
    if len(disp) <= max_print:
        print(disp.to_string(index=False))
    else:
        print(f"(Showing last {max_print} of {len(disp)} taken trades)")
        print(disp.tail(max_print).to_string(index=False))

    unreal_df = taken_df[taken_df.get("is_unrealized", False)].copy().reset_index(drop=True)
    print("\n================ UNREALISED (OPEN) TRADES - DETAILS ================")
    print(f"Open trades: {len(unreal_df)}")
    if unreal_df.empty:
        print("No unrealised open trades.")
        return

    cols_open = [
        "signal_time_ist", "ticker", "entry_price",
        "mtm_time", "mtm_price", "mtm_pnl_pct", "mtm_pnl_rs", "mtm_value",
        "max_favorable_pct", "max_adverse_pct",
        "invested_amount_constrained",
    ]
    cols_open = [c for c in cols_open if c in unreal_df.columns]
    disp_open = unreal_df[cols_open].copy()
    for c in ["signal_time_ist", "mtm_time"]:
        if c in disp_open.columns:
            disp_open[c] = disp_open[c].apply(fmt_dt)

    disp_open["mtm_value_fmt"] = disp_open.get("mtm_value", np.nan).apply(fmt_rs)
    disp_open["mtm_pnl_rs_fmt"] = disp_open.get("mtm_pnl_rs", np.nan).apply(fmt_rs)
    print(disp_open.to_string(index=False))


if __name__ == "__main__":
    main()
