# -*- coding: utf-8 -*-
"""
ALGO-SM1 ETF Multi‑TF LONG signal generator + evaluator + portfolio simulator (beginner-friendly).

KEY BEHAVIOR (updated as requested)
- NO TICKER LOCK:
    The simulator can take MULTIPLE trades from the SAME ticker,
    even if older trades for that ticker are still open/unrealised.
    (Only global capital constraint applies.)

- 1H USED FOR SIGNAL TRIGGER + EVALUATION:
    Signals are generated on 1H bars (with Weekly + Daily context merged onto each 1H bar),
    and evaluation also uses 1H OHLC to check target hits / MTM.

- UNREALISED (OPEN) TRADES:
    If a TAKEN trade does NOT hit the target within the available 1H data,
    we do NOT force-close it at the last close.
    Instead we treat it as an OPEN / UNREALISED position:
        - exit_time  = NaT
        - exit_price = NaN
        - We compute MTM (mark-to-market) using the last available close:
              mtm_time, mtm_price, mtm_pnl_pct, mtm_pnl_rs, mtm_value

    Portfolio summary reports:
        - Realised P&L (target hits)
        - Unrealised P&L (MTM of open positions)
        - Final equity = cash + MTM value of open positions

WHAT THIS SCRIPT DOES (high level)
1) Signal generation per ticker (1H trigger):
   - Load Weekly, Daily, and 1H indicator CSVs.
   - Merge Weekly + Daily context onto each 1H bar (no lookahead for weekly).
   - Apply filters:
        a) Weekly trend filter
        b) Daily previous-day momentum filter
        c) Hourly timing trigger
   - Produce candidate LONG signals (timestamp + entry price).

2) Evaluate each signal (1H evaluation):
   - If future high hits +TARGET_PCT% → exit at target at first hit time (REALIZED_TARGET).
   - Else → OPEN / UNREALISED (mark-to-market at last available close).

3) Portfolio simulation:
   - Global cash pool MAX_CAPITAL_RS.
   - Each taken trade uses CAPITAL_PER_SIGNAL_RS.
   - Trades can overlap (even same ticker).
   - Capital constraint only: skip if not enough cash at that signal time.
   - Realised trades release cash at exit_time; unrealised never release (for this run).

OUTPUT FILES (OUT_DIR)
- multi_tf_signals_<timestamp>_IST.parquet
- multi_tf_signals_etf_daily_counts_<timestamp>_IST.parquet
- multi_tf_signals_<TARGET>pct_eval_<timestamp>_IST.parquet

TIMEZONES
- CSV 'date' is parsed as UTC and converted to IST.

"""

from __future__ import annotations

import os
import glob
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz

# =============================================================================
# TIMEZONE (define BEFORE any function default uses it)
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

# =============================================================================
# PARQUET I/O HELPERS
# =============================================================================
PARQUET_ENGINE = "pyarrow"

def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Parquet support requires 'pyarrow'. Install it with: pip install pyarrow"
        ) from e


def read_tf_parquet(path: str, tz=None, fallback_csv: bool = True) -> pd.DataFrame:
    """
    Read a timeframe Parquet file and return a DataFrame sorted by 'date' (tz-aware IST).

    - Primary format: Parquet
    - Optional fallback: CSV (same path but .csv), if fallback_csv=True
    """
    _require_pyarrow()
    if tz is None:
        tz = IST  # default safely AFTER IST exists

    try:
        if os.path.exists(path):
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
        elif fallback_csv:
            csv_path = re.sub(r"\.parquet$", ".csv", path)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")

        s = df["date"]

        # Normalize to tz-aware IST
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
            # if naive -> assume UTC then convert
            if getattr(dt.dt, "tz", None) is None:
                dt = dt.dt.tz_localize("UTC")
            dt = dt.dt.tz_convert(tz)
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(tz)

        df["date"] = dt
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to Parquet (no index). Creates parent directories if missing."""
    _require_pyarrow()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False, engine=PARQUET_ENGINE)



import pandas as pd
import pytz


# =============================================================================
# CONFIG
# =============================================================================

# Where your indicator CSVs live
DIR_1H = "etf_indicators_1h_pq"
DIR_D  = "etf_indicators_daily_pq"
DIR_W  = "etf_indicators_weekly_pq"

# Where outputs will be written
OUT_DIR = "etf_signals"
os.makedirs(OUT_DIR, exist_ok=True)

# Timezone
IST = pytz.timezone("Asia/Kolkata")

# ---- Target configuration (single source of truth) ----
TARGET_PCT: float = 6.0
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)            # used in filenames (assumes integer target)
TARGET_LABEL: str = f"{TARGET_INT}%"

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 100_000.0
MAX_CAPITAL_RS: float = 3000_000.0

# A "far future" timestamp used internally to keep cash tied up for an unrealised trade
# (We do NOT write this into exit_time; exit_time stays NaT for unrealised trades.)
FAR_FUTURE_LOCK = pd.Timestamp("2099-12-31 23:59:59", tz=IST)

# Derived column names based on target
HIT_COL: str       = f"hit_{TARGET_INT}pct"
TIME_TD_COL: str   = f"time_to_{TARGET_INT}pct_days"
TIME_CAL_COL: str  = f"days_to_{TARGET_INT}pct_calendar"
EVAL_SUFFIX: str   = f"{TARGET_INT}pct_eval"

# For time-to-target in “trading days”
TRADING_HOURS_PER_DAY: float = 6.25


# =============================================================================
# SMALL UTILITIES
# =============================================================================


def read_tf_csv(path: str, tz=IST) -> pd.DataFrame:
    """
    Backwards-compatible name: reads Parquet by default.
    If a .parquet doesn't exist but a legacy .csv exists (same basename),
    we fall back to CSV automatically.
    """
    if path.endswith(".csv"):
        path = path[:-4] + ".parquet"
    return read_tf_parquet(path, tz=tz, fallback_csv=True)

def suffix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Rename all columns except 'date' by adding a suffix."""
    if df.empty:
        return df
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)


def list_tickers(directory: str, ending: str) -> set[str]:
    """List tickers in a directory by stripping a known filename ending."""
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
    """Pretty rupee formatting."""
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "₹NA"


# =============================================================================
# SIGNAL GENERATION (Weekly + Daily(prev) + 1H trigger)
# =============================================================================


def add_hourly_features(df_h: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns used in hourly trigger (rolling/shift features)."""
    if df_h.empty:
        return df_h

    df_h["ATR_H_5bar_mean"] = df_h.get("ATR_H", np.nan).rolling(5).mean()
    df_h["RSI_H_prev"] = df_h.get("RSI_H", np.nan).shift(1)
    df_h["MACD_Hist_H_prev"] = df_h.get("MACD_Hist_H", np.nan).shift(1)
    df_h["Recent_High_5_H"] = df_h.get("high_H", np.nan).rolling(5).max()
    df_h["ADX_H_5bar_mean"] = df_h.get("ADX_H", np.nan).rolling(5).mean()
    df_h["Stoch_%K_H_prev"] = df_h.get("Stoch_%K_H", np.nan).shift(1)

    return df_h


def merge_daily_and_weekly_onto_hourly(df_h: pd.DataFrame, df_d: pd.DataFrame, df_w: pd.DataFrame) -> pd.DataFrame:
    """Merge Daily + Weekly context onto each 1H row using as-of logic (no weekly lookahead)."""
    if df_h.empty or df_d.empty or df_w.empty:
        return pd.DataFrame()

    df_hd = pd.merge_asof(
        df_h.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward",
    )

    # Weekly lookahead guard: shift by +1s
    df_w_shift = df_w.sort_values("date").copy()
    df_w_shift["date"] += pd.Timedelta(seconds=1)

    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_shift,
        on="date",
        direction="backward",
    )

    return df_hdw


def compute_long_signal_mask(df: pd.DataFrame) -> pd.Series:
    """Compute boolean mask for LONG signals (same logic as before)."""
    req = [
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W", "Daily_Change_prev_D",
        "close_H", "EMA_200_H", "EMA_50_H", "RSI_H", "MACD_Hist_H", "Stoch_%K_H", "Stoch_%D_H",
        "ATR_H", "ATR_H_5bar_mean", "ADX_H", "ADX_H_5bar_mean", "VWAP_H", "Recent_High_5_H"
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return pd.Series(False, index=df.index)

    # -------------------- WEEKLY --------------------
    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 40.0) &
        (df["ADX_W"] > 10.0)
    )

    weekly_mild = (
        (df["close_W"] > df["EMA_200_W"] * 1.01) &
        (df["RSI_W"].between(40.0, 68.0)) &
        (df["ADX_W"] > 9.0)
    )

    weekly_not_extreme = df["RSI_W"] < 88.0
    weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme

    # -------------------- DAILY (prev day move) --------------------
    daily_prev = df["Daily_Change_prev_D"]
    daily_ok = (daily_prev >= 2.0) & (daily_prev <= 9.0)

    # -------------------- HOURLY --------------------
    hourly_trend = (
        (df["close_H"] >= df["EMA_200_H"] * 0.995) &
        (df["close_H"] >= df["EMA_50_H"] * 0.995)
    )

    rsi_ok = (
        (df["RSI_H"] > 49.0) |
        ((df["RSI_H"] > 46.0) & (df["RSI_H"] > df["RSI_H_prev"]))
    )

    macd_ok = (
        (df["MACD_Hist_H"] > -0.055) &
        (df["MACD_Hist_H"] >= df["MACD_Hist_H_prev"] * 0.75)
    )

    stoch_ok = (
        (df["Stoch_%K_H"] >= df["Stoch_%D_H"] * 0.98) &
        (df["Stoch_%K_H"] >= 18.0)
    )

    momentum_ok = rsi_ok & macd_ok & stoch_ok

    hourly_vol_ok = (
        (df["ATR_H"] >= df["ATR_H_5bar_mean"] * 0.92) &
        ((df["ADX_H"] >= 10.0) | (df["ADX_H"] >= df["ADX_H_5bar_mean"]))
    )

    price_above_vwap = df["close_H"] >= df["VWAP_H"]

    breakout_strict = df["close_H"] >= df["Recent_High_5_H"]
    breakout_near   = df["close_H"] >= df["Recent_High_5_H"] * 0.99

    intra_chg = df.get("Intra_Change_H", 0.0)
    intra_chg = pd.to_numeric(intra_chg, errors="coerce").fillna(0.0)
    strong_up = intra_chg >= 0.05

    breakout_ok = price_above_vwap & (
        breakout_strict |
        (breakout_near & strong_up & (df["RSI_H"] >= 47.0))
    )

    hourly_ok = hourly_trend & momentum_ok & hourly_vol_ok & breakout_ok

    return weekly_ok & daily_ok & hourly_ok


def build_signals_for_ticker(ticker: str) -> List[dict]:
    """Generate candidate LONG signals for one ticker (1H trigger)."""
    path_1h = os.path.join(DIR_1H, f"{ticker}_etf_indicators_1h.parquet")
    path_d  = os.path.join(DIR_D,  f"{ticker}_etf_indicators_daily.parquet")
    path_w  = os.path.join(DIR_W,  f"{ticker}_etf_indicators_weekly.parquet")

    df_h = read_tf_csv(path_1h)
    df_d = read_tf_csv(path_d)
    df_w = read_tf_csv(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # Daily: previous day's Daily_Change
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = pd.to_numeric(df_d["Daily_Change"], errors="coerce").shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # Suffix columns
    df_h = suffix_columns(df_h, "_H")
    df_d = suffix_columns(df_d, "_D")
    df_w = suffix_columns(df_w, "_W")

    # Hourly helper features
    df_h = add_hourly_features(df_h)

    # Merge daily + weekly onto hourly
    df_hdw = merge_daily_and_weekly_onto_hourly(df_h, df_d, df_w)
    if df_hdw.empty:
        return []

    # Remove rows missing key context
    df_hdw = df_hdw.dropna(subset=[
        "close_W", "EMA_50_W", "EMA_200_W", "RSI_W", "ADX_W", "Daily_Change_prev_D",
        "close_H"
    ])
    if df_hdw.empty:
        return []

    long_mask = compute_long_signal_mask(df_hdw)

    signals: List[dict] = []
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


# =============================================================================
# SIGNAL EVALUATION (1H OHLC, exit_time + MTM for unrealised)
# =============================================================================


def read_1h_ohlc(path: str) -> pd.DataFrame:
    """Read 1H OHLC (for evaluation). Needs date, high, low, close columns."""
    return read_tf_csv(path, tz=IST)


def evaluate_long_signal(row: pd.Series, df_1h: pd.DataFrame) -> dict:
    """Evaluate one LONG signal using 1H bars."""
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

    future = df_1h[df_1h["date"] > entry_time].copy()
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

    # Require OHLC columns
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

    # Not hit => unrealised open (MTM at last close)
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
# PORTFOLIO SIMULATION (capital constraint only; NO ticker lock)
# =============================================================================


def apply_portfolio_rules(evaluated_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply portfolio rules: capital constraint only (multiple trades per ticker allowed)."""
    df = evaluated_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()
    cash = float(MAX_CAPITAL_RS)

    # Track active trades only to release realized cash at exit_time
    open_trades: List[dict] = []

    df["taken"] = False
    df["skip_reason"] = ""  # only "NO_CAPITAL" now

    for i, row in df.iterrows():
        signal_time = row["signal_time_ist"]

        # Release realized trades that have exited by now
        still_open: List[dict] = []
        for tr in open_trades:
            if tr["release_time"] <= signal_time:
                cash += float(tr["return_value"])
            else:
                still_open.append(tr)
        open_trades = still_open

        # Capital rule
        if cash < float(CAPITAL_PER_SIGNAL_RS):
            df.at[i, "taken"] = False
            df.at[i, "skip_reason"] = "NO_CAPITAL"
            continue

        # Take trade
        df.at[i, "taken"] = True
        df.at[i, "skip_reason"] = ""
        cash -= float(CAPITAL_PER_SIGNAL_RS)

        is_unreal = bool(row.get("is_unrealized", False))
        if is_unreal:
            release_time = FAR_FUTURE_LOCK
            return_value = float(row.get("mtm_value", float(CAPITAL_PER_SIGNAL_RS)))
        else:
            exit_time = row.get("exit_time", pd.NaT)
            if pd.isna(exit_time):
                # Defensive: treat as unrealised
                is_unreal = True
                release_time = FAR_FUTURE_LOCK
                return_value = float(row.get("mtm_value", float(CAPITAL_PER_SIGNAL_RS)))
            else:
                release_time = exit_time
                return_value = float(row.get("mtm_value", float(CAPITAL_PER_SIGNAL_RS)))

        open_trades.append({
            "release_time": release_time,
            "return_value": return_value,
            "is_unrealized": is_unreal,
        })

    # After last signal: compute open positions value + final equity
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

    total_signals = int(len(df))
    taken = df[df["taken"]].copy()

    unrealised_taken = taken[taken.get("is_unrealized", False)].copy()
    realised_taken = taken[~taken.get("is_unrealized", False)].copy()

    realised_pnl_rs = float(realised_taken.get("pnl_rs", pd.Series(dtype=float)).sum()) if not realised_taken.empty else 0.0
    unrealised_pnl_rs = float(unrealised_taken.get("mtm_pnl_rs", pd.Series(dtype=float)).sum()) if not unrealised_taken.empty else 0.0

    skipped_capital = int((df["skip_reason"] == "NO_CAPITAL").sum())

    summary = {
        "total_signals": total_signals,
        "signals_taken": int(len(taken)),
        "signals_skipped_no_capital": skipped_capital,
        "realised_trades": int(len(realised_taken)),
        "unrealised_trades": int(len(unrealised_taken)),
        "realised_pnl_rs": realised_pnl_rs,
        "unrealised_pnl_rs": unrealised_pnl_rs,
        "cash_end": float(cash),
        "open_positions_value": float(open_positions_value),
        "final_equity": float(final_equity),
        "net_pnl_rs": float(net_pnl_rs),
        "net_pnl_pct": float(net_pnl_pct),
    }

    return df, summary


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    # 1) Find tickers that have all required timeframes
    tickers_1h = list_tickers(DIR_1H, "_etf_indicators_1h.parquet")
    tickers_d  = list_tickers(DIR_D,  "_etf_indicators_daily.parquet")
    tickers_w  = list_tickers(DIR_W,  "_etf_indicators_weekly.parquet")

    tickers = tickers_1h & tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across 1H, Daily, and Weekly directories.")
        return

    print(f"[INFO] Found {len(tickers)} tickers with all 3 timeframes.")

    # 2) Generate candidate signals (per ticker)
    all_signals: List[dict] = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Generating signals (1H trigger): {ticker}")
        try:
            sigs = build_signals_for_ticker(ticker)
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

    # 3) Evaluate every signal (1H evaluation)
    print(f"\n[INFO] Evaluating +{TARGET_LABEL} outcome on signals (1H evaluation).")

    tickers_in_signals = sig_df["ticker"].unique().tolist()
    one_hour_cache: Dict[str, pd.DataFrame] = {}

    for t in tickers_in_signals:
        path = os.path.join(DIR_1H, f"{t}_etf_indicators_1h.parquet")
        df_1h = read_1h_ohlc(path)
        if df_1h.empty:
            print(f"- No 1H data for {t}, its signals will be MTM at entry (flat).")
        one_hour_cache[t] = df_1h

    evaluated_rows: List[dict] = []
    for _, row in sig_df.iterrows():
        t = row["ticker"]
        df_1h = one_hour_cache.get(t, pd.DataFrame())

        if df_1h.empty:
            eval_info = {
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
        else:
            eval_info = evaluate_long_signal(row, df_1h)

        merged = row.to_dict()
        merged.update(eval_info)
        evaluated_rows.append(merged)

    out_df = pd.DataFrame(evaluated_rows).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    out_df["invested_amount"] = float(CAPITAL_PER_SIGNAL_RS)
    out_df["final_value_per_signal"] = out_df.get("mtm_value", np.nan)

    # 4) Apply portfolio rules (capital only; no ticker lock)
    print("\n[INFO] Portfolio simulation (capital only; NO ticker lock).")
    print(f"  Max capital pool           : {fmt_rs(MAX_CAPITAL_RS)}")
    print(f"  Capital per trade          : {fmt_rs(CAPITAL_PER_SIGNAL_RS)}")

    out_df_ruled, summary = apply_portfolio_rules(out_df)

    print("\n[SUMMARY]")
    print(f"  Candidate signals (all)    : {summary['total_signals']}")
    print(f"  Taken trades               : {summary['signals_taken']}")
    print(f"  Skipped (no capital)       : {summary['signals_skipped_no_capital']}")
    print(f"  Realised trades            : {summary['realised_trades']}  | P&L: {fmt_rs(summary['realised_pnl_rs'])}")
    print(f"  Unrealised trades          : {summary['unrealised_trades']} | MTM P&L: {fmt_rs(summary['unrealised_pnl_rs'])}")
    print(f"  Cash (end)                 : {fmt_rs(summary['cash_end'])}")
    print(f"  Open positions value (MTM) : {fmt_rs(summary['open_positions_value'])}")
    print(f"  Final equity               : {fmt_rs(summary['final_equity'])}")
    print(f"  Net P&L                    : {fmt_rs(summary['net_pnl_rs'])} ({summary['net_pnl_pct']:.2f}%)")

    # 5) Save outputs
    out_df_final = out_df_ruled.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    out_df_final["itr"] = np.arange(1, len(out_df_final) + 1)
    out_df_final["signal_date"] = pd.to_datetime(out_df_final["signal_time_ist"], errors="coerce").dt.tz_convert(IST).dt.date

    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_{ts}_IST.parquet")
    out_df_final[[
        "itr", "signal_time_ist", "signal_date", "ticker", "signal_side", "entry_price",
        "weekly_ok", "daily_ok", "hourly_ok", "taken", "skip_reason"
    ]].to_parquet(signals_path, index=False, engine="pyarrow")

    daily_counts = (
        out_df_final
        .groupby("signal_date")
        .size()
        .reset_index(name="LONG_signals")
        .rename(columns={"signal_date": "date"})
    )
    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_etf_daily_counts_{ts}_IST.parquet")
    daily_counts.to_parquet(daily_counts_path, index=False, engine="pyarrow")

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_{EVAL_SUFFIX}_{ts}_IST.parquet")
    out_df_final.to_parquet(eval_path, index=False, engine="pyarrow")

    print("\n[FILES SAVED]")
    print(f"  Signals         : {signals_path}")
    print(f"  Daily counts    : {daily_counts_path}")
    print(f"  Evaluation      : {eval_path}")

    # 6) FINAL PRINTS (TAKEN + UNREALISED details)
    taken_df = out_df_final[out_df_final["taken"]].copy().sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    print("\n================ TAKEN TRADES (REAL ENTRIES) ================")
    print(f"Taken trades: {len(taken_df)} / {len(out_df_final)} candidate signals")

    if taken_df.empty:
        print("No trades were taken under the current capital rules.")
        return

    cols_taken = [
        "itr",
        "signal_time_ist",
        "signal_date",
        "ticker",
        "entry_price",
        "trade_status",
        "exit_time",
        "exit_price",
        "mtm_time",
        "mtm_price",
        "mtm_pnl_pct",
        "mtm_pnl_rs",
        "mtm_value",
        "max_favorable_pct",
        "max_adverse_pct",
        "invested_amount_constrained",
        "value_constrained",
    ]
    cols_taken = [c for c in cols_taken if c in taken_df.columns]
    disp = taken_df[cols_taken].copy()

    for c in ["signal_time_ist", "exit_time", "mtm_time"]:
        if c in disp.columns:
            disp[c] = pd.to_datetime(disp[c], errors="coerce").apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S%z") if pd.notna(x) else "NA"
            )

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
            disp_open[c] = pd.to_datetime(disp_open[c], errors="coerce").apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S%z") if pd.notna(x) else "NA"
            )

    disp_open["mtm_value_fmt"] = disp_open.get("mtm_value", np.nan).apply(fmt_rs)
    disp_open["mtm_pnl_rs_fmt"] = disp_open.get("mtm_pnl_rs", np.nan).apply(fmt_rs)

    print(disp_open.to_string(index=False))


if __name__ == "__main__":
    main()
