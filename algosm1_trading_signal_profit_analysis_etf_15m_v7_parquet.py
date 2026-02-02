# -*- coding: utf-8 -*-
"""
ALGO-SM1 ETF Multi‑TF LONG signal generator + evaluator + portfolio simulator (beginner-friendly).

CHANGES IN THIS VERSION (as requested)
1) INTRADAY TIMEFRAME = 15 MINUTES
   - Uses 15m indicator CSVs (DIR_15M) as the "timing trigger" timeframe.
   - Also uses the SAME 15m OHLC for target evaluation (+TARGET_PCT).

2) REMOVED "ONE POSITION PER TICKER" LOGIC
   - No ticker lock.
   - The portfolio simulator can take multiple signals from the same ticker,
     even if an earlier trade in that ticker is still open.
   - Only constraint is available cash (MAX_CAPITAL_RS) and fixed per-trade sizing.

3) UNREALISED (OPEN) TRADES KEPT
   - If a TAKEN trade does NOT hit the target within available 15m data,
     we keep it open and compute MTM at the last available close:
         exit_time  = NaT
         exit_price = NaN
         mtm_time, mtm_price, mtm_pnl_pct, mtm_pnl_rs, mtm_value
   - Portfolio summary reports realised + unrealised and final equity.

WEEKLY FILTER (simple)
- weekly_ok = EMA_50_W > EMA_200_W
  (If EMA_200_W is NaN for early history, weekly_ok becomes False for those rows.)

TIMEZONES
- CSV 'date' assumed UTC (or parseable as UTC), then converted to IST.

OUTPUT FILES (OUT_DIR)
- multi_tf_signals_<timestamp>_IST.parquet
- multi_tf_signals_etf_daily_counts_<timestamp>_IST.parquet
- multi_tf_signals_<TARGET>pct_eval_<timestamp>_IST.parquet
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

# Indicator CSV directories
DIR_15M = "etf_indicators_15min_pq"      # <-- 15m intraday indicators
DIR_D   = "etf_indicators_daily_pq"
DIR_W   = "etf_indicators_weekly_pq"

# Output directory
OUT_DIR = "etf_signals"
os.makedirs(OUT_DIR, exist_ok=True)

# Timezone
IST = pytz.timezone("Asia/Kolkata")

# ---- Target configuration (single source of truth) ----
TARGET_PCT: float = 8.0
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)            # used in filenames (assumes integer target)
TARGET_LABEL: str = f"{TARGET_INT}%"

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 20000.0
MAX_CAPITAL_RS: float = 1000000.0

# Ledger printing (how many booking events to show in console)
LEDGER_PRINT_ROWS: int = 50

# A "far future" timestamp used internally to keep cash locked for unrealised trades
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
    """
    Rename all columns except 'date' by adding a suffix.
    Example: close -> close_I, RSI -> RSI_D, etc.
    """
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
# SIGNAL GENERATION (Weekly + Daily(prev) + 15m trigger)
# =============================================================================

def add_intraday_features(df_i: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns used in intraday trigger (rolling/shift features)."""
    if df_i.empty:
        return df_i

    df_i["ATR_I_5bar_mean"] = df_i.get("ATR_I", np.nan).rolling(5).mean()
    df_i["RSI_I_prev"] = df_i.get("RSI_I", np.nan).shift(1)
    df_i["MACD_Hist_I_prev"] = df_i.get("MACD_Hist_I", np.nan).shift(1)
    df_i["Recent_High_5_I"] = df_i.get("high_I", np.nan).rolling(5).max()
    df_i["ADX_I_5bar_mean"] = df_i.get("ADX_I", np.nan).rolling(5).mean()
    df_i["Stoch_%K_I_prev"] = df_i.get("Stoch_%K_I", np.nan).shift(1)

    return df_i


def merge_daily_and_weekly_onto_intraday(df_i: pd.DataFrame, df_d: pd.DataFrame, df_w: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Daily + Weekly context onto each 15m row using as-of logic.

    Weekly lookahead guard:
      - Shift weekly bar by +1s so that intraday bars on the same date still see previous week.
    """
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


def compute_long_signal_mask(df: pd.DataFrame) -> pd.Series:
    """
    Compute boolean mask for LONG signals.
    - Weekly: EMA_50_W > EMA_200_W
    - Daily : previous day Daily_Change between 2 and 9
    - 15m   : timing trigger on 15m columns
    """
    req = [
        "EMA_50_W", "EMA_200_W",
        "Daily_Change_prev_D",
        "close_I", "EMA_200_I", "EMA_50_I", "RSI_I", "MACD_Hist_I", "Stoch_%K_I", "Stoch_%D_I",
        "ATR_I", "ATR_I_5bar_mean", "ADX_I", "ADX_I_5bar_mean", "VWAP_I", "Recent_High_5_I",
        "RSI_I_prev", "MACD_Hist_I_prev"
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return pd.Series(False, index=df.index)

    weekly_ok = (df["EMA_50_W"] > df["EMA_200_W"])

    daily_prev = df["Daily_Change_prev_D"]
    daily_ok = (daily_prev >= 2.0) & (daily_prev <= 9.0)

    intraday_trend = (
        (df["close_I"] >= df["EMA_200_I"] * 0.995) &
        (df["close_I"] >= df["EMA_50_I"] * 0.995)
    )

    rsi_ok = (
        (df["RSI_I"] > 49.0) |
        ((df["RSI_I"] > 46.0) & (df["RSI_I"] > df["RSI_I_prev"]))
    )

    macd_ok = (
        (df["MACD_Hist_I"] > -0.055) &
        (df["MACD_Hist_I"] >= df["MACD_Hist_I_prev"] * 0.75)
    )

    stoch_ok = (
        (df["Stoch_%K_I"] >= df["Stoch_%D_I"] * 0.98) &
        (df["Stoch_%K_I"] >= 18.0)
    )

    momentum_ok = rsi_ok & macd_ok & stoch_ok

    intraday_vol_ok = (
        (df["ATR_I"] >= df["ATR_I_5bar_mean"] * 0.92) &
        ((df["ADX_I"] >= 10.0) | (df["ADX_I"] >= df["ADX_I_5bar_mean"]))
    )

    price_above_vwap = df["close_I"] >= df["VWAP_I"]

    breakout_strict = df["close_I"] >= df["Recent_High_5_I"]
    breakout_near   = df["close_I"] >= df["Recent_High_5_I"] * 0.99

    intra_chg = df.get("Intra_Change_I", 0.0)
    intra_chg = pd.to_numeric(intra_chg, errors="coerce").fillna(0.0)
    strong_up = intra_chg >= 0.05

    breakout_ok = price_above_vwap & (
        breakout_strict |
        (breakout_near & strong_up & (df["RSI_I"] >= 47.0))
    )

    intraday_ok = intraday_trend & momentum_ok & intraday_vol_ok & breakout_ok

    return weekly_ok & daily_ok & intraday_ok


def build_signals_for_ticker(ticker: str) -> List[dict]:
    """Generate candidate LONG signals for one ticker using Weekly + Daily + 15m."""
    path_15m = os.path.join(DIR_15M, f"{ticker}_etf_indicators_15min.parquet")
    path_d   = os.path.join(DIR_D,   f"{ticker}_etf_indicators_daily.parquet")
    path_w   = os.path.join(DIR_W,   f"{ticker}_etf_indicators_weekly.parquet")

    df_i = read_tf_csv(path_15m)
    df_d = read_tf_csv(path_d)
    df_w = read_tf_csv(path_w)

    if df_i.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_i = suffix_columns(df_i, "_I")
    df_d = suffix_columns(df_d, "_D")
    df_w = suffix_columns(df_w, "_W")

    df_i = add_intraday_features(df_i)

    df_idw = merge_daily_and_weekly_onto_intraday(df_i, df_d, df_w)
    if df_idw.empty:
        return []

    df_idw = df_idw.dropna(subset=[
        "Daily_Change_prev_D",
        "close_I", "EMA_200_I", "EMA_50_I",
        "EMA_50_W", "EMA_200_W",
    ])
    if df_idw.empty:
        return []

    long_mask = compute_long_signal_mask(df_idw)

    signals: List[dict] = []
    for _, r in df_idw[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_I"]),
            "weekly_ok": True,
            "daily_ok": True,
            "intraday_ok": True,
        })
    return signals


# =============================================================================
# SIGNAL EVALUATION (exit_time, MTM for unrealised)
# =============================================================================

def read_intraday_ohlc(path: str) -> pd.DataFrame:
    """Read 15m OHLC for evaluation. Needs: date, high, low, close."""
    return read_tf_csv(path, tz=IST)


def evaluate_long_signal(row: pd.Series, df_i: pd.DataFrame) -> dict:
    """
    Evaluate one LONG signal using 15m future bars.
    """
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

    future = df_i[df_i["date"] > entry_time].copy()
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
            TIME_TD_COL: time_to_target_days,
            TIME_CAL_COL: days_to_target_calendar,
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

    last_row = future.iloc[-1]
    mtm_time = last_row["date"]
    mtm_price = float(last_row["close"]) if "close" in last_row else entry_price

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
# PORTFOLIO SIMULATION (capital constraint ONLY; NO ticker lock)
# =============================================================================

def apply_portfolio_rules(evaluated_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Capital-only portfolio simulation (NO ticker lock) with PROFIT RECYCLING (compounding).

    What this means:
    - Each taken trade deploys CAPITAL_PER_SIGNAL_RS from the *current* cash pool.
    - When a trade exits (profit booking OR loss booking), we immediately add back its full
      return_value (= principal + P&L) to cash at/after its exit_time, so that this updated
      capital can be reused for subsequent trades.
    - We also record running portfolio snapshots after each signal is processed.
    """
    df = evaluated_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    cash: float = float(MAX_CAPITAL_RS)

    # Each open trade keeps its deployed principal locked until release_time.
    # For realised trades, release_time = exit_time.
    # For unrealised trades, release_time = FAR_FUTURE_LOCK (keeps cash locked).
    open_trades: List[dict] = []

    # Booking ledger: one row per realised exit (profit/loss booking)
    booking_ledger_rows: List[dict] = []
    trade_seq: int = 0

    # Running realised P&L and count, updated when trades are released back to cash.
    realised_pnl_cum: float = 0.0
    realised_trades_cum: int = 0

    df["taken"] = False
    df["skip_reason"] = ""

    # --- NEW: running portfolio snapshots (updated at each signal processing step) ---
    df["cash_before"] = np.nan
    df["cash_released"] = 0.0
    df["cash_after_release"] = np.nan
    df["cash_after_entry"] = np.nan
    df["locked_capital_after_entry"] = np.nan
    df["total_capital_after_entry"] = np.nan
    df["open_trades_after_entry"] = np.nan
    df["realised_pnl_rs_to_date"] = np.nan
    df["realised_trades_closed_to_date"] = np.nan

    for i, row in df.iterrows():
        signal_time = row["signal_time_ist"]

        # Snapshot before anything happens at this signal
        df.at[i, "cash_before"] = float(cash)

        # ------------------------------------------------------------
        # 1) RELEASE CAPITAL for trades that have exited by this time.
        #    This is where profit (or loss) gets added back to the pool
        #    and becomes available for subsequent trades.
        # ------------------------------------------------------------
        cash_released_now: float = 0.0

        still_open: List[dict] = []
        for tr in open_trades:
            if (not tr["is_unrealized"]) and (tr["release_time"] <= signal_time):
                rv = float(tr["return_value"])
                inv = float(tr["invested"])

                cash_before_rel = float(cash)
                cash += rv  # principal + P&L comes back immediately
                cash_after_rel = float(cash)

                cash_released_now += rv
                realised_pnl_cum += (rv - inv)
                realised_trades_cum += 1

                # Record booking event (profit/loss booking ledger)
                booking_ledger_rows.append({
                    "trade_id": tr.get("trade_id"),
                    "ticker": tr.get("ticker", ""),
                    "entry_time": tr.get("entry_time", pd.NaT),
                    "exit_time": tr.get("release_time", pd.NaT),
                    "invested_rs": inv,
                    "return_value_rs": rv,
                    "pnl_rs": rv - inv,
                    "pnl_pct": ((rv - inv) / inv * 100.0) if inv else np.nan,
                    "cash_before_release_rs": cash_before_rel,
                    "cash_after_release_rs": cash_after_rel,
                })
            else:
                still_open.append(tr)
        open_trades = still_open

        df.at[i, "cash_released"] = float(cash_released_now)
        df.at[i, "cash_after_release"] = float(cash)

        # ------------------------------------------------------------
        # 2) TAKE / SKIP the signal based on *current* cash AFTER releases
        # ------------------------------------------------------------
        if cash < float(CAPITAL_PER_SIGNAL_RS):
            df.at[i, "taken"] = False
            df.at[i, "skip_reason"] = "NO_CAPITAL"

            locked_after = float(sum(float(t["invested"]) for t in open_trades))
            df.at[i, "cash_after_entry"] = float(cash)
            df.at[i, "locked_capital_after_entry"] = locked_after
            df.at[i, "total_capital_after_entry"] = float(cash) + locked_after
            df.at[i, "open_trades_after_entry"] = int(len(open_trades))
            df.at[i, "realised_pnl_rs_to_date"] = float(realised_pnl_cum)
            df.at[i, "realised_trades_closed_to_date"] = int(realised_trades_cum)
            continue

        # We take the trade
        df.at[i, "taken"] = True
        df.at[i, "skip_reason"] = ""
        cash -= float(CAPITAL_PER_SIGNAL_RS)

        invested = float(CAPITAL_PER_SIGNAL_RS)

        # Determine when the capital is released and what value returns
        is_unreal = bool(row.get("is_unrealized", False))
        if is_unreal:
            release_time = FAR_FUTURE_LOCK
            return_value = float(row.get("mtm_value", invested))
        else:
            exit_time = row.get("exit_time", pd.NaT)
            if pd.isna(exit_time):
                # defensive: treat as unrealised if exit_time is missing
                release_time = FAR_FUTURE_LOCK
                is_unreal = True
                return_value = float(row.get("mtm_value", invested))
            else:
                release_time = exit_time
                return_value = float(row.get("mtm_value", invested))

        trade_seq += 1
        trade_id = int(trade_seq)

        open_trades.append({
            "trade_id": trade_id,
            "ticker": row.get("ticker", ""),
            "entry_time": signal_time,
            "release_time": release_time,
            "return_value": return_value,
            "invested": invested,
            "is_unrealized": is_unreal,
        })

        # ------------------------------------------------------------
        # 3) Snapshot AFTER taking (or skipping) the trade
        # ------------------------------------------------------------
        locked_after = float(sum(float(t["invested"]) for t in open_trades))

        df.at[i, "cash_after_entry"] = float(cash)
        df.at[i, "locked_capital_after_entry"] = locked_after
        df.at[i, "total_capital_after_entry"] = float(cash) + locked_after
        df.at[i, "open_trades_after_entry"] = int(len(open_trades))
        df.at[i, "realised_pnl_rs_to_date"] = float(realised_pnl_cum)
        df.at[i, "realised_trades_closed_to_date"] = int(realised_trades_cum)

    # ------------------------------------------------------------
    # FINAL EQUITY
    # - realised trades that haven’t been released yet (because no later signal)
    #   are added back to cash here
    # - unrealised trades remain valued at mtm_value (as already computed)
    # ------------------------------------------------------------
    open_positions_value = 0.0
    for tr in open_trades:
        if tr["is_unrealized"]:
            open_positions_value += float(tr["return_value"])
        else:
            cash += float(tr["return_value"])

    final_equity = float(cash) + float(open_positions_value)
    net_pnl_rs = final_equity - float(MAX_CAPITAL_RS)
    net_pnl_pct = (net_pnl_rs / float(MAX_CAPITAL_RS) * 100.0) if MAX_CAPITAL_RS > 0 else 0.0

    # Keep original output columns for downstream compatibility
    df["invested_amount_constrained"] = np.where(df["taken"], float(CAPITAL_PER_SIGNAL_RS), 0.0)
    df["value_constrained"] = np.where(df["taken"], df.get("mtm_value", np.nan), 0.0)
    df["final_value_constrained"] = df["value_constrained"]

    taken = df[df["taken"]].copy()
    unrealised_taken = taken[taken.get("is_unrealized", False)].copy()
    realised_taken = taken[~taken.get("is_unrealized", False)].copy()

    realised_pnl_rs = float(realised_taken.get("pnl_rs", pd.Series(dtype=float)).sum()) if not realised_taken.empty else 0.0
    unrealised_pnl_rs = float(unrealised_taken.get("mtm_pnl_rs", pd.Series(dtype=float)).sum()) if not unrealised_taken.empty else 0.0

    # ------------------------------------------------------------
    # Booking ledger dataframe (one row per realised exit)
    # ------------------------------------------------------------
    ledger_df = pd.DataFrame(booking_ledger_rows)
    if not ledger_df.empty:
        # Ensure deterministic order (exit time then trade id)
        ledger_df["exit_time"] = pd.to_datetime(ledger_df["exit_time"], errors="coerce")
        ledger_df = ledger_df.sort_values(["exit_time", "trade_id"]).reset_index(drop=True)

    ledger_rows_n = int(len(ledger_df))
    ledger_profit_n = int((ledger_df["pnl_rs"] > 0).sum()) if ledger_rows_n else 0

    summary = {
        "total_signals": int(len(df)),
        "signals_taken": int(len(taken)),
        "signals_skipped_no_capital": int((df["skip_reason"] == "NO_CAPITAL").sum()),
        "realised_trades": int(len(realised_taken)),
        "unrealised_trades": int(len(unrealised_taken)),
        "realised_pnl_rs": realised_pnl_rs,
        "unrealised_pnl_rs": unrealised_pnl_rs,
        "cash_end": float(cash),
        "open_positions_value": float(open_positions_value),
        "final_equity": float(final_equity),
        "net_pnl_rs": float(net_pnl_rs),
        "net_pnl_pct": float(net_pnl_pct),

        # Booking ledger (profit/loss recycling events)
        "booking_ledger_rows": ledger_rows_n,
        "booking_ledger_profit_rows": ledger_profit_n,
        "ledger_df": ledger_df,
    }
    return df, summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    tickers_i = list_tickers(DIR_15M, "_etf_indicators_15min.parquet")
    tickers_d = list_tickers(DIR_D,   "_etf_indicators_daily.parquet")
    tickers_w = list_tickers(DIR_W,   "_etf_indicators_weekly.parquet")

    tickers = tickers_i & tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across 15m, Daily, and Weekly directories.")
        return

    print(f"[INFO] Found {len(tickers)} tickers with all 3 timeframes (15m+Daily+Weekly).")

    all_signals: List[dict] = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Generating signals: {ticker}")
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

    print(f"\n[INFO] Evaluating +{TARGET_LABEL} outcome on signals (LONG only) using 15m future bars.")

    tickers_in_signals = sig_df["ticker"].unique().tolist()
    intraday_cache: Dict[str, pd.DataFrame] = {}

    for t in tickers_in_signals:
        path = os.path.join(DIR_15M, f"{t}_etf_indicators_15min.parquet")
        df_i = read_intraday_ohlc(path)
        if df_i.empty:
            print(f"- No 15m data for {t}, its signals will be MTM at entry (flat).")
        intraday_cache[t] = df_i

    evaluated_rows: List[dict] = []
    for _, row in sig_df.iterrows():
        t = row["ticker"]
        df_i = intraday_cache.get(t, pd.DataFrame())

        if df_i.empty:
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
            eval_info = evaluate_long_signal(row, df_i)

        merged = row.to_dict()
        merged.update(eval_info)
        evaluated_rows.append(merged)

    out_df = pd.DataFrame(evaluated_rows).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    out_df["invested_amount"] = float(CAPITAL_PER_SIGNAL_RS)
    out_df["final_value_per_signal"] = out_df.get("mtm_value", np.nan)

    print("\n[INFO] Portfolio simulation (capital only; NO ticker lock).")
    print(f"  Max capital pool           : {fmt_rs(MAX_CAPITAL_RS)}")
    print(f"  Capital per trade          : {fmt_rs(CAPITAL_PER_SIGNAL_RS)}")

    out_df_ruled, summary = apply_portfolio_rules(out_df)

    print("\n[PORTFOLIO SUMMARY]")
    print(f"  Candidate signals (all)    : {summary['total_signals']}")
    print(f"  Taken trades               : {summary['signals_taken']}")
    print(f"  Skipped (no capital)       : {summary['signals_skipped_no_capital']}")
    print(f"  Realised trades            : {summary['realised_trades']}  | P&L: {fmt_rs(summary['realised_pnl_rs'])}")
    print(f"  Unrealised trades          : {summary['unrealised_trades']} | MTM P&L: {fmt_rs(summary['unrealised_pnl_rs'])}")
    print(f"  Cash (end)                 : {fmt_rs(summary['cash_end'])}")
    print(f"  Open positions value (MTM) : {fmt_rs(summary['open_positions_value'])}")
    print(f"  Final equity               : {fmt_rs(summary['final_equity'])}")
    print(f"  Net P&L                    : {fmt_rs(summary['net_pnl_rs'])} ({summary['net_pnl_pct']:.2f}%)")

    # ------------------------------------------------------------
    # Print + save booking ledger (profit/loss booking events)
    # ------------------------------------------------------------
    ledger_df = summary.get("ledger_df", None)
    if isinstance(ledger_df, pd.DataFrame) and (not ledger_df.empty):
        # Create profit-only view
        profit_ledger = ledger_df[ledger_df["pnl_rs"] > 0].copy().reset_index(drop=True)

        print("")
        print("[BOOKING LEDGER] (Realised exits where capital was released back to cash)")
        print(f"  Total booking events   : {summary.get('booking_ledger_rows', len(ledger_df))}")
        print(f"  Profit bookings (pnl>0): {summary.get('booking_ledger_profit_rows', int((ledger_df['pnl_rs']>0).sum()))}")

        # Console print (profit bookings first; if none, print all bookings)
        to_show = profit_ledger if not profit_ledger.empty else ledger_df
        n_show = min(int(LEDGER_PRINT_ROWS), len(to_show))
        cols = [
            "trade_id", "ticker", "entry_time", "exit_time",
            "invested_rs", "return_value_rs", "pnl_rs", "pnl_pct",
            "cash_before_release_rs", "cash_after_release_rs",
        ]
        print(to_show[cols].head(n_show).to_string(index=False))

        # Save ledger files
        ledger_csv = os.path.join(OUT_DIR, f"booking_ledger_{ts}_IST.csv")
        ledger_parq = os.path.join(OUT_DIR, f"booking_ledger_{ts}_IST.parquet")
        ledger_df.to_csv(ledger_csv, index=False)
        ledger_df.to_parquet(ledger_parq, index=False, engine="pyarrow")

        if not profit_ledger.empty:
            profit_csv = os.path.join(OUT_DIR, f"profit_bookings_{ts}_IST.csv")
            profit_parq = os.path.join(OUT_DIR, f"profit_bookings_{ts}_IST.parquet")
            profit_ledger.to_csv(profit_csv, index=False)
            profit_ledger.to_parquet(profit_parq, index=False, engine="pyarrow")

        print(f"  Saved: {ledger_csv}")
        print(f"  Saved: {ledger_parq}")
        if not profit_ledger.empty:
            print(f"  Saved: {profit_csv}")
            print(f"  Saved: {profit_parq}")
    else:
        print("")
        print("[BOOKING LEDGER] No realised exits detected in this run (no capital releases).")

    out_df_ruled = out_df_ruled.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)
    out_df_ruled["itr"] = np.arange(1, len(out_df_ruled) + 1)
    out_df_ruled["signal_date"] = pd.to_datetime(out_df_ruled["signal_time_ist"], errors="coerce").dt.tz_convert(IST).dt.date

    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_{ts}_IST.parquet")
    out_df_ruled[[
        "itr", "signal_time_ist", "signal_date", "ticker", "signal_side", "entry_price",
        "weekly_ok", "daily_ok", "intraday_ok", "taken", "skip_reason"
    ]].to_parquet(signals_path, index=False, engine="pyarrow")

    daily_counts = (
        out_df_ruled
        .groupby("signal_date")
        .size()
        .reset_index(name="LONG_signals")
        .rename(columns={"signal_date": "date"})
    )
    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_etf_daily_counts_{ts}_IST.parquet")
    daily_counts.to_parquet(daily_counts_path, index=False, engine="pyarrow")

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_{EVAL_SUFFIX}_{ts}_IST.parquet")
    out_df_ruled.to_parquet(eval_path, index=False, engine="pyarrow")

    print("\n[FILES SAVED]")
    print(f"  Signals         : {signals_path}")
    print(f"  Daily counts    : {daily_counts_path}")
    print(f"  Evaluation      : {eval_path}")

    taken_df = out_df_ruled[out_df_ruled["taken"]].copy().sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    print("\n================ TAKEN TRADES (REAL ENTRIES) ================")
    print(f"Taken trades: {len(taken_df)} / {len(out_df_ruled)} candidate signals")

    if taken_df.empty:
        print("No trades were taken under the current capital rule.")
        return

    cols_taken = [
        "itr", "signal_time_ist", "signal_date", "ticker", "entry_price",
        "trade_status", "exit_time", "exit_price",
        "mtm_time", "mtm_price", "mtm_pnl_pct", "mtm_pnl_rs", "mtm_value",
        "max_favorable_pct", "max_adverse_pct",
        # NEW: portfolio snapshots (so you can see capital update after each profit booking)
        "cash_before", "cash_released", "cash_after_release", "cash_after_entry",
        "locked_capital_after_entry", "total_capital_after_entry",
        "realised_pnl_rs_to_date",
        "invested_amount_constrained", "value_constrained",
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
