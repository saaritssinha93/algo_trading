# -*- coding: utf-8 -*-
"""
Multi-timeframe positional LONG backtest framework (DAILY execution, WEEKLY context).

Execution TF:
    • Daily bars from DIR_D = "main_indicators_daily"

Context TFs:
    • Weekly from DIR_W  (NO same-week lookahead: use last fully completed week)

Flow
----
1) Generate MTF LONG signals using ONLY CUSTOM10 (Weekly + Daily positional logic)
2) Evaluate those signals on Daily data for +TARGET_PCT% target hit (positional)
3) Apply capital constraint simulation
4) Export signals + eval + constrained eval + day-wise summary

NOTE:
- This version uses ONLY Daily + Weekly data (no intraday timeframes at all).
- Strategy CUSTOM10 is designed to find stocks in a "continuously rising" structure.

Key NO-LOOKAHEAD rules:
    • Weekly:
        For every Daily bar, we use the *previous fully completed week*.
        Implementation:
            - Weekly 'date' is shifted by +1 day and merge_asof(direction="backward").
            - So the current week's bar only becomes visible from the NEXT Monday.
    • Daily:
        Daily_Change_prev_D is simply Daily_Change shifted by 1, i.e. D-1 for every entry day.
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Target configuration (FIXED) ----
TARGET_PCT: float = 2.0
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0


def _pct_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


TARGET_TAG = _pct_tag(TARGET_PCT)
TARGET_LABEL = f"{TARGET_PCT:.2f}%"

HIT_COL = f"hit_{TARGET_TAG}"
TIME_TD_COL = f"time_to_{TARGET_TAG}_days"
TIME_CAL_COL = f"days_to_{TARGET_TAG}_calendar"
EVAL_SUFFIX = f"{TARGET_TAG}_eval"

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 50000.0
MAX_CAPITAL_RS: float = 20000000.0

# Exclude last N signals in "non-recent" stats
N_RECENT_SKIP = 100

# Hardwired single strategy name
STRATEGY_NAME = "custom10"

# ----------------------- HELPERS (COMMON) -----------------------


def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """
    Read a timeframe CSV, parse 'date' as tz-aware IST, sort.

    Expected:
        - CSV has a 'date' column in UTC or naive; we coerce to tz-aware IST.
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
    """Append suffix to all columns except 'date'."""
    if df.empty:
        return df
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)


def _list_tickers_from_dir(directory: str) -> set[str]:
    """
    Robust ticker extraction:
    - Finds files containing '_main_indicators'
    - Extracts ticker as the part BEFORE '_main_indicators'
    """
    files = glob.glob(os.path.join(directory, "*_main_indicators*.csv"))
    tickers = set()
    for f in files:
        base = os.path.basename(f)
        if "_main_indicators" in base:
            ticker = base.split("_main_indicators")[0].strip()
            if ticker:
                tickers.add(ticker)
    return tickers


def _find_file(directory: str, ticker: str) -> str | None:
    """Return first matching main_indicators CSV for ticker if present."""
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None


# ======================================================================
#                 COMMON MTF LOADER (DAILY + WEEKLY)
# ======================================================================

def _load_mtf_context_daily_weekly(ticker: str) -> pd.DataFrame:
    """
    Load Daily + Weekly CSVs for a ticker, suffix columns (_D, _W),
    and merge Weekly onto Daily via as-of join.

    WEEKLY (no same-week lookahead):
        - We want for each daily bar to see ONLY the last fully completed week.
        - Implementation:
            * Take weekly dataframe, then shift its 'date' by +1 day.
            * Use merge_asof(direction="backward") onto the Daily df.
            * This means:
                - All daily bars in the current week (Mon–Fri) still see
                  the *previous* week's bar.
                - The current week's weekly bar only becomes visible from
                  the NEXT Monday.

    DAILY (previous-day change):
        - If Daily_Change exists, we create Daily_Change_prev = Daily_Change.shift(1).
        - After suffixing, it becomes Daily_Change_prev_D.
        - For any entry day D, we always use Daily_Change of D-1.

    Returns:
        A merged df with:
            * All Daily columns as *_D
            * All Weekly columns as *_W
            * Weekly columns aligned as "previous full week" context.
    """
    path_d = _find_file(DIR_D, ticker)
    path_w = _find_file(DIR_W, ticker)

    if not path_d or not path_w:
        print(f"- Skipping {ticker}: missing Daily or Weekly file.")
        return pd.DataFrame()

    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: empty Daily or Weekly data.")
        return pd.DataFrame()

    # Daily: compute previous-day Daily_Change BEFORE suffixing
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # Suffix frames
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Weekly: NO same-week lookahead (shift weekly date by +1 day)
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(days=1)

    # Merge Weekly onto Daily (backward as-of)
    df_dw = pd.merge_asof(
        df_d.sort_values("date"),
        df_w_sorted.sort_values("date"),
        on="date",
        direction="backward",
    )

    df_dw.reset_index(drop=True, inplace=True)
    return df_dw


# ======================================================================
# STRATEGY: CUSTOM 10  (DAILY POSITIONAL; WEEKLY + DAILY ONLY)
# ======================================================================

def build_signals_custom10(ticker: str) -> list[dict]:
    """
    CUSTOM 10 (DAILY positional) — STRICT, HIGH-MOMENTUM CONTINUATION

    GOAL:
      Find stocks that are in a strong, continuously rising WEEKLY trend
      and show a clean, high-momentum continuation breakout on the DAILY chart.

    Weekly filter (previous FULL week, no same-week lookahead):
      - close_W > EMA_200_W               (long-term uptrend)
      - close_W > EMA_50_W (if available) (price riding higher MA)
      - EMA_50_W >= 1.01 * EMA_200_W      (clear MA separation)
      - RSI_W >= 52                       (stronger trend)
      - ADX_W >= 20 if available          (trend strength)

    Daily filter (previous day, D-1, no same-day lookahead):
      - Daily_Change_prev_D in [+0.3%, +4.5%]
        (prior day was mildly positive / constructive, not flat or parabolic)
      - Previous close above EMA_20_D (if available)
      - Optional extra: if EMA_50_D exists, previous close above EMA_50_D too

    Entry candle (current daily bar, D):
      - Breakout above PRIOR rolling highs (20 / 50 / 100) with NO lookahead:
          * uses high_D rolling window max shifted by 1 bar
      - Green candle with decent real body (>= 0.5% of open)
      - Close above EMA_20_D and EMA_50_D (if available)
      - Momentum boosters: AT LEAST 3 of 4:
          * RSI_D >= 60  OR (RSI_D >= RSI_D_prev + 1.5)
          * MACD_Hist_D >= 0  AND >= 1.02x MACD_Hist_D_prev
          * Volume_D >= 1.6x 20-day average
          * Today’s Daily_Change_D >= +1.8%

      - Optional trend strength on daily:
          * ADX_D >= 18 if available

    Anti-chase:
      - close_D <= EMA_20_D * 1.04   (avoid entries too extended above EMA20)

    ENTRY:
      - signal_time_ist -> daily candle timestamp
      - entry_price     -> close_D  (you can execute at next-day open in live)
    """

    df = _load_mtf_context_daily_weekly(ticker)
    if df.empty:
        return []

    cols = set(df.columns)
    minimal = {"date", "open_D", "high_D", "low_D", "close_D"}
    if not minimal.issubset(cols):
        print(f"- Skipping {ticker}: missing minimal Daily columns for custom10.")
        return []

    def _col(name: str, default=None):
        if name in cols:
            return df[name]
        return pd.Series(default if default is not None else np.nan, index=df.index)

    # ===================== WEEKLY FILTER (prev FULL week) =====================
    # Start with all True, tighten as we add conditions
    weekly_ok = pd.Series(True, index=df.index)

    if {"close_W", "EMA_200_W"}.issubset(cols):
        weekly_ok &= (df["close_W"] > df["EMA_200_W"])

    if "EMA_50_W" in cols:
        weekly_ok &= (df["close_W"] > df["EMA_50_W"])
        if "EMA_200_W" in cols:
            weekly_ok &= (df["EMA_50_W"] >= df["EMA_200_W"] * 1.01)

    if "RSI_W" in cols:
        weekly_ok &= (df["RSI_W"] >= 52.0)

    if "ADX_W" in cols:
        weekly_ok &= (df["ADX_W"] >= 20.0)

    weekly_ok = weekly_ok.fillna(False)

    # ===================== DAILY PREV FILTER (D-1) =====================
    if "Daily_Change_prev_D" in cols:
        dprev = df["Daily_Change_prev_D"]          # already D-1 vs entry
    elif "Daily_Change_D" in cols:
        dprev = df["Daily_Change_D"].shift(1)      # hard fallback
    else:
        dprev = pd.Series(np.nan, index=df.index)

    daily_prev_ok = ((dprev >= 0.3) & (dprev <= 4.5)).fillna(False)

    # Previous day must also be above EMA20 / EMA50 if available
    prev_close = df["close_D"].shift(1)

    ema20 = _col("EMA_20_D")
    ema50 = _col("EMA_50_D")

    if "EMA_20_D" in cols:
        daily_prev_ok &= (prev_close >= ema20.shift(1)).fillna(False)

    if "EMA_50_D" in cols:
        daily_prev_ok &= (prev_close >= ema50.shift(1)).fillna(False)

    # ===================== DAILY BREAKOUT (NO LOOKAHEAD) =====================
    high = df["high_D"]
    close = df["close_D"]

    prior_high_20 = high.rolling(20, min_periods=20).max().shift(1)
    prior_high_50 = high.rolling(50, min_periods=50).max().shift(1)
    prior_high_100 = high.rolling(100, min_periods=100).max().shift(1)

    breakout_level = prior_high_20.copy()
    breakout_name = "PRIOR_RH20"

    fb = breakout_level.isna()
    breakout_level[fb] = prior_high_50[fb]
    breakout_name = "PRIOR_RH20_or_RH50"

    fb2 = breakout_level.isna()
    breakout_level[fb2] = prior_high_100[fb2]
    breakout_name = "PRIOR_RH20_or_RH50_or_RH100"

    # Slightly stricter breakout thresholds
    close_break = close >= breakout_level * 1.0020   # ~0.20% close breakout
    wick_break = (high >= breakout_level * 1.0010) & (close >= breakout_level * 1.0005)
    breakout = (close_break | wick_break).fillna(False)

    # ===================== CANDLE QUALITY =====================
    open_ = df["open_D"]
    green = (close > open_).fillna(False)
    body_ok = ((close - open_) >= (open_ * 0.0050)).fillna(False)  # >= 0.5% body

    # ===================== MOMENTUM BOOSTERS (>=3 of 4) =====================
    rsi = _col("RSI_D")
    rsi_prev = rsi.shift(1)
    rsi_boost = ((rsi >= 60.0) | (rsi >= (rsi_prev + 1.5))).fillna(False)

    macd_hist = _col("MACD_Hist_D")
    macd_hist_prev = macd_hist.shift(1)
    if "MACD_Hist_D" in cols:
        macd_boost = ((macd_hist >= 0.0) & (macd_hist >= macd_hist_prev * 1.02)).fillna(False)
    else:
        macd_boost = pd.Series(False, index=df.index)

    vol = _col("volume_D")
    if "volume_D" in cols:
        vol_mean_20 = vol.rolling(20, min_periods=20).mean()
        vol_boost = (vol >= (vol_mean_20 * 1.6)).fillna(False)
    else:
        vol_boost = pd.Series(False, index=df.index)

    daily_change = _col("Daily_Change_D", 0.0).fillna(0.0)
    daily_chg_boost = (daily_change >= 1.8).fillna(False)

    boosters = (
        rsi_boost.astype(int) +
        macd_boost.astype(int) +
        vol_boost.astype(int) +
        daily_chg_boost.astype(int)
    )
    momentum_ok = boosters >= 3

    # Optional: daily trend strength (if ADX_D exists)
    if "ADX_D" in cols:
        adx_d_ok = (df["ADX_D"] >= 18.0).fillna(False)
    else:
        adx_d_ok = pd.Series(True, index=df.index)

    # ===================== TREND ALIGNMENT + ANTI-CHASE =====================
    if "EMA_20_D" in cols:
        above_ema20 = (close >= ema20).fillna(False)
        not_too_far = (close <= ema20 * 1.04).fillna(False)
    else:
        above_ema20 = pd.Series(True, index=df.index)
        not_too_far = pd.Series(True, index=df.index)

    if "EMA_50_D" in cols:
        above_ema50 = (close >= ema50).fillna(False)
    else:
        above_ema50 = pd.Series(True, index=df.index)

    # ===================== FINAL ENTRY MASK =====================
    exec_ok = (
        breakout &
        green & body_ok &
        momentum_ok &
        above_ema20 & above_ema50 &
        not_too_far &
        adx_d_ok
    )

    long_mask = weekly_ok & daily_prev_ok & exec_ok

    if not long_mask.any():
        return []

    signals: list[dict] = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],              # daily candle timestamp (IST)
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_D"]),        # positional entry at close of breakout candle
            "weekly_ok": True,
            "daily_prev_ok": True,
            "exec_ok": True,
            "strategy": STRATEGY_NAME.upper(),
            "sub_strategy": (
                "W_strict_uptrend + Dprev_band_strict + "
                f"{breakout_name} + boosters>=3 + EMA20/50_align + ADX_filter + anti_chase"
            ),
        })
    return signals



# ----------------------- EVALUATION ON DAILY -----------------------

def _safe_read_daily(path: str) -> pd.DataFrame:
    """Read Daily CSV for evaluation, parse 'date' as tz-aware IST, sort."""
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(IST)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading Daily data {path}: {e}")
        return pd.DataFrame()


def evaluate_signal_on_daily_long_only(row: pd.Series, df_d: pd.DataFrame) -> dict:
    """
    Positional evaluation on Daily data (multi-day).

    - HIT_COL: target hit ANYTIME after entry (positional, no SL here)
    - Exit:
        • If target touched -> exit at target_price on first hit bar
        • Else -> exit at last available future close
    """
    entry_time = pd.to_datetime(row["signal_time_ist"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize(IST)
    else:
        entry_time = entry_time.tz_convert(IST)

    entry_price = float(row["entry_price"])

    if (not np.isfinite(entry_price)) or entry_price <= 0 or df_d is None or df_d.empty:
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

    dfx = df_d.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date"])
    if dfx["date"].dt.tz is None:
        dfx["date"] = dfx["date"].dt.tz_localize(IST)
    else:
        dfx["date"] = dfx["date"].dt.tz_convert(IST)
    dfx = dfx.sort_values("date").reset_index(drop=True)

    # Only look strictly after the entry day (no same-bar evaluation)
    future = dfx[dfx["date"] > entry_time].copy()
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

    if not {"high", "low", "close"}.issubset(future.columns):
        # minimal OHLC not available
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
        first_hit_pos = int(np.argmax(hit_mask.values))
        hit_row = future.iloc[first_hit_pos]
        exit_time = hit_row["date"]
        exit_price = float(target_price)

        dt_days = (exit_time - entry_time).days
        cal_days = dt_days
        hit_flag = True
        time_to_target_days = float(dt_days)
        days_to_target_calendar = float(cal_days)
    else:
        last_row = future.iloc[-1]
        exit_time = last_row["date"]
        exit_price = float(last_row["close"])

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
        "exit_price": float(exit_price),
        "pnl_pct": float(pnl_pct),
        "pnl_rs": float(pnl_rs),
        "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
        "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
    }


# ----------------------- CAPITAL-CONSTRAINED PORTFOLIO SIM -----------------------

def apply_capital_constraint(out_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply a portfolio-level capital constraint:

    - Start with MAX_CAPITAL_RS cash.
    - For each signal chronologically:
        * Release any open trades whose exit_time <= entry_time.
        * If available_capital >= CAPITAL_PER_SIGNAL_RS -> take trade else skip.
    """
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"])
    if df["signal_time_ist"].dt.tz is None:
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_localize(IST)
    else:
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_convert(IST)

    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    if df["exit_time"].notna().any():
        if df.loc[df["exit_time"].notna(), "exit_time"].dt.tz is None:
            df.loc[df["exit_time"].notna(), "exit_time"] = (
                df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_localize(IST)
            )
        else:
            df.loc[df["exit_time"].notna(), "exit_time"] = (
                df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)
            )

    df["pnl_rs"] = pd.to_numeric(df.get("pnl_rs", 0.0), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0.0), errors="coerce").fillna(0.0)

    df["invested_amount"] = CAPITAL_PER_SIGNAL_RS
    df["final_value_per_signal"] = df["invested_amount"] + df["pnl_rs"]

    available_capital = MAX_CAPITAL_RS
    open_trades: list[dict] = []
    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        # Release trades that have exited by this entry_time
        still_open = []
        for trade in open_trades:
            exit_time = trade["exit_time"]
            if pd.isna(exit_time) or exit_time <= entry_time:
                available_capital += trade["value"]
            else:
                still_open.append(trade)
        open_trades = still_open

        # Decide whether to take new trade
        if available_capital >= CAPITAL_PER_SIGNAL_RS:
            df.at[idx, "taken"] = True
            available_capital -= CAPITAL_PER_SIGNAL_RS

            exit_time = row["exit_time"]
            if pd.isna(exit_time):
                exit_time = entry_time

            final_value = float(row["final_value_per_signal"])
            open_trades.append({"exit_time": exit_time, "value": final_value})
        else:
            df.at[idx, "taken"] = False

    # Liquidate remaining open trades at the end
    for trade in open_trades:
        available_capital += trade["value"]

    df["invested_amount_constrained"] = np.where(df["taken"], CAPITAL_PER_SIGNAL_RS, 0.0)
    df["final_value_constrained"] = np.where(df["taken"], df["final_value_per_signal"], 0.0)

    total_signals = len(df)
    signals_taken = int(df["taken"].sum())
    gross_invested = df["invested_amount_constrained"].sum()
    final_portfolio_value = float(available_capital)
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


# ----------------------- DAILY SUMMARY -----------------------

def build_daily_summary(out_df: pd.DataFrame) -> pd.DataFrame:
    dfx = out_df.copy()

    dfx["signal_time_ist"] = pd.to_datetime(dfx["signal_time_ist"], errors="coerce")
    dfx = dfx.dropna(subset=["signal_time_ist"])

    if dfx["signal_time_ist"].dt.tz is None:
        dfx["signal_time_ist"] = dfx["signal_time_ist"].dt.tz_localize(IST)
    else:
        dfx["signal_time_ist"] = dfx["signal_time_ist"].dt.tz_convert(IST)

    dfx["date"] = dfx["signal_time_ist"].dt.date

    if "taken" not in dfx.columns:
        dfx["taken"] = True

    for c in ["pnl_rs", "pnl_pct"]:
        dfx[c] = pd.to_numeric(dfx.get(c, 0.0), errors="coerce").fillna(0.0)

    def _sum_where(series, mask):
        return float(series[mask].sum()) if len(series) else 0.0

    rows = []
    for day, g in dfx.groupby("date", dropna=False):
        taken = g["taken"].astype(bool)

        rows.append({
            "date": day,
            "signals_generated": int(len(g)),
            "signals_taken": int(taken.sum()),
            "pnl_rs_sum": _sum_where(g["pnl_rs"], slice(None)),
            "pnl_rs_taken_sum": _sum_where(g["pnl_rs"], taken),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ----------------------- MAIN -----------------------

def main():
    # --------- PART 1: GENERATE LONG SIGNALS (CUSTOM10 ONLY) ---------
    tickers_d = _list_tickers_from_dir(DIR_D)
    tickers_w = _list_tickers_from_dir(DIR_W)

    tickers = tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across Daily and Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with both Daily and Weekly timeframes.")
    print(f"Running strategy: {STRATEGY_NAME}\n")

    all_signals = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Processing {ticker} ...")
        try:
            ticker_signals = build_signals_custom10(ticker)
            all_signals.extend(ticker_signals)
            print(f"  -> {len(ticker_signals)} LONG signals for {ticker}")
        except Exception as e:
            print(f"! Error while processing {ticker}: {e}")

    if not all_signals:
        print("No LONG signals generated.")
        return

    sig_df = pd.DataFrame(all_signals)
    sig_df.sort_values(["signal_time_ist", "ticker"], inplace=True)
    sig_df["itr"] = np.arange(1, len(sig_df) + 1)

    sig_df["signal_time_ist"] = pd.to_datetime(sig_df["signal_time_ist"]).dt.tz_convert(IST)
    sig_df["signal_date"] = sig_df["signal_time_ist"].dt.date

    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")

    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_{STRATEGY_NAME}_{ts}_IST.csv")
    sig_df.to_csv(signals_path, index=False)

    print(f"\nSaved {len(sig_df)} multi-timeframe LONG signals to: {signals_path}")
    print("\nLast few LONG signals:")
    print(sig_df.tail(20).to_string(index=False))

    # --------- PART 1b: DAILY COUNTS ---------
    print("\n--- Computing per-day LONG signal counts ---")
    daily_counts = sig_df.groupby("signal_date").size().reset_index(name="LONG_signals")
    daily_counts.rename(columns={"signal_date": "date"}, inplace=True)

    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_counts_{STRATEGY_NAME}_{ts}_IST.csv")
    daily_counts.to_csv(daily_counts_path, index=False)

    print(f"\nSaved daily LONG signal counts to: {daily_counts_path}")
    print("\nDaily counts preview:")
    print(daily_counts.tail(20).to_string(index=False))

    # --------- PART 2: EVALUATE TARGET_PCT OUTCOME ON DAILY ---------
    print(f"\n--- Evaluating +{TARGET_LABEL} outcome on LONG signals (Daily eval) ---")
    print(f"Assuming capital per signal : ₹{CAPITAL_PER_SIGNAL_RS:,.2f}")
    print(f"Total LONG signals in file : {len(sig_df)}")

    tickers_in_signals = sig_df["ticker"].unique().tolist()
    print(f"Found {len(tickers_in_signals)} tickers in generated LONG signals.")

    data_cache: dict[str, pd.DataFrame] = {}
    for tkr in tickers_in_signals:
        path_d = _find_file(DIR_D, tkr)
        if not path_d:
            data_cache[tkr] = pd.DataFrame()
            continue

        df_d = _safe_read_daily(path_d)
        data_cache[tkr] = df_d

    results = []
    for _, row in sig_df.iterrows():
        ticker = row["ticker"]
        df_d = data_cache.get(ticker, pd.DataFrame())

        if df_d.empty:
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
            new_info = evaluate_signal_on_daily_long_only(row, df_d)

        merged_row = row.to_dict()
        merged_row.update(new_info)
        results.append(merged_row)

    out_df = pd.DataFrame(results).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_{STRATEGY_NAME}_{EVAL_SUFFIX}_{ts}_IST.csv")
    out_df.to_csv(eval_path, index=False)
    print(f"\nSaved {TARGET_LABEL} evaluation (LONG only, Daily eval) to: {eval_path}")

    total_long = len(out_df)
    hits_all = int(out_df[HIT_COL].sum())
    hit_pct_all = (hits_all / total_long * 100.0) if total_long else 0.0
    print(f"\n[LONG SIGNALS - ALL] Total evaluated: {total_long}, {TARGET_LABEL} hit: {hits_all} ({hit_pct_all:.1f}%)")

    if total_long > N_RECENT_SKIP:
        non_recent = out_df.iloc[:-N_RECENT_SKIP].copy()
        hits_nr = int(non_recent[HIT_COL].sum())
        hit_pct_nr = (hits_nr / len(non_recent) * 100.0) if len(non_recent) else 0.0
        print(
            f"\n[LONG SIGNALS - NON-RECENT] (excluding last {N_RECENT_SKIP}) "
            f"Evaluated: {len(non_recent)}, {TARGET_LABEL} hit: {hits_nr} ({hit_pct_nr:.1f}%)"
        )
    else:
        print(f"\n[LONG SIGNALS - NON-RECENT] Not enough rows to exclude last {N_RECENT_SKIP}.")

    # --------- CAPITAL-CONSTRAINED SIM ---------
    print("\n--- CAPITAL-CONSTRAINED PORTFOLIO SIMULATION ---")
    print(f"Max capital pool           : ₹{MAX_CAPITAL_RS:,.2f}")
    print(f"Capital per signal (fixed) : ₹{CAPITAL_PER_SIGNAL_RS:,.2f}")

    out_df_constrained, cap_summary = apply_capital_constraint(out_df)

    print(f"Signals generated (total)  : {cap_summary['total_signals']}")
    print(f"Signals actually taken     : {cap_summary['signals_taken']}")
    print(f"Gross capital deployed     : ₹{cap_summary['gross_invested']:,.2f}")
    print(f"Final portfolio value      : ₹{cap_summary['final_portfolio_value']:,.2f}")
    print(f"Net portfolio P&L          : ₹{cap_summary['net_pnl_rs']:,.2f} ({cap_summary['net_pnl_pct']:.2f}%)")

    # overwrite eval with constrained columns
    out_df_constrained.to_csv(eval_path, index=False)
    print(f"\nRe-saved evaluation with constrained-capital columns to: {eval_path}")

    # --------- Day-wise summary ---------
    print("\n--- Building day-wise summary (signals, P/L) ---")
    daily_summary = build_daily_summary(out_df_constrained)

    daily_summary_path = os.path.join(
        OUT_DIR,
        f"multi_tf_daily_summary_{STRATEGY_NAME}_{EVAL_SUFFIX}_{ts}_IST.csv"
    )
    daily_summary.to_csv(daily_summary_path, index=False)

    print(f"Saved day-wise summary to: {daily_summary_path}")
    print(daily_summary.tail(20).to_string(index=False))

    cols_preview = [
        "itr", "signal_time_ist", "signal_date", "ticker", "signal_side", "strategy",
        "entry_price", "exit_time", "exit_price", "pnl_pct", "pnl_rs",
        HIT_COL, TIME_TD_COL, TIME_CAL_COL,
        "max_favorable_pct", "max_adverse_pct",
        "taken", "invested_amount_constrained", "final_value_constrained",
    ]
    existing = [c for c in cols_preview if c in out_df_constrained.columns]
    print("\nLast few evaluated LONG signals (Daily):")
    print(out_df_constrained[existing].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
