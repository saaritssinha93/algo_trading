# -*- coding: utf-8 -*-
"""
1) Generate multi-timeframe positional LONG signals using:
    • Weekly trend filter
    • Previous-day Daily move filter (Daily_Change_prev)
    • Hourly timing trigger

2) Evaluate those LONG signals using 1H data to see if a +TARGET_PCT% move was achieved
   after entry:
       - LONG  → +TARGET_PCT% target  (high >= (1 + TARGET_PCT/100) * entry_price)

3) Assume:
    • You invest CAPITAL_PER_SIGNAL_RS in each signal you TAKE.
    • Total capital pool is limited to MAX_CAPITAL_RS.
      - If available capital < CAPITAL_PER_SIGNAL_RS at a new signal,
        that signal is SKIPPED.
      - Once some open trades exit and free capital, you can start taking
        new signals again.

Weekly rule (NO SAME-DATE LOOKAHEAD):
    • Weekly bars are stored at week-end (e.g. Friday close).
    • For any intraday bar on that same calendar date, we must still use
      the *previous* weekly bar, not the current week’s bar.
    • Implementation:
        - Before merging, shift weekly 'date' forward by a tiny delta
          (e.g. +1 second).
        - Then do pd.merge_asof(..., direction="backward").
        - This ensures all hourly bars on the weekly bar’s calendar date
          still see the previous week's weekly bar.

Daily rule uses **previous day's** Daily_Change:
    • For an hourly bar on day D, we look at Daily_Change of day D-1.

Logic per ticker:
    • Read Weekly, Daily and 1H indicator data
    • On Daily TF, create Daily_Change_prev = Daily_Change.shift(1)
    • Suffix columns (_H, _D, _W) and merge Daily & Weekly onto Hourly via
      backward as-of joins so each 1H bar knows the latest Daily & Weekly context
    • On each 1H bar, check:

      WEEKLY FILTER (slightly relaxed vs strict):
        LONG trend:
            - close_W > EMA_50_W
            - EMA_50_W > EMA_200_W
            - RSI_W > 47
            - ADX_W > 13

      DAILY FILTER (previous-day Daily_Change_prev_D):
        Using Daily_Change_prev_D from DAILY timeframe:

        LONG:
            Daily_Change_prev_D >= +3.0%

      HOURLY TRIGGER (entry timing, slightly relaxed):
        LONG intraday:
            - close_H >= VWAP_H
            - RSI_H > 48
            - MACD_Hist_H >= 0
            - ATR_H ≥ 0.93 * ATR_H_5bar_mean
            - close_H ≥ 0.997 * Recent_High_5_H

    • Whenever WEEKLY + DAILY + HOURLY blocks are TRUE:
        - Emit a LONG signal with:
            - signal_time_ist = 1H bar time (IST)
            - entry_price     = close_H
            - signal_side     = "LONG"

Additionally:
    • For each calendar date (IST), count how many LONG signals are generated
      and write a daily counts CSV.

Then:
    • For each LONG signal, scan forward 1H data:
        - target = entry_price * TARGET_MULT, check high >= target
    • Record:
        - hit_<TARGET_INT>pct (True/False)
        - time_to_<TARGET_INT>pct_days       → trading days (using TRADING_HOURS_PER_DAY)
        - days_to_<TARGET_INT>pct_calendar   → integer calendar days between entry & 1st hit
        - exit_time
        - exit_price
        - pnl_pct
        - pnl_rs
        - max_favorable_pct
        - max_adverse_pct
        - invested_amount  (unconstrained, always CAPITAL_PER_SIGNAL_RS)
        - final_value_per_signal (unconstrained, per-trade)

Portfolio-level simulation with cap MAX_CAPITAL_RS:
    • Process signals chronologically.
    • At each signal:
        - Release any trades whose exit_time <= this signal's time
          (capital + P&L returned to the pool).
        - If available_capital >= CAPITAL_PER_SIGNAL_RS:
              mark signal as taken, reserve capital, and add to open-trades list.
          else:
              mark signal as NOT taken (skipped).
    • At the end, close any remaining open trades and add their final
      values back to capital.

    Per-signal (constrained):
        - taken (bool)
        - invested_amount_constrained
        - final_value_constrained

    Portfolio summary:
        - Total capital pool (MAX_CAPITAL_RS)
        - Signals generated vs signals taken
        - Gross capital deployed (sum of per-signal constrained investments)
        - Final portfolio value
        - Net P&L (₹ and %)

Outputs:
    • signals/multi_tf_signals_<YYYYMMDD_HHMM>_IST.csv
    • signals/multi_tf_signals_daily_counts_<YYYYMMDD_HHMM>_IST.csv
    • signals/multi_tf_signals_<TARGET_INT>pct_eval_<YYYYMMDD_HHMM>_IST.csv
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_1H  = "main_indicators_1h"
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Target configuration (single source of truth) ----
TARGET_PCT: float = 5  # <-- change this once (e.g. 4.0, 5.0, 7.0, 10.0)
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)  # used in names, assume integer percent
TARGET_LABEL: str = f"{TARGET_INT}%"  # for printing

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 50000.0   # fixed rupee amount per signal
MAX_CAPITAL_RS: float = 1000000.0        # total capital pool

# Dynamic column names based on TARGET_PCT
HIT_COL: str = f"hit_{TARGET_INT}pct"
TIME_TD_COL: str = f"time_to_{TARGET_INT}pct_days"
TIME_CAL_COL: str = f"days_to_{TARGET_INT}pct_calendar"

# Eval filename suffix based on TARGET_PCT
EVAL_SUFFIX: str = f"{TARGET_INT}pct_eval"

# For outcome evaluation (LONG only)
TRADING_HOURS_PER_DAY = 6.25
N_RECENT_SKIP = 25  # how many most-recent LONG signals to treat as "recent"


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
        directory = main_indicators_1h
        ending    = "_main_indicators_1h.csv"
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


# ----------------------- SIGNAL GENERATION -----------------------
def build_signals_for_ticker(ticker: str) -> list[dict]:
    """
    Moderately-filtered MTF strategy (slightly relaxed vs previous strict version).

    • Weekly:
        - Prefers uptrend (close > EMAs, decent RSI/ADX)
        - Still filters out extreme overbought regimes, but a bit more tolerant.
    • Daily:
        - Uses previous-day Daily_Change band (~2%–9%) to avoid
          very weak and extreme blow-off moves.
    • Hourly (moderately strict):
        - Price above EMAs and VWAP
        - Reasonable RSI / MACD / Stoch profile
        - ATR & ADX support trend strength
        - Breakout / near-breakout logic slightly relaxed

    Returns:
        List of LONG signal dictionaries for this ticker.
    """
    # -------------------- READ DATA --------------------
    path_1h = os.path.join(DIR_1H, f"{ticker}_main_indicators_1h.csv")
    path_d  = os.path.join(DIR_D,  f"{ticker}_main_indicators_daily.csv")
    path_w  = os.path.join(DIR_W,  f"{ticker}_main_indicators_weekly.csv")

    df_h = _safe_read_tf(path_1h)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_h.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return []

    # -------------------- DAILY EXTRAS --------------------
    # Compute previous-day Daily_Change on daily before suffixing
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # -------------------- SUFFIX TIMEFRAMES --------------------
    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # -------------------- HOURLY EXTRAS --------------------
    # Safer construction: only roll/shift if columns exist
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

    # -------------------- MERGE D + W ONTO H --------------------
    df_hd = pd.merge_asof(
        df_h.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward"
    )

    df_w_sorted = df_w.sort_values("date").copy()
    # Tiny +1s shift to avoid same-timestamp merge edge cases if needed
    df_w_sorted["date"] += pd.Timedelta(seconds=1)

    df_hdw = pd.merge_asof(
        df_hd.sort_values("date"),
        df_w_sorted,
        on="date",
        direction="backward"
    )

    # We at least need weekly trend and previous-day daily move
    df_hdw = df_hdw.dropna(
        subset=[
            "close_W",
            "EMA_50_W",
            "EMA_200_W",
            "RSI_W",
            "ADX_W",
            "Daily_Change_prev_D",
        ]
    )

    if df_hdw.empty:
        return []

    signals: list[dict] = []

    # ============================================================
    #                     WEEKLY (MODERATE)
    # ============================================================
    weekly_strong = (
        (df_hdw["close_W"] > df_hdw["EMA_50_W"]) &
        (df_hdw["EMA_50_W"] > df_hdw["EMA_200_W"]) &
        (df_hdw["RSI_W"] > 43.0) &        # relaxed: was 45.0
        (df_hdw["ADX_W"] > 11.0)          # relaxed: was 12.0
    )

    weekly_mild = (
        (df_hdw["close_W"] > df_hdw["EMA_200_W"] * 1.010) &    # relaxed: was 1.015
        (df_hdw["RSI_W"].between(43.0, 67.0)) &                # relaxed: was 45–65
        (df_hdw["ADX_W"] > 10.0)                               # relaxed: was 11.0
    )

    weekly_not_extreme = df_hdw["RSI_W"] < 85.0                # relaxed: was 80.0

    weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme

    # ============================================================
    #                     DAILY (SLIGHTLY RELAXED BAND)
    # ============================================================
    daily_prev = df_hdw["Daily_Change_prev_D"]
    # Allow slightly weaker & slightly bigger prior move
    daily_ok = (daily_prev >= 2) & (daily_prev <= 10.0)       # relaxed: was 2.5–8.0

    # ============================================================
    #                     HOURLY (MODERATELY STRICT)
    # ============================================================
    # Trend: still above EMAs, but not as far away as strict version
    hourly_trend = (
        (df_hdw["close_H"] >= df_hdw["EMA_200_H"] * 1.00) &   # relaxed: was 1.010
        (df_hdw["close_H"] >= df_hdw["EMA_50_H"]  * 1.00)     # relaxed: was 1.010
    )

    # RSI: relax thresholds a bit, but keep “rising” logic
    rsi_ok = (
        (df_hdw["RSI_H"] > 50.0) |                             # relaxed: was 52.0
        ((df_hdw["RSI_H"] > 47.0) & (df_hdw["RSI_H"] > df_hdw["RSI_H_prev"]))  # relaxed: was 49
    )

    # MACD: allow a bit more negative and smaller improvement requirement
    macd_ok = (
        (df_hdw["MACD_Hist_H"] > 0) &                      # relaxed: was -0.02
        (df_hdw["MACD_Hist_H"] >= df_hdw["MACD_Hist_H_prev"] * 0.95)  # relaxed: was 0.95
    )

    # Stoch: slightly lower floor
    stoch_ok = (
        (df_hdw["Stoch_%K_H"] >= df_hdw["Stoch_%D_H"]) &
        (df_hdw["Stoch_%K_H"] >= 30.0)                         # relaxed: was 30.0
    )

    momentum_ok = rsi_ok & macd_ok & stoch_ok

    # Volatility / trend strength: relax ATR & ADX a bit
    hourly_vol_ok = (
        (df_hdw["ATR_H"] >= df_hdw["ATR_H_5bar_mean"] * 0.98) &   # relaxed: was 0.98
        (
            (df_hdw["ADX_H"] >= 20.0) |                           # relaxed: was 14.0
            (df_hdw["ADX_H"] >= df_hdw["ADX_H_5bar_mean"])
        )
    )

    price_above_vwap = df_hdw["close_H"] >= df_hdw["VWAP_H"]

    # Breakout rules: still demand quality, but allow a bit closer-to-high near-breakout
    breakout_strict = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"]

    breakout_near = df_hdw["close_H"] >= df_hdw["Recent_High_5_H"] * 0.999   # relaxed: was 0.999

    intra_chg = df_hdw.get("Intra_Change_H", pd.Series(0.0, index=df_hdw.index)).fillna(0.0)
    strong_up = intra_chg >= 0.3                                             # relaxed: was 0.15

    breakout_ok = price_above_vwap & (
        breakout_strict |
        (breakout_near & strong_up & (df_hdw["RSI_H"] >= 50.0))               # relaxed: was 52
    )

    hourly_ok = hourly_trend & momentum_ok & hourly_vol_ok & breakout_ok

    # ============================================================
    #                     FINAL LONG MASK
    # ============================================================
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



# ----------------------- HELPERS FOR TARGET EVAL -----------------------
def _safe_read_1h(path: str) -> pd.DataFrame:
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


def evaluate_signal_on_1h_long_only(row: pd.Series, df_1h: pd.DataFrame) -> dict:
    """
    Evaluate +TARGET_PCT% outcome for LONG signals.

    Returns a dict with keys:
        HIT_COL            (bool)
        TIME_TD_COL        (float, trading days)
        TIME_CAL_COL       (int, calendar days)
        exit_time
        exit_price
        pnl_pct
        pnl_rs
        max_favorable_pct
        max_adverse_pct
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

    future = df_1h[df_1h["date"] > entry_time].copy()
    if future.empty:
        # No future bars: treat as flat at entry
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

    # LONG: +TARGET_PCT% target
    target_price = entry_price * TARGET_MULT
    hit_mask = future["high"] >= target_price

    max_fav = ((future["high"] - entry_price) / entry_price * 100.0).max()
    max_adv = ((future["low"] - entry_price) / entry_price * 100.0).min()

    if hit_mask.any():
        # Target hit: exit at target price at first hit
        first_hit_idx = hit_mask[hit_mask].index[0]
        hit_time = future.loc[first_hit_idx, "date"]

        exit_time = hit_time
        exit_price = target_price

        # Trading days (based on configured trading hours per day)
        dt_hours = (hit_time - entry_time).total_seconds() / 3600.0
        dt_days = dt_hours / TRADING_HOURS_PER_DAY

        # Calendar day difference between entry and first hit
        cal_days = (hit_time.date() - entry_time.date()).days

        hit_flag = True
        time_to_target_days = dt_days
        days_to_target_calendar = cal_days
    else:
        # Target never hit: mark-to-market at last available close
        last_row = future.iloc[-1]
        exit_time = last_row["date"]
        exit_price = float(last_row["close"]) if "close" in last_row else entry_price

        hit_flag = False
        time_to_target_days = np.nan
        days_to_target_calendar = np.nan

    # P&L from entry to exit (per trade, assuming full CAPITAL_PER_SIGNAL_RS allocated)
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
            * First, release any open trades whose exit_time <= entry_time.
            * If available_capital >= CAPITAL_PER_SIGNAL_RS -> take trade.
              Else -> skip trade.

    Adds to out_df:
        - taken (bool)
        - invested_amount_constrained
        - final_value_constrained

    Returns a dict summary with:
        - total_signals
        - signals_taken
        - gross_invested
        - final_portfolio_value
        - net_pnl_rs
        - net_pnl_pct
    """
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    available_capital = MAX_CAPITAL_RS
    open_trades = []  # list of dicts: {"exit_time": ts, "value": float}

    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        # 1) Release any trades that have exited before or at this entry_time
        still_open = []
        for trade in open_trades:
            exit_time = trade["exit_time"]
            # If exit_time is NaT for some reason, treat as already exited
            if pd.isna(exit_time) or exit_time <= entry_time:
                available_capital += trade["value"]
            else:
                still_open.append(trade)
        open_trades = still_open

        # 2) Try to take this trade
        if available_capital >= CAPITAL_PER_SIGNAL_RS:
            # Take trade
            df.at[idx, "taken"] = True
            available_capital -= CAPITAL_PER_SIGNAL_RS

            exit_time = row["exit_time"]
            if pd.isna(exit_time):
                exit_time = entry_time  # degenerate case: no future bars

            final_value = row["final_value_per_signal"]
            open_trades.append(
                {
                    "exit_time": exit_time,
                    "value": final_value,
                }
            )
        else:
            # Skip trade due to insufficient capital
            df.at[idx, "taken"] = False

    # 3) After processing all entries, release all remaining open trades
    for trade in open_trades:
        available_capital += trade["value"]

    # Constrained invested/final per signal
    df["invested_amount_constrained"] = np.where(df["taken"], CAPITAL_PER_SIGNAL_RS, 0.0)
    df["final_value_constrained"] = np.where(df["taken"], df["final_value_per_signal"], 0.0)

    # Portfolio summary
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
def main():
    # --------- PART 1: GENERATE LONG SIGNALS ---------
    tickers_1h = _list_tickers_from_dir(DIR_1H, "_main_indicators_1h.csv")
    tickers_d  = _list_tickers_from_dir(DIR_D,  "_main_indicators_daily.csv")
    tickers_w  = _list_tickers_from_dir(DIR_W,  "_main_indicators_weekly.csv")

    tickers = tickers_1h & tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across 1H, Daily, and Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with all 3 timeframes.")

    all_signals: list[dict] = []

    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Processing {ticker} ...")
        try:
            ticker_signals = build_signals_for_ticker(ticker)
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
    sig_df["signal_date"] = sig_df["signal_time_ist"].dt.tz_convert(IST).dt.date

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

    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_counts_{ts}_IST.csv")
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
        path_1h = os.path.join(DIR_1H, f"{t}_main_indicators_1h.csv")
        df_1h = _safe_read_1h(path_1h)
        if df_1h.empty:
            print(f"- No 1H data for {t}, skipping its signals in evaluation.")
            one_hour_data_cache[t] = pd.DataFrame()
        else:
            one_hour_data_cache[t] = df_1h

    for _, row in sig_df.iterrows():
        ticker = row["ticker"]
        df_1h = one_hour_data_cache.get(ticker, pd.DataFrame())
        if df_1h.empty:
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
            new_info = evaluate_signal_on_1h_long_only(row, df_1h)

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

    # Overwrite eval file with constrained columns as well (so you can inspect taken vs skipped)
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
    main()
