# -*- coding: utf-8 -*-
"""
1) Generate multi-timeframe positional LONG signals using (RELAXED but slightly stricter):
    • Weekly trend filter
    • Previous-day Daily momentum + volatility filter
    • Hourly timing trigger

2) Evaluate those signals using 1H data to see if a +TARGET_PCT% move was achieved
   after entry (LONG side → +TARGET_PCT% target).

3) Portfolio-level simulation:
    • You invest CAPITAL_PER_SIGNAL_RS in each signal you TAKE.
    • Total capital pool is limited to MAX_CAPITAL_RS.
      - If available capital < CAPITAL_PER_SIGNAL_RS at a new signal,
        that signal is SKIPPED.
      - Once some open trades exit and free capital, you can start taking
        new signals again.

Daily / Weekly alignment rules:

    DAILY:
        • For an hourly bar on calendar day D, use the **previous day’s**
          daily bar (D-1). Implementation:
            - Shift Daily 'date' forward by +1 calendar day *for merge only*
              so that `merge_asof(..., direction="backward")` picks D-1.

    WEEKLY:
        • Weekly bars are week-end bars (e.g. Friday close).
        • For intraday bars on the same calendar date as a Weekly bar, do NOT
          use that week’s bar (no same-date lookahead).
        • Implementation:
            - Shift Weekly 'date' forward by +1 second *for merge only* so
              that all intraday bars on that calendar date still see the
              previous week's weekly bar.

For each ticker:
    • Read corresponding Weekly, Daily and 1H indicator data
    • Merge them on time (as-of joins, backward) so each 1H bar knows the
      previous-day Daily and previous-week Weekly context (per above rules)
    • On each 1H bar, check:

      WEEKLY FILTER (trend, mildly strict):
        - close_W > EMA_50_W
        - EMA_50_W > EMA_200_W
        - RSI_W > 48
        - ADX_W > 15

      DAILY FILTER (setup, mildly strict):
        - close_D > EMA_50_D
        - RSI_D > 48
        - MACD_Hist_D >= 0
        - ATR_D >= 0.93 * ATR_D_5d_mean
        - close_D >= Recent_High_10_D * 0.985

      HOURLY TRIGGER (entry timing, mildly strict):
        - close_H > VWAP_H
        - RSI_H > 49
        - MACD_Hist_H > 0
        - ATR_H >= 0.93 * ATR_H_5bar_mean
        - close_H >= Recent_High_5_H * 0.998

    • Whenever all 3 blocks are TRUE on a bar, emit a LONG signal:
        - signal_time_ist = 1H bar time (IST)
        - entry_price     = close_H
        - signal_side     = "LONG"

Then:
    • For each generated LONG signal, scan forward 1H data:
        - target = entry_price * TARGET_MULT, check high >= target
    • Record:
        - hit_<TARGET_INT>pct (True/False)
        - time_to_<TARGET_INT>pct_days       (trading days)
        - days_to_<TARGET_INT>pct_calendar   (calendar days)
        - exit_time
        - exit_price
        - pnl_pct
        - pnl_rs
        - max_favorable_pct
        - max_adverse_pct
        - invested_amount              (unconstrained, always CAPITAL_PER_SIGNAL_RS)
        - final_value_per_signal       (unconstrained, per-trade)

Additionally:
    • For each calendar date (IST), count how many LONG signals are generated
      and write a daily counts CSV.

Portfolio-level constrained sim with cap MAX_CAPITAL_RS:
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

Inputs (directories follow your existing scripts):
    • 1H data per ticker    : main_indicators_1h/<TICKER>_main_indicators_1h.csv
    • Daily data per ticker : main_indicators_daily/<TICKER>_main_indicators_daily.csv
    • Weekly data per ticker: main_indicators_weekly/<TICKER>_main_indicators_weekly.csv

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
TARGET_PCT: float = 3.0  # <-- change this once (e.g. 4.0, 5.0, 7.0, 10.0)
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0
TARGET_INT: int = int(TARGET_PCT)          # used in names, assume integer percent
TARGET_LABEL: str = f"{TARGET_INT}%"       # for printing

# Capital sizin
CAPITAL_PER_SIGNAL_RS: float = 30000.0     # fixed rupee amount per signal
MAX_CAPITAL_RS: float = 1000000.0           # total capital pool (e.g. 5 lakh)

# Dynamic column names based on TARGET_PCT
HIT_COL: str = f"hit_{TARGET_INT}pct"
TIME_TD_COL: str = f"time_to_{TARGET_INT}pct_days"
TIME_CAL_COL: str = f"days_to_{TARGET_INT}pct_calendar"

# Eval filename suffix based on TARGET_PCT
EVAL_SUFFIX: str = f"{TARGET_INT}pct_eval"

# For outcome evaluation
TRADING_HOURS_PER_DAY = 6.25
N_RECENT_SKIP = 100  # how many most-recent signals to treat as "recent" (EXCLUDE last 500)


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
    For a single ticker:
        • Read 1H, Daily, Weekly indicator CSVs
        • Merge them into a single 1H-based dataframe
        • Apply the mildly-strict relaxed multi-timeframe LONG strategy
        • Return a list of LONG signal dictionaries
    """
    path_1h = os.path.join(DIR_1H, f"{ticker}_main_indicators_1h.csv")
    path_d  = os.path.join(DIR_D,  f"{ticker}_main_indicators_daily.csv")
    path_w  = os.path.join(DIR_W,  f"{ticker}_main_indicators_weekly.csv")

    df_h = _safe_read_tf(path_1h)
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

    # ----------------- STRATEGY CONDITIONS (MILDLY STRICT LONG) -----------------

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
        (df_hdw["close_H"] > df_hdw["VWAP_H"]) &   # must be above VWAP
        (df_hdw["RSI_H"] > 49.0) &                 # slightly stronger intraday strength
        (df_hdw["MACD_Hist_H"] > 0.0) &            # momentum clearly positive
        hourly_atr_expansion &
        hourly_recent_high_break
    )

    # FINAL LONG SIGNAL: all three blocks TRUE
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
def apply_capital_constraint(out_df: pd.DataFrame):
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

    Returns:
        df_with_flags, summary_dict
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
        print("No signal evaluations produced.")
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

    print(f"\nSaved {TARGET_LABEL} evaluation for LONG signals to: {eval_path}")

    total_all = len(out_df)

    # ---------- SUMMARY 1: ALL LONG ROWS (UNCONSTRAINED) ----------
    hits_all = out_df[HIT_COL].sum()
    hit_pct_all = (hits_all / total_all * 100.0) if total_all > 0 else 0.0
    print(
        f"\n[LONG SIGNALS - ALL] (unconstrained) Total evaluated: {total_all}, "
        f"{TARGET_LABEL} target hit: {hits_all} ({hit_pct_all:.1f}%)"
    )

    # ---------- SUMMARY 2: NON-RECENT ROWS (excluding last N_RECENT_SKIP) ----------
    if total_all > N_RECENT_SKIP:
        out_df_non_recent = out_df.iloc[:-N_RECENT_SKIP].copy()
        total_non_recent = len(out_df_non_recent)
        hits_non_recent = out_df_non_recent[HIT_COL].sum()
        hit_pct_non_recent = (
            hits_non_recent / total_non_recent * 100.0
            if total_non_recent > 0 else 0.0
        )

        print(f"\n[LONG SIGNALS - NON-RECENT] (unconstrained, excluding last {N_RECENT_SKIP} signals)")
        print(
            f"Signals evaluated (non-recent): {total_non_recent}, "
            f"{TARGET_LABEL} target hit: {hits_non_recent} ({hit_pct_non_recent:.1f}%)"
        )
    else:
        print(
            f"\n[LONG SIGNALS - NON-RECENT] Not enough rows to exclude last "
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
    main()
