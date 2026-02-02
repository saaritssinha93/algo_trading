# -*- coding: utf-8 -*-
"""
Generate medium-term positional entry signals historically by combining:
    • Weekly  (trend filter)
    • Daily   (swing confirmation)
    • 1H      (entry trigger)

For EVERY 1H bar (per ticker), we:
    1) Attach latest available Daily + Weekly context using merge_asof
    2) Check Weekly bias, Daily confirmation, and 1H trigger
    3) If all align -> emit a LONG / SHORT signal row

Output:
    signals/positional_signals_history_<YYYYMMDD_HHMM>_IST.csv

You will get multiple dates of signals (not just one),
and typically a few good entries per day across the universe
(depending on how strict the rules are).
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_1H     = "main_indicators_1h"
DIR_DAILY  = "main_indicators_daily"
DIR_WEEKLY = "main_indicators_weekly"
OUT_DIR    = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# If you want to limit which tickers are processed, set a list here.
# Otherwise, tickers will be inferred from weekly files.
TICKERS = None   # e.g. ["RELIANCE", "TCS", "INFY"]


# ----------------------- HELPERS -----------------------
def _safe_read(path: str) -> pd.DataFrame:
    """Read CSV safely, parse date as tz-aware IST, sort by date."""
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(IST)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()


def _rolling_prev_high(series: pd.Series, lookback=5) -> pd.Series:
    """Previous N-bar high (excluding current bar)."""
    return series.shift(1).rolling(lookback, min_periods=1).max()


def _rolling_prev_low(series: pd.Series, lookback=5) -> pd.Series:
    """Previous N-bar low (excluding current bar)."""
    return series.shift(1).rolling(lookback, min_periods=1).min()


def infer_tickers() -> list[str]:
    """Infer tickers from weekly files if TICKERS not set."""
    if TICKERS:
        return TICKERS
    files = glob.glob(os.path.join(DIR_WEEKLY, "*_main_indicators_weekly.csv"))
    tickers = []
    for f in files:
        base = os.path.basename(f)
        t = base.replace("_main_indicators_weekly.csv", "")
        if t:
            tickers.append(t)
    return sorted(list(set(tickers)))


# ----------------------- SIGNAL LOGIC (VECTORISED) -----------------------
def build_signals_for_ticker(ticker: str) -> pd.DataFrame:
    """
    For a single ticker:
        • read W, D, 1H CSVs,
        • merge weekly+daily context onto 1H bars,
        • compute LONG/SHORT signals for EACH 1H bar (with relaxed conditions),
        • return a dataframe with one row per signal.
    """
    path_w = os.path.join(DIR_WEEKLY, f"{ticker}_main_indicators_weekly.csv")
    path_d = os.path.join(DIR_DAILY,  f"{ticker}_main_indicators_daily.csv")
    path_h = os.path.join(DIR_1H,     f"{ticker}_main_indicators_1h.csv")

    df_w = _safe_read(path_w)
    df_d = _safe_read(path_d)
    df_h = _safe_read(path_h)

    if df_w.empty or df_d.empty or df_h.empty:
        print(f"- Skipping {ticker}: missing one of W/D/H datasets.")
        return pd.DataFrame()

    # --- Keep only relevant columns for context ---
    w_cols = ["date", "close", "EMA_50", "EMA_200", "RSI", "ADX", "MACD", "MACD_Signal", "MACD_Hist"]
    d_cols = ["date", "close", "EMA_50", "RSI", "MACD_Hist", "20_SMA"]

    df_w = df_w[[c for c in w_cols if c in df_w.columns]].copy()
    df_d = df_d[[c for c in d_cols if c in df_d.columns]].copy()

    # Prefix W_ / D_ so they don't clash with 1H columns
    df_w = df_w.rename(columns={c: f"W_{c}" for c in df_w.columns if c != "date"})
    df_d = df_d.rename(columns={c: f"D_{c}" for c in df_d.columns if c != "date"})

    # --- Merge-asof Daily and Weekly context onto 1H bars ---
    df_h = df_h.sort_values("date")

    # Attach DAILY context: last daily bar at or before each 1H bar
    df_merged = pd.merge_asof(
        df_h,
        df_d.sort_values("date"),
        on="date",
        direction="backward"
    )

    # Attach WEEKLY context: last weekly bar at or before each 1H bar
    df_merged = pd.merge_asof(
        df_merged.sort_values("date"),
        df_w.sort_values("date"),
        on="date",
        direction="backward"
    )

    # ---------- WEEKLY bias (LONG / SHORT) ----------
    W_close      = df_merged["W_close"]
    W_EMA50      = df_merged["W_EMA_50"]
    W_EMA200     = df_merged["W_EMA_200"]
    W_RSI        = df_merged["W_RSI"]
    W_ADX        = df_merged["W_ADX"]
    W_MACD       = df_merged["W_MACD"]
    W_MACD_Sig   = df_merged["W_MACD_Signal"]
    W_MACD_Hist  = df_merged["W_MACD_Hist"]

    weekly_bias_long = (
        (W_close > W_EMA50) &
        (W_EMA50 > W_EMA200) &
        (W_RSI > 50) &
        (W_ADX > 20) &
        (W_MACD > W_MACD_Sig) &
        (W_MACD_Hist > 0)
    )

    weekly_bias_short = (
        (W_close < W_EMA50) &
        (W_EMA50 < W_EMA200) &
        (W_RSI < 50) &
        (W_ADX > 20) &
        (W_MACD < W_MACD_Sig) &
        (W_MACD_Hist < 0)
    )

    # ---------- DAILY confirmation (LONG / SHORT) ----------
    D_close     = df_merged["D_close"]
    D_EMA50     = df_merged["D_EMA_50"]
    D_RSI       = df_merged["D_RSI"]
    D_MACD_Hist = df_merged["D_MACD_Hist"]
    D_20SMA     = df_merged.get("D_20_SMA", D_EMA50)  # fallback

    # RELAXED long daily confirmation:
    daily_confirm_long = (
        (D_close > D_EMA50) &
        (D_RSI  > 48) &          # was 50
        (D_MACD_Hist > 0)
        # removed (D_close > D_20SMA)
    )

    # Short side left mostly as-is (you can relax similarly if needed)
    daily_confirm_short = (
        (D_close < D_EMA50) &
        (D_RSI  < 50) &
        (D_MACD_Hist < 0) &
        (D_close < D_20SMA)
    )

    # ---------- 1H ENTRY TRIGGERS (relaxed) ----------
    close_h  = df_merged["close"]
    high_h   = df_merged["high"]
    low_h    = df_merged["low"]
    rsi_h    = df_merged["RSI"]
    adx_h    = df_merged["ADX"]
    macd_h   = df_merged["MACD"]
    macds_h  = df_merged["MACD_Signal"]
    vol_h    = df_merged.get("volume", pd.Series(index=df_merged.index, data=np.nan))
    atr_h    = df_merged.get("ATR", pd.Series(index=df_merged.index, data=np.nan))

    # Previous 3-bar high/low (was 5 before)
    prev_high = _rolling_prev_high(high_h, 3)
    prev_low  = _rolling_prev_low(low_h, 3)

    # Breakout / breakdown with a tiny tolerance
    breakout_long    = close_h >= prev_high * 0.998
    breakdown_short  = close_h <= prev_low * 1.002

    # MACD crosses
    macd_cross_up = (macd_h > macds_h) & (macd_h.shift(1) <= macds_h.shift(1))
    macd_cross_dn = (macd_h < macds_h) & (macd_h.shift(1) >= macds_h.shift(1))

    # Momentum & trend strength (relaxed)
    rsi_ok_long  = rsi_h > 52          # was 55
    rsi_ok_short = rsi_h < 48          # slightly looser than 45 if you use shorts
    adx_strong   = (adx_h > 20)        # dropped the "rising vs previous" requirement

    # Volume expansion vs 10-bar avg (relaxed)
    vol_ma10 = vol_h.rolling(10, min_periods=1).mean()
    vol_ok   = vol_h >= (0.9 * vol_ma10)   # was 1.0 * vol_ma10

    # Final 1H triggers
    entry_trigger_long  = breakout_long   & macd_cross_up & rsi_ok_long  & adx_strong & vol_ok
    entry_trigger_short = breakdown_short & macd_cross_dn & rsi_ok_short & adx_strong & vol_ok

    # ---------- COMBINE MULTI-TIMEFRAME CONDITIONS ----------
    long_signal  = weekly_bias_long  & daily_confirm_long  & entry_trigger_long
    short_signal = weekly_bias_short & daily_confirm_short & entry_trigger_short

    # ---------- BUILD OUTPUT ROWS ----------
    rows = []

    # Long side
    long_idx = np.where(long_signal)[0]
    for i in long_idx:
        ep   = float(close_h.iloc[i])
        atrv = float(atr_h.iloc[i]) if pd.notnull(atr_h.iloc[i]) else np.nan
        swing_low = float(prev_low.iloc[i]) if pd.notnull(prev_low.iloc[i]) else np.nan

        stop_atr = ep - 1.5 * atrv if pd.notnull(atrv) else np.nan
        if pd.notnull(swing_low) and pd.notnull(stop_atr):
            stop_loss = min(swing_low, stop_atr)
        else:
            stop_loss = swing_low if pd.notnull(swing_low) else stop_atr

        target1 = ep + 2.0 * atrv if pd.notnull(atrv) else np.nan
        if pd.notnull(target1) and pd.notnull(stop_loss) and ep > stop_loss:
            rr = (target1 - ep) / (ep - stop_loss)
        else:
            rr = np.nan

        rows.append({
            "ticker": ticker,
            "signal_side": "LONG",
            "signal_time_ist": df_merged["date"].iloc[i],
            "entry_price": ep,
            "stop_loss": stop_loss,
            "target1": target1,
            "rr_approx": rr,
            # context flags
            "weekly_bias_long": bool(weekly_bias_long.iloc[i]),
            "daily_confirm_long": bool(daily_confirm_long.iloc[i]),
            "entry_trigger_long": bool(entry_trigger_long.iloc[i]),
            # debug snapshot
            "W_close": float(W_close.iloc[i]) if pd.notnull(W_close.iloc[i]) else np.nan,
            "W_RSI": float(W_RSI.iloc[i]) if pd.notnull(W_RSI.iloc[i]) else np.nan,
            "W_ADX": float(W_ADX.iloc[i]) if pd.notnull(W_ADX.iloc[i]) else np.nan,
            "D_close": float(D_close.iloc[i]) if pd.notnull(D_close.iloc[i]) else np.nan,
            "D_RSI": float(D_RSI.iloc[i]) if pd.notnull(D_RSI.iloc[i]) else np.nan,
            "H_RSI": float(rsi_h.iloc[i]) if pd.notnull(rsi_h.iloc[i]) else np.nan,
            "H_ADX": float(adx_h.iloc[i]) if pd.notnull(adx_h.iloc[i]) else np.nan,
            "H_volume": float(vol_h.iloc[i]) if pd.notnull(vol_h.iloc[i]) else np.nan,
        })

    # Short side (optional; can tighten/loosen separately if needed)
    short_idx = np.where(short_signal)[0]
    for i in short_idx:
        ep   = float(close_h.iloc[i])
        atrv = float(atr_h.iloc[i]) if pd.notnull(atr_h.iloc[i]) else np.nan
        swing_high = float(prev_high.iloc[i]) if pd.notnull(prev_high.iloc[i]) else np.nan

        stop_atr = ep + 1.5 * atrv if pd.notnull(atrv) else np.nan
        if pd.notnull(swing_high) and pd.notnull(stop_atr):
            stop_loss = max(swing_high, stop_atr)
        else:
            stop_loss = swing_high if pd.notnull(swing_high) else stop_atr

        target1 = ep - 2.0 * atrv if pd.notnull(atrv) else np.nan
        if pd.notnull(target1) and pd.notnull(stop_loss) and stop_loss > ep:
            rr = (ep - target1) / (stop_loss - ep)
        else:
            rr = np.nan

        rows.append({
            "ticker": ticker,
            "signal_side": "SHORT",
            "signal_time_ist": df_merged["date"].iloc[i],
            "entry_price": ep,
            "stop_loss": stop_loss,
            "target1": target1,
            "rr_approx": rr,
            "weekly_bias_short": bool(weekly_bias_short.iloc[i]),
            "daily_confirm_short": bool(daily_confirm_short.iloc[i]),
            "entry_trigger_short": bool(entry_trigger_short.iloc[i]),
            "W_close": float(W_close.iloc[i]) if pd.notnull(W_close.iloc[i]) else np.nan,
            "W_RSI": float(W_RSI.iloc[i]) if pd.notnull(W_RSI.iloc[i]) else np.nan,
            "W_ADX": float(W_ADX.iloc[i]) if pd.notnull(W_ADX.iloc[i]) else np.nan,
            "D_close": float(D_close.iloc[i]) if pd.notnull(D_close.iloc[i]) else np.nan,
            "D_RSI": float(D_RSI.iloc[i]) if pd.notnull(D_RSI.iloc[i]) else np.nan,
            "H_RSI": float(rsi_h.iloc[i]) if pd.notnull(rsi_h.iloc[i]) else np.nan,
            "H_ADX": float(adx_h.iloc[i]) if pd.notnull(adx_h.iloc[i]) else np.nan,
            "H_volume": float(vol_h.iloc[i]) if pd.notnull(vol_h.iloc[i]) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)



# ----------------------- MAIN -----------------------
def main():
    tickers = infer_tickers()
    if not tickers:
        print("No tickers found. Ensure weekly files exist or set TICKERS list/TICKERS variable.")
        return

    all_signals = []
    for t in tickers:
        print(f"\nProcessing ticker: {t}")
        df_sig = build_signals_for_ticker(t)
        if df_sig is None or df_sig.empty:
            print(f"  No signals for {t}.")
            continue
        all_signals.append(df_sig)

    if not all_signals:
        print("No signals generated for any ticker.")
        return

    out_df = pd.concat(all_signals, ignore_index=True)

    # Create date-only for grouping / filtering if you want to inspect per day
    out_df["signal_date"] = out_df["signal_time_ist"].dt.date

    # (Optional) If you want to quickly see how many signals per day:
    # print(out_df.groupby("signal_date")["ticker"].count().tail(15))

    # Save full history of signals
    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"positional_signals_history_{ts}_IST.csv")
    out_df.sort_values(["signal_time_ist", "signal_side", "ticker"], inplace=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved historical signals to: {out_path}")
    print("Last few signals:")
    print(
        out_df[["signal_time_ist", "signal_date", "ticker", "signal_side", "entry_price", "stop_loss", "target1", "rr_approx"]]
        .tail(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
