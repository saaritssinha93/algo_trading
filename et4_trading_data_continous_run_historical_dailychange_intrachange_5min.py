# update_daily_and_intra_change.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pytz

INDICATORS_DIR = "main_indicators_july_5min2"
TZ = pytz.timezone("Asia/Kolkata")

def _date_only_series(date_col):
    """
    Build a date-only series in IST without mutating the 'date' column.
    Works for naive or tz-aware timestamps.
    """
    dt = pd.to_datetime(date_col, errors="coerce")
    # If tz-naive, assume it's already IST; otherwise convert to IST
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(TZ)
    else:
        dt = dt.dt.tz_convert(TZ)
    return dt.dt.date

def _compute_intra_change(df, date_only):
    """
    % change vs previous 5-min bar within same day.
    First bar of a day will be NaN (expected).
    """
    return (
        df.groupby(date_only)["close"]
          .pct_change()
          .mul(100.0)
    )

def calculate_adx(df, period=14):
    """
    Wilder's ADX (vectorized). Returns ADX in [0, 100].
    """
    # Require columns
    for col in ("high", "low", "close"):
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)

    high  = pd.to_numeric(df["high"],  errors="coerce")
    low   = pd.to_numeric(df["low"],   errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.Series(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy()
        ]),
        index=df.index
    )

    # Directional Movement
    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing (EMA with alpha=1/period)
    alpha      = 1.0 / float(period)
    atr        = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_sm= minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # DI, DX, ADX
    eps     = 1e-10
    plus_di = 100.0 * (plus_dm_sm / (atr + eps))
    minus_di= 100.0 * (minus_dm_sm / (atr + eps))
    dx      = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)
    adx     = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.clip(lower=0, upper=100)

# -------- NEW: Robust Stochastic --------
def calculate_stochastic_fast(df, k_period=14, d_period=3):
    """Fast Stochastic: %K raw, %D = SMA(%K, d_period)."""
    low_min  = df['low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
    range_   = (high_max - low_min)

    # Avoid eps; define %K=0 when range is 0 (flat window)
    k = pd.Series(0.0, index=df.index)
    valid = range_ > 0
    k.loc[valid] = 100.0 * (df['close'].loc[valid] - low_min.loc[valid]) / range_.loc[valid]
    k = k.clip(lower=0.0, upper=100.0)

    d = k.rolling(window=d_period, min_periods=d_period).mean().clip(0.0, 100.0)
    return k, d

def _ma(x, window, kind="sma"):
    if kind == "ema":
        return x.ewm(span=window, adjust=False).mean()
    return x.rolling(window=window, min_periods=window).mean()

def calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma"):
    """Slow Stochastic: %K_slow = MA(%K_fast, k_smooth); %D = MA(%K_slow, d_period)."""
    k_fast, _ = calculate_stochastic_fast(df, k_period=k_period, d_period=1)
    k_slow = _ma(k_fast, k_smooth, kind=ma_kind).clip(0.0, 100.0)
    d      = _ma(k_slow, d_period,  kind=ma_kind).clip(0.0, 100.0)
    return k_slow, d

# --------------------------------------------

def _compute_daily_change_vs_prev_day_close(df, date_only):
    """
    % change vs previous trading day's final close.
    First trading day in file will be NaN (no prior close).
    """
    last_close_per_day = df.groupby(date_only, sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)
    base  = pd.Series(date_only).map(prev_day_last_close).to_numpy()
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_change = (close - base) / base * 100.0
    return pd.Series(daily_change, index=df.index)

def _insert_or_assign(df, col_name, values, after_col=None):
    """
    Assigns df[col_name]=values if exists; otherwise inserts it.
    If inserting and 'after_col' exists, place right after it; else append at end.
    """
    if col_name in df.columns:
        df[col_name] = values
        return
    if after_col and after_col in df.columns:
        pos = list(df.columns).index(after_col) + 1
        df.insert(pos, col_name, values)
    else:
        df[col_name] = values

def process_csv(path):
    df = pd.read_csv(path)
    # Sanity checks
    if "date" not in df.columns or "close" not in df.columns:
        print(f"[SKIP] {os.path.basename(path)} missing required columns.")
        return

    # Build date-only (IST) WITHOUT touching df['date']
    date_only = _date_only_series(df["date"])

    # Compute series
    intra_change = _compute_intra_change(df, date_only)
    daily_change = _compute_daily_change_vs_prev_day_close(df, date_only)
    adx_series   = calculate_adx(df, period=14)

    # --- NEW: Stochastic (%K, %D) ---
    stoch_k, stoch_d = calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma")

    # Update/insert columns (do not touch anything else)
    _insert_or_assign(df, "Daily_Change", daily_change, after_col=None)             # keep position if exists
    _insert_or_assign(df, "Intra_Change", intra_change, after_col="Daily_Change")   # insert right after Daily_Change if present
    _insert_or_assign(df, "ADX", adx_series, after_col=None)                        # keep position if exists
    _insert_or_assign(df, "Stoch_%K", stoch_k, after_col=None)                      # keep position if exists
    _insert_or_assign(df, "Stoch_%D", stoch_d, after_col="Stoch_%K")                # place %D right after %K if inserting

    # Save back
    df.to_csv(path, index=False)
    print(f"[OK] Updated: {os.path.basename(path)}")

def main():
    if not os.path.isdir(INDICATORS_DIR):
        print(f"Directory not found: {INDICATORS_DIR}")
        return

    for fname in os.listdir(INDICATORS_DIR):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(INDICATORS_DIR, fname)
        try:
            process_csv(path)
        except Exception as e:
            print(f"[ERR] {fname}: {e}")

if __name__ == "__main__":
    main()
