# -*- coding: utf-8 -*-
"""
Batch script: For all tickers and all dates in main_indicators_july,
detect the first valid entry of each type (bullish/bearish breakout/pullback)
for each trading date, and save results to main_indicators_history_entries.
"""

import os, glob
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import numpy as np
import pytz
from tqdm import tqdm

# === Setup ===
INDICATORS_DIR = "main_indicators_july"
OUT_ENTRIES_DIR = "main_indicators_history_entries"
os.makedirs(OUT_ENTRIES_DIR, exist_ok=True)
india_tz = pytz.timezone('Asia/Kolkata')

###############################
# Utilities (as per your code)
###############################
def normalize_time(df, tz='Asia/Kolkata'):
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, tz='Asia/Kolkata'):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, tz)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[expected_cols]
    return df

# --- Example indicator calculation functions ---
def calculate_vwap(df, price_col='close', high_col='high', low_col='low', vol_col='volume'):
    df['TP'] = (df[high_col] + df[low_col] + df[price_col]) / 3.0
    df['cum_tp_vol'] = (df['TP'] * df[vol_col]).cumsum()
    df['cum_vol'] = df[vol_col].cumsum()
    df['VWAP'] = df['cum_tp_vol'] / df['cum_vol']
    return df

def calculate_vwap_bands(df, vwap_col='VWAP', window=20, stdev_multiplier=2.0):
    rolling_diff = (df['close'] - df[vwap_col]).rolling(window)
    df['vwap_std'] = rolling_diff.std()
    df['VWAP_UpperBand'] = df[vwap_col] + stdev_multiplier * df['vwap_std']
    df['VWAP_LowerBand'] = df[vwap_col] - stdev_multiplier * df['vwap_std']
    return df

def calculate_ichimoku(df):
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    return df

def calculate_ttm_squeeze(df, bb_window=20, kc_window=20, kc_mult=1.5, tol=0.5):
    df['mean_bb'] = df['close'].rolling(bb_window).mean()
    df['std_bb'] = df['close'].rolling(bb_window).std()
    df['upper_bb'] = df['mean_bb'] + 2 * df['std_bb']
    df['lower_bb'] = df['mean_bb'] - 2 * df['std_bb']
    df['ema_kc'] = df['close'].ewm(span=kc_window).mean()
    true_range = df['high'] - df['low']
    df['atr_kc'] = true_range.rolling(kc_window).mean()
    df['upper_kc'] = df['ema_kc'] + kc_mult * df['atr_kc']
    df['lower_kc'] = df['ema_kc'] - kc_mult * df['atr_kc']
    df['squeeze_on'] = ((df['lower_bb'] > df['lower_kc'] * (1 - tol)) & (df['upper_bb'] < df['upper_kc'] * (1 + tol))).astype(int)
    df['squeeze_release'] = 0
    for i in range(1, len(df)):
        if df['squeeze_on'].iloc[i-1] == 1 and df['squeeze_on'].iloc[i] == 0:
            df.at[i, 'squeeze_release'] = 1
    return df

def is_ichimoku_bullish(df, idx, tol=0.001):
    row = df.iloc[idx]
    if pd.isna(row['senkou_span_a']) or pd.isna(row['senkou_span_b']):
        return False
    lower_span = min(row['senkou_span_a'], row['senkou_span_b'])
    if row['close'] < lower_span * (1 - tol):
        return False
    if row['tenkan_sen'] < row['kijun_sen'] * (1 - tol):
        return False
    return True

def is_ichimoku_bearish(df, idx, tol=0.001):
    row = df.iloc[idx]
    if pd.isna(row['senkou_span_a']) or pd.isna(row['senkou_span_b']):
        return False
    higher_span = max(row['senkou_span_a'], row['senkou_span_b'])
    if row['close'] > higher_span * (1 + tol):
        return False
    if row['tenkan_sen'] > row['kijun_sen'] * (1 + tol):
        return False
    return True

# [Omitted: all the *price action checks* (is_bullish_breakout_consolidation, check_bullish_breakout, etc),
# just copy from your detection logic and paste here]

# --- For brevity, below is a **reference** (reuse your earlier version):
from collections import defaultdict

def detect_signals_in_memory(ticker, df_for_rolling, df_for_detection, existing_signal_ids):
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected
    # Combine for indicator continuity
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    # Ensure indicator cols present (VWAP, bands, Ichimoku, Squeeze)
    if 'VWAP' not in combined.columns:
        combined = calculate_vwap(combined)
    combined = calculate_vwap_bands(combined, 'VWAP', window=20, stdev_multiplier=2.0)
    combined = calculate_ichimoku(combined)
    combined = calculate_ttm_squeeze(combined)
    # Rolling stats
    rolling_window = 10
    combined['rolling_high_10'] = combined['high'].rolling(rolling_window, min_periods=rolling_window).max()
    combined['rolling_low_10']  = combined['low'].rolling(rolling_window, min_periods=rolling_window).min()
    combined['rolling_vol_10']  = combined['volume'].rolling(rolling_window, min_periods=rolling_window).mean()
    combined['StochDiff']   = combined['Stochastic'].diff() if 'Stochastic' in combined.columns else 0
    combined['SMA20_Slope'] = combined['20_SMA'].diff() if '20_SMA' in combined.columns else 0

    # Use your full detection logic here, as in your last provided code...
    # ... For brevity, the rest of this function is unchanged.
    # Paste your whole detect_signals_in_memory function here!

    # The only change: we do NOT use a persistent signal ID DB, just process per batch

    # [YOUR LOGIC GOES HERE: as previously posted. Detect signals, append dicts with Signal_ID/date/Entry Type etc.]

    return signals_detected

#############################
# Main historical batch loop
#############################
def all_trading_days(df):
    """Return sorted list of trading days present in a DataFrame (as date)."""
    days = df['date'].dt.date.unique()
    return sorted(days)

def run_batch_for_all_days():
    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No indicator files found in {INDICATORS_DIR}")
        return

    print(f"Processing {len(files)} tickers found in {INDICATORS_DIR}...")
    for file_path in tqdm(files, desc="Tickers"):
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()
        print(f"\n==== {ticker} ====")
        df = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df.empty:
            print(f"  [SKIP] No data for {ticker}")
            continue

        if 'Signal_ID' not in df.columns:
            df['Signal_ID'] = [f"{ticker}-{dt.isoformat()}" for dt in df['date']]
        df.drop_duplicates(subset='date', inplace=True)

        trading_days = all_trading_days(df)
        for day in tqdm(trading_days, leave=False, desc="  Dates"):
            start_dt = india_tz.localize(datetime.combine(day, dt_time(9, 15)))
            end_dt = india_tz.localize(datetime.combine(day, dt_time(15, 30)))
            df_day = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
            if df_day.empty:
                continue

            lookback = 30
            start_idx = max(0, df[df['date'] <= start_dt].shape[0] - lookback)
            df_rolling = df.iloc[start_idx:].copy()

            detected_signals = detect_signals_in_memory(
                ticker=ticker,
                df_for_rolling=df_rolling,
                df_for_detection=df_day,
                existing_signal_ids=set()
            )
            if not detected_signals:
                continue

            signals_df = pd.DataFrame(detected_signals)
            if signals_df.empty:
                continue
            signals_df.sort_values('date', inplace=True)
            signals_df = signals_df.groupby(['Entry Type', 'Trend Type'], as_index=False).first()

            out_csv = os.path.join(OUT_ENTRIES_DIR, f"{ticker}_{day}.csv")
            signals_df.to_csv(out_csv, index=False)
            print(f"  {ticker} {day}: {signals_df.shape[0]} signal(s) stored in {out_csv}")

if __name__ == "__main__":
    run_batch_for_all_days()
