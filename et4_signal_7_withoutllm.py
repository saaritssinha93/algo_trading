# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:05:10 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
This script processes a list of user-specified dates (e.g. "2025-03-03", "2025-03-04", etc.).
For each date, it:
  1. Loads each *_main_indicators_updated.csv from the main_indicators folder.
  2. Filters rows for that day between 09:30 and 15:30 (using India timezone).
  3. Uses ~30 previous bars for rolling calculations.
  4. Runs detection logic for 4 strategies:
       - Bullish Breakout
       - Bullish Pullback
       - Bearish Breakdown
       - Bearish Pullback
  5. Marks signals in the main CSV (sets CalcFlag = 'Yes').
  6. Appends the first signal per ticker to a papertrade_<YYYY-MM-DD>.csv file.
  
The continuous loop and 20-second sleep have been removed.
"""

import os
import logging
import sys
import glob
import json
import time
import pytz
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time, date
import pandas as pd
import numpy as np
from filelock import FileLock, Timeout
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import TimedRotatingFileHandler

###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
INDICATORS_DIR = "main_indicators"
SIGNALS_DB = "generated_signals_unified.json"

market_holidays = [
    datetime(2025, 1, 26).date(),
    datetime(2025, 2, 26).date(),
    datetime(2025, 3, 6).date(),
    datetime(2025, 4, 11).date(),
    datetime(2025, 8, 15).date(),
    datetime(2025, 10, 2).date(),
    datetime(2025, 10, 17).date(),
    datetime(2025, 11, 5).date(),
    datetime(2025, 12, 25).date(),
]

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signal_merged.log",
    when="M",
    interval=30,
    backupCount=5,
    delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging with TimedRotatingFileHandler.")
print("Script start...")

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

###########################################################
# 2) JSON-based Signal Tracking
###########################################################
def load_generated_signals():
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB JSON is corrupted. Starting with empty set.")
            return set()
    else:
        return set()

def save_generated_signals(generated_signals):
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f, indent=2)

###########################################################
# 3) Time Normalization & CSV Loading
###########################################################
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

###########################################################
# 4) Indicator Calculations
###########################################################
def calculate_vwap(df: pd.DataFrame, price_col='close', high_col='high', low_col='low', vol_col='volume') -> pd.DataFrame:
    df['TP'] = (df[high_col] + df[low_col] + df[price_col]) / 3.0
    df['cum_tp_vol'] = (df['TP'] * df[vol_col]).cumsum()
    df['cum_vol'] = df[vol_col].cumsum()
    df['VWAP'] = df['cum_tp_vol'] / df['cum_vol']
    return df

def calculate_vwap_bands(df: pd.DataFrame, vwap_col='VWAP', window=20, stdev_multiplier=2.0) -> pd.DataFrame:
    rolling_diff = (df['close'] - df[vwap_col]).rolling(window)
    df['vwap_std'] = rolling_diff.std()
    df['VWAP_UpperBand'] = df[vwap_col] + stdev_multiplier * df['vwap_std']
    df['VWAP_LowerBand'] = df[vwap_col] - stdev_multiplier * df['vwap_std']
    return df

def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
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

def is_ichimoku_bullish(df: pd.DataFrame, idx: int, tol: float = 0.001) -> bool:
    row = df.iloc[idx]
    if pd.isna(row['senkou_span_a']) or pd.isna(row['senkou_span_b']):
        return False
    lower_span = min(row['senkou_span_a'], row['senkou_span_b'])
    if row['close'] < lower_span * (1 - tol):
        return False
    if row['tenkan_sen'] < row['kijun_sen'] * (1 - tol):
        return False
    return True

def is_ichimoku_bearish(df: pd.DataFrame, idx: int, tol: float = 0.001) -> bool:
    row = df.iloc[idx]
    if pd.isna(row['senkou_span_a']) or pd.isna(row['senkou_span_b']):
        return False
    higher_span = max(row['senkou_span_a'], row['senkou_span_b'])
    if row['close'] > higher_span * (1 + tol):
        return False
    if row['tenkan_sen'] > row['kijun_sen'] * (1 + tol):
        return False
    return True

def calculate_ttm_squeeze(df: pd.DataFrame, bb_window: int = 20, kc_window: int = 20, kc_mult: float = 1.5, tol: float = 0.5) -> pd.DataFrame:
    df['mean_bb'] = df['close'].rolling(bb_window).mean()
    df['std_bb'] = df['close'].rolling(bb_window).std()
    df['upper_bb'] = df['mean_bb'] + 2 * df['std_bb']
    df['lower_bb'] = df['mean_bb'] - 2 * df['std_bb']

    df['ema_kc'] = df['close'].ewm(span=kc_window).mean()
    true_range = df['high'] - df['low']
    df['atr_kc'] = true_range.rolling(kc_window).mean()
    df['upper_kc'] = df['ema_kc'] + kc_mult * df['atr_kc']
    df['lower_kc'] = df['ema_kc'] - kc_mult * df['atr_kc']

    df['squeeze_on'] = ((df['lower_bb'] > df['lower_kc'] * (1 - tol)) &
                        (df['upper_bb'] < df['upper_kc'] * (1 + tol))).astype(int)
    df['squeeze_release'] = 0
    for i in range(1, len(df)):
        if df['squeeze_on'].iloc[i-1] == 1 and df['squeeze_on'].iloc[i] == 0:
            df.at[i, 'squeeze_release'] = 1
    return df

def is_squeeze_on(df: pd.DataFrame, idx: int) -> bool:
    if 0 <= idx < len(df):
        return df.at[idx, 'squeeze_on'] == 1
    return False

def is_squeeze_releasing(df: pd.DataFrame, idx: int) -> bool:
    if 0 <= idx < len(df):
        return df.at[idx, 'squeeze_release'] == 1
    return False

###########################################################
# 5) Price-Action & Consolidation Helpers
###########################################################
def is_bullish_breakout_consolidation(df, idx, consolidation_period=10, volume_multiplier=1.5,
                                      range_threshold=0.02, breakout_buffer=0.01, min_body_ratio=0.25):
    if idx < consolidation_period or idx >= len(df):
        return False
    cdata = df.iloc[idx - consolidation_period: idx]
    ch = cdata['high'].max()
    cl = cdata['low'].min()
    ac = cdata['close'].mean()
    if (ch - cl) / ac > range_threshold:
        return False
    row = df.iloc[idx]
    if row['close'] <= row['open']:
        return False
    rng = row['high'] - row['low']
    if rng <= 0:
        return False
    body = row['close'] - row['open']
    if body < min_body_ratio * rng:
        return False
    if row['close'] <= ch * (1 + breakout_buffer):
        return False
    if 'volume' in df.columns:
        av = cdata['volume'].mean()
        if row['volume'] < volume_multiplier * av:
            return False
    return True

def is_bearish_breakdown_consolidation(df, idx, consolidation_period=10, volume_multiplier=1.5,
                                       range_threshold=0.02, breakdown_buffer=0.01, min_body_ratio=0.25):
    if idx < consolidation_period or idx >= len(df):
        return False
    cdata = df.iloc[idx - consolidation_period: idx]
    ch = cdata['high'].max()
    cl = cdata['low'].min()
    ac = cdata['close'].mean()
    if (ch - cl) / ac > range_threshold:
        return False
    row = df.iloc[idx]
    if row['close'] >= row['open']:
        return False
    rng = row['high'] - row['low']
    if rng <= 0:
        return False
    body = row['open'] - row['close']
    if body < min_body_ratio * rng:
        return False
    if row['close'] >= cl * (1 - breakdown_buffer):
        return False
    if 'volume' in df.columns:
        av = cdata['volume'].mean()
        if row['volume'] < volume_multiplier * av:
            return False
    return True

###########################################################
# 6) Checks for Bullish / Bearish Patterns
###########################################################
def check_bullish_breakout(daily_change: float, adx_val: float, price: float, high_10: float,
                           vol: float, macd_val: float, roll_vol: float,
                           macd_above_signal_now: bool, stoch_val: float, stoch_diff: float,
                           slope_okay_for_bullish: bool, is_ich_bull: bool, vwap: float) -> bool:
    def valid_adx_for_breakout(adx):
        return (20 <= adx <= 45)
    return ((2.0 <= daily_change <= 10.0)
            and valid_adx_for_breakout(adx_val)
            and (price >= 0.997 * high_10)
            and (macd_val >= 4.0)
            and (vol >= 2.0 * roll_vol)
            and macd_above_signal_now
            and (stoch_diff > 0)
            and slope_okay_for_bullish
            and is_ich_bull
            and (price > vwap))

def check_bullish_pullback(daily_change: float, adx_val: float, macd_above_signal_now: bool,
                           stoch_val: float, stoch_diff: float, slope_okay_for_bullish: bool,
                           pullback_pct: float, vwap: float, price: float) -> bool:
    def valid_adx_for_pullback(adx):
        return (20 <= adx <= 45)
    return ((daily_change >= -0.5)
            and valid_adx_for_pullback(adx_val)
            and macd_above_signal_now
            and (stoch_diff > 0)
            and slope_okay_for_bullish
            and (pullback_pct >= 0.025)
            and (price > vwap))

def check_bearish_breakdown(daily_change: float, adx_val: float, price: float, low_10: float,
                            vol: float, roll_vol: float, macd_val: float,
                            macd_above_signal_now: bool, stoch_val: float, stoch_diff: float,
                            slope_okay_for_bearish: bool, is_ich_bear: bool, vwap: float) -> bool:
    def valid_adx_for_breakout(adx):
        return (20 <= adx <= 45)
    return ((-10.0 <= daily_change <= -2.0)
            and valid_adx_for_breakout(adx_val)
            and (price <= 1.003 * low_10)
            and (vol >= 2.0 * roll_vol)
            and (macd_val <= -4.0)
            and (not macd_above_signal_now)
            and (stoch_diff < 0)
            and slope_okay_for_bearish
            and is_ich_bear
            and (price < vwap))

def check_bearish_pullback(daily_change: float, adx_val: float, macd_above_signal_now: bool,
                           stoch_val: float, stoch_diff: float, slope_okay_for_bearish: bool,
                           bounce_pct: float, vwap: float, price: float) -> bool:
    def valid_adx_for_pullback(adx):
        return (20 <= adx <= 45)
    return ((daily_change <= 0.5)
            and valid_adx_for_pullback(adx_val)
            and (not macd_above_signal_now)
            and (stoch_diff < 0)
            and slope_okay_for_bearish
            and (bounce_pct >= 0.025)
            and (price < vwap))

def check_bullish_breakout_advanced(df, idx, volume_spike_mult=1.5):
    if idx < 1:
        return False
    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]
    required_cols = ['VWAP', 'VWAP_UpperBand', 'volume', 'open', 'close']
    for rc in required_cols:
        if rc not in row:
            return False
    if row['VWAP'] <= prev_row['VWAP']:
        return False
    if row['close'] < row['VWAP']:
        return False
    if row['close'] < 1.05 * row['VWAP']:
        return False
    if row['close'] <= row['open']:
        return False
    if pd.notna(row['VWAP_UpperBand']) and row['close'] < row['VWAP_UpperBand']:
        return False
    vol_window = 10
    if idx < vol_window:
        return False
    recent_avg_vol = df['volume'].iloc[idx - vol_window: idx].mean()
    if row['volume'] < volume_spike_mult * recent_avg_vol:
        return False
    return True

def check_bearish_breakdown_advanced(df, idx, volume_spike_mult=1.5):
    if idx < 1:
        return False
    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]
    required_cols = ['VWAP', 'VWAP_LowerBand', 'volume', 'open', 'close']
    for rc in required_cols:
        if rc not in row:
            return False
    if row['VWAP'] >= prev_row['VWAP']:
        return False
    if row['close'] > row['VWAP']:
        return False
    if row['close'] > 0.95 * row['VWAP']:
        return False
    if row['close'] >= row['open']:
        return False
    if pd.notna(row['VWAP_LowerBand']) and row['close'] > row['VWAP_LowerBand']:
        return False
    vol_window = 10
    if idx < vol_window:
        return False
    recent_avg_vol = df['volume'].iloc[idx - vol_window: idx].mean()
    if row['volume'] < volume_spike_mult * recent_avg_vol:
        return False
    return True

###########################################################
# 7) detect_signals_in_memory
###########################################################
def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set
) -> list:
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    if 'VWAP' not in combined.columns:
        combined = calculate_vwap(combined)
    combined = calculate_vwap_bands(combined, 'VWAP', window=20, stdev_multiplier=2.0)
    combined = calculate_ichimoku(combined)
    combined = calculate_ttm_squeeze(combined)

    rolling_window = 10
    combined['rolling_high_10'] = combined['high'].rolling(rolling_window, min_periods=rolling_window).max()
    combined['rolling_low_10']  = combined['low'].rolling(rolling_window, min_periods=rolling_window).min()
    combined['rolling_vol_10']  = combined['volume'].rolling(rolling_window, min_periods=rolling_window).mean()

    combined['StochDiff']   = combined['Stochastic'].diff()
    combined['SMA20_Slope'] = combined['20_SMA'].diff()

    def slope_okay_for_bullish(row) -> bool:
        s20 = row['SMA20_Slope']
        s20ma = row['20_SMA']
        if pd.isna(s20) or pd.isna(s20ma) or s20ma == 0:
            return False
        return (s20 > 0) and ((s20 / s20ma) >= 0.001)

    def slope_okay_for_bearish(row) -> bool:
        s20 = row['SMA20_Slope']
        s20ma = row['20_SMA']
        if pd.isna(s20) or pd.isna(s20ma) or s20ma == 0:
            return False
        return (s20 < 0) and (abs(s20) / s20ma >= 0.001)

    def is_macd_above(ix: int) -> bool:
        if 0 <= ix < len(combined):
            m = combined.loc[ix, 'MACD']
            s = combined.loc[ix, 'Signal_Line']
            if pd.isna(m) or pd.isna(s):
                return False
            return (m > s)
        return False

    def is_adx_increasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0:
            return False
        vals = df_.loc[si:cix, 'ADX'].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        for i in range(len(vals) - 1):
            if vals[i] >= vals[i + 1]:
                return False
        return True

    def is_rsi_increasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0:
            return False
        vals = df_.loc[si:cix, 'RSI'].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        for i in range(len(vals) - 1):
            if vals[i] >= vals[i + 1]:
                return False
        return True

    def is_rsi_decreasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0:
            return False
        vals = df_.loc[si:cix, 'RSI'].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        for i in range(len(vals) - 1):
            if vals[i] <= vals[i + 1]:
                return False
        return True

    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect['date']
        match = combined[combined['date'] == dt]
        if match.empty:
            continue
        cidx = match.index[0]
        signals_for_this_bar = []

        price      = combined.loc[cidx, 'close']
        vol        = combined.loc[cidx, 'volume']
        high_10    = combined.loc[cidx, 'rolling_high_10']
        low_10     = combined.loc[cidx, 'rolling_low_10']
        roll_vol   = combined.loc[cidx, 'rolling_vol_10']
        stoch_val  = combined.loc[cidx, 'Stochastic']
        stoch_diff = combined.loc[cidx, 'StochDiff']
        adx_val    = combined.loc[cidx, 'ADX']
        macd_val   = combined.loc[cidx, 'MACD']
        macd_above_signal_now = is_macd_above(cidx)
        slope_bull = slope_okay_for_bullish(combined.loc[cidx])
        slope_bear = slope_okay_for_bearish(combined.loc[cidx])
        ich_bull   = is_ichimoku_bullish(combined, cidx)
        ich_bear   = is_ichimoku_bearish(combined, cidx)
        vwap       = combined.loc[cidx, 'VWAP']

        daily_change = row_detect.get('Daily_Change', np.nan)
        if pd.isna(daily_change):
            daily_change = 0.0

        if cidx >= 5:
            recent_high_5 = combined.loc[cidx-5:cidx, 'high'].max()
            recent_low_5  = combined.loc[cidx-5:cidx, 'low'].min()
        else:
            recent_high_5 = combined.loc[:cidx, 'high'].max()
            recent_low_5  = combined.loc[:cidx, 'low'].max()

        pullback_pct = 0.0
        if recent_high_5 and recent_high_5 > 0:
            pullback_pct = (recent_high_5 - price) / recent_high_5
        bounce_pct = 0.0
        if recent_low_5 and recent_low_5 > 0:
            bounce_pct = (price - recent_low_5) / recent_low_5

        # ---- BULLISH BREAKOUT ----
        if (check_bullish_breakout(daily_change, adx_val, price, high_10, vol, macd_val, roll_vol,
                                   macd_above_signal_now, stoch_val, stoch_diff, slope_bull, ich_bull, vwap)
            and is_bullish_breakout_consolidation(combined, cidx)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and is_rsi_increasing(combined, cidx, bars_ago=1)
            and check_bullish_breakout_advanced(combined, cidx, volume_spike_mult=3.0)):
            signal_id = f"{ticker}-{dt.isoformat()}-BULL_BREAKOUT"
            if signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Breakout',
                    'Trend Type': 'Bullish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    'VWAP': round(vwap, 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # ---- BULLISH PULLBACK ----
        if check_bullish_pullback(daily_change, adx_val, macd_above_signal_now,
                                  stoch_val, stoch_diff, slope_bull, pullback_pct, vwap, price):
            signal_id = f"{ticker}-{dt.isoformat()}-BULL_PULLBACK"
            if signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Pullback',
                    'Trend Type': 'Bullish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    'VWAP': round(vwap, 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # ---- BEARISH BREAKDOWN ----
        if (check_bearish_breakdown(daily_change, adx_val, price, low_10, vol, roll_vol, macd_val,
                                    macd_above_signal_now, stoch_val, stoch_diff, slope_bear, ich_bear, vwap)
            and is_bearish_breakdown_consolidation(combined, cidx)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and is_rsi_decreasing(combined, cidx, bars_ago=1)
            and check_bearish_breakdown_advanced(combined, cidx, volume_spike_mult=3.0)):
            signal_id = f"{ticker}-{dt.isoformat()}-BEAR_BREAKDOWN"
            if signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Breakdown',
                    'Trend Type': 'Bearish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    'VWAP': round(vwap, 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # ---- BEARISH PULLBACK ----
        if check_bearish_pullback(daily_change, adx_val, macd_above_signal_now,
                                  stoch_val, stoch_diff, slope_bear, bounce_pct, vwap, price):
            signal_id = f"{ticker}-{dt.isoformat()}-BEAR_PULLBACK"
            if signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Pullback',
                    'Trend Type': 'Bearish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    'VWAP': round(vwap, 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        for sig in signals_for_this_bar:
            existing_signal_ids.add(sig["Signal_ID"])
            signals_detected.append(sig)

    return signals_detected

###########################################################
# 8) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return
    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)
    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, expected_cols=["Signal_ID"], tz='Asia/Kolkata')
            if df_main.empty:
                logging.warning(f"[{ticker}] => main_indicators empty. Skipping.")
                return
            required_cols = ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]
            for col in required_cols:
                if col not in df_main.columns:
                    df_main[col] = ""
            signal_ids = [sig["Signal_ID"] for sig in signals_list if "Signal_ID" in sig]
            if not signal_ids:
                logging.info(f"[{ticker}] => No valid Signal_IDs to update.")
                return
            mask = df_main["Signal_ID"].isin(signal_ids)
            rows_to_update = df_main[mask]
            if not rows_to_update.empty:
                df_main.loc[mask, ["Entry Signal", "CalcFlag"]] = "Yes"
                current_time_iso = datetime.now(tz_obj).isoformat()
                df_main.loc[mask & (df_main["logtime"] == ""), "logtime"] = current_time_iso
                df_main.sort_values("date", inplace=True)
                df_main.to_csv(main_path, index=False)
                updated_count = mask.sum()
                logging.info(f"[{ticker}] => Set CalcFlag='Yes' for {updated_count} row(s).")
            else:
                logging.info(f"[{ticker}] => No matching Signal_IDs found to update.")
    except Timeout:
        logging.error(f"[{ticker}] => FileLock timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] => Error marking signals: {e}")

###########################################################
# 9) Append signals to papertrade, *only first* per Ticker
###########################################################
def append_signals_to_papertrade(signals_list, output_file, trading_day):
    if not signals_list:
        return
    df_entries = pd.DataFrame(signals_list)
    if df_entries.empty:
        return
    df_entries = normalize_time(df_entries, tz='Asia/Kolkata')
    df_entries.sort_values('date', inplace=True)
    start_dt = india_tz.localize(datetime.combine(trading_day, dt_time(0, 0)))
    end_dt   = india_tz.localize(datetime.combine(trading_day, dt_time(23, 59)))
    mask = (df_entries['date'] >= start_dt) & (df_entries['date'] <= end_dt)
    df_entries = df_entries.loc[mask].copy()
    if df_entries.empty:
        return
    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries.columns:
            df_entries[col] = ""
    for idx, row in df_entries.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price > 0:
            target = price * 1.01 if trend == "Bullish" else price * 0.99 if trend == "Bearish" else price
            qty = int(20000 / price) if price != 0 else 1
            total_val = qty * price
            df_entries.at[idx, "Target Price"] = round(target, 2)
            df_entries.at[idx, "Quantity"]     = qty
            df_entries.at[idx, "Total value"]  = round(total_val, 2)
    if 'time' not in df_entries.columns:
        df_entries['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries['time'].isna()) | (df_entries['time'] == "")
        if blank_mask.any():
            df_entries.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            existing = normalize_time(existing, tz='Asia/Kolkata')
            existing_key_set = set()
            for _, erow in existing.iterrows():
                existing_key_set.add((erow['Ticker'], erow['date'].date()))
            new_rows = []
            for _, row in df_entries.iterrows():
                key = (row['Ticker'], row['date'].date())
                if key not in existing_key_set:
                    new_rows.append(row)
            if not new_rows:
                return
            new_df = pd.DataFrame(new_rows, columns=df_entries.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['Ticker','date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")
        else:
            df_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
            df_entries.sort_values('date', inplace=True)
            df_entries.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries)} rows.")

###########################################################
# 10) find_price_action_entries_for_day
###########################################################
def find_price_action_entries_for_day(trading_day: date, papertrade_file: str):
    start_dt = india_tz.localize(datetime.combine(trading_day, dt_time(9, 30)))
    end_dt   = india_tz.localize(datetime.combine(trading_day, dt_time(15, 30)))
    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators_updated.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()
    existing_ids = load_generated_signals()
    all_signals = []
    summary_info = {}
    lock = threading.Lock()

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace("_main_indicators_updated.csv", "").upper()
        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df_full.empty:
            with lock:
                summary_info[ticker] = "No data"
            print(f"[ANALYZED] {ticker} => used 0 rows for {trading_day}, reason: No data")
            return []
        day_mask = (df_full['date'].dt.date == trading_day)
        df_day = df_full[day_mask].copy()
        if df_day.empty:
            reason = f"No bars found for {trading_day}"
            with lock:
                summary_info[ticker] = reason
            print(f"[ANALYZED] {ticker} => used 0 rows for {trading_day}, reason: {reason}")
            return []
        df_day = df_day[(df_day['date'] >= start_dt) & (df_day['date'] <= end_dt)]
        total_count = len(df_day)
        if df_day.empty:
            reason = "No rows between 9:30 and 15:30"
            with lock:
                summary_info[ticker] = reason
            print(f"[ANALYZED] {ticker} => used 0 of {total_count} rows, reason: {reason}")
            return []
        first_bar_dt = df_day['date'].min()
        df_for_rolling = df_full[df_full['date'] < first_bar_dt].tail(30)
        df_for_detection = df_day.copy()
        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_for_rolling,
            df_for_detection=df_for_detection,
            existing_signal_ids=existing_ids
        )
        if not new_signals:
            reason = "No new signals"
            with lock:
                summary_info[ticker] = reason
            print(f"[ANALYZED] {ticker} => used {total_count} rows for {trading_day}, reason: {reason}")
            return []
        reason = f"{len(new_signals)} new signals"
        with lock:
            summary_info[ticker] = reason
        mark_signals_in_main_csv(ticker, new_signals, file_path, india_tz)
        append_signals_to_papertrade(new_signals, papertrade_file, trading_day)
        print(f"[ANALYZED] {ticker} => used {total_count} rows for {trading_day}, reason: {reason}")
        return new_signals

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_path = {executor.submit(process_file, f): f for f in files}
        for fut in tqdm(as_completed(future_to_path), total=len(future_to_path),
                        desc=f"Processing {trading_day} CSVs"):
            fpath = future_to_path[fut]
            try:
                signals_result = fut.result()
                if signals_result:
                    all_signals.extend(signals_result)
            except Exception as e:
                logging.error(f"Error with {fpath}: {e}")

    save_generated_signals(existing_ids)
    print(f"\n=== SUMMARY for {trading_day} ===")
    for t, stat in summary_info.items():
        print(f"{t}: {stat}")
    if not all_signals:
        logging.info(f"No new signals overall for {trading_day}.")
        return pd.DataFrame()
    df_all = pd.DataFrame(all_signals).sort_values('date').reset_index(drop=True)
    return df_all

###########################################################
# 11) main() => Process Multiple User-Specified Dates
###########################################################
def main():
    try:
        dates_to_process = ["2025-03-03"]  # Modify this list as needed.
        for d_str in dates_to_process:
            target_day = datetime.strptime(d_str, "%Y-%m-%d").date()
            papertrade_file = f"papertrade_{d_str}.csv"
            print(f"\n===== Processing {d_str} =====")
            df_signals = find_price_action_entries_for_day(target_day, papertrade_file)
            if df_signals.empty:
                logging.info(f"No signals found for {d_str}.")
            else:
                logging.info(f"Detected {len(df_signals)} signals for {d_str}.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
