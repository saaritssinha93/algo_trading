# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 06:31:56 2025

@author: Saarit

Script now reads *all bars after 9:30 AM* from each *_main_indicators_updated.csv
(similar to Code 2), detects signals, marks CalcFlag='Yes', and immediately
updates a papertrade CSV, while still computing rolling(10) from older bars for
indicators like ADX, RSI, Ichimoku, etc.
Uses robust_prev_day_close_for_ticker to compute daily_change similar to Code 1.
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
import signal
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from filelock import FileLock, Timeout
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import TimedRotatingFileHandler

###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
INDICATORS_DIR = "main_indicators"
SIGNALS_DB = "generated_signals4.json"
market_holidays = []  # If needed, populate

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signal6.log", when="M", interval=30, backupCount=5, delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

# Prevent multiple handlers if script is reloaded
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
    """Load any previously saved signal IDs from the JSON file."""
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
    """Save updated signal IDs to the JSON file."""
    with open(SIGNALS_DB, 'w') as f:
        json.dump(list(generated_signals), f, indent=2)

###########################################################
# 3) Time Normalization & CSV Loading
###########################################################
def normalize_time(df, tz='Asia/Kolkata'):
    """Convert 'date' column to DateTime in a consistent timezone."""
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    if df['date'].dt.tz is None:
        # If no timezone, assume UTC
        df['date'] = df['date'].dt.tz_localize('UTC')

    df['date'] = df['date'].dt.tz_convert(tz)
    return df

def load_and_normalize_csv(file_path, expected_cols=None, tz='Asia/Kolkata'):
    """Load CSV and normalize 'date' column to Asia/Kolkata."""
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
        if expected_cols:
            df = df[expected_cols]
    return df

###########################################################
# 4) last_trading_day logic
###########################################################
def get_last_trading_day(ref_date=None):
    """Returns the most recent trading day on/before ref_date."""
    if ref_date is None:
        ref_date = datetime.now(india_tz).date()
    if ref_date.weekday() < 5 and ref_date not in market_holidays:
        return ref_date
    else:
        d = ref_date - timedelta(days=1)
        while d.weekday() >= 5 or d in market_holidays:
            d -= timedelta(days=1)
        return d

###########################################################
# 5) detect_signals_in_memory
###########################################################
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def robust_prev_day_close_for_ticker(ticker, date_of_interest):
    """
    Load entire CSV for 'ticker', find the previous day's final bar, return that close.
    This is used to compute daily_change EXACTLY like in Code 1.
    """
    main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker.lower()}_main_indicators_updated.csv")
    if not os.path.exists(main_csv_path):
        return None

    try:
        df_full = load_and_normalize_csv(main_csv_path, tz='Asia/Kolkata')
        if df_full.empty:
            return None

        day_date = date_of_interest.date()
        df_before = df_full[df_full['date'].dt.date < day_date]
        if df_before.empty:
            return None
        last_close = df_before.iloc[-1]['close']
        return float(last_close)
    except Exception as e:
        logging.error(f"{ticker} => robust_prev_day_close_for_ticker failed: {e}")
        return None

##############################################################################
# 1) VWAP & VWAP BANDS
##############################################################################

import numpy as np
from datetime import datetime

def calculate_vwap(df: pd.DataFrame, 
                   price_col='close', 
                   high_col='high', 
                   low_col='low', 
                   vol_col='volume') -> pd.DataFrame:
    """
    Computes VWAP from the DataFrame if you don't already have it in your CSV.
    """
    df['TP'] = (df[high_col] + df[low_col] + df[price_col]) / 3.0
    df['cum_tp_vol'] = (df['TP'] * df[vol_col]).cumsum()
    df['cum_vol'] = df[vol_col].cumsum()
    df['VWAP'] = df['cum_tp_vol'] / df['cum_vol']
    return df

def calculate_vwap_bands(df: pd.DataFrame, 
                         vwap_col='VWAP', 
                         window=20, 
                         stdev_multiplier=2.0) -> pd.DataFrame:
    """
    Adds standard-deviation bands around VWAP: VWAP ± stdev_multiplier * rolling_std.
    """
    rolling_diff = (df['close'] - df[vwap_col]).rolling(window)
    df['vwap_std'] = rolling_diff.std()
    df['VWAP_UpperBand'] = df[vwap_col] + stdev_multiplier * df['vwap_std']
    df['VWAP_LowerBand'] = df[vwap_col] - stdev_multiplier * df['vwap_std']
    return df

##############################################################################
# 2) ICHIMOKU & TTM SQUEEZE (UNMODIFIED FROM YOUR ORIGINAL)
##############################################################################

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

def calculate_ttm_squeeze(df: pd.DataFrame,
                          bb_window: int = 20,
                          kc_window: int = 20,
                          kc_mult: float = 1.5,
                          tol: float = 0.5) -> pd.DataFrame:
    df['mean_bb'] = df['close'].rolling(bb_window).mean()
    df['std_bb'] = df['close'].rolling(bb_window).std()
    df['upper_bb'] = df['mean_bb'] + 2 * df['std_bb']
    df['lower_bb'] = df['mean_bb'] - 2 * df['std_bb']
    df['ema_kc'] = df['close'].ewm(span=kc_window).mean()
    true_range = df['high'] - df['low']
    df['atr_kc'] = true_range.rolling(kc_window).mean()
    df['upper_kc'] = df['ema_kc'] + kc_mult * df['atr_kc']
    df['lower_kc'] = df['ema_kc'] - kc_mult * df['atr_kc']
    df['squeeze_on'] = (
        (df['lower_bb'] > df['lower_kc'] * (1 - tol)) &
        (df['upper_bb'] < df['upper_kc'] * (1 + tol))
    ).astype(int)
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

##############################################################################
# 3) ADVANCED CHECKS: EXAMPLE Bullish / Bearish Breakout w/ VWAP Bands & Volume
##############################################################################

def check_bullish_breakout_advanced(df, idx, volume_spike_mult=1.5):
    """
    More complex (but not overly strict) advanced bullish breakout logic:

      1) VWAP sloping up (current VWAP > previous VWAP).
      2) Price above VWAP.
      3) Close > 1.05 * VWAP (optional: at least 5% above VWAP).
      4) Candle is bullish (close > open).
      5) Price near or above VWAP_UpperBand (optional).
      6) Volume spike vs. last 10 bars (multiplier ~1.5 or 2.0).
    """
    if idx < 1:
        return False

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    # Ensure columns exist
    required_cols = ['VWAP', 'VWAP_UpperBand', 'volume', 'open', 'close']
    if any(col not in row for col in required_cols):
        return False

    # 1) VWAP sloping up
    if row['VWAP'] <= prev_row['VWAP']:
        return False

    # 2) Price above VWAP
    if row['close'] < row['VWAP']:
        return False

    # 3) Price >= 1.05 * VWAP (optional "5% above VWAP" threshold)
    #    You can lower this to 1.01, or remove if you want more signals.
    if row['close'] < 1.05 * row['VWAP']:
        return False

    # 4) Candle is bullish
    if row['close'] <= row['open']:
        return False

    # 5) Optional: Price above the VWAP Upper Band
    #    (If the upper band is NaN, we skip this check).
    if pd.notna(row['VWAP_UpperBand']) and (row['close'] < row['VWAP_UpperBand']):
        return False

    # 6) Volume spike: current volume > volume_spike_mult × average of last 10 bars
    vol_window = 10
    if idx < vol_window:
        return False
    recent_avg_vol = df['volume'].iloc[idx - vol_window : idx].mean()
    if row['volume'] < volume_spike_mult * recent_avg_vol:
        return False

    return True


def check_bearish_breakdown_advanced(df, idx, volume_spike_mult=1.5):
    """
    More complex (but not overly strict) advanced bearish breakdown logic:

      1) VWAP sloping down (current VWAP < previous VWAP).
      2) Price below VWAP.
      3) Close <= 0.95 * VWAP (optional: at least 5% below VWAP).
      4) Candle is bearish (close < open).
      5) Price near or below VWAP_LowerBand (optional).
      6) Volume spike vs. last 10 bars (multiplier ~1.5 or 2.0).
    """
    if idx < 1:
        return False

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    # Ensure columns exist
    required_cols = ['VWAP', 'VWAP_LowerBand', 'volume', 'open', 'close']
    if any(col not in row for col in required_cols):
        return False

    # 1) VWAP sloping down
    if row['VWAP'] >= prev_row['VWAP']:
        return False

    # 2) Price below VWAP
    if row['close'] > row['VWAP']:
        return False

    # 3) Price <= 0.95 * VWAP (optional "5% below VWAP" threshold)
    if row['close'] > 0.95 * row['VWAP']:
        return False

    # 4) Candle is bearish
    if row['close'] >= row['open']:
        return False

    # 5) Optional: Price below the VWAP Lower Band
    if pd.notna(row['VWAP_LowerBand']) and (row['close'] > row['VWAP_LowerBand']):
        return False

    # 6) Volume spike: current volume > volume_spike_mult × average of last 10 bars
    vol_window = 10
    if idx < vol_window:
        return False
    recent_avg_vol = df['volume'].iloc[idx - vol_window : idx].mean()
    if row['volume'] < volume_spike_mult * recent_avg_vol:
        return False

    return True


def check_bullish_breakout(
    daily_change: float,
    adx_val: float,
    price: float,
    high_10: float,
    vol: float,
    macd_val: float,
    roll_vol: float,
    macd_above_signal_now: bool,
    stoch_val: float,
    stoch_diff: float,
    slope_okay_for_bullish: bool,
    is_ich_bull: bool,
    vwap: float,
    # extra param for advanced logic if needed
) -> bool:
    def valid_adx_for_breakout(adx):
        return (29 <= adx <= 45)
    return (
        (2.0 <= daily_change <= 10.0)
        and valid_adx_for_breakout(adx_val)
        and (price >= 0.997 * high_10)
        and (macd_val >= 4.0)
        and (vol >= 3.0 * roll_vol)
        and macd_above_signal_now
        #and (stoch_val > 25)
        and (stoch_diff > 0)
        and slope_okay_for_bullish
        and is_ich_bull
        and (price > vwap)
    )

def check_bullish_pullback(
    daily_change: float,
    adx_val: float,
    macd_above_signal_now: bool,
    stoch_val: float,
    stoch_diff: float,
    slope_okay_for_bullish: bool,
    pullback_pct: float,
    vwap: float,
    price: float
) -> bool:
    def valid_adx_for_pullback(adx):
        return (29 <= adx <= 45)
    return (
        (daily_change >= -0.5)
        and valid_adx_for_pullback(adx_val)
        and macd_above_signal_now
        #and (stoch_val < 60)
        and (stoch_diff > 0)
        and slope_okay_for_bullish
        and (pullback_pct >= 0.025)
        and (pullback_pct != np.nan)
        and (vwap != np.nan)
        and (price > vwap)
    )

def check_bearish_breakdown(
    daily_change: float,
    adx_val: float,
    price: float,
    low_10: float,
    vol: float,
    roll_vol: float,
    macd_val: float,
    macd_above_signal_now: bool,
    stoch_val: float,
    stoch_diff: float,
    slope_okay_for_bearish: bool,
    is_ich_bear: bool,
    vwap: float
) -> bool:
    def valid_adx_for_breakout(adx):
        return (29 <= adx <= 45)
    return (
        (-10.0 <= daily_change <= -2.0)
        and valid_adx_for_breakout(adx_val)
        and (price <= 1.003 * low_10)
        and (vol >= 3.0 * roll_vol)
        and (macd_val <= -4.0)
        and (not macd_above_signal_now)
        #and (stoch_val < 75)
        and (stoch_diff < 0)
        and slope_okay_for_bearish
        and is_ich_bear
        and (price < vwap)
    )

def check_bearish_pullback(
    daily_change: float,
    adx_val: float,
    macd_above_signal_now: bool,
    stoch_val: float,
    stoch_diff: float,
    slope_okay_for_bearish: bool,
    bounce_pct: float,
    vwap: float,
    price: float
) -> bool:
    def valid_adx_for_pullback(adx):
        return (29 <= adx <= 45)
    return (
        (daily_change <= 0.5)
        and valid_adx_for_pullback(adx_val)
        and (not macd_above_signal_now)
        #and (stoch_val > 40)
        and (stoch_diff < 0)
        and slope_okay_for_bearish
        and (bounce_pct >= 0.025)
        and (price < vwap)
    )

##############################################################################
# 5) THE MAIN DETECT-SIGNALS FUNCTION (ADJUSTED TO USE VWAP & BANDS)
##############################################################################

##############################################################################
# 5) THE MAIN DETECT-SIGNALS FUNCTION
##############################################################################
def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set,
    prev_day_close: float = None
) -> list:
    """
    EXACT same detection function, but we ensure we pass in 'prev_day_close'
    so 'daily_change' is the same as Code 1.
    """
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    final_bar_date = df_for_detection['date'].iloc[-1]
    # If 'prev_day_close' is not provided, get from robust_prev_day_close_for_ticker
    if not prev_day_close:
        prev_day_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

    # Merge frames for rolling calculations
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    if 'VWAP' not in combined.columns:
        combined = calculate_vwap(combined)

    combined = calculate_vwap_bands(combined, 'VWAP', window=20, stdev_multiplier=2.0)

    # Rolling stats
    rolling_window = 10
    combined['rolling_high_10'] = (
        combined['high'].rolling(rolling_window, min_periods=rolling_window).max()
    )
    combined['rolling_low_10'] = (
        combined['low'].rolling(rolling_window, min_periods=rolling_window).min()
    )
    combined['rolling_vol_10'] = (
        combined['volume'].rolling(rolling_window, min_periods=rolling_window).mean()
    )

    # Additional computations
    combined['StochDiff'] = combined['Stochastic'].diff()
    combined['SMA20_Slope'] = combined['20_SMA'].diff()
    combined = calculate_ichimoku(combined)
    combined = calculate_ttm_squeeze(combined)


    # Slope checks
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
        if si < 0: return False
        vals = df_.loc[si:cix, 'ADX'].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        for i in range(len(vals) - 1):
            if vals[i] >= vals[i + 1]:
                return False
        return True

    def is_rsi_increasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, 'RSI'].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        for i in range(len(vals) - 1):
            if vals[i] >= vals[i + 1]:
                return False
        return True

    def is_rsi_decreasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, 'RSI'].to_numpy()
        if any(pd.isna(v) for v in vals):
            return False
        for i in range(len(vals) - 1):
            if vals[i] <= vals[i + 1]:
                return False
        return True

    # Compute daily_change
    try:
        latest_close = df_for_detection['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except:
        daily_change = 0.0

    # 2) Iterate over each bar in df_for_detection
    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect['date']
        match = combined[combined['date'] == dt]
        if match.empty:
            continue

        cidx = match.index[0]
        signals_for_this_bar = []

        price = combined.loc[cidx, 'close']
        vol = combined.loc[cidx, 'volume']
        high_10 = combined.loc[cidx, 'rolling_high_10']
        low_10 = combined.loc[cidx, 'rolling_low_10']
        roll_vol = combined.loc[cidx, 'rolling_vol_10']
        stoch_val = combined.loc[cidx, 'Stochastic']
        stoch_diff = combined.loc[cidx, 'StochDiff']
        adx_val = combined.loc[cidx, 'ADX']
        macd_val = combined.loc[cidx, 'MACD']
        macd_above_signal_now = is_macd_above(cidx)
        slope_bull = slope_okay_for_bullish(combined.loc[cidx])
        slope_bear = slope_okay_for_bearish(combined.loc[cidx])
        ich_bull = is_ichimoku_bullish(combined, cidx)
        ich_bear = is_ichimoku_bearish(combined, cidx)
        vwap = combined.loc[cidx, 'VWAP']

        # Price pullback & bounce calculations
        if cidx >= 5:
            recent_high_5 = combined.loc[cidx-5:cidx, 'high'].max()
            recent_low_5 = combined.loc[cidx-5:cidx, 'low'].min()
        else:
            recent_high_5 = combined.loc[:cidx, 'high'].max()
            recent_low_5 = combined.loc[:cidx, 'low'].min()

        pullback_pct = 0.0
        if recent_high_5 and recent_high_5 > 0:
            pullback_pct = (recent_high_5 - price) / recent_high_5

        bounce_pct = 0.0
        if recent_low_5 and recent_low_5 > 0:
            bounce_pct = (price - recent_low_5) / recent_low_5

        # --- 1) Bullish Breakout ---
        # Instead of elif, use if so we can get multiple signals
        if (
            check_bullish_breakout(
                daily_change=daily_change,
                adx_val=adx_val,
                price=price,
                high_10=high_10,
                vol=vol,
                roll_vol=roll_vol,
                macd_val=macd_val,
                macd_above_signal_now=macd_above_signal_now,
                stoch_val=stoch_val,
                stoch_diff=stoch_diff,
                slope_okay_for_bullish=slope_bull,
                is_ich_bull=ich_bull,
                vwap=vwap,
                #sq_on=sq_on
            )
            and is_bullish_breakout_consolidation(df=combined, idx=cidx)
            # NEW: *increasing* ADX check:
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and is_rsi_increasing(combined, cidx, bars_ago=1)
            # NEW: OBV and ROC check for bullish trend:
            #and is_obv_increasing(combined, cidx, bars_ago=2)
            #and is_roc_positive(combined, cidx)
            and check_bullish_breakout_advanced(combined, cidx, volume_spike_mult=3.0)
        ):
            # Create a unique ID for the "bullish breakout" signal
            this_signal_id = f"{ticker}-{dt.isoformat()}-BULL_BREAKOUT"
            if this_signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Breakout',
                    'Trend Type': 'Bullish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': this_signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # --- 2) Bullish Pullback ---
        if (
            check_bullish_pullback(
                daily_change=daily_change,
                adx_val=adx_val,
                macd_above_signal_now=macd_above_signal_now,
                stoch_val=stoch_val,
                stoch_diff=stoch_diff,
                slope_okay_for_bullish=slope_bull,
                pullback_pct=pullback_pct,
                price=price,
                vwap=vwap,
            )
            #and is_bullish_pullback_price_action(df=combined, idx=cidx, daily_change=daily_change)
        ):
            this_signal_id = f"{ticker}-{dt.isoformat()}-BULL_PULLBACK"
            if this_signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Pullback',
                    'Trend Type': 'Bullish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': this_signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # --- 3) Bearish Breakdown ---
        if (
            check_bearish_breakdown(
                daily_change=daily_change,
                adx_val=adx_val,
                price=price,
                low_10=low_10,
                vol=vol,
                roll_vol=roll_vol,
                macd_val=macd_val,
                macd_above_signal_now=macd_above_signal_now,
                stoch_val=stoch_val,
                stoch_diff=stoch_diff,
                slope_okay_for_bearish=slope_bear,
                is_ich_bear=ich_bear,
                #sq_release=sq_release
                vwap=vwap,
            )
            and is_bearish_breakdown_consolidation(df=combined, idx=cidx)
            # NEW: *increasing* ADX check:
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and is_rsi_decreasing(combined, cidx, bars_ago=1)
            # NEW: OBV and ROC check for bearish trend:
            #and is_obv_decreasing(combined, cidx, bars_ago=2)
            #and is_roc_negative(combined, cidx)
            and check_bearish_breakdown_advanced(combined, cidx, volume_spike_mult=3.0)
        ):
            this_signal_id = f"{ticker}-{dt.isoformat()}-BEAR_BREAKDOWN"
            if this_signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Breakdown',
                    'Trend Type': 'Bearish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': this_signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # --- 4) Bearish Pullback ---
        if (
            check_bearish_pullback(
                daily_change=daily_change,
                adx_val=adx_val,
                macd_above_signal_now=macd_above_signal_now,
                stoch_val=stoch_val,
                stoch_diff=stoch_diff,
                slope_okay_for_bearish=slope_bear,
                bounce_pct=bounce_pct,
                price=price,
                vwap=vwap,
            )
            #and is_bearish_pullback_price_action(df=combined, idx=cidx, daily_change=daily_change)
        ):
            this_signal_id = f"{ticker}-{dt.isoformat()}-BEAR_PULLBACK"
            if this_signal_id not in existing_signal_ids:
                signals_for_this_bar.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Pullback',
                    'Trend Type': 'Bearish',
                    'Price': round(price, 2),
                    'Daily Change %': round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    'ADX': round(adx_val, 2) if adx_val else None,
                    'MACD': round(macd_val, 2) if macd_val else None,
                    'Signal_ID': this_signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })

        # Finally, add all signals_for_this_bar to signals_detected
        for sig in signals_for_this_bar:
            # Mark as used so we don’t add it in future loops
            existing_signal_ids.add(sig["Signal_ID"])
            signals_detected.append(sig)

    return signals_detected

##############################################################################
# 6) Price-Action & Consolidation Helpers (Same As Your Original)
##############################################################################

def is_bullish_breakout_consolidation(df, idx, 
                                      consolidation_period=10, 
                                      volume_multiplier=1.5, 
                                      range_threshold=0.02, 
                                      breakout_buffer=0.01, 
                                      min_body_ratio=0.25):
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

def is_bearish_breakdown_consolidation(df, idx, 
                                       consolidation_period=10, 
                                       volume_multiplier=1.5, 
                                       range_threshold=0.02, 
                                       breakdown_buffer=0.01, 
                                       min_body_ratio=0.25):
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
# 6) Mark signals in main CSV => CalcFlag='Yes'
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
# 7) Immediate append to papertrade for each ticker
###########################################################
def append_signals_to_papertrade(signals_list, output_file, trading_day):
    if not signals_list:
        return

    df_entries = pd.DataFrame(signals_list)
    if df_entries.empty:
        return

    df_entries = normalize_time(df_entries, tz='Asia/Kolkata')
    df_entries.sort_values('date', inplace=True)

    start_dt = india_tz.localize(datetime.combine(trading_day, datetime.min.time()))
    end_dt = india_tz.localize(datetime.combine(trading_day, datetime.max.time()))
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
            if trend == "Bullish":
                target = price * 1.01
            elif trend == "Bearish":
                target = price * 0.99
            else:
                target = price
            qty = int(20000 / price) if price != 0 else 1
            total_val = qty * price

            df_entries.at[idx, "Target Price"] = round(target, 2)
            df_entries.at[idx, "Quantity"] = qty
            df_entries.at[idx, "Total value"] = round(total_val, 2)

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
# 8) find_price_action_entries_for_today
###########################################################
def find_price_action_entries_for_today(trading_day, papertrade_file):
    """
    We load the entire day's data from 9:30 onward and pass
    that ENTIRE subset to detect_signals_in_memory, just like Code 2 does,
    BUT we also compute 'prev_day_close' so daily_change is correct.
    """
    start_dt = india_tz.localize(datetime.combine(trading_day, dt_time(9, 30)))
    now_ist = datetime.now(india_tz)
    if now_ist.date() == trading_day:
        end_dt = now_ist
    else:
        end_dt = india_tz.localize(datetime.combine(trading_day, dt_time(14, 30)))

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators_updated.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    analysis_info = {}
    lock = threading.Lock()

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace("_main_indicators_updated.csv", "").upper()
        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")

        if df_full.empty:
            with lock:
                analysis_info[ticker] = {'rows_after_930': 0, 'rows_used': 0, 'reason': "File empty"}
            return []

        # 1) Filter only rows from 9:30 to end_dt => "df_today"
        df_today = df_full[(df_full["date"] >= start_dt) & (df_full["date"] <= end_dt)].copy()
        rows_after_930 = len(df_today)
        if df_today.empty:
            with lock:
                analysis_info[ticker] = {'rows_after_930': 0, 'rows_used': 0, 'reason': "No data after 9:30"}
            return []

        # 2) older data for rolling calculations
        df_for_rolling = df_full.tail(30).copy()

        # 3) Use entire df_today for detection
        df_for_detection = df_today.copy()
        rows_used = len(df_for_detection)

        # 4) Determine prev_day_close from the final bar in df_today
        final_bar_date = df_for_detection['date'].iloc[-1]
        prev_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

        # --- ADD LOGGING HERE ---
        if prev_close is None:
            logging.warning(
                f"[{ticker}] => No previous day close found for {final_bar_date.date()}, "
                "daily_change will default to 0.0."
            )
        else:
            logging.info(
                f"[{ticker}] => Found previous day close: {prev_close:.2f} "
                f"for {final_bar_date.date()}"
            )

        # 5) Detect signals
        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_for_rolling,
            df_for_detection=df_for_detection,
            existing_signal_ids=existing_ids,
            prev_day_close=prev_close  # pass the same way as Code 1
        )

        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])

            # Optionally mark signals in main CSV
            # mark_signals_in_main_csv(ticker, new_signals, file_path, india_tz)

            # Immediately append to papertrade
            append_signals_to_papertrade(new_signals, papertrade_file, trading_day)

        with lock:
            analysis_info[ticker] = {
                'rows_after_930': rows_after_930,
                'rows_used': rows_used,
                'reason': "" if new_signals else "No new signals"
            }

        return new_signals

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing CSVs"):
            file = future_to_file[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    save_generated_signals(existing_ids)

    print("\n=== ANALYSIS SUMMARY ===")
    for t in sorted(analysis_info.keys()):
        info = analysis_info[t]
        if info['rows_used'] == 0:
            print(f"[NOT ANALYZED] {t} => reason: {info['reason']}")
        else:
            print(f"[ANALYZED] {t} => used last {info['rows_used']} of {info['rows_after_930']} rows after 9:30, reason: {info['reason']}")

    if not all_signals:
        logging.info("No new signals detected.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values("date", inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    logging.info(f"Total new signals detected: {len(signals_df)}")
    return signals_df


###########################################################
# 9) main()
###########################################################
def main():
    """Runs the signal detection & immediate papertrade updates once."""
    try:
        last_trading_day = get_last_trading_day()
        today_str = last_trading_day.strftime('%Y-%m-%d')
        papertrade_file = f"papertrade3_{today_str}.csv"

        raw_signals_df = find_price_action_entries_for_today(last_trading_day, papertrade_file)
        if raw_signals_df.empty:
            logging.info("No signals found this cycle.")
            return

        logging.info("All signals processed & appended immediately.")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.error(traceback.format_exc())

###########################################################
# 10) Continuous run every 20 seconds
###########################################################
def signal_handler(sig, frame):
    print("\nInterrupt received, shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_continuous_every_20s():
    while True:
        try:
            main()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            logging.error(traceback.format_exc())
        time.sleep(20)

if __name__ == "__main__":
    run_continuous_every_20s()