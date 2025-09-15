# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:02 2025

@author: Saarit

Solution B (modified for multiple user-specified back-dates):
- Keep original detection data for papertrade CSV, only check CalcFlag='Yes' from main CSV.
- daily_change uses previous day's close as the reference.
- Only the *first* signal per ticker goes into papertrade CSV each day.
- 'time' column is the actual detection time.
- Allows you to run for a list of chosen dates (past or present) in one go.
"""

import os
import logging
import sys
import glob
import json
import pytz
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from filelock import FileLock, Timeout
from tqdm import tqdm
from logging.handlers import TimedRotatingFileHandler
import signal
import numpy as np

###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks import selected_stocks  # Example import for your stock list

CACHE_DIR = "data_cache_5min"
INDICATORS_DIR = "main_indicators_5min"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signalnewn5min4.log",
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

# If needed, change to your working directory:
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

# Removed any holiday/weekend logic; no more market_holidays or get_last_trading_day
SIGNALS_DB = "generated_signals_historical5min3.json"
ENTRIES_DIR = "main_indicators_history_entries_5min2"
os.makedirs(ENTRIES_DIR, exist_ok=True)


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
    """
    Convert 'date' to timezone-aware datetimes in market time.
    If the column is naive, we assume it's already in local market time (IST),
    NOT UTC, and localize directly to tz. If tz-aware, just convert to tz.
    """
    d = df.copy()
    if 'date' not in d.columns:
        raise KeyError("DataFrame missing 'date' column.")

    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d.dropna(subset=['date'], inplace=True)

    # If naive => treat as local (IST). If tz-aware => convert to IST.
    if d['date'].dt.tz is None:
        d['date'] = d['date'].dt.tz_localize(tz)
    else:
        d['date'] = d['date'].dt.tz_convert(tz)

    d.sort_values('date', inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


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
        if expected_cols:
            df = df[expected_cols]
    return df

###########################################################
# 4) detect_signals_in_memory
#    With the relaxed breakout/breakdown conditions
###########################################################
def build_signal_id(ticker, row):
    dt_str = row['date'].isoformat()
    return f"{ticker}-{dt_str}"

def is_bullish_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    curr_open = df["open"]
    curr_close = df["close"]
    return (
        (curr_close > curr_open) &
        (prev_close < prev_open) &
        (curr_open < prev_close) &
        (curr_close > prev_open)
    )

def is_bearish_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    curr_open = df["open"]
    curr_close = df["close"]
    return (
        (curr_close < curr_open) &
        (prev_close > prev_open) &
        (curr_open > prev_close) &
        (curr_close < prev_open)
    )

def robust_prev_day_close_for_ticker(ticker, date_of_interest):
    """Load entire CSV for ticker, find previous day's final bar, return that close."""
    main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker.lower()}_main_indicators.csv")
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


def calculate_session_vwap(df: pd.DataFrame,
                           price_col='close', high_col='high', low_col='low', vol_col='volume',
                           tz='Asia/Kolkata') -> pd.DataFrame:
    """
    Session VWAP (resets each trading day in local market time).
    Safe to call even if 'date' is already tz-aware; it derives a session key by .dt.date
    and recomputes VWAP per session, avoiding cross-day anchoring.
    """
    d = df.copy()

    # Ensure tz-aware timestamps in market time for correct session grouping
    dt = pd.to_datetime(d['date'], errors='coerce')
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(tz)
    else:
        dt = dt.dt.tz_convert(tz)

    d['_session'] = dt.dt.date

    # Typical Price
    tp = (d[high_col] + d[low_col] + d[price_col]) / 3.0
    tp_vol = tp * d[vol_col]

    # Cumulative per session
    d['_cum_tp_vol'] = tp_vol.groupby(d['_session']).cumsum()
    d['_cum_vol']    = d[vol_col].groupby(d['_session']).cumsum()

    d['VWAP'] = d['_cum_tp_vol'] / d['_cum_vol']

    # Clean helpers
    d.drop(columns=['_session', '_cum_tp_vol', '_cum_vol'], inplace=True)

    return d

###########################################################
# 5) Indicator Calculations (Ichimoku, TTM Squeeze, VWAP)
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
    # Bollinger Bands
    df['mean_bb'] = df['close'].rolling(bb_window).mean()
    df['std_bb'] = df['close'].rolling(bb_window).std()
    df['upper_bb'] = df['mean_bb'] + 2 * df['std_bb']
    df['lower_bb'] = df['mean_bb'] - 2 * df['std_bb']

    # Keltner Channels
    df['ema_kc'] = df['close'].ewm(span=kc_window).mean()
    true_range = df['high'] - df['low']
    df['atr_kc'] = true_range.rolling(kc_window).mean()
    df['upper_kc'] = df['ema_kc'] + kc_mult * df['atr_kc']
    df['lower_kc'] = df['ema_kc'] - kc_mult * df['atr_kc']

    # Squeeze condition
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
# 6) Price-Action & Consolidation Helpers
###########################################################
def is_bullish_breakout_consolidation(df, idx, consolidation_period=10, volume_multiplier=1.5,
                                      range_threshold=0.02, breakout_buffer=0.01, min_body_ratio=0.10):
    if idx < consolidation_period or idx >= len(df):
        return False
    cdata = df.iloc[idx - consolidation_period: idx]
    ch = cdata['high'].max()
    cl = cdata['low'].min()
    ac = cdata['close'].mean()

    # check if narrow range
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

    # breakout buffer
    if row['close'] <= ch * (1 + breakout_buffer):
        return False

    if 'volume' in df.columns:
        av = cdata['volume'].mean()
        if row['volume'] < volume_multiplier * av:
            return False
    return True

def is_bearish_breakdown_consolidation(df, idx, consolidation_period=10, volume_multiplier=1.5,
                                       range_threshold=0.02, breakdown_buffer=0.01, min_body_ratio=0.10):
    if idx < consolidation_period or idx >= len(df):
        return False
    cdata = df.iloc[idx - consolidation_period: idx]
    ch = cdata['high'].max()
    cl = cdata['low'].min()
    ac = cdata['close'].mean()

    # check if narrow range
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

    # breakdown buffer
    if row['close'] >= cl * (1 - breakdown_buffer):
        return False

    if 'volume' in df.columns:
        av = cdata['volume'].mean()
        if row['volume'] < volume_multiplier * av:
            return False
    return True

###########################################################
# 7) Checks for Bullish / Bearish Patterns
###########################################################
def check_bullish_breakout(daily_change: float, adx_val: float, price: float, high_10: float,
                           vol: float, macd_val: float, roll_vol: float,
                           macd_above_signal_now: bool, stoch_val: float, stoch_diff: float,
                           slope_okay_for_bullish: bool, is_ich_bull: bool, vwap: float) -> bool:
    def valid_adx_for_breakout(adx):
        return (25 <= adx <= 45)
    return ((2.0 <= daily_change <= 10.0)
            and valid_adx_for_breakout(adx_val)
            and (price >= 0.97 * high_10)
            and (macd_val >= 1.0)
            and (vol >= 1.5 * roll_vol)
            and macd_above_signal_now
            and (stoch_diff > 0)
            and (stoch_val > 50)
            and slope_okay_for_bullish
            #and is_ich_bull
            and (price > vwap))

def check_bullish_pullback(daily_change: float, adx_val: float, macd_above_signal_now: bool,
                           stoch_val: float, stoch_diff: float, slope_okay_for_bullish: bool,
                           pullback_pct: float, vwap: float, price: float) -> bool:
    def valid_adx_for_pullback(adx):
        return (25 <= adx <= 45)
    return ((daily_change >= -0.5)
            and valid_adx_for_pullback(adx_val)
            and macd_above_signal_now
            and (stoch_diff > 0)
            and (25 < stoch_val < 75)
            and slope_okay_for_bullish
            and (pullback_pct >= 0.025)
            and (price > vwap))

def check_bearish_breakdown(daily_change: float, adx_val: float, price: float, low_10: float,
                            vol: float, roll_vol: float, macd_val: float,
                            macd_above_signal_now: bool, stoch_val: float, stoch_diff: float,
                            slope_okay_for_bearish: bool, is_ich_bear: bool, vwap: float) -> bool:
    def valid_adx_for_breakout(adx):
        return (25 <= adx <= 45)
    return ((-10.0 <= daily_change <= -2.0)
            and valid_adx_for_breakout(adx_val)
            and (price <= 1.03 * low_10)
            and (vol >= 1.5 * roll_vol)
            and (macd_val <= -1.0)
            and (not macd_above_signal_now)
            and (stoch_val < 50)
            and (stoch_diff < 0)
            and slope_okay_for_bearish
            #and is_ich_bear
            and (price < vwap))

def check_bearish_pullback(daily_change: float, adx_val: float, macd_above_signal_now: bool,
                           stoch_val: float, stoch_diff: float, slope_okay_for_bearish: bool,
                           bounce_pct: float, vwap: float, price: float) -> bool:
    def valid_adx_for_pullback(adx):
        return (25 <= adx <= 45)
    return ((daily_change <= 0.5)
            and valid_adx_for_pullback(adx_val)
            and (not macd_above_signal_now)
            and (stoch_diff < 0)
            and (25 < stoch_val < 75)
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
    # VWAP slope check
    if row['VWAP'] <= prev_row['VWAP']:
        return False
    # Price above VWAP
    if row['close'] < row['VWAP']:
        return False
    # Some buffer above VWAP
    if row['close'] < 1.1 * row['VWAP']:
        return False
    # Candle must be green
    if row['close'] <= row['open']:
        return False
    # Must surpass upper band if it exists
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
    # VWAP slope check
    if row['VWAP'] >= prev_row['VWAP']:
        return False
    # Price below VWAP
    if row['close'] > row['VWAP']:
        return False
    # Some buffer below VWAP
    if row['close'] > 0.9 * row['VWAP']:
        return False
    # Candle must be red
    if row['close'] >= row['open']:
        return False
    # Must surpass lower band if it exists
    if pd.notna(row['VWAP_LowerBand']) and row['close'] > row['VWAP_LowerBand']:
        return False
    vol_window = 10
    if idx < vol_window:
        return False
    recent_avg_vol = df['volume'].iloc[idx - vol_window: idx].mean()
    if row['volume'] < volume_spike_mult * recent_avg_vol:
        return False
    return True

def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set
) -> list:
    """
    Minimal detector with streak filter (now supports range thresholds):
      • Bullish: Daily_Change > 2.0 AND Intra_Change positive streak (single threshold or range)
      • Bearish: Daily_Change < -2.0 AND Intra_Change negative streak (single threshold or range)

    Intra_Threshold handling:
      - If row has Intra_Threshold_L and Intra_Threshold_H -> use (L, H) as a +% range for bulls and mirrored for bears.
      - Else if globals INTRA_THRESHOLD_L/H exist -> use (L, H).
      - Else fall back to single Intra_Threshold (old behavior).
    """

    signals_detected = []
    if df_for_detection.empty:
        return signals_detected

    DEFAULT_STREAK = int(globals().get('INTRA_STREAK_BARS', 1))
    DEFAULT_THRESH = float(globals().get('INTRA_THRESHOLD', 0.0))

    # --- Light anti-chase / readability knobs (unchanged; safe defaults) ---
    BULL_VWAP_STRETCH_MAX = float(globals().get('BULL_VWAP_STRETCH_MAX', 0.0035))  # ≤0.35% above VWAP
    BULL_TOUCH_VWAP_BUFFER = float(globals().get('BULL_TOUCH_VWAP_BUFFER', 0.0015)) # low touches VWAP within 0.15%
    BULL_UPPER_WICK_MAX = float(globals().get('BULL_UPPER_WICK_MAX', 0.45))        # upper wick ≤45% of range
    BULL_NEAR_HIGH_CAP = float(globals().get('BULL_NEAR_HIGH_CAP', 0.005))         # price ≤ 0.5% above rolling_high_10
    BULL_CLOSE_TOP_FRAC = float(globals().get('BULL_CLOSE_TOP_FRAC', 0.60))        # close in top 60%
    BULL_RSI_MAX = float(globals().get('BULL_RSI_MAX', 70.0))                      # cap RSI

    def _close_in_top_fraction(row, frac=0.60):
        rng = row['high'] - row['low']
        if pd.isna(rng) or rng <= 0:
            return False
        return (row['close'] - row['low']) / rng >= frac

    def _upper_wick_fraction(row) -> float:
        rng = (row['high'] - row['low'])
        if pd.isna(rng) or rng <= 0:
            return 1.0  # treat as bad if no range
        return (row['high'] - row['close']) / rng

    # Combine & order
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined.dropna(subset=['date'], inplace=True)
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # IST session key
    dt_series = combined['date']
    try:
        ist = dt_series.dt.tz_localize('Asia/Kolkata') if getattr(dt_series.dt, 'tz', None) is None else dt_series.dt.tz_convert('Asia/Kolkata')
    except Exception:
        ist = pd.to_datetime(dt_series, errors='coerce').dt.tz_localize('Asia/Kolkata')
    combined['_date_only'] = ist.dt.date

    # Indicators (as in your pipeline)
    if 'VWAP' not in combined.columns:
        combined = calculate_session_vwap(combined)
    combined = calculate_vwap_bands(combined, 'VWAP', window=20, stdev_multiplier=2.0)
    combined = calculate_ichimoku(combined)
    combined = calculate_ttm_squeeze(combined)

    # Rolling stats
    rw = 10
    combined['rolling_high_10'] = combined['high'].rolling(rw, min_periods=rw).max()
    combined['rolling_low_10']  = combined['low'].rolling(rw, min_periods=rw).min()
    combined['rolling_vol_10']  = combined['volume'].rolling(rw, min_periods=rw).mean()

    combined['StochDiff']   = combined['Stoch_%D'].diff()
    combined['SMA20_Slope'] = combined['20_SMA'].diff()

    def slope_okay_for_bullish(row) -> bool:
        s20, s20ma = row['SMA20_Slope'], row['20_SMA']
        if pd.isna(s20) or pd.isna(s20ma) or s20ma == 0: return False
        return (s20 > 0) and ((s20 / s20ma) >= 0.001)

    def slope_okay_for_bearish(row) -> bool:
        s20, s20ma = row['SMA20_Slope'], row['20_SMA']
        if pd.isna(s20) or pd.isna(s20ma) or s20ma == 0: return False
        return (s20 < 0) and (abs(s20) / s20ma >= 0.001)

    def is_macd_above(ix: int) -> bool:
        if 0 <= ix < len(combined):
            m, s = combined.loc[ix, 'MACD'], combined.loc[ix, 'MACD_Signal']
            return pd.notna(m) and pd.notna(s) and (m > s)
        return False

    def is_adx_increasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, 'ADX'].to_numpy()
        if any(pd.isna(v) for v in vals): return False
        for j in range(len(vals)-1):
            if vals[j] >= vals[j+1]: return False
        return True

    def is_rsi_increasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, 'RSI'].to_numpy()
        if any(pd.isna(v) for v in vals): return False
        for j in range(len(vals)-1):
            if vals[j] >= vals[j+1]: return False
        return True

    def is_rsi_decreasing(df_, cix, bars_ago=1):
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, 'RSI'].to_numpy()
        if any(pd.isna(v) for v in vals): return False
        for j in range(len(vals)-1):
            if vals[j] <= vals[j+1]: return False
        return True

    # Intra_Change if missing
    if ('Intra_Change' not in combined.columns) or combined['Intra_Change'].isna().all():
        combined['Intra_Change'] = (
            combined.groupby('_date_only')['close'].pct_change().mul(100.0)
        )
    intra_by_idx = combined['Intra_Change']

    def find_cidx(ts):
        m = combined.index[combined['date'] == pd.to_datetime(ts)]
        return int(m[0]) if len(m) else None

    # --- NEW: intra streak with single threshold OR range (low, high) ---
    def has_intra_streak(cidx: int, n: int, thresh, bullish: bool) -> bool:
        """
        Accepts:
          • float/int  -> single-sided threshold (old behavior)
              - Bull: every bar > +t
              - Bear: every bar < -t
          • (low, high) or [low, high] -> band in absolute % terms
              - Bull: every bar in [+low, +high]
              - Bear: every bar in [-high, -low]
        """
        if n <= 0:
            return True
        if cidx is None or cidx - (n - 1) < 0:
            return False

        start = cidx - (n - 1)
        # ensure the n bars are on the same trading day
        cur_day = combined['_date_only'].iloc[cidx]
        window_days = combined['_date_only'].iloc[start:cidx+1]
        if window_days.nunique() != 1 or window_days.iloc[0] != cur_day:
            return False

        seg = intra_by_idx.iloc[start:cidx+1]
        if seg.isna().any():
            return False

        # Parse thresholds
        # range_mode True when thresh is (low, high)
        range_mode = False
        try:
            if isinstance(thresh, (tuple, list)) and len(thresh) == 2:
                lo = float(thresh[0]); hi = float(thresh[1])
                lo, hi = sorted((abs(lo), abs(hi)))  # use absolute magnitudes
                range_mode = True
            else:
                t = abs(float(thresh))
        except Exception:
            return False  # invalid threshold input

        if not range_mode:
            # original behavior
            return (seg > t).all() if bullish else (seg < -t).all()

        # range behavior
        if bullish:
            return ((seg >= lo) & (seg <= hi)).all()
        else:
            return ((seg <= -lo) & (seg >= -hi)).all()

    # --- Simple stochastic helpers (no lookback, minimal tolerance) ---
    STOCH_TOL = float(globals().get('STOCH_TOL', 1.0))  # 1 point tolerance by default

    def stoch_bull_cross(df, idx, tol=STOCH_TOL):
        """Bull cross = K above (or ~equal within tol) to D and K rising vs previous bar."""
        if idx < 1:
            return False
        k0 = df.at[idx, 'Stoch_%K']; d0 = df.at[idx, 'Stoch_%D']
        k1 = df.at[idx-1, 'Stoch_%K']; d1 = df.at[idx-1, 'Stoch_%D']
        if any(pd.isna(x) for x in (k0, d0, k1, d1)):
            return False
        return (k0 >= d0 - tol) and (k0 > k1)

    def stoch_bear_cross(df, idx, tol=STOCH_TOL):
        """Bear cross = K below (or ~equal within tol) to D and K falling vs previous bar."""
        if idx < 1:
            return False
        k0 = df.at[idx, 'Stoch_%K']; d0 = df.at[idx, 'Stoch_%D']
        k1 = df.at[idx-1, 'Stoch_%K']; d1 = df.at[idx-1, 'Stoch_%D']
        if any(pd.isna(x) for x in (k0, d0, k1, d1)):
            return False
        return (k0 <= d0 + tol) and (k0 < k1)

    def stoch_bull_ok(df, idx, zone=None):
        """Keep signature for compatibility; rely on the simple cross."""
        return stoch_bull_cross(df, idx)

    def stoch_bear_ok(df, idx, zone=None):
        """Keep signature for compatibility; rely on the simple cross."""
        return stoch_bear_cross(df, idx)

    # Evaluate each row to detect signals
    for _, row_detect in df_for_detection.iterrows():
        dt = row_detect.get('date', None)
        if pd.isna(dt):
            continue

        daily_change = row_detect.get('Daily_Change', np.nan)
        if pd.isna(daily_change):
            continue

        # Streak bars (N)
        n = row_detect.get('Intra_Streak_Bars', DEFAULT_STREAK)
        try:
            n = int(n) if pd.notna(n) else DEFAULT_STREAK
        except Exception:
            n = DEFAULT_STREAK

        # --- Build threshold or range (row → globals → default) ---
        # row-level range (if provided)
        tL = row_detect.get('Intra_Threshold_L', 0.25)
        tH = row_detect.get('Intra_Threshold_H', 0.75)

        thresh_value = row_detect.get('Intra_Threshold', None)
        if tL is not None and tH is not None and not (pd.isna(tL) or pd.isna(tH)):
            try:
                thresh = (float(tL), float(tH))
            except Exception:
                thresh = None
        else:
            # global range?
            gL = globals().get('INTRA_THRESHOLD_L', 0.25)
            gH = globals().get('INTRA_THRESHOLD_H', 0.75)
            if gL is not None and gH is not None:
                try:
                    thresh = (float(gL), float(gH))
                except Exception:
                    thresh = None
            else:
                # single threshold (fallback)
                try:
                    thresh = float(thresh_value) if (thresh_value is not None and not pd.isna(thresh_value)) else DEFAULT_THRESH
                except Exception:
                    thresh = DEFAULT_THRESH

        cidx = find_cidx(dt)
        if cidx is None:
            continue  # can't evaluate this row without its position

        # Read needed values from the aligned bar
        price      = combined.loc[cidx, 'close']
        vol        = combined.loc[cidx, 'volume']
        high_10    = combined.loc[cidx, 'rolling_high_10']
        low_10     = combined.loc[cidx, 'rolling_low_10']
        roll_vol   = combined.loc[cidx, 'rolling_vol_10']
        stoch_val  = combined.loc[cidx, 'Stoch_%D']
        stoch_diff = combined.loc[cidx, 'StochDiff']
        adx_val    = combined.loc[cidx, 'ADX']
        macd_val   = combined.loc[cidx, 'MACD']
        vwap       = combined.loc[cidx, 'VWAP']
        rsi        = combined.loc[cidx, 'RSI']

        macd_above_signal_now = is_macd_above(cidx)
        slope_bull = slope_okay_for_bullish(combined.loc[cidx])
        slope_bear = slope_okay_for_bearish(combined.loc[cidx])
        ich_bull   = is_ichimoku_bullish(combined, cidx)
        ich_bear   = is_ichimoku_bearish(combined, cidx)

        vwap_prev = combined.loc[cidx - 1, 'VWAP'] if cidx >= 1 else np.nan
        vwap_now  = vwap
        vwap_up   = (pd.notna(vwap_prev) and pd.notna(vwap_now) and (vwap_now > vwap_prev))

        # Simple locals for the current bar (guards are commented out below)
        row_c      = combined.loc[cidx]
        rng        = (row_c['high'] - row_c['low']) if pd.notna(row_c['high']) and pd.notna(row_c['low']) else np.nan
        # tuned_cap should tolerate both single thresholds and ranges
        try:
            if isinstance(thresh, (tuple, list)) and len(thresh) == 2:
                _tcap = max(abs(float(thresh[0])), abs(float(thresh[1])))
            else:
                _tcap = abs(float(thresh))
        except Exception:
            _tcap = abs(float(DEFAULT_THRESH))
        tuned_cap  = max(BULL_VWAP_STRETCH_MAX, (_tcap / 100.0))
        stretch    = ((price / vwap) - 1.0) if (pd.notna(vwap) and vwap > 0) else np.inf
        touched_vw = (row_c['low'] <= vwap * (1.0 + BULL_TOUCH_VWAP_BUFFER)) if pd.notna(vwap) else False
        uw_frac    = _upper_wick_fraction(row_c)
        near_high  = (price <= (row_c['rolling_high_10'] * (1.0 + BULL_NEAR_HIGH_CAP))) if pd.notna(row_c['rolling_high_10']) else True
        strong_close = _close_in_top_fraction(row_c, frac=BULL_CLOSE_TOP_FRAC)

        # ---- Bullish condition ----
        if (
            daily_change > 2
            and has_intra_streak(cidx, n, thresh, bullish=True)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and 30 < adx_val < 45
            and is_rsi_increasing(combined, cidx, bars_ago=2)
            and 50 < rsi < 75
            and stoch_val > 50
            and stoch_diff > 0
            and stoch_bull_ok(combined, cidx)
            and price > 1.005 * vwap
            and vol >= 2.5 * roll_vol
            # Optional micro-guards (left commented for now):
            # and stretch <= tuned_cap
            # and touched_vw
            # and uw_frac <= BULL_UPPER_WICK_MAX
            # and strong_close
            # and near_high
        ):
            signal_id = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_DCHANGE_INTRASTREAK"
            if signal_id not in existing_signal_ids:
                signals_detected.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'DailyChange+IntraStreak',
                    'Trend Type': 'Bullish',
                    'Price': round(float(price), 2),
                    'Daily Change %': round(float(daily_change), 2),
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })
                existing_signal_ids.add(signal_id)

        # ---- Bearish condition ----
        if (
            daily_change < -2
            and has_intra_streak(cidx, n, thresh, bullish=False)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and 30 < adx_val < 45
            and is_rsi_decreasing(combined, cidx, bars_ago=2)
            and 25 < rsi < 50
            and stoch_val < 50
            and stoch_diff < 0
            and stoch_bear_ok(combined, cidx)
            and price < 0.995 * vwap
            and vol >= 2.5 * roll_vol
        ):
            signal_id = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_DCHANGE_INTRASTREAK"
            if signal_id not in existing_signal_ids:
                signals_detected.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'DailyChange+IntraStreak',
                    'Trend Type': 'Bearish',
                    'Price': round(float(price), 2),
                    'Daily Change %': round(float(daily_change), 2),
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })
                existing_signal_ids.add(signal_id)

    return signals_detected



###########################################################
# 5) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    Updates 'main_indicators' CSV for a given ticker:
      - Sets 'Entry Signal'='Yes' and 'CalcFlag'='Yes' for all matching Signal_IDs.
    """
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

            # Ensure required columns exist
            required_cols = ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]
            for col in required_cols:
                if col not in df_main.columns:
                    df_main[col] = ""

            # Extract Signal_IDs from signals_list
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
# 6) find_price_action_entries_for_date
###########################################################
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_price_action_entries_for_date(target_date):
    """
    Detects and collects all new trading signals for the given date,
    across all *_main_indicators.csv.
    Intraday timeframe considered is 09:25 to 14:05 (example).
    """
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(14, 4)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def find_prev_day_close_for_ticker(df_full, trading_day):
        """
        Find the last close from the day BEFORE `trading_day`.
        Return None if not found.
        """
        prev_day = trading_day - timedelta(days=1)
        df_before = df_full[df_full['date'].dt.date < trading_day]
        if df_before.empty:
            return None
        return df_before.iloc[-1]['close']

    def process_file(file_path):
        ticker = os.path.basename(file_path).replace('_main_indicators.csv', '').upper()
        df_full = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
        if df_full.empty:
            return []


        # Only analyze the portion for our target_date
        df_today = df_full[(df_full['date'] >= st) & (df_full['date'] <= et)].copy()

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,   # entire dataset for correct rolling(10)
            df_for_detection=df_today,# final portion for new signals
            existing_signal_ids=existing_ids
        )

        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
                return new_signals
        return []

    # ThreadPoolExecutor for parallel reading & detection
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Processing {target_date}"):
            file = future_to_file[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logging.info(f"No new signals detected for {target_date}.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values('date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    logging.info(f"Total new signals detected for {target_date}: {len(signals_df)}")
    return signals_df

###########################################################
# 7) create_papertrade_df => final CSV
###########################################################
def create_papertrade_df(df_entries_yes, output_file, target_date):
    """
    Takes a DataFrame of signals, ensures only *first* Ticker per day is appended,
    then writes to the daily papertrade CSV with minimal duplication.
    """
    if df_entries_yes.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return

    df_entries_yes = normalize_time(df_entries_yes, tz='Asia/Kolkata')
    df_entries_yes.sort_values('date', inplace=True)

    # Example intraday timeframe for final filter
    start_dt = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    end_dt = india_tz.localize(datetime.combine(target_date, dt_time(15, 30)))
    mask = (df_entries_yes['date'] >= start_dt) & (df_entries_yes['date'] <= end_dt)
    df_entries_yes = df_entries_yes.loc[mask].copy()
    if df_entries_yes.empty:
        print("[Papertrade] No rows in [09:25 - 17:30].")
        return

    # Ensure columns for 'Target Price', 'Quantity', 'Total value'
    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df_entries_yes.columns:
            df_entries_yes[col] = ""

    # Simple example for target & quantity
    for idx, row in df_entries_yes.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price > 0:
            if trend == "Bullish":
                target = price * 1.01
            elif trend == "Bearish":
                target = price * 0.99
            else:
                target = price
            qty = int(20000 / price)
            total_val = qty * price

            df_entries_yes.at[idx, "Target Price"] = round(target, 2)
            df_entries_yes.at[idx, "Quantity"] = qty
            df_entries_yes.at[idx, "Total value"] = round(total_val, 2)

    # Set the 'time' column to the current detection time if missing
    if 'time' not in df_entries_yes.columns:
        df_entries_yes['time'] = datetime.now(india_tz).strftime('%H:%M:%S')
    else:
        blank_mask = (df_entries_yes['time'].isna()) | (df_entries_yes['time'] == "")
        if blank_mask.any():
            df_entries_yes.loc[blank_mask, 'time'] = datetime.now(india_tz).strftime('%H:%M:%S')

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
            for _, row in df_entries_yes.iterrows():
                ticker = row['Ticker']
                row_date_day = row['date'].date()
                key = (ticker, row_date_day)
                if key not in existing_key_set:
                    new_rows.append(row)

            if not new_rows:
                print("[Papertrade] No brand-new rows to append. All existing.")
                return

            new_df = pd.DataFrame(new_rows, columns=df_entries_yes.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['Ticker', 'date'], keep='first', inplace=True)
            combined.sort_values('date', inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")
        else:
            df_entries_yes.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
            df_entries_yes.sort_values('date', inplace=True)
            df_entries_yes.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df_entries_yes)} rows.")

###########################################################
# 8) run_for_date(date_str) -- no weekend/holiday skip
###########################################################
def run_for_date(date_str):
    """
    Runs the detection & CSV updates for one specific date (YYYY-MM-DD),
    with no weekend/holiday skipping.
    Also saves all signals for that date in ENTRIES_DIR as {date}_entries.csv.
    """
    try:
        # Convert user-specified string to a date
        requested_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        last_trading_day = requested_date

        date_label = last_trading_day.strftime('%Y-%m-%d')
        papertrade_file = f"papertrade_5min_v6_{date_label}.csv"
        entries_file = os.path.join(ENTRIES_DIR, f"{date_label}_entries.csv")

        # 1) Detect signals
        raw_signals_df = find_price_action_entries_for_date(last_trading_day)
        if raw_signals_df.empty:
            logging.info(f"No signals found for {date_label}.")
            # But still, save an empty file for completeness
            pd.DataFrame().to_csv(entries_file, index=False)
            return

        # ===> NEW: Save all signals for the day <===
        raw_signals_df.to_csv(entries_file, index=False)
        print(f"[ENTRIES] Wrote {len(raw_signals_df)} signals to {entries_file}")

        # 2) Mark signals in main CSV
        signals_grouped = raw_signals_df.groupby('Ticker')
        for ticker, group in signals_grouped:
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, group.to_dict('records'), main_csv_path, india_tz)

        # 3) Append to papertrade CSV
        create_papertrade_df(raw_signals_df, papertrade_file, last_trading_day)
        logging.info(f"Papertrade entries written to {papertrade_file}.")

    except Exception as e:
        logging.error(f"Error in run_for_date({date_str}): {e}")
        logging.error(traceback.format_exc())


###########################################################
# 9) Graceful Shutdown Handler
###########################################################
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

###########################################################
# Entry Point (Multiple Dates)
###########################################################


if __name__ == "__main__":
    ENTRIES_DIR = "main_indicators_history_entries"
    os.makedirs(ENTRIES_DIR, exist_ok=True)
    date_args = sys.argv[1:]
    if not date_args:
        # If no args given, define a list of back-dates here:
            date_args = [
    

    "2025-09-03"

]



    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)