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

CACHE_DIR = "data_cache_july_5min"
INDICATORS_DIR = "main_indicators_july_5min"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signalnewn5min1.log",
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
SIGNALS_DB = "generated_signals_historical5min1.json"
ENTRIES_DIR = "main_indicators_history_entries_5min"
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
    Runs the advanced detection logic on df_for_detection, using rolling-window
    from df_for_rolling. No recalculation of 'Daily_Change'; we read from the CSV.

    Returns a list of new signals, each containing a 'Signal_ID'.
    """
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    # Combine for indicator continuity
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Ensure we have VWAP, Ichimoku, TTM Squeeze
    if 'VWAP' not in combined.columns:
        combined = calculate_session_vwap(combined)
    combined = calculate_vwap_bands(combined, 'VWAP', window=20, stdev_multiplier=2.0)
    combined = calculate_ichimoku(combined)
    combined = calculate_ttm_squeeze(combined)

    # Rolling stats
    rolling_window = 10
    combined['rolling_high_10'] = combined['high'].rolling(rolling_window, min_periods=rolling_window).max()
    combined['rolling_low_10']  = combined['low'].rolling(rolling_window, min_periods=rolling_window).min()
    combined['rolling_vol_10']  = combined['volume'].rolling(rolling_window, min_periods=rolling_window).mean()

    combined['StochDiff']   = combined['Stoch_%D'].diff()
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
            s = combined.loc[ix, 'MACD_Signal']
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
        # Check strictly increasing
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
        # Check strictly increasing
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
        # Check strictly decreasing
        for i in range(len(vals) - 1):
            if vals[i] <= vals[i + 1]:
                return False
        return True

    # Evaluate each row in df_for_detection
    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect['date']

        # Match row in combined
        match = combined[combined['date'] == dt]
        if match.empty:
            continue

        cidx = match.index[0]
        signals_for_this_bar = []

        # Price, volume, rolling measures
        price = combined.loc[cidx, 'close']
        vol   = combined.loc[cidx, 'volume']
        high_10 = combined.loc[cidx, 'rolling_high_10']
        low_10  = combined.loc[cidx, 'rolling_low_10']
        roll_vol= combined.loc[cidx, 'rolling_vol_10']
        stoch_val   = combined.loc[cidx, 'Stoch_%D']
        stoch_diff  = combined.loc[cidx, 'StochDiff']
        adx_val     = combined.loc[cidx, 'ADX']
        macd_val    = combined.loc[cidx, 'MACD']
        macd_above_signal_now = is_macd_above(cidx)
        slope_bull  = slope_okay_for_bullish(combined.loc[cidx])
        slope_bear  = slope_okay_for_bearish(combined.loc[cidx])
        ich_bull    = is_ichimoku_bullish(combined, cidx)
        ich_bear    = is_ichimoku_bearish(combined, cidx)
        vwap        = combined.loc[cidx, 'VWAP']
        rsi        = combined.loc[cidx, 'RSI']

        # Pull the daily change from the CSV's row
        daily_change = row_detect.get('Daily_Change', np.nan)
        if pd.isna(daily_change):
            daily_change = 0.0  # fallback if missing

        # Check for short-term pullbacks/bounces
        if cidx >= 5:
            recent_high_5 = combined.loc[cidx-5:cidx, 'high'].max()
            recent_low_5  = combined.loc[cidx-5:cidx, 'low'].min()
        else:
            recent_high_5 = combined.loc[:cidx, 'high'].max()
            recent_low_5  = combined.loc[:cidx, 'low'].min()

        pullback_pct = 0.0
        if recent_high_5 and recent_high_5 > 0:
            pullback_pct = (recent_high_5 - price) / recent_high_5
        bounce_pct = 0.0
        if recent_low_5 and recent_low_5 > 0:
            bounce_pct = (price - recent_low_5) / recent_low_5

        # ---- BULLISH BREAKOUT ----
        if (check_bullish_breakout(daily_change, adx_val, price, high_10, vol, macd_val, roll_vol,
                                   macd_above_signal_now, stoch_val, stoch_diff, slope_bull, ich_bull, vwap)
            #and is_bullish_breakout_consolidation(combined, cidx)
            #and is_adx_increasing(combined, cidx, bars_ago=1)
            #and is_rsi_increasing(combined, cidx, bars_ago=1)
            #and check_bullish_breakout_advanced(combined, cidx, volume_spike_mult=1.5)
            ):

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
        if (check_bullish_pullback(daily_change, adx_val, macd_above_signal_now, stoch_val, stoch_diff,
                                  slope_bull, pullback_pct, vwap, price)
            and is_adx_increasing(combined, cidx, bars_ago=1)):
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
            #and is_bearish_breakdown_consolidation(combined, cidx)
            #and is_adx_increasing(combined, cidx, bars_ago=1)
            #and is_rsi_decreasing(combined, cidx, bars_ago=1)
            #and check_bearish_breakdown_advanced(combined, cidx, volume_spike_mult=1.5)
            ):

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
        if (check_bearish_pullback(daily_change, adx_val, macd_above_signal_now, stoch_val, stoch_diff,
                                  slope_bear, bounce_pct, vwap, price)
            and is_adx_increasing(combined, cidx, bars_ago=1)):
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


##############################################################################
# 7) Provide or Mock 'robust_prev_day_close_for_ticker' if needed
##############################################################################

def robust_prev_day_close_for_ticker(ticker: str, date_):
    # You can implement your own logic here or remove if not needed
    return None



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
        papertrade_file = f"papertrade_5min_{date_label}.csv"
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
    

    # January 2025
    "2025-01-01","2025-01-02","2025-01-03",
    "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
    "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
    "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
    # 2025-01-26 is Republic Day (Sunday+holiday)
    "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

    # February 2025
    "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
    "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
    "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
    "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

    # March 2025
    "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
    "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
    # 2025-03-14 is Holi (holiday)
    "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
    "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
    "2025-03-31",

    # April 2025
    "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
    "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
    # 2025-04-11 is Eid-ul-Fitr (holiday)
    "2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18",
    "2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25",
    "2025-04-28","2025-04-29","2025-04-30",

    # May 2025
    "2025-05-02",
    "2025-05-05","2025-05-06","2025-05-07","2025-05-08","2025-05-09",
    "2025-05-12","2025-05-13","2025-05-14","2025-05-15","2025-05-16",
    "2025-05-19","2025-05-20","2025-05-21","2025-05-22","2025-05-23",
    "2025-05-26","2025-05-27","2025-05-28","2025-05-29","2025-05-30",

    # June 2025
    "2025-06-02","2025-06-03","2025-06-04","2025-06-05","2025-06-06",
    "2025-06-09","2025-06-10","2025-06-11","2025-06-12","2025-06-13",
    "2025-06-16","2025-06-17","2025-06-18","2025-06-19","2025-06-20",
    "2025-06-23","2025-06-24","2025-06-25","2025-06-26","2025-06-27",
    "2025-06-30",

    # July 2025
    "2025-07-01","2025-07-02","2025-07-03","2025-07-04",
    "2025-07-07","2025-07-08","2025-07-09","2025-07-10"
]



    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)