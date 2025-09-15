# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:54:27 2025

@author: Saarit
"""

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

###########################################################
# 1) Configuration & Setup
###########################################################
india_tz = pytz.timezone('Asia/Kolkata')
# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks import selected_stocks  # Example import for your stock list

CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\signal1.log",
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
SIGNALS_DB = "generated_signals_historical.json"

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

    # If date is naive, localize to UTC first
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

###########################################################
# 4) Ichimoku & TTM Squeeze Calculations (Relaxed)
###########################################################
import pandas as pd

import pandas as pd

def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Ichimoku indicators for your DataFrame:
      - Tenkan-sen (9)
      - Kijun-sen (26)
      - Senkou Span A/B (shifted forward 26)
      - Chikou Span (shifted backward 26)
    """
    # Tenkan-sen (9)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    # Kijun-sen (26)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    # Senkou Span A
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (52)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    # Chikou Span (lagging close by 26)
    df['chikou_span'] = df['close'].shift(-26)

    return df

def is_ichimoku_bullish(df: pd.DataFrame, idx: int, tol: float = 0.05) -> bool:
    """
    An even more lenient bullish check:
      - The close must be >= (1 - tol) * the lower boundary of the cloud.
      - Tenkan-sen must be >= Kijun-sen * (1 - tol).

    Default tol is 0.1 (10%), allowing a wider margin and more bullish signals.
    """
    row = df.iloc[idx]
    if pd.isna(row['senkou_span_a']) or pd.isna(row['senkou_span_b']):
        return False

    lower_span = min(row['senkou_span_a'], row['senkou_span_b'])
    
    # Looser requirement for the price above the lower cloud boundary
    if row['close'] < lower_span * (1 - tol):
        return False

    # Looser requirement for Tenkan-sen relative to Kijun-sen
    if row['tenkan_sen'] < row['kijun_sen'] * (1 - tol):
        return False

    return True

def is_ichimoku_bearish(df: pd.DataFrame, idx: int, tol: float = 0.05) -> bool:
    """
    An even more lenient bearish check:
      - The close must be <= (1 + tol) * the higher boundary of the cloud.
      - Tenkan-sen must be <= Kijun-sen * (1 + tol).

    Default tol is 0.1 (10%), allowing a wider margin and more bearish signals.
    """
    row = df.iloc[idx]
    if pd.isna(row['senkou_span_a']) or pd.isna(row['senkou_span_b']):
        return False

    higher_span = max(row['senkou_span_a'], row['senkou_span_b'])

    # Looser requirement for price below the higher cloud boundary
    if row['close'] > higher_span * (1 + tol):
        return False

    # Looser requirement for Tenkan-sen relative to Kijun-sen
    if row['tenkan_sen'] > row['kijun_sen'] * (1 + tol):
        return False

    return True


def calculate_ttm_squeeze(df: pd.DataFrame,
                          bb_window: int = 20,
                          kc_window: int = 20,
                          kc_mult: float = 1.5,
                          tol: float = 0.3) -> pd.DataFrame:
    """
    A slightly more tolerant TTM Squeeze calculation:
      - We use Bollinger Bands (BB) and Keltner Channels (KC).
      - By default, the tolerance is set to 0.1 (10%), which is higher than usual
        so that more 'squeeze_on' bars can show up.

    squeeze_on = 1 when:
      lower_bb > lower_kc * (1 - tol) AND upper_bb < upper_kc * (1 + tol)

    squeeze_release = 1 when a squeeze_on was 1 in the previous bar but 0 now.
    """
    # Bollinger Bands
    df['mean_bb'] = df['close'].rolling(bb_window).mean()
    df['std_bb'] = df['close'].rolling(bb_window).std()
    df['upper_bb'] = df['mean_bb'] + 2 * df['std_bb']
    df['lower_bb'] = df['mean_bb'] - 2 * df['std_bb']

    # Keltner Channels (using a simple ATR)
    df['ema_kc'] = df['close'].ewm(span=kc_window).mean()
    true_range = df['high'] - df['low']
    df['atr_kc'] = true_range.rolling(kc_window).mean()
    df['upper_kc'] = df['ema_kc'] + kc_mult * df['atr_kc']
    df['lower_kc'] = df['ema_kc'] - kc_mult * df['atr_kc']

    # More lenient condition for "squeeze_on"
    df['squeeze_on'] = (
        (df['lower_bb'] > df['lower_kc'] * (1 - tol)) &
        (df['upper_bb'] < df['upper_kc'] * (1 + tol))
    ).astype(int)

    # Mark the first bar after the squeeze ends as "squeeze_release"
    df['squeeze_release'] = 0
    for i in range(1, len(df)):
        if df['squeeze_on'].iloc[i-1] == 1 and df['squeeze_on'].iloc[i] == 0:
            df.at[i, 'squeeze_release'] = 1

    return df

def is_squeeze_on(df: pd.DataFrame, idx: int) -> bool:
    """
    Checks if 'squeeze_on' is 1 at the given row.
    """
    if 0 <= idx < len(df):
        return df.at[idx, 'squeeze_on'] == 1
    return False

def is_squeeze_releasing(df: pd.DataFrame, idx: int) -> bool:
    """
    Checks if 'squeeze_release' is 1 at the given row.
    (Meaning it was 'squeeze_on' in the previous bar but not this one.)
    """
    if 0 <= idx < len(df):
        return df.at[idx, 'squeeze_release'] == 1
    return False




from datetime import datetime
import pandas as pd
import numpy as np

def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,   # bigger dataset for rolling (10+ bars)
    df_for_detection: pd.DataFrame, # last portion (e.g., last 4 bars) for new signals
    existing_signal_ids: set,
    prev_day_close: float = None
) -> list:
    """
    Stricter version to further reduce false signals and aim for higher-probability trades:
      - ADX >= 35 for breakouts, >= 30 for pullbacks
      - RSI > 45 for bullish, < 55 for bearish
      - Narrow daily_change windows
      - Higher volume multiplier for breakouts (e.g., 2.5)
      - More strict MACD offsets
      - Additional slope-of-SMA check to confirm trending direction
    """
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    # 1) Attempt to get prev_day_close if not explicitly supplied
    final_bar_date = df_for_detection['date'].iloc[-1]
    if not prev_day_close:
        prev_day_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

    # 2) Combine rolling + detection frames
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # 3) Compute daily_change from last bar in df_for_detection
    try:
        latest_close = df_for_detection['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except Exception:
        daily_change = 0.0

    # 4) Create rolling columns
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
    combined['StochDiff'] = combined['Stochastic'].diff()
    combined['SMA20_Slope'] = combined['20_SMA'].diff()

    # NEW: Calculate OBV
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        obv = [0]
        for i in range(1, len(df)):
            if df.loc[i, 'close'] > df.loc[i-1, 'close']:
                obv.append(obv[-1] + df.loc[i, 'volume'])
            elif df.loc[i, 'close'] < df.loc[i-1, 'close']:
                obv.append(obv[-1] - df.loc[i, 'volume'])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)
    combined['OBV'] = calculate_obv(combined)

    # NEW: Calculate ROC (5-period)
    def calculate_roc(series: pd.Series, period: int = 5) -> pd.Series:
        return series.pct_change(periods=period) * 100
    combined['ROC_5'] = calculate_roc(combined['close'], period=5)

    # 5) Helper checks
    def valid_adx_for_breakout(adx_val: float) -> bool:
        # Require ADX >= 35 for breakouts
        return (not pd.isna(adx_val)) and (adx_val >= 25) and (adx_val <= 45)

    def valid_adx_for_pullback(adx_val: float) -> bool:
        # Require ADX >= 30 for pullbacks
        return (not pd.isna(adx_val)) and (adx_val >= 25) and (adx_val <= 45)

    def is_macd_above(idx: int) -> bool:
        if 0 <= idx < len(combined):
            macd_val = combined.loc[idx, 'MACD']
            sig_val = combined.loc[idx, 'Signal_Line']
            if pd.isna(macd_val) or pd.isna(sig_val):
                return False
            return (macd_val > sig_val)
        return False

    def get_trend_type(row) -> str:
        """
        Stricter RSI thresholds for identifying a bullish/bearish environment:
         - Bullish if RSI > 45 and close > 1.003×VWAP
         - Bearish if RSI < 55 and close < 0.997×VWAP
        """
        rsi_val = row.get("RSI", 50)
        cls_val = row.get("close", 0)
        vwap_val = row.get("VWAP", 0)
        if (rsi_val > 20) and (cls_val > 1.003 * vwap_val):
            return "Bullish"
        elif (rsi_val < 80) and (cls_val < 0.997 * vwap_val):
            return "Bearish"
        else:
            return ""  # No clear trend

    # Slope checks for bullish/bearish
    # "Significant" means the slope is at least 0.1% of the SMA value
    def slope_okay_for_bullish(row) -> bool:
        slope_20 = row['SMA20_Slope']
        sma_20 = row['20_SMA']
        if pd.isna(slope_20) or pd.isna(sma_20) or sma_20 == 0:
            return False
        return (slope_20 > 0) and ((slope_20 / sma_20) >= 0.001)

    def slope_okay_for_bearish(row) -> bool:
        slope_20 = row['SMA20_Slope']
        sma_20 = row['20_SMA']
        if pd.isna(slope_20) or pd.isna(sma_20) or sma_20 == 0:
            return False
        return (slope_20 < 0) and ((abs(slope_20) / sma_20) >= 0.001)

    def is_adx_increasing(df: pd.DataFrame, current_idx: int, bars_ago: int = 1) -> bool:
        """
        Checks if ADX is strictly increasing for the last `bars_ago` intervals
        ending at `current_idx`.
        """
        start_idx = current_idx - bars_ago
        if start_idx < 0:
            return False  # Not enough data to check
        adx_values = df.loc[start_idx:current_idx, 'ADX'].to_numpy()
        if any(pd.isna(val) for val in adx_values):
            return False
        for i in range(len(adx_values) - 1):
            if adx_values[i] >= adx_values[i + 1]:
                return False
        return True

    def is_rsi_increasing(df: pd.DataFrame, current_idx: int, bars_ago: int = 1) -> bool:
        """
        Checks if RSI is strictly increasing over the last `bars_ago + 1` bars ending at current_idx.
        """
        start_idx = current_idx - bars_ago
        if start_idx < 0:
            return False
        rsi_values = df.loc[start_idx:current_idx, 'RSI'].to_numpy()
        if any(pd.isna(val) for val in rsi_values):
            return False
        for i in range(len(rsi_values) - 1):
            if rsi_values[i] >= rsi_values[i + 1]:
                return False
        return True

    def is_rsi_decreasing(df: pd.DataFrame, current_idx: int, bars_ago: int = 3) -> bool:
        """
        Checks if RSI is strictly decreasing over the last `bars_ago + 1` bars ending at current_idx.
        """
        start_idx = current_idx - bars_ago
        if start_idx < 0:
            return False
        rsi_values = df.loc[start_idx:current_idx, 'RSI'].to_numpy()
        if any(pd.isna(val) for val in rsi_values):
            return False
        for i in range(len(rsi_values) - 1):
            if rsi_values[i] <= rsi_values[i + 1]:
                return False
        return True

    # NEW: OBV helper checks
    def is_obv_increasing(df: pd.DataFrame, current_idx: int, bars_ago: int = 3) -> bool:
        start_idx = current_idx - bars_ago
        if start_idx < 0:
            return False
        obv_values = df.loc[start_idx:current_idx, 'OBV'].to_numpy()
        return all(obv_values[i] < obv_values[i+1] for i in range(len(obv_values)-1))

    def is_obv_decreasing(df: pd.DataFrame, current_idx: int, bars_ago: int = 3) -> bool:
        start_idx = current_idx - bars_ago
        if start_idx < 0:
            return False
        obv_values = df.loc[start_idx:current_idx, 'OBV'].to_numpy()
        return all(obv_values[i] > obv_values[i+1] for i in range(len(obv_values)-1))

    # NEW: ROC helper checks
    def is_roc_positive(df: pd.DataFrame, current_idx: int) -> bool:
        if pd.isna(df.loc[current_idx, 'ROC_5']):
            return False
        return df.loc[current_idx, 'ROC_5'] > 0

    def is_roc_negative(df: pd.DataFrame, current_idx: int) -> bool:
        if pd.isna(df.loc[current_idx, 'ROC_5']):
            return False
        return df.loc[current_idx, 'ROC_5'] < 0
    
    # 5) Ichimoku + TTM Squeeze
    combined = calculate_ichimoku(combined)
    combined = calculate_ttm_squeeze(combined)


    # 6) Parameter constants for stricter logic
    VOLUME_MULTIPLIER_BREAKOUT = 1.5     # higher volume requirement
    MACD_BULLISH_OFFSET_BREAKOUT = 3.0
    MACD_BULLISH_OFFSET_BREAKOUT_max = 20
    MACD_BULLISH_OFFSET_PULLBACK = 1
    MACD_BEARISH_OFFSET_BREAKOUT = -3.0
    MACD_BEARISH_OFFSET_BREAKOUT_min = -20
    MACD_BEARISH_OFFSET_PULLBACK = -1

    # 7) Loop over detection bars
    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect['date']
        match = combined[combined['date'] == dt]
        if match.empty:
            continue
        cidx = match.index[0]

        # Build unique signal ID
        signal_id = f"{ticker}-{dt.isoformat()}"
        if signal_id in existing_signal_ids:
            continue

        # Determine trend
        row_trend = get_trend_type(combined.loc[cidx])
        if not row_trend:
            continue

        # Check slope alignment
        if row_trend == "Bullish":
            if not slope_okay_for_bullish(combined.loc[cidx]):
                continue
        else:  # row_trend == "Bearish"
            if not slope_okay_for_bearish(combined.loc[cidx]):
                continue

        # Gather relevant columns
        adx_val = combined.loc[cidx, 'ADX']
        stoch_val = combined.loc[cidx, 'Stochastic']
        stoch_diff = combined.loc[cidx, 'StochDiff']
        macd_val = combined.loc[cidx, 'MACD']
        macd_above_signal_now = is_macd_above(cidx)
        price = combined.loc[cidx, 'close']
        vol = combined.loc[cidx, 'volume']
        roll_vol = combined.loc[cidx, 'rolling_vol_10']
        high_10 = combined.loc[cidx, 'rolling_high_10']
        low_10 = combined.loc[cidx, 'rolling_low_10']
        sma_20_val = combined.loc[cidx, '20_SMA']
        vwap = combined.loc[cidx, 'VWAP']
        rsi = combined.loc[cidx, 'RSI']
        
        # Ichimoku & Squeeze checks
        is_ich_bull = is_ichimoku_bullish(combined, cidx)
        is_ich_bear = is_ichimoku_bearish(combined, cidx)
        sq_on = is_squeeze_on(combined, cidx)
        sq_release = is_squeeze_releasing(combined, cidx)

        # Branch for bullish vs. bearish
        if row_trend == "Bullish":
            # A) Bullish breakout
            bullish_breakout = (
                #(2.0 <= daily_change)
                (2.0 <= daily_change <= 7.0)
                and (price > 1.09 * vwap)  # equivalent to: VWAP < 0.9434 * price
                and valid_adx_for_breakout(adx_val)
                #and (rsi > 50)
                # and (price >= 0.999 * high_10)
                and (vol >= VOLUME_MULTIPLIER_BREAKOUT * roll_vol)
                # and macd_above_signal_now
                #and (macd_val >= MACD_BULLISH_OFFSET_BREAKOUT)
                and (stoch_val > 50)
                and (stoch_diff > 0)
                # NEW: slope check for bullish
                and slope_okay_for_bullish(combined.loc[cidx])
                # NEW: *increasing* ADX check:
                and is_adx_increasing(combined, cidx, bars_ago=2)
                and is_rsi_increasing(combined, cidx, bars_ago=2)
                # NEW: OBV and ROC check for bullish trend:
                and is_obv_increasing(combined, cidx, bars_ago=2)
                and is_roc_positive(combined, cidx)
                #and is_bullish_breakout_price_action(combined, cidx, daily_change)
                and is_ich_bull
                #and sq_release
            )

            # B) Bullish pullback
            bullish_pullback = (
                (daily_change >= 0.5)
                and (price > 1.01 * vwap)
                and valid_adx_for_pullback(adx_val)
                and macd_above_signal_now
                #and (rsi > 30)
                #and (macd_val >= MACD_BULLISH_OFFSET_PULLBACK)
                # and (price >= 0.98 * sma_20_val)
                and (stoch_val < 60)
                and (stoch_diff > 0)
                # NEW: slope check for bullish (optional commented out)
                and slope_okay_for_bullish(combined.loc[cidx])
                # NEW: *increasing* ADX check (optional commented out)
                # and is_adx_increasing(combined, cidx, bars_ago=2)
                # and is_rsi_increasing(combined, cidx, bars_ago=2)
                # NEW: OBV and ROC check for bullish trend (optional commented out)
                # and is_obv_increasing(combined, cidx, bars_ago=2)
                # and is_roc_positive(combined, cidx)
                and is_bullish_pullback_price_action(combined, cidx, daily_change)
            )

            if bullish_breakout or bullish_pullback:
                signals_detected.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Breakout' if bullish_breakout else 'Pullback',
                    'Trend Type': 'Bullish',
                    'Price': round(row_detect.get('close', 0), 2),
                    'Daily Change %': round(daily_change, 2),
                    'VWAP': round(row_detect.get("VWAP", 0), 2),
                    'ADX': round(adx_val, 2),
                    'MACD': round(macd_val, 2) if macd_val is not None else None,
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })
                existing_signal_ids.add(signal_id)

        else:  # Bearish branch
            # A) Bearish breakdown
            bearish_breakdown = (
                #(daily_change <= -2.0)
                (-7.0 <= daily_change <= -2.0)
                and (price < 0.91 * vwap)  # equivalent to: VWAP > 1.0638 * price
                and valid_adx_for_breakout(adx_val)
                # and (price <= 1.001 * low_10)
                and (vol >= VOLUME_MULTIPLIER_BREAKOUT * roll_vol)
                # and (not macd_above_signal_now)
                #and (macd_val <= MACD_BEARISH_OFFSET_BREAKOUT)
                and (stoch_val < 50)
                and (stoch_diff < 0)
                #and (rsi < 50)
                # NEW: slope check for bearish
                and slope_okay_for_bearish(combined.loc[cidx])
                # NEW: *increasing* ADX check:
                and is_adx_increasing(combined, cidx, bars_ago=2)
                and is_rsi_decreasing(combined, cidx, bars_ago=2)
                # NEW: OBV and ROC check for bearish trend:
                and is_obv_decreasing(combined, cidx, bars_ago=2)
                and is_roc_negative(combined, cidx)
                #and is_bearish_breakdown_price_action(combined, cidx, daily_change)
                and is_ich_bear
                #and sq_release
            )

            # B) Bearish pullback
            bearish_pullback = (
                (daily_change <= -0.5)
                and valid_adx_for_pullback(adx_val)
                and (price < 0.99 * vwap)
                and (not macd_above_signal_now)
                #and (macd_val <= MACD_BEARISH_OFFSET_PULLBACK)
                # and (price <= 1.02 * sma_20_val)
                and (stoch_val > 40)
                and (stoch_diff < 0)
                #and (rsi < 70)
                # NEW: slope check for bearish (optional commented out)
                and slope_okay_for_bearish(combined.loc[cidx])
                # NEW: *increasing* ADX check (optional commented out)
                # and is_adx_increasing(combined, cidx, bars_ago=1)
                # and is_rsi_decreasing(combined, cidx, bars_ago=1)
                # NEW: OBV and ROC check for bearish trend (optional commented out)
                # and is_obv_decreasing(combined, cidx, bars_ago=1)
                # and is_roc_negative(combined, cidx)
                and is_bearish_pullback_price_action(combined, cidx, daily_change)
            )

            if bearish_breakdown or bearish_pullback:
                signals_detected.append({
                    'Ticker': ticker,
                    'date': dt,
                    'Entry Type': 'Breakdown' if bearish_breakdown else 'Pullback',
                    'Trend Type': 'Bearish',
                    'Price': round(row_detect.get('close', 0), 2),
                    'Daily Change %': round(daily_change, 2),
                    'VWAP': round(row_detect.get("VWAP", 0), 2),
                    'ADX': round(adx_val, 2),
                    'MACD': round(macd_val, 2) if macd_val is not None else None,
                    'Signal_ID': signal_id,
                    'logtime': row_detect.get("logtime", datetime.now().isoformat()),
                    'Entry Signal': "Yes"
                })
                existing_signal_ids.add(signal_id)

    return signals_detected

def is_bullish_breakout_price_action(df, idx, daily_change) -> bool:
    """
    Even more relaxed logic for a bullish breakout:
      1. Candle is bullish (close > open).
      2. Candle's body >= 10% of total range (instead of 15–20%).
      3. Candle's upper wick <= 80% of total range (instead of 70–75%).
      4. Candle's close is within 8% of the highest high over the last 10 bars (instead of 4–6%).
      5. Daily change >= +0.2% (lower than +0.3% or +0.5%).
      6. Close >= 92% of its 20-SMA (instead of 93–95%).
      7. Volume >= 50% of the 10-bar average (instead of 60–70%).
      8. Previous candle’s high can be up to 12% above current candle’s high (instead of 5–10%).
      9. Open must not be more than 20% gap down from previous close (instead of 10–15%).
      10. Price >= 96% of its 50-SMA (instead of 97–98%).
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]

    # 1) Must be bullish
    if row['close'] <= row['open']:
        return False

    # 2) Candle body check (>= 10% of total range)
    body = row['close'] - row['open']
    range_ = row['high'] - row['low']
    if range_ <= 0 or body < 0.10 * range_:
        return False

    # 3) Upper wick check (<= 80% of range)
    upper_wick = row['high'] - row['close']
    if upper_wick > 0.80 * range_:
        return False

    # 4) Close near recent high (within 8%)
    if idx < 9:
        return False  # Not enough bars for a 10-bar rolling window
    recent_high = df['high'].rolling(window=10, min_periods=10).max().iloc[idx]
    if recent_high <= 0 or row['close'] < 0.92 * recent_high:
        return False

    # 5) Strength of move
    if daily_change < 0.2:
        return False

    # 6) 20-SMA check
    if idx >= 19:
        ma20 = df['close'].rolling(window=20, min_periods=20).mean().iloc[idx]
        if row['close'] < 0.92 * ma20:
            return False

    # 7) Volume check
    if 'volume' in df.columns and idx >= 9:
        avg_vol = df['volume'].rolling(window=10, min_periods=10).mean().iloc[idx]
        if avg_vol > 0 and row['volume'] < 0.50 * avg_vol:
            return False

    # 8) Previous candle’s high
    if idx > 0:
        prev_high = df.iloc[idx - 1]['high']
        if prev_high > 1.12 * row['high']:
            return False

    # 9) Acceptable gap at open
    if idx > 0:
        prev_close = df.iloc[idx - 1]['close']
        if row['open'] < 0.80 * prev_close:
            return False

    # 10) Broader trend => 50-SMA
    if idx >= 49:
        ma50 = df['close'].rolling(window=50, min_periods=50).mean().iloc[idx]
        if row['close'] < 0.96 * ma50:
            return False

    return True


def is_bearish_breakdown_price_action(df, idx, daily_change) -> bool:
    """
    Even more relaxed logic for a bearish breakdown:
      1. Candle is bearish (close < open).
      2. Candle's body >= 10% of total range.
      3. Candle's lower wick <= 80% of total range.
      4. Candle's close is within 8% of the lowest low over the last 10 bars.
      5. Daily change <= -0.2% (instead of -0.3% or -0.5%).
      6. Candle closes below 105% of its 20-SMA (instead of 103%).
      7. Volume >= 50% of the 10-bar average (instead of 60–70%).
      8. Previous candle’s low can be 15% lower than the current candle’s low (instead of 8–12%).
      9. Open must not be more than 15% gap up from previous close (instead of 10–12%).
      10. Price <= 105% of its 50-SMA (instead of 102–103%).
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]

    # 1) Must be bearish
    if row['close'] >= row['open']:
        return False

    # 2) Candle body check (>= 10% of range)
    body = row['open'] - row['close']
    range_ = row['high'] - row['low']
    if range_ <= 0 or body < 0.10 * range_:
        return False

    # 3) Lower wick check (<= 80% of range)
    lower_wick = row['close'] - row['low']
    if lower_wick > 0.80 * range_:
        return False

    # 4) Close near recent low (within 8%)
    if idx < 9:
        return False
    recent_low = df['low'].rolling(window=10, min_periods=10).min().iloc[idx]
    if recent_low <= 0 or row['close'] > 1.08 * recent_low:
        return False

    # 5) Strength of move
    if daily_change > -0.2:
        return False

    # 6) 20-SMA check
    if idx >= 19:
        ma20 = df['close'].rolling(window=20, min_periods=20).mean().iloc[idx]
        if row['close'] > 1.05 * ma20:
            return False

    # 7) Volume check
    if 'volume' in df.columns and idx >= 9:
        avg_vol = df['volume'].rolling(window=10, min_periods=10).mean().iloc[idx]
        if avg_vol > 0 and row['volume'] < 0.50 * avg_vol:
            return False

    # 8) Previous candle’s low
    if idx > 0:
        prev_low = df.iloc[idx - 1]['low']
        if prev_low < 0.85 * row['low']:
            return False

    # 9) Acceptable gap at open
    if idx > 0:
        prev_close = df.iloc[idx - 1]['close']
        if row['open'] > 1.15 * prev_close:
            return False

    # 10) Broader trend => 50-SMA
    if idx >= 49:
        ma50 = df['close'].rolling(window=50, min_periods=50).mean().iloc[idx]
        if row['close'] > 1.05 * ma50:
            return False

    return True


def is_bullish_pullback_price_action(df, idx, daily_change) -> bool:
    """
    Further relaxed logic for a bullish pullback:
      1. Candle is bullish (close > open).
      2. Lower wick >= 0.5% of total range (instead of 1–2%).
      3. Upper wick <= 95% of total range (looser than 70–90%).
      4. Low is within 20% of the recent 10-bar low (instead of 8–15%).
      5. Daily change >= -0.5% (allows more negative than -0.3%).
      6. Open is within 10% of the candle's low (instead of 5–8%).
      7. Close >= 90% of its 10-SMA (instead of 95–97%).
      8. Volume >= 20% of the 10-bar average (instead of 25–30%).
      9. Price >= 90% of its 50-SMA for a broader uptrend (instead of 95–97%).
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]

    # 1) Must be bullish
    if row['close'] <= row['open']:
        return False

    range_ = row['high'] - row['low']
    if range_ <= 0:
        return False

    # 2) Lower wick check (>= 0.5% of total range)
    lower_wick = row['open'] - row['low']
    if lower_wick < 0.005 * range_:
        return False

    # 3) Upper wick check (<= 95% of total range)
    upper_wick = row['high'] - row['close']
    if upper_wick > 0.95 * range_:
        return False

    # 4) Near support: within 20% of recent 10-bar low
    if idx < 9:
        return False
    recent_low = df['low'].rolling(window=10, min_periods=10).min().iloc[idx]
    if recent_low <= 0 or row['low'] > 1.20 * recent_low:
        return False

    # 5) Daily change can be as low as -0.5%
    if daily_change < -0.5:
        return False

    # 6) Open near low (10% gap)
    if row['open'] > 1.10 * row['low']:
        return False

    # 7) 10-SMA check (>= 90%)
    if idx >= 9:
        ma10 = df['close'].rolling(window=10, min_periods=10).mean().iloc[idx]
        if row['close'] < 0.90 * ma10:
            return False

    # 8) Volume check (>= 20% of avg)
    if 'volume' in df.columns and idx >= 9:
        avg_vol = df['volume'].rolling(window=10, min_periods=10).mean().iloc[idx]
        if avg_vol > 0 and row['volume'] < 0.20 * avg_vol:
            return False

    # 9) Broader uptrend => 50-SMA (>= 90%)
    if idx >= 49:
        ma50 = df['close'].rolling(window=50, min_periods=50).mean().iloc[idx]
        if row['close'] < 0.90 * ma50:
            return False

    return True


def is_bearish_pullback_price_action(df, idx, daily_change) -> bool:
    """
    Further relaxed logic for a bearish pullback:
      1. Candle is bearish (close < open).
      2. Upper wick >= 0.5% of total range (instead of 1–2%).
      3. Lower wick <= 95% of total range (looser than 70–90%).
      4. High is within 20% of the recent 10-bar high (instead of 8–15%).
      5. Daily change <= +0.5% (allows more positive than +0.3%).
      6. Open is within 10% of the candle's high (instead of 5–8%).
      7. Close <= 110% of its 10-SMA (instead of 102–105%).
      8. Volume >= 20% of the 10-bar average (instead of 25–30%).
      9. Price <= 110% of its 50-SMA for a broader downtrend (instead of 102–105%).
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]

    # 1) Must be bearish
    if row['close'] >= row['open']:
        return False

    range_ = row['high'] - row['low']
    if range_ <= 0:
        return False

    # 2) Upper wick check (>= 0.5% of range)
    upper_wick = row['high'] - row['open']
    if upper_wick < 0.005 * range_:
        return False

    # 3) Lower wick check (<= 95% of range)
    lower_wick = row['close'] - row['low']
    if lower_wick > 0.95 * range_:
        return False

    # 4) Near resistance: within 20% of recent 10-bar high
    if idx < 9:
        return False
    recent_high = df['high'].rolling(window=10, min_periods=10).max().iloc[idx]
    if recent_high <= 0 or row['high'] < 0.80 * recent_high:
        return False

    # 5) Daily move can be as high as +0.5%
    if daily_change > 0.5:
        return False

    # 6) Open near high (10% gap)
    if row['open'] < 0.90 * row['high']:
        return False

    # 7) 10-SMA check (<= 110%)
    if idx >= 9:
        ma10 = df['close'].rolling(window=10, min_periods=10).mean().iloc[idx]
        if row['close'] > 1.10 * ma10:
            return False

    # 8) Volume check (>= 20% of avg)
    if 'volume' in df.columns and idx >= 9:
        avg_vol = df['volume'].rolling(window=10, min_periods=10).mean().iloc[idx]
        if avg_vol > 0 and row['volume'] < 0.20 * avg_vol:
            return False

    # 9) Broader downtrend => 50-SMA (<= 110%)
    if idx >= 49:
        ma50 = df['close'].rolling(window=50, min_periods=50).mean().iloc[idx]
        if row['close'] > 1.10 * ma50:
            return False

    return True




'''
def is_bullish_breakout_price_action(df, idx, daily_change) -> bool:
    """
    Less strict Price Action logic for a bullish breakout:
      - The current candle is bullish (close > open).
      - The candle's body is moderately large (at least 40% of the candle's range).
      - The candle's close is near the highest high of the last 10 bars (within 1.5%).
      - The move is strong: daily_change is at least +1.5%.
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]

    # Must be a bullish candle.
    if row['close'] <= row['open']:
        return False

    # Calculate body and range.
    body = row['close'] - row['open']
    range_ = row['high'] - row['low']
    # Relax the body requirement: at least 40% of the range.
    if range_ <= 0 or body < 0.4 * range_:
        return False

    # Get the rolling 10-bar high.
    recent_high = df['high'].rolling(window=10, min_periods=10).max().iloc[idx]
    if recent_high <= 0:
        return False

    # Allow the current close to be within 1.5% of the recent high.
    if row['close'] < 0.985 * recent_high:
        return False

    # Require a moderately strong move.
    if daily_change < 1.5:
        return False

    return True


def is_bearish_breakdown_price_action(df, idx, daily_change) -> bool:
    """
    Less strict Price Action logic for a bearish breakdown:
      - The current candle is bearish (close < open).
      - The candle's body is moderately large (at least 40% of the candle's range).
      - The candle's close is near the lowest low of the last 10 bars (within 1.5%).
      - The move is strong: daily_change is at most -1.5%.
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]

    # Must be a bearish candle.
    if row['close'] >= row['open']:
        return False

    # Calculate body and range.
    body = row['open'] - row['close']
    range_ = row['high'] - row['low']
    if range_ <= 0 or body < 0.4 * range_:
        return False

    # Get the rolling 10-bar low.
    recent_low = df['low'].rolling(window=10, min_periods=10).min().iloc[idx]
    if recent_low <= 0:
        return False

    # Allow the current close to be within 1.5% above the recent low.
    if row['close'] > 1.015 * recent_low:
        return False

    # Require a moderately strong move.
    if daily_change > -1.5:
        return False

    return True


def is_bullish_pullback_price_action(df, idx, daily_change) -> bool:
    """
    Less strict Price Action logic for a bullish pullback:
      - The current candle is bullish (close > open).
      - The candle shows evidence of a pullback with a moderately long lower wick.
      - The candle's low is near the support level of the last 10 bars (within 3%).
      - The daily change is modest positive (at least +0.3%).
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]
    
    # Must be bullish.
    if row['close'] <= row['open']:
        return False

    range_ = row['high'] - row['low']
    if range_ <= 0:
        return False

    # Calculate lower wick length.
    lower_wick = row['open'] - row['low']
    # Relax requirement: lower wick must be at least 40% of the range.
    if lower_wick < 0.4 * range_:
        return False

    # Get the rolling 10-bar low as a proxy for support.
    recent_low = df['low'].rolling(window=10, min_periods=10).min().iloc[idx]
    if recent_low <= 0:
        return False

    # Check if the candle's low is near the support (within 3%).
    if row['low'] > 1.03 * recent_low:
        return False

    if daily_change < 0.3:
        return False

    return True


def is_bearish_pullback_price_action(df, idx, daily_change) -> bool:
    """
    Less strict Price Action logic for a bearish pullback:
      - The current candle is bearish (close < open).
      - The candle shows evidence of a pullback with a moderately long upper wick.
      - The candle's high is near the resistance level of the last 10 bars (within 3%).
      - The daily change is modest negative (at most -0.3%).
    """
    if idx < 0 or idx >= len(df):
        return False

    row = df.iloc[idx]
    
    # Must be bearish.
    if row['close'] >= row['open']:
        return False

    range_ = row['high'] - row['low']
    if range_ <= 0:
        return False

    # Calculate upper wick length.
    upper_wick = row['high'] - row['open']
    if upper_wick < 0.4 * range_:
        return False

    # Get the rolling 10-bar high as a proxy for resistance.
    recent_high = df['high'].rolling(window=10, min_periods=10).max().iloc[idx]
    if recent_high <= 0:
        return False

    # Check if the candle's high is near the resistance (within 3%).
    if row['high'] < 0.97 * recent_high:
        return False

    if daily_change > -0.3:
        return False

    return True
'''





    
'''
def detect_signals_in_memory(
    ticker,
    df_for_rolling,
    df_for_detection,
    existing_signal_ids,
    prev_day_close=None
):
    """
    Revised to tighten up signals and reduce false entries.
    1) df_for_rolling: a bigger dataset (e.g. last 20+ bars) so rolling(10/20) is accurate
    2) df_for_detection: only the final portion (last ~4 bars) for new signals
    3) 'existing_signal_ids' to avoid duplicates
    4) prev_day_close used for daily % change
    """

    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    # -----------------------------------------------------
    # A) If prev_day_close not supplied, fallback:
    # -----------------------------------------------------
    final_bar_date = df_for_detection['date'].iloc[-1]
    if not prev_day_close:
        prev_day_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

    # -----------------------------------------------------
    # B) Combine frames for rolling context
    # -----------------------------------------------------
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------
    # C) Daily change
    # -----------------------------------------------------
    try:
        latest_close = df_for_detection['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except:
        daily_change = 0.0

    # -----------------------------------------------------
    # D) Precompute Price Action columns
    # -----------------------------------------------------
    # (Optional) You can keep bullish_engulfing/bearish_engulfing checks if desired

    # Basic "bullish" or "bearish" candle shape: 
    # e.g. close in top 25% of bar => "Bullish_PriceAction", etc.
    combined["Bullish_PriceAction"] = (
        (combined["close"] > combined["open"]) &
        ((combined["close"] - combined["low"]) > 0.5 * (combined["high"] - combined["low"])) &  # Body >= 50% range
        ((combined["high"] - combined["close"]) < 0.3 * (combined["high"] - combined["low"]))   # close near high
    )
    combined["Bearish_PriceAction"] = (
        (combined["close"] < combined["open"]) &
        ((combined["high"] - combined["close"]) > 0.5 * (combined["high"] - combined["low"])) &  # Body >= 50% range
        ((combined["close"] - combined["low"]) < 0.3 * (combined["high"] - combined["low"]))     # close near low
    )

    # "Pullback" concept: near 20 SMA, requires last bar is a bounce from minor retracement
    combined["Bullish_PullbackPriceAction"] = (
        (combined["close"] > combined["open"]) &
        (combined["low"] >= 0.99 * combined["20_SMA"]) &
        (combined["low"] <= 1.01 * combined["20_SMA"])
    )
    combined["Bearish_PullbackPriceAction"] = (
        (combined["close"] < combined["open"]) &
        (combined["high"] <= 1.01 * combined["20_SMA"]) &
        (combined["high"] >= 0.99 * combined["20_SMA"])
    )

    # -----------------------------------------------------
    # E) Rolling columns for breakout checks
    # -----------------------------------------------------
    rolling_window = 10
    combined["rolling_high_10"] = (
        combined["high"].rolling(rolling_window, min_periods=rolling_window).max()
    )
    combined["rolling_low_10"] = (
        combined["low"].rolling(rolling_window, min_periods=rolling_window).min()
    )
    combined["rolling_vol_10"] = (
        combined["volume"].rolling(rolling_window, min_periods=rolling_window).mean()
    )

    # Stochastic difference
    combined["StochDiff"] = combined["Stochastic"].diff()

    # We may also want slope of 20 SMA for trend confirmation:
    combined["SMA20_Slope"] = combined["20_SMA"].diff()

    # -----------------------------------------------------
    # F) MACD crossing logic
    # -----------------------------------------------------
    def is_macd_above(row_idx):
        if row_idx < 0 or row_idx >= len(combined):
            return False
        macd_val = combined.loc[row_idx, "MACD"]
        signal_val = combined.loc[row_idx, "Signal_Line"]
        if pd.isna(macd_val) or pd.isna(signal_val):
            return False
        return (macd_val > signal_val)

    def macd_cross_up(row_idx):
        if row_idx <= 0 or row_idx >= len(combined):
            return False
        macd_now    = combined.loc[row_idx, "MACD"]
        signal_now  = combined.loc[row_idx, "Signal_Line"]
        macd_prev   = combined.loc[row_idx-1, "MACD"]
        signal_prev = combined.loc[row_idx-1, "Signal_Line"]
        if any(pd.isna(x) for x in [macd_now, signal_now, macd_prev, signal_prev]):
            return False
        return (macd_prev <= signal_prev) and (macd_now > signal_now)

    # Basic up-/down-trend check (RSI & VWAP from the original code):
    def get_trend_type(row):
        rsi_val = row.get("RSI", 0)
        cls_val = row.get("close", 0)
        vwap_val = row.get("VWAP", 0)
        # You can refine or skip VWAP checks if you want.
        if (rsi_val > 25) and (cls_val > 1.005 * vwap_val):
            return "Bullish"
        elif (rsi_val < 75) and (cls_val < 0.995 * vwap_val):
            return "Bearish"
        return None

    def valid_adx(val):
        # We want stronger trends => raise from >25 to >30 or so
        if pd.isna(val):
            return False
        return (val >= 30)  # No upper bound forced here

    volume_multiplier = 1.5
    macd_extreme_upper_limit = 50
    macd_extreme_lower_limit = -50

    # Adjust offsets if needed
    MACD_BULLISH_OFFSET = 2   # e.g. want MACD > 2 for stronger signals
    MACD_BEARISH_OFFSET = -2

    # -----------------------------------------------------
    # G) Loop over detection bars only
    # -----------------------------------------------------
    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect["date"]
        match = combined[combined["date"] == dt]
        if match.empty:
            continue
        cidx = match.index[0]

        adx_val = combined.loc[cidx, "ADX"]
        # Trend must be sufficiently strong
        if not valid_adx(adx_val):
            continue

        # Basic up/down trend or skip
        row_trend = get_trend_type(combined.loc[cidx])
        if not row_trend:
            continue

        # Confirm 20SMA slope for directional trades
        # For bullish: 20 SMA slope > 0, for bearish: slope < 0
        slope_20 = combined["SMA20_Slope"].iloc[cidx]
        if row_trend == "Bullish" and (slope_20 <= 0):
            continue
        if row_trend == "Bearish" and (slope_20 >= 0):
            continue

        # Check prior bar for "pullback" scenario
        bullish_pullback_check = False
        bearish_pullback_check = False
        if cidx > 0:
            # For bullish pullback, previous bar was a mild negative candle
            if (combined.loc[cidx-1, "close"] < combined.loc[cidx-1, "open"]):
                bullish_pullback_check = True
            # For bearish pullback, previous bar was a mild positive candle
            if (combined.loc[cidx-1, "close"] > combined.loc[cidx-1, "open"]):
                bearish_pullback_check = True

        macd_above_signal_now = is_macd_above(cidx)
        # crossing_up_now = macd_cross_up(cidx) # optional usage

        # *** Refined thresholds *** 
        # For ~2% target/SL, we want a decent daily move. 
        # Example: for bullish pullback => daily_change > +1.0
        # for bullish breakout => daily_change between +2 and +12 (some upper bound to skip extremes)
        # Adjust as per your testing:
        bullish_conditions = {
            "Pullback": (
                (daily_change >= 1.0) &
                (combined.loc[cidx, "Bullish_PullbackPriceAction"]) &
                bullish_pullback_check &
                macd_above_signal_now &
                (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSET) &
                (combined.loc[cidx, "Stochastic"] < 60) &
                (combined.loc[cidx, "StochDiff"] > 0) &
                (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"]) &
                # optional: ensure candle is not too small
                ((combined.loc[cidx, "high"] - combined.loc[cidx, "low"]) > 0.005 * combined.loc[cidx, "low"])
            ),
            "Breakout": (
                (combined.loc[cidx, "close"] >= 0.999 * combined.loc[cidx, "rolling_high_10"]) &
                (2.0 <= daily_change <= 12.0) &
                (combined.loc[cidx, "volume"] >= volume_multiplier * combined.loc[cidx, "rolling_vol_10"]) &
                macd_above_signal_now &
                (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSET) &
                (combined.loc[cidx, "MACD"] < macd_extreme_upper_limit) &
                (combined.loc[cidx, "Bullish_PriceAction"]) &
                (combined.loc[cidx, "Stochastic"] > 60) &
                (combined.loc[cidx, "StochDiff"] > 0) &
                (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
            )
        }

        bearish_conditions = {
            "Pullback": (
                (daily_change <= -1.0) &
                (combined.loc[cidx, "Bearish_PullbackPriceAction"]) &
                bearish_pullback_check &
                (not macd_above_signal_now) &
                (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSET) &
                (combined.loc[cidx, "Stochastic"] > 40) &
                (combined.loc[cidx, "StochDiff"] < 0) &
                (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
            ),
            "Breakdown": (
                (combined.loc[cidx, "close"] <= 1.001 * combined.loc[cidx, "rolling_low_10"]) &
                (-12.0 <= daily_change <= -2.0) &
                (combined.loc[cidx, "volume"] >= volume_multiplier * combined.loc[cidx, "rolling_vol_10"]) &
                (not macd_above_signal_now) &
                (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSET) &
                (combined.loc[cidx, "MACD"] > macd_extreme_lower_limit) &
                (combined.loc[cidx, "Bearish_PriceAction"]) &
                (combined.loc[cidx, "Stochastic"] < 40) &
                (combined.loc[cidx, "StochDiff"] < 0) &
                (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
            ),
        }

        # Decide which dictionary to apply
        chosen_dict = bullish_conditions if (row_trend == "Bullish") else bearish_conditions

        for entry_type, condition_bool in chosen_dict.items():
            if condition_bool:
                # Build a unique signal id
                sig_id = row_detect.get("Signal_ID")
                if not sig_id:
                    dt_str = row_detect["date"].isoformat()
                    sig_id = f"{ticker}-{dt_str}"

                # Avoid duplicates
                if sig_id in existing_signal_ids:
                    continue

                signals_detected.append({
                    "Ticker": ticker,
                    "date": row_detect["date"],
                    "Entry Type": entry_type,
                    "Trend Type": row_trend,
                    "Price": round(row_detect.get("close", 0), 2),
                    "Daily Change %": round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    "ADX": round(adx_val, 2),
                    "MACD": row_detect.get("MACD", None),
                    "Signal_ID": sig_id,
                    "logtime": row_detect.get("logtime", datetime.now(india_tz).isoformat()),
                    "Entry Signal": "Yes"
                })

    return signals_detected


def detect_signals_in_memory(
    ticker,
    df_for_rolling,
    df_for_detection,
    existing_signal_ids,
    prev_day_close=None
):
    """
    1) df_for_rolling: a bigger dataset (e.g. last 20+ bars) so rolling(10) is accurate
    2) df_for_detection: only the final portion (last 4 bars) for new signals
    3) We do rolling computations (like rolling high/low) on df_for_rolling
       but only "detect signals" in the last 4 bars.
    """
    signals_detected = []
    if df_for_detection.empty or df_for_rolling.empty:
        return signals_detected

    # (A) Handling prev_day_close if not supplied
    final_bar_date = df_for_detection['date'].iloc[-1]
    if not prev_day_close:
        prev_day_close = robust_prev_day_close_for_ticker(ticker, final_bar_date)

    # (B) Combine frames for rolling context
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined.sort_values('date', inplace=True)

    # (C) daily_change based on final detection bar
    try:
        latest_close = df_for_detection['close'].iloc[-1]
        if prev_day_close and prev_day_close > 0:
            daily_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        else:
            daily_change = 0.0
    except:
        daily_change = 0.0

    combined = combined.copy()

    # (D) Precompute Price Action columns
    combined["Bullish_Engulfing"] = is_bullish_engulfing(combined)
    combined["Bearish_Engulfing"] = is_bearish_engulfing(combined)

    combined["Bullish_PriceAction"] = (
        (combined["high"] > combined["high"].shift(1) * 1.003)
        & (combined["low"] > combined["low"].shift(1) * 1.003)
        & ((combined["close"] - combined["open"]) > 0.002 * combined["open"])
        & ((combined["high"] - combined["close"]) < 0.25 * (combined["high"] - combined["low"]))
    )
    combined["Bearish_PriceAction"] = (
        (combined["high"] < combined["high"].shift(1) * 0.997)
        & (combined["low"] < combined["low"].shift(1) * 0.997)
        & ((combined["open"] - combined["close"]) > 0.002 * combined["open"])
        & ((combined["close"] - combined["low"]) < 0.25 * (combined["high"] - combined["low"]))
    )

    # Pullback requires referencing shift(1); handle if cidx==0 or missing
    combined["Bullish_PullbackPriceAction"] = (
        (combined["close"] > combined["open"])
        & (combined["low"] >= combined["20_SMA"] * 0.99)
        & (combined["low"] <= combined["20_SMA"] * 1.01)
    )
    combined["Bearish_PullbackPriceAction"] = (
        (combined["close"] < combined["open"])
        & (combined["high"] <= combined["20_SMA"] * 1.01)
        & (combined["high"] >= combined["20_SMA"] * 0.99)
    )

    # (E) Rolling columns (with min_periods=10 to avoid NaN)
    rolling_window = 10
    combined["rolling_high_10"] = (
        combined["high"].rolling(rolling_window, min_periods=rolling_window).max()
    )
    combined["rolling_low_10"]  = (
        combined["low"].rolling(rolling_window, min_periods=rolling_window).min()
    )
    combined["rolling_vol_10"]  = (
        combined["volume"].rolling(rolling_window, min_periods=rolling_window).mean()
    )

    # (F) Stochastic difference
    combined["StochDiff"] = combined["Stochastic"].diff()

    combined.reset_index(drop=True, inplace=True)

    # (G) MACD crossing logic
    def is_macd_above(row_idx):
        if row_idx < 0 or row_idx >= len(combined):
            return False
        macd_val    = combined.loc[row_idx, "MACD"]
        signal_val  = combined.loc[row_idx, "Signal_Line"]
        if pd.isna(macd_val) or pd.isna(signal_val):
            return False
        return macd_val > signal_val

    def macd_cross_up(row_idx):
        if row_idx <= 0 or row_idx >= len(combined):
            return False
        macd_now    = combined.loc[row_idx, "MACD"]
        signal_now  = combined.loc[row_idx, "Signal_Line"]
        macd_prev   = combined.loc[row_idx-1, "MACD"]
        signal_prev = combined.loc[row_idx-1, "Signal_Line"]
        if (
            pd.isna(macd_now) or pd.isna(signal_now)
            or pd.isna(macd_prev) or pd.isna(signal_prev)
        ):
            return False
        return (macd_prev <= signal_prev) and (macd_now > signal_now)

    def get_trend_type(row):
        rsi_val = row.get("RSI", 0)
        cls_val = row.get("close", 0)
        vwap_val = row.get("VWAP", 0)

        if (rsi_val > 20) and (cls_val > vwap_val * 1.01):
            return "Bullish"
        elif (rsi_val < 80) and (cls_val < vwap_val * 0.99):
            return "Bearish"
        return None

    def valid_adx(val):
        if pd.isna(val):
            return False
        return (val > 25)

    volume_multiplier = 1.5
    macd_extreme_upper_limit = 50
    macd_extreme_lower_limit = -50
    MACD_BULLISH_OFFSET = 3
    MACD_BULLISH_OFFSETp = 1
    MACD_BEARISH_OFFSET = -3
    MACD_BEARISH_OFFSETp = -1

    # (H) Loop over detection bars only
    for idx_detect, row_detect in df_for_detection.iterrows():
        dt = row_detect["date"]
        match = combined[combined["date"] == dt]
        if match.empty:
            continue
        cidx = match.index[0]

        adx_val = combined.loc[cidx, "ADX"]
        if not valid_adx(adx_val):
            continue

        row_trend = get_trend_type(combined.loc[cidx])
        if not row_trend:
            continue

        # Pullback checks
        bullish_pullback_check = False
        if cidx > 0:
            if (
                not pd.isna(combined.loc[cidx-1, "close"])
                and not pd.isna(combined.loc[cidx-1, "open"])
                and (combined.loc[cidx-1, "close"] < combined.loc[cidx-1, "open"])
            ):
                bullish_pullback_check = True

        bearish_pullback_check = False
        if cidx > 0:
            if (
                not pd.isna(combined.loc[cidx-1, "close"])
                and not pd.isna(combined.loc[cidx-1, "open"])
                and (combined.loc[cidx-1, "close"] > combined.loc[cidx-1, "open"])
            ):
                bearish_pullback_check = True

        macd_above_signal_now = is_macd_above(cidx)

        bullish_conditions = {
            "Pullback": (
                (combined.loc[cidx, "close"] >= combined.loc[cidx, "open"])
                and (daily_change > 0.5)
                and (combined.loc[cidx, "low"] >= combined.loc[cidx, "VWAP"] * 0.98)
                and macd_above_signal_now
                and (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSETp)
                and (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] < 55)
                and (combined.loc[cidx, "StochDiff"] > 0)
                and (adx_val > 25)
                #and bullish_pullback_check
                and (combined.loc[cidx, "Bullish_PullbackPriceAction"])
            ),
            "Breakout": (
                (combined.loc[cidx, "close"] > (combined.loc[cidx, "rolling_high_10"] * 0.998))
                and (10 > daily_change > 1.5)
                and (combined.loc[cidx, "volume"] > volume_multiplier * combined.loc[cidx, "rolling_vol_10"])
                and macd_above_signal_now
                and (combined.loc[cidx, "MACD"] > MACD_BULLISH_OFFSET)
                and (combined.loc[cidx, "MACD"] < macd_extreme_upper_limit)
                and (combined.loc[cidx, "close"] > combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] > 50)
                and (combined.loc[cidx, "StochDiff"] > 0)
                and (adx_val > 35 and adx_val < 45)
                and (combined.loc[cidx, "Bullish_PriceAction"])
            ),
        }

        bearish_conditions = {
            "Pullback": (
                (combined.loc[cidx, "close"] <= combined.loc[cidx, "open"])
                and (daily_change < -0.5)
                and (combined.loc[cidx, "high"] <= combined.loc[cidx, "VWAP"] * 1.02)
                and not macd_above_signal_now
                and (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSETp)
                and (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] > 45)
                and (combined.loc[cidx, "StochDiff"] < 0)
                and (adx_val > 25)
                #and bearish_pullback_check
                and (combined.loc[cidx, "Bearish_PullbackPriceAction"])
            ),
            "Breakdown": (
                (combined.loc[cidx, "close"] < (combined.loc[cidx, "rolling_low_10"] * 1.002))
                and (-10 < daily_change < -1.5)
                and (combined.loc[cidx, "volume"] > volume_multiplier * combined.loc[cidx, "rolling_vol_10"])
                and not macd_above_signal_now
                and (combined.loc[cidx, "MACD"] < MACD_BEARISH_OFFSET)
                and (combined.loc[cidx, "MACD"] > macd_extreme_lower_limit)
                and (combined.loc[cidx, "close"] < combined.loc[cidx, "20_SMA"])
                and (combined.loc[cidx, "Stochastic"] < 50)
                and (combined.loc[cidx, "StochDiff"] < 0)
                and (adx_val > 35 and adx_val < 45)
                and (combined.loc[cidx, "Bearish_PriceAction"])
            ),
        }

        chosen_dict = bullish_conditions if (row_trend == "Bullish") else bearish_conditions

        for entry_type, condition_bool in chosen_dict.items():
            if condition_bool:
                sig_id = row_detect.get("Signal_ID")
                if not sig_id:
                    dt_str = row_detect["date"].isoformat()
                    sig_id = f"{ticker}-{dt_str}"
                if sig_id in existing_signal_ids:
                    continue

                signals_detected.append({
                    "Ticker": ticker,
                    "date": row_detect["date"],
                    "Entry Type": entry_type,
                    "Trend Type": row_trend,
                    "Price": round(row_detect.get("close", 0), 2),
                    "Daily Change %": round(daily_change, 2),
                    "VWAP": round(row_detect.get("VWAP", 0), 2),
                    "ADX": round(adx_val, 2),
                    "MACD": row_detect.get("MACD", None),
                    "Signal_ID": sig_id,
                    "logtime": row_detect.get("logtime", datetime.now(india_tz).isoformat()),
                    "Entry Signal": "Yes"
                })

    return signals_detected
'''
###########################################################
# 5) Mark signals in main CSV => CalcFlag='Yes'
###########################################################
def mark_signals_in_main_csv(ticker, signals_list, main_path, tz_obj):
    """
    Updates 'main_indicators_updated' CSV for a given ticker:
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
    across all *_main_indicators_updated.csv.
    Intraday timeframe considered is 09:25 to 14:05 (example).
    """
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(14, 46)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators_updated.csv')
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
        ticker = os.path.basename(file_path).replace('_main_indicators_updated.csv', '').upper()
        df_full = load_and_normalize_csv(file_path, tz='Asia/Kolkata')
        if df_full.empty:
            return []

        # Get prev_day_close
        prev_day_close = find_prev_day_close_for_ticker(df_full, target_date)

        # Only analyze the portion for our target_date
        df_today = df_full[(df_full['date'] >= st) & (df_full['date'] <= et)].copy()

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,   # entire dataset for correct rolling(10)
            df_for_detection=df_today,# final portion for new signals
            existing_signal_ids=existing_ids,
            prev_day_close=prev_day_close
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
    end_dt = india_tz.localize(datetime.combine(target_date, dt_time(17, 30)))
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
    """
    try:
        # Convert user-specified string to a date
        requested_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # Force EXACT date instead of skipping weekends/holidays
        last_trading_day = requested_date

        date_label = last_trading_day.strftime('%Y-%m-%d')
        papertrade_file = f"papertrade_{date_label}.csv"

        # 1) Detect signals
        raw_signals_df = find_price_action_entries_for_date(last_trading_day)
        if raw_signals_df.empty:
            logging.info(f"No signals found for {date_label}.")
            return

        # 2) Mark signals in main CSV
        signals_grouped = raw_signals_df.groupby('Ticker')
        for ticker, group in signals_grouped:
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_updated.csv")
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
    # If multiple dates are provided on the command line, we'll iterate over them.
    # Example usage:
    #   python script.py 2025-01-14 2025-01-15 2025-01-16
    date_args = sys.argv[1:]
    if not date_args:
        # If no args given, define a list of back-dates here:
        date_args = [
            "2025-01-27", "2025-01-28", "2025-01-29",
            "2025-01-30", "2025-01-31",
            "2025-02-03", "2025-02-04", "2025-02-05", 
            "2025-02-07"
        ]
        #date_args = ["2025-02-04"]

    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
