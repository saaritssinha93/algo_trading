# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:15:21 2025

Author: Saarit

This script filters a large list of NSE stocks using
Zerodha's KiteConnect historical data for daily candles.
Generates three files:
  - et4_filtered_stocks_bullish.py        (Bullish watchlist)
  - et4_filtered_stocks_bearish.py        (Bearish watchlist)
  - et4_filtered_stocks.py                (Combined watchlist)
"""

import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from kiteconnect import KiteConnect
from logging.handlers import TimedRotatingFileHandler
import math

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    "logs\\filter.log", when="M", interval=30,
    backupCount=5, delay=True
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

def setup_kite_session():
    """
    Setup the KiteConnect session using saved API keys and access tokens.
    """
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()
        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        print("Kite Connect session initialized successfully.")
        return kite
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

kite = setup_kite_session()

# Define Market Holidays for 2024 (no trading occurs on these dates).
market_holidays = [
    datetime(2024, 1, 26).date(),
    datetime(2024, 3, 8).date(),
    datetime(2024, 3, 25).date(),
    datetime(2024, 3, 29).date(),
    datetime(2024, 4, 11).date(),
    datetime(2024, 4, 17).date(),
    datetime(2024, 5, 1).date(),
    datetime(2024, 6, 17).date(),
    datetime(2024, 7, 17).date(),
    datetime(2024, 8, 15).date(),
    datetime(2024, 10, 2).date(),
    datetime(2024, 11, 1).date(),
    datetime(2024, 11, 15).date(),
    datetime(2024, 11, 20).date(),
    datetime(2024, 12, 25).date(),
]

# --------------------------------------------------------------------------
# Instrument Cache to avoid multiple calls for each symbol
# --------------------------------------------------------------------------
def build_instrument_cache(kite, exchange="NSE"):
    """
    Fetch all instruments for an exchange once,
    and return a dict {tradingsymbol: instrument_token}.
    """
    cache = {}
    try:
        all_instruments = kite.instruments(exchange=exchange)
        for inst in all_instruments:
            cache[inst["tradingsymbol"].upper()] = inst["instrument_token"]
    except Exception as e:
        logger.error(f"Error fetching instruments for {exchange}: {e}")
    return cache

instrument_cache = build_instrument_cache(kite, "NSE")

def get_instrument_token_from_cache(symbol):
    """
    Return the instrument token from the global cache if it exists, else None.
    """
    return instrument_cache.get(symbol.upper())

# Import your large list of stocks (1225 in your example)
from et4_filtered_stocks_market_cap1 import shares

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def calculate_rsi(df, period=14):
    """
    Calculate RSI; skip if length mismatch is encountered.
    """
    try:
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = (-1 * delta.clip(upper=0)).abs()
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return df
    return df

def calculate_macd(df, span_short=12, span_long=26, span_signal=9):
    """
    Calculate MACD; skip if length mismatch is encountered.
    """
    try:
        df['ema_short'] = df['close'].ewm(span=span_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=span_long, adjust=False).mean()
        df['macd'] = df['ema_short'] - df['ema_long']
        df['macd_signal'] = df['macd'].ewm(span=span_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return df
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands; skip if length mismatch is encountered.
    """
    try:
        df['sma'] = df['close'].rolling(window).mean()
        df['std'] = df['close'].rolling(window).std()
        df['upper_band'] = df['sma'] + (df['std'] * num_std)
        df['lower_band'] = df['sma'] - (df['std'] * num_std)
    except Exception as e:
        logger.error(f"Bollinger calculation error: {e}")
        return df
    return df

def detect_volume_spike(df, window=20, multiplier=2):
    """
    True if latest volume > multiplier * rolling avg volume.
    """
    try:
        avg_volume = df['volume'].rolling(window).mean().iloc[-1]
        latest_volume = df['volume'].iloc[-1]
        return latest_volume > (avg_volume * multiplier)
    except:
        return False

def detect_pullback(df, retracement=0.618):
    """
    True if current price is <= (recent high - retracement * (recent_high - recent_low)).
    """
    try:
        recent_high = df['close'].max()
        recent_low = df['close'].min()
        target_price = recent_high - (retracement * (recent_high - recent_low))
        current_price = df['close'].iloc[-1]
        return current_price <= target_price
    except:
        return False
    
    
def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Simple Stochastic Oscillator calculation:
    stoch_k = 100 * ((Close - LowestLow) / (HighestHigh - LowestLow)) over k_period
    stoch_d = stoch_k rolling mean over d_period
    """
    try:
        df['lowest_low'] = df['low'].rolling(k_period).min()
        df['highest_high'] = df['high'].rolling(k_period).max()
        df['stoch_k'] = 100 * ( (df['close'] - df['lowest_low']) /
                                (df['highest_high'] - df['lowest_low']) )
        df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()
    except Exception as e:
        logger.error(f"Stochastic calc error: {e}")
    return df


logger = logging.getLogger()

def calculate_true_range(df):
    """
    True Range (TR):
      max( high - low,
           abs(high - prev_close),
           abs(low  - prev_close) )
    """
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
    df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
    return df[['H-L','H-PC','L-PC']].max(axis=1)

def calculate_plus_dm(df):
    """
    +DM:
      if (high[t] - high[t-1]) > (low[t-1] - low[t]) and (high[t] - high[t-1]) > 0 => up_move
      else 0
    """
    plus_dm = [0.0]
    for i in range(1, len(df)):
        up_move = df['high'].iloc[i] - df['high'].iloc[i-1]
        down_move = df['low'].iloc[i-1] - df['low'].iloc[i]
        if (up_move > down_move) and (up_move > 0):
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)
    return pd.Series(plus_dm, index=df.index)

def calculate_minus_dm(df):
    """
    -DM:
      if (low[t-1] - low[t]) > (high[t] - high[t-1]) and (low[t-1] - low[t]) > 0 => down_move
      else 0
    """
    minus_dm = [0.0]
    for i in range(1, len(df)):
        up_move = df['high'].iloc[i] - df['high'].iloc[i-1]
        down_move = df['low'].iloc[i-1] - df['low'].iloc[i]
        if (down_move > up_move) and (down_move > 0):
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)
    return pd.Series(minus_dm, index=df.index)

def wilder_smooth(series, period):
    """
    Wilder's smoothing function for a Pandas Series.
    smoothed[period-1] = sum of first 'period' raw values
    Then for t >= period:
      smoothed[t] = ( smoothed[t-1] - (smoothed[t-1]/period) ) + raw[t]
    Returns a new Series of same length.
    """
    result = [None]*len(series)
    # initial sum for the first 'period' points
    if len(series) < period:
        return pd.Series(result, index=series.index)

    first_val = series.iloc[0:period].sum()
    result[period-1] = first_val

    for i in range(period, len(series)):
        prev = result[i-1]
        raw_val = series.iloc[i]
        result[i] = prev - (prev/period) + raw_val

    return pd.Series(result, index=series.index)

def calculate_adx(df, period=14):
    """
    Welles Wilder's ADX full implementation with Wilder's smoothing.
    """
    # 1) True Range
    df['TR'] = calculate_true_range(df)

    # 2) +DM, -DM
    df['+DM'] = calculate_plus_dm(df)
    df['-DM'] = calculate_minus_dm(df)

    # 3) Wilder-smooth TR, +DM, -DM over 'period'
    df['TR_smooth'] = wilder_smooth(df['TR'], period)
    df['+DM_smooth'] = wilder_smooth(df['+DM'], period)
    df['-DM_smooth'] = wilder_smooth(df['-DM'], period)

    # 4) +DI, -DI
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    # 5) DX
    df['DX'] = 100 * ( (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI']) )

    # 6) ADX = Wilder-smooth DX
    # Typically you might do: 
    # first 14 DX values => average => ADX(14) => then wilder smoothing 
    # We'll do a direct approach:
    df['ADX_smooth'] = wilder_smooth(df['DX'], period)
    # dividing by period to approximate the average:
    df['ADX'] = df['ADX_smooth'] / period

    return df

def calculate_adx_all(df, period=14):
    """
    Convenience function to attach ADX, +DI, -DI columns to df.
    """
    try:
        df = calculate_adx(df, period=period)
    except Exception as e:
        logger.error(f"ADX calc error: {e}")
    return df


# --------------------------------------------------------------------------
# Bullish Filter
# --------------------------------------------------------------------------
def filter_stocks_bullish(kite, symbols):
    """
    Filter logic for bullish stocks:
      1) Min volume
      2) Price range
      3) ATR ratio
      4) 20-SMA > 50-SMA
      5) RSI < 75
      6) MACD > MACD Signal
      ...
    """
    selected = []

    MIN_AVG_VOLUME = 50000
    MIN_PRICE = 50
    MAX_PRICE = 25000
    MIN_ATR_RATIO = 0.01
    RSI_THRESHOLD = 75

    today = datetime.now().date()
    from_date = today - timedelta(days=150)
    to_date = today

    total_stocks = len(symbols)
    print(f"Analyzing {total_stocks} stocks for bullish filter...")
    logger.info(f"Analyzing {total_stocks} stocks for bullish filter...")

    for sym in tqdm(symbols, desc="Filtering stocks (Bullish)"):
        try:
            token = get_instrument_token_from_cache(sym)
            if not token:
                logger.debug(f"{sym} SKIP: No instrument token (NSE?)")
                continue

            time.sleep(0.2)  # throttle to avoid rate-limit

            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            if not data or len(data) < 50:
                logger.debug(f"{sym} SKIP: insufficient data (<50 candles)")
                continue

            df = pd.DataFrame(data).dropna()
            if len(df) < 20:
                logger.debug(f"{sym} SKIP: cannot compute 20-day volume average")
                continue

            # 1) Volume
            avg_vol_20 = df['volume'].tail(20).mean()
            if avg_vol_20 < MIN_AVG_VOLUME:
                logger.debug(f"{sym} SKIP: volume={avg_vol_20:.0f} < {MIN_AVG_VOLUME}")
                continue

            # 2) Price range
            last_close = df['close'].iloc[-1]
            if not (MIN_PRICE <= last_close <= MAX_PRICE):
                logger.debug(f"{sym} SKIP: last_close={last_close} not in range")
                continue

            # 3) ATR ratio
            df['H-L'] = df['high'] - df['low']
            df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
            df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(14).mean()
            latest_atr = df['ATR'].iloc[-1] if not math.isnan(df['ATR'].iloc[-1]) else 0
            if latest_atr <= 0:
                logger.debug(f"{sym} SKIP: ATR<=0")
                continue
            atr_ratio = latest_atr / last_close
            if atr_ratio < MIN_ATR_RATIO:
                logger.debug(f"{sym} SKIP: ATR ratio {atr_ratio:.2f} < {MIN_ATR_RATIO}")
                continue

            # 4) 20-SMA > 50-SMA
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            if pd.isna(df['sma20'].iloc[-1]) or pd.isna(df['sma50'].iloc[-1]):
                logger.debug(f"{sym} SKIP: SMA NaN")
                continue
            if df['sma20'].iloc[-1] <= df['sma50'].iloc[-1]:
                logger.debug(f"{sym} SKIP: 20-SMA <= 50-SMA")
                continue

            # 5) RSI < 75
            df = calculate_rsi(df, period=14)
            if df['rsi'].iloc[-1] >= RSI_THRESHOLD:
                logger.debug(f"{sym} SKIP: RSI over {RSI_THRESHOLD}")
                continue

            # 6) MACD > MACD Signal
            df = calculate_macd(df)
            if df['macd'].iloc[-1] <= df['macd_signal'].iloc[-1]:
                logger.debug(f"{sym} SKIP: MACD <= Signal")
                continue

            # If passes all checks:
            selected.append(sym)

        except Exception as e:
            logger.error(f"{sym} ERROR: {e}")
            continue

    print(f"Completed bullish filter. {len(selected)} selected out of {total_stocks} analyzed.")
    logger.info(f"Bullish filter done. {len(selected)} of {total_stocks} selected.")
    return selected

def filter_and_save_stocks():
    """
    Run the bullish filter and save to et4_filtered_stocks_bullish.py
    as `selected_stocks = [...]`.
    """
    symbols_analyzed = shares
    bullish_list = filter_stocks_bullish(kite, symbols_analyzed)

    out_file = "et4_filtered_stocks_bullish.py"
    try:
        with open(out_file, "w") as f:
            f.write("# This file is auto-generated by filter_and_save_stocks()\n")
            f.write("selected_stocks = [\n")
            for s in bullish_list:
                f.write(f"    '{s}',\n")
            f.write("]\n")
        logger.warning(f"{len(bullish_list)} stocks saved to {out_file}")
        print(f"{len(bullish_list)} stocks saved to {out_file}")
    except Exception as e:
        logger.error(f"Error writing to file {out_file}: {e}")

# --------------------------------------------------------------------------
# Bearish Filter
# --------------------------------------------------------------------------
def filter_stocks_bearish(kite, symbols):
    """
    Bearish filter. Similar approach to bullish, but with 20-SMA<50-SMA,
    RSI>25, MACD<Signal, etc.
    """
    selected = []

    MIN_AVG_VOLUME = 50000
    MIN_PRICE = 50
    MAX_PRICE = 25000
    MIN_ATR_RATIO = 0.01
    RSI_THRESHOLD = 25

    today = datetime.now().date()
    from_date = today - timedelta(days=150)
    to_date = today

    total_stocks = len(symbols)
    print(f"Analyzing {total_stocks} stocks for bearish filter...")
    logger.info(f"Analyzing {total_stocks} stocks for bearish filter...")

    for sym in tqdm(symbols, desc="Filtering stocks (Bearish)"):
        try:
            token = get_instrument_token_from_cache(sym)
            if not token:
                logger.debug(f"{sym} SKIP: No instrument token (NSE?)")
                continue

            time.sleep(0.2)

            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            if not data or len(data) < 50:
                logger.debug(f"{sym} SKIP: Not enough data (<50 candles)")
                continue

            df = pd.DataFrame(data).dropna()
            if len(df) < 20:
                logger.debug(f"{sym} SKIP: cannot compute 20-day volume average")
                continue

            # 1) Volume
            avg_vol_20 = df['volume'].tail(20).mean()
            if avg_vol_20 < MIN_AVG_VOLUME:
                logger.debug(f"{sym} SKIP: volume={avg_vol_20:.0f} < {MIN_AVG_VOLUME}")
                continue

            # 2) Price range
            last_close = df['close'].iloc[-1]
            if not (MIN_PRICE <= last_close <= MAX_PRICE):
                logger.debug(f"{sym} SKIP: last_close={last_close} not in range")
                continue

            # 3) ATR ratio
            df['H-L'] = df['high'] - df['low']
            df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
            df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(14).mean()
            latest_atr = df['ATR'].iloc[-1] if not math.isnan(df['ATR'].iloc[-1]) else 0
            if latest_atr <= 0:
                logger.debug(f"{sym} SKIP: ATR<=0")
                continue
            atr_ratio = latest_atr / last_close
            if atr_ratio < MIN_ATR_RATIO:
                logger.debug(f"{sym} SKIP: ATR ratio {atr_ratio:.2f} < {MIN_ATR_RATIO}")
                continue

            # 4) 20-SMA < 50-SMA
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            if pd.isna(df['sma20'].iloc[-1]) or pd.isna(df['sma50'].iloc[-1]):
                logger.debug(f"{sym} SKIP: SMA NaN")
                continue
            if df['sma20'].iloc[-1] >= df['sma50'].iloc[-1]:
                logger.debug(f"{sym} SKIP: 20-SMA >= 50-SMA")
                continue

            # 5) RSI > 25
            df = calculate_rsi(df, period=14)
            if df['rsi'].iloc[-1] <= RSI_THRESHOLD:
                logger.debug(f"{sym} SKIP: RSI below or eq {RSI_THRESHOLD}")
                continue

            # 6) MACD < MACD Signal
            df = calculate_macd(df)
            if df['macd'].iloc[-1] >= df['macd_signal'].iloc[-1]:
                logger.debug(f"{sym} SKIP: MACD >= Signal")
                continue

            # If passes all checks:
            selected.append(sym)

        except Exception as e:
            logger.error(f"{sym} ERROR: {e}")
            continue

    print(f"Completed bearish filter. {len(selected)} selected out of {total_stocks} analyzed.")
    logger.info(f"Bearish filter done. {len(selected)} of {total_stocks} selected.")
    return selected

def filter_and_save_stocks_bearish():
    """
    Run the bearish filter and save to et4_filtered_stocks_bearish.py
    as `selected_stocks_bearish = [...]`.
    """
    symbols_analyzed = shares
    bearish_list = filter_stocks_bearish(kite, symbols_analyzed)

    out_file = "et4_filtered_stocks_bearish.py"
    try:
        with open(out_file, "w") as f:
            f.write("# This file is auto-generated by filter_and_save_stocks_bearish()\n")
            f.write("selected_stocks_bearish = [\n")
            for s in bearish_list:
                f.write(f"    '{s}',\n")
            f.write("]\n")
        logger.warning(f"{len(bearish_list)} stocks saved to {out_file}")
        print(f"{len(bearish_list)} stocks saved to {out_file}")
    except Exception as e:
        logger.error(f"Error writing to file {out_file}: {e}")


def filter_stocks_pullback_bullish(kite, symbols):
    """
    Bullish Pullback Filter - example conditions:
      1) Volume above threshold
      2) ADX > 20 (uptrend strength)
      3) Stochastic < 30 and stoch_k is turning up
      4) Price near 20-day SMA
    """
    selected = []

    MIN_AVG_VOLUME = 25000
    ADX_THRESHOLD = 20
    STOCH_OVERSOLD = 40
    MAX_PRICE = 25000
    MIN_PRICE = 50

    today = datetime.now().date()
    from_date = today - timedelta(days=200)  # Enough data for ADX & Stoch
    to_date = today

    print(f"Analyzing {len(symbols)} stocks for **bullish pullback** filter...")

    for sym in tqdm(symbols, desc="Filtering Pullback (Bullish)"):
        try:
            token = get_instrument_token_from_cache(sym)
            if not token:
                continue

            time.sleep(0.2)  # Rate-limit safeguard

            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            if not data or len(data) < 50:
                continue

            df = pd.DataFrame(data).dropna()
            if len(df) < 20:
                continue

            # Condition 1: Volume
            avg_vol_20 = df["volume"].tail(20).mean()
            if avg_vol_20 < MIN_AVG_VOLUME:
                continue

            # Condition 2: ADX > 20
            # We assume you have 'calculate_adx_all(df)' from your ADX logic
            df = calculate_adx_all(df, period=14)
            last_adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 0
            if last_adx <= ADX_THRESHOLD:
                continue

            # Condition 3: Stochastic < 40, turning up
            df = calculate_stochastic(df, k_period=14, d_period=3)
            if "stoch_k" not in df.columns:
                continue
            stoch_k_now = df["stoch_k"].iloc[-1]
            stoch_k_prev = df["stoch_k"].iloc[-2] if len(df) > 1 else 100

            # Must be oversold *and* stoch is increasing
            if stoch_k_now >= STOCH_OVERSOLD:
                continue
            if not (stoch_k_now > stoch_k_prev):
                continue

            # Condition 4: Price near 20-SMA
            df["sma20"] = df["close"].rolling(20).mean()
            last_close = df["close"].iloc[-1]
            last_sma20 = df["sma20"].iloc[-1]
            if pd.isna(last_sma20):
                continue

            if last_close < MIN_PRICE or last_close > MAX_PRICE:
                continue

            # within 5% of the 20-SMA
            if abs(last_close - last_sma20)/last_sma20 > 0.05:
                continue

            selected.append(sym)

        except Exception as e:
            logger.error(f"{sym} ERROR (Bullish Pullback): {e}")
            continue

    return selected


def filter_and_save_stocks_pullback_bullish():
    """
    Run the bullish pullback filter and save results to 
    et4_filtered_stocks_pullback_bullish.py
    """
    pullback_bullish = filter_stocks_pullback_bullish(kite, shares)

    out_file = "et4_filtered_stocks_pullback_bullish.py"
    try:
        with open(out_file, "w") as f:
            f.write("# This file is auto-generated by filter_and_save_stocks_pullback_bullish()\n")
            f.write("selected_stocks_pullback_bullish = [\n")
            for s in pullback_bullish:
                f.write(f"    '{s}',\n")
            f.write("]\n")
        print(f"{len(pullback_bullish)} stocks saved to {out_file}")
    except Exception as e:
        logger.error(f"Error writing to file {out_file}: {e}")




def filter_stocks_pullback_bearish(kite, symbols):
    """
    Bearish Pullback Filter - example conditions:
      1) Volume above threshold
      2) ADX > 20 (downtrend strength)
      3) Stochastic > 70, stoch_k turning down
      4) Price near 20-day SMA
    """
    selected = []

    MIN_AVG_VOLUME = 25000
    ADX_THRESHOLD = 20
    STOCH_OVERBOUGHT = 60
    MAX_PRICE = 25000
    MIN_PRICE = 50

    today = datetime.now().date()
    from_date = today - timedelta(days=200)
    to_date = today

    print(f"Analyzing {len(symbols)} stocks for **bearish pullback** filter...")

    for sym in tqdm(symbols, desc="Filtering Pullback (Bearish)"):
        try:
            token = get_instrument_token_from_cache(sym)
            if not token:
                continue

            time.sleep(0.2)

            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            if not data or len(data) < 50:
                continue

            df = pd.DataFrame(data).dropna()
            if len(df) < 20:
                continue

            # Condition 1: Volume
            avg_vol_20 = df["volume"].tail(20).mean()
            if avg_vol_20 < MIN_AVG_VOLUME:
                continue

            # Condition 2: ADX > 20
            df = calculate_adx_all(df, period=14)
            last_adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 0
            if last_adx <= ADX_THRESHOLD:
                continue

            # Condition 3: Stochastic > 60, stoch_k turning down
            df = calculate_stochastic(df, k_period=14, d_period=3)
            stoch_k_now = df["stoch_k"].iloc[-1] if "stoch_k" in df.columns else 0
            stoch_k_prev = df["stoch_k"].iloc[-2] if len(df) > 1 else 0

            if stoch_k_now <= STOCH_OVERBOUGHT:
                continue
            if not (stoch_k_now < stoch_k_prev):
                continue

            # Condition 4: Price near 20-day SMA
            df["sma20"] = df["close"].rolling(20).mean()
            last_close = df["close"].iloc[-1]
            last_sma20 = df["sma20"].iloc[-1]
            if pd.isna(last_sma20):
                continue

            if last_close < MIN_PRICE or last_close > MAX_PRICE:
                continue
            if abs(last_close - last_sma20) / last_sma20 > 0.05:
                continue

            selected.append(sym)
        except Exception as e:
            logger.error(f"{sym} ERROR (Bearish Pullback): {e}")
            continue

    return selected


def filter_and_save_stocks_pullback_bearish():
    """
    Run the bearish pullback filter and save results to 
    et4_filtered_stocks_pullback_bearish.py
    """
    pullback_bearish = filter_stocks_pullback_bearish(kite, shares)

    out_file = "et4_filtered_stocks_pullback_bearish.py"
    try:
        with open(out_file, "w") as f:
            f.write("# This file is auto-generated by filter_and_save_stocks_pullback_bearish()\n")
            f.write("selected_stocks_pullback_bearish = [\n")
            for s in pullback_bearish:
                f.write(f"    '{s}',\n")
            f.write("]\n")
        print(f"{len(pullback_bearish)} stocks saved to {out_file}")
    except Exception as e:
        logger.error(f"Error writing to file {out_file}: {e}")



# --------------------------------------------------------------------------
# Combine bullish and bearish
# --------------------------------------------------------------------------
def combine_filtered_stocks_into_single_list(
    bullish_file='et4_filtered_stocks_bullish.py',
    bearish_file='et4_filtered_stocks_bearish.py',
    pbullish_file='et4_filtered_stocks_pullback_bullish.py',
    pbearish_file='et4_filtered_stocks_pullback_bearish.py',
    combined_file='et4_filtered_stocks.py'
):
    """
    Combine standard bullish, standard bearish, pullback_bullish, pullback_bearish
    stocks into a single list named 'selected_stocks'.
    If any file does not exist or lacks the expected variable, default to an empty list.
    """
    import importlib.util

    def safe_load_list(file_path, var_name):
        """
        Attempt to load `var_name` from the Python file at `file_path`.
        If the file or variable doesn't exist, return an empty list.
        """
        if not os.path.isfile(file_path):
            logger.warning(f"File '{file_path}' does not exist. Returning empty list for {var_name}.")
            return []
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            result_list = getattr(module, var_name, [])
            if not isinstance(result_list, list):
                logger.warning(f"'{var_name}' in {file_path} is not a list. Returning empty list.")
                return []
            return result_list
        except Exception as e:
            logger.error(f"Failed to load variable '{var_name}' from {file_path}: {e}")
            return []

    try:
        # Load standard bullish, default to empty if missing
        bullish_list = safe_load_list(bullish_file, 'selected_stocks')
        # Some scripts store as 'selected_stocks_bullish', so let's try that too if empty:
        if not bullish_list:
            bullish_list = safe_load_list(bullish_file, 'selected_stocks_bullish')

        # Load standard bearish, default to empty if missing
        bearish_list = safe_load_list(bearish_file, 'selected_stocks_bearish')
        # Some scripts store as 'selected_stocks', so let's try that if empty:
        if not bearish_list:
            bearish_list = safe_load_list(bearish_file, 'selected_stocks')

        # Pullback bullish
        pbullish_list = safe_load_list(pbullish_file, 'selected_stocks_pullback_bullish')
        # Pullback bearish
        pbearish_list = safe_load_list(pbearish_file, 'selected_stocks_pullback_bearish')

        # Identify overlap among all four? or just standard + pullback? 
        # We'll do a single union for everything, but print out how many total
        # and how many are common to all:
        all_lists = [bullish_list, bearish_list, pbullish_list, pbearish_list]
        combined_set = set()
        for lst in all_lists:
            combined_set.update(lst)

        # For curiosity, let's see how many are in ALL four:
        common_all = set(bullish_list).intersection(bearish_list, pbullish_list, pbearish_list)
        print(f"Number of stocks in all four lists: {len(common_all)}")

        # Write combined
        combined = list(combined_set)
        with open(combined_file, "w") as f:
            f.write("# Auto-generated by combine_filtered_stocks_into_single_list()\n\n")
            f.write("selected_stocks = [\n")
            for s in combined:
                f.write(f"    '{s}',\n")
            f.write("]\n")

        print(f"Combined {len(combined)} stocks saved to {combined_file}")
        logger.warning(f"Combined {len(combined)} stocks saved to {combined_file}")

    except Exception as e:
        logger.error(f"Error combining filtered stocks: {e}")
        print(f"Error combining filtered stocks: {e}")


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Standard Bullish Filter
    filter_and_save_stocks()  # => et4_filtered_stocks_bullish.py

    # 2) Standard Bearish Filter
    filter_and_save_stocks_bearish()  # => et4_filtered_stocks_bearish.py

    # 3) Bullish Pullback Filter
    filter_and_save_stocks_pullback_bullish()  # => et4_filtered_stocks_pullback_bullish.py

    # 4) Bearish Pullback Filter
    filter_and_save_stocks_pullback_bearish()  # => et4_filtered_stocks_pullback_bearish.py

    # 5) Combine them all safely
    combine_filtered_stocks_into_single_list(
        bullish_file='et4_filtered_stocks_bullish.py',
        bearish_file='et4_filtered_stocks_bearish.py',
        pbullish_file='et4_filtered_stocks_pullback_bullish.py',
        pbearish_file='et4_filtered_stocks_pullback_bearish.py',
        combined_file='et4_filtered_stocks.py'
    )

