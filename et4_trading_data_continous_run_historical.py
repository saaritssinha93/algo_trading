# -*- coding: utf-8 -*-
"""
Batch download 1-year (or as available) 15-min data for all selected_stocks.
Calculate all technical indicators for each bar and output full CSV.
"""

import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
from kiteconnect import KiteConnect, exceptions as kexc

# ========== SETUP ==========
india_tz = pytz.timezone('Asia/Kolkata')
CACHE_DIR = "data_cache_july"
INDICATORS_DIR = "main_indicators_july"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks_MIS import selected_stocks  # Make sure this list is available

# ==== Load KiteConnect session ====
def setup_kite_session():
    with open("access_token.txt", 'r') as f:
        access_token = f.read().strip()
    with open("api_key.txt", 'r') as f:
        api_key = f.read().split()[0]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    df = pd.DataFrame(kite.instruments("NSE"))
    tokens = df[df['tradingsymbol'].isin(stocks)][['tradingsymbol','instrument_token']]
    return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))

shares_tokens = get_tokens_for_stocks(selected_stocks)

# ========== TECHNICAL INDICATORS ==========

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(close, period=20, up=2, dn=2):
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = sma + up * std
    lower = sma - dn * std
    return upper, lower

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
    percent_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return percent_k, percent_d

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff(-1)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx

def calculate_vwap(df):
    vwap = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-10)
    return vwap

def calculate_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def calculate_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return cci

def calculate_mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp.diff() > 0, 0)
    neg_mf = mf.where(tp.diff() < 0, 0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum().abs()
    mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
    return mfi

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

# --- Add more indicators as needed using your originals above...

# ========== HISTORICAL FETCH ==========
def fetch_historical_15min_df(ticker, token, days=370):
    # Fetch at most 370 calendar days of 15-min data, API allows only 60 days per call.
    frames = []
    end = datetime.now(india_tz)
    start = end - timedelta(days=days)
    chunk = timedelta(days=60)
    s = start
    while s < end:
        e = min(s + chunk, end)
        df = pd.DataFrame(
            kite.historical_data(token, s, e, "15minute")
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.tz_convert(india_tz)
            frames.append(df)
        s = e + timedelta(minutes=15)  # API can return overlapping candle, so skip 1 step
        time.sleep(0.5)  # throttle for rate-limit
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    return result

# ========== PROCESS EACH TICKER ==========

def process_ticker_full_history(ticker, token):
    print(f"Fetching {ticker} ...")
    df = fetch_historical_15min_df(ticker, token, days=370)
    if df.empty:
        print(f"  No data for {ticker}")
        return

    # Add all indicators:
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = calculate_atr(df)
    df['EMA_50'] = calculate_ema(df['close'], 50)
    df['EMA_200'] = calculate_ema(df['close'], 200)
    df['20_SMA'] = df['close'].rolling(window=20, min_periods=20).mean()
    df['VWAP'] = calculate_vwap(df)
    df['CCI'] = calculate_cci(df)
    df['MFI'] = calculate_mfi(df)
    df['OBV'] = calculate_obv(df)
    # MACD, BB, Stoch, ADX, etc.
    macd, macd_signal, macd_hist = calculate_macd(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['close'])
    df['Recent_High'] = df['high'].rolling(window=5, min_periods=5).max()
    df['Recent_Low'] = df['low'].rolling(window=5, min_periods=5).min()
    stoch_k, stoch_d = calculate_stochastic(df)
    df['Stoch_%K'] = stoch_k
    df['Stoch_%D'] = stoch_d
    df['ADX'] = calculate_adx(df)
    # Add more indicators here as needed!

    # Calculate Daily Change %
    df['date_only'] = df['date'].dt.date
    prev_close = df.groupby('date_only')['close'].shift(1)
    df['Daily_Change'] = 100 * (df['close'] - prev_close) / (prev_close + 1e-10)

    # Write output
    out_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    df.to_csv(out_path, index=False)
    print(f"  {ticker} history saved ({len(df)} rows).")
    print(df.head(2))
    print(df.tail(2))

# ========== MAIN LOOP ==========
if __name__ == "__main__":
    for ticker in selected_stocks:
        token = shares_tokens.get(ticker)
        if not token:
            print(f"No token for {ticker}, skipping.")
            continue
        try:
            process_ticker_full_history(ticker, token)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
