# -*- coding: utf-8 -*-
"""
Batch download 1-year (or as available) 5-min data for all selected_stocks.
Calculate all technical indicators for each bar and output full CSV.

Adds:
    • Daily_Change  -> % vs previous trading day's final close
    • Intra_Change  -> % vs previous 5-min bar (intraday)
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
CACHE_DIR = "data_cache_August_5min"
INDICATORS_DIR = "main_indicators_August_5min"
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


def calculate_adx(df, period=14):
    """
    Wilder's ADX (vectorized):
      1) True Range (TR)
      2) Directional Movement (+DM, -DM)
      3) Wilder EMA smoothing for TR, +DM, -DM (alpha=1/period)
      4) +DI, -DI, DX, ADX

    Returns: pd.Series ADX in [0, 100]
    """
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # 1) True Range
    tr = pd.Series(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy()
        ]),
        index=df.index
    )

    # 2) Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # 3) Wilder smoothing (EMA with alpha=1/period)
    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # 4) DI, DX, ADX
    eps = 1e-10
    plus_di = 100.0 * (plus_dm_sm / (atr + eps))
    minus_di = 100.0 * (minus_dm_sm / (atr + eps))
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    # Sanity clamp to [0, 100]
    return adx.clip(lower=0, upper=100)


def calculate_vwap(df):
    # Session-agnostic cumulative VWAP (resetting by day is also possible if needed)
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

# ========== HISTORICAL FETCH ==========
def fetch_historical_5min_df(ticker, token, days=370):
    """
    Fetch at most 370 calendar days of 5-min data.
    Kite allows ~60 days per call; we loop in 60-day chunks.
    """
    frames = []
    end = datetime.now(india_tz)
    start = end - timedelta(days=days)
    chunk = timedelta(days=60)
    s = start
    while s < end:
        e = min(s + chunk, end)
        df = pd.DataFrame(
            kite.historical_data(token, s, e, "5minute")
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.tz_convert(india_tz)
            frames.append(df)
        # Advance by 5 minutes to avoid overlapping the last candle of the previous chunk
        s = e + timedelta(minutes=5)
        time.sleep(0.5)  # throttle for rate-limit

    if not frames:
        return pd.DataFrame()

    result = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset='date')
          .sort_values('date')
          .reset_index(drop=True)
    )
    return result

# ========== PROCESS EACH TICKER ==========

def process_ticker_full_history(ticker, token):
    print(f"Fetching {ticker} ...")
    df = fetch_historical_5min_df(ticker, token, days=370)
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
    stoch_k, stoch_d = calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma")
    df['Stoch_%K'] = stoch_k
    df['Stoch_%D'] = stoch_d
    df['ADX'] = calculate_adx(df)

    # ========== CHANGES BELOW ==========
    # Compute date-only for grouping
    df['date_only'] = df['date'].dt.date

    # 1) Intra_Change: % change vs previous 5-min bar *within the same day*
    df['Intra_Change'] = (
        df.groupby('date_only')['close']
          .pct_change()
          .mul(100.0)
    )

    # 2) Daily_Change: % vs previous trading day's FINAL close
    #    - First, get last close for each day (index is the date_only)
    last_close_per_day = (
        df.groupby('date_only', sort=True)['close']
          .last()
    )
    #    - Shift by one day to align each day with the *previous* day's close
    prev_day_last_close = last_close_per_day.shift(1)

    #    - Map each row's date_only to the previous day's final close
    df['Prev_Day_Close'] = df['date_only'].map(prev_day_last_close)

    #    - Compute Daily_Change; remains NaN when previous day's close is not available
    df['Daily_Change'] = (df['close'] - df['Prev_Day_Close']) / (df['Prev_Day_Close']) * 100.0

    # ========== /CHANGES ==========

    # Write output
    out_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    df.to_csv(out_path, index=False)
    print(f"  {ticker} history saved ({len(df)} rows).")
    print(df[['date','close','Intra_Change','Prev_Day_Close','Daily_Change']].head(10).to_string(index=False))
    print(df[['date','close','Intra__change','Prev_Day_Close','Daily_Change']] if False else "")  # placeholder to keep IDE quiet

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
