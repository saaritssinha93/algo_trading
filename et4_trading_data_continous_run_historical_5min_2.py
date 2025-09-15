# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:51:47 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Fetch 5-min historical data for selected_stocks from a FIXED start date to present,
compute indicators, and write *_main_indicators.csv files.

Window:
    • From: 2025-07-08 00:00 IST
    • To  : now (IST)

Adds (per bar):
    • Daily_Change  -> % vs previous trading day's final close
    • Intra_Change  -> % vs previous 5-min bar (intraday)
"""

import os
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kexc

# ========= CONFIG =========
india_tz = pytz.timezone("Asia/Kolkata")

# Fixed start date as requested
START_DATE = india_tz.localize(datetime(2025, 8, 25, 0, 0, 0))

# Directories
CACHE_DIR = "data_cache_August_5min"
INDICATORS_DIR = "main_indicators_August_5min"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Stock universe (make sure this exists in your env)
# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks_MIS import selected_stocks

# ========= KITE SESSION =========
def setup_kite_session():
    with open("access_token.txt", "r") as f:
        access_token = f.read().strip()
    with open("api_key.txt", "r") as f:
        api_key = f.read().split()[0]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    ins = pd.DataFrame(kite.instruments("NSE"))
    tokens = ins[ins["tradingsymbol"].isin(stocks)][["tradingsymbol", "instrument_token"]]
    return dict(zip(tokens["tradingsymbol"], tokens["instrument_token"]))

shares_tokens = get_tokens_for_stocks(selected_stocks)

# ========= INDICATORS =========
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(close, period=20, up=2, dn=2):
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return sma + up * std, sma - dn * std

def _ma(x, window, kind="sma"):
    if kind == "ema":
        return x.ewm(span=window, adjust=False).mean()
    return x.rolling(window, min_periods=window).mean()

def calculate_stochastic_fast(df, k_period=14, d_period=3):
    low_min  = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = (high_max - low_min)
    k = pd.Series(0.0, index=df.index)
    valid = rng > 0
    k.loc[valid] = 100.0 * (df["close"].loc[valid] - low_min.loc[valid]) / rng.loc[valid]
    k = k.clip(0.0, 100.0)
    d = k.rolling(d_period, min_periods=d_period).mean().clip(0.0, 100.0)
    return k, d

def calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma"):
    k_fast, _ = calculate_stochastic_fast(df, k_period=k_period, d_period=1)
    k_slow = _ma(k_fast, k_smooth, kind=ma_kind).clip(0.0, 100.0)
    d = _ma(k_slow, d_period, kind=ma_kind).clip(0.0, 100.0)
    return k_slow, d

def calculate_adx(df, period=14):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.Series(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy()
        ]),
        index=df.index
    )

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    eps = 1e-10
    plus_di = 100.0 * (plus_dm_sm / (atr + eps))
    minus_di = 100.0 * (minus_dm_sm / (atr + eps))
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.clip(0, 100)

def calculate_vwap(df):
    # Session-agnostic cumulative VWAP (resetting by day can be plugged in later)
    return (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)

def calculate_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def calculate_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)

def calculate_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp.diff() > 0, 0)
    neg_mf = mf.where(tp.diff() < 0, 0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum().abs()
    return 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

# ========= DATA FETCH =========
def fetch_historical_5min_df(token, start_dt_ist):
    """
    Fetch 5-min bars from start_dt_ist (tz-aware IST) to now in ~60-day chunks.
    Retries each chunk up to MAX_RETRIES times.
    """
    end = datetime.now(india_tz)
    chunk = timedelta(days=60)
    s = start_dt_ist
    frames = []
    MAX_RETRIES = 3
    SLEEP_BETWEEN_CALLS = 0.5

    print(f"  Window: {s.strftime('%Y-%m-%d %H:%M %Z')}  →  {end.strftime('%Y-%m-%d %H:%M %Z')}")

    while s < end:
        e = min(s + chunk, end)

        # Avoid overlapping last candle across chunks
        query_from = s
        query_to = e

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = kite.historical_data(token, query_from, query_to, "5minute")
                df = pd.DataFrame(raw)
                if df.empty:
                    break
                # Ensure tz-aware (IST)
                df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(india_tz)
                frames.append(df)
                break
            except (kexc.NetworkException, kexc.DataException, kexc.TokenException, kexc.InputException) as ex:
                if attempt == MAX_RETRIES:
                    print(f"    ! Failed chunk {query_from} → {query_to}: {ex}")
                else:
                    time.sleep(1.0 * attempt)  # simple backoff
            finally:
                time.sleep(SLEEP_BETWEEN_CALLS)

        s = e + timedelta(minutes=5)

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset="date")
          .sort_values("date")
          .reset_index(drop=True)
    )
    return out

# ========= PER-TICKER PIPELINE =========
def process_ticker_since_fixed_start(ticker, token):
    print(f"\nFetching {ticker} ...")
    df = fetch_historical_5min_df(token, START_DATE)
    if df.empty:
        print(f"  No data for {ticker} in the given window.")
        return

    # Indicators
    df["RSI"]       = calculate_rsi(df["close"])
    df["ATR"]       = calculate_atr(df)
    df["EMA_50"]    = calculate_ema(df["close"], 50)
    df["EMA_200"]   = calculate_ema(df["close"], 200)
    df["20_SMA"]    = df["close"].rolling(20, min_periods=20).mean()
    df["VWAP"]      = calculate_vwap(df)
    df["CCI"]       = calculate_cci(df)
    df["MFI"]       = calculate_mfi(df)
    df["OBV"]       = calculate_obv(df)

    macd, macd_sig, macd_hist = calculate_macd(df["close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"] = macd_hist

    df["Upper_Band"], df["Lower_Band"] = calculate_bollinger_bands(df["close"])
    df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
    df["Recent_Low"]  = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma")
    df["Stoch_%K"] = stoch_k
    df["Stoch_%D"] = stoch_d

    df["ADX"] = calculate_adx(df)

    # ======== CHANGES: Daily_Change & Intra_Change (IST day) ========
    df["date_only"] = df["date"].dt.tz_convert(india_tz).dt.date

    # Intra_Change: within-day pct change vs previous 5-min bar
    df["Intra_Change"] = (
        df.groupby("date_only")["close"]
          .pct_change()
          .mul(100.0)
    )

    # Daily_Change: % vs previous trading day's final close
    last_close_per_day = df.groupby("date_only", sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)
    df["Prev_Day_Close"] = df["date_only"].map(prev_day_last_close)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"]) * 100.0

    # Write
    out_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
    df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}  ({len(df)} rows)")
    head_cols = ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"]
    print(df[head_cols].head(10).to_string(index=False))

# ========= MAIN =========
if __name__ == "__main__":
    print(f"Fetching window: {START_DATE.strftime('%Y-%m-%d')} to {datetime.now(india_tz).strftime('%Y-%m-%d')} (IST)")
    for ticker in selected_stocks:
        token = shares_tokens.get(ticker)
        if not token:
            print(f"! No token for {ticker}, skipping.")
            continue
        try:
            process_ticker_since_fixed_start(ticker, token)
        except Exception as e:
            print(f"! Error processing {ticker}: {e}")
