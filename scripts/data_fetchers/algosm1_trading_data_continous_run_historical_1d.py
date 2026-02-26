# -*- coding: utf-8 -*-
"""
Fetch DAILY candles for selected_stocks for the LAST 1 YEAR up to now (IST),
compute indicators, and write *_main_indicators_daily.csv files.

Adds (per bar):
    • Daily_Change  -> % vs previous trading day's final close
    • Intra_Change  -> % vs previous DAILY bar (day-over-day), for schema consistency
"""

import os, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kexc

# ========= CONFIG =========
india_tz = pytz.timezone("Asia/Kolkata")
NOW_IST = datetime.now(india_tz)
START_DATE = (NOW_IST - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)

CACHE_DIR = "data_cache_daily"
INDICATORS_DIR = "main_indicators_daily"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# from et4_filtered_stocks_market_cap import selected_stocks
from et4_filtered_stocks_MIS import selected_stocks

# ========= KITE SESSION =========
def setup_kite_session():
    with open("access_token.txt", "r") as f:
        access_token = f.read().strip()
    with open("api_key.txt", "r") as f:
        api_key = f.read().split()[0]
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access_token)
    return k

kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    ins = pd.DataFrame(kite.instruments("NSE"))
    tokens = ins[ins["tradingsymbol"].isin(stocks)][["tradingsymbol","instrument_token"]]
    return dict(zip(tokens["tradingsymbol"], tokens["instrument_token"]))

shares_tokens = get_tokens_for_stocks(selected_stocks)

# ========= INDICATORS =========
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
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
    return macd, signal_line, macd - signal_line

def calculate_bollinger_bands(close, period=20, up=2, dn=2):
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return sma + up*std, sma - dn*std

def _ma(x, window, kind="sma"):
    return x.ewm(span=window, adjust=False).mean() if kind=="ema" \
           else x.rolling(window, min_periods=window).mean()

def calculate_stochastic_fast(df, k_period=14, d_period=3):
    low_min  = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = (high_max - low_min)
    k = pd.Series(0.0, index=df.index)
    valid = rng > 0
    k.loc[valid] = 100.0 * (df["close"].loc[valid] - low_min.loc[valid]) / rng.loc[valid]
    k = k.clip(0,100)
    d = k.rolling(d_period, min_periods=d_period).mean().clip(0,100)
    return k, d

def calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma"):
    k_fast, _ = calculate_stochastic_fast(df, k_period, d_period=1)
    k_slow = _ma(k_fast, k_smooth, kind=ma_kind).clip(0,100)
    d = _ma(k_slow, d_period, kind=ma_kind).clip(0,100)
    return k_slow, d

def calculate_adx(df, period=14):
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)
    tr = pd.Series(np.maximum.reduce([
        (high-low).to_numpy(), (high-prev_close).abs().to_numpy(), (low-prev_close).abs().to_numpy()
    ]), index=df.index)
    up_move, down_move = high - prev_high, prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    alpha = 1.0/float(period)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    eps = 1e-10
    plus_di  = 100.0 * (plus_dm_sm  / (atr + eps))
    minus_di = 100.0 * (minus_dm_sm / (atr + eps))
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)
    return dx.ewm(alpha=alpha, adjust=False).mean().clip(0,100)

def calculate_vwap(df):
    # cumulative VWAP across the period (not session-reset)
    return (df["close"]*df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)

def calculate_ema(close, span): return close.ewm(span=span, adjust=False).mean()

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
        if df["close"].iloc[i] > df["close"].iloc[i-1]: obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]: obv.append(obv[-1] - df["volume"].iloc[i])
        else: obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

# ========= DATA FETCH (DAILY) =========
def fetch_historical_daily_df(token, start_dt_ist):
    """
    Fetch daily bars from start_dt_ist to now in ~180-day chunks.
    """
    end = datetime.now(india_tz)
    chunk = timedelta(days=180)
    s = start_dt_ist
    frames = []
    MAX_RETRIES = 3
    SLEEP = 0.5

    print(f"  Window: {s.strftime('%Y-%m-%d %H:%M %Z')}  →  {end.strftime('%Y-%m-%d %H:%M %Z')}")
    while s < end:
        e = min(s + chunk, end)
        for attempt in range(1, MAX_RETRIES+1):
            try:
                raw = kite.historical_data(token, s, e, "day")
                df = pd.DataFrame(raw)
                if df.empty: break
                # tz-aware IST
                df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(india_tz)
                frames.append(df)
                break
            except (kexc.NetworkException, kexc.DataException, kexc.TokenException, kexc.InputException) as ex:
                if attempt == MAX_RETRIES:
                    print(f"    ! Failed chunk {s} → {e}: {ex}")
                else:
                    time.sleep(1.0 * attempt)
            finally:
                time.sleep(SLEEP)
        s = e + timedelta(days=1)

    if not frames: return pd.DataFrame()
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset="date")
             .sort_values("date")
             .reset_index(drop=True))
    return out

# ========= PER-TICKER PIPELINE =========
def process_ticker_last_year_daily(ticker, token):
    print(f"\nFetching {ticker} ...")
    df = fetch_historical_daily_df(token, START_DATE)
    if df.empty:
        print(f"  No daily data for {ticker} in the given 1y window.")
        return

    # Indicators on daily candles
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
    df["MACD"]         = macd
    df["MACD_Signal"]  = macd_sig
    df["MACD_Hist"]    = macd_hist

    df["Upper_Band"], df["Lower_Band"] = calculate_bollinger_bands(df["close"])
    df["Recent_High"] = df["high"].rolling(20, min_periods=20).max()
    df["Recent_Low"]  = df["low"].rolling(20, min_periods=20).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    # Change features
    df["date_only"] = df["date"].dt.tz_convert(india_tz).dt.date

    # Intra_Change (daily-over-daily change for schema continuity)
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)

    # Daily_Change: % vs previous trading day's final close (same as above on daily data)
    df["Prev_Day_Close"] = df["close"].shift(1)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"]) * 100.0

    # ---- Write ----
    out_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators_daily.csv")
    df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}  ({len(df)} rows)")
    preview_cols = ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"]
    print(df[preview_cols].head(10).to_string(index=False))

# ========= MAIN =========
if __name__ == "__main__":
    print(f"Fetching window (daily): {START_DATE.strftime('%Y-%m-%d')} to {datetime.now(india_tz).strftime('%Y-%m-%d')} (IST)")
    for ticker in selected_stocks:
        token = shares_tokens.get(ticker)
        if not token:
            print(f"! No token for {ticker}, skipping.")
            continue
        try:
            process_ticker_last_year_daily(ticker, token)
        except Exception as e:
            print(f"! Error processing {ticker}: {e}")
