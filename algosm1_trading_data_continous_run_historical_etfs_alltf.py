# -*- coding: utf-8 -*-
"""
Unified historical fetch + indicator generator for ETF timeframes.

MODES (per-ticker, fetched in parallel using a thread pool):

    daily    -> main_indicators_etf_daily      (1-year window, DAILY candles)
    weekly   -> main_indicators_etf_weekly     (1-year window, WEEKLY candles)
    monthly  -> main_indicators_etf_monthly    (1-year window, MONTHLY candles)
    1h       -> main_indicators_etf_1h         (1-year window, 60-minute candles)
    3h       -> main_indicators_etf_3h         (1-year window, resampled 3-hour from 60-minute)
    5min     -> main_indicators_etf_5min       (from fixed 2025-08-25, 5-minute candles)
    15min    -> main_indicators_etf_15min      (from fixed 2025-08-25, 15-minute candles)

Usage:
    python algosm1_unified_history_etf.py daily
    python algosm1_unified_history_etf.py weekly
    python algosm1_unified_history_etf.py monthly
    python algosm1_unified_history_etf.py 1h
    python algosm1_unified_history_etf.py 3h
    python algosm1_unified_history_etf.py 5min
    python algosm1_unified_history_etf.py 15min
    python algosm1_unified_history_etf.py all

Requires:
    - api_key.txt       (first word is API key)
    - access_token.txt  (full access token string)
    - algosm1_filtered_etfs.py providing 'selected_stocks' list
"""

import os
import sys
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kexc

# ========= GLOBAL CONFIG =========

india_tz = pytz.timezone("Asia/Kolkata")
NOW_IST = datetime.now(india_tz)

# Directories (ETF-specific)
DIRS = {
    "daily": {
        "cache": "data_cache_etf_daily",
        "out":   "main_indicators_etf_daily",
    },
    "weekly": {
        "cache": "data_cache_etf_weekly",
        "out":   "main_indicators_etf_weekly",
    },
    "monthly": {
        "cache": "data_cache_etf_monthly",
        "out":   "main_indicators_etf_monthly",
    },
    "1h": {
        "cache": "data_cache_etf_1h",
        "out":   "main_indicators_etf_1h",
    },
    "3h": {
        "cache": "data_cache_etf_3h",
        "out":   "main_indicators_etf_3h",
    },
    "5min": {
        "cache": "data_cache_etf_5min",
        "out":   "main_indicators_etf_5min",
    },
    "15min": {
        "cache": "data_cache_etf_15min",
        "out":   "main_indicators_etf_15min",
    },
}

for cfg in DIRS.values():
    os.makedirs(cfg["cache"], exist_ok=True)
    os.makedirs(cfg["out"], exist_ok=True)

# Parallelism + behaviour toggles
DEFAULT_MAX_WORKERS = 6          # tune based on your machine + API limits
SKIP_IF_OUT_EXISTS = True        # skip ticker if final CSV already exists

# ========= STOCK UNIVERSE (ETFs) =========

from algosm1_filtered_etfs import selected_stocks

# ========= KITE SESSION & TOKENS =========

def setup_kite_session() -> KiteConnect:
    with open("access_token.txt", "r") as f:
        access_token = f.read().strip()
    with open("api_key.txt", "r") as f:
        api_key = f.read().split()[0]
    kite_local = KiteConnect(api_key=api_key)
    kite_local.set_access_token(access_token)
    return kite_local

kite = setup_kite_session()


def get_tokens_for_stocks(stocks):
    """
    Single instruments() call, then map tradingsymbol -> instrument_token.
    """
    ins = pd.DataFrame(kite.instruments("NSE"))
    tokens = ins[ins["tradingsymbol"].isin(stocks)][[
        "tradingsymbol", "instrument_token"
    ]]
    return dict(zip(tokens["tradingsymbol"], tokens["instrument_token"]))


shares_tokens = get_tokens_for_stocks(selected_stocks)

# ========= TIME + DATE HELPERS =========

def _to_ist(series_dt: pd.Series) -> pd.Series:
    """
    Robustly convert Kite 'date' to tz-aware IST.
    Handles naive timestamps by localizing to IST,
    and tz-aware timestamps by converting to IST.
    """
    dt = pd.to_datetime(series_dt, errors="coerce")
    if dt.dt.tz is None:
        return dt.dt.tz_localize(india_tz)
    return dt.dt.tz_convert(india_tz)


# ========= INDICATOR FUNCTIONS =========

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
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = high_max - low_min
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
            (low - prev_close).abs().to_numpy(),
        ]),
        index=df.index,
    )

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

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
    # Cumulative VWAP
    return (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)


def calculate_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def calculate_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
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


def add_standard_indicators(df):
    """Add RSI, ATR, EMAs, VWAP, CCI, MFI, OBV, MACD, Bollinger."""
    df["RSI"] = calculate_rsi(df["close"])
    df["ATR"] = calculate_atr(df)
    df["EMA_50"] = calculate_ema(df["close"], 50)
    df["EMA_200"] = calculate_ema(df["close"], 200)
    df["20_SMA"] = df["close"].rolling(20, min_periods=20).mean()
    df["VWAP"] = calculate_vwap(df)
    df["CCI"] = calculate_cci(df)
    df["MFI"] = calculate_mfi(df)
    df["OBV"] = calculate_obv(df)

    macd, macd_sig, macd_hist = calculate_macd(df["close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"] = macd_hist

    df["Upper_Band"], df["Lower_Band"] = calculate_bollinger_bands(df["close"])
    return df


# ========= START DATE PER MODE =========

def get_start_date(mode: str) -> datetime:
    """
    Returns tz-aware IST start datetime for each mode.

    daily/weekly/monthly/1h/3h -> NOW_IST - 365 days
    5min/15min                 -> fixed 2025-08-25 00:00 IST (as in original script)
    """
    if mode in ("5min", "15min"):
        return india_tz.localize(datetime(2025, 8, 25, 0, 0, 0))

    base = NOW_IST - timedelta(days=365)

    if mode in ("1h", "3h"):
        base = base.replace(minute=0, second=0, microsecond=0)
    else:  # daily / weekly / monthly
        base = base.replace(hour=0, minute=0, second=0, microsecond=0)

    return base


# ========= FETCHERS =========

def fetch_historical_generic(token, start_dt_ist, interval, chunk_days, step_td):
    """
    Generic chunked fetcher for day/week/month/60min/15min/5min.
    Chunked to respect Kite's historical data window limits.
    """
    end = datetime.now(india_tz)
    chunk = timedelta(days=chunk_days)
    s = start_dt_ist
    frames = []

    MAX_RETRIES = 3
    SLEEP_BETWEEN_CALLS = 0.4  # small pause between API calls

    print(
        f"  Window: {s.strftime('%Y-%m-%d %H:%M %Z')}  →  "
        f"{end.strftime('%Y-%m-%d %H:%M %Z')}"
    )

    while s < end:
        e = min(s + chunk, end)
        query_from = s
        query_to = e

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = kite.historical_data(token, query_from, query_to, interval)
                df = pd.DataFrame(raw)
                if df.empty:
                    break
                df["date"] = _to_ist(df["date"])
                frames.append(df)
                break
            except (
                kexc.NetworkException,
                kexc.DataException,
                kexc.TokenException,
                kexc.InputException,
            ) as ex:
                if attempt == MAX_RETRIES:
                    print(f"    ! Failed chunk {query_from} → {query_to}: {ex}")
                else:
                    time.sleep(1.0 * attempt)
            finally:
                time.sleep(SLEEP_BETWEEN_CALLS)

        s = e + step_td

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset="date")
          .sort_values("date")
          .reset_index(drop=True)
    )
    return out


def fetch_historical_etf_daily_df(token, start_dt_ist):
    return fetch_historical_generic(
        token=token,
        start_dt_ist=start_dt_ist,
        interval="day",
        chunk_days=180,
        step_td=timedelta(days=1),
    )


def fetch_historical_etf_weekly_df(token, start_dt_ist):
    return fetch_historical_generic(
        token=token,
        start_dt_ist=start_dt_ist,
        interval="week",
        chunk_days=365,
        step_td=timedelta(days=7),
    )


def fetch_historical_etf_monthly_df(token, start_dt_ist):
    """
    Build MONTHLY candles by:
        1) Fetching DAILY data from Kite (interval='day')
        2) Resampling it to calendar months (OHLCV)

    This avoids using interval='month', which Kite does NOT support.
    """
    # 1) Fetch daily data from the same start date
    df_daily = fetch_historical_etf_daily_df(token, start_dt_ist)
    if df_daily.empty:
        return df_daily  # nothing to resample

    # 2) Resample daily → monthly on IST 'date'
    df = df_daily.copy()
    df = df.set_index("date").sort_index()

    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }

    # Calendar month resample (month end timestamps)
    df_m = df.resample("M", label="right", closed="right").agg(agg)

    # Drop any incomplete/empty months
    df_m = df_m.dropna(subset=["open", "high", "low", "close"])

    # Back to a normal dataframe with 'date' column
    df_m = df_m.reset_index()  # 'date' is now a column again
    return df_m


def process_ticker_monthly(ticker, token, start_dt_ist):
    """
    Monthly timeframe pipeline.

    Note: naming is monthly-specific:
        • Prev_Month_Close = previous month's close
        • Monthly_Change   = % change vs previous month's close
    """
    mode = "monthly"
    if SKIP_IF_OUT_EXISTS and _out_exists(mode, ticker):
        print(f"[MONTHLY] {ticker}: output exists, skipping.")
        return

    print(f"\n[MONTHLY] Fetching {ticker} ...")
    df = fetch_historical_etf_monthly_df(token, start_dt_ist)
    if df.empty:
        print(f"  No monthly data for {ticker} in the given window.")
        return

    # Add indicators on the MONTHLY candles
    df = add_standard_indicators(df)

    # Monthly-specific extras (e.g. last 12 months window for highs/lows)
    df["Recent_High"] = df["high"].rolling(12, min_periods=3).max()
    df["Recent_Low"]  = df["low"].rolling(12, min_periods=3).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    # Keep date as IST & also store month period
    df["date"] = _to_ist(df["date"])
    df["month_only"] = df["date"].dt.to_period("M")

    # Simple month-over-month change
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)

    df["Prev_Month_Close"] = df["close"].shift(1)
    df["Monthly_Change"] = (
        (df["close"] - df["Prev_Month_Close"]) / (df["Prev_Month_Close"]) * 100.0
    )

    _finalize_and_save(
        df,
        mode,
        ticker,
        ["date", "close", "Intra_Change", "Prev_Month_Close", "Monthly_Change"],
    )


def fetch_historical_etf_1h_df(token, start_dt_ist):
    return fetch_historical_generic(
        token=token,
        start_dt_ist=start_dt_ist,
        interval="60minute",
        chunk_days=60,
        step_td=timedelta(hours=1),
    )


def fetch_historical_etf_5min_df(token, start_dt_ist):
    return fetch_historical_generic(
        token=token,
        start_dt_ist=start_dt_ist,
        interval="5minute",
        chunk_days=60,
        step_td=timedelta(minutes=5),
    )


def fetch_historical_etf_15min_df(token, start_dt_ist):
    return fetch_historical_generic(
        token=token,
        start_dt_ist=start_dt_ist,
        interval="15minute",
        chunk_days=120,
        step_td=timedelta(minutes=15),
    )


def fetch_historical_60m_for_etf_3h(token, start_dt_ist):
    return fetch_historical_etf_1h_df(token, start_dt_ist)


# ========= 3H RESAMPLE =========

def resample_to_etf_3h(df_60m):
    """
    Take a 60m OHLCV dataframe and build 3h candles aligned to 09:15/12:15/15:15 IST.
    Assumes df_60m has columns: date, open, high, low, close, volume.
    """
    if df_60m.empty:
        return df_60m

    df = df_60m.set_index("date").sort_index()

    # Shift index back by 15 min so 3h bins land at ... 09:15, 12:15, 15:15 after we shift back.
    df_shift = df.copy()
    df_shift.index = df_shift.index - pd.Timedelta(minutes=15)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_etf_3h = df_shift.resample("3h", label="right", closed="right").agg(agg)

    # Shift timestamps forward by 15 min to restore real session times
    df_etf_3h.index = df_etf_3h.index + pd.Timedelta(minutes=15)

    # Drop empty bins
    df_etf_3h = df_etf_3h.dropna(subset=["open", "high", "low", "close"])

    # Keep only canonical intraday endpoints
    keep_times = {(9, 15), (12, 15), (15, 15)}
    mask = df_etf_3h.index.map(lambda ts: (ts.hour, ts.minute) in keep_times)
    df_etf_3h = df_etf_3h[mask]

    return df_etf_3h.reset_index().rename(columns={"index": "date"})


# ========= CHANGE FEATURES (shared intraday) =========

def add_change_features_intraday(df):
    """
    Adds date_only, Intra_Change (within day), Prev_Day_Close (prev day's last close),
    and Daily_Change (% vs Prev_Day_Close) for intraday frames.
    """
    df["date"] = _to_ist(df["date"])
    df["date_only"] = df["date"].dt.date

    df["Intra_Change"] = df.groupby("date_only")["close"].pct_change().mul(100.0)

    last_close_per_day = df.groupby("date_only", sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)

    df["Prev_Day_Close"] = df["date_only"].map(prev_day_last_close)
    df["Daily_Change"] = (
        (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"]) * 100.0
    )
    return df


# ========= OUTPUT PATH HELPERS =========

def get_out_path(mode: str, ticker: str) -> str:
    """
    Central place to define ETF filename patterns per timeframe.
    We keep original ETF naming for existing modes and extend for 15min + monthly.
    """
    out_dir = DIRS[mode]["out"]

    if mode == "daily":
        fname = f"{ticker}_main_indicators_etf_daily.csv"
    elif mode == "weekly":
        fname = f"{ticker}_main_indicators_etf_weekly.csv"
    elif mode == "monthly":
        fname = f"{ticker}_main_indicators_etf_monthly.csv"
    elif mode == "1h":
        fname = f"{ticker}_main_indicators_etf_1h.csv"
    elif mode == "3h":
        fname = f"{ticker}_main_indicators_etf_3h.csv"
    elif mode == "5min":
        # original ETF 5min script used this simpler name
        fname = f"{ticker}_main_indicators.csv"
    elif mode == "15min":
        # follow same pattern as 5min but in a different folder
        fname = f"{ticker}_main_indicators.csv"
    else:
        raise ValueError(f"Unknown mode for out_path: {mode}")

    return os.path.join(out_dir, fname)


def _finalize_and_save(df, mode: str, ticker: str, head_cols=None):
    out_path = get_out_path(mode, ticker)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df)} rows)")
    if head_cols:
        print(df[head_cols].head(5).to_string(index=False))


def _out_exists(mode: str, ticker: str) -> bool:
    """
    Check if final output file exists for (mode, ticker).
    Used to optionally skip already-processed tickers.
    """
    return os.path.exists(get_out_path(mode, ticker))


# ========= PER-TICKER PIPELINES =========

def process_ticker_daily(ticker, token, start_dt_ist):
    mode = "daily"
    if SKIP_IF_OUT_EXISTS and _out_exists(mode, ticker):
        print(f"[DAILY] {ticker}: output exists, skipping.")
        return

    print(f"\n[DAILY] Fetching {ticker} ...")
    df = fetch_historical_etf_daily_df(token, start_dt_ist)
    if df.empty:
        print(f"  No daily data for {ticker} in the given 1y window.")
        return

    df = add_standard_indicators(df)

    # Daily-specific extras
    df["Recent_High"] = df["high"].rolling(20, min_periods=20).max()
    df["Recent_Low"] = df["low"].rolling(20, min_periods=20).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    df["date"] = _to_ist(df["date"])
    df["date_only"] = df["date"].dt.date

    # Day-over-day change
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)

    df["Prev_Day_Close"] = df["close"].shift(1)
    df["Daily_Change"] = (
        (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"]) * 100.0
    )

    _finalize_and_save(
        df,
        mode,
        ticker,
        ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"],
    )


def process_ticker_weekly(ticker, token, start_dt_ist):
    mode = "weekly"
    if SKIP_IF_OUT_EXISTS and _out_exists(mode, ticker):
        print(f"[WEEKLY] {ticker}: output exists, skipping.")
        return

    print(f"\n[WEEKLY] Fetching {ticker} ...")
    df = fetch_historical_etf_weekly_df(token, start_dt_ist)
    if df.empty:
        print(f"  No weekly data for {ticker} in the given 1y window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(20, min_periods=20).max()
    df["Recent_Low"] = df["low"].rolling(20, min_periods=20).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    df["date"] = _to_ist(df["date"])
    df["date_only"] = df["date"].dt.date

    # Week-over-week change
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)

    df["Prev_Day_Close"] = df["close"].shift(1)
    df["Daily_Change"] = (
        (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"]) * 100.0
    )

    _finalize_and_save(
        df,
        mode,
        ticker,
        ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"],
    )




def process_ticker_1h(ticker, token, start_dt_ist):
    mode = "1h"
    if SKIP_IF_OUT_EXISTS and _out_exists(mode, ticker):
        print(f"[1H] {ticker}: output exists, skipping.")
        return

    print(f"\n[1H] Fetching {ticker} ...")
    df = fetch_historical_etf_1h_df(token, start_dt_ist)
    if df.empty:
        print(f"  No 1h data for {ticker} in the given 1y window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
    df["Recent_Low"]  = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(
        df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma"
    )
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)

    _finalize_and_save(
        df,
        mode,
        ticker,
        ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"],
    )


def process_ticker_3h(ticker, token, start_dt_ist):
    mode = "3h"
    if SKIP_IF_OUT_EXISTS and _out_exists(mode, ticker):
        print(f"[3H] {ticker}: output exists, skipping.")
        return

    print(f"\n[3H] Fetching {ticker} ...")
    df_60m = fetch_historical_60m_for_etf_3h(token, start_dt_ist)
    if df_60m.empty:
        print(f"  No 60m data for {ticker} in the given 1y window (3h base).")
        return

    df = resample_to_etf_3h(df_60m)
    if df.empty:
        print(f"  After resample, no 3h data for {ticker}.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
    df["Recent_Low"]  = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)

    _finalize_and_save(
        df,
        mode,
        ticker,
        ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"],
    )


def _process_ticker_intraday_generic(
    ticker,
    token,
    start_dt_ist,
    mode,
    fetch_fn,
    recent_window=5,
):
    if SKIP_IF_OUT_EXISTS and _out_exists(mode, ticker):
        print(f"[{mode.upper()}] {ticker}: output exists, skipping.")
        return

    label = mode.upper()
    print(f"\n[{label}] Fetching {ticker} ...")
    df = fetch_fn(token, start_dt_ist)
    if df.empty:
        print(f"  No {mode} data for {ticker} in the given window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(recent_window, min_periods=recent_window).max()
    df["Recent_Low"]  = df["low"].rolling(recent_window, min_periods=recent_window).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d

    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)

    _finalize_and_save(
        df,
        mode,
        ticker,
        ["date", "close", "Intra_Change", "Prev_Day_Close", "Daily_Change"],
    )


def process_ticker_5min(ticker, token, start_dt_ist):
    _process_ticker_intraday_generic(
        ticker=ticker,
        token=token,
        start_dt_ist=start_dt_ist,
        mode="5min",
        fetch_fn=fetch_historical_etf_5min_df,
        recent_window=5,
    )


def process_ticker_15min(ticker, token, start_dt_ist):
    _process_ticker_intraday_generic(
        ticker=ticker,
        token=token,
        start_dt_ist=start_dt_ist,
        mode="15min",
        fetch_fn=fetch_historical_etf_15min_df,
        recent_window=5,
    )


# ========= DRIVER (PARALLEL BY TICKER) =========

VALID_MODES = ("daily", "weekly", "monthly", "1h", "3h", "5min", "15min")


def _run_single_ticker_mode(mode: str, ticker: str, token: int, start_dt: datetime):
    """
    Worker for a single (mode, ticker). Used inside the thread pool.
    """
    try:
        if mode == "daily":
            process_ticker_daily(ticker, token, start_dt)
        elif mode == "weekly":
            process_ticker_weekly(ticker, token, start_dt)
        elif mode == "monthly":
            process_ticker_monthly(ticker, token, start_dt)
        elif mode == "1h":
            process_ticker_1h(ticker, token, start_dt)
        elif mode == "3h":
            process_ticker_3h(ticker, token, start_dt)
        elif mode == "5min":
            process_ticker_5min(ticker, token, start_dt)
        elif mode == "15min":
            process_ticker_15min(ticker, token, start_dt)
    except Exception as e:
        print(f"! Error processing {ticker} in mode {mode}: {e}")


def run_mode(mode: str, max_workers: int = DEFAULT_MAX_WORKERS):
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected: {', '.join(VALID_MODES)}")

    start_dt = get_start_date(mode)
    print(
        f"\n=== MODE: {mode} | Window: {start_dt.strftime('%Y-%m-%d %H:%M')} "
        f"to {datetime.now(india_tz).strftime('%Y-%m-%d %H:%M')} (IST) ==="
    )

    work_items = []
    for ticker in selected_stocks:
        token = shares_tokens.get(ticker)
        if not token:
            print(f"! No token for {ticker}, skipping.")
            continue
        work_items.append((ticker, token))

    if not work_items:
        print("No valid tickers with tokens. Nothing to do.")
        return

    print(f"Processing {len(work_items)} ETF tickers with max_workers={max_workers} ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_ticker_mode, mode, ticker, token, start_dt): ticker
            for (ticker, token) in work_items
        }

        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"! Worker crashed for {t} in mode {mode}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        arg_mode = "all"   # default to all ETF timeframes
        print("No mode passed on command line. Defaulting to: all")
    else:
        arg_mode = sys.argv[1].lower()

    if arg_mode == "all":
        for m in VALID_MODES:
            run_mode(m, max_workers=DEFAULT_MAX_WORKERS)
    else:
        run_mode(arg_mode, max_workers=DEFAULT_MAX_WORKERS)
