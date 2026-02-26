# -*- coding: utf-8 -*-
"""
Unified historical fetch + indicator generator for multiple timeframes.

Supports TWO context modes:
1) previous  (NO LOOKAHEAD / previous context)
   - daily/intraday capped to previous calendar day (D-1)
   - weekly capped to previous fully completed week (previous Friday strictly before run date)

2) live (fetch "at the moment")
   - intraday capped to NOW (IST)
   - weekly capped to NOW (IST)  (weekly bar may be partial depending on broker semantics)
   - daily:
        * default: include today's daily candle ONLY if after market close time (IST)
        * optional: force include today's daily candle anytime (NOT recommended)

Freshness skip:
- If output exists and last candle date >= expected cutoff for that mode, skip fetching.

Note:
- Does NOT account for NSE holidays. "previous trading day" is approximated as previous calendar day.
"""

import os
import sys
import time as _time
import argparse
from datetime import datetime, timedelta, date, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kexc

# ========= GLOBAL CONFIG =========

india_tz = pytz.timezone("Asia/Kolkata")

DIRS = {
    "daily":  {"cache": "data_cache_daily",  "out": "main_indicators_daily"},
    "weekly": {"cache": "data_cache_weekly", "out": "main_indicators_weekly"},
    "1h":     {"cache": "data_cache_1h",     "out": "main_indicators_1h"},
    "3h":     {"cache": "data_cache_3h",     "out": "main_indicators_3h"},
    "5min":   {"cache": "data_cache_5min",   "out": "main_indicators_5min"},
    "15min":  {"cache": "data_cache_15min",  "out": "main_indicators_15min"},
}

for cfg in DIRS.values():
    os.makedirs(cfg["cache"], exist_ok=True)
    os.makedirs(cfg["out"], exist_ok=True)

DEFAULT_MAX_WORKERS = 6
SKIP_IF_FRESH = True

# Market timing (IST)
MARKET_CLOSE_TIME = time(15, 35)  # conservative close time for "day candle ready"
MARKET_OPEN_TIME  = time(9, 15)

# ========= STOCK UNIVERSE =========
from et4_filtered_stocks_MIS import selected_stocks

# ========= KITE SESSION =========

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
    ins = pd.DataFrame(kite.instruments("NSE"))
    tokens = ins[ins["tradingsymbol"].isin(stocks)][["tradingsymbol", "instrument_token"]]
    return dict(zip(tokens["tradingsymbol"], tokens["instrument_token"]))

shares_tokens = get_tokens_for_stocks(selected_stocks)

# ========= TIME HELPERS =========

def _to_ist(series_dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series_dt, errors="coerce")
    if dt.dt.tz is None:
        return dt.dt.tz_localize(india_tz)
    return dt.dt.tz_convert(india_tz)

def _prev_calendar_day(d: date) -> date:
    return (datetime.combine(d, datetime.min.time()) - timedelta(days=1)).date()

def last_friday_before(d: date) -> date:
    """Most recent Friday strictly before date d."""
    dd = d - timedelta(days=1)
    while dd.weekday() != 4:  # Friday = 4
        dd -= timedelta(days=1)
    return dd

def get_previous_daily_cutoff_dt(now_ist: datetime) -> datetime:
    d_prev = _prev_calendar_day(now_ist.date())
    return india_tz.localize(datetime(d_prev.year, d_prev.month, d_prev.day, 23, 59, 59))

def get_previous_weekly_cutoff_dt(now_ist: datetime) -> datetime:
    fri = last_friday_before(now_ist.date())
    return india_tz.localize(datetime(fri.year, fri.month, fri.day, 23, 59, 59))

def _round_down_to_minute(dt: datetime, minute_step: int) -> datetime:
    # round down to nearest minute_step
    m = (dt.minute // minute_step) * minute_step
    return dt.replace(minute=m, second=0, microsecond=0)

def get_end_dt_for_mode(mode: str, now_ist: datetime, context: str, force_today_daily: bool) -> datetime:
    """
    context:
      - previous: cap daily/intraday to D-1 and weekly to previous completed week
      - live: cap intraday/weekly to now; daily to today only after close (unless forced)
    """
    context = context.lower().strip()

    if context == "previous":
        if mode == "weekly":
            return get_previous_weekly_cutoff_dt(now_ist)
        return get_previous_daily_cutoff_dt(now_ist)

    # live
    if mode in ("5min",):
        return _round_down_to_minute(now_ist, 5)
    if mode in ("15min",):
        return _round_down_to_minute(now_ist, 15)
    if mode in ("1h", "3h"):
        return now_ist.replace(minute=0, second=0, microsecond=0)

    if mode == "daily":
        if force_today_daily:
            # include "today" even during market hours (not recommended)
            return now_ist
        # only include daily candle after market close
        if now_ist.time() >= MARKET_CLOSE_TIME:
            return now_ist
        return get_previous_daily_cutoff_dt(now_ist)

    if mode == "weekly":
        return now_ist

    return now_ist

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

def get_start_date(mode: str, now_ist: datetime) -> datetime:
    if mode in ("5min", "15min"):
        return india_tz.localize(datetime(2025, 8, 25, 0, 0, 0))

    base = now_ist - timedelta(days=365)
    if mode in ("1h", "3h"):
        base = base.replace(minute=0, second=0, microsecond=0)
    else:
        base = base.replace(hour=0, minute=0, second=0, microsecond=0)
    return base

# ========= FETCHERS =========

def fetch_historical_generic(token, start_dt_ist, end_dt_ist, interval, chunk_days, step_td):
    end = end_dt_ist
    chunk = timedelta(days=chunk_days)
    s = start_dt_ist
    frames = []

    MAX_RETRIES = 3
    SLEEP_BETWEEN_CALLS = 0.35

    print(f"  Window: {s.strftime('%Y-%m-%d %H:%M %Z')}  →  {end.strftime('%Y-%m-%d %H:%M %Z')}")

    while s < end:
        e = min(s + chunk, end)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = kite.historical_data(token, s, e, interval)
                df = pd.DataFrame(raw)
                if df.empty:
                    break
                df["date"] = _to_ist(df["date"])
                frames.append(df)
                break
            except (kexc.NetworkException, kexc.DataException, kexc.TokenException, kexc.InputException) as ex:
                if attempt == MAX_RETRIES:
                    print(f"    ! Failed chunk {s} → {e}: {ex}")
                else:
                    _time.sleep(1.0 * attempt)
            finally:
                _time.sleep(SLEEP_BETWEEN_CALLS)

        s = e + step_td

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset="date")
          .sort_values("date")
          .reset_index(drop=True)
    )
    out = out[out["date"] <= end_dt_ist].reset_index(drop=True)
    return out

def fetch_historical_daily_df(token, start_dt_ist, end_dt_ist):
    return fetch_historical_generic(token, start_dt_ist, end_dt_ist, "day", chunk_days=180, step_td=timedelta(days=1))

def fetch_historical_weekly_df(token, start_dt_ist, end_dt_ist):
    return fetch_historical_generic(token, start_dt_ist, end_dt_ist, "week", chunk_days=365, step_td=timedelta(days=7))

def fetch_historical_1h_df(token, start_dt_ist, end_dt_ist):
    return fetch_historical_generic(token, start_dt_ist, end_dt_ist, "60minute", chunk_days=60, step_td=timedelta(hours=1))

def fetch_historical_5min_df(token, start_dt_ist, end_dt_ist):
    return fetch_historical_generic(token, start_dt_ist, end_dt_ist, "5minute", chunk_days=60, step_td=timedelta(minutes=5))

def fetch_historical_15min_df(token, start_dt_ist, end_dt_ist):
    return fetch_historical_generic(token, start_dt_ist, end_dt_ist, "15minute", chunk_days=120, step_td=timedelta(minutes=15))

def fetch_historical_60m_for_3h(token, start_dt_ist, end_dt_ist):
    return fetch_historical_1h_df(token, start_dt_ist, end_dt_ist)

# ========= 3H RESAMPLE =========

def resample_to_3h(df_60m):
    if df_60m.empty:
        return df_60m

    df = df_60m.set_index("date").sort_index()

    df_shift = df.copy()
    df_shift.index = df_shift.index - pd.Timedelta(minutes=15)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_3h = df_shift.resample("3h", label="right", closed="right").agg(agg)

    df_3h.index = df_3h.index + pd.Timedelta(minutes=15)
    df_3h = df_3h.dropna(subset=["open", "high", "low", "close"])

    keep_times = {(9, 15), (12, 15), (15, 15)}
    mask = df_3h.index.map(lambda ts: (ts.hour, ts.minute) in keep_times)
    df_3h = df_3h[mask]

    return df_3h.reset_index().rename(columns={"index": "date"})

# ========= CHANGE FEATURES =========

def add_change_features_intraday(df):
    df["date_only"] = df["date"].dt.tz_convert(india_tz).dt.date
    df["Intra_Change"] = df.groupby("date_only")["close"].pct_change().mul(100.0)

    last_close_per_day = df.groupby("date_only", sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)

    df["Prev_Day_Close"] = df["date_only"].map(prev_day_last_close)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"] + 1e-10) * 100.0
    return df

# ========= FRESHNESS CHECK =========

def _read_last_date_from_csv(path: str) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(path, usecols=["date"])
        if df.empty:
            return None
        dt = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dt.empty:
            return None
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(india_tz)
        else:
            dt = dt.dt.tz_convert(india_tz)
        return dt.max()
    except Exception:
        return None

def should_skip_ticker(out_path: str, cutoff_dt_ist: datetime) -> bool:
    if not os.path.exists(out_path):
        return False
    last_dt = _read_last_date_from_csv(out_path)
    if last_dt is None:
        return False
    return last_dt >= cutoff_dt_ist

# ========= SAVE =========

def _finalize_and_save(df, out_path, head_cols=None):
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df)} rows)")
    if head_cols:
        print(df[head_cols].head(5).to_string(index=False))

# ========= PER-TICKER PIPELINES =========

def process_ticker_daily(ticker, token, start_dt_ist, end_dt_ist):
    out_path = os.path.join(DIRS["daily"]["out"], f"{ticker}_main_indicators_daily.csv")

    if SKIP_IF_FRESH and should_skip_ticker(out_path, end_dt_ist):
        print(f"[DAILY] {ticker}: output fresh till cutoff, skipping.")
        return

    print(f"\n[DAILY] Fetching {ticker} ...")
    df = fetch_historical_daily_df(token, start_dt_ist, end_dt_ist)
    if df.empty:
        print(f"  No daily data for {ticker} in the given window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(20, min_periods=20).max()
    df["Recent_Low"] = df["low"].rolling(20, min_periods=20).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    df["date_only"] = df["date"].dt.tz_convert(india_tz).dt.date
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)
    df["Prev_Day_Close"] = df["close"].shift(1)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"] + 1e-10) * 100.0

    _finalize_and_save(df, out_path, ["date", "close", "Prev_Day_Close", "Daily_Change"])

def process_ticker_weekly(ticker, token, start_dt_ist, end_dt_ist):
    out_path = os.path.join(DIRS["weekly"]["out"], f"{ticker}_main_indicators_weekly.csv")

    if SKIP_IF_FRESH and should_skip_ticker(out_path, end_dt_ist):
        print(f"[WEEKLY] {ticker}: output fresh till cutoff, skipping.")
        return

    print(f"\n[WEEKLY] Fetching {ticker} ...")
    df = fetch_historical_weekly_df(token, start_dt_ist, end_dt_ist)
    if df.empty:
        print(f"  No weekly data for {ticker} in the given window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(20, min_periods=20).max()
    df["Recent_Low"] = df["low"].rolling(20, min_periods=20).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    df["date_only"] = df["date"].dt.tz_convert(india_tz).dt.date
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)
    df["Prev_Day_Close"] = df["close"].shift(1)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"] + 1e-10) * 100.0

    _finalize_and_save(df, out_path, ["date", "close", "Prev_Day_Close", "Daily_Change"])

def process_ticker_1h(ticker, token, start_dt_ist, end_dt_ist):
    out_path = os.path.join(DIRS["1h"]["out"], f"{ticker}_main_indicators_1h.csv")

    if SKIP_IF_FRESH and should_skip_ticker(out_path, end_dt_ist):
        print(f"[1H] {ticker}: output fresh till cutoff, skipping.")
        return

    print(f"\n[1H] Fetching {ticker} ...")
    df = fetch_historical_1h_df(token, start_dt_ist, end_dt_ist)
    if df.empty:
        print(f"  No 1h data for {ticker} in the given window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
    df["Recent_Low"] = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)
    _finalize_and_save(df, out_path, ["date", "close", "Prev_Day_Close", "Daily_Change"])

def process_ticker_3h(ticker, token, start_dt_ist, end_dt_ist):
    out_path = os.path.join(DIRS["3h"]["out"], f"{ticker}_main_indicators_3h.csv")

    if SKIP_IF_FRESH and should_skip_ticker(out_path, end_dt_ist):
        print(f"[3H] {ticker}: output fresh till cutoff, skipping.")
        return

    print(f"\n[3H] Fetching {ticker} ...")
    df_60m = fetch_historical_60m_for_3h(token, start_dt_ist, end_dt_ist)
    if df_60m.empty:
        print(f"  No 60m data for {ticker} in the given window (3h base).")
        return

    df = resample_to_3h(df_60m)
    if df.empty:
        print(f"  After resample, no 3h data for {ticker}.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
    df["Recent_Low"] = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)
    _finalize_and_save(df, out_path, ["date", "close", "Prev_Day_Close", "Daily_Change"])

def _process_ticker_intraday_generic(ticker, token, start_dt_ist, end_dt_ist, mode, fetch_fn, recent_window=5):
    out_path = os.path.join(DIRS[mode]["out"], f"{ticker}_main_indicators_{mode}.csv")

    if SKIP_IF_FRESH and should_skip_ticker(out_path, end_dt_ist):
        print(f"[{mode.upper()}] {ticker}: output fresh till cutoff, skipping.")
        return

    print(f"\n[{mode.upper()}] Fetching {ticker} ...")
    df = fetch_fn(token, start_dt_ist, end_dt_ist)
    if df.empty:
        print(f"  No {mode} data for {ticker} in the given window.")
        return

    df = add_standard_indicators(df)
    df["Recent_High"] = df["high"].rolling(recent_window, min_periods=recent_window).max()
    df["Recent_Low"] = df["low"].rolling(recent_window, min_periods=recent_window).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)
    _finalize_and_save(df, out_path, ["date", "close", "Prev_Day_Close", "Daily_Change"])

def process_ticker_5min(ticker, token, start_dt_ist, end_dt_ist):
    _process_ticker_intraday_generic(ticker, token, start_dt_ist, end_dt_ist, "5min", fetch_historical_5min_df, recent_window=5)

def process_ticker_15min(ticker, token, start_dt_ist, end_dt_ist):
    _process_ticker_intraday_generic(ticker, token, start_dt_ist, end_dt_ist, "15min", fetch_historical_15min_df, recent_window=5)

# ========= DRIVER =========

VALID_MODES = ("daily", "weekly", "1h", "3h", "5min", "15min")

def _run_single_ticker_mode(mode: str, ticker: str, token: int, start_dt: datetime, end_dt: datetime):
    try:
        if mode == "daily":
            process_ticker_daily(ticker, token, start_dt, end_dt)
        elif mode == "weekly":
            process_ticker_weekly(ticker, token, start_dt, end_dt)
        elif mode == "1h":
            process_ticker_1h(ticker, token, start_dt, end_dt)
        elif mode == "3h":
            process_ticker_3h(ticker, token, start_dt, end_dt)
        elif mode == "5min":
            process_ticker_5min(ticker, token, start_dt, end_dt)
        elif mode == "15min":
            process_ticker_15min(ticker, token, start_dt, end_dt)
    except Exception as e:
        print(f"! Error processing {ticker} in mode {mode}: {e}")

def run_mode(mode: str, context: str, max_workers: int, force_today_daily: bool):
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected: {', '.join(VALID_MODES)}")

    now_ist = datetime.now(india_tz)
    start_dt = get_start_date(mode, now_ist)
    end_dt = get_end_dt_for_mode(mode, now_ist, context=context, force_today_daily=force_today_daily)

    print(
        f"\n=== MODE: {mode} | CONTEXT: {context} | Window: {start_dt.strftime('%Y-%m-%d %H:%M')} "
        f"to {end_dt.strftime('%Y-%m-%d %H:%M')} (IST) ==="
    )

    if end_dt <= start_dt:
        print("End cutoff is <= start. Nothing to fetch for this mode.")
        return

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

    print(f"Processing {len(work_items)} tickers with max_workers={max_workers} ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_ticker_mode, mode, ticker, token, start_dt, end_dt): ticker
            for (ticker, token) in work_items
        }

        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"! Worker crashed for {t} in mode {mode}: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="all", help="daily|weekly|1h|3h|5min|15min|all")
    p.add_argument("--context", default="live", choices=["live", "previous"], help="live fetches up to now; previous caps to D-1/prev week")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--force-today-daily", action="store_true", help="Include today's daily candle even before market close (NOT recommended)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    arg_mode = args.mode.lower()

    if arg_mode == "all":
        for m in VALID_MODES:
            run_mode(m, context=args.context, max_workers=args.max_workers, force_today_daily=args.force_today_daily)
    else:
        run_mode(arg_mode, context=args.context, max_workers=args.max_workers, force_today_daily=args.force_today_daily)
