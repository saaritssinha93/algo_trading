# -*- coding: utf-8 -*-
"""
Single-run script to replicate the live 15-min scenario for the past 2 months:
  1) We generate all 15-minute intervals from (NOW - 60 days) up to NOW.
  2) For each interval (call it t_candle), we fetch historical data from
     (t_candle - 10 days) to t_candle in 15-min interval (the continuity window).
  3) We merge that into a DataFrame, compute all indicators, and then
     take the row corresponding to t_candle as the "latest candle".
  4) We append that single row to {ticker}_main_indicators.csv,
     exactly as if the code had just run in real-time for that candle.
     
After the simulation, the code then checks all CSV files in the
'main_indicators_new' folder. For each CSV, if the most recent candle
is older than the latest 15-min interval (i.e. the CSV is not up-to-date),
it generates and prints the missing candle data.
     
Indicators included:
- RSI, ATR, MACD (with signal line & histogram)
- Bollinger Bands (Upper_Band, Lower_Band)
- Stochastic, ADX, VWAP
- 20-SMA, Recent High/Low (5-bar)
- EMA_50, EMA_200
- CCI, MFI, OBV, Parabolic_SAR
- Ichimoku (Tenkan_Sen, Kijun_Sen, Senkou_Span_A, Senkou_Span_B, Chikou_Span)
- Williams %R, Rate of Change (ROC), Bollinger Band Width (BB_Width)
- Daily_Change (vs. previous day's close)
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException, KiteException
from requests.exceptions import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

###############################################################################
# 1) User Configuration
###############################################################################
INDIA_TZ = pytz.timezone('Asia/Kolkata')

# Provide your list of tickers:
from et4_filtered_stocks_MIS import selected_stocks

# CSV folder for appending each "live" candle
INDICATORS_DIR = "main_indicators_new1"
os.makedirs(INDICATORS_DIR, exist_ok=True)

# We'll simulate 2 months (~60 days) of 15-min data
DAYS_BACK = 60

# Candle interval for the "live" simulation
INTERVAL = "15minute"

# When fetching continuity data, how many days of past data to fetch
# so that indicators have enough lookback (ATR, ADX, etc.)
CONTINUITY_DAYS = 10

###############################################################################
# 2) Logging Setup
###############################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)

logging.info("Initialized logger for 2-month simulation script.")

###############################################################################
# 3) KiteConnect Session
###############################################################################
def setup_kite_session() -> KiteConnect:
    """
    Reads local `access_token.txt` and `api_key.txt` to establish a KiteConnect session.
    """
    try:
        with open("access_token.txt", 'r') as tf:
            access_token = tf.read().strip()
        with open("api_key.txt", 'r') as kf:
            key_data = kf.read().split()
        if not key_data:
            raise ValueError("api_key.txt is empty or invalid.")
        kite_ = KiteConnect(api_key=key_data[0])
        kite_.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        return kite_
    except FileNotFoundError as e:
        logging.error(f"File not found for API keys/tokens: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

kite = setup_kite_session()

###############################################################################
# 4) Instrument Tokens
###############################################################################
def fetch_tokens_for_stocks(stocks):
    """
    Retrieves instrument tokens (NSE) for the given list of `stocks`.
    """
    try:
        instruments = kite.instruments(exchange="NSE")
        df_instr = pd.DataFrame(instruments)
        sub_df = df_instr[df_instr['tradingsymbol'].isin(stocks)][['tradingsymbol','instrument_token']]
        tk_map = dict(zip(sub_df['tradingsymbol'], sub_df['instrument_token']))
        return tk_map
    except Exception as e:
        logging.error(f"Error fetching instrument tokens: {e}")
        raise

try:
    shares_tokens = fetch_tokens_for_stocks(selected_stocks)
except Exception as e:
    logging.error("Could not fetch instrument tokens. Exiting.")
    sys.exit(1)

###############################################################################
# 5) Time Utility Functions
###############################################################################
def generate_15min_time_slots(days_back=60, interval_minutes=15):
    """
    Returns a list of candle-end timestamps from (NOW - days_back) up to 'now'.
    This version generates times labeled from 9:30 up to 15:15.
    """
    end_dt = datetime.now(INDIA_TZ).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)

    # Floor the start to a multiple of 15 minutes
    start_minute_block = (start_dt.minute // interval_minutes) * interval_minutes
    start_dt = start_dt.replace(minute=start_minute_block, second=0, microsecond=0)

    all_slots = []
    current = start_dt
    while current <= end_dt:
        local_time = current.astimezone(INDIA_TZ)
        # Keep only times between 9:30 and 15:15
        if ((local_time.hour > 9 or (local_time.hour == 9 and local_time.minute >= 30)) and
            (local_time.hour < 15 or (local_time.hour == 15 and local_time.minute <= 15))):
            all_slots.append(current)
        current += timedelta(minutes=interval_minutes)
    return all_slots

def generate_time_slots_from(start_dt: datetime, end_dt: datetime, interval_minutes=15):
    """
    Generate 15-min time slots from start_dt up to end_dt, filtering by trading hours.
    """
    slots = []
    current = start_dt
    while current <= end_dt:
        local_time = current.astimezone(INDIA_TZ)
        if ((local_time.hour > 9 or (local_time.hour == 9 and local_time.minute >= 30)) and
            (local_time.hour < 15 or (local_time.hour == 15 and local_time.minute <= 15))):
            slots.append(current)
        current += timedelta(minutes=interval_minutes)
    return slots

TIME_SLOTS = generate_15min_time_slots(DAYS_BACK, 15)

###############################################################################
# 6) Indicator Calculation Functions
###############################################################################
def calculate_rsi(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_atr(df: pd.DataFrame, period=14) -> pd.Series:
    d = df.copy()
    d['prev_close'] = d['close'].shift(1)
    d['range1'] = d['high'] - d['low']
    d['range2'] = (d['high'] - d['prev_close']).abs()
    d['range3'] = (d['low']  - d['prev_close']).abs()
    tr = d[['range1','range2','range3']].max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({'MACD': macd, 'Signal_Line': sig, 'Histogram': hist})

def calculate_bollinger_bands(df: pd.DataFrame, period=20, up=2, dn=2) -> pd.DataFrame:
    sma = df['close'].rolling(period, min_periods=1).mean()
    std = df['close'].rolling(period, min_periods=1).std()
    upper = sma + (std * up)
    lower = sma - (std * dn)
    return pd.DataFrame({'Upper_Band': upper, 'Lower_Band': lower})

def calculate_stochastic(df: pd.DataFrame, fastk=14, slowk=3) -> pd.Series:
    low_min = df['low'].rolling(window=fastk).min()
    high_max = df['high'].rolling(window=fastk).max()
    percent_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    return percent_k.rolling(window=slowk).mean()

def calculate_adx(df: pd.DataFrame, period=14) -> pd.Series:
    d = df.copy()
    d['prev_close'] = d['close'].shift(1)
    d['TR'] = d.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['prev_close']),
        abs(row['low']  - row['prev_close'])
    ), axis=1)
    d['+DM'] = np.where(
        (d['high'] - d['high'].shift(1)) > (d['low'].shift(1) - d['low']),
        np.where((d['high'] - d['high'].shift(1))>0, d['high'] - d['high'].shift(1), 0),
        0
    )
    d['-DM'] = np.where(
        (d['low'].shift(1) - d['low']) > (d['high'] - d['high'].shift(1)),
        np.where((d['low'].shift(1) - d['low'])>0, d['low'].shift(1) - d['low'], 0),
        0
    )
    d['TR_smooth'] = d['TR'].rolling(period).sum()
    d['+DM_smooth'] = d['+DM'].rolling(period).sum()
    d['-DM_smooth'] = d['-DM'].rolling(period).sum()
    # Wilder smoothing
    for i in range(period, len(d)):
        d.at[d.index[i], 'TR_smooth'] = d.at[d.index[i-1], 'TR_smooth'] - \
            (d.at[d.index[i-1], 'TR_smooth']/period) + d.at[d.index[i], 'TR']
        d.at[d.index[i], '+DM_smooth'] = d.at[d.index[i-1], '+DM_smooth'] - \
            (d.at[d.index[i-1], '+DM_smooth']/period) + d.at[d.index[i], '+DM']
        d.at[d.index[i], '-DM_smooth'] = d.at[d.index[i-1], '-DM_smooth'] - \
            (d.at[d.index[i-1], '-DM_smooth']/period) + d.at[d.index[i], '-DM']
    d['+DI'] = 100 * d['+DM_smooth'] / (d['TR_smooth'] + 1e-10)
    d['-DI'] = 100 * d['-DM_smooth'] / (d['TR_smooth'] + 1e-10)
    d['DX'] = 100 * (abs(d['+DI'] - d['-DI'])) / (d['+DI'] + d['-DI'] + 1e-10)
    d['ADX'] = d['DX'].rolling(period).mean()
    for i in range(2*period, len(d)):
        d.at[d.index[i], 'ADX'] = (
            (d.at[d.index[i-1], 'ADX']*(period-1)) + d.at[d.index[i], 'DX']
        ) / period
    return d['ADX']

def calculate_ema(series: pd.Series, span=50) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_cci(df: pd.DataFrame, period=20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close'])/3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015*mad + 1e-10)

def calculate_mfi(df: pd.DataFrame, period=14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close'])/3
    mf = tp * df['volume']
    diff_tp = tp.diff()
    pos_flow = mf.where(diff_tp > 0, 0)
    neg_flow = mf.where(diff_tp < 0, 0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum().abs()
    return 100 - (100 / (1 + (pos_sum/(neg_sum + 1e-10))))

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_parabolic_sar(df: pd.DataFrame, initial_af=0.02, step_af=0.02, max_af=0.2) -> pd.Series:
    length = len(df)
    if length == 0:
        return pd.Series([], dtype=float)
    psar = df['low'].iloc[0]
    ep = df['high'].iloc[0]
    af = initial_af
    trend = 1  # 1=up, -1=down
    sar_vals = [psar]
    for i in range(1, length):
        prev_psar = psar
        if trend == 1:
            psar = prev_psar + af*(ep - prev_psar)
            psar = min(psar, df['low'].iloc[i-1], df['low'].iloc[i])
            if df['low'].iloc[i] < psar:
                trend = -1
                psar = ep
                ep = df['low'].iloc[i]
                af = initial_af
        else:
            psar = prev_psar + af*(ep - prev_psar)
            psar = max(psar, df['high'].iloc[i-1], df['high'].iloc[i])
            if df['high'].iloc[i] > psar:
                trend = 1
                psar = ep
                ep = df['high'].iloc[i]
                af = initial_af
        if trend == 1:
            if df['high'].iloc[i] > ep:
                ep = df['high'].iloc[i]
                af = min(af+step_af, max_af)
        else:
            if df['low'].iloc[i] < ep:
                ep = df['low'].iloc[i]
                af = min(af+step_af, max_af)
        sar_vals.append(psar)
    return pd.Series(sar_vals, index=df.index)

def calculate_ichimoku(df: pd.DataFrame, conv=9, base=26, span_b=52, disp=26) -> pd.DataFrame:
    h = df['high']
    l = df['low']
    tenkan = (h.rolling(conv).max() + l.rolling(conv).min())/2
    kijun  = (h.rolling(base).max() + l.rolling(base).min())/2
    span_a = ((tenkan + kijun)/2).shift(disp)
    span_b_ = ((h.rolling(span_b).max() + l.rolling(span_b).min())/2).shift(disp)
    chikou = df['close'].shift(-disp)
    return pd.DataFrame({
        'Tenkan_Sen': tenkan,
        'Kijun_Sen': kijun,
        'Senkou_Span_A': span_a,
        'Senkou_Span_B': span_b_,
        'Chikou_Span': chikou
    })

def calculate_williams_r(df: pd.DataFrame, period=14) -> pd.Series:
    highest = df['high'].rolling(period).max()
    lowest  = df['low'].rolling(period).min()
    return -100 * (highest - df['close']) / (highest - lowest + 1e-10)

def calculate_roc(df: pd.DataFrame, period=12) -> pd.Series:
    return (df['close'].diff(period) / df['close'].shift(period)) * 100

def calculate_bb_width(df: pd.DataFrame, period=20, up=2, dn=2) -> pd.Series:
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + std * up
    lower = sma - std * dn
    return upper - lower

###############################################################################
# 7) Single Candle Fetch (simulating "live" scenario)
###############################################################################
def make_api_call_with_retry(token: int, from_d: str, to_d: str, interval: str,
                             max_retries=5, delay=1, backoff=2) -> pd.DataFrame:
    """
    Safe wrapper around kite.historical_data with retry.
    """
    attempts = 0
    while attempts < max_retries:
        try:
            logging.info(f"Fetching historical_data(token={token}, interval={interval}) Attempt:{attempts+1}")
            raw = kite.historical_data(token, from_d, to_d, interval)
            if not raw:
                logging.warning("No data returned by historical_data, retrying...")
                time.sleep(delay)
                attempts += 1
                delay *= backoff
                continue
            return pd.DataFrame(raw)
        except (NetworkException, KiteException, HTTPError) as e:
            e_str = str(e).lower()
            if "too many requests" in e_str or "rate limit" in e_str:
                logging.warning(f"Rate limit => wait {delay}s, attempt {attempts+1}")
                time.sleep(delay)
                attempts += 1
                delay *= backoff
            else:
                logging.error(f"API call error => {e}")
                time.sleep(delay)
                attempts += 1
                delay *= backoff
        except Exception as ex:
            logging.error(f"Unexpected error => {ex}")
            time.sleep(delay)
            attempts += 1
            delay *= backoff

    logging.error(f"Exceeded max_retries for token={token}")
    return pd.DataFrame()

def fetch_single_candle_data(ticker: str, token: int, candle_end_dt: datetime) -> pd.DataFrame:
    """
    Simulates a single "live" run for the 15-min candle ending at `candle_end_dt`.
    We fetch from (candle_end_dt - CONTINUITY_DAYS) to candle_end_dt with 15-min intervals,
    then re-calc the indicators and extract the row that matches candle_end_dt.
    """
    fetch_start = candle_end_dt - timedelta(days=CONTINUITY_DAYS)
    fetch_end   = candle_end_dt
    from_str = fetch_start.strftime("%Y-%m-%d %H:%M:%S")
    to_str   = fetch_end.strftime("%Y-%m-%d %H:%M:%S")

    df = make_api_call_with_retry(token, from_str, to_str, INTERVAL)
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', drop=False, inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(INDIA_TZ)
    df.sort_index(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = recalc_indicators_for_entire_df(df)

    target_minute_block = candle_end_dt.replace(second=0, microsecond=0)
    df_candidate = df[df['date'] == target_minute_block]
    if df_candidate.empty:
        # If exact match not found, pick the last row <= target_minute_block
        df_candidate = df[df['date'] <= target_minute_block]
        if df_candidate.empty:
            return pd.DataFrame()
        final_row = df_candidate.iloc[[-1]].copy()
    else:
        final_row = df_candidate.iloc[[-1]].copy()

    return final_row

###############################################################################
# 8) Recalc Over Entire Range (for continuity)
###############################################################################
def recalc_indicators_for_entire_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recomputes all columns over the entire DataFrame (not just the last row).
    Also sets 'Daily_Change' by comparing each day's close with previous day's close.
    """
    if df.empty:
        return df
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    required_cols = ['open','high','low','close','volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.warning(f"Data missing required columns: {missing}")
        return df

    # RSI
    df['RSI'] = calculate_rsi(df['close'])
    # ATR
    df['ATR'] = calculate_atr(df)
    # MACD
    macd_df = calculate_macd(df)
    df['MACD']        = macd_df['MACD']
    df['Signal_Line'] = macd_df['Signal_Line']
    df['Histogram']   = macd_df['Histogram']
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(df)
    df['Upper_Band'] = bb_df['Upper_Band']
    df['Lower_Band'] = bb_df['Lower_Band']
    # Stochastic
    df['Stochastic'] = calculate_stochastic(df)
    # VWAP
    df['VWAP'] = ((df['close'] * df['volume']).cumsum()) / (df['volume'].cumsum() + 1e-10)
    # 20_SMA
    df['20_SMA'] = df['close'].rolling(20).mean()
    # Recent_High/Low
    df['Recent_High'] = df['high'].rolling(5).max()
    df['Recent_Low']  = df['low'].rolling(5).min()
    # ADX
    df['ADX'] = calculate_adx(df)
    # EMA_50, EMA_200
    df['EMA_50']  = calculate_ema(df['close'], 50)
    df['EMA_200'] = calculate_ema(df['close'], 200)
    # CCI
    df['CCI'] = calculate_cci(df)
    # MFI
    df['MFI'] = calculate_mfi(df)
    # OBV
    df['OBV'] = calculate_obv(df)
    # Parabolic_SAR
    df['Parabolic_SAR'] = calculate_parabolic_sar(df)
    # Ichimoku
    ich = calculate_ichimoku(df)
    for c in ich.columns:
        df[c] = ich[c]
    # Williams %R
    df['Williams_%R'] = calculate_williams_r(df)
    # ROC
    df['ROC'] = calculate_roc(df)
    # BB_Width
    df['BB_Width'] = calculate_bb_width(df)

    # Daily_Change
    df['Daily_Change'] = np.nan
    for idx in range(len(df)):
        if idx == 0:
            continue
        today_dt = df.at[idx, 'date']
        prev_dt  = df.at[idx-1, 'date']
        if today_dt.date() != prev_dt.date():
            prev_close = df.at[idx-1, 'close']
            if prev_close != 0:
                chg = ((df.at[idx, 'close'] - prev_close)/prev_close)*100
                df.at[idx, 'Daily_Change'] = chg
        else:
            df.at[idx, 'Daily_Change'] = df.at[idx-1, 'Daily_Change']

    return df

###############################################################################
# 9) Append Single Candle to CSV
###############################################################################
def append_candle_to_csv(ticker: str, row_df: pd.DataFrame):
    """
    Takes a single-row DataFrame for a candle and physically appends it
    to {ticker}_main_indicators.csv (one row at a time).
    Does NOT re-read or merge old CSV data; simply writes each new row at the end.
    """
    if row_df.empty or 'date' not in row_df.columns:
        return

    csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")

    # Ensure columns you might want to see
    for c in ['Entry Signal','Signal_ID','logtime','CalcFlag']:
        if c not in row_df.columns:
            row_df[c] = ''

    # If CSV doesn't exist, we'll write the header once; otherwise just append
    write_header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode='a', index=False, header=write_header)
    logging.info(f"[{ticker}] => Appended candle => {row_df.iloc[0]['date']}")

###############################################################################
# 10) Main Single-Run Process (Simulating the 15-min loop)
###############################################################################
def simulate_2month_live_for_ticker(ticker: str):
    """
    For the given ticker:
      - Build a list of 15-min time slots for the last 2 months
      - For each time slot, fetch data from (slot_dt - CONTINUITY_DAYS) to slot_dt
      - Recompute indicators, pick the final candle row
      - Append that single row to CSV
    """
    token = shares_tokens.get(ticker)
    if not token:
        logging.warning(f"[{ticker}] => No instrument token found. Skipping.")
        return

    # Track last appended timestamp to avoid repeating the same candle
    last_appended_time = None

    for slot_dt in TIME_SLOTS:
        single_row = fetch_single_candle_data(ticker, token, slot_dt)
        if single_row.empty:
            continue
        current_time = single_row.iloc[0]['date']

        # Skip if it's the same timestamp as last appended
        if last_appended_time is not None and current_time == last_appended_time:
            continue

        append_candle_to_csv(ticker, single_row)
        last_appended_time = current_time

    logging.info(f"[{ticker}] => Completed 2-month simulation with {len(TIME_SLOTS)} candle fetches.")

def main_simulation_run():
    """
    Simulates the "every 15 mins" approach for all tickers over the last 2 months,
    but **only** for tickers whose CSV file does not yet exist.
    """
    logging.info("Starting 2-month simulation run, replicating live 15-min fetch.")
    results = {}

    # Use ThreadPoolExecutor to run each ticker in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {}
        for tkr in selected_stocks:
            # Skip if CSV already exists
            csv_path = os.path.join(INDICATORS_DIR, f"{tkr}_main_indicators.csv")
            if os.path.exists(csv_path):
                logging.info(f"[{tkr}] => main_indicators.csv already exists; skipping simulation.")
                results[tkr] = True
                continue

            future = executor.submit(simulate_2month_live_for_ticker, tkr)
            future_to_ticker[future] = tkr

        for future in as_completed(future_to_ticker):
            tkr = future_to_ticker[future]
            try:
                future.result()  # raises any exception from simulate_2month_live_for_ticker
                results[tkr] = True
            except Exception as e:
                logging.error(f"Error processing {tkr}: {e}")
                traceback.print_exc()
                results[tkr] = False

    done = [k for k,v in results.items() if v]
    not_done = [k for k,v in results.items() if not v]
    logging.info(f"\nCompleted tickers: {done}")
    if not_done:
        logging.info(f"Tickers with errors: {not_done}")
    logging.info("All done with 2-month live-like simulation process.")

# 11) Missing-data updater  ###############################################

def simulate_missing_candles_for_ticker(ticker: str):
    """
    If {ticker}_main_indicators.csv is not current, work out every 15-min slot
    from the last recorded candle up to ‘now’, fetch each candle (with the same
    continuity-window logic you already use), recompute indicators, and
    APPEND the rows to the CSV so the file becomes fully up-to-date.
    """
    csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")

    # ── 1. Decide our start-point ─────────────────────────────────────────
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, usecols=["date"])
            df["date"] = pd.to_datetime(df["date"])
            last_date = df["date"].max().tz_localize(INDIA_TZ) \
                       if df["date"].max().tzinfo is None else df["date"].max()
        except Exception as e:
            logging.error(f"[{ticker}] CSV read/parsing error: {e}")
            return
    else:
        # No CSV yet – start 60 days back so we create the whole file.
        last_date = datetime.now(INDIA_TZ) - timedelta(days=DAYS_BACK)

    # If nothing to fill, quit fast
    now = datetime.now(INDIA_TZ).replace(second=0, microsecond=0)
    if now - last_date <= timedelta(minutes=15):
        logging.info(f"[{ticker}] already current (last: {last_date}).")
        return

    # ── 2. Build every missing 15-min slot ───────────────────────────────
    start_missing = last_date + timedelta(minutes=15)
    missing_slots = generate_time_slots_from(start_missing, now, 15)
    if not missing_slots:
        logging.info(f"[{ticker}] No slots to fill.")
        return

    # ── 3. Fetch & append for each slot ──────────────────────────────────
    token = shares_tokens.get(ticker)
    if not token:
        logging.warning(f"[{ticker}] No instrument token – skipped.")
        return

    for slot in missing_slots:
        row = fetch_single_candle_data(ticker, token, slot)
        if row.empty:
            logging.warning(f"[{ticker}] No data for {slot}.")
            continue
        append_candle_to_csv(ticker, row)      # <- **NOW WE APPEND**
    logging.info(f"[{ticker}] Filled {len(missing_slots)} missing candles.")


def update_all_csvs_to_current():
    """
    Runs simulate_missing_candles_for_ticker() for every symbol in
    `selected_stocks` – whether the CSV exists or not.
    """
    for symbol in selected_stocks:
        simulate_missing_candles_for_ticker(symbol)


def check_missing_data_for_all_csvs():
    """
    Loops over all CSV files in INDICATORS_DIR and for each, extracts the ticker
    (assumed to be the portion before the first underscore) and then simulates missing candles.
    """
    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators*.csv")
    csv_files = [f for f in os.listdir(INDICATORS_DIR) if f.endswith(".csv")]
    for file in csv_files:
        # Assuming the file name format is {ticker}_main_indicators.csv
        ticker = file.split("_")[0]
        simulate_missing_candles_for_ticker(ticker)

###############################################################################
# 12) Script Entry
###############################################################################
if __name__ == "__main__":
    # First, run the 2-month simulation for tickers without an existing CSV.
    #main_simulation_run()
    
    # Then, check each CSV file for missing recent candle data and print the missing data.
    #check_missing_data_for_all_csvs()
    update_all_csvs_to_current()
