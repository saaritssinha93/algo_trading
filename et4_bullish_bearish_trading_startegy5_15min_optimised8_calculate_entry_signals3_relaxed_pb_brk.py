# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:20:29 2024

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
This script automates fetching intraday stock data, updating and calculating technical indicators,
detecting trend-based entry signals, and maintaining records (main_indicators CSV files) without overwriting old data.

Key points:
- Data is fetched every 15 minutes (intraday intervals).
- Old data is never refreshed or overwritten. Only new data points are appended.
- Indicators are recalculated on a recent subset of data to save computation time.
- Entry signals are determined from the existing indicators in the main CSV files.
- A unique Signal_ID prevents reprocessing old rows.

The code also handles logging, error scenarios, and schedules tasks during market hours.
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect.exceptions import NetworkException, KiteException
import pandas as pd
import numpy as np
import pytz
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
import traceback
from requests.exceptions import HTTPError
import shutil
import csv
import glob
import argparse
import signal
import sys

# ================================
# Global Configuration
# ================================

india_tz = pytz.timezone('Asia/Kolkata')  # Timezone for all date operations

# We import a list of stocks from another file
from et4_filtered_stocks_market_cap import selected_stocks

# Directories for caching data and storing main indicators
CACHE_DIR = "data_cache"
INDICATORS_DIR = "main_indicators"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICATORS_DIR, exist_ok=True)

# Limit how many threads can make API calls simultaneously
api_semaphore = threading.Semaphore(2)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # We use WARNING as the base level to avoid too verbose logs

file_handler = logging.FileHandler("trading_script_main.log")
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging initialized successfully.")
print("Logging initialized and script execution started.")

# Attempt to set working directory
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
    logger.info(f"Changed working directory to {cwd}.")
except FileNotFoundError:
    logger.error(f"Working directory {cwd} not found.")
    logger.debug(traceback.format_exc())
    raise
except Exception as e:
    logger.error(f"Unexpected error changing directory: {e}")
    logger.debug(traceback.format_exc())
    raise

# Define Market Holidays for 2024 - These are days no trading occurs.
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

class APICallError(Exception):
    """Custom Exception for API call failures after retries"""
    pass

def normalize_time(df, timezone='Asia/Kolkata'):
    """
    Ensure 'date' column is in datetime format and set to the correct timezone.
    If the 'date' column is naive (no timezone), we first treat it as UTC and then convert.
    """
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError("The 'date' column could not be converted to datetime.")

    # If no timezone, assume UTC then convert to given timezone
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(timezone)

    return df

def load_and_normalize_csv(file_path, expected_cols=None, timezone='Asia/Kolkata'):
    """
    Load a CSV, ensure it has a 'date' column in proper timezone,
    and if some expected columns are missing, add them.
    """
    if not os.path.exists(file_path):
        # If file not found, return empty DataFrame with the expected columns
        return pd.DataFrame(columns=expected_cols if expected_cols else [])

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df = normalize_time(df, timezone=timezone)
        df = df.sort_values(by='date').reset_index(drop=True)
    else:
        # If no 'date' column, just ensure other columns exist
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""

    # Ensure all expected columns are present
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[expected_cols]

    return df


def setup_kite_session():
    """
    Setup the KiteConnect session using saved API keys and access tokens.
    Without proper keys, the script cannot fetch data.
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
        print(f"Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        print(f"Error: {e}")
        raise

kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    """
    Given a list of stock symbols, fetch their corresponding instrument tokens from the Kite API.
    Tokens are needed to request historical data for each stock.
    """
    try:
        logging.info("Fetching tokens for provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][['tradingsymbol','instrument_token']]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        raise

shares_tokens = get_tokens_for_stocks(selected_stocks)


def detect_signals_in_memory(ticker: str, df_today: pd.DataFrame) -> list:
    """
    Signal detection with relaxed thresholds for pullbacks while maintaining quality breakouts/breakdowns.
    """

    signals_detected = []

    if df_today.empty:
        return signals_detected  # No data -> no signals

    # -- 1) Daily change
    try:
        first_open = df_today["open"].iloc[0]
        latest_close = df_today["close"].iloc[-1]
        daily_change = ((latest_close - first_open) / first_open) * 100
    except Exception as e:
        logging.error(f"[{ticker}] Error computing daily_change in detect_signals_in_memory: {e}")
        return signals_detected

    # -- 2) Check if ADX indicates momentum (e.g., > 35 for focused momentum)
    latest_row = df_today.iloc[-1]
    adx_val = latest_row.get("ADX", 0)
    if adx_val <= 22:  # Maintain ADX threshold for general momentum
        return signals_detected

    # Decide if it's Bullish or Bearish scenario:
    if (
        daily_change > 1.5
        and latest_row.get("RSI", 0) > 35
        and latest_row.get("close", 0) > latest_row.get("VWAP", 0) * 1.01
    ):
        trend_type = "Bullish"
    elif (
        daily_change < -1.5
        and latest_row.get("RSI", 0) < 65
        and latest_row.get("close", 0) < latest_row.get("VWAP", 0) * 0.99
    ):
        trend_type = "Bearish"
    else:
        return signals_detected

    # -- 3) Define "Pullback" & "Breakout/Breakdown" conditions
    volume_multiplier = 1.8
    rolling_window = 10
    MACD_BULLISH_OFFSET = 0.1
    MACD_BEARISH_OFFSET = -0.1

    if trend_type == "Bullish":
        conditions = {
            "Pullback": (
                (df_today["low"] >= df_today["VWAP"] * 0.99)  # Loosened proximity to VWAP
                & (df_today["close"] > df_today["open"])
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] < 47)  # Slightly relaxed stochastic threshold
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 22)  # Reduced ADX threshold for pullbacks
            ),
            "Breakout": (
                (df_today["close"] > df_today["high"].rolling(window=rolling_window).max() * 0.995)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] > df_today["Signal_Line"])
                & (df_today["MACD"] > MACD_BULLISH_OFFSET)
                & (df_today["close"] > df_today["20_SMA"])
                & (df_today["Stochastic"] > 60)
                & (df_today["Stochastic"].diff() > 0)
                & (df_today["ADX"] > 42)
            ),
        }

    else:  # Bearish
        conditions = {
            "Pullback": (
                (df_today["high"] <= df_today["VWAP"] * 1.01)  # Loosened proximity to VWAP
                & (df_today["close"] < df_today["open"])
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] > 53)  # Slightly relaxed stochastic threshold
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 22)  # Reduced ADX threshold for pullbacks
            ),
            "Breakdown": (
                (df_today["close"] < df_today["low"].rolling(window=rolling_window).min() * 1.005)
                & (df_today["volume"] > volume_multiplier * df_today["volume"].rolling(window=rolling_window).mean())
                & (df_today["MACD"] < df_today["Signal_Line"])
                & (df_today["MACD"] < MACD_BEARISH_OFFSET)
                & (df_today["close"] < df_today["20_SMA"])
                & (df_today["Stochastic"] < 40)
                & (df_today["Stochastic"].diff() < 0)
                & (df_today["ADX"] > 42)
            ),
        }

    # -- 4) Collect rows matching conditions
    for entry_type, cond in conditions.items():
        selected = df_today.loc[cond]
        if not selected.empty:
            for _, row in selected.iterrows():
                signals_detected.append({
                    "Ticker": ticker,
                    "date": row["date"],
                    "Entry Type": entry_type,
                    "Trend Type": trend_type,
                    "Price": round(row.get("close", 0), 2),
                    "Daily Change %": round(daily_change, 2),
                    "VWAP": round(row.get("VWAP", 0), 2),
                    "ADX": round(row.get("ADX", 0), 2),
                    "MACD": row.get("MACD", None),
                    "Signal_ID": row.get("Signal_ID", ""),
                    "logtime": row.get("logtime", ""),
                    "Entry Signal": "Yes"
                })

    return signals_detected


def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day:
    - If today is trading day, return today
    - Else, go back until we find the last valid trading day.
    """
    if reference_date is None:
        reference_date=datetime.now().date()
    logging.info(f"Checking last trading day from {reference_date}.")
    print(f"Reference Date: {reference_date}")
    if reference_date.weekday()<5 and reference_date not in market_holidays:
        last_trading_day=reference_date
    else:
        last_trading_day=reference_date-timedelta(days=1)
        while last_trading_day.weekday()>=5 or last_trading_day in market_holidays:
            last_trading_day-=timedelta(days=1)
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")
    return last_trading_day




def signal_handler(sig, frame):
    logging.info("Interrupt received, shutting down.")
    print("Interrupt received. Shutting down gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)



def find_price_action_entries_for_today(last_trading_day):
    """
    Read each ticker's main_indicators.csv, slice today's data,
    call detect_signals_in_memory(),
    combine all signals into a single DataFrame.
    """
    india_tz = pytz.timezone('Asia/Kolkata')
    # We'll define today's start and end times
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,25)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14,50)))

    pattern = os.path.join(INDICATORS_DIR, '*_main_indicators.csv')
    csv_files = glob.glob(pattern)

    all_signals = []
    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace('_main_indicators.csv','')
        df = load_and_normalize_csv(file_path, timezone='Asia/Kolkata')
        if df.empty:
            continue

        # slice today's data
        df_today = df[(df['date'] >= start_time)&(df['date'] <= end_time)].copy()
        if df_today.empty:
            continue

        # detect signals
        signals_list = detect_signals_in_memory(ticker, df_today)
        if signals_list:
            all_signals.extend(signals_list)

    # Return them as a DataFrame
    if not all_signals:
        return pd.DataFrame()
    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values(by='date', inplace=True)
    signals_df.reset_index(drop=True, inplace=True)
    return signals_df




def mark_signals_in_main_csv(ticker, signals_list, main_path, india_tz):
    """
    Given a list of signals (each with 'date', 'Entry Signal', 'Signal_ID', etc.),
    update the corresponding rows in `*_main_indicators.csv` so that 'Entry Signal'='Yes'.
    Also updates 'logtime' for those rows.
    """
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] No existing main_indicators.csv to update.")
        return

    try:
        # Load existing main CSV
        df_main = load_and_normalize_csv(main_path, timezone='Asia/Kolkata')
        if df_main.empty:
            logging.warning(f"[{ticker}] main_indicators.csv is empty. No signals updated.")
            return

        if 'Entry Signal' not in df_main.columns:
            df_main['Entry Signal'] = 'No'

        # Iterate over each signal and update the corresponding row
        for signal in signals_list:
            sig_id = signal.get('Signal_ID', "")
            sig_date = signal.get('date', None)

            # Option A: Use Signal_ID if available
            if sig_id and 'Signal_ID' in df_main.columns:
                mask = (df_main['Signal_ID'] == str(sig_id))
            else:
                # Option B: Match exact date
                mask = (df_main['date'] == sig_date)

            # Update 'Entry Signal' and 'logtime' for matched rows
            df_main.loc[mask, 'Entry Signal'] = 'Yes'
            df_main.loc[mask, 'logtime'] = datetime.now(india_tz).isoformat()

        # Write back the updated CSV
        df_main.sort_values('date', inplace=True)
        df_main.to_csv(main_path, index=False)
        logging.info(f"[{ticker}] Updated Entry Signal='Yes' for {len(signals_list)} signals.")
    except Exception as e:
        logging.error(f"[{ticker}] Error updating signals in main CSV: {e}")


def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads the full-day 'price_action_entries' from input_file,
    ensures only the *first* (earliest) intraday entry of each Ticker
    is kept, calculates target price & quantity, then merges
    into 'papertrade_{YYYY-MM-DD}.csv' (no duplicates).
    """

    india_tz = pytz.timezone('Asia/Kolkata')
    if not os.path.exists(input_file):
        print(f"No entries file {input_file}, skipping papertrade creation.")
        return

    df_entries = pd.read_csv(input_file)
    if 'date' not in df_entries.columns:
        logging.error(f"No 'date' column in {input_file}")
        raise KeyError("No 'date' in entries file.")

    # Convert 'date' to Asia/Kolkata time
    df_entries = normalize_time(df_entries, timezone='Asia/Kolkata')
    df_entries.sort_values('date', inplace=True)

    # Filter strictly to today's (9:25–15:50)
    start_time = india_tz.localize(datetime.combine(last_trading_day, datetime_time(9,25)))
    end_time   = india_tz.localize(datetime.combine(last_trading_day, datetime_time(14,50)))
    todays_entries = df_entries[(df_entries['date'] >= start_time) & (df_entries['date'] <= end_time)].copy()

    if todays_entries.empty:
        print(f"No valid entries found for {last_trading_day}.")
        return

    # ---------------------------
    # Only keep *earliest* entry
    # of the day for each Ticker
    # ---------------------------
    # 1) Sort by 'date' ascending
    todays_entries.sort_values('date', inplace=True)
    # 2) Drop duplicates on ['Ticker'], keep='first'
    #    so we only have the earliest row for each Ticker
    todays_entries.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)

    # Ensure required columns
    required_cols = [
        "Ticker","date","Entry Type","Trend Type","Price",
        "Daily Change %","VWAP","ADX","Signal_ID","logtime"
    ]
    for col in required_cols:
        if col not in todays_entries.columns:
            todays_entries[col] = ""

    # Separate Bullish vs Bearish to assign targets
    if "Trend Type" in todays_entries.columns:
        bullish = todays_entries[todays_entries["Trend Type"]=="Bullish"].copy()
        bearish = todays_entries[todays_entries["Trend Type"]=="Bearish"].copy()

        # Example logic
        if not bullish.empty:
            bullish["Target Price"] = bullish["Price"] * 1.01
            bullish["Quantity"] = (20000 / bullish["Price"]).astype(int)
            bullish["Total value"] = bullish["Quantity"] * bullish["Price"]

        if not bearish.empty:
            bearish["Target Price"] = bearish["Price"] * 0.99
            bearish["Quantity"] = (20000 / bearish["Price"]).astype(int)
            bearish["Total value"] = bearish["Quantity"] * bearish["Price"]

        combined_entries = pd.concat([bullish, bearish], ignore_index=True)
    else:
        combined_entries = todays_entries.copy()

    # Convert 'date' again and re-sort
    combined_entries['date'] = pd.to_datetime(combined_entries['date'], errors='coerce')
    combined_entries = normalize_time(combined_entries, timezone='Asia/Kolkata')
    combined_entries.sort_values('date', inplace=True)

    # If 'logtime' missing, fill
    if 'logtime' not in combined_entries.columns:
        combined_entries['logtime'] = datetime.now(india_tz).isoformat()

    # Merge into existing papertrade CSV
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        existing = normalize_time(existing, timezone='Asia/Kolkata')
        existing.sort_values('date', inplace=True)

        # Combine, drop duplicates on [Ticker, date]
        all_paper = pd.concat([existing, combined_entries], ignore_index=True)
        all_paper.drop_duplicates(subset=['Ticker','date'], keep='last', inplace=True)
        all_paper.sort_values('date', inplace=True)
        all_paper.to_csv(output_file, index=False)
    else:
        combined_entries.to_csv(output_file, index=False)

    print(f"Papertrade data (only earliest Ticker entries) saved to '{output_file}'")

    # Debug output
    try:
        papertrade_data = pd.read_csv(output_file)
        print("\nPapertrade Data for Today (Earliest Ticker entries):")
        print(papertrade_data)
    except Exception as e:
        logging.error(f"Error reading {output_file}: {e}")
        print(f"Error reading {output_file}: {e}")




def main():
    """
    1) Determine today's (or last) trading day
    2) Find signals for each ticker from existing main_indicators
    3) Write them to price_action_entries_s_15min_{today}.csv
    4) Update 'Entry Signal' in main_indicators.csv
    5) Then create papertrade_s_{today}.csv from those signals
    """
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')

    # Step 2: gather signals
    signals_df = find_price_action_entries_for_today(last_trading_day)
    entries_filename = f"price_action_entries_s_15min_{today_str}.csv"
    papertrade_filename = f"papertrade_s_{today_str}.csv"

    if signals_df.empty:
        print(f"No signals detected for {last_trading_day}. Exiting.")
        return

    # Step 3: write them to entries CSV
    signals_df.to_csv(entries_filename, index=False)
    print(f"Wrote {len(signals_df)} signals to '{entries_filename}'.")
    print("\nDetected Signals for Today:")
    print(signals_df)

    # Step 4: Update 'Entry Signal' in main_indicators.csv
    for _, signal in signals_df.iterrows():
        ticker = signal['Ticker']
        main_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
        mark_signals_in_main_csv(ticker, [signal.to_dict()], main_path, india_tz)

    # Step 5: create papertrade
    create_papertrade_file(entries_filename, papertrade_filename, last_trading_day)
    print(f"\nPapertrade file => {papertrade_filename}")



def continuous_run():
    """
    Runs the 'main()' function in a continuous loop every 15 seconds,
    until user interrupts (Ctrl+C) or an exception breaks out.
    """
    while True:
        try:
            main()          # the function that does your daily signals logic
            time.sleep(15)  # wait 15 seconds
        except KeyboardInterrupt:
            print("User interrupted. Exiting continuous run.")
            break
        except Exception as e:
            logging.error(f"Error in continuous_run: {e}")
            # If some error occurs, wait briefly and then try again
            time.sleep(15)

if __name__=="__main__":


    # Then run in a continuous mode
    continuous_run()


