# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:01:27 2024

@author: Saarit
"""

import datetime as dt
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Importing the shares list from et1_select_stocklist.py
from et4_nfo_stock_tickers import shares

# Define the correct path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)


# Set up logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_script.log"),  # Log to file
        logging.StreamHandler()                    # Log to console
    ]
)

# Test logging initialization
logging.info("Logging has been initialized successfully.")
print("Logging initialized and script execution started.")

# Known market holidays for 2024
market_holidays = [
    dt.date(2024, 1, 26),  # Republic Day
    dt.date(2024, 3, 8),   # Mahashivratri
    dt.date(2024, 3, 25),  # Holi
    dt.date(2024, 3, 29),  # Good Friday
    dt.date(2024, 4, 11),  # Id-Ul-Fitr (Ramadan Eid)
    dt.date(2024, 4, 17),  # Shri Ram Navami
    dt.date(2024, 5, 1),   # Maharashtra Day
    dt.date(2024, 6, 17),  # Bakri Id
    dt.date(2024, 7, 17),  # Moharram
    dt.date(2024, 8, 15),  # Independence Day
    dt.date(2024, 10, 2),  # Gandhi Jayanti
    dt.date(2024, 11, 1),  # Diwali Laxmi Pujan
    dt.date(2024, 11, 15), # Gurunanak Jayanti
    dt.date(2024, 11, 20), # Maha elections
    dt.date(2024, 12, 25), # Christmas
]

def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day, starting from the given reference date.
    If no reference date is provided, default to yesterday.
    """
    if reference_date is None:
        reference_date = dt.date.today() - timedelta(days=1)  # Start from yesterday

    logging.info(f"Checking for the last trading day. Reference date: {reference_date}.")
    print(f"Reference Date: {reference_date}")
    
    while reference_date.weekday() >= 5 or reference_date in market_holidays:  # 5=Saturday, 6=Sunday
        logging.debug(f"{reference_date} is a holiday or weekend.")
        reference_date -= timedelta(days=1)
        
    logging.info(f"Last trading day determined as: {reference_date}")
    print(f"Last Trading Day: {reference_date}")
    return reference_date


def setup_kite_session():
    """
    Initialize Kite Connect session.
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

# Setup Kite session
kite = setup_kite_session()

def get_tokens_for_stocks(stocks):
    """
    Fetch tokens for the provided stock symbols.
    """
    try:
        logging.info("Fetching tokens for the provided stocks.")
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        tokens = instrument_df[instrument_df['tradingsymbol'].isin(stocks)][['tradingsymbol', 'instrument_token']]
        logging.info(f"Successfully fetched tokens for {len(tokens)} stocks.")
        print(f"Tokens fetched for {len(tokens)} stocks.")
        return dict(zip(tokens['tradingsymbol'], tokens['instrument_token']))
    except Exception as e:
        logging.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        raise

# Read tokens for shares from et1_select_stocklist.py
shares_tokens = get_tokens_for_stocks(shares)


def find_price_action_entry(last_trading_day):
    """
    Identify potential trading entries based on stringent criteria, including daily change and price action.
    """
    def calculate_rsi(series, window=14):
        """Calculate RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(data, window=14):
        """Calculate Average True Range (ATR)."""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()

    def calculate_daily_change(data):
        """Calculate daily percentage change."""
        return ((data['close'].iloc[-1] - data['open'].iloc[0]) / data['open'].iloc[0]) * 100

    # Define past_5_days relative to last_trading_day
    past_5_days = last_trading_day - timedelta(days=5)
    logging.info(f"Fetching data from {past_5_days} to {last_trading_day}")
    entries = []

    for ticker, token in shares_tokens.items():
        try:
            logging.info(f"Processing ticker: {ticker}")
            print(f"Fetching data for {ticker}...")

            # Fetch intraday data for the past 5 days
            data = pd.DataFrame(
                kite.historical_data(
                    instrument_token=token,
                    from_date=past_5_days.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=(last_trading_day + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    interval="15minute"
                )
            )

            if data.empty:
                logging.warning(f"No data available for {ticker}. Skipping.")
                continue

            logging.info(f"Data fetched successfully for {ticker}")

            # Check for required columns
            required_columns = ['date', 'close', 'volume', 'high', 'low', 'open']
            if not all(col in data.columns for col in required_columns):
                logging.warning(f"Missing required columns for {ticker}. Skipping.")
                continue

            # Convert and set 'date' column
            data['date'] = pd.to_datetime(data['date'])
            if data['date'].dt.tz is None:
                data['date'] = data['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            else:
                data['date'] = data['date'].dt.tz_convert('Asia/Kolkata')
            data.set_index('date', inplace=True)

            # Calculate indicators
            data['RSI'] = calculate_rsi(data['close'])
            data['ATR'] = calculate_atr(data)
            data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            data['20_SMA'] = data['close'].rolling(window=20).mean()
            data['Recent_High'] = data['high'].rolling(window=5).max()
            data['Recent_Low'] = data['low'].rolling(window=5).min()

            daily_change = calculate_daily_change(data)

            # Screener filters
            if (
                daily_change > 1.0 and
                data['RSI'].iloc[-1] > 60 and
                data['close'].iloc[-1] > data['VWAP'].iloc[-1] and
                data['volume'].iloc[-1] > 2 * data['volume'].rolling(window=5).mean().iloc[-1] and
                data['ATR'].iloc[-1] > 1.2
            ):
                logging.info(f"Screener passed for {ticker}")
                print(f"Screener passed for {ticker}")

                # Pullback Entry
                for i in range(1, len(data)):
                    current_row = data.iloc[i]

                    if (
                        current_row['low'] <= current_row['VWAP'] * 1.02 and
                        current_row['low'] >= current_row['VWAP'] * 0.98 and
                        current_row['close'] > current_row['open']
                    ):
                        entries.append({
                            "Ticker": ticker,
                            "Time": current_row.name,
                            "Entry Type": "Pullback",
                            "Buy Price": current_row['close'],
                            "VWAP": current_row['VWAP'],
                            "Daily Change %": daily_change,
                        })

                    # Breakout Entry
                    recent_high = data['high'].iloc[max(0, i-5):i].max()
                    if (
                        current_row['close'] > recent_high and
                        current_row['volume'] > 1.5 * data['volume'].rolling(window=5).mean().iloc[i]
                    ):
                        entries.append({
                            "Ticker": ticker,
                            "Time": current_row.name,
                            "Entry Type": "Breakout",
                            "Buy Price": current_row['close'],
                            "Recent High": recent_high,
                            "Daily Change %": daily_change,
                        })

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")
            print(f"Error processing ticker {ticker}: {e}")

    logging.info("Price action entry detection completed.")
    print("Price action entry detection completed.")
    return pd.DataFrame(entries)


def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads price action entries from a CSV file and creates a papertrade.csv with target prices.
    Includes only entries after 9:20 AM on the last trading day.
    """
    try:
        logging.info(f"Reading data from {input_file}.")
        price_action_entries = pd.read_csv(input_file)

        # Define required and optional columns
        required_columns = ["Ticker", "Time", "Entry Type", "Buy Price", "20_SMA", "Recent_Low"]
        missing_cols = [col for col in required_columns if col not in price_action_entries.columns]

        # Add default values for missing columns
        if missing_cols:
            logging.warning(f"Missing columns in input file: {missing_cols}. Adding default values.")
            for col in missing_cols:
                price_action_entries[col] = None  # Add NaN or a suitable default value

        # Process data
        price_action_entries["Time"] = pd.to_datetime(price_action_entries["Time"])

        # Define 9:20 AM threshold for filtering
        threshold_time = pd.Timestamp(
            datetime.combine(last_trading_day, datetime.min.time()) + timedelta(hours=9, minutes=20),
            tz="Asia/Kolkata"
        )

        # Filter entries for the last trading day and after 9:20 AM
        filtered_entries = price_action_entries[
            (price_action_entries["Time"].dt.date == last_trading_day) &
            (price_action_entries["Time"] >= threshold_time)
        ]

        if filtered_entries.empty:
            logging.warning(f"No entries found for {last_trading_day} after 9:20 AM.")
            print(f"No entries found for {last_trading_day} after 9:20 AM.")
            return

        # Earliest entry for each ticker
        earliest_entries = filtered_entries.sort_values(by="Time").groupby("Ticker").first().reset_index()
        earliest_entries["Target Price"] = earliest_entries["Buy Price"] * 1.01
        earliest_entries["Quantity"] = (20000 / earliest_entries["Buy Price"]).astype(int)
        earliest_entries["Total Value Bought"] = earliest_entries["Quantity"] * earliest_entries["Buy Price"]

        # Save
        earliest_entries.to_csv(output_file, index=False)
        logging.info(f"Papertrade data saved to {output_file}")
        print(f"Papertrade data saved to {output_file}")

    except Exception as e:
        logging.error(f"Error creating papertrade file: {e}")
        print(f"Error creating papertrade file: {e}")




def run_for_last_30_trading_days():
    """
    Iterates over the last 10 trading days, fetching data, identifying entries,
    and creating paper trade files for each day.
    """
    # Start with yesterday as the reference date
    reference_date = dt.date.today() - timedelta(days=1)

    for day in range(30):
        logging.info(f"Processing for trading day: {reference_date}")
        
        # Get the last trading day
        last_trading_day = get_last_trading_day(reference_date)
        logging.info(f"Last trading day: {last_trading_day}")

        # Run find_price_action_entry for the current trading day
        price_action_entries = find_price_action_entry(last_trading_day)
        
        if not price_action_entries.empty:
            # Save price_action_entries CSV for the current trading day
            entries_filename = f"price_action_entries_5min_{last_trading_day}.csv"
            price_action_entries.to_csv(entries_filename, index=False)
            logging.info(f"Entries detected. Saved results to {entries_filename}.")
            print(f"Entries detected and saved to '{entries_filename}'.")
        else:
            logging.info(f"No entries detected for {last_trading_day}.")
            print(f"No entries detected for {last_trading_day}.")
            entries_filename = None  # No entries for this day, no file created
        
        # Run create_papertrade_file for the current trading day, if entries exist
        if entries_filename:
            papertrade_filename = f"papertrade_{last_trading_day}.csv"
            create_papertrade_file(input_file=entries_filename, output_file=papertrade_filename, last_trading_day=last_trading_day)
        
        # Move the reference date back by one day for the next iteration
        reference_date -= timedelta(days=1)

# Call the function to process the last 10 trading days
run_for_last_30_trading_days()







