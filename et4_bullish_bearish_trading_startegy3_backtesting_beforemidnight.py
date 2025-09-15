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
from et1_stock_tickers import shares

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
        reference_date = dt.date.today() - timedelta(days=0)  # Start from yesterday

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
    Identify potential trading entries, classifying stocks as Bullish or Bearish.
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

    past_5_days = last_trading_day - timedelta(days=5)
    logging.info(f"Fetching data from {past_5_days} to {last_trading_day}")
    entries = []

    for ticker, token in shares_tokens.items():
        try:
            logging.info(f"Processing ticker: {ticker}")

            # Fetch intraday data
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

            # Bullish Screener
            if (
                daily_change > 1.0 and
                data['RSI'].iloc[-1] > 65 and
                data['close'].iloc[-1] > data['VWAP'].iloc[-1] and
                data['volume'].iloc[-1] > 2.2 * data['volume'].rolling(window=5).mean().iloc[-1] and
                data['ATR'].iloc[-1] > 1.4
            ):
                trend_type = "Bullish"

                for i in range(1, len(data)):
                    current_row = data.iloc[i]
                    recent_high = data['high'].iloc[max(0, i-5):i].max()

                 # Relaxed pullback conditions
                    if (
                        current_row['low'] <= current_row['VWAP'] * 1.03 and  # Wider deviation from VWAP
                        current_row['low'] >= current_row['VWAP'] * 0.97 and
                        current_row['close'] > current_row['open']  # Bullish close
                    ):
                        entries.append({
                            "Ticker": ticker,
                            "Time": current_row.name,
                            "Entry Type": "Pullback",
                            "Trend Type": trend_type,
                            "Price": current_row['close'],
                            "VWAP": current_row['VWAP'],
                            "Daily Change %": daily_change,
                        })

                    if (
                        current_row['close'] > recent_high and
                        current_row['volume'] > 1.2 * data['volume'].rolling(window=5).mean().iloc[i]  # Lower volume threshold
                    ):
                        entries.append({
                            "Ticker": ticker,
                            "Time": current_row.name,
                            "Entry Type": "Breakout",
                            "Trend Type": trend_type,
                            "Price": current_row['close'],
                            "Recent High": recent_high,
                            "Daily Change %": daily_change,
                        })

            # Bearish Screener
            elif (
                daily_change < -1.0 and
                data['RSI'].iloc[-1] < 35 and
                data['close'].iloc[-1] < data['VWAP'].iloc[-1] and
                data['volume'].iloc[-1] > 2.2 * data['volume'].rolling(window=5).mean().iloc[-1] and
                data['ATR'].iloc[-1] > 1.4
            ):
                trend_type = "Bearish"

                for i in range(1, len(data)):
                    current_row = data.iloc[i]
                    recent_low = data['low'].iloc[max(0, i-5):i].min()

            # Relaxed pullback conditions
                    if (
                       current_row['high'] >= current_row['VWAP'] * 0.97 and  # Wider deviation from VWAP
                       current_row['high'] <= current_row['VWAP'] * 1.03 and
                       current_row['close'] < current_row['open']  # Bearish close
                    ):
                        entries.append({
                            "Ticker": ticker,
                            "Time": current_row.name,
                            "Entry Type": "Pullback",
                            "Trend Type": trend_type,
                            "Price": current_row['close'],
                            "VWAP": current_row['VWAP'],
                            "Daily Change %": daily_change,
                        })

            # Relaxed breakdown conditions
                    if (
                        current_row['close'] < recent_low and
                        current_row['volume'] > 1.2 * data['volume'].rolling(window=5).mean().iloc[i]  # Lower volume threshold
                    ):
                        entries.append({
                            "Ticker": ticker,
                            "Time": current_row.name,
                            "Entry Type": "Breakdown",
                            "Trend Type": trend_type,
                            "Price": current_row['close'],
                            "Recent Low": recent_low,
                            "Daily Change %": daily_change,
                        })

        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")

    logging.info("Price action entry detection completed.")
    return pd.DataFrame(entries)



def create_papertrade_file(input_file, output_file, last_trading_day):
    """
    Reads price action entries from a CSV file and creates a papertrade file for bullish and bearish strategies.
    Includes only entries after 9:20 AM on the last trading day.

    Args:
    - input_file (str): Path to the file with price action entries.
    - output_file (str): Path to save the papertrade file.
    - last_trading_day (datetime.date): The last trading day being processed.
    """
    try:
        logging.info(f"Reading data from {input_file}.")
        price_action_entries = pd.read_csv(input_file)

        # Ensure required columns exist and add default values for missing columns
        required_columns = [
            "Ticker", "Time", "Entry Type", "Trend Type"
        ]
        missing_cols = [col for col in required_columns if col not in price_action_entries.columns]

        # Add default values for missing columns
        if missing_cols:
            logging.warning(f"Missing columns in input file: {missing_cols}. Adding default values.")
            for col in missing_cols:
                price_action_entries[col] = None  # Add NaN or a suitable default value

        # Convert 'Time' column to datetime
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
            logging.warning(f"No valid entries for {last_trading_day} after 9:20 AM.")
            print(f"No valid entries for {last_trading_day} after 9:20 AM.")
            return

        # Sort by time and get the earliest entry for each ticker
        earliest_entries = filtered_entries.sort_values(by="Time").groupby("Ticker").first().reset_index()

        # Calculate target prices, quantities, and total values based on trend type
        if "Trend Type" in earliest_entries.columns:
            bullish_entries = earliest_entries[earliest_entries["Trend Type"] == "Bullish"]
            bearish_entries = earliest_entries[earliest_entries["Trend Type"] == "Bearish"]
            
            # Create copies of the slices to modify them
            bullish_entries = bullish_entries.copy()
            bearish_entries = bearish_entries.copy()
            
            # Bullish processing: set Price and Total value to 0.0
            if not bullish_entries.empty:
                bullish_entries.loc[:, "Target Price"] = bullish_entries["Price"] * 1.01
                bullish_entries.loc[:, "Quantity"] = (20000 / bullish_entries["Price"]).astype(int)
                bullish_entries.loc[:, "Total value"] = bullish_entries["Quantity"] * bullish_entries["Price"]


            # Bearish processing: set Price and Total value to 0.0
            if not bearish_entries.empty:
                bearish_entries.loc[:, "Target Price"] = bearish_entries["Price"] * 0.99
                bearish_entries.loc[:, "Quantity"] = (20000 / bearish_entries["Price"]).astype(int)
                bearish_entries.loc[:, "Total value"] = bearish_entries["Quantity"] * bearish_entries["Price"]


            # Combine bullish and bearish entries
            combined_entries = pd.concat([bullish_entries, bearish_entries], ignore_index=True)

        else:
            logging.warning("Trend Type column is missing in the input file. Skipping processing.")
            print("Trend Type column is missing in the input file. Skipping processing.")
            return

        # Save the combined results to the output file
        combined_entries.to_csv(output_file, index=False)
        logging.info(f"Combined papertrade data saved to {output_file}")
        print(f"Combined papertrade data saved to {output_file}")

    except Exception as e:
        logging.error(f"Error creating combined papertrade file: {e}")
        print(f"Error creating combined papertrade file: {e}")



def run_for_last_30_trading_days():
    """
    Iterates over the last 30 trading days, fetching data, identifying entries,
    and creating combined paper trade files for both bullish and bearish strategies.
    """
    # Start with a reference date to identify trading days
    reference_date = dt.date.today() - timedelta(days=0)  # Start from yesterday
    
    for day in range(2):  # Iterate over the last 30 trading days
        logging.info(f"Processing for trading day: {reference_date}")
        
        # Get the last trading day
        last_trading_day = get_last_trading_day(reference_date)
        logging.info(f"Last trading day: {last_trading_day}")
        
        # Run find_price_action_entry for the current trading day
        price_action_entries = find_price_action_entry(last_trading_day)
        
        entries_filename = f"price_action_entries_15min_{last_trading_day}.csv"
        papertrade_filename = f"papertrade_{last_trading_day}.csv"

        if not price_action_entries.empty:
            # Save price_action_entries CSV for the current trading day
            price_action_entries.to_csv(entries_filename, index=False)
            logging.info(f"Entries detected. Saved results to {entries_filename}.")
            print(f"Entries detected and saved to '{entries_filename}'.")
            
            # Create the combined papertrade file for bullish and bearish strategies
            create_papertrade_file(
                input_file=entries_filename,
                output_file=papertrade_filename,
                last_trading_day=last_trading_day
            )
        else:
            # No entries detected: create empty files with placeholder headers
            logging.info(f"No entries detected for {last_trading_day}. Creating empty files.")
            print(f"No entries detected for {last_trading_day}. Creating empty files.")

            # Save an empty price_action_entries file
            pd.DataFrame(columns=[
                "Ticker", "Time", "Price", "Quantity", 
                "Total value", "Trend Type"
            ]).to_csv(entries_filename, index=False)
            
            # Create an empty papertrade file
            pd.DataFrame(columns=[
                "Ticker", "Time", "Price", "Quantity", 
                "Total value", "Target Price"
            ]).to_csv(papertrade_filename, index=False)
            
            logging.info(f"Empty files created: {entries_filename} and {papertrade_filename}")

        # Move the reference date back by one day for the next iteration
        reference_date -= timedelta(days=1)

# Call the function to process the last 30 trading days
run_for_last_30_trading_days()







