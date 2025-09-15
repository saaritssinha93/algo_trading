# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:47:21 2024

@author: Saarit
"""

import time
import threading
from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
import pytz
import tkinter as tk  # Import tkinter for GUI and alias as tk
from datetime import datetime
import csv




# Define the correct path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)

# Set up logging
logging.basicConfig(filename='trading_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Known market holidays for 2024 (example)
market_holidays = [
    dt.date(2024, 10, 2),  # Gandhi Jayanti
    # Add other known holidays for the year here
]

# Function to setup Kite Connect session
def setup_kite_session():
    """Establishes a Kite Connect session."""
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()

        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        return kite

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise

# Function to fetch all NSE instruments
def fetch_instruments(kite):
    """Fetches all NSE instruments and returns a DataFrame."""
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        logging.info("NSE instrument data fetched successfully.")
        return instrument_df

    except Exception as e:
        logging.error(f"Error fetching instruments: {e}")
        raise

# Function to lookup instrument token
def instrument_lookup(instrument_df, symbol):
    """Looks up instrument token for a given symbol in the instrument dump."""
    try:
        instrument_token = instrument_df[instrument_df.tradingsymbol == symbol].instrument_token.values[0]
        return instrument_token
    except IndexError:
        logging.error(f"Symbol {symbol} not found in instrument dump.")
        return -1
    except Exception as e:
        logging.error(f"Error in instrument lookup: {e}")
        return -1
    
    
def fetch_nse_indices_trend(kite, instrument_df, index_list):
    """
    Fetches the daily closing prices of specified NSE indices for the past 6 months
    and adds a trend column based on daily percentage growth:
      - Bullish: Growth >= 0.5%
      - Bearish: Growth <= -0.5%
      - Sideways: -0.5% < Growth < 0.5%

    Args:
    - kite: KiteConnect session object.
    - instrument_df: DataFrame of NSE instruments.
    - index_list: List of NSE index symbols to process.

    Returns:
    - trend_df: DataFrame with indices, their daily closing prices, and trend classification.
    """
    try:
        # Get today's date and date six months ago
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=201)

        trends = []

        for index in index_list:
            logging.info(f"Fetching historical data for index: {index}")
            try:
                # Fetch instrument token
                token = instrument_lookup(instrument_df, index)
                if token == -1:
                    logging.warning(f"Index {index} not found in instruments. Skipping.")
                    continue

                # Fetch historical data
                historical_data = kite.historical_data(
                    instrument_token=token,
                    from_date=start_date,
                    to_date=end_date,
                    interval="day"
                )

                # Convert to DataFrame
                if not historical_data:
                    logging.warning(f"No historical data found for {index}. Skipping.")
                    continue

                data = pd.DataFrame(historical_data)

                # Ensure the `date` column exists
                if 'date' not in data.columns:
                    logging.warning(f"'date' column missing for {index}. Adding manually.")
                    data['date'] = pd.to_datetime([row['date'] for row in historical_data])

                # Ensure `date` is the index
                if 'date' not in data.index.names:
                    data.set_index('date', inplace=True)

                # Calculate daily percentage change
                data['daily_change_%'] = data['close'].pct_change() * 100

                # Determine trend
                data['trend'] = np.where(
                    data['daily_change_%'] >= 0.5, 'Bullish',
                    np.where(data['daily_change_%'] <= -0.5, 'Bearish', 'Sideways')
                )

                # Append to trends
                data['index'] = index  # Add index name for identification
                trends.append(data[['index', 'close', 'daily_change_%', 'trend']].reset_index())

            except Exception as e:
                logging.error(f"Error fetching data for {index}: {e}")
                continue

        # Combine all trends into a single DataFrame
        if trends:
            trend_df = pd.concat(trends, ignore_index=True)
        else:
            logging.error("No data collected for trends. Exiting.")
            return pd.DataFrame()

        # Save to a CSV file for further analysis
        trend_df.to_csv('nse_indices_trends.csv', index=False)
        logging.info("Trend data saved to 'nse_indices_trends.csv'.")

        return trend_df

    except Exception as e:
        logging.error(f"Error fetching NSE indices trend data: {e}")
        raise

def sort_nse_indices_trends():
    """
    Sorts the nse_indices_trends.csv file by date from recent to older for each index.
    It groups the data by 'index' and sorts the dates within each index group.
    """
    try:
        # Load the CSV file into a DataFrame
        trend_df = pd.read_csv('nse_indices_trends.csv')

        if trend_df.empty:
            logging.warning("The nse_indices_trends.csv file is empty. Nothing to sort.")
            return

        # Ensure 'date' column is in datetime format
        trend_df['date'] = pd.to_datetime(trend_df['date'], errors='coerce')

        # Group by 'index' and sort each group by date in descending order
        trend_df_sorted = trend_df.groupby('index', group_keys=False).apply(lambda x: x.sort_values('date', ascending=False))

        # Save the sorted DataFrame back to CSV
        trend_df_sorted.to_csv('nse_indices_trends_sorted.csv', index=False)
        logging.info("Sorted trend data saved to 'nse_indices_trends_sorted.csv'.")

        # Print the first few rows of the sorted data for verification
        print(trend_df_sorted.head())

    except Exception as e:
        logging.error(f"Error sorting NSE indices trends: {e}")
        print("An error occurred while sorting the data. Please check the logs for details.")


def main():
    """
    Main function to fetch and display NSE indices trends, then sort the data.
    """
    logging.info("Starting the script to fetch and sort NSE indices trends...")

    # List of NSE indices symbols
    index_list = [
        "NIFTY 50", "NIFTY NEXT 50", "NIFTY MIDCAP 50", "NIFTY MIDCAP 100",
        "NIFTY SMLCAP 50", "NIFTY SMLCAP 100"
    ]

    # Setup Kite session
    kite = setup_kite_session()
    instrument_df = fetch_instruments(kite)

    # Fetch trends for NSE indices
    try:
        trend_df = fetch_nse_indices_trend(kite, instrument_df, index_list)

        # Display the data in tabular format
        print("\nNSE Indices Trends (Last 6 Months):")
        print(trend_df.to_string(index=False, columns=["index", "date", "close", "daily_change_%", "trend"]))

        # Now sort the data
        sort_nse_indices_trends()

    except Exception as e:
        logging.error(f"Error during main execution: {e}")
        print("An error occurred. Please check the logs for details.")

    logging.info("Script completed successfully.")



if __name__ == "__main__":
    main()
