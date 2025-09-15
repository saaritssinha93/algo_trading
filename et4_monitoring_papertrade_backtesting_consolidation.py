# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:56:01 2024

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
    
    
def consolidate_results(last_trading_days, output_file="consolidated_results.csv"):
    """
    Reads all result_final_{last_trading_day}.csv files, extracts overall tickers
    (Overall, Overall Bearish, and Overall Bullish), their net percentages, and the corresponding date,
    and stores them in a single file.

    Args:
    - last_trading_days (list): List of trading days to process.
    - output_file (str): Name of the output file to save the consolidated data.

    Returns:
    - None
    """
    consolidated_data = []

    for trading_day in last_trading_days:
        file_path = f"result_final_{trading_day}.csv"
        try:
            # Read the result_final file for the trading day
            df = pd.read_csv(file_path)

            # Filter rows for Overall, Overall Bearish, and Overall Bullish
            for trend in ["Overall", "Overall Bearish", "Overall Bullish"]:
                if trend in df["Ticker"].values:
                    overall_row = df[df["Ticker"] == trend].iloc[0]
                    consolidated_data.append({
                        "Date": trading_day,
                        "Ticker": trend,
                        "Net Percentage": overall_row["Net Percentage"]
                    })

        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping this trading day.")
            continue
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # Create a DataFrame from the consolidated data
    if consolidated_data:
        consolidated_df = pd.DataFrame(consolidated_data, columns=["Date", "Ticker", "Net Percentage"])
        consolidated_df.to_csv(output_file, index=False)
        print(f"Consolidated results saved to {output_file}")
    else:
        print("No data to consolidate.")


def analyze_consolidated_results(input_file="consolidated_results.csv"):
    """
    Reads consolidated_results.csv, calculates the sum and average of Net Percentages
    for Overall, Overall Bearish, and Overall Bullish, and prints the results.
    Additionally, counts and prints the number of unique entries (dates) across all trend types.

    Args:
    - input_file (str): Path to the consolidated results file.

    Returns:
    - None
    """
    try:
        # Read the consolidated results file
        df = pd.read_csv(input_file)

        # Ensure Net Percentage is numerical
        df['Net Percentage'] = pd.to_numeric(df['Net Percentage'])

        # Count unique dates
        unique_dates_count = df['Date'].nunique()

        # Print the number of unique dates
        print(f"\nTrading days: {unique_dates_count}")

        # Filter rows by Ticker and calculate totals and averages for each trend
        for trend in ["Overall", "Overall Bearish", "Overall Bullish"]:
            trend_df = df[df['Ticker'] == trend]
            total_net_percentage = trend_df['Net Percentage'].sum()
            count_trend = len(trend_df)
            average_net_percentage = total_net_percentage / count_trend if count_trend > 0 else 0

            print(f"\nTotal Net Percentage for '{trend}': {total_net_percentage:.4f}")
            print(f"Average Net Percentage for '{trend}': {average_net_percentage:.4f}")

    except FileNotFoundError:
        print(f"File {input_file} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")






def recalculate_overall(last_trading_day):
    """
    Recalculates the overall net percentage based on the sum of net percentages for bullish and bearish rows.
    It then updates the 'result_final_{last_trading_day}.csv' file with the new overall values, replacing existing ones.
    
    Args:
    - last_trading_day (str): The trading day for which the overall calculation is performed.
    
    Returns:
    - None
    """
    file_path = f"result_final_{last_trading_day}.csv"
    try:
        # Read the result_final file for the trading day
        df = pd.read_csv(file_path)

        # Ensure 'Net Percentage' is numerical
        df['Net Percentage'] = pd.to_numeric(df['Net Percentage'], errors='coerce')

        # Filter bearish and bullish rows, excluding 'Overall Bearish' and 'Overall Bullish'
        bearish_rows = df[(df['Trend Type'] == 'Bearish') & (df['Ticker'] != 'Overall Bearish') & (df['Ticker'] != 'All')]
        bullish_rows = df[(df['Trend Type'] == 'Bullish') & (df['Ticker'] != 'Overall Bullish') & (df['Ticker'] != 'All')]

        bearish_count = len(bearish_rows)
        bullish_count = len(bullish_rows)

        # Check if bearish_count or bullish_count is zero and avoid division by zero
        if bearish_count > 0:
            total_bearish_net_percentage = bearish_rows['Net Percentage'].sum() / bearish_count
        else:
            total_bearish_net_percentage = 0

        if bullish_count > 0:
            total_bullish_net_percentage = bullish_rows['Net Percentage'].sum() / bullish_count
        else:
            total_bullish_net_percentage = 0

        # Calculate the overall net percentage
        overall_net_percentage = (
            (total_bearish_net_percentage * bearish_count) + 
            (total_bullish_net_percentage * bullish_count)
        ) / (bearish_count + bullish_count) if (bearish_count + bullish_count) > 0 else 0

        # Round to two decimal places
        overall_net_percentage = round(overall_net_percentage, 2)
        total_bearish_net_percentage = round(total_bearish_net_percentage, 2)
        total_bullish_net_percentage = round(total_bullish_net_percentage, 2)

        # Remove the existing 'Overall', 'Overall Bearish', and 'Overall Bullish' rows if they exist
        df = df[~df['Ticker'].isin(['Overall', 'Overall Bearish', 'Overall Bullish'])]

        # Create the updated rows for Overall, Overall Bearish, and Overall Bullish
        overall_row = pd.DataFrame([{
            'Ticker': 'Overall',
            'Net Percentage': overall_net_percentage,
            'Trend Type': 'All'
        }])

        overall_bearish_row = pd.DataFrame([{
            'Ticker': 'Overall Bearish',
            'Net Percentage': total_bearish_net_percentage,
            'Trend Type': 'Bearish'
        }])

        overall_bullish_row = pd.DataFrame([{
            'Ticker': 'Overall Bullish',
            'Net Percentage': total_bullish_net_percentage,
            'Trend Type': 'Bullish'
        }])

        # Append the new overall rows to the DataFrame
        df = pd.concat([df, overall_row, overall_bearish_row, overall_bullish_row], ignore_index=True)

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False)
        print(f"Overall net percentage recalculated and replaced in {file_path}.")

    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error recalculating overall for {last_trading_day}: {e}")





# Example list of trading days
last_trading_days = [
    '2024-12-02',
    '2024-11-29', '2024-11-28', '2024-11-27', '2024-11-26', '2024-11-25',    
    '2024-11-22', '2024-11-21', '2024-11-19', '2024-11-18', '2024-11-14',
    '2024-11-13', '2024-11-12', '2024-11-11', '2024-11-08', '2024-11-07',
    '2024-11-06', '2024-11-05', '2024-11-04', '2024-10-31', '2024-10-30',
    '2024-10-29', '2024-10-28', '2024-10-25', '2024-10-23', '2024-10-22',
    '2024-10-21', '2024-10-18', '2024-10-17', '2024-10-16', '2024-10-15',
    '2024-10-14', '2024-10-11', '2024-10-10', '2024-10-09', '2024-10-08',
    '2024-10-07', '2024-10-04', '2024-10-03', '2024-10-01', '2024-09-30',
    '2024-09-27', '2024-09-26', '2024-09-25', '2024-09-24', '2024-09-23',
    '2024-09-20', '2024-09-19', '2024-09-18', '2024-09-17', '2024-09-16',
    '2024-09-13', '2024-09-12', '2024-09-11', '2024-09-10', '2024-09-09',
    '2024-09-06', '2024-09-05', '2024-09-04', '2024-09-03', '2024-08-30',
    '2024-08-29', '2024-08-28', '2024-08-27', '2024-08-26', '2024-08-23',
    '2024-08-22', '2024-08-21', '2024-08-20', '2024-08-19', '2024-08-16',
    '2024-08-14', '2024-08-13', '2024-08-12', '2024-08-09', '2024-08-08',
    '2024-08-07', '2024-08-06', '2024-08-05', '2024-08-02', '2024-08-01',
    '2024-07-31', '2024-07-30', '2024-07-29', '2024-07-26', '2024-07-25',
    '2024-07-24', '2024-07-23', '2024-07-22', '2024-07-19', '2024-07-18',
    '2024-07-17', '2024-07-16', '2024-07-15', '2024-07-12', '2024-07-11',
    '2024-07-10', '2024-07-09', '2024-07-08', '2024-07-05', '2024-07-04',
    '2024-07-03', '2024-07-02', '2024-07-01', '2024-06-28', '2024-06-27',
    '2024-06-26', '2024-06-25', '2024-06-24', '2024-06-21', '2024-06-20',
    '2024-06-19', '2024-06-18', '2024-06-17', '2024-06-14', '2024-06-13',
    '2024-06-12', '2024-06-11', '2024-06-10', '2024-06-07', '2024-06-06',
    '2024-06-05', '2024-06-04', '2024-06-03', '2024-05-31', '2024-05-30',
    '2024-05-29', '2024-05-28', '2024-05-27', '2024-05-26', '2024-05-23',
    '2024-05-22', '2024-05-21', '2024-05-20', '2024-05-19', '2024-05-18',
    '2024-05-17', '2024-05-16', '2024-05-15', '2024-05-14', '2024-05-13',
    '2024-05-12', '2024-05-11', '2024-05-10', '2024-05-09', '2024-05-08',
    '2024-05-07'
]



for trading_day in last_trading_days:
    recalculate_overall(trading_day)

# Call the function to consolidate results
consolidate_results(last_trading_days)

# Analyze the consolidated results
analyze_consolidated_results()