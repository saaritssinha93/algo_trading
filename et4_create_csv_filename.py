# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:13:43 2024

@author: Saarit
"""

import os
import csv

# List of tickers that need CSV files
tickers = [
    "INDOUS"
]

# We import a list of stocks from another file
#from et4_filtered_stocks_market_cap import selected_stocks

# Directory where the CSV files should be created
indicators_dir = "main_indicators"

# Ensure the directory exists
os.makedirs(indicators_dir, exist_ok=True)

# Define the headers
headers = [
    "date", "open", "high", "low", "close", "volume",
    "RSI", "ATR", "MACD", "Signal_Line", "Histogram",
    "Upper_Band", "Lower_Band", "Stochastic",
    "VWAP", "20_SMA", "Recent_High", "Recent_Low",
    "ADX", "Entry Signal", "Signal_ID", "logtime"
]

# Create each CSV file with headers
for ticker in tickers:
    filename = f"{ticker}_main_indicators.csv"
    filepath = os.path.join(indicators_dir, filename)
    
    # Check if file already exists to avoid overwriting
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        print(f"Created {filename} with headers.")
    else:
        print(f"{filename} already exists. Skipping creation.")
