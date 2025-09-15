# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:22:10 2025

@author: Saarit

Modified so that ONLY data from *_historical.csv is copied on top of
*_main_indicators_updated.csv (no reading from *_main_indicators.csv).
"""

import os
import glob
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MAIN_INDICATORS_DIR = "main_indicators"
DATA_CACHE_DIR = "data_cache"

# Key cutoff times
CUTOFF_DATETIME = "2025-01-24 09:15:00+05:30"  # historical data must be < this
MAIN_START_DATETIME = "2025-01-24 09:30:00+05:30"  # main indicators start time

# Columns in the final "updated" file
MAIN_COLUMNS = [
    "date", "open", "high", "low", "close", "volume",
    "RSI", "ATR", "MACD", "Signal_Line", "Histogram",
    "Upper_Band", "Lower_Band", "Stochastic", "VWAP",
    "20_SMA", "Recent_High", "Recent_Low", "ADX",
    "Entry Signal", "logtime", "Signal_ID", "CalcFlag"
]

# Columns in historical file
HIST_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# ------------------------------------------------------------------------------
# Find all historical CSV files: "*_historical.csv" in data_cache/
# ------------------------------------------------------------------------------
historical_files = glob.glob(os.path.join(DATA_CACHE_DIR, "*_historical.csv"))
if not historical_files:
    print("No historical files found in 'data_cache'.")
    exit()

# ------------------------------------------------------------------------------
# Process each historical file
# ------------------------------------------------------------------------------
for hist_file_path in historical_files:
    # Example: "ABC_historical.csv" -> Ticker = "ABC"
    base_name = os.path.basename(hist_file_path)
    ticker = base_name.replace("_historical.csv", "")

    # Construct the output file path => {Ticker}_main_indicators_updated.csv
    output_file_path = os.path.join(MAIN_INDICATORS_DIR, f"{ticker}_main_indicators_updated.csv")

    print(f"\nProcessing Ticker: {ticker}")
    print(f"  Historical File:      {hist_file_path}")
    print(f"  Updated Output File:  {output_file_path}")

    # --------------------------------------------------------------------------
    # Step 1: Load Historical
    # --------------------------------------------------------------------------
    try:
        hist_df = pd.read_csv(hist_file_path, parse_dates=["date"])
        hist_df = hist_df[HIST_COLUMNS]
    except Exception as e:
        print(f"[ERROR] Could not read '{hist_file_path}': {e}")
        continue

    # --------------------------------------------------------------------------
    # Step 2: Load any EXISTING _main_indicators_updated.csv
    #         If it doesn't exist, create an empty DataFrame with MAIN_COLUMNS
    # --------------------------------------------------------------------------
    if os.path.exists(output_file_path):
        try:
            updated_df = pd.read_csv(output_file_path, parse_dates=["date"])
        except Exception as e:
            print(f"[WARNING] Could not read '{output_file_path}': {e}")
            updated_df = pd.DataFrame(columns=MAIN_COLUMNS)
    else:
        updated_df = pd.DataFrame(columns=MAIN_COLUMNS)

    # Ensure updated_df has exactly MAIN_COLUMNS (adding if missing)
    for col in MAIN_COLUMNS:
        if col not in updated_df.columns:
            updated_df[col] = np.nan
    updated_df = updated_df[MAIN_COLUMNS]

    # --------------------------------------------------------------------------
    # Step 3: Filter Historical Data (before the cutoff)
    # --------------------------------------------------------------------------
    filtered_hist_df = hist_df[hist_df["date"] < CUTOFF_DATETIME].copy()

    # --------------------------------------------------------------------------
    # Step 4: Fill extra columns in filtered_hist_df with NaN
    #         so it matches the MAIN_COLUMNS structure.
    # --------------------------------------------------------------------------
    extra_cols = [col for col in MAIN_COLUMNS if col not in HIST_COLUMNS and col != "date"]
    for col in extra_cols:
        filtered_hist_df[col] = np.nan

    # Reorder columns to match MAIN_COLUMNS
    filtered_hist_df = filtered_hist_df[MAIN_COLUMNS]

    # --------------------------------------------------------------------------
    # Step 5: Combine the filtered historical with the existing "updated" DataFrame
    # --------------------------------------------------------------------------
    combined_df = pd.concat([filtered_hist_df, updated_df], ignore_index=True)

    # --------------------------------------------------------------------------
    # Step 6: Sort by Date
    # --------------------------------------------------------------------------
    combined_df.sort_values(by="date", inplace=True)

    # --------------------------------------------------------------------------
    # Step 7: Save the result back to {Ticker}_main_indicators_updated.csv
    # --------------------------------------------------------------------------
    try:
        combined_df.to_csv(output_file_path, index=False)
        print(f"  => [SUCCESS] Updated file saved to: {output_file_path}")
    except Exception as e:
        print(f"[ERROR] Could not write '{output_file_path}': {e}")
