# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:35:11 2025

@author: Saarit
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

def add_daily_change_for_directory(
    directory: str = "main_indicators",
    start_str: str = "2025-01-17 09:15:00+05:30",
    end_str: str   = "2025-02-27 13:30:00+05:30"
):
    """
    Reads all CSV files in the specified directory (e.g., main_indicators1) that match the
    naming pattern *_main_indicators_updated.csv. Each file is expected to have columns:
      date, open, high, low, close, volume, RSI, ATR, MACD, Signal_Line, Histogram,
      Upper_Band, Lower_Band, Stochastic, VWAP, 20_SMA, Recent_High, Recent_Low, ADX,
      Entry Signal, logtime, Signal_ID, CalcFlag
    Adds a column 'Daily_Change' for all 15-min rows in the specified date/time range
    (start_str to end_str, inclusive).

    The formula for Daily_Change in each 15-min candle is:
         Daily_Change = ((close - prev_day_close) / prev_day_close) * 100

    Where prev_day_close is the last 15-min candle close from the previous trading day.
    If no previous day data is found, Daily_Change is set to NaN for that day.

    Overwrites each CSV with the new 'Daily_Change' column.
    """

    # Parse the start/end times for filtering
    start_time = pd.to_datetime(start_str)
    end_time   = pd.to_datetime(end_str)

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist. Aborting.")
        return

    # Pattern for matching the CSV files we want to process
    csv_pattern = os.path.join(directory, "*_main_indicators_updated.csv")
    all_csvs = glob.glob(csv_pattern)

    if not all_csvs:
        print(f"No matching CSVs found in '{directory}'.")
        return

    for csv_path in all_csvs:
        # Load CSV
        df = pd.read_csv(csv_path)

        # Check that 'date' column exists
        if 'date' not in df.columns:
            print(f"Skipping {csv_path}; missing 'date' column.")
            continue

        # Parse dates (these may already have timezone info; if not, we localize)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop any rows where date parsing failed
        df = df.dropna(subset=['date'])

        # Sort by date ascending
        df.sort_values('date', inplace=True)

        # Restrict to the desired 15-min timeframe from start_time to end_time
        mask = (df['date'] >= start_time) & (df['date'] <= end_time)
        df_in_range = df.loc[mask].copy()

        if df_in_range.empty:
            # If there's no data in that range, just continue to next file
            print(f"No rows in desired date range for {csv_path}")
            # Optionally continue or write file with no changes
            continue

        # Add an empty column 'Daily_Change' if it doesn't exist, else we'll overwrite
        if 'Daily_Change' not in df.columns:
            df['Daily_Change'] = np.nan

        # =====================
        # Compute Daily_Change
        # =====================
        # Strategy:
        # 1) Group the in-range data by date (yyyy-mm-dd).
        # 2) For each day, find the "previous trading day's last close" and
        #    set the Daily_Change for all rows in that day.
        # 3) If there's no previous trading day in the data, set NaN.

        # We'll do it for the in-range subset, then put those results back.

        # Create a 'date_only' to group by day
        df_in_range['date_only'] = df_in_range['date'].dt.date

        # We'll get the unique days in ascending order
        unique_days = sorted(df_in_range['date_only'].unique())

        # Build a helper dictionary: day -> last close from that day
        day_to_last_close = {}

        # First, we want to fill day_to_last_close for all days in the entire DF
        # to get continuity from the previous day *before* the start_time
        # so that if start_time is in the middle of a day, we still can find
        # the previous day's last close.

        # Make sure the full DF (not just in_range) is also grouped
        df['date_only'] = df['date'].dt.date
        df_listed_days  = sorted(df['date_only'].unique())

        for d in df_listed_days:
            daily_data = df[df['date_only'] == d]
            if not daily_data.empty:
                # last close for day d
                day_to_last_close[d] = daily_data.iloc[-1]['close']

        # Now we iterate over each day in df_in_range, find the previous day's close
        # from day_to_last_close, and update daily_change
        daily_changes = []

        for day_idx, day_val in enumerate(unique_days):
            # If it's the first day in in_range, there's no previous day in this subset.
            # But we might have a previous day in the overall data. That day is the
            # one just before day_val in df_listed_days.
            # We'll find day_val's index in df_listed_days and subtract 1
            if day_val in df_listed_days:
                idx_in_full = df_listed_days.index(day_val)
                if idx_in_full == 0:
                    # No prior day in the entire dataset => daily change = NaN
                    prev_day_close = np.nan
                else:
                    # We do have a previous day
                    prev_day = df_listed_days[idx_in_full - 1]
                    prev_day_close = day_to_last_close.get(prev_day, np.nan)
            else:
                # Should not happen if day_val is from the data, but just in case
                prev_day_close = np.nan

            # For each row in day_val, daily_change = (close - prev_day_close) / prev_day_close * 100
            day_mask = (df_in_range['date_only'] == day_val)
            if np.isnan(prev_day_close) or prev_day_close == 0:
                df_in_range.loc[day_mask, 'Daily_Change'] = np.nan
            else:
                df_in_range.loc[day_mask, 'Daily_Change'] = (
                    (df_in_range.loc[day_mask, 'close'] - prev_day_close) / prev_day_close
                ) * 100

        # Now we have updated 'Daily_Change' values for the in-range subset in df_in_range.
        # We merge these changes back into df (the full data).
        # Easiest approach: for the in-range rows, update df['Daily_Change'] by matching on index.

        # We should ensure the indexes line up. Let's set or keep them in sync:
        df_in_range_idx = df_in_range.index
        df.loc[df_in_range_idx, 'Daily_Change'] = df_in_range['Daily_Change']

        # Clean up
        df.drop(columns=['date_only'], inplace=True, errors='ignore')
        df_in_range.drop(columns=['date_only'], inplace=True, errors='ignore')

        # Sort by date again (not strictly necessary if we kept order, but safe)
        df.sort_values('date', inplace=True)

        # Overwrite the CSV
        df.to_csv(csv_path, index=False)
        print(f"Updated 'Daily_Change' in file: {os.path.basename(csv_path)}")

def main():
    # Call the function to update daily change in the specified directory
    add_daily_change_for_directory(
        directory="main_indicators",
        start_str="2025-01-17 09:15:00+05:30",
        end_str="2025-02-27 13:30:00+05:30"
    )

if __name__ == "__main__":
    main()
