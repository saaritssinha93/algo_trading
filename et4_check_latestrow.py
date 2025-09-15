# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:54:01 2024

Author: Saarit
Description:
    This script checks for the latest expected 15-minute interval entries
    in all *_main_indicators.csv files in a given directory. It aligns
    timestamps to Asia/Kolkata (+05:30) so that both the CSV entries and
    the expected interval have a timezone-aware datetime.

    Additional logic:
    - If the latest expected 15-min interval entry exists, check the "Entry Signal" column.
    - Only print tickers where "Entry Signal" is missing (not "Yes").
"""

import os
import glob
from datetime import datetime
import pandas as pd
import argparse
import pytz

def get_latest_expected_interval(current_time=None, tz='Asia/Kolkata'):
    """
    Given a current_time (datetime), find the most recent 15-minute interval
    in the specified timezone.

    For example:
    - If current_time is 9:40, the interval is 9:30.
    - If current_time is 10:10, the interval is 10:00.
    - If current_time is 12:20, the interval is 12:15.
    """
    timezone = pytz.timezone(tz)
    if current_time is None:
        current_time = datetime.now(timezone)
    else:
        if current_time.tzinfo is None:
            current_time = timezone.localize(current_time)
        else:
            current_time = current_time.astimezone(timezone)

    minute_block = (current_time.minute // 15) * 15
    expected_interval = current_time.replace(minute=minute_block, second=0, microsecond=0)
    return expected_interval

def check_csv_intervals(data_dir='main_indicators', current_time=None, tz='Asia/Kolkata'):
    """
    Reads all *_main_indicators.csv files in the given directory, extracts the ticker from the filename,
    and checks if the last expected 15-min interval entry exists in the CSV.

    Returns a dictionary: { 'TICKER': True/False, ... } indicating whether the latest expected
    entry was found for each ticker.
    """
    expected_interval = get_latest_expected_interval(current_time, tz=tz)
    pattern = os.path.join(data_dir, '*_main_indicators.csv')
    files = glob.glob(pattern)
    timezone = pytz.timezone(tz)

    results = {}
    for f in files:
        ticker = os.path.basename(f).replace('_main_indicators.csv', '')
        
        # Read the CSV
        try:
            df = pd.read_csv(f)
        except Exception:
            results[ticker] = False
            continue
        
        if 'date' not in df.columns:
            results[ticker] = False
            continue
        
        try:
            df['date'] = pd.to_datetime(df['date'], utc=False)
            if df['date'].dt.tz is None:
                df['date'] = df['date'].dt.tz_localize(timezone)
            else:
                df['date'] = df['date'].dt.tz_convert(timezone)
        except Exception:
            results[ticker] = False
            continue
        
        df['floored_date'] = df['date'].dt.floor('15min')
        interval_exists = (df['floored_date'] == expected_interval).any()
        
        results[ticker] = interval_exists
    return results

def main():
    parser = argparse.ArgumentParser(description="Check CSV intervals for tickers (timezone aware).")
    parser.add_argument("--data_dir", type=str, default="main_indicators",
                        help="Directory containing *_main_indicators.csv files.")
    parser.add_argument("--timezone", type=str, default="Asia/Kolkata",
                        help="Timezone for matching intervals.")
    args = parser.parse_args()
    
    current_time = datetime.now()  # naive current time
    results = check_csv_intervals(data_dir=args.data_dir, current_time=current_time, tz=args.timezone)

    tz = args.timezone
    timezone = pytz.timezone(tz)
    expected_interval = get_latest_expected_interval(current_time, tz=tz)

    for ticker, exists in results.items():
        file_path = os.path.join(args.data_dir, f"{ticker}_main_indicators.csv")
        
        if not exists:
            print(f"[{ticker}] No entry found for the expected interval ({expected_interval}), so missing Entry Signal.")
        else:
            # Interval exists, now check the Entry Signal
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'], utc=False)
                if df['date'].dt.tz is None:
                    df['date'] = df['date'].dt.tz_localize(timezone)
                else:
                    df['date'] = df['date'].dt.tz_convert(timezone)

                df['floored_date'] = df['date'].dt.floor('15min')
                matched = df[df['floored_date'] == expected_interval]

                if matched.empty:
                    # Should not happen if exists is True, but in case
                    print(f"[{ticker}] Interval found previously, but now no matching rows? Possibly no Entry Signal.")
                    continue

                # Get the latest row for that interval
                latest_row = matched.iloc[-1]
                entry_signal = latest_row.get('Entry Signal', None)

                # Check the entry signal
                if entry_signal is None or str(entry_signal).strip().lower() in ['', 'nan']:
                    # Print ticker only if Entry Signal is missing
                    print(f"[{ticker}] Entry Signal: missing at {expected_interval}.")
                # If Entry Signal == "Yes", we do NOT print anything as per request.

            except Exception as e:
                # If there's any error reading or processing, assume no entry signal info
                print(f"[{ticker}] Error processing file, assuming missing Entry Signal: {e}")

if __name__ == "__main__":
    main()
