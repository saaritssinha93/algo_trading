import os
import glob
import pandas as pd
from datetime import datetime
import pytz

# Define directories
source_dir = "main_indicators2"
dest_dir = "main_indicators"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Define the starting datetime for new data (timezone-aware)
# Here we use the given timestamp with +05:30 offset.
start_datetime = pd.Timestamp("2025-03-06 10:00:00+05:30")

# Get list of all source CSV files
source_files = glob.glob(os.path.join(source_dir, "*_main_indicators.csv"))

for source_file in source_files:
    # Assume file naming convention: {ticker}_main_indicators.csv
    basename = os.path.basename(source_file)
    ticker = basename.split("_")[0]
    
    # Define destination filename
    dest_file = os.path.join(dest_dir, f"{ticker}_main_indicators_updated.csv")
    
    try:
        # Read source CSV
        df_source = pd.read_csv(source_file)
    except Exception as e:
        print(f"Error reading {source_file}: {e}")
        continue

    # Convert 'date' column to datetime (assumed to be in a proper format)
    try:
        # First, parse as UTC if not tz-aware then convert to Asia/Kolkata
        df_source['date'] = pd.to_datetime(df_source['date'], errors='coerce', utc=True)
        df_source['date'] = df_source['date'].dt.tz_convert("Asia/Kolkata")
    except Exception as e:
        print(f"Error converting date column in {source_file}: {e}")
        continue

    # Filter rows with date >= start_datetime
    df_new = df_source[df_source['date'] >= start_datetime]
    if df_new.empty:
        print(f"No new rows to append for {ticker} from {source_file}")
        continue

    # If the destination file exists, read existing data and append new rows.
    if os.path.exists(dest_file):
        try:
            df_dest = pd.read_csv(dest_file)
            # Convert the destination date column similarly
            df_dest['date'] = pd.to_datetime(df_dest['date'], errors='coerce', utc=True)
            df_dest['date'] = df_dest['date'].dt.tz_convert("Asia/Kolkata")
        except Exception as e:
            print(f"Error reading destination file {dest_file}: {e}")
            df_dest = pd.DataFrame()
        # Append new rows to the destination DataFrame
        df_combined = pd.concat([df_dest, df_new], ignore_index=True)
    else:
        # If destination file does not exist, use the new rows directly.
        df_combined = df_new.copy()
    
    # Write the combined DataFrame back to the destination file
    try:
        df_combined.to_csv(dest_file, index=False)
        print(f"Appended {len(df_new)} new rows for {ticker} to {dest_file}")
    except Exception as e:
        print(f"Error writing to {dest_file}: {e}")
