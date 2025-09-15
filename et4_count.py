import os
import glob
import pandas as pd
from datetime import datetime, timedelta

# Define the directory path
directory = "main_indicators_new"

# Create the pattern for matching CSV files in the directory
pattern = os.path.join(directory, "*_main_indicators*.csv")
csv_files = glob.glob(pattern)

# Print the total number of CSV files in the directory
total_files = len(csv_files)
print(f"Total CSV files in directory '{directory}': {total_files}\n")

recent_count = 0

for file in csv_files:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Could not read file {file}: {e}")
        continue

    if df.empty:
        print(f"{file} is empty. Skipping.")
        continue

    # Parse the 'date' column as datetime.
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Error parsing dates in file {file}: {e}")
        continue

    # Extract the latest date from the 'date' column.
    last_date = df['date'].max()

    # If last_date is tz-aware, create a tz-aware current datetime.
    if last_date.tzinfo is not None:
        now = datetime.now(last_date.tzinfo)
    else:
        now = datetime.now()

    # Check if the last_date is within the past 15 minutes.
    if now - last_date <= timedelta(minutes=15):
        print(f"File {file} is recent (last date: {last_date}).")
        recent_count += 1
    else:
        print(f"File {file} is not recent (last date: {last_date}).")

print(f"\nNumber of recent CSV files (with a candle in the last 15 minutes): {recent_count}")


# Print the total number of CSV files in the directory
total_files = len(csv_files)
print(f"Total CSV files in directory '{directory}': {total_files}\n")