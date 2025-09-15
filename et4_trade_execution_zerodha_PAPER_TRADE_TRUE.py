# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:01:37 2024

Author: Saarit
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect import KiteConnect
import pytz
import threading
import time
import signal
import sys
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import uuid
import csv
import traceback

# ================================
# Configuration Parameters
# ================================

PAPER_TRADE = True
CSV_DIRECTORY = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
CSV_FILENAME_PATTERN = "papertrade_{}.csv"
CSV_FILES_TO_MONITOR = [
    "papertrade_2024-12-20.csv"
]

EXECUTED_SIGNALS_FILE = "executed_signals.json"



# ================================
# Setup Logging
# ================================

cwd = CSV_DIRECTORY
os.chdir(cwd)  # Change the current working directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_script.log"),
        logging.StreamHandler()
    ]
)

logging.info("Logging initialized.")
print("Logging initialized and script execution started.")

# ================================
# Kite Connect Session Setup
# ================================

def setup_kite_session():
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()

        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().strip().split()
        if len(key_secret) < 2:
            raise ValueError("API key and secret must be provided in 'api_key.txt' separated by space.")
        api_key, api_secret = key_secret[0], key_secret[1]

        kite = KiteConnect(api_key=api_key)
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

kite = setup_kite_session()

# ================================
# Utility Functions
# ================================

def load_executed_signals():
    if os.path.exists(EXECUTED_SIGNALS_FILE):
        with open(EXECUTED_SIGNALS_FILE, 'r') as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                logging.error("Executed signals file is corrupted. Starting with an empty set.")
                return set()
    return set()

def save_executed_signals():
    with open(EXECUTED_SIGNALS_FILE, 'w') as f:
        json.dump(list(executed_signals), f)


# Initialize executed_signals
executed_signals = load_executed_signals()

# Initialize a set to track tickers already simulated for today
simulated_tickers = set()


def generate_signal_id(ticker, date, transaction_type):
    unique_id = uuid.uuid4()
    return f"{ticker}-{date.isoformat()}-{transaction_type}-{unique_id}"

def log_simulation_result(ticker, transaction_type, price, quantity, target_price, stop_loss_price, order_id):
    india_tz = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
    simulation_filename = f"simulation_papertrade_{today_str}.csv"

    # Create the row of data
    row_data = {
        "Timestamp": datetime.now(india_tz).isoformat(),
        "Ticker": ticker,
        "Transaction Type": transaction_type,
        "Quantity": quantity,  # <--- NEW
        "Executed Price": price,
        "Target Price": target_price,
        "Stop Loss Price": stop_loss_price,
        "Order ID": order_id
    }

    # Append to CSV
    file_exists = os.path.exists(simulation_filename)
    df = pd.DataFrame([row_data])
    df.to_csv(
        simulation_filename,
        mode='a',
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        doublequote=True
    )


# ================================
# Read and Normalize CSV Function
# ================================

def read_and_normalize_csv(file_path, expected_columns, timezone='Asia/Kolkata'):
    """
    Reads a CSV file, ensures it has the expected columns, and normalizes the 'date' column.

    Parameters:
        file_path (str): Path to the CSV file.
        expected_columns (list): List of expected column names.
        timezone (str): Timezone for 'logtime'.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    try:
        logging.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(
            file_path,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            on_bad_lines='warn',  # Log and skip malformed lines
            engine='python'        # Use Python engine for better handling
        )

        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing columns {missing_cols} in {file_path}. Adding them with default values.")
            for col in missing_cols:
                if col == 'logtime':
                    india_tz = pytz.timezone(timezone)
                    current_logtime = datetime.now(india_tz).isoformat()
                    df[col] = current_logtime
                else:
                    df[col] = ""  # Assign default empty string or appropriate default

        # Handle extra columns by keeping only the expected ones
        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            df = df.drop(columns=extra_columns)
            logging.warning(f"Dropped extra columns {extra_columns} from {file_path}")

        # Normalize 'date' column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df['date'] = df['date'].dt.tz_convert(timezone)
        else:
            logging.error(f"'date' column missing in {file_path}. Cannot normalize time.")
            df['date'] = pd.NaT  # Assign Not-a-Time

        # Drop rows with NaT in 'date' after normalization
        before_drop = len(df)
        df = df.dropna(subset=['date'])
        after_drop = len(df)
        if before_drop != after_drop:
            logging.warning(f"Dropped {before_drop - after_drop} rows with invalid 'date' in {file_path}")

        # Assign unique Signal_ID if missing
        if 'Signal_ID' not in df.columns:
            df['Signal_ID'] = df.apply(
                lambda row: generate_signal_id(row['Ticker'], row['date'], row['Trend Type']),
                axis=1
            )
            logging.info(f"Assigned 'Signal_ID' for all rows in {file_path}")
        else:
            # For rows with missing Signal_ID, assign one
            missing_signal_ids = df['Signal_ID'].isna()
            if missing_signal_ids.any():
                df.loc[missing_signal_ids, 'Signal_ID'] = df.loc[missing_signal_ids].apply(
                    lambda row: generate_signal_id(row['Ticker'], row['date'], row['Trend Type']),
                    axis=1
                )
                logging.info(f"Assigned missing 'Signal_ID' in {file_path}")

        # Reorder columns to match expected_columns
        df = df[expected_columns]

        return df

    except pd.errors.ParserError as e:
        logging.error(f"Pandas ParserError while reading {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"Unexpected error while reading {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise

# ================================
# Write CSV Function
# ================================

def write_csv(df, file_path, expected_columns):
    """
    Writes a DataFrame to a CSV file with consistent formatting.

    Parameters:
        df (pd.DataFrame): The DataFrame to write.
        file_path (str): Destination CSV file path.
        expected_columns (list): List of expected column names.
    """
    try:
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""

        # Reorder columns
        df = df[expected_columns]

        # Write to CSV with consistent quoting
        df.to_csv(
            file_path,
            index=False,
            quoting=csv.QUOTE_ALL,    # Enclose all fields in quotes
            escapechar='\\',
            doublequote=True
        )
        logging.info(f"Successfully wrote to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write to {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise

# ================================
# execute_paper_trades Function
# ================================

# Add a global set to track executed trades (by Signal_ID)
# Already initialized earlier as executed_signals

expected_columns = [
    "Ticker", "date", "Trend Type", "Signal_ID",
    "Price"
]

def execute_paper_trades(last_trading_day, kite, api_semaphore, paper_trade=True):
    filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    if not os.path.exists(filename):
        logging.error(f"File {filename} does not exist.")
        print(f"File {filename} does not exist.")
        return

    try:
        df = read_and_normalize_csv(filename, expected_columns)
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        print(f"Error reading {filename}: {e}")
        return

    india_tz = pytz.timezone('Asia/Kolkata')

    # Filter out trades that have already been executed
    new_trades_df = df[~df['Signal_ID'].isin(executed_signals)]

    if new_trades_df.empty:
        logging.info("No new trades found to execute.")
        print("No new trades found to execute.")
        return

    # Process only new trades
    for index, row in new_trades_df.iterrows():
        try:
            ticker = row['Ticker']
            time_entry = row['date']
            trend_type = row['Trend Type']
            price = float(row['Price'])
            quantity = int(row.get('Quantity', 1))  # Default to 1 if Quantity not provided
            signal_id = row['Signal_ID']

            # Determine transaction type based on trend_type
            if trend_type.lower() == 'bullish':
                transaction_type = 'BUY'
                target_price = round(price * 1.02, 2)  # 2% Target
                stop_loss_price = round(price * 0.98, 2)  # 2% Stop Loss
            elif trend_type.lower() == 'bearish':
                transaction_type = 'SELL'
                target_price = round(price * 0.98, 2)  # 2% Target
                stop_loss_price = round(price * 1.02, 2) # 2% Stop Loss
            else:
                logging.warning(f"Unknown Trend Type '{trend_type}' for Ticker '{ticker}'. Skipping.")
                print(f"Unknown Trend Type '{trend_type}' for Ticker '{ticker}'. Skipping.")
                continue

            exchange = 'NSE'
            tradingsymbol = ticker
            product = kite.PRODUCT_MIS
            order_type = kite.ORDER_TYPE_LIMIT
            validity = kite.VALIDITY_DAY
            variety = kite.VARIETY_REGULAR
            tag = 'PaperTrade'

            if paper_trade:
                # Check if ticker has been simulated already
                if ticker in simulated_tickers:
                    logging.info(f"Ticker {ticker} has already been simulated. Skipping duplicate entry.")
                    print(f"Ticker {ticker} has already been simulated. Skipping duplicate entry.")
                    continue  # Skip this trade if already simulated

                # Simulate order placement
                simulated_order_id = f"PAPER-{signal_id}-{datetime.now(india_tz).strftime('%H%M%S')}"
                logging.info(
                    f"[SIMULATION] Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                    f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Simulated Order ID: {simulated_order_id}"
                    )
                print(
                    f"[SIMULATION] Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                    f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Simulated Order ID: {simulated_order_id}"
                    )

                # Log the simulation result
                log_simulation_result(
                    ticker,
                    transaction_type,
                    price,
                    quantity,   # Pass the quantity here
                    target_price,
                    stop_loss_price,
                    simulated_order_id
                    )

                # Add this signal_id to executed_signals
                executed_signals.add(signal_id)
                # Mark ticker as simulated to avoid duplicates
                simulated_tickers.add(ticker)

            else:
                # For live trading, check if ticker has already been processed
                if ticker in simulated_tickers:
                    logging.info(f"Ticker {ticker} has already been traded. Skipping duplicate live order.")
                    print(f"Ticker {ticker} has already been traded. Skipping duplicate live order.")
                    continue  # Skip placing this trade
                    
                with api_semaphore:
                    try:
                        order_id = kite.place_order(
                            variety=variety,
                            exchange=exchange,
                            tradingsymbol=tradingsymbol,
                            transaction_type=transaction_type,
                            quantity=quantity,
                            product=product,
                            order_type=order_type,
                            price=price,
                            validity=validity,
                            squareoff=target_price,
                            stoploss=stop_loss_price,
                            trailing_stoploss=0,
                            tag=tag
                        )

                        logging.info(
                            f"Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                            f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Order ID: {order_id}"
                        )
                        print(
                            f"Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                            f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Order ID: {order_id}"
                        )

                        # Add this signal_id to executed_signals after successful order placement
                        executed_signals.add(signal_id)
                        
                        # Mark ticker as traded to avoid duplicate live trades
                        simulated_tickers.add(ticker)

                    except Exception as e:
                        logging.error(f"Failed to place order for {ticker}: {e}")
                        print(f"Failed to place order for {ticker}: {e}")

        except Exception as e:
            logging.error(f"Error processing new trade row {index} in {filename}: {e}")
            print(f"Error processing new trade row {index} in {filename}: {e}")
            continue

    # Save the executed signals after processing all trades
    save_executed_signals()

    logging.info("Finished processing new trades.")
    print("Finished processing new trades.")

# ================================
# CSVChangeHandler Class
# ================================

class CSVChangeHandler(FileSystemEventHandler):
    def __init__(self, target_file, callback):
        super().__init__()
        self.target_file = target_file
        self.callback = callback

    def on_modified(self, event):
        if not event.is_directory and os.path.basename(event.src_path) == self.target_file:
            self.callback()

# ================================
# main Function
# ================================

def main():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Updated start and end times
    start_time_naive = datetime.combine(now.date(), datetime_time(9, 31, 0))  # 9:31:00 AM
    end_time_naive = datetime.combine(now.date(), datetime_time(18, 30, 0))   # 3:30:00 PM

    start_time = india_tz.localize(start_time_naive)
    end_time = india_tz.localize(end_time_naive)

    if now > end_time:
        logging.warning("Current time is past the trading window. Exiting the script.")
        print("Current time is past the trading window. Exiting the script.")
        return
    elif now > start_time:
        logging.info("Current time is within the trading window. Scheduling remaining jobs.")
    else:
        logging.info("Current time is before the trading window. Waiting to start scheduling.")

    # Generate scheduled times at 1-minute intervals
    scheduled_times = []
    current_time = start_time
    while current_time <= end_time:
        scheduled_times.append(current_time.strftime("%H:%M:%S"))
        current_time += timedelta(minutes=1)

    # Determine last trading day
    def get_last_trading_day():
        last_day = datetime.now(india_tz).date()
        while last_day.weekday() >= 5 or last_day in market_holidays:
            last_day -= timedelta(days=1)
        return last_day.strftime('%Y-%m-%d')

    market_holidays = [
        datetime(2024, 1, 26).date(),
        datetime(2024, 3, 8).date(),
        datetime(2024, 3, 25).date(),
        datetime(2024, 3, 29).date(),
        datetime(2024, 4, 11).date(),
        datetime(2024, 4, 17).date(),
        datetime(2024, 5, 1).date(),
        datetime(2024, 6, 17).date(),
        datetime(2024, 7, 17).date(),
        datetime(2024, 8, 15).date(),
        datetime(2024, 10, 2).date(),
        datetime(2024, 11, 1).date(),
        datetime(2024, 11, 15).date(),
        datetime(2024, 11, 20).date(),
        datetime(2024, 12, 25).date(),
    ]

    last_trading_day = get_last_trading_day()
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last trading day: {last_trading_day}")

    api_semaphore = threading.Semaphore(2)

    for scheduled_time in scheduled_times:
        # Parse the scheduled_time string back to a datetime object for today
        scheduled_datetime_naive = datetime.strptime(scheduled_time, "%H:%M:%S")
        scheduled_datetime = india_tz.localize(datetime.combine(now.date(), scheduled_datetime_naive.time()))
        
        if scheduled_datetime > now:
            schedule.every().day.at(scheduled_time).do(
                execute_paper_trades,
                last_trading_day=last_trading_day,
                kite=kite,
                api_semaphore=api_semaphore,
                paper_trade=PAPER_TRADE
            )
            logging.info(f"Scheduled job at {scheduled_time} IST for last_trading_day: {last_trading_day}")
        else:
            logging.debug(f"Skipped scheduling job at {scheduled_time} IST as it's already past.")

    logging.info("Scheduled execute_paper_trades to run every minute starting at 9:31:00 AM IST up to 3:30:00 PM IST.")
    print("Scheduled execute_paper_trades to run every minute starting at 9:31:00 AM IST up to 3:30:00 PM IST.")

    def execute_trades_callback():
        logging.info("CSV file change detected after debounce delay. Executing trades.")
        print("CSV file change detected after debounce delay. Executing trades.")
        execute_paper_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    def on_csv_change():
        global debounce_timer
        if debounce_timer:
            debounce_timer.cancel()
        debounce_timer = threading.Timer(DEBOUNCE_DELAY, execute_trades_callback)
        debounce_timer.start()

    DEBOUNCE_DELAY = 5  # seconds
    debounce_timer = None

    def setup_file_observer():
        observers = []
        for csv_file in CSV_FILES_TO_MONITOR:
            event_handler = CSVChangeHandler(target_file=csv_file, callback=on_csv_change)
            observer = Observer()
            observer.schedule(event_handler, path=CSV_DIRECTORY, recursive=False)
            observer.start()
            logging.info(f"Started file observer for {csv_file}.")
            print(f"Started file observer for {csv_file}.")
            observers.append(observer)
        return observers

    observers = setup_file_observer()

    # Execute immediately if within trading window
    if start_time <= now <= end_time:
        logging.info("Executing initial run of trading strategy.")
        print("Executing initial run of trading strategy.")
        execute_paper_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        for observer in observers:
            observer.stop()
        logging.info("Shutting down file observers and exiting script.")
        print("Shutting down file observers and exiting script.")
    for observer in observers:
        observer.join()

if __name__ == "__main__":
    main()
