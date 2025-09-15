# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:41:31 2025

@author: Saarit
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect import KiteConnect
import pytz
import threading
import time
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

# Global lock for synchronizing trade execution
trade_lock = threading.Lock()

# ================================
# Setup Logging
# ================================

os.chdir(CSV_DIRECTORY)  # Change the current working directory

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

def generate_signal_id(ticker, date, transaction_type):
    unique_id = uuid.uuid4()
    return f"{ticker}-{date.isoformat()}-{transaction_type}-{unique_id}"

def log_simulation_result(ticker, transaction_type, price, target_price, stop_loss_price, order_id):
    """
    Appends a single trade record to simulation_papertrade_{todaydate}.csv.
    """
    india_tz = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
    simulation_filename = f"simulation_papertrade_{today_str}.csv"

    row_data = {
        "Timestamp": datetime.now(india_tz).isoformat(),
        "Ticker": ticker,
        "Transaction Type": transaction_type,
        "Executed Price": price,
        "Target Price": target_price,
        "Stop Loss Price": stop_loss_price,
        "Order ID": order_id
    }

    file_exists = os.path.exists(simulation_filename)
    df = pd.DataFrame([row_data])
    df.to_csv(simulation_filename, mode='a', header=not file_exists, index=False,
              quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)

# ================================
# Read and Normalize CSV Function
# ================================

def read_and_normalize_csv(file_path, expected_columns, timezone='Asia/Kolkata'):
    try:
        logging.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(
            file_path,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            on_bad_lines='warn',
            engine='python'
        )

        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing columns {missing_cols} in {file_path}. Adding them with default values.")
            for col in missing_cols:
                df[col] = ""  # Add missing columns with default empty values

        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            df = df.drop(columns=extra_columns)
            logging.warning(f"Dropped extra columns {extra_columns} from {file_path}")

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df['date'] = df['date'].dt.tz_convert(timezone)
        else:
            logging.error(f"'date' column missing in {file_path}. Cannot normalize time.")
            df['date'] = pd.NaT

        before_drop = len(df)
        df = df.dropna(subset=['date'])
        after_drop = len(df)
        if before_drop != after_drop:
            logging.warning(f"Dropped {before_drop - after_drop} rows with invalid 'date' in {file_path}")

        if 'Signal_ID' not in df.columns:
            df['Signal_ID'] = df.apply(
                lambda row: generate_signal_id(row['Ticker'], row['date'], row['Trend Type']),
                axis=1
            )
            logging.info(f"Assigned 'Signal_ID' for all rows in {file_path}")
        else:
            missing_signal_ids = df['Signal_ID'].isna()
            if missing_signal_ids.any():
                df.loc[missing_signal_ids, 'Signal_ID'] = df.loc[missing_signal_ids].apply(
                    lambda row: generate_signal_id(row['Ticker'], row['date'], row['Trend Type']),
                    axis=1
                )
                logging.info(f"Assigned missing 'Signal_ID' in {file_path}")

        before_dedup = len(df)
        df = df.drop_duplicates(subset=['Signal_ID'])
        after_dedup = len(df)
        if before_dedup != after_dedup:
            logging.info(f"Removed {before_dedup - after_dedup} duplicate rows based on 'Signal_ID'")

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
    try:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        df = df[expected_columns]
        df.to_csv(
            file_path,
            index=False,
            quoting=csv.QUOTE_ALL,
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

    with trade_lock:
        new_trades_df = df[~df['Signal_ID'].isin(executed_signals)]

        if new_trades_df.empty:
            logging.info("No new trades found to execute.")
            print("No new trades found to execute.")
            return

        india_tz = pytz.timezone('Asia/Kolkata')

        # Process only new trades
        for index, row in new_trades_df.iterrows():
            try:
                ticker = row['Ticker']
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
                    stop_loss_price = round(price * 1.02, 2)  # 2% Stop Loss
                else:
                    logging.warning(f"Unknown Trend Type '{trend_type}' for Ticker '{ticker}'. Skipping.")
                    continue

                if paper_trade:
                    simulated_order_id = f"PAPER-{signal_id}-{datetime.now(india_tz).strftime('%H%M%S')}"
                    logging.info(
                        f"[SIMULATION] Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                        f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Simulated Order ID: {simulated_order_id}"
                    )
                    print(
                        f"[SIMULATION] Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                        f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Simulated Order ID: {simulated_order_id}"
                    )

                    log_simulation_result(ticker, transaction_type, price, target_price, stop_loss_price, simulated_order_id)

                    executed_signals.add(signal_id)

                else:
                    with api_semaphore:
                        try:
                            order_id = kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange='NSE',
                                tradingsymbol=ticker,
                                transaction_type=transaction_type,
                                quantity=quantity,
                                product=kite.PRODUCT_MIS,
                                order_type=kite.ORDER_TYPE_LIMIT,
                                price=price,
                                validity=kite.VALIDITY_DAY,
                                squareoff=target_price,
                                stoploss=stop_loss_price,
                                trailing_stoploss=0,
                                tag='PaperTrade'
                            )

                            logging.info(
                                f"Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                                f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Order ID: {order_id}"
                            )
                            print(
                                f"Placed {transaction_type} order for {ticker} at {price} with Target {target_price} "
                                f"and Stop Loss {stop_loss_price}. Signal_ID: {signal_id}, Order ID: {order_id}"
                            )

                            executed_signals.add(signal_id)

                        except Exception as e:
                            logging.error(f"Failed to place order for {ticker}: {e}")
                            print(f"Failed to place order for {ticker}: {e}")

            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
                print(f"Error processing row {index}: {e}")
                continue

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

    start_time_naive = datetime.combine(now.date(), datetime_time(9, 31, 0))
    end_time_naive = datetime.combine(now.date(), datetime_time(18, 30, 0))

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

    scheduled_times = []
    current_time = start_time
    while current_time <= end_time:
        scheduled_times.append(current_time.strftime("%H:%M:%S"))
        current_time += timedelta(minutes=1)

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
        nonlocal debounce_timer
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
