# -*- coding: utf-8 -*-
"""
Rewrite of the merged code, removing the on-demand menu,
linked only to PAPER_TRADE=True/False and file-change triggers,
with the Quantity forced to 1 regardless of CSV input.

SL-M and target are snapped to the nearest multiple of 0.5
around the 2% offset from the entry price.
"""

import os
import pandas as pd
import logging
import time
import csv
import traceback
import json
import uuid
import threading
from datetime import datetime, timedelta, time as datetime_time

import pytz
from kiteconnect import KiteConnect
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================================
# Configuration
# ================================
PAPER_TRADE = False  # Toggle between paper trades and real trades
CSV_DIRECTORY = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
CSV_FILENAME_PATTERN = "papertrade_{}.csv"
EXECUTED_SIGNALS_FILE = "executed_signals_mis_mkt.json"

EXPECTED_COLUMNS = ["Ticker", "date", "Trend Type", "Signal_ID", "Price", "Quantity"]

# ================================
# Logging
# ================================
os.chdir(CSV_DIRECTORY)
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
# Kite Connect Session
# ================================
def setup_kite_session():
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()

        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().strip().split()
        if len(key_secret) < 2:
            raise ValueError("API key and secret must be in 'api_key.txt' separated by space.")
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
# Globals & Utilities
# ================================
def load_executed_signals():
    if os.path.exists(EXECUTED_SIGNALS_FILE):
        try:
            with open(EXECUTED_SIGNALS_FILE, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Executed signals file is corrupted. Starting with empty set.")
            return set()
    return set()

def save_executed_signals():
    with open(EXECUTED_SIGNALS_FILE, 'w') as f:
        json.dump(list(executed_signals), f)

executed_signals = load_executed_signals()

paper_traded_tickers = set()  # For paper trades
real_traded_tickers = set()   # For real trades

def generate_signal_id(ticker, date, transaction_type):
    unique_id = uuid.uuid4()
    return f"{ticker}-{date.isoformat()}-{transaction_type}-{unique_id}"

def current_market_price(ticker):
    try:
        ltp_data = kite.ltp(f"NSE:{ticker}")
        return ltp_data[f"NSE:{ticker}"]['last_price']
    except Exception as e:
        logging.error(f"Failed to fetch LTP for {ticker}: {e}")
        print(f"Failed to fetch LTP for {ticker}: {e}")
        return 0.0

def round_to_nearest_05(value: float) -> float:
    """
    Round a given price to the nearest 0.5 increment.
    Example: 101.26 -> 101.5, 101.24 -> 101.0, etc.
    """
    return round(value * 2) / 2.0

# ================================
# Logging for Paper & Real trades
# ================================
def log_simulation_result(ticker, transaction_type, price, quantity, target_price, stop_loss_price, order_id):
    india_tz = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
    sim_file = f"simulation_papertrade_{today_str}.csv"

    row_data = {
        "Timestamp": datetime.now(india_tz).isoformat(),
        "Ticker": ticker,
        "Transaction Type": transaction_type,
        "Quantity": quantity,
        "Executed Price": price,
        "Target Price": target_price,
        "Stop Loss Price": stop_loss_price,
        "Order ID": order_id
    }

    file_exists = os.path.exists(sim_file)
    df = pd.DataFrame([row_data])
    df.to_csv(
        sim_file,
        mode='a',
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        doublequote=True
    )

def log_trade_result(ticker, transaction_type, price, quantity, target_price, stop_loss_price, order_id):
    india_tz = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
    trade_file = f"trade_log_{today_str}.csv"

    row_data = {
        "Timestamp": datetime.now(india_tz).isoformat(),
        "Ticker": ticker,
        "Transaction Type": transaction_type,
        "Quantity": quantity,
        "Executed Price": price,
        "Target Price": target_price,
        "Stop Loss Price": stop_loss_price,
        "Order ID": order_id
    }

    file_exists = os.path.exists(trade_file)
    df = pd.DataFrame([row_data])
    df.to_csv(
        trade_file,
        mode='a',
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        doublequote=True
    )

# ================================
# Reading & Normalizing CSV
# ================================
def read_and_normalize_csv(file_path, expected_columns, timezone='Asia/Kolkata'):
    try:
        logging.info(f"Reading CSV: {file_path}")
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
            logging.warning(f"Missing columns {missing_cols} in {file_path}, adding defaults.")
            for mc in missing_cols:
                df[mc] = ""

        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            df = df.drop(columns=extra_cols)
            logging.warning(f"Dropped extra columns {extra_cols} from {file_path}")

        # Convert 'date' to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df['date'] = df['date'].dt.tz_convert(timezone)
        else:
            df['date'] = pd.NaT

        before_drop = len(df)
        df = df.dropna(subset=['date'])
        after_drop = len(df)
        if before_drop != after_drop:
            logging.warning(f"Dropped {before_drop - after_drop} rows with invalid 'date' in {file_path}")

        # Assign Signal_ID if missing
        if 'Signal_ID' not in df.columns:
            df['Signal_ID'] = df.apply(
                lambda r: generate_signal_id(r['Ticker'], r['date'], r['Trend Type']),
                axis=1
            )
        else:
            missing_sids = df['Signal_ID'].isna()
            if missing_sids.any():
                df.loc[missing_sids, 'Signal_ID'] = df.loc[missing_sids].apply(
                    lambda r: generate_signal_id(r['Ticker'], r['date'], r['Trend Type']),
                    axis=1
                )

        df = df[expected_columns]
        return df

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        logging.error(traceback.format_exc())
        raise

# ================================
# Real Trade MIS + SL + Target
# ================================
def place_mis_trade(kite, ticker, transaction_type, quantity):
    """
    Places 3 orders:
      1) Entry as MARKET
      2) Target as LIMIT (2% away, snapped to nearest 0.5)
      3) SL as SL-M (2% away, snapped to nearest 0.5)
    Then monitors to cancel the remaining order once one is executed.
    """
    def _log(m):
        logging.info(m)
        print(m)

    current_price = current_market_price(ticker)
    if current_price == 0.0:
        msg = f"Could not fetch LTP for {ticker}. Aborting."
        _log(msg)
        raise ValueError(msg)

    if transaction_type.upper() == 'BUY':
        raw_stop_loss_price = current_price * 0.98
        raw_target_price    = current_price * 1.02
        sl_txn  = 'SELL'
        tgt_txn = 'SELL'
    else:
        raw_stop_loss_price = current_price * 1.02
        raw_target_price    = current_price * 0.98
        sl_txn  = 'BUY'
        tgt_txn = 'BUY'

    # Snap SL and Target to nearest 0.5
    stop_loss_price = round_to_nearest_05(raw_stop_loss_price)
    target_price    = round_to_nearest_05(raw_target_price)

    # 1) Entry as MARKET
    entry_order_id = kite.place_order(
        variety='regular',
        exchange='NSE',
        tradingsymbol=ticker,
        transaction_type=transaction_type.upper(),
        quantity=quantity,
        product=kite.PRODUCT_MIS,
        order_type='MARKET',
        validity='DAY',
        tag='EntryOrder'
    )
    _log(f"[REAL] Placed {transaction_type} MARKET for {ticker}, qty={quantity}, ID={entry_order_id}")

    # Wait for fill
    filled_price = current_price
    filled = False
    while not filled:
        orders = kite.orders()
        for o in orders:
            if o['order_id'] == entry_order_id and o['status'] == 'COMPLETE':
                filled = True
                filled_price = float(o.get('average_price', current_price))
                break
        if not filled:
            time.sleep(2)

    _log(f"[REAL] Entry {entry_order_id} filled at {filled_price}")

    # 2) Target as LIMIT
    target_order_id = kite.place_order(
        variety='regular',
        exchange='NSE',
        tradingsymbol=ticker,
        transaction_type=tgt_txn,
        quantity=quantity,
        product=kite.PRODUCT_MIS,
        order_type='LIMIT',
        price=target_price,
        validity='DAY',
        tag='TargetOrder'
    )
    _log(f"[REAL] Placed {tgt_txn} LIMIT for {ticker} @ {target_price}, ID={target_order_id}")

    # 3) SL as SL-M
    stop_loss_order_id = kite.place_order(
        variety='regular',
        exchange='NSE',
        tradingsymbol=ticker,
        transaction_type=sl_txn,
        quantity=quantity,
        product=kite.PRODUCT_MIS,
        order_type='SL-M',
        trigger_price=stop_loss_price,
        validity='DAY',
        tag='StopLossOrder'
    )
    _log(f"[REAL] Placed {sl_txn} SL-M for {ticker} trigger={stop_loss_price}, ID={stop_loss_order_id}")

    # 4) Monitor to see if target or SL fills
    trade_active = True
    while trade_active:
        orders = kite.orders()

        target_filled = any(o['order_id'] == target_order_id and o['status'] == 'COMPLETE' for o in orders)
        sl_filled     = any(o['order_id'] == stop_loss_order_id and o['status'] == 'COMPLETE' for o in orders)

        if target_filled:
            # Cancel SL
            try:
                kite.cancel_order(variety='regular', order_id=stop_loss_order_id)
                _log(f"[REAL] Target hit => canceled SL {stop_loss_order_id}")
            except Exception as ce:
                _log(f"[REAL] Error canceling SL {stop_loss_order_id}: {ce}")
            trade_active = False

        elif sl_filled:
            # Cancel Target
            try:
                kite.cancel_order(variety='regular', order_id=target_order_id)
                _log(f"[REAL] SL triggered => canceled Target {target_order_id}")
            except Exception as ce:
                _log(f"[REAL] Error canceling Target {target_order_id}: {ce}")
            trade_active = False

        else:
            time.sleep(5)

    return {
        "entry_order_id": entry_order_id,
        "entry_fill_price": filled_price,
        "stop_loss_price": stop_loss_price,
        "target_price": target_price
    }

# ================================
# Master Execute Trades
# ================================
def execute_trades(last_trading_day, kite, api_semaphore, paper_trade=True):
    """
    Reads the CSV `papertrade_{last_trading_day}.csv`.
    If paper_trade => sim logs, else => real MIS+SL+target (2% from LTP, snapped to 0.5).
    """
    filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    file_path = os.path.join(CSV_DIRECTORY, filename)

    if not os.path.exists(file_path):
        msg = f"File {file_path} does not exist. No trades."
        logging.error(msg)
        print(msg)
        return

    try:
        df = read_and_normalize_csv(file_path, EXPECTED_COLUMNS)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        print(f"Error reading {file_path}: {e}")
        return

    # Filter out signals that have already been executed
    new_df = df[~df['Signal_ID'].isin(executed_signals)]
    if new_df.empty:
        logging.info("No new trades to execute.")
        print("No new trades to execute.")
        return

    india_tz = pytz.timezone('Asia/Kolkata')

    for idx, row in new_df.iterrows():
        try:
            ticker = row['Ticker']
            trend_type = row['Trend Type'].lower().strip()
            signal_id = row['Signal_ID']
            csv_price = float(row['Price'])

            # Force quantity = 1 always
            quantity = 1

            if trend_type == 'bullish':
                transaction_type = 'BUY'
            elif trend_type == 'bearish':
                transaction_type = 'SELL'
            else:
                print(f"Unknown Trend Type '{trend_type}' for {ticker}. Skipping.")
                logging.info(f"Unknown Trend Type '{trend_type}' for {ticker}. Skipping.")
                continue

            if paper_trade:
                # PAPER
                if ticker in paper_traded_tickers:
                    logging.info(f"{ticker} already paper-traded. Skipping.")
                    print(f"{ticker} already paper-traded. Skipping.")
                    continue

                # 2% SL from CSV Price, then snapped to 0.5 (optional if needed for paper trades)
                if transaction_type == 'BUY':
                    raw_sl_price = csv_price * 0.98
                else:
                    raw_sl_price = csv_price * 1.02
                stop_loss_price = round_to_nearest_05(raw_sl_price)

                sim_id = f"SIM-{signal_id}-{datetime.now(india_tz).strftime('%H%M%S')}"
                logging.info(f"[SIM] {transaction_type} {ticker}, price={csv_price}, SL={stop_loss_price}, ID={sim_id}")
                print(f"[SIM] {transaction_type} {ticker}, price={csv_price}, SL={stop_loss_price}, ID={sim_id}")

                # Log simulation
                log_simulation_result(
                    ticker=ticker,
                    transaction_type=transaction_type,
                    price=csv_price,
                    quantity=quantity,
                    target_price=None,
                    stop_loss_price=stop_loss_price,
                    order_id=sim_id
                )

                executed_signals.add(signal_id)
                paper_traded_tickers.add(ticker)

            else:
                # REAL
                if ticker in real_traded_tickers:
                    logging.info(f"{ticker} already real-traded. Skipping.")
                    print(f"{ticker} already real-traded. Skipping.")
                    continue

                with api_semaphore:
                    try:
                        trade_info = place_mis_trade(kite, ticker, transaction_type, quantity)
                        log_trade_result(
                            ticker=ticker,
                            transaction_type=transaction_type,
                            price=trade_info['entry_fill_price'],
                            quantity=quantity,
                            target_price=trade_info['target_price'],
                            stop_loss_price=trade_info['stop_loss_price'],
                            order_id=trade_info['entry_order_id']
                        )
                        executed_signals.add(signal_id)
                        real_traded_tickers.add(ticker)

                    except Exception as ex:
                        msg = f"Real trade failed for {ticker}: {ex}"
                        logging.error(msg)
                        print(msg)

        except Exception as ee:
            logging.error(f"Error processing row {idx}: {ee}")
            print(f"Error processing row {idx}: {ee}")

    save_executed_signals()
    logging.info("Finished processing new trades.")
    print("Finished processing new trades.")

# ================================
# Watchdog Handler
# ================================
class CSVChangeHandler(FileSystemEventHandler):
    def __init__(self, target_file, callback, debounce_sec=5):
        super().__init__()
        self.target_file = target_file
        self.callback = callback
        self.debounce_sec = debounce_sec
        self.timer = None

    def on_modified(self, event):
        if not event.is_directory and os.path.basename(event.src_path) == self.target_file:
            logging.info(f"Detected modification in {event.src_path}")
            print(f"Detected modification in {event.src_path}")
            self._debounce()

    def _debounce(self):
        if self.timer and self.timer.is_alive():
            self.timer.cancel()
        self.timer = threading.Timer(self.debounce_sec, self.callback)
        self.timer.start()

# ================================
# main
# ================================
def main():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Trading window
    start_naive = datetime.combine(now.date(), datetime_time(9, 31, 0))
    end_naive   = datetime.combine(now.date(), datetime_time(18, 30, 0))
    start_time  = india_tz.localize(start_naive)
    end_time    = india_tz.localize(end_naive)

    if now < start_time:
        logging.info("Before trading window. We'll wait/monitor for file changes, no immediate trades.")
        print("Before trading window. We'll wait/monitor for file changes, no immediate trades.")
    elif now > end_time:
        logging.warning("Past trading window. Will still react to file changes, but be mindful.")
        print("Past trading window. Will still react to file changes, but be mindful.")
    else:
        logging.info("Within trading window. Ready to trade if CSV changes.")
        print("Within trading window. Ready to trade if CSV changes.")

    # Market holidays for 2024
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

    def get_last_trading_day():
        d = datetime.now(india_tz).date()
        while d.weekday() >= 5 or d in market_holidays:
            d -= timedelta(days=1)
        return d.strftime('%Y-%m-%d')

    last_trading_day = get_last_trading_day()
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last trading day: {last_trading_day}")

    api_semaphore = threading.Semaphore(2)

    # Watchdog
    csv_filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    full_path = os.path.join(CSV_DIRECTORY, csv_filename)

    def file_change_callback():
        logging.info("CSV changed -> reading file & executing trades.")
        print("CSV changed -> reading file & executing trades.")
        execute_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    handler = CSVChangeHandler(csv_filename, file_change_callback, debounce_sec=5)
    observer = Observer()
    observer.schedule(handler, path=CSV_DIRECTORY, recursive=False)
    observer.start()
    logging.info(f"Started Watchdog on {csv_filename}")
    print(f"Started Watchdog on {csv_filename}")

    # If within window, optionally do one immediate check
    if start_time <= now <= end_time:
        logging.info("Doing an initial read and trade execution.")
        print("Doing an initial read and trade execution.")
        execute_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    # Keep script alive until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt -> shutting down.")
        print("KeyboardInterrupt -> shutting down.")
    finally:
        observer.stop()
        observer.join()
        logging.info("Watchdog observer stopped. Script ended.")
        print("Watchdog observer stopped. Script ended.")

if __name__ == "__main__":
    main()
