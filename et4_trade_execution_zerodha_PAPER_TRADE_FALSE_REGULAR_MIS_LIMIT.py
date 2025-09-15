# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:59:52 2025

@author: Saarit

Script that:
1. Reads 'papertrade_{yyyy-mm-dd}.csv' for signals
2. Places a LIMIT entry (MIS) using CSV 'Price' for real trades, or sim for paper trades
3. Once entry is filled, places separate LIMIT target & SL (stop-loss limit) order
4. Polls until one is filled, cancels the other
5. Uses watchdog to automatically re-check CSV on file changes
6. PAPER_TRADE=True => simulation, PAPER_TRADE=False => real trades
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
PAPER_TRADE = True  # Toggle between paper trades and real trades
CSV_DIRECTORY = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
CSV_FILENAME_PATTERN = "papertrade_{}.csv"
EXECUTED_SIGNALS_FILE = "executed_signals_cover_mkt.json"

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
    """
    (Optional) You might not rely on this if you're using CSV price for the limit entry,
    but we can still keep it if needed for logs or checking.
    """
    try:
        ltp_data = kite.ltp(f"NSE:{ticker}")
        return ltp_data[f"NSE:{ticker}"]['last_price']
    except Exception as e:
        logging.error(f"Failed to fetch LTP for {ticker}: {e}")
        print(f"Failed to fetch LTP for {ticker}: {e}")
        return 0.0

# ================================
# Paper vs Real Trade Logging
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
# CSV Reading & Normalization
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
# Real Trade with LIMIT Entry, LIMIT Target, SL (Stop-Loss Limit)
# ================================
def place_mis_limit_trade(kite, ticker, transaction_type, quantity, limit_price):
    """
    1) Place a LIMIT entry order at 'limit_price' (from CSV).
    2) Wait for fill => get fill_price (or assume limit_price).
    3) Place a separate LIMIT target order ~ +2% or -2%.
    4) Place an SL (stop-loss limit) order (not SL-M), so we set both 'price' and 'trigger_price'.
       - For SELL SL: price <= trigger_price
       - For BUY SL:  price >= trigger_price
    5) Poll until target or SL is filled => cancel the other
    """

    def _log(msg):
        logging.info(msg)
        print(msg)

    # 1) Entry Limit Order
    entry_order_id = kite.place_order(
        variety='regular',
        exchange='NSE',
        tradingsymbol=ticker,
        transaction_type=transaction_type.upper(),
        quantity=quantity,
        product=kite.PRODUCT_MIS,
        order_type='LIMIT',    # <--- LIMIT entry
        price=limit_price,     # from CSV
        validity='DAY',
        tag='EntryOrder'
    )
    _log(f"[REAL] Placed {transaction_type} LIMIT for {ticker} @ {limit_price}, ID={entry_order_id}")

    # 2) Wait for fill
    filled = False
    fill_price = limit_price
    while not filled:
        orders = kite.orders()
        for o in orders:
            if o['order_id'] == entry_order_id and o['status'] == 'COMPLETE':
                filled = True
                fill_price = float(o.get('average_price', limit_price))
                break
        if not filled:
            time.sleep(2)

    _log(f"[REAL] Entry {entry_order_id} filled at {fill_price}")

    # 3) Compute target & SL price (2% away)
    if transaction_type.upper() == 'BUY':
        target_price_val = round(fill_price * 1.02, 2)
        sl_price_val = round(fill_price * 0.98, 2)
        # For a SELL SL (to exit a BUY), we want 'price <= trigger_price'.
        # We'll keep them the same to do a stop-loss limit
        # e.g. price=sl_price_val, trigger_price=sl_price_val
        sl_txn = 'SELL'
        tgt_txn = 'SELL'
        sl_trigger = sl_price_val  # same
        sl_limit = sl_price_val    # same
        # If the difference is too large, Zerodha might throw an error.

    else:
        # SELL entry
        target_price_val = round(fill_price * 0.98, 2)
        sl_price_val = round(fill_price * 1.02, 2)
        # For a BUY SL (to exit a SELL), we want 'price >= trigger_price'
        sl_txn = 'BUY'
        tgt_txn = 'BUY'
        sl_trigger = sl_price_val
        sl_limit = sl_price_val

    # 4) Place Target (LIMIT)
    target_order_id = kite.place_order(
        variety='regular',
        exchange='NSE',
        tradingsymbol=ticker,
        transaction_type=tgt_txn,
        quantity=quantity,
        product=kite.PRODUCT_MIS,
        order_type='LIMIT',
        price=target_price_val,
        validity='DAY',
        tag='TargetOrder'
    )
    _log(f"[REAL] Placed {tgt_txn} LIMIT for {ticker} @ {target_price_val}, ID={target_order_id}")

    # 5) Place Stop-Loss (SL)
    #    order_type='SL' => stop-loss limit
    #    must set both price= and trigger_price=
    stop_loss_order_id = kite.place_order(
        variety='regular',
        exchange='NSE',
        tradingsymbol=ticker,
        transaction_type=sl_txn,
        quantity=quantity,
        product=kite.PRODUCT_MIS,
        order_type='SL',
        price=sl_limit,            # limit price for the stop-loss
        trigger_price=sl_trigger,  # trigger price for the stop-loss
        validity='DAY',
        tag='StopLossOrder'
    )
    _log(f"[REAL] Placed {sl_txn} SL order for {ticker}, price={sl_limit}, trigger={sl_trigger}, ID={stop_loss_order_id}")

    # 6) Poll for exit
    trade_active = True
    while trade_active:
        orders = kite.orders()
        target_filled = any(o['order_id'] == target_order_id and o['status'] == 'COMPLETE' for o in orders)
        sl_filled = any(o['order_id'] == stop_loss_order_id and o['status'] == 'COMPLETE' for o in orders)

        if target_filled:
            # Cancel SL
            try:
                kite.cancel_order(variety='regular', order_id=stop_loss_order_id)
                _log(f"[REAL] Target filled => canceled SL order {stop_loss_order_id}. Exiting trade.")
            except Exception as ce:
                _log(f"[REAL] Error canceling SL {stop_loss_order_id}: {ce}")
            trade_active = False

        elif sl_filled:
            # Cancel Target
            try:
                kite.cancel_order(variety='regular', order_id=target_order_id)
                _log(f"[REAL] Stop-loss triggered => canceled Target {target_order_id}. Exiting trade.")
            except Exception as ce:
                _log(f"[REAL] Error canceling Target {target_order_id}: {ce}")
            trade_active = False

        else:
            time.sleep(5)

    # Return final info for logging
    return {
        "entry_order_id": entry_order_id,
        "entry_fill_price": fill_price,
        "stop_loss_price": sl_price_val,
        "target_price": target_price_val
    }

# ================================
# Master Execute Trades
# ================================
def execute_trades(last_trading_day, kite, api_semaphore, paper_trade=True):
    """
    Reads `papertrade_{last_trading_day}.csv`.
    - If PAPER_TRADE=True => simulate with a 'paper fill' at CSV price
    - If PAPER_TRADE=False => place real MIS limit entry (with CSV price),
      then limit target & SL (stop-loss limit).
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

    # Filter out signals already executed
    new_df = df[~df['Signal_ID'].isin(executed_signals)]
    if new_df.empty:
        logging.info("No new trades to execute.")
        print("No new trades to execute.")
        return

    india_tz = pytz.timezone('Asia/Kolkata')

    for idx, row in new_df.iterrows():
        try:
            ticker = row['Ticker'].strip().upper()
            trend_type = row['Trend Type'].lower().strip()
            signal_id = row['Signal_ID']
            limit_price = float(row['Price'])  # CSV limit for entry
            qty_str = row.get('Quantity', 1)
            try:
                quantity = int(qty_str)
            except:
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
                # PAPER TRADES
                if ticker in paper_traded_tickers:
                    logging.info(f"{ticker} already paper-traded. Skipping.")
                    print(f"{ticker} already paper-traded. Skipping.")
                    continue

                # For simulation, we 'assume' the limit entry is filled at CSV price
                # Then we do a 2% 'target' & 'SL' for logs
                if transaction_type == 'BUY':
                    sim_target_price = round(limit_price * 1.02, 2)
                    sim_sl_price = round(limit_price * 0.98, 2)
                else:
                    sim_target_price = round(limit_price * 0.98, 2)
                    sim_sl_price = round(limit_price * 1.02, 2)

                sim_id = f"SIM-{signal_id}-{datetime.now(india_tz).strftime('%H%M%S')}"
                logging.info(
                    f"[SIM] {transaction_type} {ticker}, entry@{limit_price}, "
                    f"target@{sim_target_price}, sl@{sim_sl_price}, ID={sim_id}"
                )
                print(
                    f"[SIM] {transaction_type} {ticker}, entry@{limit_price}, "
                    f"target@{sim_target_price}, sl@{sim_sl_price}, ID={sim_id}"
                )

                # Log simulation
                log_simulation_result(
                    ticker=ticker,
                    transaction_type=transaction_type,
                    price=limit_price,
                    quantity=quantity,
                    target_price=sim_target_price,
                    stop_loss_price=sim_sl_price,
                    order_id=sim_id
                )

                executed_signals.add(signal_id)
                paper_traded_tickers.add(ticker)

            else:
                # REAL TRADES
                if ticker in real_traded_tickers:
                    logging.info(f"{ticker} already real-traded. Skipping.")
                    print(f"{ticker} already real-traded. Skipping.")
                    continue

                with api_semaphore:
                    try:
                        # Place real MIS limit entry & separate limit target + stop-loss limit
                        trade_info = place_mis_limit_trade(
                            kite, ticker, transaction_type, quantity, limit_price
                        )

                        # Log final result
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
    end_naive = datetime.combine(now.date(), datetime_time(18, 30, 0))
    start_time = india_tz.localize(start_naive)
    end_time = india_tz.localize(end_naive)

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

    # If within window, do an initial read once
    if start_time <= now <= end_time:
        logging.info("Doing an initial read and trade execution.")
        print("Doing an initial read and trade execution.")
        execute_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    # Keep script alive
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
