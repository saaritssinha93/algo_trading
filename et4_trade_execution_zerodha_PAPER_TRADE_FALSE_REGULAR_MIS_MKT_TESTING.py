# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:11:27 2025

@author: Saarit
"""
import os
import pandas as pd
import logging
import threading
import time
import json
import uuid
import csv
import traceback
from datetime import datetime, timedelta, time as datetime_time

import pytz
from kiteconnect import KiteConnect

# ================================
# Configuration
# ================================
CSV_DIRECTORY = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
EXECUTED_SIGNALS_FILE = "executed_signals_real_time.json"

# ================================
# Setup Logging
# ================================
os.chdir(CSV_DIRECTORY)  # Change working directory

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
            raise ValueError("API key and secret must be in 'api_key.txt', separated by space.")

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
# Utilities
# ================================
def load_executed_signals():
    if os.path.exists(EXECUTED_SIGNALS_FILE):
        try:
            with open(EXECUTED_SIGNALS_FILE, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Executed signals file is corrupted. Starting with an empty set.")
            return set()
    return set()

def save_executed_signals():
    with open(EXECUTED_SIGNALS_FILE, 'w') as f:
        json.dump(list(executed_signals), f)

executed_signals = load_executed_signals()
traded_tickers = set()  # Tickers traded today to avoid duplicates

def generate_trade_id(ticker, date, transaction_type):
    unique_id = uuid.uuid4()
    return f"{ticker}-{date.isoformat()}-{transaction_type}-{unique_id}"

def log_trade_result(ticker, transaction_type, price, quantity, target_price, stop_loss_price, order_id):
    india_tz = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
    trade_log_filename = f"trade_log_{today_str}.csv"

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

    file_exists = os.path.exists(trade_log_filename)
    df = pd.DataFrame([row_data])
    df.to_csv(
        trade_log_filename,
        mode='a',
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        doublequote=True
    )

def current_market_price(ticker):
    """
    Fetches the current market price for the given ticker.
    """
    try:
        ltp_data = kite.ltp(f"NSE:{ticker}")
        return ltp_data[f"NSE:{ticker}"]['last_price']
    except Exception as e:
        logging.error(f"Failed to fetch current market price for {ticker}: {e}")
        print(f"Failed to fetch current market price for {ticker}: {e}")
        return 0.0

# ================================
# execute_trade Function
# ================================
def execute_trade(ticker, transaction_type, quantity):
    """
    Executes a real-time trade in MIS with:
      1. MARKET entry
      2. Separate LIMIT target order
      3. SL-M (stop-loss market) order
      4. OCO logic: if target is hit => cancel SL; if SL is hit => cancel target.
    """
    def _log(msg):
        logging.info(msg)
        print(msg)

    try:
        india_tz = pytz.timezone('Asia/Kolkata')

        # 1) Fetch current market price
        current_price = current_market_price(ticker)
        if current_price == 0.0:
            _log(f"Unable to fetch current market price for {ticker}. Trade aborted.")
            return

        # 2) Calculate Stop-Loss & Target (example: 2% away)
        if transaction_type == 'BUY':
            stop_loss_price = round(current_price * 0.98, 2)  # 2% below
            target_price = round(current_price * 1.02, 2)     # 2% above
            sl_txn = 'SELL'
            target_txn = 'SELL'
        else:  # SELL
            stop_loss_price = round(current_price * 1.02, 2)  # 2% above
            target_price = round(current_price * 0.98, 2)     # 2% below
            sl_txn = 'BUY'
            target_txn = 'BUY'

        # 3) Place the initial MARKET order
        entry_order_id = kite.place_order(
            variety='regular',
            exchange='NSE',
            tradingsymbol=ticker,
            transaction_type=transaction_type,
            quantity=quantity,
            product=kite.PRODUCT_MIS,
            order_type='MARKET',
            validity='DAY',
            tag='EntryOrder'
        )
        _log(f"Placed {transaction_type} MARKET order for {ticker}, qty={quantity}, ID={entry_order_id}")

        # 4) Wait for market order fill
        filled = False
        filled_price = 0.0
        while not filled:
            orders = kite.orders()
            for order in orders:
                if order['order_id'] == entry_order_id and order['status'] == 'COMPLETE':
                    filled = True
                    filled_price = float(order.get('average_price', current_price))
                    break
            if not filled:
                time.sleep(2)

        _log(f"Market entry for {ticker} filled at {filled_price}, order_id={entry_order_id}")

        # 5) Place Target (LIMIT) order
        target_order_id = kite.place_order(
            variety='regular',
            exchange='NSE',
            tradingsymbol=ticker,
            transaction_type=target_txn,
            quantity=quantity,
            product=kite.PRODUCT_MIS,
            order_type='LIMIT',
            price=target_price,
            validity='DAY',
            tag='TargetOrder'
        )
        _log(f"Placed Target {target_txn} LIMIT order for {ticker} @ {target_price}, ID={target_order_id}")

        # 6) Place Stop-Loss (SL-M) order
        # For an SL-M order: order_type='SL-M'
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
        _log(f"Placed Stop-Loss {sl_txn} SL-M order for {ticker} trigger={stop_loss_price}, ID={stop_loss_order_id}")

        # 7) Monitor until either Target or SL is hit
        trade_active = True
        target_filled = False
        sl_triggered = False

        while trade_active:
            orders = kite.orders()
            for order in orders:
                # If target is COMPLETED
                if order['order_id'] == target_order_id and order['status'] == 'COMPLETE':
                    target_filled = True
                    break
                # If SL is COMPLETED
                if order['order_id'] == stop_loss_order_id and order['status'] == 'COMPLETE':
                    sl_triggered = True
                    break

            if target_filled:
                try:
                    kite.cancel_order(variety='regular', order_id=stop_loss_order_id)
                    _log(f"Target filled => canceled SL (order_id={stop_loss_order_id}). Trade exited.")
                except Exception as ce:
                    _log(f"Failed to cancel SL order {stop_loss_order_id}: {ce}")
                trade_active = False

            elif sl_triggered:
                try:
                    kite.cancel_order(variety='regular', order_id=target_order_id)
                    _log(f"SL triggered => canceled Target (order_id={target_order_id}). Trade exited.")
                except Exception as ce:
                    _log(f"Failed to cancel Target order {target_order_id}: {ce}")
                trade_active = False

            else:
                time.sleep(5)

        # 8) Log final trade result
        log_trade_result(
            ticker=ticker,
            transaction_type=transaction_type,
            price=filled_price,
            quantity=quantity,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            order_id=entry_order_id
        )

        # 9) Mark as executed
        trade_id = generate_trade_id(ticker, datetime.now(pytz.timezone('Asia/Kolkata')), transaction_type)
        executed_signals.add(trade_id)
        save_executed_signals()

        _log(f"Trade completed for {ticker}, EntryID={entry_order_id}.\n")

    except Exception as e:
        _log(f"Error executing trade for {ticker}: {e}")
        traceback.print_exc()

# ================================
# Multi-Threading Start
# ================================
def start_trade_thread(ticker, transaction_type, quantity):
    """
    Start a new thread for the trade so the main menu isn't blocked.
    """
    trade_thread = threading.Thread(
        target=execute_trade,
        args=(ticker, transaction_type, quantity),
        daemon=True
    )
    trade_thread.start()

# ================================
# main Function
# ================================
def main():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Define your trading window if desired
    start_naive = datetime.combine(now.date(), time(9, 31, 0))
    end_naive = datetime.combine(now.date(), time(18, 30, 0))
    start_time = india_tz.localize(start_naive)
    end_time = india_tz.localize(end_naive)

    if now > end_time:
        logging.warning("Current time is past the trading window. Exiting the script.")
        print("Current time is past the trading window. Exiting the script.")
        return
    elif now < start_time:
        logging.info("Current time is before the trading window. Waiting to start trading.")
        print("Current time is before the trading window. Waiting to start trading.")
    else:
        logging.info("Current time is within the trading window. You can start trading now.")
        print("Current time is within the trading window. You can start trading now.")

    # Main menu loop
    while True:
        current_time = datetime.now(india_tz)
        if current_time > end_time:
            logging.info("Trading window closed. Exiting script.")
            print("Trading window closed. Script ended.")
            break

        try:
            print("\n--- Real-Time Trading Menu ---")
            print("1. Place a Trade")
            print("2. Exit")
            choice = input("Enter your choice (1 or 2): ").strip()

            if choice == '1':
                ticker = input("Enter the ticker symbol (e.g., RELIANCE): ").strip().upper()
                txn = input("Enter transaction type (BUY/SELL): ").strip().upper()
                if txn not in ['BUY', 'SELL']:
                    print("Invalid type. Must be 'BUY' or 'SELL'.")
                    continue

                try:
                    qty = int(input("Enter quantity: ").strip())
                    if qty <= 0:
                        print("Quantity must be > 0.")
                        continue
                except ValueError:
                    print("Invalid quantity. Please enter a positive integer.")
                    continue

                # Check if ticker already traded
                if ticker in traded_tickers:
                    print(f"Ticker {ticker} has already been traded today. Skipping.")
                    logging.info(f"Attempted to trade {ticker} again. Skipping.")
                    continue

                # Launch trade in background thread
                start_trade_thread(ticker, txn, qty)
                traded_tickers.add(ticker)

            elif choice == '2':
                print("Exiting the script.")
                logging.info("User exit request.")
                break
            else:
                print("Invalid choice. Enter 1 or 2.")

        except KeyboardInterrupt:
            print("\nInterrupted by user, exiting.")
            logging.info("Script interrupted by user.")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            print(f"Unexpected error: {e}")

    logging.info("Script ended cleanly.")
    print("Script ended.")

if __name__ == "__main__":
    main()
