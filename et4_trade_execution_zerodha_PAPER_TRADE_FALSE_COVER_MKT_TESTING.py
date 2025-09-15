# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:27:23 2025

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
import json
import uuid
import csv
import traceback

# ================================
# Configuration Parameters
# ================================

CSV_DIRECTORY = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
EXECUTED_SIGNALS_FILE = "executed_signals_real_time.json"

# ================================
# Setup Logging
# ================================

cwd = CSV_DIRECTORY
os.chdir(cwd)  # Change working directory

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
# Utility Functions
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

# Global sets
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

# ================================
# current_market_price Function
# ================================

def current_market_price(ticker):
    """
    Fetches the current market price for the given ticker.

    Parameters:
        ticker (str): The trading symbol.

    Returns:
        float: Current market price.
    """
    try:
        # Last traded price from NSE
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
    Executes a real-time trade with a Cover Order at MARKET + trigger_price for SL,
    then places a separate LIMIT Target, implementing OCO logic.
    """
    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(india_tz)

        # 1) Fetch current market price
        current_price = current_market_price(ticker)
        if current_price == 0.0:
            print(f"Unable to fetch current market price for {ticker}. Trade aborted.")
            logging.error(f"Unable to fetch current market price for {ticker}. Trade aborted.")
            return

        # 2) Calculate Stop-Loss & Target
        if transaction_type.upper() == 'BUY':
            stop_loss_price = round(current_price * 0.98, 2)  # 2% below
            target_price = round(current_price * 1.02, 2)    # 2% above
        elif transaction_type.upper() == 'SELL':
            stop_loss_price = round(current_price * 1.02, 2) # 2% above
            target_price = round(current_price * 0.98, 2)    # 2% below
        else:
            logging.error(f"Invalid transaction type: {transaction_type}")
            print(f"Invalid transaction type: {transaction_type}. Use 'BUY' or 'SELL'.")
            return

        # 3) Place Cover Order (variety='co') with trigger_price for SL
        order_id = kite.place_order(
            variety='co',             # <-- KEY FIX: 'co' for Cover Orders
            exchange='NSE',
            tradingsymbol=ticker,
            transaction_type=transaction_type.upper(),
            quantity=quantity,
            product=kite.PRODUCT_MIS, # Intraday
            order_type='MARKET',      # Market entry
            trigger_price=stop_loss_price,
            validity='DAY',
            tag='RealTimeTrade'
        )

        logging.info(
            f"Placed Cover {transaction_type.upper()} MARKET order for {ticker} SL@{stop_loss_price}, Order ID: {order_id}"
        )
        print(
            f"Placed Cover {transaction_type.upper()} MARKET order for {ticker} SL@{stop_loss_price}, Order ID: {order_id}"
        )

        # 4) Monitor Cover Order fill
        filled = False
        filled_price = current_price
        while not filled:
            orders = kite.orders()
            for order in orders:
                if order['order_id'] == order_id and order['status'] == 'COMPLETE':
                    filled = True
                    filled_price = float(order.get('average_price', current_price))
                    break
            if not filled:
                time.sleep(2)

        logging.info(f"Cover order {order_id} for {ticker} filled at {filled_price}.")
        print(f"Cover order {order_id} for {ticker} filled at {filled_price}.")

        # 5) Place Target (LIMIT) order
        if transaction_type.upper() == 'BUY':
            target_txn = 'SELL'
        else:
            target_txn = 'BUY'

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
        logging.info(
            f"Placed Target {target_txn} LIMIT order for {ticker} @ {target_price}, ID: {target_order_id}"
        )
        print(
            f"Placed Target {target_txn} LIMIT order for {ticker} @ {target_price}, ID: {target_order_id}"
        )

        # 6) OCO Logic
        oco_active = True
        while oco_active:
            orders = kite.orders()
            target_filled = False
            sl_triggered = False
            for order in orders:
                if order['order_id'] == target_order_id and order['status'] == 'COMPLETE':
                    target_filled = True
                    break
                
                # Check if the child SL order is filled
                if (
                    order.get("parent_order_id") == order_id  # i.e. the main CO order_id
                    and order["status"] == "COMPLETE"
                ):
                    # If the SL child order is "COMPLETE", it means the stop-loss has actually triggered
                    sl_triggered = True
                    break
                
                #if order['order_id'] == order_id and order['status'] == 'COMPLETE':
                    # Means stop loss leg of the cover order triggered
                    #sl_triggered = True
                    #break
            if target_filled:
                # Cancel the SL (cover) if target is hit
                try:
                    kite.cancel_order(order_id=order_id, variety='co')
                    logging.info(f"Target order {target_order_id} filled. Canceled cover SL {order_id}.")
                    print(f"Target filled -> canceled SL. Target ID: {target_order_id}, SL ID: {order_id}")
                except Exception as e:
                    logging.error(f"Failed to cancel SL order {order_id}: {e}")
                    print(f"Failed to cancel SL order {order_id}: {e}")
                oco_active = False
            elif sl_triggered:
                # Cancel the Target if SL triggered
                try:
                    kite.cancel_order(order_id=target_order_id, variety='regular')
                    logging.info(f"SL triggered for cover order {order_id}. Canceled target ID: {target_order_id}.")
                    print(f"SL triggered -> canceled target. Cover ID: {order_id}, Target ID: {target_order_id}")
                except Exception as e:
                    logging.error(f"Failed to cancel target order {target_order_id}: {e}")
                    print(f"Failed to cancel target order {target_order_id}: {e}")
                oco_active = False
            else:
                time.sleep(5)

        # 7) Log final trade
        log_trade_result(
            ticker=ticker,
            transaction_type=transaction_type.upper(),
            price=filled_price,
            quantity=quantity,
            target_price=target_price,
            stop_loss_price=stop_loss_price,
            order_id=target_order_id
        )

        # 8) Mark as executed
        trade_id = generate_trade_id(ticker, datetime.now(pytz.timezone('Asia/Kolkata')), transaction_type.upper())
        executed_signals.add(trade_id)
        save_executed_signals()

    except Exception as e:
        logging.error(f"Error executing trade for {ticker}: {e}")
        print(f"Error executing trade for {ticker}: {e}")

# ================================
# main Function
# ================================

def main():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Define trading window
    start_naive = datetime.combine(now.date(), datetime_time(9, 31, 0))  # 9:31 AM
    end_naive = datetime.combine(now.date(), datetime_time(18, 30, 0))   # 6:30 PM
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

    while start_time <= datetime.now(india_tz) <= end_time:
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

                # Check if the ticker has already been traded
                if ticker in traded_tickers:
                    print(f"Ticker {ticker} has already been traded today.")
                    logging.info(f"Attempted to trade {ticker} again. Skipping.")
                    continue

                # Execute the trade
                execute_trade(ticker, txn, qty)
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

    logging.info("Trading window closed. Script ended.")
    print("Trading window closed. Script ended.")

if __name__ == "__main__":
    main()
