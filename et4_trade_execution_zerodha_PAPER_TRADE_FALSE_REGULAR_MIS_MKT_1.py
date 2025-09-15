# -*- coding: utf-8 -*-
"""
Cleaned-up Code 2:
- Reads from 'papertrade_{YYYY-MM-DD}.csv' (matching your signals from Code 1).
- If PAPER_TRADE=True, logs a "simulation" order.
- If PAPER_TRADE=False, places a real MIS trade with a 2% target and 2% SL-M order,
  both snapped to the nearest 0.5.
- Uses Watchdog to trigger whenever the CSV changes.
- Uses executed_signals_mis_mkt.json to store which Signal_IDs were already traded
  so it doesn't trade them again.
- Forces Quantity=1 for all trades.
- Observes a daily “paper_traded_tickers” and “real_traded_tickers” set, so you only
  trade each ticker once (remove if you want multiple signals per ticker in a single day).
"""

import os
import time
import csv
import json
import uuid
import threading
import logging
import traceback
import pytz

from datetime import datetime, timedelta, time as datetime_time

import pandas as pd
from kiteconnect import KiteConnect

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================================
# 1) Configuration
# ================================
PAPER_TRADE = False  # Flip True/False to simulate or place real orders
CSV_DIRECTORY = r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo"
CSV_FILENAME_PATTERN = "papertrade_{}.csv"

EXECUTED_SIGNALS_FILE = "executed_signals_mis_mkt.json"
EXPECTED_COLUMNS = ["Ticker", "date", "Trend Type", "Signal_ID", "Price", "Quantity"]

# If you want to block multiple signals for same ticker on the same day
paper_traded_tickers = set()
real_traded_tickers = set()

# ================================
# 2) Logging Setup
# ================================
os.makedirs(CSV_DIRECTORY, exist_ok=True)
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
print("Logging initialized. Script execution started.")


# ================================
# 3) Kite Connect Session
# ================================
def setup_kite_session():
    """
    Reads 'access_token.txt' for the access token,
    'api_key.txt' for [api_key, api_secret],
    sets up KiteConnect session.
    """
    try:
        # Read Access Token
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()

        # Read API key + secret
        with open("api_key.txt", "r") as f:
            key_data = f.read().strip().split()
        if len(key_data) < 2:
            raise ValueError("api_key.txt must contain 'API_KEY' and 'API_SECRET' separated by space.")
        api_key, api_secret = key_data[0], key_data[1]

        # Initialize Kite
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
        print(f"Error setting up Kite session: {e}")
        raise


try:
    kite = setup_kite_session()
except Exception as e:
    kite = None
    logging.warning("Kite session not established (check credentials). Continuing in PAPER_TRADE mode only if needed.")


# ================================
# 4) Executed Signals (JSON)
# ================================
def load_executed_signals():
    """
    Load the set of Signal_IDs that have already been processed.
    """
    if os.path.exists(EXECUTED_SIGNALS_FILE):
        try:
            with open(EXECUTED_SIGNALS_FILE, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Executed signals file is corrupted. Starting with empty set.")
            return set()
    else:
        return set()


def save_executed_signals(executed_signals_set):
    """
    Save the updated set of Signal_IDs to JSON.
    """
    with open(EXECUTED_SIGNALS_FILE, 'w') as f:
        json.dump(list(executed_signals_set), f)


executed_signals = load_executed_signals()


# ================================
# 5) Utility Functions
# ================================
def generate_signal_id(ticker, dt_obj, transaction_type):
    """
    Fallback generator if the CSV is missing a 'Signal_ID'.
    """
    unique_id = uuid.uuid4()
    return f"{ticker}-{dt_obj.isoformat()}-{transaction_type}-{unique_id}"


def current_market_price(kite_, ticker):
    """
    Safely fetch Last Traded Price from Kite.
    """
    try:
        if kite_ is None:
            return 0.0
        data = kite_.ltp(f"NSE:{ticker}")
        return float(data[f"NSE:{ticker}"]['last_price'])
    except Exception as e:
        logging.error(f"Failed to fetch LTP for {ticker}: {e}")
        print(f"Failed to fetch LTP for {ticker}: {e}")
        return 0.0


def round_to_nearest_05(value: float) -> float:
    """Round the given float to the nearest 0.5."""
    return round(value * 2) / 2.0


# ================================
# 6) Logging Paper/Real Trades
# ================================
def log_simulation_result(ticker, transaction_type, price, quantity,
                          target_price, stop_loss_price, order_id):
    """
    Appends a record to simulation_papertrade_{YYYY-MM-DD}.csv
    so you can track your hypothetical orders.
    """
    india_tz = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(india_tz).strftime('%Y-%m-%d')
    sim_file = f"simulation_papertrade_{today_str}.csv"

    row_data = {
        "Timestamp": datetime.now(india_tz).isoformat(),
        "Ticker": ticker,
        "Transaction Type": transaction_type,
        "Quantity": quantity,
        "Executed Price": price,
        "Target Price": target_price if target_price else "",
        "Stop Loss Price": stop_loss_price if stop_loss_price else "",
        "Order ID": order_id
    }
    file_exists = os.path.exists(sim_file)

    pd.DataFrame([row_data]).to_csv(
        sim_file,
        mode='a',
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_ALL
    )


def log_trade_result(ticker, transaction_type, price, quantity,
                     target_price, stop_loss_price, order_id):
    """
    Appends a record to trade_log_{YYYY-MM-DD}.csv
    to track real MIS trades.
    """
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

    pd.DataFrame([row_data]).to_csv(
        trade_file,
        mode='a',
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_ALL
    )


# ================================
# 7) Reading & Normalizing CSV
# ================================
def read_and_normalize_csv(file_path, expected_columns, timezone='Asia/Kolkata'):
    """
    Reads the CSV, ensures columns exist, normalizes 'date' column to Asia/Kolkata,
    and returns a clean DataFrame.
    """
    logging.info(f"Reading CSV: {file_path}")
    try:
        df = pd.read_csv(
            file_path,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            on_bad_lines='warn',
            engine='python'
        )
    except Exception as e:
        logging.error(f"Error reading CSV {file_path}: {e}")
        raise

    # Ensure expected columns exist, fill missing
    missing_cols = set(expected_columns) - set(df.columns)
    for mc in missing_cols:
        df[mc] = ""

    # Remove extra columns we don't need
    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        logging.warning(f"Dropping extra columns: {extra_cols}")
        df.drop(columns=list(extra_cols), inplace=True)

    # Convert date -> datetime w/ tz
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df.dropna(subset=['date'], inplace=True)
        df['date'] = df['date'].dt.tz_convert(timezone)
    else:
        df['date'] = pd.NaT

    # If 'Signal_ID' is empty, create a fallback
    if 'Signal_ID' in df.columns:
        missing_sid_mask = df['Signal_ID'].isna() | (df['Signal_ID'] == "")
        if missing_sid_mask.any():
            df.loc[missing_sid_mask, 'Signal_ID'] = df[missing_sid_mask].apply(
                lambda row: generate_signal_id(
                    row['Ticker'],
                    row['date'] if pd.notna(row['date']) else datetime.now(pytz.timezone(timezone)),
                    row['Trend Type']
                ),
                axis=1
            )

    # Force columns to expected order
    df = df[expected_columns].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ================================
# 8) Real MIS: place entry / target / SL
# ================================
def place_mis_trade(kite_, ticker, transaction_type, quantity):
    """
    For real trades:
      1. Get LTP
      2. Place Market entry order
      3. Wait until filled (polling)
      4. Place limit target (2% away)
      5. Place stop-loss SL-M (2% away)
      6. Wait for fill of either one, cancel the other.

    Returns a dict with final fill info, to be logged.
    """
    def _log(msg):
        logging.info(msg)
        print(msg)

    # 1) Current LTP
    ltp = current_market_price(kite_, ticker)
    if ltp == 0.0:
        msg = f"Could not fetch LTP for {ticker}. Aborting real trade."
        _log(msg)
        raise ValueError(msg)

    # 2% offsets
    if transaction_type.upper() == "BUY":
        raw_sl_price = ltp * 0.98
        raw_tgt_price = ltp * 1.02
        sl_transaction = "SELL"
        tgt_transaction = "SELL"
    else:
        raw_sl_price = ltp * 1.02
        raw_tgt_price = ltp * 0.98
        sl_transaction = "BUY"
        tgt_transaction = "BUY"

    # Snap to nearest 0.5
    sl_price = round_to_nearest_05(raw_sl_price)
    tgt_price = round_to_nearest_05(raw_tgt_price)

    # 2) Place Market Entry
    entry_order_id = kite_.place_order(
        variety=kite_.VARIETY_REGULAR,
        exchange=kite_.EXCHANGE_NSE,
        tradingsymbol=ticker,
        transaction_type=transaction_type.upper(),
        quantity=quantity,
        product=kite_.PRODUCT_MIS,
        order_type=kite_.ORDER_TYPE_MARKET,
        validity=kite_.VALIDITY_DAY,
        tag="EntryOrder"
    )
    _log(f"[REAL] Placed {transaction_type.upper()} MARKET for {ticker}, qty={quantity}, ID={entry_order_id}")

    # 3) Wait for fill
    filled_price = ltp
    filled = False
    while not filled:
        orders = kite_.orders()
        for o in orders:
            if o['order_id'] == entry_order_id and o['status'] == 'COMPLETE':
                filled = True
                filled_price = float(o.get('average_price', ltp))
                break
        if not filled:
            time.sleep(2)
    _log(f"[REAL] Entry {entry_order_id} filled at {filled_price}")

    # 4) Target (LIMIT)
    target_order_id = kite_.place_order(
        variety=kite_.VARIETY_REGULAR,
        exchange=kite_.EXCHANGE_NSE,
        tradingsymbol=ticker,
        transaction_type=tgt_transaction,
        quantity=quantity,
        product=kite_.PRODUCT_MIS,
        order_type=kite_.ORDER_TYPE_LIMIT,
        price=tgt_price,
        validity=kite_.VALIDITY_DAY,
        tag="TargetOrder"
    )
    _log(f"[REAL] Placed {tgt_transaction} LIMIT for {ticker} @ {tgt_price}, ID={target_order_id}")

    # 5) Stop-loss (SL-M)
    stop_loss_order_id = kite_.place_order(
        variety=kite_.VARIETY_REGULAR,
        exchange=kite_.EXCHANGE_NSE,
        tradingsymbol=ticker,
        transaction_type=sl_transaction,
        quantity=quantity,
        product=kite_.PRODUCT_MIS,
        order_type=kite_.ORDER_TYPE_SLM,
        trigger_price=sl_price,
        validity=kite_.VALIDITY_DAY,
        tag="StopLossOrder"
    )
    _log(f"[REAL] Placed {sl_transaction} SL-M for {ticker}, trigger={sl_price}, ID={stop_loss_order_id}")

    # 6) Wait for either target or SL to fill
    trade_active = True
    while trade_active:
        orders = kite_.orders()

        target_filled = any(
            o['order_id'] == target_order_id and o['status'] == 'COMPLETE'
            for o in orders
        )
        sl_filled = any(
            o['order_id'] == stop_loss_order_id and o['status'] == 'COMPLETE'
            for o in orders
        )

        if target_filled:
            # Cancel the SL order
            try:
                kite_.cancel_order(
                    variety=kite_.VARIETY_REGULAR,
                    order_id=stop_loss_order_id
                )
                _log(f"[REAL] Target hit => canceled SL {stop_loss_order_id}")
            except Exception as ce:
                _log(f"[REAL] Error canceling SL {stop_loss_order_id}: {ce}")
            trade_active = False

        elif sl_filled:
            # Cancel the target order
            try:
                kite_.cancel_order(
                    variety=kite_.VARIETY_REGULAR,
                    order_id=target_order_id
                )
                _log(f"[REAL] SL triggered => canceled Target {target_order_id}")
            except Exception as ce:
                _log(f"[REAL] Error canceling Target {target_order_id}: {ce}")
            trade_active = False

        else:
            time.sleep(5)

    return {
        "entry_order_id": entry_order_id,
        "entry_fill_price": filled_price,
        "stop_loss_price": sl_price,
        "target_price": tgt_price
    }


# ================================
# 9) Master “Execute Trades” Flow
# ================================
def execute_trades(last_trading_day, kite_, api_semaphore, paper_trade=True):
    """
    1) Reads the CSV => papertrade_{YYYY-MM-DD}.csv
    2) For each row, skip if:
       - The 'Signal_ID' is already in executed_signals
       - The ticker is already traded (if you want only one trade/ticker/day)
    3) Force quantity=1.
    4) If paper_trade => log a simulation.
       If real => place MIS entry, target, & SL orders.
    5) Store the 'Signal_ID' in executed_signals to avoid duplication.
    """

    filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    file_path = os.path.join(CSV_DIRECTORY, filename)

    if not os.path.exists(file_path):
        msg = f"File {file_path} not found. No trades."
        logging.error(msg)
        print(msg)
        return

    # Read CSV
    try:
        df = read_and_normalize_csv(file_path, EXPECTED_COLUMNS)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return

    # Filter out signals we already executed
    fresh_df = df[~df['Signal_ID'].isin(executed_signals)].copy()
    if fresh_df.empty:
        logging.info("No new trades to execute.")
        print("No new trades to execute.")
        return

    india_tz = pytz.timezone('Asia/Kolkata')

    # Process row by row
    for idx, row in fresh_df.iterrows():
        try:
            ticker = str(row['Ticker']).upper()
            trend = str(row['Trend Type']).lower().strip()
            signal_id = row['Signal_ID']
            csv_price = float(row.get('Price', 0) or 0)

            # Force quantity = 1
            quantity = 1

            # Determine transaction type
            if trend == "bullish":
                txn_type = "BUY"
            elif trend == "bearish":
                txn_type = "SELL"
            else:
                logging.info(f"Unknown Trend Type '{trend}' for {ticker}. Skipping.")
                continue

            # Skip if we have already traded this ticker (one trade per day)
            if paper_trade and (ticker in paper_traded_tickers):
                logging.info(f"{ticker} already paper-traded today. Skipping.")
                continue
            if (not paper_trade) and (ticker in real_traded_tickers):
                logging.info(f"{ticker} already real-traded today. Skipping.")
                continue

            # If PAPER_TRADE, do a "simulation"
            if paper_trade:
                # Simulate ~2% SL from CSV Price, snapped to 0.5
                if txn_type == 'BUY':
                    raw_sl_price = csv_price * 0.98
                else:
                    raw_sl_price = csv_price * 1.02
                sl_price = round_to_nearest_05(raw_sl_price)

                sim_id = f"SIM-{signal_id}-{datetime.now(india_tz).strftime('%H%M%S')}"
                print(f"[SIM] {txn_type} {ticker} @ {csv_price}, SL={sl_price}, ID={sim_id}")
                logging.info(f"[SIM] {txn_type} {ticker} @ {csv_price}, SL={sl_price}, ID={sim_id}")

                # Log
                log_simulation_result(
                    ticker=ticker,
                    transaction_type=txn_type,
                    price=csv_price,
                    quantity=quantity,
                    target_price=None,
                    stop_loss_price=sl_price,
                    order_id=sim_id
                )

                executed_signals.add(signal_id)
                paper_traded_tickers.add(ticker)

            else:
                # REAL TRADE
                if kite_ is None:
                    print("Kite session not available. Skipping real trade.")
                    continue

                with api_semaphore:  # concurrency limit
                    try:
                        trade_info = place_mis_trade(kite_, ticker, txn_type, quantity)
                        # Log real trade
                        log_trade_result(
                            ticker=ticker,
                            transaction_type=txn_type,
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

        except Exception as row_exc:
            logging.error(f"Error processing row {idx}: {row_exc}")
            print(f"Error processing row {idx}: {row_exc}")

    # Finally, save updated signals
    save_executed_signals(executed_signals)
    logging.info("Finished processing trades.")
    print("Finished processing trades.")


# ================================
# 10) Watchdog Handler
# ================================
class CSVChangeHandler(FileSystemEventHandler):
    """
    On file 'papertrade_YYYY-MM-DD.csv' changed,
    wait a few seconds (debounce) then call 'callback()'.
    """
    def __init__(self, target_file, callback, debounce_sec=5):
        super().__init__()
        self.target_file = target_file
        self.callback = callback
        self.debounce_sec = debounce_sec
        self.timer = None

    def on_modified(self, event):
        if event.is_directory:
            return
        if os.path.basename(event.src_path) == self.target_file:
            logging.info(f"Detected modification in {event.src_path}")
            print(f"Detected modification in {event.src_path}")
            self._debounce()

    def _debounce(self):
        if self.timer and self.timer.is_alive():
            self.timer.cancel()
        self.timer = threading.Timer(self.debounce_sec, self.callback)
        self.timer.start()


# ================================
# 11) Main
# ================================
def main():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Trading window example: 9:31 to 15:30 or so
    start_naive = datetime.combine(now.date(), datetime_time(9, 31))
    end_naive = datetime.combine(now.date(), datetime_time(15, 30))
    start_time = india_tz.localize(start_naive)
    end_time = india_tz.localize(end_naive)

    if now < start_time:
        print("Currently before trading window. Will watch for CSV changes.")
        logging.info("Currently before trading window. Will watch for CSV changes.")
    elif now > end_time:
        print("Currently after trading window. Watchdog still runs, but no real trades recommended.")
        logging.warning("Currently after trading window. We might place trades if code is triggered.")
    else:
        print("Within trading window. Ready to execute trades if CSV changes.")
        logging.info("Within trading window. Ready to execute trades if CSV changes.")

    # Example: Market holidays
    market_holidays = [
        datetime(2024, 1, 26).date(),
        datetime(2024, 3, 8).date(),
        datetime(2024, 3, 25).date(),
        datetime(2024, 3, 29).date(),
        datetime(2024, 4, 11).date(),
        datetime(2024, 8, 15).date(),
        datetime(2024, 10, 2).date(),
        datetime(2024, 11, 1).date(),
        datetime(2024, 11, 15).date(),
        datetime(2024, 12, 25).date(),
    ]

    def get_last_trading_day():
        d = now.date()
        # If weekend or holiday, step backward
        while d.weekday() >= 5 or d in market_holidays:
            d -= timedelta(days=1)
        return d.strftime('%Y-%m-%d')

    last_trading_day = get_last_trading_day()
    print(f"Last trading day: {last_trading_day}")
    logging.info(f"Last trading day: {last_trading_day}")

    api_semaphore = threading.Semaphore(2)  # limit concurrency of real trades

    csv_filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    file_path = os.path.join(CSV_DIRECTORY, csv_filename)

    def on_csv_changed():
        logging.info("CSV changed -> reading file & executing trades.")
        print("CSV changed -> reading file & executing trades.")
        execute_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    # Watchdog observer
    event_handler = CSVChangeHandler(csv_filename, on_csv_changed, debounce_sec=5)
    observer = Observer()
    observer.schedule(event_handler, path=CSV_DIRECTORY, recursive=False)
    observer.start()
    print(f"Watchdog started on {csv_filename}")
    logging.info(f"Watchdog started on {csv_filename}")

    # Optionally do one immediate check if within the trading window
    if start_time <= now <= end_time:
        print("Performing immediate trade execution check.")
        logging.info("Performing immediate trade execution check.")
        execute_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

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

