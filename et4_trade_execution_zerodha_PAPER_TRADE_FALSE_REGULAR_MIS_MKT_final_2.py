# -*- coding: utf-8 -*-
"""
Code 2 (Concurrent Real Trades) with Forced Close at 15:15:
 - Reads trades from 'papertrade_{YYYY-MM-DD}.csv' in CSV_DIRECTORY.
 - Places real or paper trades for each CSV row, concurrently.
 - DOES NOT skip repeated tickers; multiple trades per ticker are allowed.
 - Skips re-trading the same Signal_ID multiple times.
 - Places MIS order with 2% target & 2% SL, snapped to nearest 0.5 increments.
 - Spawns a background thread per real trade, so multiple real trades can run in parallel.
 - Any real trade still open at 15:15 is forcibly closed automatically (with retries).
FIXES:
 - When SL or target is hit, we reliably cancel the other (with retries).
 - At 15:15, we reliably cancel all pending orders & close the position.
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

import pandas as pd
from kiteconnect import KiteConnect

from datetime import datetime, timedelta, time as datetime_time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

############################################
# 1) Configuration
############################################
PAPER_TRADE = False  # Toggle True/False for simulation vs. real trades
CSV_DIRECTORY = r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo"
CSV_FILENAME_PATTERN = "papertrade_{}.csv"

EXECUTED_SIGNALS_FILE = "executed_signals_mis_mkt.json"
EXPECTED_COLUMNS = ["Ticker", "date", "Trend Type", "Signal_ID", "Price", "Quantity"]

# Max number of real-trade threads that can run at once
MAX_CONCURRENT_TRADES = 20

############################################
# 2) Logging Setup
############################################
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

############################################
# 3) Kite Connect Session
############################################
def setup_kite_session():
    """
    Reads 'access_token.txt' and 'api_key.txt' to set up KiteConnect.
    """
    try:
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()

        with open("api_key.txt", "r") as f:
            key_data = f.read().strip().split()
        if len(key_data) < 2:
            raise ValueError("api_key.txt must have 'API_KEY' and 'API_SECRET' separated by space.")

        api_key, api_secret = key_data[0], key_data[1]
        kite_ = KiteConnect(api_key=api_key)
        kite_.set_access_token(access_token)

        logging.info("Kite session established.")
        print("Kite Connect session initialized successfully.")
        return kite_

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
    logging.warning("Kite session not established. PAPER_TRADE only?")

############################################
# 4) Executed Signals (JSON)
############################################
def load_executed_signals():
    """
    Load already executed Signal_IDs to avoid re-trading them.
    """
    if os.path.exists(EXECUTED_SIGNALS_FILE):
        try:
            with open(EXECUTED_SIGNALS_FILE, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Executed signals file corrupted. Starting empty.")
            return set()
    else:
        return set()

def save_executed_signals(executed_signals_set):
    with open(EXECUTED_SIGNALS_FILE, 'w') as f:
        json.dump(list(executed_signals_set), f)

executed_signals = load_executed_signals()

############################################
# 5) Utilities
############################################
def generate_signal_id(ticker, dt_obj, transaction_type):
    """
    Fallback generator if CSV lacks 'Signal_ID'
    """
    unique_id = uuid.uuid4()
    return f"{ticker}-{dt_obj.isoformat()}-{transaction_type}-{unique_id}"

def current_market_price(kite_, ticker):
    """
    Get LTP from Kite. If no Kite session, or error => returns 0.0
    """
    if kite_ is None:
        return 0.0
    try:
        data = kite_.ltp(f"NSE:{ticker}")
        return float(data[f"NSE:{ticker}"]['last_price'])
    except Exception as e:
        logging.error(f"Failed fetching LTP for {ticker}: {e}")
        print(f"Failed fetching LTP for {ticker}: {e}")
        return 0.0

def round_to_nearest_05(value: float) -> float:
    """
    Round float to nearest 0.5
    """
    return round(value * 2) / 2.0

def cancel_order_with_retry(kite_, variety, order_id, max_retries=3, wait_sec=2):
    """
    Tries up to `max_retries` times to cancel the order.
    Returns True if successful, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            kite_.cancel_order(variety, order_id)
            logging.info(f"Cancel order {order_id} success on attempt {attempt}.")
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} to cancel order {order_id} failed: {e}")
            time.sleep(wait_sec)
    return False

############################################
# 6) Logging Paper/Real Trades
############################################
def log_simulation_result(ticker, transaction_type, price, quantity,
                          target_price, stop_loss_price, order_id):
    """
    Append a record to simulation_papertrade_{YYYY-MM-DD}.csv
    for hypothetical trades.
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
    Append real MIS trade record to trade_log_{YYYY-MM-DD}.csv
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

############################################
# 7) Reading & Normalizing CSV
############################################
def read_and_normalize_csv(file_path, expected_columns, timezone='Asia/Kolkata'):
    """
    Reads CSV, ensures columns exist, normalizes 'date' to Asia/Kolkata,
    returns cleaned DataFrame.
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

    missing_cols = set(expected_columns) - set(df.columns)
    for mc in missing_cols:
        df[mc] = ""

    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        logging.warning(f"Dropping extra columns: {extra_cols}")
        df.drop(columns=list(extra_cols), inplace=True)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df.dropna(subset=['date'], inplace=True)
        df['date'] = df['date'].dt.tz_convert(timezone)
    else:
        df['date'] = pd.NaT

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

    df = df[expected_columns].copy()
    df.reset_index(drop=True, inplace=True)
    return df

############################################
# 8) Real MIS: place entry / target / SL
#    + Force close any open trade at 15:15
############################################
def place_mis_trade(kite_, ticker, transaction_type, quantity):
    """
    Real trade logic:
      1) place market entry
      2) wait fill
      3) place 2% target (LIMIT) + 2% SL (SL-M)
      4) monitor until target/SL hits or 15:15 => forcibly close
FIXES:
 - If SL is hit, we now reliably cancel the Target with retry (and vice versa).
 - At 15:15, we retry cancellations, then place a market close.
"""

    def _log(msg):
        logging.info(msg)
        print(msg)

    india_tz = pytz.timezone("Asia/Kolkata")
    now_date = datetime.now(india_tz).date()
    forced_close_dt = india_tz.localize(datetime.combine(now_date, datetime_time(15, 15)))

    # 1) LTP
    ltp = current_market_price(kite_, ticker)
    if ltp == 0.0:
        msg = f"Could not fetch LTP for {ticker}. Aborting real trade."
        _log(msg)
        raise ValueError(msg)

    # 2% offsets
    if transaction_type.upper() == "BUY":
        raw_sl_price = ltp * 0.98
        raw_tgt_price = ltp * 1.015
        sl_transaction = "SELL"
        tgt_transaction = "SELL"
    else:
        raw_sl_price = ltp * 1.02
        raw_tgt_price = ltp * 0.985
        sl_transaction = "BUY"
        tgt_transaction = "BUY"

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
    _log(f"[REAL] Placed {transaction_type.upper()} MARKET for {ticker} qty={quantity}, ID={entry_order_id}")

    # 3) Wait fill
    filled = False
    filled_price = ltp
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

    # 4) Place target (LIMIT)
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
    _log(f"[REAL] Placed {tgt_transaction} LIMIT @ {tgt_price}, ID={target_order_id}")

    # 5) Place stop-loss (SL-M)
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
    _log(f"[REAL] Placed {sl_transaction} SL-M trigger={sl_price}, ID={stop_loss_order_id}")

    # 6) Wait loop: check target/SL fill or 15:15 forced close
    trade_active = True
    while trade_active:
        now_ist = datetime.now(india_tz)

        # Force close if time >= 15:15
        if now_ist >= forced_close_dt:
            _log(f"[REAL] Time >=15:15. Forcing close for {ticker}.")
            # Cancel target with retry
            cancelled_target = cancel_order_with_retry(kite_, kite_.VARIETY_REGULAR, target_order_id, max_retries=3)
            if not cancelled_target:
                _log(f"[REAL] Could NOT cancel target {target_order_id} after retries.")

            # Cancel SL with retry
            cancelled_sl = cancel_order_with_retry(kite_, kite_.VARIETY_REGULAR, stop_loss_order_id, max_retries=3)
            if not cancelled_sl:
                _log(f"[REAL] Could NOT cancel SL {stop_loss_order_id} after retries.")

            # Place forced close market
            if transaction_type.upper() == "BUY":
                close_txn_type = "SELL"
            else:
                close_txn_type = "BUY"

            try:
                close_order_id = kite_.place_order(
                    variety=kite_.VARIETY_REGULAR,
                    exchange=kite_.EXCHANGE_NSE,
                    tradingsymbol=ticker,
                    transaction_type=close_txn_type,
                    quantity=quantity,
                    product=kite_.PRODUCT_MIS,
                    order_type=kite_.ORDER_TYPE_MARKET,
                    validity=kite_.VALIDITY_DAY,
                    tag="ForcedClose"
                )
                _log(f"[REAL] Forced close placed => {close_txn_type} {ticker} qty={quantity}, ID={close_order_id}")
            except Exception as fe:
                _log(f"[REAL] Forced close attempt failed: {fe}")

            trade_active = False
            break

        # Otherwise check if target or SL is filled
        orders = kite_.orders()
        target_filled = any(o['order_id'] == target_order_id and o['status'] == 'COMPLETE' for o in orders)
        sl_filled = any(o['order_id'] == stop_loss_order_id and o['status'] == 'COMPLETE' for o in orders)

        if target_filled:
            _log(f"[REAL] Target {target_order_id} filled => Canceling SL {stop_loss_order_id}")
            cancel_order_with_retry(kite_, kite_.VARIETY_REGULAR, stop_loss_order_id, max_retries=3)
            trade_active = False

        elif sl_filled:
            _log(f"[REAL] SL {stop_loss_order_id} triggered => Canceling Target {target_order_id}")
            cancel_order_with_retry(kite_, kite_.VARIETY_REGULAR, target_order_id, max_retries=3)
            trade_active = False
        else:
            time.sleep(3)

    return {
        "entry_order_id": entry_order_id,
        "entry_fill_price": filled_price,
        "stop_loss_price": sl_price,
        "target_price": tgt_price
    }

############################################
# 9) Concurrency for Real Trades
############################################
def real_trade_thread(kite_, ticker, txn_type, quantity, signal_id, on_success, on_failure, semaphore=None):
    """
    Background thread to call place_mis_trade.
    We use a semaphore to limit concurrency.
    """
    if semaphore:
        semaphore.acquire()

    try:
        trade_info = place_mis_trade(kite_, ticker, txn_type, quantity)
        on_success(trade_info)
    except Exception as e:
        on_failure(e)
    finally:
        if semaphore:
            semaphore.release()

############################################
# 10) Master “Execute Trades” Flow
############################################
def execute_trades(last_trading_day, kite_, api_semaphore, paper_trade=True):
    """
    Reads 'papertrade_{last_trading_day}.csv', skip Signal_ID if executed,
    place trades (paper or real).
    """
    filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    file_path = os.path.join(CSV_DIRECTORY, filename)

    if not os.path.exists(file_path):
        msg = f"File {file_path} not found. No trades."
        logging.error(msg)
        print(msg)
        return

    try:
        df = read_and_normalize_csv(file_path, EXPECTED_COLUMNS)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return

    # Skip previously executed signals
    fresh_df = df[~df['Signal_ID'].isin(executed_signals)].copy()
    if fresh_df.empty:
        logging.info("No new trades to execute.")
        print("No new trades to execute.")
        return

    india_tz = pytz.timezone('Asia/Kolkata')
    threads = []

    for idx, row in fresh_df.iterrows():
        try:
            ticker = str(row['Ticker']).upper()
            trend = str(row['Trend Type']).lower().strip()
            signal_id = row['Signal_ID']
            csv_price = float(row.get('Price', 0) or 0)

            # Force quantity=1
            quantity = 1

            if trend == "bullish":
                txn_type = "BUY"
            elif trend == "bearish":
                txn_type = "SELL"
            else:
                logging.info(f"Unknown Trend Type '{trend}' => skip {ticker}")
                continue

            executed_signals.add(signal_id)

            if paper_trade:
                # PAPER TRADE
                if txn_type == 'BUY':
                    raw_sl_price = csv_price * 0.98
                else:
                    raw_sl_price = csv_price * 1.02
                sl_price = round_to_nearest_05(raw_sl_price)

                sim_id = f"SIM-{signal_id}-{datetime.now(india_tz).strftime('%H%M%S')}"
                print(f"[SIM] {txn_type} {ticker} @ {csv_price}, SL={sl_price}, ID={sim_id}")
                logging.info(f"[SIM] {txn_type} {ticker} @ {csv_price}, SL={sl_price}, ID={sim_id}")

                log_simulation_result(
                    ticker=ticker,
                    transaction_type=txn_type,
                    price=csv_price,
                    quantity=quantity,
                    target_price=None,
                    stop_loss_price=sl_price,
                    order_id=sim_id
                )

            else:
                # REAL TRADE
                if kite_ is None:
                    print("Kite session not available => skipping real trade.")
                    continue

                def on_success(trade_info):
                    log_trade_result(
                        ticker=ticker,
                        transaction_type=txn_type,
                        price=trade_info['entry_fill_price'],
                        quantity=quantity,
                        target_price=trade_info['target_price'],
                        stop_loss_price=trade_info['stop_loss_price'],
                        order_id=trade_info['entry_order_id']
                    )

                def on_failure(err):
                    msg = f"[REAL] Trade failed for {ticker}: {err}"
                    logging.error(msg)
                    print(msg)
                    # Optionally revert executed_signals.remove(signal_id)

                t = threading.Thread(
                    target=real_trade_thread,
                    args=(
                        kite_,
                        ticker,
                        txn_type,
                        quantity,
                        signal_id,
                        on_success,
                        on_failure,
                        api_semaphore
                    ),
                    daemon=True
                )
                t.start()
                threads.append(t)

        except Exception as row_exc:
            logging.error(f"Error processing row {idx}: {row_exc}")
            print(f"Error processing row {idx}: {row_exc}")

    save_executed_signals(executed_signals)
    logging.info("Submitted trades. Running in background threads.")
    print("Submitted trades. Background threads working...")

############################################
# 11) Watchdog Handler
############################################
class CSVChangeHandler(FileSystemEventHandler):
    """
    On papertrade_{YYYY-MM-DD}.csv changed, we wait a few seconds (debounce)
    then call `callback()`.
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

############################################
# 12) Main
############################################
def main():
    india_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india_tz)

    # Trading window e.g. 9:31 to 15:30
    start_naive = datetime.combine(now.date(), datetime_time(9, 30))
    end_naive = datetime.combine(now.date(), datetime_time(15, 30))
    start_time = india_tz.localize(start_naive)
    end_time = india_tz.localize(end_naive)

    if now < start_time:
        print("Currently before trading window. Watchdog active.")
        logging.info("Currently before trading window. Watchdog active.")
    elif now > end_time:
        print("Currently after trading window. We'll watch for CSV changes, but it may be out-of-hours.")
        logging.warning("Currently after trading window. May be OOH for trades.")
    else:
        print("Within trading window. Ready to trade if CSV changes.")
        logging.info("Within trading window. Ready for trades if CSV changes.")

    # Example market holidays
    market_holidays = [
        datetime(2024, 1, 26).date(),
        datetime(2024, 3, 14).date(),
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
        while d.weekday() >= 5 or d in market_holidays:
            d -= timedelta(days=1)
        return d.strftime('%Y-%m-%d')

    last_trading_day = get_last_trading_day()
    print(f"Last trading day: {last_trading_day}")
    logging.info(f"Last trading day: {last_trading_day}")

    # Concurrency semaphore
    api_semaphore = threading.Semaphore(MAX_CONCURRENT_TRADES)

    csv_filename = CSV_FILENAME_PATTERN.format(last_trading_day)
    file_path = os.path.join(CSV_DIRECTORY, csv_filename)

    def on_csv_changed():
        logging.info("CSV changed -> reading & executing trades.")
        print("CSV changed -> reading & executing trades.")
        execute_trades(last_trading_day, kite, api_semaphore, paper_trade=PAPER_TRADE)

    event_handler = CSVChangeHandler(csv_filename, on_csv_changed, debounce_sec=5)
    observer = Observer()
    observer.schedule(event_handler, path=CSV_DIRECTORY, recursive=False)
    observer.start()
    print(f"Watchdog started on {csv_filename}")
    logging.info(f"Watchdog started on {csv_filename}")

    # One immediate check if within trading window
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
