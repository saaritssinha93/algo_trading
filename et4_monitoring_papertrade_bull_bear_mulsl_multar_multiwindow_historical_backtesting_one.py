# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:55:42 2025

@author: Saarit

Edited version:
 - Only ONE target and ONE stop-loss at ±2%.
 - calculate_net_percentage() simplified to average out each ticker's 'Percentage Profit/Loss'.
 - Removed partial exit logic.
"""

import time
import threading
from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
import pytz
import tkinter as tk  # for GUI
from datetime import datetime
import csv
from zoneinfo import ZoneInfo

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)

logging.basicConfig(filename='trading_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

market_holidays = [
    dt.date(2024, 10, 2),  # Example holiday
    # Add more 2024 holidays here
]

def setup_kite_session():
    """Establishes a Kite Connect session."""
    try:
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()

        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        return kite

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error setting up Kite session: {e}")
        raise


def fetch_instruments(kite):
    """Fetch all NSE instruments and return a DataFrame."""
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        logging.info("NSE instrument data fetched successfully.")
        return instrument_df

    except Exception as e:
        logging.error(f"Error fetching instruments: {e}")
        raise


def instrument_lookup(instrument_df, symbol):
    """Look up instrument token for a given symbol."""
    try:
        instrument_token = instrument_df[instrument_df.tradingsymbol == symbol].instrument_token.values[0]
        return instrument_token
    except IndexError:
        logging.error(f"Symbol {symbol} not found in instrument dump.")
        return -1
    except Exception as e:
        logging.error(f"Error in instrument lookup: {e}")
        return -1


def calculate_atr(df, n=20):
    """Calculate Average True Range (ATR) over 'n' periods."""
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n, min_periods=1).mean()
    return df['ATR']


def atr_trailing_stop(df, atr_multiplier=3, atr_period=20):
    """Compute an ATR-based trailing stop for long positions."""
    atr = calculate_atr(df, n=atr_period)
    if len(atr) > atr_period:
        initial_stop = df['close'].iloc[0] - (atr_multiplier * atr.iloc[atr_period - 1])
    else:
        initial_stop = df['close'].iloc[0]

    trailing_stop = df['close'] - (atr_multiplier * atr)
    trailing_stop.iloc[:atr_period] = initial_stop
    return trailing_stop


def fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval='minute'):
    """Fetch historical data from Kite."""
    instrument_token = instrument_lookup(instrument_df, ticker)
    if instrument_token == -1:
        print(f"Instrument lookup failed for {ticker}")
        return pd.DataFrame()

    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        data = kite.historical_data(
            instrument_token,
            from_date=start_str,
            to_date=end_str,
            interval=interval
        )
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()


def monitor_ticker_in_window(
    ticker, buy_price, quantity, total_value_bought, trade_time,
    kite, instrument_df, interval, trend_type
):
    """
    Monitors a BULLISH (long) trade with:
      - +2% target
      - -2% stop-loss
      - forced exit at 3:15 PM
    """
    # Ensure it's actually bullish
    if trend_type.strip().lower() != 'bullish':
        print(f"Trend type for {ticker} is not Bullish. Skipping monitoring.")
        return

    # Create tkinter window for logging
    root = tk.Tk()
    root.title(f"Monitoring {ticker}")
    text_widget = tk.Text(root, height=20, width=100)
    text_widget.pack()

    def append_text(msg):
        text_widget.insert(tk.END, msg + '\n')
        text_widget.see(tk.END)
        root.update()

    append_text(f"Monitoring {ticker} from {trade_time} with a 2% target & stop-loss...")

    target_price = buy_price * 1.02
    stop_loss_price = buy_price * 0.98

    local_tz = ZoneInfo("Asia/Kolkata")
    start_time = trade_time
    end_time = dt.datetime(2025, 2, 6, tzinfo=local_tz)

    hist_df = fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval)
    if hist_df.empty:
        append_text(f"No historical data available for {ticker}. Skipping.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No data for {ticker}.")
        root.destroy()
        return

    # Prepare the result file
    result_file = f"result_papertrade_{trade_time.strftime('%Y-%m-%d')}.csv"
    if not os.path.isfile(result_file):
        with open(result_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Ticker", "date", "Shares Sold", "Total Buy", "Total Sold",
                             "Percentage Profit/Loss", "Trend Type"])

    # Monitor the price
    for _, row in hist_df.iterrows():
        current_time = pd.to_datetime(row['date']).tz_convert(local_tz)
        current_price = row['close']

        append_text(f"[{current_time.strftime('%H:%M:%S')}] {ticker} Price: ₹{current_price:.2f}")

        # 1) If it's 3:15 PM, exit the trade at market
        if current_time.hour == 15 and current_time.minute == 15:
            total_value_sold = current_price * quantity
            profit_or_loss = total_value_sold - total_value_bought
            percent_pnl = (profit_or_loss / total_value_bought) * 100
            append_text(f"Market Close => Sold all {quantity} @ ₹{current_price:.2f}, P/L: {percent_pnl:.2f}%")
            with open(result_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity,
                    total_value_bought, total_value_sold, f"{percent_pnl:.2f}%", trend_type
                ])
            break

        # 2) Check target
        if current_price >= target_price:
            total_value_sold = current_price * quantity
            profit = total_value_sold - total_value_bought
            percent_pnl = (profit / total_value_bought) * 100
            append_text(f"Target Hit => Sold {quantity} @ ₹{current_price:.2f}, Profit: {percent_pnl:.2f}%")
            with open(result_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity,
                    total_value_bought, total_value_sold, f"{percent_pnl:.2f}%", trend_type
                ])
            break

        # 3) Check stop-loss
        if current_price <= stop_loss_price:
            total_value_sold = current_price * quantity
            profit_or_loss = total_value_sold - total_value_bought
            percent_pnl = (profit_or_loss / total_value_bought) * 100
            append_text(f"Stop-loss Hit => Sold {quantity} @ ₹{current_price:.2f}, P/L: {percent_pnl:.2f}%")
            with open(result_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity,
                    total_value_bought, total_value_sold, f"{percent_pnl:.2f}%", trend_type
                ])
            break

        time.sleep(1)  # small delay to simulate real-time

    root.destroy()


def monitor_ticker_in_window_bear(
    ticker, sell_price, quantity, total_value_sold, trade_time,
    kite, instrument_df, interval, trend_type
):
    """
    Monitors a BEARISH (short) trade with:
      - +2% stop-loss (if price rises 2% from sell_price, we exit)
      - -2% target (if price falls 2% from sell_price, we exit)
      - forced exit at 3:15 PM
    """
    if trend_type.strip().lower() != 'bearish':
        print(f"Trend type for {ticker} is not Bearish. Skipping monitoring.")
        return

    root = tk.Tk()
    root.title(f"Monitoring {ticker}")
    text_widget = tk.Text(root, height=20, width=100)
    text_widget.pack()

    def append_text(msg):
        text_widget.insert(tk.END, msg + '\n')
        text_widget.see(tk.END)
        root.update()

    append_text(f"Monitoring {ticker} (Bearish) from {trade_time} with a 2% target & stop-loss...")

    target_price = sell_price * 0.98   # 2% down from sell_price
    stop_loss_price = sell_price * 1.02

    local_tz = ZoneInfo("Asia/Kolkata")
    start_time = trade_time
    end_time = dt.datetime(2025, 2, 6, tzinfo=local_tz)

    hist_df = fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval)
    if hist_df.empty:
        append_text(f"No historical data for {ticker}. Skipping.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No data for {ticker}.")
        root.destroy()
        return

    # Prepare result file
    result_file = f"result_papertrade_{trade_time.strftime('%Y-%m-%d')}.csv"
    if not os.path.isfile(result_file):
        with open(result_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Ticker", "date", "Shares Bought Back",
                "Total Sold", "Total Bought", "Percentage Profit/Loss", "Trend Type"
            ])

    for _, row in hist_df.iterrows():
        current_time = pd.to_datetime(row['date']).tz_convert(local_tz)
        current_price = row['close']

        append_text(f"[{current_time.strftime('%H:%M:%S')}] {ticker} Price: ₹{current_price:.2f}")

        # 1) If it's 3:15 PM, force buy-back at market
        if current_time.hour == 15 and current_time.minute == 15:
            total_value_bought = current_price * quantity
            pl = total_value_sold - total_value_bought
            percent_pnl = (pl / total_value_sold) * 100
            append_text(f"Market Close => Bought back {quantity} @ ₹{current_price:.2f}, P/L: {percent_pnl:.2f}%")
            with open(result_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    quantity, total_value_sold, total_value_bought,
                    f"{percent_pnl:.2f}%", trend_type
                ])
            break

        # 2) Target => price <= target_price
        if current_price <= target_price:
            total_value_bought = current_price * quantity
            # Profit for short = initial sold - final buy
            pl = total_value_sold - total_value_bought
            percent_pnl = (pl / total_value_sold) * 100
            append_text(f"Bearish Target Hit => Bought back {quantity} @ ₹{current_price:.2f}, P/L: {percent_pnl:.2f}%")
            with open(result_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    quantity, total_value_sold, total_value_bought,
                    f"{percent_pnl:.2f}%", trend_type
                ])
            break

        # 3) Stop-loss => price >= stop_loss_price
        if current_price >= stop_loss_price:
            total_value_bought = current_price * quantity
            pl = total_value_sold - total_value_bought
            percent_pnl = (pl / total_value_sold) * 100
            append_text(f"Bearish Stop-loss Hit => Bought back {quantity} @ ₹{current_price:.2f}, P/L: {percent_pnl:.2f}%")
            with open(result_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    quantity, total_value_sold, total_value_bought,
                    f"{percent_pnl:.2f}%", trend_type
                ])
            break

        time.sleep(1)

    root.destroy()


def monitor_paper_trades_backtest(kite, instrument_df, file_path, interval='minute'):
    """
    For all 'Bullish' trades in the specified CSV, monitor them using +2% target, -2% stop-loss.
    """
    try:
        paper_trades = pd.read_csv(file_path)
        if paper_trades.empty:
            print(f"No trades found in {file_path}.")
            return
    except FileNotFoundError:
        print(f"{file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    local_tz = pytz.timezone('Asia/Kolkata')
    threads = []

    # We'll iterate only over bullish trades
    bullish_mask = paper_trades['Trend Type'].str.strip().str.lower() == 'bullish'
    bullish_trades = paper_trades[bullish_mask]
    if bullish_trades.empty:
        print("No Bullish trades to monitor in this file.")
        return

    for _, row in bullish_trades.iterrows():
        ticker = row['Ticker']
        buy_price = float(row['Price'])
        quantity = int(row['Quantity'])
        total_value_bought = float(row['Total value'])
        trade_time = pd.to_datetime(row['date'])
        trend_type = row['Trend Type']

        # Localize or convert to IST
        if trade_time.tzinfo is None:
            trade_time = trade_time.tz_localize(local_tz)
        else:
            trade_time = trade_time.tz_convert(local_tz)

        t = threading.Thread(
            target=monitor_ticker_in_window,
            args=(ticker, buy_price, quantity, total_value_bought,
                  trade_time, kite, instrument_df, interval, trend_type)
        )
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()


def monitor_paper_trades_backtest_bear(kite, instrument_df, file_path, interval='minute'):
    """
    For all 'Bearish' trades in the specified CSV, monitor them using -2% target, +2% stop-loss.
    """
    try:
        paper_trades = pd.read_csv(file_path)
        if paper_trades.empty:
            print(f"No trades found in {file_path}.")
            return
    except FileNotFoundError:
        print(f"{file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    local_tz = pytz.timezone('Asia/Kolkata')
    threads = []

    # We'll iterate only over bearish trades
    bear_mask = paper_trades['Trend Type'].str.strip().str.lower() == 'bearish'
    bear_trades = paper_trades[bear_mask]
    if bear_trades.empty:
        print("No Bearish trades to monitor in this file.")
        return

    for _, row in bear_trades.iterrows():
        ticker = row['Ticker']
        sell_price = float(row['Price'])
        quantity = int(row['Quantity'])
        total_value_sold = float(row['Total value'])
        trade_time = pd.to_datetime(row['date'])
        trend_type = row['Trend Type']

        # Localize or convert to IST
        if trade_time.tzinfo is None:
            trade_time = trade_time.tz_localize(local_tz)
        else:
            trade_time = trade_time.tz_convert(local_tz)

        t = threading.Thread(
            target=monitor_ticker_in_window_bear,
            args=(ticker, sell_price, quantity, total_value_sold,
                  trade_time, kite, instrument_df, interval, trend_type)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def calculate_net_percentage(last_trading_days):
    """
    Calculate net percentage P/L for each trading day.
    Each day has a CSV 'result_papertrade_{trading_day}.csv'.
    We'll average the 'Percentage Profit/Loss' by ticker, then
    compute an overall average across tickers.

    For each day, create `result_final_{trading_day}.csv`.
    """
    for trading_day in last_trading_days:
        file_path = f"result_papertrade_{trading_day}.csv"
        output_file = f"result_final_{trading_day}.csv"

        if not os.path.isfile(file_path):
            print(f"No result file for {trading_day}. Skipping.")
            continue

        df = pd.read_csv(file_path)
        if df.empty or 'Percentage Profit/Loss' not in df.columns:
            print(f"No results or missing columns in {file_path}.")
            continue

        # Convert 'xx%' strings to float
        df['Percentage Profit/Loss'] = df['Percentage Profit/Loss'].str.rstrip('%').astype(float)

        # Group by Ticker and average
        grouped = df.groupby('Ticker')['Percentage Profit/Loss'].mean().reset_index()
        grouped.columns = ['Ticker', 'Net Percentage']

        # Calculate overall average
        overall_avg = grouped['Net Percentage'].mean()

        # Append an 'Overall' row
        overall_row = pd.DataFrame([['Overall', round(overall_avg, 4)]],
                                   columns=['Ticker', 'Net Percentage'])
        final_df = pd.concat([grouped, overall_row], ignore_index=True)

        # Save to CSV
        final_df.to_csv(output_file, index=False)
        print(f"Net Percentage for {trading_day} = {round(overall_avg, 4)}%")
        print(f"Saved => {output_file}")


def main():
    """
    Main driver:
      - Attempt to set up Kite session,
      - fetch instruments,
      - for each trading day, monitor bullish/bearish entries,
      - then compute final net percentages.
    """
    last_trading_days = [
        '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30',
        '2025-01-31', '2025-02-03', '2025-02-04', '2025-02-05'
    ]

    logging.info("Starting the trading algorithm...")
    kite = setup_kite_session()
    instrument_df = fetch_instruments(kite)

    for trading_day in last_trading_days:
        logging.info(f"Processing trades for {trading_day}...")
        file_path = f'papertrade_{trading_day}.csv'

        if not os.path.isfile(file_path):
            logging.error(f"File not found => {file_path}. Skipping day.")
            continue

        # Monitor bullish
        monitor_paper_trades_backtest(kite, instrument_df, file_path, interval='minute')
        # Monitor bearish
        monitor_paper_trades_backtest_bear(kite, instrument_df, file_path, interval='minute')

    # After monitoring all days, calculate net P/L
    calculate_net_percentage(last_trading_days)
    logging.info("All done. Net percentages computed.")


if __name__ == "__main__":
    main()
    logging.shutdown()
