# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:26:33 2024

@author: Saarit
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
import tkinter as tk  # Import tkinter for GUI and alias as tk
from datetime import datetime
import csv
from zoneinfo import ZoneInfo



# Define the correct path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)

# Set up logging
logging.basicConfig(filename='trading_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Known market holidays for 2024 (example)
market_holidays = [
    dt.date(2024, 10, 2),  # Gandhi Jayanti
    # Add other known holidays for the year here
]

# Function to setup Kite Connect session
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

# Function to fetch all NSE instruments
def fetch_instruments(kite):
    """Fetches all NSE instruments and returns a DataFrame."""
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        logging.info("NSE instrument data fetched successfully.")
        return instrument_df

    except Exception as e:
        logging.error(f"Error fetching instruments: {e}")
        raise

# Function to lookup instrument token
def instrument_lookup(instrument_df, symbol):
    """Looks up instrument token for a given symbol in the instrument dump."""
    try:
        instrument_token = instrument_df[instrument_df.tradingsymbol == symbol].instrument_token.values[0]
        return instrument_token
    except IndexError:
        logging.error(f"Symbol {symbol} not found in instrument dump.")
        return -1
    except Exception as e:
        logging.error(f"Error in instrument lookup: {e}")
        return -1


# Function to calculate the ATR (Average True Range)
def calculate_atr(df, n=20):
    """Calculate the Average True Range (ATR) over 'n' periods."""
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n, min_periods=1).mean()
    return df['ATR']

# Function to initialize and update the ATR trailing stop
def atr_trailing_stop(df, atr_multiplier=3, atr_period=20):
    """
    Calculate an ATR-based trailing stop for long positions.
    
    Parameters:
    - atr_multiplier: Multiplier for ATR to set stop distance.
    - atr_period: Period for ATR calculation.
    """
    atr = calculate_atr(df, n=atr_period)
    
    # Set the initial trailing stop as a fixed distance from the first close price
    initial_stop = df['close'].iloc[0] - (atr_multiplier * atr.iloc[atr_period-1]) if len(atr) > atr_period else df['close'].iloc[0]
    
    # Calculate the trailing stop distance for subsequent periods
    trailing_stop = df['close'] - (atr_multiplier * atr)
    
    # Set initial stop level for NaN values at the start
    trailing_stop.iloc[:atr_period] = initial_stop
    
    return trailing_stop




# Function to fetch historical price data
def fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval='minute'):
    """
    Fetch historical price data for backtesting.

    Args:
    - ticker (str): Stock ticker symbol.
    - kite (KiteConnect): Kite Connect instance.
    - instrument_df (pd.DataFrame): DataFrame containing instrument information.
    - start_time (datetime): Start time for historical data.
    - end_time (datetime): End time for historical data.
    - interval (str): Time interval for the data (e.g., 'minute', '5minute').

    Returns:
    - pd.DataFrame: DataFrame of historical price data.
    """
    instrument_token = instrument_lookup(instrument_df, ticker)
    if instrument_token == -1:
        print(f"Instrument lookup failed for {ticker}")
        return None

    # Convert start and end times to required string format
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Fetch historical data from KiteConnect
        historical_data = kite.historical_data(
            instrument_token, 
            from_date=start_time_str, 
            to_date=end_time_str, 
            interval=interval
        )
        return pd.DataFrame(historical_data)

    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    
    

def monitor_ticker_in_window(ticker, buy_price, quantity, total_value_bought, trade_time, kite, instrument_df, target_percentage, interval, trend_type):
        # Check if the trend type is Bullish
    if trend_type.strip().lower() != 'bullish':
        print(f"Trend type for {ticker} is not Bearish. Skipping monitoring.")
        return
    root = tk.Tk()
    root.title(f"Monitoring {ticker}")
    text_widget = tk.Text(root, height=20, width=100)
    text_widget.pack()

    def append_text(message, window=text_widget):
        window.insert(tk.END, message + '\n')
        window.see(tk.END)
        root.update()

    append_text(f"Monitoring {ticker} from {trade_time}...")

    half_quantity = quantity // 2
    remaining_quantity = quantity - half_quantity

    first_target_price = buy_price * (1 + 0.015)  # First target at 1.5% above buy price
    first_stop_loss_price = buy_price * (1 - 0.015)  # Initial stop-loss at 1.5% below buy price

    second_stop_loss_price = None  # Placeholder for second stop-loss after first execution
    second_target_price = None  # Placeholder for second target after first execution

    local_tz = pytz.timezone('Asia/Kolkata')
    start_time = trade_time
    

    # Specify the local timezone
    local_tz = ZoneInfo("Asia/Kolkata")  # Replace with your local timezone
    end_time = dt.datetime(2025, 3, 7, tzinfo=local_tz)

    historical_prices = fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval)
    if historical_prices.empty:
        append_text(f"No historical data available for {ticker}. Skipping.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No historical data available for {ticker}. Skipping.")
        root.mainloop()
        return

    # Prepare the result file header if it doesn't exist
    result_file = f"result_papertrade2_{trade_time.strftime('%Y-%m-%d')}.csv"
    if not os.path.isfile(result_file):
        with open(result_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Ticker", "date", "Shares Sold", "Total Buy", "Total Sold", "Percentage Profit/Loss", "Trend Type"])

    first_execution_done = False

    for _, price_row in historical_prices.iterrows():
        current_time = pd.to_datetime(price_row['date']).tz_convert(local_tz)
        current_price = price_row['close']

        # Print detailed ticker data on console
        if not first_execution_done:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring {ticker} | "
                  f"Current Price: ₹{current_price:.2f} | "
                  f"Buy Price: ₹{buy_price:.2f} | "
                  f"First Target: ₹{first_target_price:.2f} | "
                  f"First Stop Loss: ₹{first_stop_loss_price:.2f}")
        else:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring {ticker} | "
                  f"Current Price: ₹{current_price:.2f} | "
                  f"Buy Price: ₹{buy_price:.2f} | "
                  f"Second Target: ₹{second_target_price:.2f} | "
                  f"Second Stop Loss: ₹{second_stop_loss_price:.2f}")

        append_text(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Ticker: {ticker}, Current Price: ₹{current_price:.2f}")

        # Check if the current time is 3:15 PM
        if current_time.hour == 15 and current_time.minute == 15:
            total_value_sold = current_price * quantity
            profit_or_loss = total_value_sold - total_value_bought
            percentage_change = (profit_or_loss / total_value_bought) * 100
            append_text(f"Market close sell for {ticker} at ₹{current_price} on {current_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"Profit/Loss: ₹{profit_or_loss} ({percentage_change:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FINAL: Market close for {ticker}. Sold all {quantity} shares at ₹{current_price:.2f}. "
                  f"Profit/Loss: ₹{profit_or_loss:.2f} ({percentage_change:.2f}%)")

            # Write result to CSV file
            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity, total_value_bought, total_value_sold, f"{percentage_change:.2f}%", trend_type])
            break

        # Handle the first target and stop-loss
        if not first_execution_done:
            if current_price >= first_target_price:
                total_value_sold = current_price * half_quantity
                profit = total_value_sold - (half_quantity * buy_price)
                percentage_profit = (profit / (half_quantity * buy_price)) * 100  # Adjusted for first execution
                append_text(f"First target hit for {ticker}! Sold {half_quantity} shares at ₹{current_price:.2f}. "
                            f"Profit: ₹{profit:.2f} ({percentage_profit:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: First target hit for {ticker}. Sold {half_quantity} shares at ₹{current_price:.2f}. "
                      f"Updated Target and Stop Loss.")

                # Update stop-loss and target for remaining quantities
                second_stop_loss_price = first_target_price * (1 - 0.005)  # 1% below the current price
                second_target_price = first_target_price * (1 + 0.005)  # 1% above the current price
                first_execution_done = True

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), half_quantity, total_value_bought, total_value_sold, f"{percentage_profit:.2f}%", trend_type])

            elif current_price <= first_stop_loss_price:
                total_value_sold = first_stop_loss_price * quantity
                loss = (quantity * buy_price) - total_value_sold
                percentage_loss = (loss / (quantity * buy_price)) * -100  # Adjusted for first execution
                append_text(f"Stop-loss hit for {ticker}! Sold all {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                            f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL: Stop-loss hit for {ticker}. Sold all {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                      f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity, total_value_bought, total_value_sold, f"{percentage_loss:.2f}%", trend_type])
                break

        # Handle the second target and stop-loss
        elif first_execution_done:
            if current_price >= second_target_price:
                total_value_sold = current_price * remaining_quantity
                profit = total_value_sold - (remaining_quantity * first_target_price)
                percentage_profit = (profit / (remaining_quantity * first_target_price)) * 100  # Adjusted for second execution
                append_text(f"Second target hit for {ticker}! Sold remaining {remaining_quantity} shares at ₹{current_price:.2f}. "
                            f"Profit: ₹{profit:.2f} ({percentage_profit:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Second target hit for {ticker}. Sold remaining {remaining_quantity} shares.")

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), remaining_quantity, total_value_bought, total_value_sold, f"{percentage_profit:.2f}%", trend_type])
                break

            elif current_price <= second_stop_loss_price:
                total_value_sold = second_stop_loss_price * remaining_quantity
                loss = (remaining_quantity * first_target_price) - total_value_sold
                percentage_loss = (loss / (remaining_quantity * first_target_price)) * -100  # Adjusted for second execution
                append_text(f"Second stop-loss hit for {ticker}! Sold remaining {remaining_quantity} shares at ₹{second_stop_loss_price:.2f}. "
                            f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL: Second stop-loss hit for {ticker}. Sold remaining {remaining_quantity} shares.")

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), remaining_quantity, total_value_bought, total_value_sold, f"{percentage_loss:.2f}%", trend_type])
                break

        time.sleep(1)  # Simulate time passing

    print(f"Monitoring complete for {ticker}. Closing console and GUI.")
    root.destroy()  # Close the tkinter window



def monitor_paper_trades_backtest(kite, instrument_df, file_path, target_percentage=1.5, interval='minute'):
    try:
        paper_trades = pd.read_csv(file_path)
        if paper_trades.empty:
            print("No active trades found.")
            return
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    local_tz = pytz.timezone('Asia/Kolkata')
    threads = []

    for _, row in paper_trades.iterrows():
        ticker = row['Ticker']
        buy_price = row['Price']
        quantity = row['Quantity']
        total_value_bought = row['Total value']
        trade_time = pd.to_datetime(row['date'])
        trend_type = row['Trend Type']
        
        # Use tz_convert if the timestamp is already tz-aware
        if trade_time.tzinfo:
            trade_time = trade_time.tz_convert(local_tz)
        else:  # Use tz_localize if the timestamp is naive
            trade_time = trade_time.tz_localize(local_tz)

        thread = threading.Thread(
            target=monitor_ticker_in_window,
            args=(ticker, buy_price, quantity, total_value_bought, trade_time, kite, instrument_df, target_percentage, interval, trend_type)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def monitor_ticker_in_window_bear(ticker, sell_price, quantity, total_value_sold, trade_time, kite, instrument_df, interval, trend_type):
        # Check if the trend type is Bearish
    if trend_type.strip().lower() != 'bearish':
        print(f"Trend type for {ticker} is not Bullish. Skipping monitoring.")
        return
    root = tk.Tk()
    root.title(f"Monitoring {ticker}")
    text_widget = tk.Text(root, height=20, width=100)
    text_widget.pack()

    def append_text(message, window=text_widget):
        window.insert(tk.END, message + '\n')
        window.see(tk.END)
        root.update()

    append_text(f"Monitoring {ticker} from {trade_time}...")

    half_quantity = quantity // 2
    remaining_quantity = quantity - half_quantity

    first_target_price = sell_price * (1 - 0.015)  # First target at 2% below sell price
    first_stop_loss_price = sell_price * (1 + 0.015)  # Stop-loss at 2% above sell price

    second_target_price = None  # Placeholder for second target after first execution
    second_stop_loss_price = None  # Placeholder for second stop-loss after first execution

    local_tz = pytz.timezone('Asia/Kolkata')
    start_time = trade_time
    # Specify the local timezone
    local_tz = ZoneInfo("Asia/Kolkata")  # Replace with your local timezone
    end_time = dt.datetime(2025, 3, 7, tzinfo=local_tz)

    historical_prices = fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval)
    if historical_prices.empty:
        append_text(f"No historical data available for {ticker}. Skipping.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No historical data available for {ticker}. Skipping.")
        root.destroy()
        return

    # Prepare the result file header if it doesn't exist
    result_file = f"result_papertrade2_{trade_time.strftime('%Y-%m-%d')}.csv"
    if not os.path.isfile(result_file):
        with open(result_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Ticker", "date", "Shares Bought Back", "Total Sold", "Total Bought", "Percentage Profit/Loss", "Trend Type"])

    first_execution_done = False

    for _, price_row in historical_prices.iterrows():
        current_time = pd.to_datetime(price_row['date']).tz_convert(local_tz)
        current_price = price_row['close']

        # Print detailed ticker data on console
        if not first_execution_done:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring {ticker} | "
                  f"Current Price: ₹{current_price:.2f} | "
                  f"Sell Price: ₹{sell_price:.2f} | "
                  f"First Target: ₹{first_target_price:.2f} | "
                  f"First Stop Loss: ₹{first_stop_loss_price:.2f}")
        else:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring {ticker} | "
                  f"Current Price: ₹{current_price:.2f} | "
                  f"Sell Price: ₹{sell_price:.2f} | "
                  f"Second Target: ₹{second_target_price:.2f} | "
                  f"Second Stop Loss: ₹{second_stop_loss_price:.2f}")

        append_text(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Ticker: {ticker}, Current Price: ₹{current_price:.2f}")

        # Check if the current time is 3:15 PM
        if current_time.hour == 15 and current_time.minute == 15:
            total_value_bought = current_price * quantity
            profit_or_loss = total_value_sold - total_value_bought
            percentage_change = (profit_or_loss / total_value_sold) * 100
            append_text(f"Market close buy-back for {ticker} at ₹{current_price} on {current_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"Profit/Loss: ₹{profit_or_loss} ({percentage_change:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FINAL: Market close for {ticker}. Bought back all {quantity} shares at ₹{current_price:.2f}. "
                  f"Profit/Loss: ₹{profit_or_loss:.2f} ({percentage_change:.2f}%)")

            # Write result to CSV file
            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity, total_value_sold, total_value_bought, f"{percentage_change:.2f}%", trend_type])
            break

        # Handle the first target and stop-loss
        if not first_execution_done:
            if current_price <= first_target_price:
                total_value_bought = current_price * half_quantity
                profit = -(total_value_bought - (half_quantity * sell_price))
                percentage_profit = profit / (half_quantity * sell_price) * 100
                append_text(f"First target hit for {ticker}! Bought back {half_quantity} shares at ₹{current_price:.2f}. "
                            f"Profit: ₹{profit:.2f} ({percentage_profit:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: First target hit for {ticker}. Bought back {half_quantity} shares at ₹{current_price:.2f}. "
                      f"Updated Target and Stop Loss.")

                # Update second target and stop-loss for remaining quantities
                second_target_price = first_target_price * (1 - 0.005)  # 1% below the current target
                second_stop_loss_price = first_target_price * (1 + 0.005)  # 1% above the current target
                first_execution_done = True

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), half_quantity, total_value_sold, total_value_bought, f"{percentage_profit:.2f}%", trend_type])

            elif current_price >= first_stop_loss_price:
                total_value_bought = first_stop_loss_price * quantity
                loss = (quantity * sell_price) - total_value_bought
                percentage_loss = (loss / (quantity * sell_price)) * 100  # Adjusted for first execution
                append_text(f"Stop-loss hit for {ticker}! Bought back all {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                            f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL: Stop-loss hit for {ticker}. Bought back all {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                      f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), quantity, total_value_sold, total_value_bought, f"{percentage_loss:.2f}%", trend_type])
                break

        # Handle the second target and stop-loss
        elif first_execution_done:
            if current_price <= second_target_price:
                total_value_bought = current_price * remaining_quantity
                profit = -(total_value_bought - (remaining_quantity * first_target_price))
                percentage_profit = (profit / (remaining_quantity * first_target_price)) * 100  # Adjusted for second execution
                append_text(f"Second target hit for {ticker}! Bought back remaining {remaining_quantity} shares at ₹{current_price:.2f}. "
                            f"Profit: ₹{profit:.2f} ({percentage_profit:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Second target hit for {ticker}. Bought back remaining {remaining_quantity} shares.")

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), remaining_quantity, total_value_sold, total_value_bought, f"{percentage_profit:.2f}%", trend_type])
                break

            elif current_price >= second_stop_loss_price:
                total_value_bought = second_stop_loss_price * remaining_quantity
                loss = (remaining_quantity * first_target_price) - total_value_bought
                percentage_loss = (loss / (remaining_quantity * first_target_price)) * 100  # Adjusted for second execution
                append_text(f"Second stop-loss hit for {ticker}! Bought back remaining {remaining_quantity} shares at ₹{second_stop_loss_price:.2f}. "
                            f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL: Second stop-loss hit for {ticker}. Bought back remaining {remaining_quantity} shares.")

                # Write result to CSV file
                with open(result_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ticker, current_time.strftime('%Y-%m-%d %H:%M:%S'), remaining_quantity, total_value_sold, total_value_bought, f"{percentage_loss:.2f}%", trend_type])
                break

        time.sleep(1)  # Simulate time passing

    print(f"Monitoring completed for {ticker}. Closing console and GUI.")
    root.destroy()  # Close the tkinter window



def monitor_paper_trades_backtest_bear(kite, instrument_df, file_path, target_percentage=1.5, interval='minute'):
    try:
        paper_trades = pd.read_csv(file_path)
        if paper_trades.empty:
            print("No active trades found.")
            return
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    local_tz = pytz.timezone('Asia/Kolkata')
    threads = []

    for index, row in paper_trades.iterrows():
        ticker = row['Ticker']
        sell_price = row['Price']  # Change from 'Buy Price' to 'Sell Price' for bearish strategy
        quantity = row['Quantity']
        total_value_sold = row['Total value']  # Change from 'Total Value Bought' to 'Total Value Sold'
        trade_time = pd.to_datetime(row['date'])
        trend_type = row['Trend Type']

        if trade_time.tzinfo is None:
            trade_time = trade_time.tz_localize(local_tz)
        else:
            trade_time = trade_time.tz_convert(local_tz)

        # Start a thread to monitor each ticker
        thread = threading.Thread(
            target=monitor_ticker_in_window_bear,
            args=(ticker, sell_price, quantity, total_value_sold, trade_time, kite, instrument_df, interval, trend_type)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

from glob import glob

def calculate_net_percentage(last_trading_days):
    """
    Calculate the net percentage profit/loss for each trading day and save results to individual files.
    Each file will be named as `result_final2_{last_trading_day}.csv`.

    Args:
    - last_trading_days (list): List of trading days to process.

    Returns:
    - None
    """
    for trading_day in last_trading_days:
        file_path = f"result_papertrade2_{trading_day}.csv"
        output_file = f"result_final2_{trading_day}.csv"
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping this trading day.")
            continue

        # Ensure Percentage Profit/Loss is numerical after removing '%'
        df['Percentage Profit/Loss'] = df['Percentage Profit/Loss'].str.rstrip('%').astype(float)

        # Initialize variables
        total_percentage = 0
        unique_ticker_count = 0
        results = []

        # Group by Ticker and calculate weighted profit/loss
        grouped = df.groupby('Ticker')
        for ticker, group in grouped:
            if len(group) > 1:  # If ticker has multiple entries
                calculated_percentage = (
                    group.iloc[0]['Percentage Profit/Loss'] * 2 + group.iloc[1]['Percentage Profit/Loss']
                ) / 2
            else:  # If ticker has a single entry
                calculated_percentage = group.iloc[0]['Percentage Profit/Loss']

            total_percentage += calculated_percentage
            unique_ticker_count += 1
            results.append({"Ticker": ticker, "Net Percentage": calculated_percentage})

        if unique_ticker_count == 0:
            print(f"No valid tickers found in {file_path}.")
            continue

        # Calculate the net percentage
        net_percentage = total_percentage / unique_ticker_count

        # Add overall result
        results.append({"Ticker": "Overall", "Net Percentage": round(net_percentage, 4)})

        # Save the results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

        print(f"Net Percentage for {trading_day}: {round(net_percentage, 4)}%")
        print(f"Results saved to {output_file}")




def main():
    '''
    last_trading_days = [
        '2025-02-10'
    ]
    '''
    last_trading_days = [
         "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
         "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
         "2025-02-24","2025-02-25","2025-02-27","2025-02-28",

         # March 2025
         "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
         "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
         "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
         "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
         "2025-03-31",

         # April 2025
         "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
         "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
         "2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18",
         "2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25",
         "2025-04-28","2025-04-29","2025-04-30"
        ]
    
    


    logging.info("Starting the trading algorithm...")
    kite = setup_kite_session()
    instrument_df = fetch_instruments(kite)

    for trading_day in last_trading_days:
        logging.info(f"Processing trades for {trading_day}...")
        
        file_path = f'papertrade2_{trading_day}.csv'
        try:
            # Read the paper trade file
            df = pd.read_csv(file_path)
            required_columns = ['Ticker', 'date', 'Entry Type', 'Trend Type', 'Price', 'VWAP']
            
            # Ensure all required columns exist
            if not set(required_columns).issubset(df.columns):
                logging.warning(f"Missing required columns in {file_path}. Skipping.")
                continue
            
            # Process trades based on Trend Type collectively
            try:
                bullish_entries = df[df['Trend Type'].str.strip().str.lower() == 'bullish']
                bearish_entries = df[df['Trend Type'].str.strip().str.lower() == 'bearish']

                if not bullish_entries.empty:
                    logging.info(f"Processing {len(bullish_entries)} bullish entries.")
                    monitor_paper_trades_backtest(kite, instrument_df, file_path, bullish_entries)

                if not bearish_entries.empty:
                    logging.info(f"Processing {len(bearish_entries)} bearish entries.")
                    monitor_paper_trades_backtest_bear(kite, instrument_df, file_path, bearish_entries)

            except Exception as e:
                logging.error(f"Error processing trades for {trading_day}: {e}")

        
        except FileNotFoundError:
            logging.error(f"File {file_path} not found. Skipping {trading_day}.")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            
        # Calculate net percentage profit/loss and save results
        net_percentage = calculate_net_percentage(last_trading_days)
        print(f"Net Percentage Profit/Loss: {net_percentage}%")
        logging.info(f"Net Percentage Profit/Loss: {net_percentage}% saved to results_final.csv")    





if __name__ == "__main__":
    main()
    logging.shutdown()
