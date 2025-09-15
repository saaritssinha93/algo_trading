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
import tkinter as tk  # Import tkinter for GUI
from datetime import datetime
import csv
from zoneinfo import ZoneInfo

cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)

logging.basicConfig(filename='trading_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

market_holidays = [
    dt.date(2024, 10, 2),  # Example holiday
]

def setup_kite_session():
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
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        logging.info("NSE instrument data fetched successfully.")
        return instrument_df
    except Exception as e:
        logging.error(f"Error fetching instruments: {e}")
        raise

def instrument_lookup(instrument_df, symbol):
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
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n, min_periods=1).mean()
    return df['ATR']

def atr_trailing_stop(df, atr_multiplier=3, atr_period=20):
    atr = calculate_atr(df, n=atr_period)
    initial_stop = df['close'].iloc[0] - (atr_multiplier * atr.iloc[atr_period-1]) if len(atr) > atr_period else df['close'].iloc[0]
    trailing_stop = df['close'] - (atr_multiplier * atr)
    trailing_stop.iloc[:atr_period] = initial_stop
    return trailing_stop

def fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval='minute'):
    instrument_token = instrument_lookup(instrument_df, ticker)
    if instrument_token == -1:
        print(f"Instrument lookup failed for {ticker}")
        return None
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        historical_data = kite.historical_data(
            instrument_token,
            from_date=start_time_str,
            to_date=end_time_str,
            interval=interval
        )
        return pd.DataFrame(historical_data)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

####################################################################
# REWRITTEN monitor_ticker_in_window => ONLY 1 TARGET, 1 STOP LOSS
####################################################################
def monitor_ticker_in_window(ticker, buy_price, quantity, total_value_bought, trade_time, kite, instrument_df, target_percentage, interval, trend_type):
    if trend_type.strip().lower() != 'bullish':
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

    # Single target & single stop-loss for full quantity
    first_target_price = buy_price * (1 + 0.02)   # 1.5% above buy price
    first_stop_loss_price = buy_price * (1 - 0.015)# 1.5% below buy price

    local_tz = pytz.timezone('Asia/Kolkata')
    start_time = trade_time
    local_tz = ZoneInfo("Asia/Kolkata")
    end_time = dt.datetime(2025, 5, 1, tzinfo=local_tz)

    historical_prices = fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval)
    if historical_prices.empty:
        append_text(f"No historical data available for {ticker}. Skipping.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No historical data for {ticker}. Skipping.")
        root.mainloop()
        return

    # Prepare the result file
    result_file = f"result_papertrade3_{trade_time.strftime('%Y-%m-%d')}.csv"
    if not os.path.isfile(result_file):
        with open(result_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Ticker", "date", "Shares Sold", "Total Buy", "Total Sold", "Percentage Profit/Loss", "Trend Type"])

    for _, price_row in historical_prices.iterrows():
        current_time = pd.to_datetime(price_row['date']).tz_convert(local_tz)
        current_price = price_row['close']

        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring {ticker} | "
              f"Current Price: ₹{current_price:.2f} | "
              f"Buy Price: ₹{buy_price:.2f} | "
              f"Target: ₹{first_target_price:.2f} | "
              f"Stop Loss: ₹{first_stop_loss_price:.2f}")

        append_text(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Ticker: {ticker}, Current: ₹{current_price:.2f}")

        # 3:15 PM => close the position
        if current_time.hour == 15 and current_time.minute == 15:
            total_value_sold = current_price * quantity
            profit_or_loss = total_value_sold - total_value_bought
            percentage_change = (profit_or_loss / total_value_bought) * 100

            append_text(f"Market close sell for {ticker} at ₹{current_price}, P/L: ₹{profit_or_loss} ({percentage_change:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FINAL: Market close for {ticker}. "
                  f"Sold {quantity} shares at ₹{current_price:.2f}. "
                  f"P/L: ₹{profit_or_loss:.2f} ({percentage_change:.2f}%)")

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker,
                                 current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 quantity,
                                 total_value_bought,
                                 total_value_sold,
                                 f"{percentage_change:.2f}%",
                                 trend_type])
            break

        # SINGLE Target => close full quantity
        if current_price >= first_target_price:
            total_value_sold = current_price * quantity
            profit = total_value_sold - total_value_bought
            percentage_profit = (profit / total_value_bought) * 100
            append_text(f"Target hit for {ticker}! Sold {quantity} shares at ₹{current_price:.2f}. "
                        f"Profit: ₹{profit:.2f} ({percentage_profit:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Target hit for {ticker}. "
                  f"Sold {quantity} shares at ₹{current_price:.2f}. "
                  f"P/L: ₹{profit:.2f} ({percentage_profit:.2f}%)")

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker,
                                 current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 quantity,
                                 total_value_bought,
                                 total_value_sold,
                                 f"{percentage_profit:.2f}%",
                                 trend_type])
            break

        # SINGLE Stop-loss => close full quantity
        elif current_price <= first_stop_loss_price:
            total_value_sold = first_stop_loss_price * quantity
            loss = total_value_bought - total_value_sold
            percentage_loss = (loss / total_value_bought) * 100
            append_text(f"Stop-loss hit for {ticker}! Sold {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                        f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL: Stop-loss hit for {ticker}. "
                  f"Sold {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                  f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker,
                                 current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 quantity,
                                 total_value_bought,
                                 total_value_sold,
                                 f"{-percentage_loss:.2f}%",  # negative
                                 trend_type])
            break

        time.sleep(1)  # simulate time

    print(f"Monitoring complete for {ticker}. Closing console and GUI.")
    root.destroy()

####################################################################
# REWRITTEN monitor_ticker_in_window_bear => ONLY 1 TARGET, 1 STOP
####################################################################
def monitor_ticker_in_window_bear(ticker, sell_price, quantity, total_value_sold, trade_time, kite, instrument_df, interval, trend_type):
    if trend_type.strip().lower() != 'bearish':
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

    # Single target & single stop-loss for full quantity (Bearish side)
    first_target_price = sell_price * (1 - 0.02)   # e.g., 1.5% below sell price
    first_stop_loss_price = sell_price * (1 + 0.015)# 1.5% above sell price

    local_tz = pytz.timezone('Asia/Kolkata')
    start_time = trade_time
    local_tz = ZoneInfo("Asia/Kolkata")
    end_time = dt.datetime(2025, 5, 1, tzinfo=local_tz)

    historical_prices = fetch_historical_price(ticker, kite, instrument_df, start_time, end_time, interval)
    if historical_prices.empty:
        append_text(f"No historical data for {ticker}. Skipping.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: No historical data for {ticker}. Skipping.")
        root.destroy()
        return

    result_file = f"result_papertrade3_{trade_time.strftime('%Y-%m-%d')}.csv"
    if not os.path.isfile(result_file):
        with open(result_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Ticker","date","Shares Bought Back","Total Sold","Total Bought","Percentage Profit/Loss","Trend Type"])

    for _, price_row in historical_prices.iterrows():
        current_time = pd.to_datetime(price_row['date']).tz_convert(local_tz)
        current_price = price_row['close']

        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring {ticker} | "
              f"Current Price: ₹{current_price:.2f} | "
              f"Sell Price: ₹{sell_price:.2f} | "
              f"Target: ₹{first_target_price:.2f} | "
              f"Stop Loss: ₹{first_stop_loss_price:.2f}")

        append_text(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {ticker}, Current: ₹{current_price:.2f}")

        # 3:15 => forced close
        if current_time.hour == 15 and current_time.minute == 15:
            total_value_bought = current_price * quantity
            profit_or_loss = total_value_sold - total_value_bought
            percentage_change = (profit_or_loss / total_value_sold) * 100
            append_text(f"Market close buy-back for {ticker} at ₹{current_price} => P/L: {profit_or_loss} ({percentage_change:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FINAL: Market close for {ticker}. "
                  f"Bought back {quantity} shares at ₹{current_price:.2f}. "
                  f"Profit/Loss: ₹{profit_or_loss:.2f} ({percentage_change:.2f}%)")

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker,
                                 current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 quantity,
                                 total_value_sold,
                                 total_value_bought,
                                 f"{percentage_change:.2f}%",
                                 trend_type])
            break

        # SINGLE target => close entire quantity
        if current_price <= first_target_price:
            total_value_bought = current_price * quantity
            # For a short, profit is 'initial_sold - final_buy'
            profit = total_value_sold - total_value_bought
            percentage_profit = (profit / total_value_sold) * 100
            append_text(f"Target hit for {ticker}! Bought back {quantity} shares at ₹{current_price:.2f}. "
                        f"Profit: ₹{profit:.2f} ({percentage_profit:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Bearish target for {ticker}. "
                  f"Bought back {quantity} at ₹{current_price:.2f}, P/L: ₹{profit:.2f} ({percentage_profit:.2f}%)")

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker,
                                 current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 quantity,
                                 total_value_sold,
                                 total_value_bought,
                                 f"{percentage_profit:.2f}%",
                                 trend_type])
            break

        # SINGLE stop-loss => close entire quantity
        elif current_price >= first_stop_loss_price:
            total_value_bought = first_stop_loss_price * quantity
            loss = total_value_sold - total_value_bought
            percentage_loss = (loss / total_value_sold) * 100
            append_text(f"Stop-loss hit for {ticker}! Bought back {quantity} shares at ₹{first_stop_loss_price:.2f}. "
                        f"Loss: ₹{loss:.2f} ({percentage_loss:.2f}%)")
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL: Stop-loss for {ticker}. "
                  f"Bought back {quantity} at ₹{first_stop_loss_price:.2f}, L: ₹{loss:.2f} ({percentage_loss:.2f}%)")

            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([ticker,
                                 current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                 quantity,
                                 total_value_sold,
                                 total_value_bought,
                                 f"{percentage_loss:.2f}%",
                                 trend_type])
            break

        time.sleep(1)

    print(f"Monitoring completed for {ticker}. Closing console and GUI.")
    root.destroy()

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

        if trade_time.tzinfo:
            trade_time = trade_time.tz_convert(local_tz)
        else:
            trade_time = trade_time.tz_localize(local_tz)

        thread = threading.Thread(
            target=monitor_ticker_in_window,
            args=(ticker, buy_price, quantity, total_value_bought, trade_time, kite, instrument_df, target_percentage, interval, trend_type)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

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

    for _, row in paper_trades.iterrows():
        ticker = row['Ticker']
        sell_price = row['Price']
        quantity = row['Quantity']
        total_value_sold = row['Total value']
        trade_time = pd.to_datetime(row['date'])
        trend_type = row['Trend Type']

        if trade_time.tzinfo is None:
            trade_time = trade_time.tz_localize(local_tz)
        else:
            trade_time = trade_time.tz_convert(local_tz)

        thread = threading.Thread(
            target=monitor_ticker_in_window_bear,
            args=(ticker, sell_price, quantity, total_value_sold, trade_time, kite, instrument_df, interval, trend_type)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def calculate_net_percentage(last_trading_days):
    for trading_day in last_trading_days:
        file_path = f"result_papertrade3_{trading_day}.csv"
        output_file = f"result_final_{trading_day}.csv"
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping {trading_day}.")
            continue

        df['Percentage Profit/Loss'] = df['Percentage Profit/Loss'].str.rstrip('%').astype(float)
        total_percentage = 0
        unique_ticker_count = 0
        results = []

        grouped = df.groupby('Ticker')
        for ticker, group in grouped:
            if len(group) > 1:
                calculated_percentage = (
                    group.iloc[0]['Percentage Profit/Loss']*2 + group.iloc[1]['Percentage Profit/Loss']
                ) / 2
            else:
                calculated_percentage = group.iloc[0]['Percentage Profit/Loss']

            total_percentage += calculated_percentage
            unique_ticker_count += 1
            results.append({"Ticker": ticker, "Net Percentage": calculated_percentage})

        if unique_ticker_count == 0:
            print(f"No valid tickers found in {file_path}.")
            continue

        net_percentage = total_percentage / unique_ticker_count
        results.append({"Ticker": "Overall", "Net Percentage": round(net_percentage, 4)})

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Net Percentage for {trading_day}: {round(net_percentage, 4)}%")
        print(f"Results saved to {output_file}")

def main():
    last_trading_days = [
        
        "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
        "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
        "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
        "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
        "2025-03-31",
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
        file_path = f'papertrade3_{trading_day}.csv'
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Ticker','date','Entry Type','Trend Type','Price','VWAP']
            if not set(required_columns).issubset(df.columns):
                logging.warning(f"Missing required columns in {file_path}. Skipping.")
                continue

            bullish_entries = df[df['Trend Type'].str.strip().str.lower() == 'bullish']
            bearish_entries = df[df['Trend Type'].str.strip().str.lower() == 'bearish']

            if not bullish_entries.empty:
                logging.info(f"Processing {len(bullish_entries)} bullish entries.")
                monitor_paper_trades_backtest(kite, instrument_df, file_path, bullish_entries)

            if not bearish_entries.empty:
                logging.info(f"Processing {len(bearish_entries)} bearish entries.")
                monitor_paper_trades_backtest_bear(kite, instrument_df, file_path, bearish_entries)

        except FileNotFoundError:
            logging.error(f"File {file_path} not found. Skipping {trading_day}.")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    calculate_net_percentage(last_trading_days)

if __name__ == "__main__":
    main()
    logging.shutdown()
