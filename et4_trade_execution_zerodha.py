# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:01:37 2024

Author: Saarit
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta, time as datetime_time
from kiteconnect import KiteConnect, KiteTicker
import pytz
import threading
import time
import signal
import sys
import schedule

# ================================
# Setup Logging
# ================================

# Define the correct working directory path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et5\\trading_strategies_algo"
os.chdir(cwd)  # Change the current working directory to 'cwd'

# Set up logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.FileHandler("trading_script.log"),  # Log messages will be written to 'trading_script.log'
        logging.StreamHandler()  # Log messages will also be printed to the console
    ]
)

# Test logging initialization by logging an info message
logging.info("Logging has been initialized successfully.")
print("Logging initialized and script execution started.")  # Print to console

# ================================
# Kite Connect Session Setup
# ================================

def setup_kite_session():
    """
    Initialize Kite Connect session using API key and access token.

    Returns:
        KiteConnect: An instance of KiteConnect with an active session.

    Raises:
        FileNotFoundError: If 'access_token.txt' or 'api_key.txt' is not found.
        Exception: For other errors during session setup.
    """
    try:
        # Read access token from file
        with open("access_token.txt", 'r') as token_file:
            access_token = token_file.read().strip()

        # Read API key from file
        with open("api_key.txt", 'r') as key_file:
            key_secret = key_file.read().split()

        # Initialize KiteConnect with API key
        kite = KiteConnect(api_key=key_secret[0])
        kite.set_access_token(access_token)  # Set access token for the session
        logging.info("Kite session established successfully.")
        print("Kite Connect session initialized successfully.")
        return kite  # Return the KiteConnect instance

    except FileNotFoundError as e:
        # Log and print error if files are not found
        logging.error(f"File not found: {e}")
        print(f"Error: {e}")
        raise  # Re-raise the exception to halt execution
    except Exception as e:
        # Log and print any other exceptions
        logging.error(f"Error setting up Kite session: {e}")
        print(f"Error: {e}")
        raise  # Re-raise the exception to halt execution

# Setup Kite session by initializing the KiteConnect instance
kite = setup_kite_session()

# ================================
# Define the execute_paper_trades Function
# ================================

def execute_paper_trades(last_trading_day, kite, api_semaphore):
    """
    Execute paper trades based on the signals from the CSV file.

    Parameters:
        last_trading_day (str or datetime.date): The last trading day in 'YYYY-MM-DD' format or date object.
        kite (KiteConnect): An authenticated KiteConnect object.
        api_semaphore (threading.Semaphore): Semaphore to limit concurrent API calls.

    Returns:
        None
    """
    # Define the filename
    filename = f"papertrade_{last_trading_day}.csv"

    # Check if the file exists
    if not os.path.exists(filename):
        logging.error(f"File {filename} does not exist.")
        print(f"File {filename} does not exist.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(filename, parse_dates=['Time'])
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        print(f"Error reading {filename}: {e}")
        return

    # Timezone object for IST
    india_tz = pytz.timezone('Asia/Kolkata')

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        try:
            ticker = row['Ticker']
            time_entry = row['Time']  # Assuming this is in IST timezone
            trend_type = row['Trend Type']
            price = float(row['Price'])
            quantity = int(row['Quantity'])

            # Determine the transaction type based on Trend Type
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
                print(f"Unknown Trend Type '{trend_type}' for Ticker '{ticker}'. Skipping.")
                continue  # Skip this row

            # Define additional order parameters
            exchange = 'NSE'  # Modify if trading on a different exchange
            tradingsymbol = ticker  # Ensure this matches Zerodha's tradingsymbol
            product = kite.PRODUCT_MIS
            order_type = kite.ORDER_TYPE_LIMIT
            validity = kite.VALIDITY_DAY
            variety = kite.VARIETY_BRACKET
            tag = 'PaperTrade'

            # Place the bracket order within semaphore to limit concurrent API calls
            with api_semaphore:
                try:
                    order_id = kite.place_order(
                        variety=variety,
                        exchange=exchange,
                        tradingsymbol=tradingsymbol,
                        transaction_type=transaction_type,
                        quantity=quantity,
                        product=product,
                        order_type=order_type,
                        price=price,
                        validity=validity,
                        squareoff=target_price,
                        stoploss=stop_loss_price,
                        trailing_stoploss=0,  # Not used in this case
                        tag=tag
                    )

                    logging.info(f"Placed {transaction_type} order for {ticker} at {price} with Target {target_price} and Stop Loss {stop_loss_price}. Order ID: {order_id}")
                    print(f"Placed {transaction_type} order for {ticker} at {price} with Target {target_price} and Stop Loss {stop_loss_price}. Order ID: {order_id}")

                except Exception as e:
                    logging.error(f"Failed to place order for {ticker}: {e}")
                    print(f"Failed to place order for {ticker}: {e}")

        except Exception as e:
            logging.error(f"Error processing row {index} in {filename}: {e}")
            print(f"Error processing row {index} in {filename}: {e}")
            continue  # Proceed to the next row

# ================================
# Define the main Function
# ================================

def main():
    """
    Main function to execute the trading script.
    - Determines the last trading day.
    - Selects stocks based on RSI criteria.
    - Schedules the entry detection to run every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.
    """
    import pytz
    from datetime import datetime, timedelta, time as datetime_time
    import schedule
    import logging
    import time
    import signal
    import sys
    import threading

    # ================================
    # Setup Signal Handling for Graceful Shutdown
    # ================================

    def signal_handler(sig, frame):
        logging.info("Interrupt received. Shutting down gracefully...")
        print("Interrupt received. Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ================================
    # Timezone Configuration
    # ================================

    india_tz = pytz.timezone('Asia/Kolkata')  # Timezone object for IST
    now = datetime.now(india_tz)  # Current time in IST

    # ================================
    # Define the Start and End Times
    # ================================

    # Define the start and end times with timezone awareness
    start_time_naive = datetime.combine(now.date(), datetime_time(9, 15, 5))
    end_time_naive = datetime.combine(now.date(), datetime_time(15, 0, 5))
    
    start_time = india_tz.localize(start_time_naive)
    end_time = india_tz.localize(end_time_naive)

    # ================================
    # Handle Past Start Times
    # ================================

    if now > end_time:
        logging.warning("Current time is past the trading window. Exiting the script.")
        print("Current time is past the trading window. Exiting the script.")
        return
    elif now > start_time:
        logging.info("Current time is within the trading window. Scheduling remaining jobs.")
    else:
        logging.info("Current time is before the trading window. Waiting to start scheduling.")

    # ================================
    # Generate Scheduled Times at 15-Minute Intervals
    # ================================

    # Generate the list of scheduled times at 15-minute intervals between start_time and end_time
    scheduled_times = []
    current_time = start_time
    while current_time <= end_time:
        scheduled_times.append(current_time.strftime("%H:%M:%S"))
        current_time += timedelta(minutes=15)

    # ================================
    # Determine Last Trading Day
    # ================================

    # Assuming last_trading_day is yesterday if today is a trading day,
    # else the last weekday that was not a holiday.
    # You need to implement 'get_last_trading_day' based on your specific requirements.
    # Here's a simple placeholder implementation:

    def get_last_trading_day():
        """
        Determine the last trading day. This function assumes that trading days are Monday to Friday,
        excluding weekends. You may need to extend this to account for market holidays.
        
        Returns:
            str: Last trading day in 'YYYY-MM-DD' format.
        """
        last_day = datetime.now(india_tz).date() - timedelta(days=0)
        # Loop back if the last_day is weekend or a market holiday
        while last_day.weekday() >= 5 or last_day in market_holidays:
            last_day -= timedelta(days=1)
        return last_day.strftime('%Y-%m-%d')

    # Define market holidays (add more dates as needed)
    market_holidays = [
        datetime(2024, 1, 26).date(),  # Republic Day
        datetime(2024, 3, 8).date(),   # Mahashivratri
        datetime(2024, 3, 25).date(),  # Holi
        datetime(2024, 3, 29).date(),  # Good Friday
        datetime(2024, 4, 11).date(),  # Id-Ul-Fitr (Ramadan Eid)
        datetime(2024, 4, 17).date(),  # Shri Ram Navami
        datetime(2024, 5, 1).date(),   # Maharashtra Day
        datetime(2024, 6, 17).date(),  # Bakri Id
        datetime(2024, 7, 17).date(),  # Moharram
        datetime(2024, 8, 15).date(),  # Independence Day
        datetime(2024, 10, 2).date(),  # Gandhi Jayanti
        datetime(2024, 11, 1).date(),  # Diwali Laxmi Pujan
        datetime(2024, 11, 15).date(), # Gurunanak Jayanti
        datetime(2024, 11, 20).date(), # Maha elections
        datetime(2024, 12, 25).date(), # Christmas
    ]

    last_trading_day = get_last_trading_day()  # Determine the last trading day

    logging.info(f"Last trading day determined as: {last_trading_day}")
    print(f"Last trading day determined as: {last_trading_day}")

    # ================================
    # Initialize Semaphore
    # ================================

    # Limit to 2 concurrent API calls to prevent rate limit issues
    api_semaphore = threading.Semaphore(2)

    # ================================
    # Schedule execute_paper_trades at Each Scheduled Time
    # ================================

    for scheduled_time in scheduled_times:
        # Convert scheduled_time string back to datetime object for comparison
        scheduled_datetime_naive = datetime.strptime(scheduled_time, "%H:%M:%S")
        scheduled_datetime = india_tz.localize(datetime.combine(now.date(), scheduled_datetime_naive.time()))
        
        if scheduled_datetime > now:
            schedule.every().day.at(scheduled_time).do(execute_paper_trades, last_trading_day, kite, api_semaphore)
            logging.info(f"Scheduled job at {scheduled_time} IST for last_trading_day: {last_trading_day}")
        else:
            logging.debug(f"Skipped scheduling job at {scheduled_time} IST as it's already past.")

    logging.info("Scheduled `execute_paper_trades` to run every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")
    print("Scheduled `execute_paper_trades` to run every 15 minutes starting at 9:15:05 AM IST up to 3:00:05 PM IST.")

    # ================================
    # Optional: Execute the Job Immediately if Within Trading Window
    # ================================

    if start_time <= now <= end_time:
        logging.info("Executing initial run of trading strategy.")
        print("Executing initial run of trading strategy.")
        execute_paper_trades(last_trading_day, kite, api_semaphore)

    # ================================
    # Keep the Script Running to Allow Scheduled Jobs to Execute
    # ================================

    while True:
        schedule.run_pending()  # Run any pending scheduled jobs
        time.sleep(1)  # Sleep for a short time to prevent high CPU usage

# ================================
# Execute the main Function
# ================================

if __name__ == "__main__":
    main()  # Execute the main function when the script is run
