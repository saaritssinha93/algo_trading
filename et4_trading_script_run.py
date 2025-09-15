# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:04:17 2024

Author: Saarit
"""

import subprocess
import time
from datetime import datetime, time as datetime_time
import pytz
import sys
import logging
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from tqdm import tqdm
import glob
import os

# ================================
# Configuration
# ================================

# Define the timezone
india_tz = pytz.timezone('Asia/Kolkata')

# Paths to your scripts
TRADING_SCRIPT_PATH = 'C:/Users/Saarit/OneDrive/Desktop/Trading/et4/trading_strategies_algo/et4_bullish_bearish_trading_startegy_15min_optimised_main.py'

# Log files for each script
TRADING_LOG_FILE = 'trading_script.log'
RUN_LOG_FILE = 'et4_run.log'

# ================================
# Logging Setup
# ================================

# Setup main logger
logger = logging.getLogger('et4_run_logger')
logger.setLevel(logging.INFO)

# Prevent adding multiple handlers if they already exist
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Rotating File Handler
    file_handler = RotatingFileHandler(RUN_LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Override logging
logging = logger

# ================================
# Function Definitions
# ================================

def run_trading_script():
    """
    Executes the trading strategy script every 15 minutes.
    Logs output and errors to TRADING_LOG_FILE.
    """
    try:
        with open(TRADING_LOG_FILE, 'a') as trading_log:
            process = subprocess.run(
                ['python', TRADING_SCRIPT_PATH],
                stdout=trading_log,
                stderr=trading_log,
                check=True
            )
        logging.info(f"Successfully ran trading script at {datetime.now(india_tz).strftime('%Y-%m-%d %H:%M:%S')} IST.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running trading script: {e}")
    except Exception as e:
        logging.error(f"Unexpected error running trading script: {e}")

from datetime import timedelta

# ================================
# Helper Functions
# ================================

# Define Market Holidays for 2024 - These are days no trading occurs.
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

def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day:
    - If today is a trading day, return today
    - Else, go back until we find the last valid trading day.
    """
    if reference_date is None:
        reference_date = datetime.now(india_tz).date()
    logging.info(f"Checking last trading day from {reference_date}.")
    print(f"Reference Date: {reference_date}")
    if reference_date.weekday() < 5 and reference_date not in market_holidays:
        last_trading_day = reference_date
    else:
        last_trading_day = reference_date - timedelta(days=1)
        while last_trading_day.weekday() >= 5 or last_trading_day in market_holidays:
            last_trading_day -= timedelta(days=1)
    logging.info(f"Last trading day: {last_trading_day}")
    print(f"Last Trading Day: {last_trading_day}")
    return last_trading_day


# ================================
# Main Execution Flow
# ================================

def main():
    """
    Orchestrates the scheduling and execution of the trading script.
    """
    last_trading_day = get_last_trading_day()
    today_str = last_trading_day.strftime('%Y-%m-%d')

    logging.info(f"Starting main process for trading day: {last_trading_day}")
    print(f"Starting main process for trading day: {last_trading_day}")

    # Define start and end times
    start_time_naive = datetime.combine(last_trading_day, datetime_time(9, 30, 15))
    end_time_naive = datetime.combine(last_trading_day, datetime_time(23, 30, 15))
    start_time = india_tz.localize(start_time_naive)
    end_time = india_tz.localize(end_time_naive)

    now = datetime.now(india_tz)
    if now > end_time:
        logging.warning("Current time past trading window, exiting.")
        print("Past trading window, exiting.")
        return
    elif now >= start_time:
        logging.info("Current time within trading window.")
    else:
        logging.info("Current time before trading window.")

    # Initialize the scheduler
    scheduler = BackgroundScheduler(timezone=india_tz)
    scheduler.start()

    # Define CronTrigger to run every 15 minutes at second=15 between 9 and 15 hours
    trading_trigger = CronTrigger(
        minute='0,15,30,45',
        second='15',
        hour='9-15',
        timezone=india_tz,
        start_date=start_time,
        end_date=end_time
    )

    # Schedule the trading script
    scheduler.add_job(run_trading_script, trigger=trading_trigger)
    logging.info(f"Scheduled trading script every 15 minutes starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')} IST.")
    print(f"Scheduled trading script every 15 minutes starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')} IST.")

    try:
        # Run until end_time
        while True:
            now = datetime.now(india_tz)
            if now >= end_time:
                logging.info(f"Reached end time {end_time.strftime('%Y-%m-%d %H:%M:%S')} IST. Stopping scripts.")
                print(f"Reached end time {end_time.strftime('%Y-%m-%d %H:%M:%S')} IST. Stopping scripts.")
                break
            time.sleep(1)  # Sleep to reduce CPU usage
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down.")
        print("Interrupted by user. Shutting down.")
    finally:
        # Shutdown scheduler
        scheduler.shutdown(wait=False)
        logging.info("Scheduler shut down.")
        print("Scheduler shut down.")

        # Since we removed the signals_script, no need to terminate any subprocesses
        logging.info("All tasks have been stopped.")
        print("All tasks have been stopped.")


if __name__ == "__main__":
    main()
