# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 18:50:14 2025

@author: Saarit
"""

import os
import glob
import pandas as pd
from filelock import FileLock, Timeout
import logging
import traceback

def reset_calcflag_in_main_indicators(indicators_dir="main_indicators"):
    """
    For every *_main_indicators.csv in 'indicators_dir',
    set 'CalcFlag' = 'No' for all rows, ignoring existing values.
    Uses a FileLock to avoid concurrency issues.
    """

    pattern = os.path.join(indicators_dir, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No main_indicator CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        # Create a .lock path for concurrency control
        lock_path = file_path + ".lock"
        lock = FileLock(lock_path, timeout=10)

        try:
            with lock:
                # Read CSV
                df = pd.read_csv(file_path)
                # Ensure 'CalcFlag' column exists
                if "CalcFlag" not in df.columns:
                    df["CalcFlag"] = ""
                # Set every row's CalcFlag to "No"
                df["CalcFlag"] = "No"

                # Overwrite the file
                df.to_csv(file_path, index=False)
                logging.info(f"Set CalcFlag='No' for all rows in {file_path}.")
                print(f"Set CalcFlag='No' for all rows in {os.path.basename(file_path)}.")

        except Timeout:
            logging.error(f"File lock timeout for {file_path}.")
        except Exception as e:
            logging.error(f"Error resetting CalcFlag in {file_path}: {e}")
            logging.error(traceback.format_exc())



reset_calcflag_in_main_indicators()





import os
import glob
import pandas as pd
from filelock import FileLock, Timeout
import logging
import traceback

def reset_entry_signal_in_main_indicators(indicators_dir="main_indicators"):
    """
    For every *_main_indicators.csv in 'indicators_dir',
    set 'Entry Signal' = 'No' for all rows, ignoring existing values.
    Uses a FileLock to avoid concurrency issues.
    """

    pattern = os.path.join(indicators_dir, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No main_indicator CSV files found in {indicators_dir}.")
        return

    for file_path in files:
        # Create a .lock path for concurrency control
        lock_path = file_path + ".lock"
        lock = FileLock(lock_path, timeout=10)

        try:
            with lock:
                # Read CSV
                df = pd.read_csv(file_path)
                # Ensure 'Entry Signal' column exists
                if "Entry Signal" not in df.columns:
                    df["Entry Signal"] = ""
                # Set every row's Entry Signal to "No"
                df["Entry Signal"] = "No"

                # Overwrite the file
                df.to_csv(file_path, index=False)
                logging.info(f"Set Entry Signal='No' for all rows in {file_path}.")
                print(f"Set Entry Signal='No' for all rows in {os.path.basename(file_path)}.")

        except Timeout:
            logging.error(f"File lock timeout for {file_path}.")
        except Exception as e:
            logging.error(f"Error resetting Entry Signal in {file_path}: {e}")
            logging.error(traceback.format_exc())

reset_entry_signal_in_main_indicators()