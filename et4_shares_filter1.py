# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:22:42 2024

@author: Saarit
"""

# ================================
# Import Necessary Modules
# ================================

import os
import logging
import time
import threading
from datetime import datetime, timedelta, time as datetime_time

import pandas as pd
import numpy as np
import pytz
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect

# Import the shares list from et1_select_stocklist.py
from et1_stock_tickers_test import shares  # 'shares' should be a list of stock tickers

# Define the correct working directory path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)  # Change the current working directory to 'cwd'


# Filtering out based on categories to be removed
categories_to_remove = [
    'indices', 'etfs', 'bonds and debts', 'small and microcap', 'mutual funds', 
    'penny or illiquid stocks', 'international assets', 
    'reits and infrastructure funds', 'derivatives or leveraged instruments'
]

# Keywords indicating categories to remove
keywords = [
    'NIFTY', 'ETF', 'BEES', 'DEBT', 'SM', 'ST', 'BE', 'RR', 'IV', 'BZ', 'GLOBAL', 'LIQUID',
    'ADD', 'INVIT', 'GOLD', 'SILVER', 'MF', 'FUND', 'ASM', 'GSM'
]

# Filter shares
filtered_shares = [
    share for share in shares
    if not any(keyword in share for keyword in keywords)
]

filtered_shares

# Save the filtered shares to a new Python file
filtered_shares_content = "shares = [\n"
filtered_shares_content += ", ".join(f"'{share}'" for share in filtered_shares)
filtered_shares_content += "\n]"

# Define the file path
file_path = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo\\et4_filtered_stocks.py"

# Save the content to the file
with open(file_path, "w") as file:
    file.write(filtered_shares_content)

file_path