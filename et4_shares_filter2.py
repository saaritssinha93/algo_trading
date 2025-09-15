# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:22:42 2024

@author: Saarit
"""

# ================================
# Import Necessary Modules
# ================================

import os
import yfinance as yf

# Import the shares list from et4_filtered_stocks.py
from et4_filtered_stocks import shares  # Adjust the import path as needed

# Define the correct working directory path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)  # Change the current working directory to 'cwd'

# Function to fetch and filter market caps using yfinance
def filter_market_caps(shares, min_cap=500):
    """
    Fetches and filters shares with a market cap greater than the specified minimum,
    and prints the ticker along with its market cap to the console.

    Args:
        shares (list): List of stock tickers (e.g., ['RELIANCE.NS', 'TCS.NS']).
        min_cap (int): Minimum market cap in crores to include in the filtered list.

    Returns:
        list: A list of shares that meet the market cap criteria.
    """
    filtered_shares = []
    for share in shares:
        try:
            # Create a ticker object
            stock = yf.Ticker(share)

            # Fetch info which includes market cap among other things
            info = stock.info
            market_cap = info.get('marketCap')

            # Convert from INR to crores
            market_cap_crores = market_cap / 1e7 if market_cap else None

            # Check if market cap meets the specified minimum
            if market_cap_crores and market_cap_crores > min_cap:
                filtered_shares.append(share[:-3])  # Remove the '.NS' for cleaner output
                print(f"{share[:-3]}: Market Cap = ₹{market_cap_crores:.2f} crores")
            else:
                print(f"{share[:-3]}: Market Cap data not available or below threshold")
        except Exception as e:
            print(f"Error retrieving data for {share}: {e}")

    return filtered_shares

# Convert stock symbols to Yahoo Finance format if necessary
formatted_shares = [share + '.NS' if not share.endswith('.NS') else share for share in shares]

# Filter shares based on market cap
filtered_shares = filter_market_caps(formatted_shares, 500)

# Save the filtered shares to a new Python file
filtered_shares_content = "shares = [\n    '" + "',\n    '".join(filtered_shares) + "'\n]"

file_path = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo\\et4_filtered_stocks_market_cap.py"
with open(file_path, "w") as file:
    file.write(filtered_shares_content)

print(f"Filtered shares saved to {file_path}")
