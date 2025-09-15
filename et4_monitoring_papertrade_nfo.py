# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:01:27 2024

@author: Saarit
"""

import datetime as dt
from kiteconnect import KiteConnect
import pandas as pd
import logging
import os
from datetime import timedelta

# Define the working directory and set it
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_script.log"),
        logging.StreamHandler(),
    ]
)

logging.info("Logging initialized and script execution started.")

# Define market holidays for 2024
MARKET_HOLIDAYS = [
    dt.date(2024, 1, 26), dt.date(2024, 3, 8), dt.date(2024, 3, 25),
    dt.date(2024, 3, 29), dt.date(2024, 4, 11), dt.date(2024, 4, 17),
    dt.date(2024, 5, 1), dt.date(2024, 6, 17), dt.date(2024, 7, 17),
    dt.date(2024, 8, 15), dt.date(2024, 10, 2), dt.date(2024, 11, 1),
    dt.date(2024, 11, 15), dt.date(2024, 11, 20), dt.date(2024, 12, 25),
]

def get_last_trading_day(reference_date=None):
    """
    Determine the last valid trading day, avoiding weekends and holidays.
    """
    reference_date = reference_date or (dt.date.today() - timedelta(days=1))
    logging.info(f"Determining last trading day from reference date: {reference_date}")

    while reference_date.weekday() >= 5 or reference_date in MARKET_HOLIDAYS:
        logging.debug(f"{reference_date} is a holiday or weekend.")
        reference_date -= timedelta(days=1)

    logging.info(f"Last trading day determined as: {reference_date}")
    return reference_date

def setup_kite_session():
    """
    Initialize Kite Connect session using stored credentials.
    """
    try:
        with open("access_token.txt", "r") as token_file:
            access_token = token_file.read().strip()

        with open("api_key.txt", "r") as key_file:
            api_key = key_file.read().strip()

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logging.info("Kite session established successfully.")
        return kite
    except FileNotFoundError as e:
        logging.error(f"Required credentials file not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error initializing Kite session: {e}")
        raise

kite = setup_kite_session()

def load_papertrade_data(filepath):
    """
    Load papertrade.csv and validate the data.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded papertrade.csv with {len(df)} rows.")
        logging.debug(f"First few rows of papertrade.csv:\n{df.head()}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading papertrade.csv: {e}")
        raise

papertrade_path = os.path.join(cwd, "papertrade.csv")
papertrade_df = load_papertrade_data(papertrade_path)

def fetch_atm_strike_price(current_price):
    """
    Calculate the ATM strike price rounded to the nearest 50.
    """
    strike_price = round(current_price / 50) * 50
    logging.debug(f"ATM Strike Price for LTP {current_price}: {strike_price}")
    return strike_price

def fetch_nearest_option_contract(ticker, atm_strike, expiry_type="weekly"):
    """
    Fetch the nearest options contract based on the expiry type.
    """
    today = dt.date.today()

    if expiry_type == "weekly":
        expiry_date = today + timedelta((3 - today.weekday()) % 7)  # Nearest Thursday
    elif expiry_type == "monthly":
        next_month = today.replace(day=28) + timedelta(days=4)
        expiry_date = next_month - timedelta(days=next_month.weekday() + 3)
    else:
        logging.error("Invalid expiry type specified.")
        raise ValueError("Invalid expiry type specified.")

    expiry_format = expiry_date.strftime('%d%b%Y').upper()
    logging.debug(f"Nearest expiry for {ticker} (ATM {atm_strike}): {expiry_format}")
    return f"{ticker}{expiry_format}CE", f"{ticker}{expiry_format}PE"

def simulate_trade(row, kite):
    """
    Simulate a trade based on a row from the papertrade.csv.
    """
    try:
        ticker = row["Ticker"]
        ltp_response = kite.ltp(f"NSE:{ticker}")
        logging.debug(f"LTP Response for {ticker}: {ltp_response}")
        ltp = ltp_response[f"NSE:{ticker}"]["last_price"]

        atm_strike = fetch_atm_strike_price(ltp)
        ce_contract, pe_contract = fetch_nearest_option_contract(ticker, atm_strike)

        ce_ltp_response = kite.ltp(f"NFO:{ce_contract}")
        pe_ltp_response = kite.ltp(f"NFO:{pe_contract}")
        logging.debug(f"CE LTP Response: {ce_ltp_response}, PE LTP Response: {pe_ltp_response}")

        ce_ltp = ce_ltp_response[f"NFO:{ce_contract}"]["last_price"]
        pe_ltp = pe_ltp_response[f"NFO:{pe_contract}"]["last_price"]

        ce_qty = int(20000 // ce_ltp)
        pe_qty = int(20000 // pe_ltp)

        trade_details = {
            "Ticker": ticker,
            "CE_Contract": ce_contract,
            "PE_Contract": pe_contract,
            "CE_LTP": ce_ltp,
            "PE_LTP": pe_ltp,
            "CE_Qty": ce_qty,
            "PE_Qty": pe_qty,
            "CE_SL": ce_ltp * 0.98,
            "PE_SL": pe_ltp * 0.98,
            "CE_Target": ce_ltp * 1.04,
            "PE_Target": pe_ltp * 1.04,
        }
        logging.info(f"Simulated trade for {ticker}: {trade_details}")
        return trade_details
    except Exception as e:
        logging.error(f"Error simulating trade for {row['Ticker']}: {e}")
        return None

# Process each row in papertrade_df
trade_results = []
for _, row in papertrade_df.iterrows():
    trade = simulate_trade(row, kite)
    if trade:
        trade_results.append(trade)

# Save trade results
if trade_results:
    results_df = pd.DataFrame(trade_results)
    results_path = os.path.join(cwd, "papertrade_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"Trade results saved to {results_path}")
else:
    logging.warning("No trade results to save. Please check the input data.")
