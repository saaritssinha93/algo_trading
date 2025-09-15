# -*- coding: utf-8 -*-
"""
Zerodha kiteconnect automated authentication
"""

from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the correct path
cwd = r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo"
os.chdir(cwd)

# List of ETFs
shares = [
    # NSE-listed ETFs
    'ABGSEC', 'ABSLBANETF', 'ABSLNN50ET', 'ABSLPSE', 'AUTOIETF', 'AXISBPSETF',
    'AXISCETF', 'AXISBNKETF', 'AXISGOLD', 'AXISILVER', 'AXISTECETF', 'AXISNIFTY',
    'BANKBEES', 'BANKBETF', 'BANKETF', 'BANKETFADD', 'BBETF0432', 'BBNPPGOLD',
    'BBNPNBETF', 'BFSI', 'BSE500IETF', 'BSLGOLDETF', 'BSLNIFTY', 'BSLSENETFG',
    'COMMOIETF', 'CONS', 'CONSUMBEES', 'CONSUMIETF', 'CPSEETF', 'DIVOPPBEES',
    'EQUAL50ADD', 'EBANKNIFTY', 'EBBETF0425', 'EBBETF0430', 'EBBETF0431',
    'EBBETF0432', 'EBBETF0433', 'EGOLD', 'FINIETF', 'GILT5YBEES', 'GSEC10ABSL',
    'GSEC10IETF', 'GSEC10YEAR', 'GSEC5IETF', 'GOLD1', 'GOLDBEES', 'GOLDCASE',
    'GOLDETF', 'GOLDETFADD', 'GOLDIETF', 'GOLDSHARE', 'HDFCLOWVOL', 'HDFCMID150',
    'HDFCGOLD', 'HDFCGROWTH', 'HDFCLIQUID', 'HDFCNIF100', 'HDFCNIFBAN', 'HDFCNIFTY',
    'HDFCSENSEX', 'HDFCSILVER', 'HDFCSML250', 'HDFCVALUE', 'HEALTHADD',
    'HEALTHIETF', 'HEALTHY', 'INFRAIETF', 'IT', 'ITBEES', 'ITIETF', 'ITETF',
    'ITETFADD', 'IVZINNIFTY', 'IVZINGOLD', 'JUNIORBEES', 'LICMFGOLD', 'LICNETFN50',
    'LICNETFGSC', 'LICNETFSEN', 'LICNMID100', 'LIQUIDADD', 'LIQUIDBEES',
    'LIQUIDCASE', 'LIQUIDETF', 'LIQUIDIETF', 'LIQUIDSHRI', 'LIQUIDSBI', 'LIQUID',
    'LIQUIDBETF', 'LIQUID1', 'LOWVOL', 'LOWVOL1', 'LOWVOLIETF', 'MAFANG',
    'MAKEINDIA', 'MASPTOP50', 'MID150BEES', 'MID150CASE', 'MIDCAP', 'MIDCAPETF',
    'MIDCAPIETF', 'MIDQ50ADD', 'MIDSMALL', 'MIDSELIETF', 'MOM100', 'MOM30IETF',
    'MOM50', 'MOMOMENTUM', 'MOVALUE', 'MULTICAP', 'NIF100BEES', 'NIF100IETF',
    'NIF5GETF', 'NIF10GETF', 'NIFITETF', 'NIFTY1', 'NIFTY50ADD', 'NIFTYBEES',
    'NIFTYBETF', 'NIFTYETF', 'NIFTYIETF', 'NIFTYQLITY', 'PHARMABEES', 
    'PVTBANKADD', 'PVTBANIETF', 'PSUBANK', 'PSUBANKADD', 'PSUBNKIETF', 'PSUBNKBEES', 
    'QGOLDHALF', 'QNIFTY', 'QUAL30IETF', 'RELIANCEETF', 'SBIETF50', 'SBIETFCON',
    'SBIETFPB', 'SBIETFIT', 'SBIGOLDETF',
    'SBISILVER', 'SBINEQWETF', 'SDL26BEES', 'SETF10GILT', 'SETFNN50', 'SETFNIF50',
    'SETFNIFBK', 'SETFGOLD', 'SENSEXADD', 'SENSEXIETF', 'SENSEXETF', 'SHARIABEES',
    'SILVER', 'SILVER1', 'SILVERADD', 'SILVERBEES', 'SILVERETF', 'SILVERIETF', 
    'SILVRETF', 'SILVER', 'SILVERIETF', 'TATAGOLD', 'TATSILV', 
    'TOP100CASE', 'UTIBANKETF', 'UTINIFTETF', 'UTINEXT50', 'UTISENSETF', 'UTISXN50',
    # BSE-listed ETFs
    'LIQUIDBEES', 'NIFTYBEES', 'GOLDBEES', 'BANKBEES', 'HNGSNGBEES', 
    'JUNIORBEES', 'CPSE ETF', 'LIQUIDETF', 'NIFTYIETF', 'PSUBNKBEES', 
    'HDFCGOLD', 'SETFGOLD', 'SBISENSEX', 'GOLDIETF', 'ICICIB22', 
    'UTINIFTETF', 'GOLD1', 'MON100', 'NIFTY1', 'UTI GOLD ETF', 
    'SENSEXIETF', 'BSLGOLDETF', 'MOM100', 'LICMFGOLD', 'PSUBANK', 
    'SETFSN50', 'NIF100IETF', 'NIF100BEES', 'HDFCSENSEX', 
    'UTISENSETF', 'MOM50', 'AXISGOLD', 'SHARIABEES', 'SETFBSE100', 
    'QUANTUM GOLD', 'SENSEX1', 'QNIFTY', 'LICNETFSEN'
]


# Known market holidays for 2024 (example, you can update this list)
market_holidays = [
    dt.date(2024, 10, 2),  # Gandhi Jayanti
    # Add other known holidays for the year here
]

# Generate trading session
try:
    with open("access_token.txt", 'r') as token_file:
        access_token = token_file.read().strip()

    with open("api_key.txt", 'r') as key_file:
        key_secret = key_file.read().split()

    kite = KiteConnect(api_key=key_secret[0])
    kite.set_access_token(access_token)
    logging.info("Kite session established successfully.")

except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    raise
except Exception as e:
    logging.error(f"Error setting up Kite session: {e}")
    raise

# Get dump of all NSE instruments
try:
    instrument_dump_nse = kite.instruments("NSE")
    instrument_df_nse = pd.DataFrame(instrument_dump_nse)
    logging.info("NSE instrument data fetched successfully.")
    
except Exception as e:
    logging.error(f"Error fetching NSE instruments: {e}")
    raise

# Get dump of all BSE instruments
try:
    instrument_dump_bse = kite.instruments("BSE")
    instrument_df_bse = pd.DataFrame(instrument_dump_bse)
    logging.info("BSE instrument data fetched successfully.")

except Exception as e:
    logging.error(f"Error fetching BSE instruments: {e}")
    raise

# Combine both instrument DataFrames if needed
instrument_df = pd.concat([instrument_df_nse, instrument_df_bse], ignore_index=True)

# Update the instrument lookup function
def instrument_lookup(instrument_df, symbol):
    """Looks up instrument token for a given symbol in the instrument dump."""
    try:
        instrument_token = instrument_df[instrument_df.tradingsymbol == symbol].instrument_token.values[0]
        logging.info(f"Instrument token for {symbol}: {instrument_token}")
        return instrument_token
    except IndexError:
        logging.error(f"Symbol {symbol} not found in instrument dump.")
        return -1
    except Exception as e:
        logging.error(f"Error in instrument lookup: {e}")
        return -1

# Function to check if a day is a market open day (skips weekends and holidays)
def is_market_open(day):
    """Returns True if the day is a market open day (i.e., not a weekend or a holiday)."""
    return day.weekday() < 5 and day not in market_holidays  # Monday=0, ..., Friday=4 (market open), skip weekends and holidays

# Function to find the most recent trading day
def get_last_trading_day():
    """Returns the most recent trading day (skips weekends and holidays)."""
    today = dt.date.today()
    day = today - dt.timedelta(days=1)  # Start with yesterday
    
    # Keep going back until we find a market open day
    while not is_market_open(day):
        day -= dt.timedelta(days=1)
    
    return day

# Function to fetch OHLC data for a specific date range
def fetch_ohlc(ticker, interval, start_date, end_date):
    """Extracts historical OHLC data for a specific date range and returns it as a DataFrame."""
    try:
        instrument = instrument_lookup(instrument_df, ticker)
        if instrument == -1:
            raise ValueError(f"Instrument lookup failed for ticker {ticker}")

        # Fetch historical data for the given date range
        data = pd.DataFrame(
            kite.historical_data(
                instrument, 
                start_date, 
                end_date, 
                interval
            )
        )

        # Print full response for debugging
        logging.info(f"Raw API response for {ticker}: {kite.historical_data(instrument, start_date, end_date, interval)}")

        if not data.empty:
            logging.info(f"Fetched data columns for {ticker}: {data.columns}")
            data.set_index("date", inplace=True)
            return data
        else:
            logging.error(f"No data returned for {ticker}")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Error fetching OHLC data for {ticker}: {e}")
        return None

# Function to fetch live price
def fetch_live_price(symbol):
    """Fetches the current live price of the given stock."""
    try:
        instrument = instrument_lookup(instrument_df, symbol)
        live_price_data = kite.ltp(f"NSE:{symbol}")
        return live_price_data[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        logging.error(f"Error fetching live price for {symbol}: {e}")
        return None

# Global dictionary to store stocks and their percentage changes
high_growth_stocks = {}

# Modify the print_price_comparison function to store percentage change
def print_price_comparison(symbol):
    """Prints last market day closing price, current price, and percentage change."""    
    # Find the last trading day
    last_trading_day = get_last_trading_day()
    
    # Set date range (last trading day to today)
    start_date = last_trading_day
    end_date = last_trading_day
    
    logging.info(f"Fetching data for {symbol} on last trading day: {last_trading_day}")
    
    # Fetch OHLC data for last trading day and today's live price
    ohlc_data = fetch_ohlc(symbol, "day", start_date, end_date)
    
    if ohlc_data is not None and not ohlc_data.empty:
        if 'close' in ohlc_data.columns:
            last_close = ohlc_data['close'].iloc[-1]
            live_price = fetch_live_price(symbol)
            
            if live_price is not None:
                percent_change = ((live_price - last_close) / last_close) * 100
                
                logging.info(f"{symbol}:")
                logging.info(f"Last Trading Day Close: {last_close}")
                logging.info(f"Current Price: {live_price}")
                logging.info(f"Percentage Change: {percent_change:.2f}%")
                
                # Check if the percentage change is 2% or more
                if percent_change >= 2:
                    logging.info(f"{symbol} has increased by {percent_change:.2f}%, which is above 2%.")
                    high_growth_stocks[symbol] = percent_change  # Store percentage change
                    print(f"{symbol}: Last Close: {last_close}, Current Price: {live_price}, Percentage Change: {percent_change:.2f}% (Above 2% growth)")
                else:
                    print(f"{symbol}: Last Close: {last_close}, Current Price: {live_price}, Percentage Change: {percent_change:.2f}%")

# Function to calculate metrics for each ETF
def calculate_etf_metrics(symbol):
    """Calculates and prints metrics for a given ETF."""
    last_trading_day = get_last_trading_day()
    start_date = last_trading_day - dt.timedelta(days=30)  # 30 days back for data
    end_date = last_trading_day

    # Fetch OHLC data for the last 30 days
    ohlc_data = fetch_ohlc(symbol, "day", start_date, end_date)

    if ohlc_data is not None and not ohlc_data.empty:
        # Calculate CMP
        cmp = fetch_live_price(symbol)

        # Ensure we have enough data for 20DMA calculation
        if 'close' in ohlc_data.columns and len(ohlc_data) >= 20:
            ohlc_data['20DMA'] = ohlc_data['close'].rolling(window=20).mean()
            twenty_dma = ohlc_data['20DMA'].iloc[-1]  # Latest 20 DMA
            
            if cmp is not None:
                diff = cmp - twenty_dma
                percent_change = ((cmp - twenty_dma) / twenty_dma) * 100 if twenty_dma != 0 else None
                
                logging.info(f"{symbol}: CMP: {cmp}, 20DMA: {twenty_dma}, CMP - 20DMA: {diff}, % Change: {percent_change:.2f}%")
                return (symbol, diff, percent_change)  # Return values for ranking

    return (symbol, None, None)  # In case of issues, return None

def run_etf_analysis_every_30mins(shares):
    """Run the ETF analysis every 30 minutes starting at 9:15 AM or immediately if after 9:15 AM."""

    # Define the first run time at 9:15 AM
    now = datetime.now()
    first_run_time = now.replace(hour=9, minute=15, second=0, microsecond=0)

    # If the current time is past 9:15 AM today, start immediately
    if now > first_run_time:
        print("It's past 9:15 AM, starting immediately...")
        first_run_time = now  # Start right away
    else:
        # Wait until 9:15 AM to start
        wait_time = (first_run_time - now).total_seconds()
        print(f"Waiting until {first_run_time} to start...")
        time.sleep(wait_time)

    # Start the loop to run every 30 minutes
    while True:
        etf_metrics = []
        
        # Perform the ETF analysis
        for share in shares:
            metrics = calculate_etf_metrics(share)
            if metrics and metrics[2] is not None:  # Ensure percent_change is not None
                etf_metrics.append(metrics)

        # Rank the ETFs based on % Change in ascending order
        ranked_etfs = sorted(etf_metrics, key=lambda x: x[2] if x[2] is not None else float('inf'))
        bottom_ten_etfs = ranked_etfs[:10]  # Bottom 10 based on % Change

        # Prepare data for CSV
        ranked_data = [{"Rank": rank, "Symbol": symbol, "% Change": f"{percent_change:.2f}%"} 
                       for rank, (symbol, diff, percent_change) in enumerate(bottom_ten_etfs, start=1)]

        # Create a DataFrame and save it to CSV
        ranked_df = pd.DataFrame(ranked_data)
        ranked_df.to_csv("etfs_to_buy.csv", index=False)

        # Print the ranking list
        print("Bottom 10 ETFs based on % Change from 20DMA:")
        print(ranked_df)

        # Wait for 30 minutes before the next iteration
        print(f"Next run at {datetime.now() + timedelta(minutes=30)}")
        time.sleep(1800)  # 1800 seconds = 30 minutes
        
        
        
# Example usage
if __name__ == "__main__":
            run_etf_analysis_every_30mins(shares)