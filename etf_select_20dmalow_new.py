# -*- coding: utf-8 -*-
"""
Zerodha KiteConnect – Bottom 10 ETFs by daily % Change
Automatically fetches yesterday's close and today's live price,
ranks ETFs ascending by percentage change, and outputs the bottom 10.
"""

from kiteconnect import KiteConnect
import logging
import os
import pandas as pd
from datetime import datetime, timedelta, date

# ——— Configuration ———
cwd = r"C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
os.chdir(cwd)

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

# Logging setup
logging.basicConfig(level=logging.INFO)

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
    'SBIETFPB', 'SBIETFIT', 'SBIGOLDETF', 'SBIETFQLTY', 'SBIBPB',
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


# Known market holidays (update each year)
market_holidays = {
    date(2024, 10, 2),  # Gandhi Jayanti
    # add more dates here
}

# Load instrument dumps once
inst_nse = pd.DataFrame(kite.instruments("NSE"))
inst_bse = pd.DataFrame(kite.instruments("BSE"))
instrument_df = pd.concat([inst_nse, inst_bse], ignore_index=True)

LOOKBACK_DAYS = 60  # fetch this many calendar days, enough for 20 trading bars


def is_market_open(d: date) -> bool:
    return d.weekday() < 5 and d not in market_holidays


def last_trading_day() -> date:
    d = date.today() - timedelta(days=1)
    while not is_market_open(d):
        d -= timedelta(days=1)
    return d


def lookup(symbol: str):
    df = instrument_df[instrument_df.tradingsymbol == symbol]
    if df.empty:
        logging.error(f"{symbol} not found in instrument dump")
        return None, None
    row = df.iloc[0]
    return row.instrument_token, row.exchange


def fetch_ohlc(token, start_date: date, end_date: date):
    try:
        data = pd.DataFrame(kite.historical_data(token, start_date, end_date, "day"))
        return data if not data.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching OHLC data: {e}")
        return pd.DataFrame()


def fetch_live(token, exchange, symbol):
    key = f"{exchange}:{symbol}"
    resp = kite.ltp(key)
    return resp.get(key, {}).get("last_price")


def calculate_change(symbol):
    token, exch = lookup(symbol)
    if not token:
        return None

    td = last_trading_day()
    start = td - timedelta(days=LOOKBACK_DAYS)

    df_ohlc = fetch_ohlc(token, start, td)
    df_ohlc = df_ohlc.dropna(subset=["close"]).reset_index(drop=True)

    if len(df_ohlc) < 20:
        logging.warning(f"Insufficient data for {symbol} to compute 20DMA (have {len(df_ohlc)} bars)")
        return None

    df_ohlc["20DMA"] = df_ohlc["close"].rolling(window=20).mean()
    twenty_dma = df_ohlc["20DMA"].iloc[-1]
    if pd.isna(twenty_dma):
        logging.warning(f"20DMA is NaN for {symbol}")
        return None

    live = fetch_live(token, exch, symbol)
    if live is None:
        logging.warning(f"No live price for {symbol}")
        return None

    pct = (live - twenty_dma) / twenty_dma * 100
    return symbol, pct


def main():
    changes = [res for s in shares if (res := calculate_change(s))]

    if not changes:
        logging.error("No % changes calculated — check symbols or data availability.")
        return

    changes.sort(key=lambda x: x[1])  # ascending → most below 20DMA first
    bottom10 = changes[:20]

    df = pd.DataFrame(bottom10, columns=["Symbol", "% Change"])
    df["% Change"] = df["% Change"].map(lambda x: f"{x:.2f}%")
    df.to_csv("etfs_to_buy.csv", index=False)

    print("Bottom 10 ETFs by % Change from 20DMA (look‑back up to 60 days):")
    print(df)


if __name__ == "__main__":
    main()

