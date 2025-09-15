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
# Holidays
market_holidays = {date(2024, 10, 2)}

# Instrument dump (cached once per run)
instrument_df = pd.concat([
    pd.DataFrame(kite.instruments("NSE")),
    pd.DataFrame(kite.instruments("BSE"))
], ignore_index=True)

LOOKBACK_DAYS = 100         # enough calendar days to ensure ≥50 trading bars
DMA_WINDOW     = 50         # use 50‑day moving average

# ── Helpers ────────────────────────────────────────────────────────────────

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
        logging.warning(f"{symbol}: not found in instrument dump")
        return None, None
    r = df.iloc[0]
    return r.instrument_token, r.exchange


def fetch_ohlc(token: int, start: date, end: date) -> pd.DataFrame:
    try:
        return pd.DataFrame(kite.historical_data(token, start, end, "day"))
    except Exception as e:
        logging.warning(f"token {token}: {e}")
        return pd.DataFrame()


def live_price(token: int, exch: str, sym: str):
    key = f"{exch}:{sym}"
    try:
        return kite.ltp(key)[key]["last_price"]
    except Exception:
        return None

# ── Core calculation ──────────────────────────────────────────────────────

def pct_vs_dma(symbol: str):
    token, exch = lookup(symbol)
    if not token:
        return None

    end = last_trading_day()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = fetch_ohlc(token, start, end)
    df = df.dropna(subset=["close"])  # guard against missing rows

    if len(df) < DMA_WINDOW:
        logging.info(f"{symbol}: only {len(df)} bars – need {DMA_WINDOW}")
        return None

    dma = df["close"].rolling(DMA_WINDOW).mean().iloc[-1]
    if pd.isna(dma):
        return None

    price = live_price(token, exch, symbol)
    if price is None:
        return None

    pct = (price - dma) / dma * 100
    return symbol, pct

# ── Driver ────────────────────────────────────────────────────────────────

def main():
    results = [res for s in shares if (res := pct_vs_dma(s))]
    if not results:
        logging.error("No results – check data/API")
        return

    results.sort(key=lambda x: x[1])  # ascending: most below 50‑DMA first
    bottom10 = results[:30]

    df = pd.DataFrame(bottom10, columns=["Symbol", "% Change vs 50DMA"])
    df["% Change vs 50DMA"] = df["% Change vs 50DMA"].map(lambda x: f"{x:.2f}%")
    df.to_csv("etfs_to_buy.csv", index=False)

    print("Bottom 30 ETFs by % change vs 50‑DMA:")
    print(df)


if __name__ == "__main__":
    main()
