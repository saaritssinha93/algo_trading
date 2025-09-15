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


# ──────────────────────────────────────────────────────────────
#  Calendar helpers
# ──────────────────────────────────────────────────────────────
market_holidays = {date(2024, 10, 2)}      # update yearly

def is_market_open(d: date) -> bool:
    return d.weekday() < 5 and d not in market_holidays

def last_trading_day() -> date:
    d = date.today()
    while not is_market_open(d):
        d -= timedelta(days=1)
    return d

# ──────────────────────────────────────────────────────────────
#  Exchange / token resolution & live price
# ──────────────────────────────────────────────────────────────
def lookup(symbol: str):
    """Return (instrument_token, exchange) preferring NSE ETFs."""
    df = instrument_df[instrument_df.tradingsymbol.eq(symbol)]
    if df.empty:
        logging.warning("%s not in instrument dump", symbol)
        return None, None
    df = df.copy()
    df["rank"] = df.exchange.eq("NSE").astype(int) + df.instrument_type.eq("ETF").astype(int)
    row = df.sort_values(["rank", "expiry"], ascending=False).iloc[0]
    return int(row.instrument_token), row.exchange

def fetch_live(token: int, exch: str, symbol: str):
    """Best-effort last traded price."""
    try:
        key = str(token)
        data = kite.ltp([key])
        if key in data:
            return data[key]["last_price"]
    except Exception:
        pass
    try:
        key = f"{exch}:{symbol}"
        return kite.ltp([key]).get(key, {}).get("last_price")
    except Exception as e:
        logging.warning("%s live price error: %s", symbol, e)
        return None

# ──────────────────────────────────────────────────────────────
#  Historical OHLC fetch
# ──────────────────────────────────────────────────────────────
def fetch_ohlc(token: int, start: date, end: date):
    try:
        return pd.DataFrame(kite.historical_data(token, start, end, "day"))
    except Exception as e:
        logging.warning("OHLC fetch failed: %s", e)
        return pd.DataFrame()

# ──────────────────────────────────────────────────────────────
#  DMA frame (yesterday + today with live price)
# ──────────────────────────────────────────────────────────────
LOOKBACK_CAL_DAYS = 250        # ≈165 trading sessions  → always >= 100 closes

def dma_frame(symbol: str):
    token, exch = lookup(symbol)
    if not token:
        return None
    end = last_trading_day()
    start = end - timedelta(days=LOOKBACK_CAL_DAYS)
    df = fetch_ohlc(token, start, end)
    if len(df) < 101:
        return None
    for win in (20, 50, 100):
        df[f"{win}DMA"] = df["close"].rolling(win).mean()
    fr = df.tail(2).copy()
    if fr[["20DMA", "50DMA", "100DMA"]].isna().any().any():
        return None
    live = fetch_live(token, exch, symbol)
    if live is None:
        return None
    fr.loc[fr.index[-1], "live"] = live
    return fr.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
#  Trend flag extractor
# ──────────────────────────────────────────────────────────────
def trend_flags(two_rows: pd.DataFrame):
    """Return (early_break, bullish_stack, long_uptrend) booleans."""
    y, t = two_rows.iloc[0], two_rows.iloc[1]
    early_break   = (t.live > t["20DMA"]) and (t["20DMA"] > y["20DMA"])
    bullish_stack =  t.live > t["20DMA"] > t["50DMA"] > t["100DMA"]
    long_uptrend  = (t["50DMA"] > t["100DMA"]) and (t.live > t["50DMA"]) and (t["100DMA"] > y["100DMA"])
    return early_break, bullish_stack, long_uptrend

# ──────────────────────────────────────────────────────────────
#  Bottom-10 laggard scan
# ──────────────────────────────────────────────────────────────
def laggard_scan():
    rows = []
    for sym in shares:
        fr = dma_frame(sym)
        if fr is None:
            continue
        pct = (fr.loc[1, "live"] - fr.loc[1, "20DMA"]) / fr.loc[1, "20DMA"] * 100
        rows.append((sym, pct))
    out = pd.DataFrame(rows, columns=["Symbol", "% from 20DMA"]).sort_values("% from 20DMA")
    laggards = out.head(10)
    laggards["% from 20DMA"] = laggards["% from 20DMA"].map(lambda x: f"{x:.2f}%")
    laggards.to_csv("etfs_to_buy.csv", index=False)
    print("\nBottom 10 ETFs vs 20-DMA:\n", laggards.to_string(index=False))

# ──────────────────────────────────────────────────────────────
#  Bullish-stack top-10 scan (price > 20>50>100)
# ──────────────────────────────────────────────────────────────
def bullish_scan():
    rows = []
    for sym in shares:
        fr = dma_frame(sym)
        if fr is None:
            continue
        late = fr.loc[1]
        if late.live > late["20DMA"] > late["50DMA"] > late["100DMA"]:
            pct = (late.live - late["100DMA"]) / late["100DMA"] * 100
            rows.append((sym, pct))
    out = pd.DataFrame(rows, columns=["Symbol", "% above 100DMA"]).sort_values("% above 100DMA", ascending=False)
    leaders = out.head(10)
    leaders["% above 100DMA"] = leaders["% above 100DMA"].map(lambda x: f"{x:.2f}%")
    leaders.to_csv("bullish_etfs.csv", index=False)
    print("\nTop 10 bullish-stack ETFs:\n", leaders.to_string(index=False))

import os
from datetime import datetime

def _safe_csv_write(df: pd.DataFrame, path: str):
    """
    Try to write CSV.  If the file is locked, write
    <base>_YYYYMMDD_HHMMSS.csv and warn the user.
    """
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(path)[0]
        alt  = f"{base}_{ts}.csv"
        df.to_csv(alt, index=False)
        logging.warning("%s is locked – wrote to %s instead.", path, alt)
        return alt


# ──────────────────────────────────────────────────────────────
#  Unified summary (all ETFs, all flags)
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
#  Unified summary (all ETFs, all flags)   << REPLACE THIS WHOLE DEF
# ──────────────────────────────────────────────────────────────
def unified_scan(out_path="etf_dma_summary.csv"):
    """
    Builds / refreshes the master sheet with:
      • % from 20-, 50- and 100-DMA
      • three trend flags
    and saves the result sorted by the 20-DMA column (smallest first).
    """
    rows = []
    for sym in shares:
        fr = dma_frame(sym)
        if fr is None:
            rows.append([sym, None, None, None, None, None, None])
            continue

        y, t   = fr.iloc[0], fr.iloc[1]
        pct20  = (t.live - t["20DMA"])  / t["20DMA"]  * 100
        pct50  = (t.live - t["50DMA"])  / t["50DMA"]  * 100
        pct100 = (t.live - t["100DMA"]) / t["100DMA"] * 100

        e_break, b_stack, l_up = trend_flags(fr)
        rows.append([sym, pct20, pct50, pct100, e_break, b_stack, l_up])

    df = (pd.DataFrame(rows, columns=[
            "Symbol",
            "% from 20DMA",
            "% from 50DMA",
            "% from 100DMA",
            "Early 20DMA Break",
            "Bullish Stack (Price > 20>50>100)",
            "Long-Term Uptrend (50 > 100 DMA)",
        ])
        .sort_values("% from 20DMA"))      # smallest distance to 20-DMA first

    # round numeric cols
    for c in ("% from 20DMA", "% from 50DMA", "% from 100DMA"):
        df[c] = df[c].map(lambda x: None if x is None else round(x, 2))

    # safe write (handles locked file)
    _safe_csv_write(df, out_path)
    print(f"\nFull DMA summary (sorted by % from 20DMA) written to {out_path}")

# ──────────────────────────────────────────────────────────────
#  Show only the fully-confirmed bullish ETFs
# ──────────────────────────────────────────────────────────────
def show_confirmed_bullish(csv_path="etf_dma_summary.csv",
                           out_csv="etf_dma_summary_true.csv"):
    """
    • Reads the master CSV.
    • Filters rows where all three trend flags are True.
    • Prints the result and writes it to out_csv.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌  File not found: {csv_path}")
        return

    mask = (
        df["Early 20DMA Break"].eq(True) &
        df["Bullish Stack (Price > 20>50>100)"].eq(True) &
        df["Long-Term Uptrend (50 > 100 DMA)"].eq(True)
    )
    winners = df[mask]

    if winners.empty:
        print("\nNo ETF currently meets **all three** bullish-confirmation conditions.")
    else:
        print("\nETFs satisfying every bullish filter "
              "(early 20-DMA break + bullish stack + long-term uptrend):")
        print(winners.to_string(index=False))
        winners.to_csv(out_csv, index=False)              # ← NEW line
        print(f"\n✔️  Saved {len(winners)} rows to {out_csv}")   # ← NEW line


# ──────────────────────────────────────────────────────────────
#  Main routine
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #laggard_scan()
    #bullish_scan()
    unified_scan()
    show_confirmed_bullish()