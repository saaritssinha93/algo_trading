# -*- coding: utf-8 -*-
"""
Zerodha KiteConnect – Bottom 10 ETFs by daily % Change
Automatically fetches yesterday's close and today's live price,
ranks ETFs ascending by percentage change, and outputs the bottom 10.
"""


import os, logging, pandas as pd, numpy as np
from datetime import date, timedelta, datetime
from kiteconnect import KiteConnect

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


LOOKBACK_CAL_DAYS = 250        #  ≈ 165 trading sessions, ≥100-DMA
YEAR_DAYS          = 365        #  52-week stats

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

# ───────────────────────── Instruments ─────────────────────
inst_nse = pd.DataFrame(kite.instruments("NSE"))
inst_bse = pd.DataFrame(kite.instruments("BSE"))
instrument_df = pd.concat([inst_nse, inst_bse], ignore_index=True)

# ───────────────────────── lookup  (no SettingWithCopy) ──────────
def lookup(symbol: str):
    """
    Return (instrument_token, exchange) preferring:
      1) NSE over BSE
      2) rows whose instrument_type == 'ETF'
      3) latest expiry (rarely used for ETFs anyway)
    """
    sub = instrument_df.loc[instrument_df.tradingsymbol.eq(symbol)]
    if sub.empty:
        logging.warning("%s not in instrument dump", symbol)
        return None, None

    # make an independent copy, then add helper column cleanly
    sub = (sub.copy()
              .assign(rank = sub.exchange.eq("NSE").astype(int)
                                + sub.instrument_type.eq("ETF").astype(int)))

    best = sub.sort_values(["rank", "expiry"], ascending=False).iloc[0]
    return int(best.instrument_token), best.exchange


def fetch_ohlc(token, start, end):
    try:
        return pd.DataFrame(kite.historical_data(token, start, end, "day"))
    except Exception as e:
        logging.warning("OHLC fetch failed: %s", e); return pd.DataFrame()

def fetch_live(token, exch, sym):
    try:
        data = kite.ltp([str(token)])
        if str(token) in data:  return data[str(token)]["last_price"]
    except: pass
    try:
        key = f"{exch}:{sym}"
        return kite.ltp([key]).get(key, {}).get("last_price")
    except: return None

# ───────────────────────── Indicator helpers ───────────────
def rsi(series, n=14):
    delta = series.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(n).mean(); roll_dn = dn.rolling(n).mean()
    rs = roll_up/(roll_dn+1e-10)
    return 100 - (100/(1+rs))

def macd_hist(df, fast=12, slow=26, sig=9):
    ema_fast = df.close.ewm(span=fast, adjust=False).mean()
    ema_slow = df.close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast-ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd-signal

def adx(df, n=14):
    hi, lo, cl = df.high, df.low, df.close
    plus_dm  = (hi.diff()>lo.diff()) & (hi.diff()>0) * hi.diff()
    minus_dm = (lo.diff()>hi.diff()) & (lo.diff()>0) * -lo.diff()
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100*(plus_dm.rolling(n).mean()/atr)
    minus_di = 100*(minus_dm.rolling(n).mean()/atr)
    dx = 100*(plus_di-minus_di).abs()/(plus_di+minus_di)
    return dx.rolling(n).mean()

def mfi(df, n=14):
    tp = (df.high+df.low+df.close)/3
    mf = tp*df.volume
    up = mf.where(tp>tp.shift(), 0); dn = mf.where(tp<tp.shift(), 0)
    pos = up.rolling(n).sum(); neg = dn.rolling(n).sum()
    return 100-(100/(1+pos/(neg+1e-10)))

def obv(df):
    direction = np.sign(df.close.diff().fillna(0))
    return (df.volume*direction).cumsum()

# ───────────────────────── DMA frame (+live) ───────────────
def dma_frame(sym):
    token, exch = lookup(sym)
    if not token: return None
    end = date.today(); start = end - timedelta(days=LOOKBACK_CAL_DAYS)
    df = fetch_ohlc(token, start, end)
    if len(df)<101:  return None
    for w in (20,50,100): df[f"{w}DMA"]=df.close.rolling(w).mean()
    live = fetch_live(token, exch, sym)
    if live is None: return None
    fr = df.tail(2).copy(); fr.loc[fr.index[-1],"live"]=live
    return fr.reset_index(drop=True)

def trend_flags(two):
    y,t = two.iloc[0], two.iloc[1]
    early = (t.live>t["20DMA"]) and (t["20DMA"]>y["20DMA"])
    stack = t.live>t["20DMA"]>t["50DMA"]>t["100DMA"]
    up    = (t["50DMA"]>t["100DMA"]) and (t.live>t["50DMA"]) and (t["100DMA"]>y["100DMA"])
    return early, stack, up

# ───────────────────────── Build master summary ─────────────
def build_dma_summary(out="etf_dma_summary.csv"):
    rows=[]
    for sym in shares:
        fr = dma_frame(sym)
        if fr is None:
            rows.append([sym]+[None]*6); continue
        y,t = fr.iloc[0], fr.iloc[1]
        pct20=(t.live-t["20DMA"])/t["20DMA"]*100
        pct50=(t.live-t["50DMA"])/t["50DMA"]*100
        pct100=(t.live-t["100DMA"])/t["100DMA"]*100
        rows.append([sym, pct20, pct50, pct100, *trend_flags(fr)])
    df = pd.DataFrame(rows, columns=[
        "Symbol","% from 20DMA","% from 50DMA","% from 100DMA",
        "Early 20DMA Break","Bullish Stack (Price > 20>50>100)","Long-Term Uptrend (50 > 100 DMA)"
    ]).sort_values("% from 20DMA")
    for c in ("% from 20DMA","% from 50DMA","% from 100DMA"):
        df[c]=df[c].map(lambda x: None if x is None else round(x,2))
    df.to_csv(out, index=False); return df

# ───────────────────────── Enrich summary with momentum ─────
# ───────────────────────── Enrich summary with momentum ─────
def enrich_with_momentum(df, out="etf_dma_summary.csv"):
    today      = date.today()
    start_yr   = today - timedelta(days=YEAR_DAYS)
    extras     = []

    for _, row in df.iterrows():
        token, _ = lookup(row.Symbol)
        if not token:
            continue

        ohlc = fetch_ohlc(token, start_yr, today)
        if ohlc.empty or len(ohlc) < 15:      # need at least ~3 weeks
            continue

        close = ohlc.close
        lookback = min(len(close), 252)       # use whatever history we have
        low_n  = close.tail(lookback).min()
        high_n = close.tail(lookback).max()

        pct_low  = (close.iloc[-1] - low_n)  / low_n  * 100
        pct_high = (close.iloc[-1] - high_n) / high_n * 100

        extras.append({
            "Symbol":            row.Symbol,
            "Pct from 52w Low":  round(pct_low,  2),
            "Pct from 52w High": round(pct_high, 2),
            "RSI14":   round(rsi(close).iloc[-1], 1),
            "MACD_Hist": round(macd_hist(ohlc).iloc[-1], 3),
            "ADX14":   round(adx(ohlc).iloc[-1], 1),
            "MFI14":   round(mfi(ohlc).iloc[-1], 1),
            "OBV":     int(obv(ohlc).iloc[-1]),
        })

    merged = df.merge(pd.DataFrame(extras), on="Symbol", how="left")
    merged.to_csv(out, index=False)
    return merged


# ───────────────────────── True-flag list ───────────────────
def save_true_flags(df, out="etf_dma_summary_true.csv"):
    mask = (
        df["Early 20DMA Break"] &
        df["Bullish Stack (Price > 20>50>100)"] &
        df["Long-Term Uptrend (50 > 100 DMA)"]
    )
    winners=df[mask]
    winners.to_csv(out,index=False)
    logging.info("Saved %d rows to %s", len(winners), out)
    return winners

# ───────────────────────── Bottom-finder list ───────────────
# ───────────────────────── Bottom-finder (relaxed) ──────────────
def save_bottom_watch(df, out="etf_dma_bottoms.csv"):
    """
    Looser rules:
      • still within 10 % of the 52-week low   (was 5 %)
      • bullish-stack confirmed   (price > 20 > 50 > 100)
      • Early-break flag is now optional
      • long-term up-trend flag kept (50 > 100 DMA)
      • ADX ≥ 20   (was 25)  – trend gaining but not yet strong
      • RSI between 30-65    (was 30-60) – allows a bit more momentum
    """
    mask = (
        (df["Pct from 52w Low"] <= 20) &                        # ← wider bottom band
        df["Bullish Stack (Price > 20>50>100)"] &               # must be stacked
        df["Long-Term Uptrend (50 > 100 DMA)"] &
        df["Early 20DMA Break"] &                # keep major trend
        (df["ADX14"] >= 25) &                                   # milder ADX threshold
        df["RSI14"].between(30, 60) &
        #
        # --- NEW momentum / volume confirmations ---
        (df["MACD_Hist"] > -1) &                 # histogram crossed above zero
        (df["MFI14"]  > 30) &                         # positive money-flow
        (df["OBV"] > 0)                         # 5-day OBV is rising                          # wider RSI window
        # Early-20DMA break no longer mandatory, but include it
        # if you want a “priority” column:
        # & df["Early 20DMA Break"]
    )

        # sort by “distance to 20-DMA” (smallest first) and save
    bot = (df[mask]
           .sort_values("% from 20DMA", ascending=True))

    bot.to_csv(out, index=False)
    logging.info("Saved %d relaxed bottom-candidates to %s",
                 len(bot), out)



# ───────────────────────── MAIN ─────────────────────────────
if __name__ == "__main__":
    # full refresh
    base = build_dma_summary()
    full = enrich_with_momentum(base)
    save_true_flags(full)
    save_bottom_watch(full)
    print("\nDone – check the three CSV files for results.")