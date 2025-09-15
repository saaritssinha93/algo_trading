# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:26:05 2025

@author: Saarit
"""
# hist_2min_run_dates_indicators.py
# ───────────────────────────────────────────────────────────────
# One-shot script
#   • for every date in DATES_LIST and every symbol in selected_stocks
#       – pull 2-minute candles (incl. a 14-calendar-day *warm-up* window)
#       – add on every bar
#            · Daily_Change   (% vs. previous trading-day 15 :30 close)
#            · Change_2min    (% vs. previous bar)
#            · RSI-14
#            · ADX-14
#            · MFI-14
#            · OBV           (cum volume × direction)
#            · Force_Index   (ΔClose × Volume)
#   • write rows that belong to the requested dates to
#         main_indicators_smlitvl_his/{TICKER}_smlitvl.csv
# ───────────────────────────────────────────────────────────────
import os, time, logging, threading, glob
from datetime import datetime, timedelta, time as dtime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

import numpy  as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kex

# ───────── USER SETTINGS ─────────
from et4_filtered_stocks_MIS import selected_stocks
# All NSE / BSE equity-segment trading days in May 2025
# NSE/BSE equity-segment trading days
# (Wed 1 Jan 2025 → Thu 29 May 2025)
# --- weekends removed
# --- weekday market holidays removed:
#       26-Feb (Maha Shivaratri) 14-Mar (Holi) 31-Mar (Eid-ul-Fitr)
#       10-Apr (Mahavir Jayanti) 14-Apr (Ambedkar Jayanti)
#       18-Apr (Good Friday)     01-May (Maharashtra Day) :contentReference[oaicite:0]{index=0}
# Total sessions: 100

DATES_LIST: list[str] = [
    "02-05-2025",  # Fri
    "05-05-2025",  # Mon
    "06-05-2025",  # Tue
    "07-05-2025",  # Wed
    "08-05-2025",  # Thu
    "09-05-2025",  # Fri
    "12-05-2025",  # Mon
    "13-05-2025",  # Tue
    "14-05-2025",  # Wed
    "15-05-2025",  # Thu
    "16-05-2025",  # Fri
    "19-05-2025",  # Mon
    "20-05-2025",  # Tue
    "21-05-2025",  # Wed
    "22-05-2025",  # Thu
    "23-05-2025",  # Fri
    "26-05-2025",  # Mon
    "27-05-2025",  # Tue
    "28-05-2025",  # Wed
    "29-05-2025",  # Thu
    "30-05-2025",  # Fri
    "02-06-2025",  # Mon
    "03-06-2025",  # Tue
    "04-06-2025",  # Wed
    "05-06-2025",  # Thu
    "06-06-2025",  # Fri
    "09-06-2025",  # Mon
    "10-06-2025",  # Tue
    "11-06-2025",  # Wed
    "12-06-2025",  # Thu
    "13-06-2025",  # Fri
    "16-06-2025",  # Mon
    "17-06-2025",  # Tue
]

                    # DD-MM-YYYY …

# ───────── CONSTANTS ─────────
INDIA_TZ = pytz.timezone("Asia/Kolkata")
OUT_DIR  = "main_indicators_smlitvl_his"
THREADS, QPS = 4, 8                                        # parallelism & crude rate-limit
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs("logs", exist_ok=True)

HOLIDAYS = {date(2025, 1,26), date(2025, 3,14), date(2025, 4,11),
            date(2025, 8,15), date(2025,10, 2), date(2025,11, 5),
            date(2025,12,25)}

# ───────── LOGGING ─────────
log = logging.getLogger("hist_dates")
log.setLevel(logging.INFO); log.propagate = False
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
for h in list(log.handlers): log.removeHandler(h)
fh = logging.FileHandler("logs/hist_2min_dates.log", encoding="utf-8"); fh.setFormatter(_fmt); log.addHandler(fh)
sh = logging.StreamHandler(); sh.setFormatter(_fmt); log.addHandler(sh)

# ───────── KiteConnect ─────────
def kite_session() -> KiteConnect:
    with open("api_key.txt")      as f: api_key, *_ = f.read().split()
    with open("access_token.txt") as f: access  = f.read().strip()
    kc = KiteConnect(api_key=api_key); kc.set_access_token(access)
    log.info("Kite session ready.")
    return kc
kite = kite_session()

TOKENS: Dict[str,int] = {d["tradingsymbol"]: d["instrument_token"]
                         for d in kite.instruments("NSE")
                         if d["tradingsymbol"] in selected_stocks}

# ───────── crude token-bucket ─────────
class Bucket:
    def __init__(self, cap:int):
        self.cap, self.t = cap, cap
        self.lock = threading.Lock(); self.ts = time.perf_counter()
    def take(self):
        while True:
            with self.lock:
                if time.perf_counter() - self.ts >= 1:
                    self.t = self.cap; self.ts = time.perf_counter()
                if self.t: self.t -= 1; return
            time.sleep(0.02)
bucket = Bucket(QPS)

# ───────── HELPERS ─────────
def last_trade_day(d: date) -> date:
    while d.weekday() >= 5 or d in HOLIDAYS: d -= timedelta(days=1)
    return d

def to_ist(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    if s.dt.tz is None: s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert("Asia/Kolkata")

def fetch_history(token:int, start:datetime, end:datetime) -> pd.DataFrame:
    bucket.take()
    for _ in range(3):
        try:
            data = kite.historical_data(
                token,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end  .strftime("%Y-%m-%d %H:%M:%S"),
                "2minute")
            return pd.DataFrame(data)
        except kex.NetworkException: time.sleep(0.5)
    return pd.DataFrame()

def prev_close(token:int, prev_day:date) -> float:
    end   = INDIA_TZ.localize(datetime.combine(prev_day, dtime(15,30)))
    start = end - timedelta(minutes=15)
    df = fetch_history(token, start, end)
    return np.nan if df.empty else df["close"].iloc[-1]

# ───────── TA INDICATORS ─────────
def rsi(close: pd.Series, n:int=14) -> pd.Series:
    delta = close.diff()
    up  = delta.clip(lower=0);  dn = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean(); ma_dn = dn.rolling(n).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def adx(df: pd.DataFrame, n:int=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = (high.diff()).where((high.diff() >  low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (low.diff() ).where((low.diff()  > high.diff()) & (low.diff()  > 0), 0.0)
    tr = pd.concat([(high-low).abs(),
                    (high-close.shift()).abs(),
                    (low -close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * plus_dm.rolling(n).sum()  / atr
    minus_di = 100 * minus_dm.rolling(n).sum() / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di)
    return dx.rolling(n).mean()

def mfi(df: pd.DataFrame, n:int=14) -> pd.Series:
    tp = (df["high"]+df["low"]+df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0).abs()
    mfr = pos.rolling(n).sum() / neg.rolling(n).sum()
    return 100 - 100 / (1 + mfr)

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (df["volume"] * direction).cumsum()

def force_index(df: pd.DataFrame) -> pd.Series:
    return df["volume"] * df["close"].diff()

def ema(s: pd.Series, span:int):
    return s.ewm(span=span, adjust=False).mean()

def macd(df: pd.DataFrame):
    fast, slow, sig = 12, 26, 9
    line = ema(df.close, fast) - ema(df.close, slow)
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return pd.DataFrame({"MACD_Line": line, "MACD_Signal": signal, "MACD_Hist": hist})

def stochastic(df: pd.DataFrame):
    k_len, d_len = 14, 3
    low_min  = df.low.rolling(k_len).min()
    high_max = df.high.rolling(k_len).max()
    pct_k = 100 * (df.close - low_min) / (high_max - low_min)
    pct_d = pct_k.rolling(d_len).mean()
    return pd.DataFrame({"Stoch_%K": pct_k, "Stoch_%D": pct_d})

def vwap_daily(df: pd.DataFrame):
    grp = df.groupby(df.date.dt.date)
    return grp.apply(lambda g: ( ((g.high+g.low+g.close)/3 * g.volume).cumsum() / g.volume.cumsum() ))\
              .droplevel(0)


# ───────── PROCESS ONE SYMBOL ─────────
def process_symbol(sym:str) -> None:
    token = TOKENS[sym]
    # parse dates once
    trg_dates = [datetime.strptime(d, "%d-%m-%Y").date() for d in DATES_LIST]

    # ---- fetch a 14-calendar-day warm-up window before the earliest target ----
    earliest = min(trg_dates)
    warm_start_day = earliest - timedelta(days=14)
    warm_start     = INDIA_TZ.localize(datetime.combine(warm_start_day, dtime(9,15)))
    warm_end       = INDIA_TZ.localize(datetime.combine(max(trg_dates), dtime(15,30)))

    df_full = fetch_history(token, warm_start, warm_end)
    if df_full.empty:
        log.warning("%s – no data retrieved", sym); return
    df_full["date"] = to_ist(df_full["date"])

    # ---- Daily_Change requires previous trading-day 15 :30 close ----
    # cache previous day close for speed
    prev_close_cache: Dict[date,float] = {}
    closes = []
    for ts, cls in zip(df_full["date"], df_full["close"]):
        cur_day = ts.date()
        if cur_day not in prev_close_cache:
            prev_day = last_trade_day(cur_day - timedelta(days=1))
            prev_close_cache[cur_day] = prev_close(token, prev_day)
        closes.append(prev_close_cache[cur_day])
    df_full["Prev_Close"]   = closes
    df_full["Daily_Change"] = (df_full["close"] - df_full["Prev_Close"]) / df_full["Prev_Close"] * 100
    df_full.loc[df_full["Prev_Close"].isna(), "Daily_Change"] = np.nan
    df_full["Change_2min"]  = df_full["close"].pct_change() * 100

    # ---- INDICATORS (need full df for warm-up) ----
    df_full["RSI_14"]      = rsi(df_full["close"])
    df_full["ADX_14"]      = adx(df_full)
    df_full["MFI_14"]      = mfi(df_full)
    df_full["OBV"]         = obv(df_full)
    df_full["Force_Index"] = force_index(df_full)
    df_full["EMA_5"]       = ema(df_full.close, 5)
    df_full["EMA_20"]      = ema(df_full.close, 20)
    df_full               = pd.concat([df_full, macd(df_full), stochastic(df_full)], axis=1)
    df_full["VWAP"]       = vwap_daily(df_full)    

    # ---- keep only rows whose calendar date is in DATES_LIST ----
    mask = df_full["date"].dt.date.isin(trg_dates)
    out  = df_full.loc[mask].copy()

    out_cols = [
        "date","open","high","low","close","volume",
        "Prev_Close","Daily_Change","Change_2min",
        "RSI_14","ADX_14","MFI_14","OBV","Force_Index",
        "EMA_5","EMA_20",
        "MACD_Line","MACD_Signal","MACD_Hist",
        "Stoch_%K","Stoch_%D","VWAP"
    ]
    out[out_cols].to_csv(os.path.join(OUT_DIR, f"{sym}_smlitvl.csv"), index=False)
    log.info("%s saved – rows=%d  days=%d", sym, len(out), len(trg_dates))

# ───────── MAIN ─────────
if __name__ == "__main__":
    log.info("hist-2min with indicators (symbols=%d  dates=%s)", len(selected_stocks), DATES_LIST)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futs = {pool.submit(process_symbol, s): s for s in selected_stocks}
        for f in as_completed(futs):
            if (e := f.exception()) is not None:
                log.error("ERR %s – %s", futs[f], e)