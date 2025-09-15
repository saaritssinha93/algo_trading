# live_2min_collector_stable.py
# ------------------------------------------------------------------
# • polls kite.quote once a second (bulk mode)
# • rolls 2-minute candles, keeps a 14-calendar-day warm-up so that:
#       Daily_Change ‖ Change_2min ‖ RSI-14 ‖ ADX-14 ‖ MFI-14 ‖ OBV ‖ Force_Index
#   are NEVER NaN
# • *Force_Index* now uses **true 2-minute volume** (Δ of cumulative volume)
# ------------------------------------------------------------------
import os, sys, csv, time, signal, logging, threading
from datetime import datetime, timedelta, time as dtime, date
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kex

from et4_filtered_stocks_MIS import selected_stocks           # user list
# ───────── BASIC CONFIG ─────────
INDIA_TZ  = pytz.timezone("Asia/Kolkata")
OUT_DIR   = "main_indicators_smlitvl"
BAR_MIN   = 2
BATCH_SZ  = 500
QPS       = 10

HOLIDAYS = {date(2025,1,26), date(2025,3,14), date(2025,4,11),
            date(2025,8,15), date(2025,10,2), date(2025,11,5),
            date(2025,12,25)}
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# ───────── LOGGING ─────────
log = logging.getLogger("collector")
log.setLevel(logging.INFO); log.propagate = False
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
log.addHandler(logging.FileHandler("logs/collect_2min_live.log", encoding="utf-8"))
_cons = logging.StreamHandler(sys.stdout); _cons.setFormatter(_fmt); log.addHandler(_cons)

# ───────── KITE SESSION ─────────
def kite_session() -> KiteConnect:
    with open("api_key.txt") as f: api_key, *_ = f.read().split()
    with open("access_token.txt") as f: access = f.read().strip()
    kc = KiteConnect(api_key=api_key); kc.set_access_token(access)
    log.info("Kite session ready.")
    return kc
kite = kite_session()

TOKENS: Dict[str,int] = {d["tradingsymbol"]: d["instrument_token"]
                         for d in kite.instruments("NSE")
                         if d["tradingsymbol"] in selected_stocks}
TOKEN2SYM = {v:k for k,v in TOKENS.items()}
TOKENS_LIST = list(TOKENS.values())

# ───────── RATE BUCKET ─────────
class Bucket:
    def __init__(self, cap:int):
        self.cap=cap; self.t=cap; self.lock=threading.Lock(); self.ts=time.perf_counter()
    def take(self):
        while True:
            with self.lock:
                if time.perf_counter()-self.ts>=1:
                    self.t=self.cap; self.ts=time.perf_counter()
                if self.t:
                    self.t-=1; return
            time.sleep(0.01)
bucket = Bucket(QPS)

# ───────── TIME HELPERS ─────────
def floor_minute(ts:datetime, m:int=BAR_MIN) -> datetime:
    return ts - timedelta(minutes=ts.minute % m,
                          seconds=ts.second, microseconds=ts.microsecond)

def last_trade_day(d:date) -> date:
    while d.weekday()>=5 or d in HOLIDAYS: d -= timedelta(days=1)
    return d

# ───────── TA FUNCTIONS ─────────
def rsi14(close):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def adx14(df):
    hi, lo, cl = df["high"], df["low"], df["close"]
    plus  = (hi.diff()).where((hi.diff() >  lo.diff()) & (hi.diff() > 0), 0.)
    minus = (lo.diff()).where((lo.diff() >  hi.diff()) & (lo.diff() > 0), 0.)
    tr = pd.concat([(hi-lo).abs(), (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    pdi = 100 * plus.rolling(14).sum() / atr
    mdi = 100 * minus.rolling(14).sum() / atr
    dx  = 100 * (pdi-mdi).abs() / (pdi+mdi)
    return dx.rolling(14).mean()

def mfi14(df):
    tp = (df["high"]+df["low"]+df["close"])/3
    mf = tp * df["volume"]
    pos = mf.where(tp.diff() > 0, 0.)
    neg = mf.where(tp.diff() < 0, 0.).abs()
    mfr = pos.rolling(14).sum() / neg.rolling(14).sum()
    return 100 - 100 / (1 + mfr)

def obv(df):   return (df["volume"] * np.sign(df["close"].diff()).fillna(0)).cumsum()
def force(df): return df["volume"] * df["close"].diff()

# ───────── WARM-UP HISTORY (14 calendar days) ─────────
def preload_history(tok:int, days:int=14)->pd.DataFrame:
    start = INDIA_TZ.localize(
                datetime.combine(last_trade_day(datetime.now(INDIA_TZ).date()-timedelta(days=days)),
                                 dtime(9,15)))
    end   = datetime.now(INDIA_TZ)
    bucket.take()
    data = kite.historical_data(tok, start.strftime("%Y-%m-%d %H:%M:%S"),
                                     end  .strftime("%Y-%m-%d %H:%M:%S"), "2minute")
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    dt = pd.to_datetime(df["date"])
    df["date"] = dt.tz_localize("UTC").dt.tz_convert(INDIA_TZ) if dt.dt.tz is None else dt.dt.tz_convert(INDIA_TZ)
    return df

# ───────── STATE ─────────
class Slot:
    __slots__ = ("bar", "prev_close", "df", "warm_done", "last_cum_vol")
    def __init__(self):
        self.bar            = None           # current building bar
        self.prev_close     = None           # previous bar close
        self.df             = pd.DataFrame() # full history (warm-up + new bars)
        self.warm_done      = False
        self.last_cum_vol   = 0              # ← NEW  (last cumulative volume)

state: Dict[int,Slot] = defaultdict(Slot)
locks = defaultdict(threading.Lock)

def csv_path(sym): return os.path.join(OUT_DIR, f"{sym}_smlitvl.csv")

def write_bar(sym:str, end:datetime, b:dict, daily:float, chg2m:float|None, ind:dict)->None:
    hdr = ["date","open","high","low","close","volume",
           "Daily_Change","Change_2min",
           "RSI_14","ADX_14","MFI_14","OBV","Force_Index"]
    row = [end.isoformat(), b["open"], b["high"], b["low"], b["close"], b["volume"],
           round(daily,4),
           "" if chg2m is None else round(chg2m,4),
           ind["RSI_14"], ind["ADX_14"], ind["MFI_14"], ind["OBV"], ind["Force"]]
    p = csv_path(sym)
    with locks[p]:
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(hdr)
            w.writerow(row)

def prev_close(tok:int, bar_time:datetime)->float:
    ref   = last_trade_day(bar_time.date()-timedelta(days=1))
    end   = INDIA_TZ.localize(datetime.combine(ref, dtime(15,30)))
    start = end - timedelta(minutes=15)
    bucket.take()
    data  = kite.historical_data(tok, start.strftime("%Y-%m-%d %H:%M:%S"),
                                      end  .strftime("%Y-%m-%d %H:%M:%S"), "15minute")
    return data[-1]["close"] if data else np.nan

def flush_bar(tok:int, allow_net:bool=True)->None:
    slot = state[tok]; b = slot.bar
    if not b: return
    st = b["bar_time"]
    if st.minute % BAR_MIN: return                         # incomplete bar
    sym = TOKEN2SYM[tok]; end = st + timedelta(minutes=BAR_MIN)

    if allow_net and not slot.warm_done:                   # preload once
        slot.df = preload_history(tok); slot.warm_done = True

    if allow_net and slot.prev_close is None:              # fetch prev-day close once
        slot.prev_close = prev_close(tok, st)

    # append finished bar to history
    slot.df = pd.concat([slot.df,
                         pd.DataFrame([{"date":end,"open":b["open"],"high":b["high"],
                                        "low":b["low"],"close":b["close"],"volume":b["volume"]}])],
                        ignore_index=True)

    df = slot.df
    df["RSI_14"] = rsi14(df["close"])
    df["ADX_14"] = adx14(df)
    df["MFI_14"] = mfi14(df)
    df["OBV"]    = obv(df)
    df["Force"]  = force(df)

    ind   = df.iloc[-1][["RSI_14","ADX_14","MFI_14","OBV","Force"]].to_dict()
    daily = np.nan if slot.prev_close is None or np.isnan(slot.prev_close) \
                 else (b["close"]-slot.prev_close)/slot.prev_close*100
    chg2m = None if slot.prev_close is None \
                 else (b["close"]-slot.prev_close)/slot.prev_close*100

    write_bar(sym, end, b, daily, chg2m, ind)
    slot.prev_close = b["close"]
    slot.bar = None

# ───────── QUOTE POLLING ─────────
def fetch_quotes(batch: list[int])->dict:
    bucket.take()
    try:
        return kite.quote(batch)
    except kex.NetworkException as e:
        log.error("quote err %s", e)
        return {}

def poll_once()->None:
    key = floor_minute(datetime.now(INDIA_TZ))
    for i in range(0, len(TOKENS_LIST), BATCH_SZ):
        batch  = TOKENS_LIST[i:i+BATCH_SZ]
        quotes = fetch_quotes(batch)
        for tok in batch:
            q    = quotes.get(str(tok), {})
            ltp  = q.get("last_price")
            cumv = q.get("volume") or 0            # cumulative volume from exchange
            if ltp is None: continue

            slot = state[tok]
            dvol = max(cumv - slot.last_cum_vol, 0) # Δvolume since previous tick
            slot.last_cum_vol = cumv

            bar  = slot.bar
            if bar is None or bar["bar_time"] != key:      # new bar
                if bar: flush_bar(tok)
                bar = {"open": ltp, "high": ltp, "low": ltp,
                       "close": ltp, "volume": 0,  "bar_time": key}
                slot.bar = bar

            # update current bar
            bar["high"]   = max(bar["high"], ltp)
            bar["low"]    = min(bar["low"],  ltp)
            bar["close"]  = ltp
            bar["volume"]+= dvol                # <- use delta volume

# ───────── CLEAN SHUTDOWN ─────────
exit_flag = False
def on_signal(*_): global exit_flag; exit_flag = True
signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)

def market_open()->bool:
    now = datetime.now(INDIA_TZ)
    return now.replace(hour=9,minute=15,second=0) <= now <= \
           now.replace(hour=15,minute=30,second=0)

log.info("live 2-minute collector – %d symbols", len(selected_stocks))
heartbeat = time.time()

while not exit_flag:
    if market_open():
        poll_once()
        if time.time() - heartbeat >= 10:
            log.info("heartbeat …"); heartbeat = time.time()
        time.sleep(1)
    else:
        time.sleep(5)

# graceful exit (no new network calls)
for tok in list(state):
    flush_bar(tok, allow_net=False)
logging.shutdown()
sys.exit(0)
