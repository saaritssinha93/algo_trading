# -*- coding: utf-8 -*-
"""
Ultra-lean live 2-minute collector (bulk kite.quote)

• polls once a second
• 500 tokens per quote-call  →  3 calls/s are enough for ~1 100 symbols
• finished bars are flushed to main_indicators_smlitvl/{TICKER}_smlitvl.csv
"""

import os, sys, csv, time, signal, logging, threading
from datetime import datetime, timedelta, time as dtime
from itertools   import islice
from collections import defaultdict, deque
import numpy as np, pytz
from kiteconnect import KiteConnect, exceptions as kex

# ───────────────────────── CONFIG ─────────────────────────
INDIA_TZ   = pytz.timezone("Asia/Kolkata")
OUT_DIR    = "main_indicators_smlitvl"
BAR_MIN    = 2
BATCH_SZ   = 500              # max tokens per quote() call (Kite doc)
QPS        = 10                # 3 quote-calls / second – very safe
from et4_filtered_stocks_MIS import selected_stocks    # your list

HOLIDAYS = {datetime(2025,1,26).date(), datetime(2025,3,14).date(),
            datetime(2025,4,11).date(), datetime(2025,8,15).date(),
            datetime(2025,10,2).date(), datetime(2025,11,5).date(),
            datetime(2025,12,25).date()}
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────── LOGGING ─────────────────────────
log = logging.getLogger("collector"); log.setLevel(logging.INFO)
log.propagate = False
for h in list(log.handlers):              # clear dupes on re-run
    log.removeHandler(h)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh  = logging.FileHandler("logs/collect_2min_live.log", encoding="utf-8")
fh.setFormatter(fmt); log.addHandler(fh)
sh  = logging.StreamHandler(sys.stdout);  sh.setFormatter(fmt); log.addHandler(sh)

# ───────────────────────── KITE ─────────────────────────
def kite_session():
    with open("api_key.txt") as f: api_key, *_ = f.read().split()
    with open("access_token.txt") as f: token = f.read().strip()
    kc = KiteConnect(api_key=api_key); kc.set_access_token(token)
    log.info("Kite session ready."); return kc
kite = kite_session()

TOKENS    = {d["tradingsymbol"]: d["instrument_token"]
             for d in kite.instruments("NSE")
             if d["tradingsymbol"] in selected_stocks}
TOKEN2SYM = {v:k for k,v in TOKENS.items()}

# ───────────────────────── BUCKET ─────────────────────────
class Bucket:
    def __init__(self, cap):
        self.cap=cap; self.t=cap; self.lock=threading.Lock(); self.ts=time.perf_counter()
    def take(self):
        while True:
            with self.lock:
                if time.perf_counter()-self.ts >= 1:
                    self.t=self.cap; self.ts=time.perf_counter()
                if self.t: self.t-=1; return
            time.sleep(0.01)
bucket = Bucket(QPS)

# ───────────────────────── UTIL ─────────────────────────
def floor_minute(dt, m=BAR_MIN):
    return dt - timedelta(minutes=dt.minute% m,
                          seconds=dt.second, microseconds=dt.microsecond)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        head = list(islice(it, n))
        if not head: break
        yield head

def last_trade_day(d=None):
    d = d or datetime.now(INDIA_TZ).date()
    while d.weekday()>=5 or d in HOLIDAYS: d-=timedelta(days=1)
    return d

def yesterday_close(token):
    ltd = last_trade_day() - timedelta(days=1)
    end = INDIA_TZ.localize(datetime.combine(ltd, dtime(15,30)))
    start = end - timedelta(minutes=15)
    bucket.take()
    data = kite.historical_data(token, start.strftime("%Y-%m-%d %H:%M:%S"),
                                end.strftime("%Y-%m-%d %H:%M:%S"), "15minute")
    return data[-1]["close"] if data else np.nan

# ───────────────────────── STATE ─────────────────────────
class Slot:
    __slots__=("bar","yclose","prev_close","chg2m")
    def __init__(self):
        self.bar=None; self.yclose=np.nan; self.prev_close=None; self.chg2m=np.nan
state  = defaultdict(Slot)
locks  = defaultdict(threading.Lock)   # one lock per CSV

def csv_path(sym): return os.path.join(OUT_DIR, f"{sym}_smlitvl.csv")

# ───────────────────────── CSV write ─────────────────────────
def write_bar(sym, bar_end, bdict, daily, chg2m):
    row = [bar_end.isoformat(), bdict["open"], bdict["high"], bdict["low"],
           bdict["close"], bdict["volume"], round(daily,4),
           round(chg2m,4) if not np.isnan(chg2m) else ""]
    p = csv_path(sym)
    with locks[p]:
        new = not os.path.exists(p)
        with open(p,"a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["date","open","high","low","close","volume",
                                "Daily_Change","Change_2min"])
            w.writerow(row)
    log.info("[%s] %s bar", sym, bar_end.strftime("%H:%M"))
        

# ───────────────────────── FLAG UTILITY ─────────────────────────
def add_flag_column(sym: str):
    """
    Post-process an existing
        main_indicators_smlitvl/{sym}_smlitvl.csv

    • Adds/overwrites a column called   flag
    • Rule:
        flag = "YES"  if
            (Daily_Change ≥ +2.00  and the *current* and *previous*
             Change_2min are both > 0)
         or (Daily_Change ≤ −2.00  and the *current* and *previous*
             Change_2min are both < 0)
        otherwise "NO"

    The function is *not* called by the collector – run it whenever you
    want to refresh the flags (for one ticker or in a loop over all files).
    """
    path = csv_path(sym)               # re-use helper from your script
    if not os.path.exists(path):
        log.warning("add_flag_column: %s not found", path)
        return

    try:
        df = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                           encoding="utf-8", missing_values="")
        if df.size == 0:
            log.warning("add_flag_column: %s is empty", path); return
        # Work in pandas for convenience
        import pandas as pd
        pdf = pd.read_csv(path, parse_dates=["date"])

        # Ensure numeric
        pdf["Daily_Change"]  = pd.to_numeric(pdf["Daily_Change"],  errors="coerce")
        pdf["Change_2min"]   = pd.to_numeric(pdf["Change_2min"],   errors="coerce")

        # Default NO
        pdf["flag"] = "NO"

        pos = (pdf["Daily_Change"] >=  2.0) & \
              (pdf["Change_2min"]  >  0)   & \
              (pdf["Change_2min"].shift(1) > 0)

        neg = (pdf["Daily_Change"] <= -2.0) & \
              (pdf["Change_2min"]  <  0)   & \
              (pdf["Change_2min"].shift(1) < 0)

        pdf.loc[pos | neg, "flag"] = "YES"

        # Overwrite file (keep column order)
        cols = ["date","open","high","low","close","volume",
                "Daily_Change","Change_2min","flag"]
        pdf.to_csv(path, index=False, columns=cols)
        log.info("add_flag_column: updated %s (%d rows processed)", sym, len(pdf))
    except Exception as e:
        log.error("add_flag_column: failed for %s – %s", sym, e)

def update_flags_all():
    """
    Loop over every ticker in `selected_stocks` and rebuild the flag column
    in its *_smlitvl.csv file.  Safe to run multiple times.
    """
    for sym in selected_stocks:
        add_flag_column(sym)

# ───────────────────────── MAIN LOOP helpers ─────────────────────────
def flush(token):
    sl = state[token]
    b  = sl.bar
    if b is None: return
    bar_time = b["time"]
    if bar_time.minute % BAR_MIN: return        # ignore incomplete
    sym = TOKEN2SYM[token]
    if np.isnan(sl.yclose):
        sl.yclose = yesterday_close(token)
    daily = np.nan if np.isnan(sl.yclose) else (b["close"]-sl.yclose)/sl.yclose*100
    chg2m = np.nan if sl.prev_close is None else (b["close"]-sl.prev_close)/sl.prev_close*100
    write_bar(sym, bar_time + timedelta(minutes=BAR_MIN), b, daily, chg2m)
    sl.prev_close = b["close"]       # for next Δ2m
    sl.bar = None                    # reset

# ───────────────────────── QUOTE FETCH ─────────────────────────
def fetch_quotes(batch):
    bucket.take()
    try:
        return kite.quote(batch)
    except kex.NetworkException as e:
        log.error("quote err: %s", e); return {}

# ───────────────────────── POLL ─────────────────────────
def poll_once():
    for batch in batched(list(TOKENS.values()), BATCH_SZ):
        q = fetch_quotes(batch)
        now = datetime.now(INDIA_TZ)
        key = floor_minute(now)
        for tok in batch:
            tok_str = str(tok)
            if tok_str not in q: continue
            data = q[tok_str]
            ltp  = data["last_price"]
            vol  = (data.get("volume") or data.get("volume_traded") or
                    data.get("volume_traded_today") or 0)
            sl = state[tok]
            b  = sl.bar
            if b is None:          # first tick for this bar
                b = {"open":ltp,"high":ltp,"low":ltp,"close":ltp,
                     "volume":0,"time":key}
                sl.bar = b
            if b["time"] != key:   # bar complete → persist
                flush(tok)
                b = {"open":ltp,"high":ltp,"low":ltp,"close":ltp,
                     "volume":0,"time":key}
                sl.bar = b
            # update running aggregates
            b["high"]  = max(b["high"], ltp)
            b["low"]   = min(b["low"],  ltp)
            b["close"] = ltp
            b["volume"] += vol

# ───────────────────────── SHUTDOWN ─────────────────────────
def graceful(*_):
    log.info("Exiting – flushing bars")
    for tok in list(state):
        flush(tok)
    sys.exit(0)
signal.signal(signal.SIGINT, graceful); signal.signal(signal.SIGTERM, graceful)

# ───────────────────────── MARKET WINDOW ─────────────────────────
def mkt_open():
    now = datetime.now(INDIA_TZ)
    return now.replace(hour=9,minute=15,second=0) <= now <= \
           now.replace(hour=15,minute=30,second=0)

# ───────────────────────── RUN ─────────────────────────
log.info("FAST-v6 2-min collector – %d symbols", len(selected_stocks))
last_ping = time.time()

while True:
    if mkt_open():
        poll_once()
        if time.time() - last_ping >= 1:
            log.info("Heartbeat …"); last_ping = time.time()
        time.sleep(1)
    else:
        for tok in list(state): flush(tok)
        now = datetime.now(INDIA_TZ)
        nxt = (now + timedelta(days=1)).replace(hour=9,minute=15,second=0,microsecond=0)
        while nxt.date().weekday()>=5 or nxt.date() in HOLIDAYS:
            nxt += timedelta(days=1)
        sleep = (nxt - now).total_seconds()
        log.info("Market closed – sleeping %.0f s", sleep)
        time.sleep(sleep)
