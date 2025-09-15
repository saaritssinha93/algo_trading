# hist_2min_run_dates.py
# ───────────────────────────────────────────────────────────────
# For each symbol in `selected_stocks` and for each date
# in DATES_LIST (DD-MM-YYYY):
#   • fetch 2-minute candles from 09:15 to 15:30 IST
#   • compute Daily_Change  (vs. previous trading-day close)
#   • compute Change_2min   (row-to-row % change)
#   • save / overwrite
#         main_indicators_smlitvl_his/{TICKER}_smlitvl.csv
#
# Run once, then exit.
# ───────────────────────────────────────────────────────────────
import os, time, logging, threading
from datetime import datetime, timedelta, time as dtime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

import numpy  as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kex

from et4_filtered_stocks_MIS import selected_stocks          # ← your tickers

# ───────── USER: set target dates here (DD-MM-YYYY) ─────────
DATES_LIST: List[str] = ["27-05-2025"]           # add more, e.g. ["27-05-2025","28-05-2025"]

# ───────── constants ─────────
INDIA_TZ  = pytz.timezone("Asia/Kolkata")
OUT_DIR   = "main_indicators_smlitvl_his"
THREADS   = 4
QPS       = 8                                     # crude req cap

HOLIDAYS = {date(2025, 1,26), date(2025, 3,14), date(2025, 4,11),
            date(2025, 8,15), date(2025,10, 2), date(2025,11, 5),
            date(2025,12,25)}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# ───────── logging ─────────
log = logging.getLogger("hist_dates")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh = logging.FileHandler("logs/hist_2min_dates.log", encoding="utf-8"); fh.setFormatter(fmt)
sh = logging.StreamHandler();                                             sh.setFormatter(fmt)
log.addHandler(fh); log.addHandler(sh); log.propagate = False

# ───────── Kite session ─────────
def kite_session() -> KiteConnect:
    with open("api_key.txt")      as f: api_key, *_ = f.read().strip().split()
    with open("access_token.txt") as f: access = f.read().strip()
    kc = KiteConnect(api_key=api_key); kc.set_access_token(access)
    log.info("Kite session ready.")
    return kc
kite = kite_session()

TOKENS = {d["tradingsymbol"]: d["instrument_token"]
          for d in kite.instruments("NSE")
          if d["tradingsymbol"] in selected_stocks}

# ───────── crude rate-limit bucket ─────────
class Bucket:
    def __init__(self, cap:int):
        self.cap = cap; self.t = cap
        self.lock = threading.Lock(); self.ts = time.perf_counter()
    def take(self):
        while True:
            with self.lock:
                if time.perf_counter() - self.ts >= 1:
                    self.t = self.cap; self.ts = time.perf_counter()
                if self.t:
                    self.t -= 1; return
            time.sleep(0.02)
bucket = Bucket(QPS)

# ───────── helpers ─────────
def last_trade_day(d: Optional[date]) -> date:
    """Return most recent trading day before (or equal to) d-1."""
    d = d or datetime.now(INDIA_TZ).date()
    while d.weekday() >= 5 or d in HOLIDAYS:
        d -= timedelta(days=1)
    return d

def to_ist(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert("Asia/Kolkata")

def fetch_history(token:int, start:datetime, end:datetime) -> pd.DataFrame:
    bucket.take()
    for _ in range(3):
        try:
            data = kite.historical_data(token,
                                        start.strftime("%Y-%m-%d %H:%M:%S"),
                                        end.strftime("%Y-%m-%d %H:%M:%S"),
                                        "2minute")
            return pd.DataFrame(data)
        except kex.NetworkException:
            time.sleep(0.5)
    return pd.DataFrame()

def prev_close(token:int, prev_day:date) -> float:
    """15:30 close of previous trading day."""
    end   = INDIA_TZ.localize(datetime.combine(prev_day, dtime(15,30)))
    start = end - timedelta(minutes=15)
    df = fetch_history(token, start, end)
    return df["close"].iloc[-1] if not df.empty else np.nan

# ───────── per-symbol task ─────────
def process_symbol(sym:str) -> None:
    token = TOKENS[sym]
    all_frames = []

    for d_str in DATES_LIST:
        try:
            trade_day = datetime.strptime(d_str, "%d-%m-%Y").date()
        except ValueError:
            log.error("bad date format '%s' (DD-MM-YYYY)", d_str); continue

        start = INDIA_TZ.localize(datetime.combine(trade_day, dtime(9,15)))
        end   = INDIA_TZ.localize(datetime.combine(trade_day, dtime(15,30)))

        df = fetch_history(token, start, end)
        if df.empty:
            log.warning("%s %s – no data", sym, d_str); continue

        df["date"] = to_ist(df["date"])

        pday = last_trade_day(trade_day - timedelta(days=1))
        pcl  = prev_close(token, pday)
        df["Daily_Change"] = np.nan if np.isnan(pcl) else (df["close"]-pcl)/pcl*100
        df["Change_2min"]  = df["close"].pct_change()*100

        all_frames.append(df)

    if not all_frames:
        return

    final = pd.concat(all_frames, ignore_index=True)
    keep  = ["date","open","high","low","close","volume",
             "Daily_Change","Change_2min"]
    final[keep].to_csv(os.path.join(OUT_DIR, f"{sym}_smlitvl.csv"), index=False)
    log.info("%s saved (%d rows over %d day(s))", sym, len(final), len(all_frames))

# ───────── run once ─────────
if __name__ == "__main__":
    log.info("hist-2min one-shot (%d symbols, dates=%s)", len(selected_stocks), DATES_LIST)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futs = {pool.submit(process_symbol, s): s for s in selected_stocks}
        for f in as_completed(futs):
            if exc := f.exception():
                log.error("ERR %s – %s", futs[f], exc)
