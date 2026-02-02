# etf_live_15m_scheduler.py
# Runs the ETF 15-min parquet updater every 15 minutes during market hours (IST).

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, time as dtime
import pytz

from kiteconnect import KiteConnect

import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_etfsonly as core


IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
STEP_MIN     = 15

# NEW: hard stop for this process run
SESSION_END  = dtime(15, 50)

API_KEY_FILE = "api_key.txt"
ACCESS_TOKEN_FILE = "access_token.txt"


def read_first_nonempty_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise ValueError(f"{path} is empty or whitespace-only")


def setup_kite_session_fixed() -> KiteConnect:
    """
    FIX: only take the first non-empty line from the files (no embedded newlines).
    """
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# Monkeypatch the core script session creator so everything downstream uses the fixed behavior
core.setup_kite_session = setup_kite_session_fixed


def now_ist() -> datetime:
    return datetime.now(IST)


def dt_today_at(t: dtime) -> datetime:
    n = now_ist()
    return IST.localize(datetime(n.year, n.month, n.day, t.hour, t.minute, 0))


def is_within_session(t: datetime) -> bool:
    tt = t.time()
    return (tt >= MARKET_OPEN) and (tt <= MARKET_CLOSE)


def next_boundary(t: datetime) -> datetime:
    """
    Next wall-clock boundary aligned to 15 minutes, anchored naturally.
    Example: 09:15, 09:30, 09:45, ...
    """
    # Round up to next STEP_MIN boundary
    minute = (t.minute // STEP_MIN) * STEP_MIN
    base = t.replace(minute=minute, second=0, microsecond=0)
    if base < t:
        base = base + timedelta(minutes=STEP_MIN)
    return base


def sleep_until(target: datetime) -> None:
    while True:
        n = now_ist()
        secs = (target - n).total_seconds()
        if secs <= 0:
            return
        _time.sleep(min(secs, 2.0))


def run_update_15m(holidays: set) -> None:
    """
    Calls your existing updater. It will append missing 15m rows and write parquet.
    """
    core.run_mode(
        mode="15min",
        context="live",
        max_workers=4,              # tune 2-6 depending on rate limits
        force_today_daily=False,
        skip_if_fresh=True,         # key: avoids rewriting if already up-to-date
        intraday_ts="end",          # use candle END timestamps
        holidays=holidays,
        refresh_tokens=False,       # uses cached tokens unless you force refresh
        report_dir="reports",
        print_missing_rows=False,
        print_missing_rows_max=200,
    )


def main() -> None:
    holidays = core._read_holidays(core.HOLIDAYS_FILE_DEFAULT)
    special_open = core._read_special_trading_days(getattr(core, "SPECIAL_TRADING_DAYS_FILE_DEFAULT", "nse_special_trading_days.csv"))


    print("[LIVE] ETF 15m scheduler started.")
    print("       Runs every 15 mins between 09:15 and 15:30 IST (trading days).")
    print("       Process will exit at 15:50 IST.")

    while True:
        n = now_ist()
        session_end = dt_today_at(SESSION_END)

        # NEW: stop this run at 15:50 IST
        if n >= session_end:
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # Skip non-trading days (for a scheduled run, just exit)
        
        if not core._is_trading_day(n.date(), holidays, special_open):
            print("[INFO] Non-trading day. Exiting.")
            return

        start = dt_today_at(MARKET_OPEN)
        end = dt_today_at(MARKET_CLOSE)

        if n < start:
            print(f"[INFO] Before market. Sleeping until {start}.")
            sleep_until(min(start, session_end))
            continue

        if n > end:
            # NEW: don't sleep to next day; end this run at 15:50
            if n < session_end:
                print(f"[INFO] After market. Sleeping until session end {session_end.strftime('%Y-%m-%d %H:%M:%S%z')} then exiting.")
                sleep_until(session_end)
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # If we're exactly on a boundary (±2s), run now; else wait for next boundary.
        boundary = n.replace(second=0, microsecond=0)
        on_boundary = (boundary.minute % STEP_MIN == 0) and (abs(n.second) <= 2)

        if not on_boundary:
            nb = next_boundary(n)
            # ensure we stay within session
            if nb > end:
                nb = end
            print(f"[INFO] Next run at {nb.strftime('%Y-%m-%d %H:%M:%S%z')}")
            sleep_until(min(nb + timedelta(seconds=2), session_end))

            # loop continues; SESSION_END check at top will exit if needed

        try:
            print(f"[RUN ] Updating 15m at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
            run_update_15m(holidays)
        except Exception as e:
            print(f"[ERR ] Update failed: {e}")
            _time.sleep(3)


if __name__ == "__main__":
    main()
