# -*- coding: utf-8 -*-
"""
eqidv1_eod_scheduler_for_15mins_data.py
========================================

Runs the **EQIDV1** 15-minute parquet updater every 15 minutes during
market hours (IST), then exits after session close.

This is adapted from stocks_eod_daily_weekly_scheduler_for_15mins_data.py
but wired to the eqidv1 backtesting core:
    backtesting/eqidv1/trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py

Typical usage:
- Run once per day via Windows Task Scheduler / batch file at ~09:10 IST.
- It will:
    * Exit immediately on non-trading days
    * Sleep until market open if started early
    * Run at each 15m boundary (09:15, 09:30, 09:45, ..., 15:30)
    * Stop and exit at 15:50 IST

Notes:
- Only 15-min data is updated (the eqidv1 core supports 5min & 15min only).
- Includes the KiteConnect auth-file "first non-empty line" fix.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from datetime import datetime, timedelta, time as dtime, date as ddate

import pytz
from kiteconnect import KiteConnect

# =============================================================================
# Wire eqidv1 core into sys.path
# =============================================================================
_ROOT = Path(__file__).resolve().parent
_EQIDV1 = _ROOT / "backtesting" / "eqidv1"
if str(_EQIDV1) not in sys.path:
    sys.path.insert(0, str(_EQIDV1))

import trading_data_continous_run_historical_alltf_v3_parquet_stocksonly as core  # noqa: E402


# =============================================================================
# CONFIG
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
STEP_MIN     = 15

# Hard stop for this process run
SESSION_END  = dtime(15, 50)

# Optional: allowlist extra special trading days (e.g., weekend budget session).
EXTRA_TRADING_DAYS: set[ddate] = set()

# Auth files (same folder as this scheduler)
API_KEY_FILE = _ROOT / "api_key.txt"
ACCESS_TOKEN_FILE = _ROOT / "access_token.txt"

REPORT_DIR = _ROOT / "reports" / "eqidv1_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# AUTH FIX (monkeypatch)
# =============================================================================
def read_first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise ValueError(f"{path} is empty or whitespace-only")


def setup_kite_session_fixed() -> KiteConnect:
    """
    FIX: only take the first non-empty token from the files (no embedded newlines).
    """
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE).split()[0]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# Ensure the updater uses the fixed session creator
core.setup_kite_session = setup_kite_session_fixed


# =============================================================================
# TIME HELPERS
# =============================================================================
def now_ist() -> datetime:
    return datetime.now(IST)


def dt_today_at(t: dtime) -> datetime:
    n = now_ist()
    return IST.localize(datetime(n.year, n.month, n.day, t.hour, t.minute, 0))


def next_boundary(t: datetime) -> datetime:
    """
    Next wall-clock boundary aligned to STEP_MIN minutes.
    Example (15m): 09:15, 09:30, 09:45, ...
    """
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


# =============================================================================
# TRADING DAY CHECK
# =============================================================================
def is_trading_day(d: ddate, holidays: set[ddate]) -> bool:
    if d in EXTRA_TRADING_DAYS:
        return True
    if hasattr(core, "_is_trading_day"):
        try:
            return bool(core._is_trading_day(d, holidays))
        except Exception:
            pass
    return d.weekday() < 5 and d not in holidays


# =============================================================================
# UPDATER CALL
# =============================================================================
def run_update_15m(holidays: set[ddate]) -> None:
    """
    Calls the eqidv1 updater for 15-min parquet data.
    """
    core.run_mode(
        mode="15min",
        max_workers=4,
        skip_if_fresh=True,
        intraday_ts="end",
        holidays=holidays,
        refresh_tokens=False,
        report_dir=str(REPORT_DIR),
        print_missing_rows=False,
        print_missing_rows_max=200,
    )


# =============================================================================
# MAIN LOOP
# =============================================================================
def main() -> None:
    try:
        holidays = set(core._read_holidays(core.HOLIDAYS_FILE_DEFAULT))
    except Exception:
        holidays = set()

    print("[LIVE] EQIDV1 15m scheduler started.")
    print("       Runs every 15 mins between 09:15 and 15:30 IST (trading days).")
    print("       Process will exit at 15:50 IST.")
    if EXTRA_TRADING_DAYS:
        show = ", ".join(sorted(d.isoformat() for d in EXTRA_TRADING_DAYS))
        print(f"       EXTRA_TRADING_DAYS allowlist: {show}")

    while True:
        n = now_ist()
        session_end = dt_today_at(SESSION_END)

        # Hard stop
        if n >= session_end:
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # Non-trading day => exit
        if not is_trading_day(n.date(), holidays):
            print("[INFO] Non-trading day. Exiting.")
            return

        start = dt_today_at(MARKET_OPEN)
        end = dt_today_at(MARKET_CLOSE)

        if n < start:
            print(f"[INFO] Before market. Sleeping until {start.strftime('%Y-%m-%d %H:%M:%S%z')}.")
            sleep_until(min(start, session_end))
            continue

        if n > end:
            if n < session_end:
                print(
                    f"[INFO] After market. Sleeping until session end "
                    f"{session_end.strftime('%Y-%m-%d %H:%M:%S%z')} then exiting."
                )
                sleep_until(session_end)
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # If we're exactly on a boundary (+-2s), run now; else wait for next boundary.
        boundary = n.replace(second=0, microsecond=0)
        on_boundary = (boundary.minute % STEP_MIN == 0) and (abs(n.second) <= 2)

        if not on_boundary:
            nb = next_boundary(n)
            if nb > end:
                nb = end
            print(f"[INFO] Next run at {nb.strftime('%Y-%m-%d %H:%M:%S%z')}")
            # Sleep slightly past boundary to ensure the new candle has closed
            sleep_until(min(nb + timedelta(seconds=2), session_end))
            continue

        try:
            print(f"[RUN ] Updating EQIDV1 15m at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
            run_update_15m(holidays)
        except Exception as e:
            print(f"[ERR ] Update failed: {e}")
            _time.sleep(3)


if __name__ == "__main__":
    main()
