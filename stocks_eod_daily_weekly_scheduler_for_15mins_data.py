# -*- coding: utf-8 -*-
"""
algosm1_live_15m_scheduler_stocksonly.py
======================================

Runs the **STOCKS** 15-minute parquet updater (ALGO-SM1 stocksonly core)
every 15 minutes during market hours (IST), and exits at SESSION_END.

This is adapted from your ETF 15m scheduler pattern, but wired to:
    algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py

Typical usage:
- Run this script once per day (e.g., via Windows Task Scheduler at 09:10 IST).
- It will:
    * Exit immediately on non-trading days
    * Sleep until market open if started early
    * Run at each 15m boundary (09:15, 09:30, 09:45, ...)
    * Stop after market close and exit at 15:50 IST

Notes:
- Includes the KiteConnect auth-file "first non-empty line" fix.
- Handles both cores that implement _is_trading_day(d, holidays) and
  cores that implement _is_trading_day(d, holidays, special_open).
"""

from __future__ import annotations

import time as _time
import inspect
from pathlib import Path
from datetime import datetime, timedelta, time as dtime, date as ddate

import pytz
from kiteconnect import KiteConnect

import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly as core


# =============================================================================
# CONFIG
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
STEP_MIN     = 15

# Hard stop for this process run (so a daily task doesn't keep running forever)
SESSION_END  = dtime(15, 50)

# Optional: Run DAILY+WEEKLY once near EOD (useful if you want daily/weekly parquet too)
RUN_EOD_DAILY_WEEKLY = True
EOD_RUN_AT = dtime(15, 40)

# Optional: allowlist extra special trading days (e.g., weekend budget session).
# Keep this EMPTY unless you explicitly want to force-run on these dates.
EXTRA_TRADING_DAYS: set[ddate] = set([
    # ddate(2026, 2, 1),
])

# Auth files (same folder as this scheduler)
ROOT = Path(__file__).resolve().parent
API_KEY_FILE = ROOT / "api_key.txt"
ACCESS_TOKEN_FILE = ROOT / "access_token.txt"

REPORT_DIR = ROOT / "reports"
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
    eod_done = False

    while True:
        n = now_ist()
        secs = (target - n).total_seconds()
        if secs <= 0:
            return
        _time.sleep(min(secs, 2.0))


# =============================================================================
# TRADING DAY CHECK
# =============================================================================
def _load_special_open_days() -> set[ddate]:
    """
    Some cores support special trading days via _read_special_trading_days().
    If not present, return empty set.
    """
    if hasattr(core, "_read_special_trading_days"):
        try:
            path = getattr(core, "SPECIAL_TRADING_DAYS_FILE_DEFAULT", "nse_special_trading_days.csv")
            return set(core._read_special_trading_days(path))  # type: ignore[attr-defined]
        except Exception:
            return set()
    return set()


def is_trading_day(d: ddate, holidays: set[ddate], special_open: set[ddate]) -> bool:
    """
    Supports both signatures:
      - _is_trading_day(d, holidays)
      - _is_trading_day(d, holidays, special_open)
    plus EXTRA_TRADING_DAYS allowlist.
    """
    if d in EXTRA_TRADING_DAYS:
        return True

    if not hasattr(core, "_is_trading_day"):
        # Fail-safe: assume weekdays excluding holidays
        if d.weekday() >= 5:
            return False
        return d not in holidays

    try:
        sig = inspect.signature(core._is_trading_day)  # type: ignore[attr-defined]
        if len(sig.parameters) >= 3:
            return bool(core._is_trading_day(d, holidays, special_open))  # type: ignore[misc]
        return bool(core._is_trading_day(d, holidays))  # type: ignore[misc]
    except Exception:
        # Fallback: simple weekday check
        if d.weekday() >= 5:
            return False
        return d not in holidays


# =============================================================================
# UPDATER CALL
# =============================================================================
def run_update_15m(holidays: set[ddate]) -> None:
    """
    Calls your existing stocksonly updater. It will append missing 15m rows and write parquet.
    """
    core.run_mode(
        mode="15min",
        context="live",
        max_workers=4,              # tune 2-6 depending on rate limits
        force_today_daily=False,
        skip_if_fresh=True,         # avoids rewriting if already up-to-date
        intraday_ts="end",          # use candle END timestamps
        holidays=holidays,
        refresh_tokens=False,       # uses cached tokens unless you force refresh
        report_dir=str(REPORT_DIR),
        print_missing_rows=False,
        print_missing_rows_max=200,
    )

def run_eod_daily_weekly(holidays: set[ddate]) -> None:
    """
    Optional EOD job: update DAILY and WEEKLY parquet once per trading day.
    - Run time default: 15:40 IST (config EOD_RUN_AT).
    - Uses skip_if_fresh=True to avoid rework.
    """
    for mode in ("daily", "weekly"):
        core.run_mode(
            mode=mode,
            context="live",
            max_workers=4,
            force_today_daily=False,
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
    # Trading calendar inputs
    try:
        holidays = set(core._read_holidays(core.HOLIDAYS_FILE_DEFAULT))  # type: ignore[attr-defined]
    except Exception:
        holidays = set()

    special_open = _load_special_open_days()

    print("[LIVE] STOCKS 15m scheduler started.")
    print("       Runs every 15 mins between 09:15 and 15:30 IST (trading days).")
    print("       Process will exit at 15:50 IST.")
    if EXTRA_TRADING_DAYS:
        show = ", ".join(sorted([d.isoformat() for d in EXTRA_TRADING_DAYS]))
        print(f"       EXTRA_TRADING_DAYS allowlist: {show}")

    eod_done = False

    while True:
        n = now_ist()
        session_end = dt_today_at(SESSION_END)

        # Hard stop
        if n >= session_end:
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # Non-trading day => exit (for scheduled runs)
        if not is_trading_day(n.date(), holidays, special_open):
            print("[INFO] Non-trading day. Exiting.")
            return

        start = dt_today_at(MARKET_OPEN)
        end = dt_today_at(MARKET_CLOSE)
        eod_time = dt_today_at(EOD_RUN_AT)

        # Optional EOD DAILY+WEEKLY update
        if RUN_EOD_DAILY_WEEKLY and (not eod_done) and (n >= eod_time):
            try:
                print(f"[EOD ] Updating DAILY+WEEKLY at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
                run_eod_daily_weekly(holidays)
                eod_done = True
            except Exception as e:
                print(f"[ERR ] EOD update failed: {e}")
                _time.sleep(3)

        if n < start:
            print(f"[INFO] Before market. Sleeping until {start.strftime('%Y-%m-%d %H:%M:%S%z')}.")
            sleep_until(min(start, session_end))
            continue

        if n > end:
            # Do not sleep to next day; exit at SESSION_END
            if n < session_end:
                print(
                    f"[INFO] After market. Sleeping until session end "
                    f"{session_end.strftime('%Y-%m-%d %H:%M:%S%z')} then exiting."
                )
                sleep_until(session_end)
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # If we're exactly on a boundary (±2s), run now; else wait for next boundary.
        boundary = n.replace(second=0, microsecond=0)
        on_boundary = (boundary.minute % STEP_MIN == 0) and (abs(n.second) <= 2)

        if not on_boundary:
            nb = next_boundary(n)
            if nb > end:
                nb = end
            print(f"[INFO] Next run at {nb.strftime('%Y-%m-%d %H:%M:%S%z')}")
            # Sleep slightly past boundary to ensure the new candle has closed and is available
            sleep_until(min(nb + timedelta(seconds=2), session_end))
            continue

        try:
            print(f"[RUN ] Updating 15m at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
            run_update_15m(holidays)
        except Exception as e:
            print(f"[ERR ] Update failed: {e}")
            _time.sleep(3)


if __name__ == "__main__":
    main()
