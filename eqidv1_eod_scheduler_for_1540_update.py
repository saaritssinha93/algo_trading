# -*- coding: utf-8 -*-
"""
eqidv1_eod_scheduler_for_1540_update.py
========================================

Runs once per trading day at 15:40 IST to perform a final flush of
EQIDV1 intraday parquet data (15-minute and 5-minute) with all of
today's candles, ensuring nothing is stale heading into the next session.

Adapted from stocks_eod_daily_weekly_scheduler_for_daily_1540_update.py.
Since the eqidv1 core only supports intraday modes (5min, 15min), this
replaces the daily+weekly update with a final intraday catch-up.

Usage:
- Schedule via Windows Task Scheduler or batch file at ~15:30 IST.
- At 15:40 it runs 15min then 5min update with skip_if_fresh=False.
- Process exits at 16:00 IST.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from datetime import datetime, time as dtime, date as ddate

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

RUN_AT      = dtime(15, 40)    # trigger time
SESSION_END = dtime(16, 0)     # hard stop

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
    FIX: only take the first non-empty token from the files.
    """
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE).split()[0]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


core.setup_kite_session = setup_kite_session_fixed


# =============================================================================
# HELPERS
# =============================================================================
def now_ist() -> datetime:
    return datetime.now(IST)


def dt_today_at(t: dtime) -> datetime:
    n = now_ist()
    return IST.localize(datetime(n.year, n.month, n.day, t.hour, t.minute, 0))


def sleep_until(target: datetime) -> None:
    while True:
        n = now_ist()
        secs = (target - n).total_seconds()
        if secs <= 0:
            return
        _time.sleep(min(secs, 5.0))


# =============================================================================
# EOD UPDATER
# =============================================================================
def run_eod_updates(holidays: set[ddate]) -> None:
    """
    Final intraday flush: update both 15min and 5min parquet with all
    of today's candles. Uses skip_if_fresh=False to force a complete
    re-evaluation at end of day.
    """
    for mode in ("15min", "5min"):
        print(f"[RUN ] Updating {mode.upper()} ...")
        core.run_mode(
            mode=mode,
            max_workers=4,
            skip_if_fresh=False,       # force update at EOD
            intraday_ts="end",
            holidays=holidays,
            refresh_tokens=False,
            report_dir=str(REPORT_DIR),
            print_missing_rows=False,
            print_missing_rows_max=200,
        )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    try:
        holidays = set(core._read_holidays(core.HOLIDAYS_FILE_DEFAULT))
    except Exception:
        holidays = set()

    print("[EOD] EQIDV1 1540 scheduler started.")
    print("      Runs once per trading day at 15:40 IST (15min + 5min final flush).")
    print("      Process will exit at 16:00 IST.")

    last_run_date = None  # prevents accidental double-run same day

    while True:
        n = now_ist()
        session_end = dt_today_at(SESSION_END)

        # Hard stop
        if n >= session_end:
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # Non-trading day => exit
        if not core._is_trading_day(n.date(), holidays):
            print("[INFO] Non-trading day. Exiting.")
            return

        target = dt_today_at(RUN_AT)

        # Before 15:40 -> wait
        if n < target:
            print(f"[INFO] Next EOD run at {target.strftime('%Y-%m-%d %H:%M:%S%z')}")
            sleep_until(min(target, session_end))
            continue

        # After 15:40 -> if not run today, run now
        if last_run_date != n.date():
            try:
                print(f"[RUN ] EOD update starting at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
                run_eod_updates(holidays)
                last_run_date = n.date()
                print("[DONE] EOD update complete.")
            except Exception as e:
                print(f"[ERR ] EOD update failed: {e}")
                _time.sleep(5)

        # Wait until session end, then exit
        if now_ist() < session_end:
            print(f"[INFO] Sleeping until session end: {session_end.strftime('%Y-%m-%d %H:%M:%S%z')}")
            sleep_until(session_end)

        print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
        return


if __name__ == "__main__":
    main()
