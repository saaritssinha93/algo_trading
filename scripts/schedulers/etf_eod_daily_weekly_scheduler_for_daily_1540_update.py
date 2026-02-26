# etf_eod_daily_weekly_scheduler_for_daily_1540_update.py
# Runs once per trading day at 15:40 IST:
#   - updates DAILY parquet + indicators
#   - updates WEEKLY parquet + indicators
#
# Assumes you have:
#   algosm1_trading_data_continous_run_historical_alltf_v3_parquet_etfsonly.py
# in the same folder (imported as `core` below)

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, time as dtime
import pytz

from kiteconnect import KiteConnect

import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_etfsonly as core


IST = pytz.timezone("Asia/Kolkata")

RUN_AT = dtime(15, 50)   # 15:40 IST daily
# NEW: hard stop for this process run
SESSION_END = dtime(16, 00)

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
    FIX: only take the first non-empty line from the files (no embedded newlines),
    preventing 'InvalidHeader' errors.
    """
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# Monkeypatch core so it uses the fixed session loader everywhere
core.setup_kite_session = setup_kite_session_fixed


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


def run_eod_updates(holidays: set) -> None:
    """
    Update DAILY + WEEKLY using the same indicator pipeline in `core`.
    """
    print("[RUN ] Updating DAILY .")
    core.run_mode(
        mode="daily",               # FIX: was "day"
        context="live",
        max_workers=4,
        force_today_daily=True,     # end-of-day: include today's completed day
        skip_if_fresh=True,
        intraday_ts="end",
        holidays=holidays,
        refresh_tokens=False,
        report_dir="etf_missing_reports",
        print_missing_rows=False,
        print_missing_rows_max=200,
    )

    print("[RUN ] Updating WEEKLY .")
    core.run_mode(
        mode="weekly",              # FIX: was "week"
        context="live",
        max_workers=4,
        force_today_daily=False,
        skip_if_fresh=True,
        intraday_ts="end",
        holidays=holidays,
        refresh_tokens=False,
        report_dir="etf_missing_reports",
        print_missing_rows=False,
        print_missing_rows_max=200,
    )


def main() -> None:
    holidays = core._read_holidays(core.HOLIDAYS_FILE_DEFAULT)

    print("[EOD] Daily+Weekly scheduler started.")
    print("      Runs once per trading day at 15:40 IST.")
    print("      Process will exit at 15:50 IST.")

    last_run_date = None  # prevents accidental double-run same day

    while True:
        n = now_ist()
        session_end = dt_today_at(SESSION_END)

        # NEW: stop this run at 15:50 IST
        if n >= session_end:
            print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
            return

        # If non-trading day, for a scheduled run just exit
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

        # NEW: do not sleep until next day; end this run at 15:50
        if now_ist() < session_end:
            print(f"[INFO] Sleeping until session end: {session_end.strftime('%Y-%m-%d %H:%M:%S%z')}")
            sleep_until(session_end)

        print(f"[EXIT] Session end reached ({session_end.strftime('%Y-%m-%d %H:%M:%S%z')}).")
        return


if __name__ == "__main__":
    main()
