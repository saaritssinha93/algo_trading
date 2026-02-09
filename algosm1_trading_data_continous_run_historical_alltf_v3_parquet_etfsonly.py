# -*- coding: utf-8 -*-
"""
Unified Zerodha (KiteConnect) ETF data fetch + indicator generator for multiple timeframes (Parquet storage).

CHANGES DONE (as requested):
0) CSV -> PARQUET storage for indicator outputs (and missing-rows reports).
1) STOCK -> ETF universe loader:
   - Tries to import: etf_filtered_etfs_all.py
     Supports:
       - etf_tokens = {ETFSYMBOL: TOKEN, ...}   OR
       - selected_etfs = [ETFSYMBOL, ...] / {ETFSYMBOL, ...} / {ETFSYMBOL: TOKEN, ...}
   - Fallback file: etf_tickers.txt

2) Directories renamed to ETF-specific:
   - Cache dirs: etf_cache_<tf>
   - Output dirs: etf_indicators_<tf>
   - Reports default: etf_missing_reports
   - Token cache file: etf_tokens_cache.json
   - Log file: etf_fetcher_run.log

Behavior remains the same:
- Fetch & print ONLY missing files and missing evaluation rows.
- Per-ticker freshness skip is retained.
- Indicators and fetching logic unchanged.
"""

import os
import sys
import time as _time
import json
import re
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteConnect, exceptions as kexc

# ========= GLOBAL CONFIG =========

IST_TZ = pytz.timezone("Asia/Kolkata")

# ETF-specific directories
DIRS = {
    "daily":  {"cache": "etf_cache_daily_pq",  "out": "etf_indicators_daily_pq"},
    "weekly": {"cache": "etf_cache_weekly_pq", "out": "etf_indicators_weekly_pq"},
    "1h":     {"cache": "etf_cache_1h_pq",     "out": "etf_indicators_1h_pq"},
    "3h":     {"cache": "etf_cache_3h_pq",     "out": "etf_indicators_3h_pq"},
    "5min":   {"cache": "etf_cache_5min_pq",   "out": "etf_indicators_5min_pq"},
    "15min":  {"cache": "etf_cache_15min_pq",  "out": "etf_indicators_15min_pq"},
}
for cfg in DIRS.values():
    os.makedirs(cfg["cache"], exist_ok=True)
    os.makedirs(cfg["out"], exist_ok=True)

VALID_MODES = ("daily", "weekly", "1h", "3h", "5min", "15min")
DEFAULT_MAX_WORKERS = 6

# Market timing (IST)
MARKET_OPEN_TIME = time(9, 15)
MARKET_CLOSE_TIME_INTRADAY = time(15, 30)      # last intraday candle end (5m/15m)
MARKET_CLOSE_TIME_DAILY_READY = time(15, 35)   # daily bar safely "final"

# Candle-end timestamps for intraday
DEFAULT_INTRADAY_TIMESTAMP = "end"  # "end" or "start"

# Incremental fetch warmup bars (to re-stabilize indicators near the tail)
WARMUP_BARS = {
    "5min":  600,
    "15min": 400,
    "1h":    300,
    "3h":    300,
    "daily": 260,
    "weekly": 120,
}

# Token cache (ETF-specific)
TOKENS_CACHE_FILE = "etf_tokens_cache.json"
TOKENS_CACHE_MAX_AGE_DAYS = 7

# Optional NSE holidays file (one date per line or CSV column "date")
HOLIDAYS_FILE_DEFAULT = "nse_holidays.csv"

# ========= STORAGE (PARQUET) =========
# Main indicator outputs are stored as Parquet:
#   etf_indicators_<tf> / <TICKER>_etf_indicators_<tf>.parquet
#
# For smooth migration, the script can read legacy CSV outputs (if present) and will
# start writing Parquet going forward.
MIGRATE_LEGACY_CSV = True        # can be disabled via CLI: --no-migrate-csv
DELETE_LEGACY_CSV = False        # can be enabled via CLI: --delete-legacy-csv

# Add near HOLIDAYS_FILE_DEFAULT
SPECIAL_TRADING_DAYS_FILE_DEFAULT = "nse_special_trading_days.csv"  # one date per line or CSV col "date"

def _read_special_trading_days(path: str) -> set[date]:
    """Read special trading days (e.g. weekend sessions) from file and merge hard-coded dates."""
    days = _read_holidays(path)  # parses date lines/csv (same style as holidays)
    days.update(HARD_CODED_SPECIAL_OPEN_DAYS)
    return days




# ========= LOGGING =========

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("etf_fetcher")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler("etf_fetcher_run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ========= ETF UNIVERSE =========

import importlib
from types import ModuleType
from typing import Optional

def _normalize_ticker_list(obj) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        arr = list(obj.keys())
    elif isinstance(obj, (set, list, tuple)):
        arr = list(obj)
    else:
        if isinstance(obj, str):
            arr = re.split(r"[\s,;]+", obj.strip())
        else:
            try:
                arr = list(obj)  # type: ignore
            except Exception:
                arr = [obj]

    tickers: list[str] = []
    for x in arr:
        s = str(x).strip().upper()
        if not s:
            continue
        s = s.replace("NSE:", "").replace("BSE:", "")
        tickers.append(s)

    return sorted(set(tickers))


def load_etf_universe(logger: logging.Logger) -> tuple[list[str], dict[str, int]]:
    """
    ETF universe loader:
    - Preferred: etf_filtered_etfs_all.py with:
        etf_tokens OR selected_etfs
    - Fallback: etf_tickers.txt (one ETF symbol per line)
    """
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    for p in (str(script_dir), str(parent_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    token_map: dict[str, int] = {}
    mod: Optional[ModuleType] = None

    # Primary ETF module
    try:
        mod = importlib.import_module("etf_filtered_etfs_all")
    except Exception:
        mod = None

    if mod is not None:
        # Option A: etf_tokens
        if hasattr(mod, "etf_tokens") and isinstance(getattr(mod, "etf_tokens"), dict):
            raw = getattr(mod, "etf_tokens")
            try:
                token_map = {str(k).strip().upper(): int(v) for k, v in raw.items() if str(k).strip()}
                tickers = sorted(token_map.keys())
                if tickers:
                    logger.info("Loaded %d ETFs from etf_filtered_etfs.etf_tokens", len(tickers))
                    return tickers, token_map
            except Exception:
                pass

        # Option B: selected_etfs
        if hasattr(mod, "selected_etfs"):
            ss = getattr(mod, "selected_etfs")
            if isinstance(ss, dict):
                tickers = _normalize_ticker_list(ss)
                try:
                    if ss and all(isinstance(v, (int, float)) for v in ss.values()):
                        token_map = {str(k).strip().upper(): int(v) for k, v in ss.items() if str(k).strip()}
                        tickers = sorted(token_map.keys())
                except Exception:
                    pass
                if tickers:
                    logger.info("Loaded %d ETFs from etf_filtered_etfs_all.selected_etfs (dict/set-like)", len(tickers))
                    return tickers, token_map

            tickers = _normalize_ticker_list(ss)
            if tickers:
                logger.info("Loaded %d ETFs from etf_filtered_etfs_all.selected_etfs", len(tickers))
                return tickers, token_map

    # Fallback file: etf_tickers.txt
    for base in (cwd, script_dir, parent_dir):
        f = base / "etf_tickers.txt"
        if f.exists():
            arr = [x.strip().upper() for x in f.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            tickers = _normalize_ticker_list(arr)
            if tickers:
                logger.info("Loaded %d ETFs from %s", len(tickers), str(f))
                return tickers, token_map

    raise RuntimeError(
        "Could not load ETF symbols.\n"
        "Fix options:\n"
        "  1) Ensure etf_filtered_etfs_all.py is importable and define either:\n"
        "       - etf_tokens = {ETFSYMBOL: TOKEN, ...}   OR\n"
        "       - selected_etfs = [ETFSYMBOL, ...] / {ETFSYMBOL, ...} / {ETFSYMBOL: TOKEN, ...}\n"
        "  2) Or create etf_tickers.txt (one ETF symbol per line) in cwd / script dir / parent dir.\n\n"
        f"Diagnostics:\n  cwd={cwd}\n  script_dir={script_dir}\n  parent_dir={parent_dir}"
    )


# ========= KITE SESSION =========

def setup_kite_session() -> KiteConnect:
    with open("access_token.txt", "r", encoding="utf-8") as f:
        access_token = f.read().strip()
    with open("api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().split()[0]
    kite_local = KiteConnect(api_key=api_key)
    kite_local.set_access_token(access_token)
    return kite_local


# ========= HOLIDAYS =========

def _read_holidays(path: str) -> set[date]:
    holidays: set[date] = set()
    if not path or not os.path.exists(path):
        return holidays

    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            if "date" in df.columns:
                ds = pd.to_datetime(df["date"], errors="coerce").dropna()
                holidays.update(ds.dt.date.tolist())
            else:
                ds = pd.to_datetime(df.iloc[:, 0], errors="coerce").dropna()
                holidays.update(ds.dt.date.tolist())
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    d = pd.to_datetime(s, errors="coerce")
                    if pd.notna(d):
                        holidays.add(d.date())
    except Exception:
        return set()

    return holidays


# ========= TRADING CALENDAR HELPERS =========

# Hard-coded special trading sessions (weekend trading days).
# Keep this in sync with your live signal scripts.
HARD_CODED_SPECIAL_OPEN_DAYS: set[date] = {
    date(2026, 2, 1),  # Special trading Sunday
}

def _is_trading_day(d: date, holidays: set[date], special_open: set[date] | None = None) -> bool:
    # Merge optional file-based specials with hard-coded specials
    merged_specials = set(HARD_CODED_SPECIAL_OPEN_DAYS)
    if special_open:
        merged_specials.update(special_open)

    # Allow special sessions (even on weekends)
    if d in merged_specials:
        return True

    # Normal calendar: Mon-Fri, excluding holidays
    if d.weekday() >= 5:
        return False
    if d in holidays:
        return False
    return True


def _prev_trading_day(d: date, holidays: set[date], special_open: set[date] | None = None) -> date:
    x = d - timedelta(days=1)
    while not _is_trading_day(x, holidays, special_open):
        x -= timedelta(days=1)
    return x


def last_friday_strictly_before(d: date) -> date:
    dd = d - timedelta(days=1)
    while dd.weekday() != 4:
        dd -= timedelta(days=1)
    return dd

def monday_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())

def _round_down_session_anchored(ts: datetime, step_min: int) -> datetime:
    if ts.tzinfo is None:
        ts = IST_TZ.localize(ts)

    anchor = IST_TZ.localize(datetime(ts.year, ts.month, ts.day, 9, 15, 0))
    if ts <= anchor:
        return anchor.replace(second=0, microsecond=0)

    delta_min = int((ts - anchor).total_seconds() // 60)
    steps = max(0, delta_min // step_min)
    out = anchor + timedelta(minutes=steps * step_min)
    return out.replace(second=0, microsecond=0)

def last_completed_intraday_end(now_ist: datetime, step_min: int, holidays: set[date]) -> datetime:
    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    d = now_ist.date()

    if not _is_trading_day(d, holidays):
        d = _prev_trading_day(d, holidays)
        close_dt = IST_TZ.localize(datetime(d.year, d.month, d.day, 15, 30, 0))
        return _round_down_session_anchored(close_dt, step_min)

    if now_ist.time() < MARKET_OPEN_TIME:
        d = _prev_trading_day(d, holidays)
        close_dt = IST_TZ.localize(datetime(d.year, d.month, d.day, 15, 30, 0))
        return _round_down_session_anchored(close_dt, step_min)

    if now_ist.time() >= MARKET_CLOSE_TIME_INTRADAY:
        close_dt = IST_TZ.localize(datetime(d.year, d.month, d.day, 15, 30, 0))
        return _round_down_session_anchored(close_dt, step_min)

    return _round_down_session_anchored(now_ist, step_min)

def daily_cutoff_date_live(now_ist: datetime, holidays: set[date], force_today: bool) -> date:
    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    d = now_ist.date()
    if not _is_trading_day(d, holidays):
        return _prev_trading_day(d, holidays)

    if force_today:
        return d

    if now_ist.time() >= MARKET_CLOSE_TIME_DAILY_READY:
        return d
    return _prev_trading_day(d, holidays)

def daily_cutoff_date_previous(now_ist: datetime, holidays: set[date]) -> date:
    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)
    return _prev_trading_day(now_ist.date(), holidays)

def weekly_cutoff_date_previous(now_ist: datetime) -> date:
    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)
    return last_friday_strictly_before(now_ist.date())

def get_end_dt_for_mode(mode: str, now_ist: datetime, context: str, holidays: set[date], force_today_daily: bool) -> datetime:
    context = context.lower().strip()
    mode = mode.lower().strip()

    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    if context == "previous":
        if mode == "weekly":
            d = weekly_cutoff_date_previous(now_ist)
            return IST_TZ.localize(datetime(d.year, d.month, d.day, 23, 59, 59))
        else:
            d = daily_cutoff_date_previous(now_ist, holidays)
            return IST_TZ.localize(datetime(d.year, d.month, d.day, 23, 59, 59))

    if mode == "5min":
        return last_completed_intraday_end(now_ist, 5, holidays)
    if mode == "15min":
        return last_completed_intraday_end(now_ist, 15, holidays)
    if mode == "1h":
        return last_completed_intraday_end(now_ist, 60, holidays)
    if mode == "3h":
        return last_completed_intraday_end(now_ist, 60, holidays)
    if mode == "daily":
        d = daily_cutoff_date_live(now_ist, holidays, force_today_daily)
        return IST_TZ.localize(datetime(d.year, d.month, d.day, 23, 59, 59))
    if mode == "weekly":
        return now_ist

    return now_ist


# ========= START DATE PER MODE =========

def get_start_date(mode: str, now_ist: datetime) -> datetime:
    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    # Keep your original behaviour for intraday start anchor
    if mode in ("5min", "15min"):
        return IST_TZ.localize(datetime(2025, 8, 25, 0, 0, 0))

    base = now_ist - timedelta(days=365)
    if mode in ("1h", "3h"):
        return base.replace(minute=0, second=0, microsecond=0)
    return base.replace(hour=0, minute=0, second=0, microsecond=0)


# ========= DATA HELPERS =========

def _to_ist(series_dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series_dt, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize(IST_TZ)
    return dt.dt.tz_convert(IST_TZ)

def _read_last_ts_fast_csv(path: str):
    """Legacy CSV tail reader (kept only for migration)."""
    try:
        with open(path, "rb") as f:
            header = f.readline().decode("utf-8", errors="ignore").strip()
            if not header:
                return None
            cols = [c.strip().strip('"') for c in header.split(",")]
            if "date" not in cols:
                return None
            date_idx = cols.index("date")

            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            if file_size <= 0:
                return None

            block = 8192
            data = b""
            offset = 0
            while file_size - offset > 0 and len(data.splitlines()) < 8:
                read_sz = min(block, file_size - offset)
                offset += read_sz
                f.seek(file_size - offset)
                chunk = f.read(read_sz)
                data = chunk + data

            lines = data.splitlines()
            for raw in reversed(lines):
                if not raw.strip():
                    continue
                try:
                    line = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) <= date_idx:
                    continue
                ds = parts[date_idx]
                ts = pd.to_datetime(ds, errors="coerce")
                if pd.isna(ts):
                    continue
                if ts.tzinfo is None:
                    ts = ts.tz_localize(IST_TZ)
                else:
                    ts = ts.tz_convert(IST_TZ)
                return ts
    except Exception:
        return None
    return None


def _ensure_parquet_engine():
    """Parquet I/O requires an engine. We use pyarrow."""
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Parquet storage requires 'pyarrow'.\n"
            "Install it once:  pip install pyarrow\n"
            f"Original import error: {e}"
        ) from e


def _read_last_ts_fast_parquet(path: str):
    """
    Read the last 'date' value efficiently from a Parquet file using pyarrow metadata/row-groups.
    Returns pd.Timestamp localized/converted to IST, or None.
    """
    try:
        _ensure_parquet_engine()
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        md = pf.metadata
        if md is None or md.num_rows <= 0:
            return None

        # Read only the last row-group's 'date' column.
        last_rg = md.num_row_groups - 1
        if last_rg < 0:
            return None

        table = pf.read_row_group(last_rg, columns=["date"])
        if table.num_rows <= 0:
            return None

        # Convert the last value to pandas Timestamp
        col = table.column(0)
        # pyarrow scalar -> python datetime / pandas Timestamp
        val = col[col.length() - 1].as_py()
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None

        if ts.tzinfo is None:
            ts = ts.tz_localize(IST_TZ)
        else:
            ts = ts.tz_convert(IST_TZ)
        return ts
    except Exception:
        return None


def _read_last_ts_from_store(path: str):
    """Read last timestamp from either parquet (preferred) or legacy csv."""
    ext = str(Path(path).suffix).lower()
    if ext == ".parquet":
        return _read_last_ts_fast_parquet(path)
    if ext == ".csv":
        return _read_last_ts_fast_csv(path)
    # Fallback: try parquet first, then csv
    ts = _read_last_ts_fast_parquet(path)
    if ts is not None:
        return ts
    return _read_last_ts_fast_csv(path)
def _intraday_end_shift_minutes(interval: str) -> int:
    return {"5minute": 5, "15minute": 15, "60minute": 60}.get(interval, 0)

def _maybe_convert_existing_intraday_to_end(df: pd.DataFrame, step_min: int) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    s = pd.to_datetime(df["date"], errors="coerce")
    if s.isna().all():
        return df
    s = _to_ist(s)
    min_ts = s.min()
    if (min_ts.hour, min_ts.minute) == (9, 15):
        s = s + pd.Timedelta(minutes=step_min)
        df = df.copy()
        df["date"] = s
    return df


# ========= PER-TICKER FRESHNESS (FAST SKIP) =========

def expected_last_stamp(mode: str, now_ist: datetime, context: str, holidays: set[date],
                        force_today_daily: bool, intraday_ts: str) -> dict:
    mode = mode.lower().strip()
    context = context.lower().strip()

    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    if mode == "daily":
        cutoff = daily_cutoff_date_previous(now_ist, holidays) if context == "previous" else \
                 daily_cutoff_date_live(now_ist, holidays, force_today_daily)
        return {"kind": "date", "value": cutoff}

    if mode == "weekly":
        if context == "previous":
            cutoff_friday = weekly_cutoff_date_previous(now_ist)
            wk_start = monday_of_week(cutoff_friday)
        else:
            wk_start = monday_of_week(now_ist.date())
        return {"kind": "date", "value": wk_start}

    step_map = {"5min": 5, "15min": 15, "1h": 60, "3h": 60}
    step = step_map.get(mode, 0)

    if step <= 0:
        return {"kind": "ts", "value": now_ist}

    if context == "previous":
        prev_d = daily_cutoff_date_previous(now_ist, holidays)
        ref = IST_TZ.localize(datetime(prev_d.year, prev_d.month, prev_d.day, 15, 30, 0))
        exp_end = last_completed_intraday_end(ref, step, holidays)
    else:
        exp_end = last_completed_intraday_end(now_ist, step, holidays)

    if mode in ("5min", "15min", "1h"):
        if intraday_ts.lower() == "start":
            exp_end = exp_end - timedelta(minutes=step)
        return {"kind": "ts", "value": exp_end, "step_min": step}

    return {"kind": "ts", "value": exp_end, "step_min": step}

def ticker_is_fresh(mode: str, out_path: str, now_ist: datetime, context: str,
                    holidays: set[date], force_today_daily: bool, intraday_ts: str) -> bool:
    # out_path is the TARGET parquet path; we may read legacy CSV if enabled.
    existing_path = _resolve_existing_store_path(out_path)
    if not os.path.exists(existing_path):
        return False

    last_ts = _read_last_ts_from_store(existing_path)
    if last_ts is None:
        return False

    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(IST_TZ)
    else:
        last_ts = last_ts.tz_convert(IST_TZ)

    spec = expected_last_stamp(mode, now_ist, context, holidays, force_today_daily, intraday_ts)

    if spec.get("kind") == "date":
        exp_d: date = spec["value"]
        return last_ts.date() >= exp_d

    exp_ts: datetime = spec["value"]
    if exp_ts.tzinfo is None:
        exp_ts = IST_TZ.localize(exp_ts)

    tol = timedelta(seconds=1)
    step_min = int(spec.get("step_min", 0) or 0)
    step_td = timedelta(minutes=step_min) if step_min > 0 else timedelta(0)

    if last_ts >= (exp_ts - tol):
        return True
    if step_min > 0:
        if (last_ts + step_td) >= (exp_ts - tol):
            return True
        if (last_ts - step_td) >= (exp_ts - tol):
            return True

    return False


def missing_spec(mode: str, out_path: str, now_ist: datetime, context: str,
                 holidays: set[date], force_today_daily: bool, intraday_ts: str) -> dict:
    # out_path is the TARGET parquet path; we may read legacy CSV if enabled.
    existing_path = _resolve_existing_store_path(out_path)

    if not os.path.exists(existing_path):
        spec = expected_last_stamp(mode, now_ist, context, holidays, force_today_daily, intraday_ts)
        return {"kind": "file_missing", "last_ts": None, "expected": spec}

    last_ts = _read_last_ts_from_store(existing_path)
    if last_ts is None:
        spec = expected_last_stamp(mode, now_ist, context, holidays, force_today_daily, intraday_ts)
        return {"kind": "rows_missing", "last_ts": None, "expected": spec}

    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(IST_TZ)
    else:
        last_ts = last_ts.tz_convert(IST_TZ)

    spec = expected_last_stamp(mode, now_ist, context, holidays, force_today_daily, intraday_ts)

    if ticker_is_fresh(mode, out_path, now_ist, context, holidays, force_today_daily, intraday_ts):
        return {"kind": "fresh", "last_ts": last_ts, "expected": spec}

    return {"kind": "rows_missing", "last_ts": last_ts, "expected": spec}




# ========= FETCHERS =========

def fetch_historical_generic(kite: KiteConnect, token: int, start_dt_ist: datetime, end_dt_ist: datetime, interval: str,
                            chunk_days: int, step_td: timedelta, logger: logging.Logger,
                            intraday_ts: str) -> pd.DataFrame:
    end = end_dt_ist if end_dt_ist.tzinfo else IST_TZ.localize(end_dt_ist)
    s = start_dt_ist if start_dt_ist.tzinfo else IST_TZ.localize(start_dt_ist)

    chunk = timedelta(days=chunk_days)
    frames = []

    MAX_RETRIES = 3
    SLEEP_BETWEEN_CALLS = 0.35

    while s < end:
        e = min(s + chunk, end)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = kite.historical_data(token, s, e, interval)
                df = pd.DataFrame(raw)
                if df.empty:
                    break

                df["date"] = _to_ist(df["date"])

                if intraday_ts.lower() == "end":
                    shift_min = _intraday_end_shift_minutes(interval)
                    if shift_min > 0:
                        df["date"] = df["date"] + pd.Timedelta(minutes=shift_min)

                frames.append(df)
                break
            except (kexc.NetworkException, kexc.DataException, kexc.TokenException, kexc.InputException) as ex:
                if attempt == MAX_RETRIES:
                    logger.warning("Failed chunk %s → %s (%s): %s", s, e, interval, ex)
                else:
                    _time.sleep(1.0 * attempt)
            finally:
                _time.sleep(SLEEP_BETWEEN_CALLS)

        s = e + step_td

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset="date")
          .sort_values("date")
          .reset_index(drop=True)
    )
    out = out[out["date"] <= end_dt_ist].reset_index(drop=True)
    return out

def fetch_historical_daily_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "day", 180, timedelta(days=1), logger, intraday_ts)

def fetch_historical_weekly_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "week", 365, timedelta(days=7), logger, intraday_ts)

def fetch_historical_1h_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "60minute", 60, timedelta(hours=1), logger, intraday_ts)

def fetch_historical_5min_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "5minute", 60, timedelta(minutes=5), logger, intraday_ts)

def fetch_historical_15min_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "15minute", 120, timedelta(minutes=15), logger, intraday_ts)


# ========= INDICATORS (unchanged) =========

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(close, period=20, up=2, dn=2):
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return sma + up * std, sma - dn * std

def _ma(x, window, kind="sma"):
    if kind == "ema":
        return x.ewm(span=window, adjust=False).mean()
    return x.rolling(window, min_periods=window).mean()

def calculate_stochastic_fast(df, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = high_max - low_min
    k = pd.Series(0.0, index=df.index)
    valid = rng > 0
    k.loc[valid] = 100.0 * (df["close"].loc[valid] - low_min.loc[valid]) / rng.loc[valid]
    k = k.clip(0.0, 100.0)
    d = k.rolling(d_period, min_periods=d_period).mean().clip(0.0, 100.0)
    return k, d

def calculate_stochastic_slow(df, k_period=14, k_smooth=3, d_period=3, ma_kind="sma"):
    k_fast, _ = calculate_stochastic_fast(df, k_period=k_period, d_period=1)
    k_slow = _ma(k_fast, k_smooth, kind=ma_kind).clip(0.0, 100.0)
    d = _ma(k_slow, d_period, kind=ma_kind).clip(0.0, 100.0)
    return k_slow, d

def calculate_adx(df, period=14):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.Series(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy(),
        ]),
        index=df.index,
    )

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    eps = 1e-10
    plus_di = 100.0 * (plus_dm_sm / (atr + eps))
    minus_di = 100.0 * (minus_dm_sm / (atr + eps))
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + eps)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.clip(0, 100)

def calculate_vwap(df):
    return (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)

def calculate_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def calculate_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)

def calculate_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp.diff() > 0, 0)
    neg_mf = mf.where(tp.diff() < 0, 0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum().abs()
    return 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def add_standard_indicators(df):
    df["RSI"] = calculate_rsi(df["close"])
    df["ATR"] = calculate_atr(df)
    df["EMA_50"] = calculate_ema(df["close"], 50)
    df["EMA_200"] = calculate_ema(df["close"], 200)
    df["20_SMA"] = df["close"].rolling(20, min_periods=20).mean()
    df["VWAP"] = calculate_vwap(df)
    df["CCI"] = calculate_cci(df)
    df["MFI"] = calculate_mfi(df)
    df["OBV"] = calculate_obv(df)

    macd, macd_sig, macd_hist = calculate_macd(df["close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"] = macd_hist

    df["Upper_Band"], df["Lower_Band"] = calculate_bollinger_bands(df["close"])
    return df


# ========= 3H RESAMPLE =========

def resample_to_3h(df_60m: pd.DataFrame) -> pd.DataFrame:
    if df_60m.empty:
        return df_60m

    df = df_60m.set_index("date").sort_index()

    df_shift = df.copy()
    df_shift.index = df_shift.index - pd.Timedelta(minutes=15)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_3h = df_shift.resample("3h", label="right", closed="right").agg(agg)

    df_3h.index = df_3h.index + pd.Timedelta(minutes=15)
    df_3h = df_3h.dropna(subset=["open", "high", "low", "close"])

    keep_times = {(9, 15), (12, 15), (15, 15)}
    mask = df_3h.index.map(lambda ts: (ts.hour, ts.minute) in keep_times)
    df_3h = df_3h[mask]

    return df_3h.reset_index().rename(columns={"index": "date"})


# ========= CHANGE FEATURES =========

def add_change_features_intraday(df: pd.DataFrame) -> pd.DataFrame:
    df["date_only"] = df["date"].dt.tz_convert(IST_TZ).dt.date
    df["Intra_Change"] = df.groupby("date_only")["close"].pct_change().mul(100.0)

    last_close_per_day = df.groupby("date_only", sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)

    df["Prev_Day_Close"] = df["date_only"].map(prev_day_last_close)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"] + 1e-10) * 100.0
    return df

def add_change_features_daily_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df["Intra_Change"] = df["close"].pct_change().mul(100.0)
    df["Prev_Day_Close"] = df["close"].shift(1)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"] + 1e-10) * 100.0
    return df


# ========= TOKEN CACHE =========

def load_or_fetch_tokens(kite: KiteConnect, etfs: list[str], logger: logging.Logger, refresh: bool = False) -> dict[str, int]:
    etfs_u = sorted({t.upper().strip() for t in etfs if t.strip()})

    if (not refresh) and os.path.exists(TOKENS_CACHE_FILE):
        try:
            st = os.stat(TOKENS_CACHE_FILE)
            age_days = (datetime.now() - datetime.fromtimestamp(st.st_mtime)).days
            if age_days <= TOKENS_CACHE_MAX_AGE_DAYS:
                cache = json.loads(Path(TOKENS_CACHE_FILE).read_text(encoding="utf-8"))
                if isinstance(cache, dict) and all(t in cache for t in etfs_u):
                    return {t: int(cache[t]) for t in etfs_u}
        except Exception:
            pass

    logger.info("Fetching NSE instruments for ETF token map (this can take time)...")
    ins = pd.DataFrame(kite.instruments("NSE"))
    tokens = ins[ins["tradingsymbol"].isin(etfs_u)][["tradingsymbol", "instrument_token"]]
    mp = dict(zip(tokens["tradingsymbol"], tokens["instrument_token"]))

    try:
        existing = {}
        if os.path.exists(TOKENS_CACHE_FILE):
            existing = json.loads(Path(TOKENS_CACHE_FILE).read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        existing.update({k: int(v) for k, v in mp.items()})
        Path(TOKENS_CACHE_FILE).write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {t: int(mp[t]) for t in etfs_u if t in mp}


# ========= SAVE =========

def _finalize_and_save(df: pd.DataFrame, out_path: str):
    """Write dataframe to Parquet (preferred) or CSV (legacy / reports, if any)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = str(Path(out_path).suffix).lower()

    if ext == ".parquet":
        _ensure_parquet_engine()
        # Use snappy for good speed/size tradeoff
        df.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")
        return

    # Fallback/legacy
    df.to_csv(out_path, index=False)

# ========= INCREMENTAL LOAD + MERGE =========

def _legacy_csv_path_for(parquet_path: str) -> str:
    return str(Path(parquet_path).with_suffix(".csv"))

def _resolve_existing_store_path(target_parquet_path: str) -> str:
    """
    If parquet exists -> return parquet.
    Else if legacy CSV exists and migration enabled -> return CSV.
    Else return parquet (even if missing).
    """
    if os.path.exists(target_parquet_path):
        return target_parquet_path
    if MIGRATE_LEGACY_CSV:
        legacy = _legacy_csv_path_for(target_parquet_path)
        if os.path.exists(legacy):
            return legacy
    return target_parquet_path


def _load_existing_ohlc(out_path: str, intraday_ts: str, mode: str) -> pd.DataFrame:
    """
    Load existing OHLCV for incremental merge.
    'out_path' is expected to be the TARGET parquet path; we may read a legacy CSV if present.
    """
    existing_path = _resolve_existing_store_path(out_path)
    if not os.path.exists(existing_path):
        return pd.DataFrame()

    try:
        keep_cols = ["date", "open", "high", "low", "close", "volume"]
        ext = str(Path(existing_path).suffix).lower()

        if ext == ".parquet":
            _ensure_parquet_engine()
            df = pd.read_parquet(existing_path, columns=keep_cols, engine="pyarrow")
        else:
            df = pd.read_csv(existing_path)

        if df.empty or "date" not in df.columns:
            return pd.DataFrame()

        df["date"] = _to_ist(df["date"])

        # If legacy file stored intraday as candle-start, convert to candle-end for consistency
        if mode in ("5min", "15min", "1h") and intraday_ts.lower() == "end":
            step = {"5min": 5, "15min": 15, "1h": 60}[mode]
            df = _maybe_convert_existing_intraday_to_end(df, step)

        keep = [c for c in keep_cols if c in df.columns]
        return df[keep].drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _incremental_start_from_existing(mode: str, out_path: str, default_start: datetime) -> datetime:
    existing_path = _resolve_existing_store_path(out_path)
    if not os.path.exists(existing_path):
        return default_start

    last_ts = _read_last_ts_from_store(existing_path)
    if last_ts is None:
        return default_start

    warm = int(WARMUP_BARS.get(mode, 0))
    if warm <= 0:
        return default_start

    if mode == "5min":
        back = timedelta(minutes=5 * warm)
    elif mode == "15min":
        back = timedelta(minutes=15 * warm)
    elif mode == "1h":
        back = timedelta(hours=1 * warm)
    elif mode == "3h":
        back = timedelta(hours=1 * warm)
    elif mode == "daily":
        back = timedelta(days=warm)
    elif mode == "weekly":
        back = timedelta(days=7 * warm)
    else:
        back = timedelta(days=30)

    s = (last_ts - back)
    s = s.to_pydatetime() if isinstance(s, pd.Timestamp) else s
    if s.tzinfo is None:
        s = IST_TZ.localize(s)
    return max(default_start, s)


# ========= PER-ETF PIPELINES =========

def _compute_common_features(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = add_standard_indicators(df)

    if mode in ("daily", "weekly"):
        df["Recent_High"] = df["high"].rolling(20, min_periods=20).max()
        df["Recent_Low"] = df["low"].rolling(20, min_periods=20).min()
    else:
        df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
        df["Recent_Low"] = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    if mode in ("daily", "weekly"):
        df = add_change_features_daily_weekly(df)
    else:
        df = add_change_features_intraday(df)

    return df

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _fmt_expected(spec: dict) -> str:
    try:
        if spec.get("kind") == "date":
            return f"date>={spec['value']}"
        if spec.get("kind") == "ts":
            v = spec["value"]
            if isinstance(v, datetime):
                return f"ts>={v.strftime('%Y-%m-%d %H:%M:%S')}"
            return f"ts>={str(v)}"
    except Exception:
        pass
    return str(spec)

@dataclass
class UpdateReport:
    mode: str
    ticker: str
    status: str            # created|updated|noop|failed
    out_path: str
    existed_before: bool
    last_before: str | None
    expected: str | None
    new_rows_count: int
    new_first: str | None
    new_last: str | None
    new_rows_path: str | None

def process_ticker(mode: str, ticker: str, token: int, kite: KiteConnect,
                   start_dt_ist: datetime, end_dt_ist: datetime,
                   logger: logging.Logger, context: str, holidays: set[date],
                   force_today_daily: bool, skip_if_fresh: bool, intraday_ts: str,
                   report_dir: str, print_missing_rows: bool, print_missing_rows_max: int) -> UpdateReport:
    out_path = os.path.join(DIRS[mode]["out"], f"{ticker}_etf_indicators_{mode}.parquet")
    now_ist = datetime.now(IST_TZ)

    existing_path = _resolve_existing_store_path(out_path)
    existed_before = os.path.exists(existing_path)
    last_before_ts = _read_last_ts_from_store(existing_path) if existed_before else None
    if last_before_ts is not None:
        if last_before_ts.tzinfo is None:
            last_before_ts = last_before_ts.tz_localize(IST_TZ)
        else:
            last_before_ts = last_before_ts.tz_convert(IST_TZ)

    exp = expected_last_stamp(mode, now_ist, context, holidays, force_today_daily, intraday_ts)
    exp_str = _fmt_expected(exp)

    if skip_if_fresh and ticker_is_fresh(mode, out_path, now_ist, context, holidays, force_today_daily, intraday_ts):
        return UpdateReport(mode, ticker, "noop", out_path, existed_before,
                            last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                            exp_str, 0, None, None, None)

    inc_start = _incremental_start_from_existing(mode, out_path, start_dt_ist)
    if inc_start >= end_dt_ist:
        return UpdateReport(mode, ticker, "noop", out_path, existed_before,
                            last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                            exp_str, 0, None, None, None)

    existing = _load_existing_ohlc(out_path, intraday_ts, mode)

    try:
        if mode == "daily":
            fetched = fetch_historical_daily_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
        elif mode == "weekly":
            fetched = fetch_historical_weekly_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
        elif mode == "1h":
            fetched = fetch_historical_1h_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
        elif mode == "5min":
            fetched = fetch_historical_5min_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
        elif mode == "15min":
            fetched = fetch_historical_15min_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
        elif mode == "3h":
            df_60m = fetch_historical_1h_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
            if df_60m.empty:
                return UpdateReport(mode, ticker, "noop", out_path, existed_before,
                                    last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                                    exp_str, 0, None, None, None)
            fetched = resample_to_3h(df_60m)
        else:
            return UpdateReport(mode, ticker, "failed", out_path, existed_before,
                                last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                                exp_str, 0, None, None, None)
    except Exception as e:
        logger.exception("[%s] %s fetch failed: %s", mode.upper(), ticker, e)
        return UpdateReport(mode, ticker, "failed", out_path, existed_before,
                            last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                            exp_str, 0, None, None, None)

    if fetched is None or fetched.empty:
        return UpdateReport(mode, ticker, "noop", out_path, existed_before,
                            last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                            exp_str, 0, None, None, None)

    merged = fetched
    if not existing.empty:
        merged = (
            pd.concat([existing, fetched], ignore_index=True)
              .drop_duplicates(subset="date", keep="last")
              .sort_values("date")
              .reset_index(drop=True)
        )

    try:
        merged = _compute_common_features(merged, mode)

        if mode == "daily":
            cutoff_d = daily_cutoff_date_previous(now_ist, holidays) if context.lower() == "previous" \
                       else daily_cutoff_date_live(now_ist, holidays, force_today_daily)
            merged = merged[merged["date"].dt.date <= cutoff_d].reset_index(drop=True)

        _finalize_and_save(merged, out_path)


        # Optional: if we migrated from a legacy CSV, delete it after successful parquet write
        if DELETE_LEGACY_CSV and existed_before and str(existing_path).lower().endswith(".csv"):
            try:
                os.remove(existing_path)
            except Exception:
                pass

        if existed_before and last_before_ts is not None:
            new_rows = merged[merged["date"] > last_before_ts].copy()
        else:
            new_rows = merged.copy()

        new_rows_count = int(len(new_rows))
        new_first = None
        new_last = None
        if new_rows_count > 0:
            nf = pd.to_datetime(new_rows["date"], errors="coerce").dropna().min()
            nl = pd.to_datetime(new_rows["date"], errors="coerce").dropna().max()
            new_first = nf.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(nf) else None
            new_last = nl.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(nl) else None

        new_rows_path = None
        if new_rows_count > 0:
            rep_dir = os.path.join(report_dir, "missing_rows", mode)
            _safe_mkdir(rep_dir)
            new_rows_path = os.path.join(rep_dir, f"{ticker}_missing_rows_{mode}.parquet")
            _finalize_and_save(new_rows, new_rows_path)

            if print_missing_rows:
                show = new_rows.tail(print_missing_rows_max)
                logger.info("[%s] %s NEW ROWS (last %d):\n%s",
                            mode.upper(), ticker, min(print_missing_rows_max, len(show)),
                            show.to_string(index=False))

        status = "created" if not existed_before else ("updated" if new_rows_count > 0 else "noop")

        return UpdateReport(
            mode=mode,
            ticker=ticker,
            status=status,
            out_path=out_path,
            existed_before=existed_before,
            last_before=last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
            expected=exp_str,
            new_rows_count=new_rows_count,
            new_first=new_first,
            new_last=new_last,
            new_rows_path=new_rows_path
        )

    except Exception as e:
        logger.exception("[%s] %s indicator/save failed: %s", mode.upper(), ticker, e)
        return UpdateReport(mode, ticker, "failed", out_path, existed_before,
                            last_before_ts.strftime("%Y-%m-%d %H:%M:%S") if last_before_ts is not None else None,
                            exp_str, 0, None, None, None)


# ========= DRIVER =========

def run_mode(mode: str, context: str, max_workers: int, force_today_daily: bool,
             skip_if_fresh: bool, intraday_ts: str, holidays: set[date],
             refresh_tokens: bool, report_dir: str,
             print_missing_rows: bool, print_missing_rows_max: int):
    logger = logging.getLogger("etf_fetcher")
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected: {', '.join(VALID_MODES)}")

    now_ist = datetime.now(IST_TZ)
    start_dt = get_start_date(mode, now_ist)
    end_dt = get_end_dt_for_mode(mode, now_ist, context=context, holidays=holidays, force_today_daily=force_today_daily)

    logger.info("=== MODE=%s | CONTEXT=%s | intraday_ts=%s | Window: %s → %s (IST) ===",
                mode, context, intraday_ts, start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M"))

    if end_dt <= start_dt:
        logger.info("End cutoff <= start. Nothing to fetch for %s.", mode)
        return

    etfs, pre_token_map = load_etf_universe(logger)

    missing_files: list[str] = []
    missing_rows: list[str] = []
    fresh: list[str] = []

    if skip_if_fresh:
        for t in etfs:
            t = t.upper()
            out_path = os.path.join(DIRS[mode]["out"], f"{t}_etf_indicators_{mode}.parquet")
            ms = missing_spec(mode, out_path, now_ist, context, holidays, force_today_daily, intraday_ts)
            if ms["kind"] == "fresh":
                fresh.append(t)
            elif ms["kind"] == "file_missing":
                missing_files.append(t)
                missing_rows.append(t)
            else:
                missing_rows.append(t)
    else:
        missing_rows = [t.upper() for t in etfs]

    if skip_if_fresh:
        logger.info("[%s] Missing files: %d", mode.upper(), len(missing_files))
        if missing_files:
            rep_dir = os.path.join(report_dir, "missing_files")
            _safe_mkdir(rep_dir)
            miss_file_path = os.path.join(rep_dir, f"missing_files_{mode}.txt")
            Path(miss_file_path).write_text("\n".join(missing_files), encoding="utf-8")
            logger.info("[%s] Missing files list saved: %s", mode.upper(), miss_file_path)
            logger.info("[%s] Missing files sample: %s", mode.upper(), ", ".join(missing_files[:50]))

        logger.info("[%s] Missing evaluation rows (stale ETFs): %d", mode.upper(), len(missing_rows))
    else:
        logger.info("[%s] no-skip enabled => processing all ETFs: %d", mode.upper(), len(missing_rows))

    if not missing_rows:
        logger.info("[%s] Nothing missing — all ETFs fresh.", mode.upper())
        return

    kite = setup_kite_session()

    token_map = {k.upper(): int(v) for k, v in dict(pre_token_map).items()}
    need_tokens = [t for t in missing_rows if t.upper() not in token_map]

    if need_tokens:
        fetched = load_or_fetch_tokens(kite, need_tokens, logger, refresh=refresh_tokens)
        token_map.update({k.upper(): int(v) for k, v in fetched.items()})

    work_items = []
    for t in missing_rows:
        tok = token_map.get(t.upper())
        if not tok:
            logger.warning("No token for %s, skipping.", t)
            continue
        work_items.append((t.upper(), int(tok)))

    if not work_items:
        logger.info("No valid ETFs with tokens.")
        return

    logger.info("[%s] Processing ONLY missing ETFs=%d with max_workers=%d ...", mode.upper(), len(work_items), max_workers)

    updated_reports: list[UpdateReport] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_ticker,
                mode, tkr, tok, kite, start_dt, end_dt,
                logger, context, holidays, force_today_daily,
                skip_if_fresh, intraday_ts,
                report_dir, print_missing_rows, print_missing_rows_max
            ): tkr
            for (tkr, tok) in work_items
        }
        for fut in as_completed(futures):
            tkr = futures[fut]
            try:
                rep: UpdateReport = fut.result()
                if rep.status == "failed":
                    failed += 1
                if rep.status in ("created", "updated"):
                    updated_reports.append(rep)
            except Exception as e:
                failed += 1
                logger.exception("Worker crashed for %s (%s): %s", tkr, mode, e)

    if updated_reports:
        logger.info("[%s] Updated ETFs: %d", mode.upper(), len(updated_reports))
        for r in sorted(updated_reports, key=lambda x: x.ticker):
            logger.info(
                "[%s] %s %s | last_before=%s | expected=%s | new_rows=%d | new_range=%s → %s | new_rows_store=%s",
                mode.upper(),
                r.ticker,
                r.status,
                r.last_before,
                r.expected,
                r.new_rows_count,
                r.new_first,
                r.new_last,
                r.new_rows_path
            )
    else:
        logger.info("[%s] No new rows were appended (everything ended up noop).", mode.upper())

    if failed:
        logger.warning("[%s] Failed ETFs: %d (see etf_fetcher_run.log)", mode.upper(), failed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="all", help="daily|weekly|1h|3h|5min|15min|all")
    p.add_argument("--context", default="live", choices=["live", "previous"],
                   help="live fetches up to last completed candle; previous caps to prev trading day / prev week")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--force-today-daily", action="store_true",
                   help="Include today's daily candle even before market close (NOT recommended)")
    p.add_argument("--no-skip", action="store_true",
                   help="Disable freshness skip (will refetch/recompute even if fresh)")
    p.add_argument("--intraday-ts", default=DEFAULT_INTRADAY_TIMESTAMP, choices=["start", "end"],
                   help="Store intraday timestamps as candle start or candle end (end recommended)")
    p.add_argument("--holidays-file", default=HOLIDAYS_FILE_DEFAULT,
                   help="Optional NSE holidays file (CSV with 'date' or one date per line)")
    p.add_argument("--refresh-tokens", action="store_true",
                   help="Force refresh token cache (kite.instruments NSE)")


    p.add_argument("--no-migrate-csv", action="store_true",
                   help="Do NOT read legacy CSV outputs (Parquet-only).")
    p.add_argument("--delete-legacy-csv", action="store_true",
                   help="After successful Parquet write, delete legacy CSV outputs (if they exist).")

    # ETF-specific reporting default
    p.add_argument("--report-dir", default="etf_missing_reports",
                   help="Directory to write missing-files and missing-rows reports")
    p.add_argument("--print-missing-rows", action="store_true",
                   help="Print a small preview of newly appended rows per ETF")
    p.add_argument("--print-missing-rows-max", type=int, default=5,
                   help="Max rows to print per ETF when --print-missing-rows is enabled")

    return p.parse_args()


def main():
    logger = setup_logger()
    args = parse_args()

    global MIGRATE_LEGACY_CSV, DELETE_LEGACY_CSV
    MIGRATE_LEGACY_CSV = not args.no_migrate_csv
    DELETE_LEGACY_CSV = bool(args.delete_legacy_csv)

    if MIGRATE_LEGACY_CSV:
        logger.info("Legacy CSV migration ENABLED: will read *.csv if *.parquet is missing.")
    else:
        logger.info("Legacy CSV migration DISABLED: Parquet-only.")

    if DELETE_LEGACY_CSV:
        logger.warning("Legacy CSV deletion ENABLED: legacy *.csv files will be deleted after successful Parquet writes.")

    holidays = _read_holidays(args.holidays_file)
    if holidays:
        logger.info("Loaded %d holidays from %s", len(holidays), args.holidays_file)
    else:
        logger.info("No holidays loaded (weekend-only calendar).")

    mode = args.mode.lower().strip()
    skip_if_fresh = not args.no_skip

    if mode == "all":
        for m in VALID_MODES:
            run_mode(
                m,
                context=args.context,
                max_workers=args.max_workers,
                force_today_daily=args.force_today_daily,
                skip_if_fresh=skip_if_fresh,
                intraday_ts=args.intraday_ts,
                holidays=holidays,
                refresh_tokens=args.refresh_tokens,
                report_dir=args.report_dir,
                print_missing_rows=args.print_missing_rows,
                print_missing_rows_max=args.print_missing_rows_max
            )
    else:
        run_mode(
            mode,
            context=args.context,
            max_workers=args.max_workers,
            force_today_daily=args.force_today_daily,
            skip_if_fresh=skip_if_fresh,
            intraday_ts=args.intraday_ts,
            holidays=holidays,
            refresh_tokens=args.refresh_tokens,
            report_dir=args.report_dir,
            print_missing_rows=args.print_missing_rows,
            print_missing_rows_max=args.print_missing_rows_max
        )


if __name__ == "__main__":
    main()
