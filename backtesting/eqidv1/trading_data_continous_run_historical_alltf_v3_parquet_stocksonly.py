# -*- coding: utf-8 -*-
"""
Unified Zerodha (KiteConnect) data fetch + indicator generator
**ONLY for intraday 5-minute and 15-minute timeframes** (Parquet storage).

This is a trimmed / rewritten version of:
  trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py

What was removed:
- daily / weekly / 1h / 3h modes
- 3h resampling pipeline
- daily/weekly cutoff logic
- all mode names except: 5min, 15min
- related directories and warmup settings

What remains:
- ETF universe loader (filtered_stocks_MIS.py or stocks_tickers.txt)
- Kite session setup (api_key.txt + access_token.txt)
- Trading calendar helpers (weekends + optional holidays file)
- Robust missing/freshness detection for intraday candles
- Incremental fetching with warmup re-stabilization
- Indicator computation (RSI/ATR/MACD/BB/ADX/VWAP/EMA/CCI/MFI/OBV, etc.)
- Parquet outputs + optional legacy CSV migration (read-only or delete after write)
- Reports for missing files / newly appended rows

Outputs (Parquet):
- stocks_indicators_5min_eq / <TICKER>_stocks_indicators_5min.parquet
- stocks_indicators_15min_eq / <TICKER>_stocks_indicators_15min.parquet

Usage examples:
- Fetch only 5-min:
    python trading_data_continous_run_historical_5m_15m_parquet.py 5min
- Fetch only 15-min:
    python trading_data_continous_run_historical_5m_15m_parquet.py 15min
- Fetch both:
    python trading_data_continous_run_historical_5m_15m_parquet.py all

Notes:
- Intraday timestamps can be stored as candle "end" (recommended) or "start".
- By default, the script skips tickers that are already "fresh".
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

# Directories (only intraday)
DIRS = {
    "5min":   {"cache": "stocks_cache_5min_eq",   "out": "stocks_indicators_5min_eq"},
    "15min":  {"cache": "stocks_cache_15min_eq",  "out": "stocks_indicators_15min_eq"},
}
for cfg in DIRS.values():
    os.makedirs(cfg["cache"], exist_ok=True)
    os.makedirs(cfg["out"], exist_ok=True)

VALID_MODES = ("5min", "15min")
DEFAULT_MAX_WORKERS = 6

# Market timing (IST)
MARKET_OPEN_TIME = time(9, 15)
MARKET_CLOSE_TIME_INTRADAY = time(15, 30)      # last intraday candle end (5m/15m)

# Candle-end timestamps for intraday
DEFAULT_INTRADAY_TIMESTAMP = "end"  # "end" or "start"

# Incremental fetch warmup bars (to re-stabilize indicators near the tail)
WARMUP_BARS = {
    "5min":  600,
    "15min": 400,
}

# Token cache
TOKENS_CACHE_FILE = "stocks_tokens_cache.json"
TOKENS_CACHE_MAX_AGE_DAYS = 7

# Optional NSE holidays file (one date per line or CSV column "date")
HOLIDAYS_FILE_DEFAULT = "nse_holidays.csv"

# ========= STORAGE (PARQUET) =========
MIGRATE_LEGACY_CSV = True
DELETE_LEGACY_CSV = False


# ========= LOGGING =========

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("stocks_fetcher")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler("stocks_fetcher_run.log", mode="w", encoding="utf-8")
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


def load_stocks_universe(logger: logging.Logger) -> tuple[list[str], dict[str, int]]:
    """
    Universe loader (ETF-ready):
    - Preferred: filtered_stocks_MIS.py with either:
        - stocks_tokens = {SYMBOL: TOKEN, ...}
        - selected_stocks = [...]
    - Fallback: stocks_tickers.txt (one symbol per line)
    """
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    for p in (str(script_dir), str(parent_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    token_map: dict[str, int] = {}
    mod: Optional[ModuleType] = None

    try:
        mod = importlib.import_module("filtered_stocks_MIS")
    except Exception:
        mod = None

    if mod is not None:
        if hasattr(mod, "stocks_tokens") and isinstance(getattr(mod, "stocks_tokens"), dict):
            raw = getattr(mod, "stocks_tokens")
            try:
                token_map = {str(k).strip().upper(): int(v) for k, v in raw.items() if str(k).strip()}
                tickers = sorted(token_map.keys())
                if tickers:
                    logger.info("Loaded %d symbols from filtered_stocks_MIS.stocks_tokens", len(tickers))
                    return tickers, token_map
            except Exception:
                pass

        if hasattr(mod, "selected_stocks"):
            ss = getattr(mod, "selected_stocks")
            if isinstance(ss, dict):
                tickers = _normalize_ticker_list(ss)
                try:
                    if ss and all(isinstance(v, (int, float)) for v in ss.values()):
                        token_map = {str(k).strip().upper(): int(v) for k, v in ss.items() if str(k).strip()}
                        tickers = sorted(token_map.keys())
                except Exception:
                    pass
                if tickers:
                    logger.info("Loaded %d symbols from filtered_stocks_MIS.selected_stocks", len(tickers))
                    return tickers, token_map

            tickers = _normalize_ticker_list(ss)
            if tickers:
                logger.info("Loaded %d symbols from filtered_stocks_MIS.selected_stocks", len(tickers))
                return tickers, token_map

    for base in (cwd, script_dir, parent_dir):
        f = base / "stocks_tickers.txt"
        if f.exists():
            arr = [x.strip().upper() for x in f.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            tickers = _normalize_ticker_list(arr)
            if tickers:
                logger.info("Loaded %d symbols from %s", len(tickers), str(f))
                return tickers, token_map

    raise RuntimeError(
        "Could not load symbols.\n"
        "Fix options:\n"
        "  1) Ensure filtered_stocks_MIS.py is importable and define either:\n"
        "       - stocks_tokens = {SYMBOL: TOKEN, ...}   OR\n"
        "       - selected_stocks = [SYMBOL, ...] / {SYMBOL, ...} / {SYMBOL: TOKEN, ...}\n"
        "  2) Or create stocks_tickers.txt (one symbol per line) in cwd / script dir / parent dir.\n\n"
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

def _is_trading_day(d: date, holidays: set[date]) -> bool:
    if d.weekday() >= 5:
        return False
    if d in holidays:
        return False
    return True

def _prev_trading_day(d: date, holidays: set[date]) -> date:
    x = d - timedelta(days=1)
    while not _is_trading_day(x, holidays):
        x -= timedelta(days=1)
    return x

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


# ========= START DATE PER MODE =========

def get_start_date(mode: str, now_ist: datetime) -> datetime:
    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    # Keep your original intraday start anchor behaviour
    return IST_TZ.localize(datetime(2025, 8, 25, 0, 0, 0))


# ========= DATA HELPERS =========

def _to_ist(series_dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series_dt, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize(IST_TZ)
    return dt.dt.tz_convert(IST_TZ)

def _ensure_parquet_engine():
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Parquet storage requires 'pyarrow'.\n"
            "Install it once:  pip install pyarrow\n"
            f"Original import error: {e}"
        ) from e

def _read_last_ts_fast_parquet(path: str):
    try:
        _ensure_parquet_engine()
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        md = pf.metadata
        if md is None or md.num_rows <= 0:
            return None

        last_rg = md.num_row_groups - 1
        if last_rg < 0:
            return None

        table = pf.read_row_group(last_rg, columns=["date"])
        if table.num_rows <= 0:
            return None

        col = table.column(0)
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

def _read_last_ts_from_store(path: str):
    ext = str(Path(path).suffix).lower()
    if ext == ".parquet":
        return _read_last_ts_fast_parquet(path)
    if ext == ".csv":
        return _read_last_ts_fast_csv(path)
    ts = _read_last_ts_fast_parquet(path)
    if ts is not None:
        return ts
    return _read_last_ts_fast_csv(path)

def _intraday_end_shift_minutes(interval: str) -> int:
    return {"5minute": 5, "15minute": 15}.get(interval, 0)

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

def expected_last_stamp(mode: str, now_ist: datetime, holidays: set[date], intraday_ts: str) -> dict:
    mode = mode.lower().strip()

    if now_ist.tzinfo is None:
        now_ist = IST_TZ.localize(now_ist)

    step_map = {"5min": 5, "15min": 15}
    step = step_map.get(mode, 0)

    exp_end = last_completed_intraday_end(now_ist, step, holidays)

    if intraday_ts.lower() == "start":
        exp_end = exp_end - timedelta(minutes=step)

    return {"kind": "ts", "value": exp_end, "step_min": step}

def _legacy_csv_path_for(parquet_path: str) -> str:
    return str(Path(parquet_path).with_suffix(".csv"))

def _resolve_existing_store_path(target_parquet_path: str) -> str:
    if os.path.exists(target_parquet_path):
        return target_parquet_path
    if MIGRATE_LEGACY_CSV:
        legacy = _legacy_csv_path_for(target_parquet_path)
        if os.path.exists(legacy):
            return legacy
    return target_parquet_path

def ticker_is_fresh(mode: str, out_path: str, now_ist: datetime, holidays: set[date], intraday_ts: str) -> bool:
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

    spec = expected_last_stamp(mode, now_ist, holidays, intraday_ts)
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

def missing_spec(mode: str, out_path: str, now_ist: datetime, holidays: set[date], intraday_ts: str) -> dict:
    existing_path = _resolve_existing_store_path(out_path)

    spec = expected_last_stamp(mode, now_ist, holidays, intraday_ts)

    if not os.path.exists(existing_path):
        return {"kind": "file_missing", "last_ts": None, "expected": spec}

    last_ts = _read_last_ts_from_store(existing_path)
    if last_ts is None:
        return {"kind": "rows_missing", "last_ts": None, "expected": spec}

    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(IST_TZ)
    else:
        last_ts = last_ts.tz_convert(IST_TZ)

    if ticker_is_fresh(mode, out_path, now_ist, holidays, intraday_ts):
        return {"kind": "fresh", "last_ts": last_ts, "expected": spec}

    return {"kind": "rows_missing", "last_ts": last_ts, "expected": spec}


# ========= FETCHERS =========

def fetch_historical_generic(
    kite: KiteConnect,
    token: int,
    start_dt_ist: datetime,
    end_dt_ist: datetime,
    interval: str,
    chunk_days: int,
    step_td: timedelta,
    logger: logging.Logger,
    intraday_ts: str
) -> pd.DataFrame:
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

def fetch_historical_5min_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "5minute", 60, timedelta(minutes=5), logger, intraday_ts)

def fetch_historical_15min_df(kite, token, start_dt_ist, end_dt_ist, logger, intraday_ts):
    return fetch_historical_generic(kite, token, start_dt_ist, end_dt_ist, "15minute", 120, timedelta(minutes=15), logger, intraday_ts)


# ========= INDICATORS =========
# (Copied as-is from your original script to preserve feature parity.)

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
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
    close = df["close"].values
    volume = df["volume"].values
    direction = np.sign(np.diff(close, prepend=close[0]))
    direction[0] = 0
    return pd.Series(np.cumsum(direction * volume), index=df.index)

def add_standard_indicators(df):
    df["RSI"] = calculate_rsi(df["close"])
    df["ATR"] = calculate_atr(df)
    df["EMA_20"] = calculate_ema(df["close"], 20)
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


# ========= CHANGE FEATURES (intraday only) =========

def add_change_features_intraday(df: pd.DataFrame) -> pd.DataFrame:
    df["date_only"] = df["date"].dt.tz_convert(IST_TZ).dt.date
    df["Intra_Change"] = df.groupby("date_only")["close"].pct_change().mul(100.0)

    last_close_per_day = df.groupby("date_only", sort=True)["close"].last()
    prev_day_last_close = last_close_per_day.shift(1)

    df["Prev_Day_Close"] = df["date_only"].map(prev_day_last_close)
    df["Daily_Change"] = (df["close"] - df["Prev_Day_Close"]) / (df["Prev_Day_Close"] + 1e-10) * 100.0
    return df


# ========= TOKEN CACHE =========

def load_or_fetch_tokens(kite: KiteConnect, symbols: list[str], logger: logging.Logger, refresh: bool = False) -> dict[str, int]:
    syms_u = sorted({t.upper().strip() for t in symbols if t.strip()})

    if (not refresh) and os.path.exists(TOKENS_CACHE_FILE):
        try:
            st = os.stat(TOKENS_CACHE_FILE)
            age_days = (datetime.now() - datetime.fromtimestamp(st.st_mtime)).days
            if age_days <= TOKENS_CACHE_MAX_AGE_DAYS:
                cache = json.loads(Path(TOKENS_CACHE_FILE).read_text(encoding="utf-8"))
                if isinstance(cache, dict) and all(t in cache for t in syms_u):
                    return {t: int(cache[t]) for t in syms_u}
        except Exception:
            pass

    logger.info("Fetching NSE instruments for token map (this can take time)...")
    ins = pd.DataFrame(kite.instruments("NSE"))
    tokens = ins[ins["tradingsymbol"].isin(syms_u)][["tradingsymbol", "instrument_token"]]
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

    return {t: int(mp[t]) for t in syms_u if t in mp}


# ========= SAVE =========

def _finalize_and_save(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = str(Path(out_path).suffix).lower()

    if ext == ".parquet":
        _ensure_parquet_engine()
        df.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")
        return

    df.to_csv(out_path, index=False)


# ========= INCREMENTAL LOAD + MERGE =========

def _load_existing_ohlc(out_path: str, intraday_ts: str, mode: str) -> pd.DataFrame:
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

        if intraday_ts.lower() == "end":
            step = {"5min": 5, "15min": 15}[mode]
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
    else:
        back = timedelta(days=30)

    s = (last_ts - back)
    s = s.to_pydatetime() if isinstance(s, pd.Timestamp) else s
    if s.tzinfo is None:
        s = IST_TZ.localize(s)
    return max(default_start, s)


# ========= PER-SYMBOL PIPELINE =========

def _compute_common_features(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = add_standard_indicators(df)

    # Intraday recent high/low (shorter window)
    df["Recent_High"] = df["high"].rolling(5, min_periods=5).max()
    df["Recent_Low"] = df["low"].rolling(5, min_periods=5).min()

    stoch_k, stoch_d = calculate_stochastic_slow(df, 14, 3, 3, "sma")
    df["Stoch_%K"], df["Stoch_%D"] = stoch_k, stoch_d
    df["ADX"] = calculate_adx(df)

    df = add_change_features_intraday(df)
    return df

def _downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "date":
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
    return out

def _indicator_quality_snapshot(df: pd.DataFrame) -> dict[str, float]:
    checks: dict[str, float] = {}
    total = max(len(df), 1)

    for col in ("ATR", "RSI", "Stoch_%K", "Stoch_%D", "ADX", "EMA_20", "EMA_50"):
        if col in df.columns:
            checks[f"{col}_nan_pct"] = float(df[col].isna().sum() * 100.0 / total)

    if "RSI" in df.columns:
        r = pd.to_numeric(df["RSI"], errors="coerce")
        checks["RSI_out_of_range_pct"] = float(((r < 0) | (r > 100)).sum() * 100.0 / total)

    if "Stoch_%K" in df.columns:
        k = pd.to_numeric(df["Stoch_%K"], errors="coerce")
        checks["StochK_out_of_range_pct"] = float(((k < 0) | (k > 100)).sum() * 100.0 / total)

    return checks

def _log_indicator_quality(logger: logging.Logger, ticker: str, mode: str, df: pd.DataFrame) -> None:
    q = _indicator_quality_snapshot(df)
    if not q:
        return
    severe = [k for k, v in q.items() if v > 15.0]
    if severe:
        logger.warning("[%s] %s indicator quality warning: %s", mode.upper(), ticker, q)

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _fmt_expected(spec: dict) -> str:
    try:
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

def process_ticker(
    mode: str,
    ticker: str,
    token: int,
    kite: KiteConnect,
    start_dt_ist: datetime,
    end_dt_ist: datetime,
    logger: logging.Logger,
    holidays: set[date],
    skip_if_fresh: bool,
    intraday_ts: str,
    report_dir: str,
    print_missing_rows: bool,
    print_missing_rows_max: int
) -> UpdateReport:
    out_path = os.path.join(DIRS[mode]["out"], f"{ticker}_stocks_indicators_{mode}.parquet")
    now_ist = datetime.now(IST_TZ)

    existing_path = _resolve_existing_store_path(out_path)
    existed_before = os.path.exists(existing_path)
    last_before_ts = _read_last_ts_from_store(existing_path) if existed_before else None
    if last_before_ts is not None:
        if last_before_ts.tzinfo is None:
            last_before_ts = last_before_ts.tz_localize(IST_TZ)
        else:
            last_before_ts = last_before_ts.tz_convert(IST_TZ)

    exp = expected_last_stamp(mode, now_ist, holidays, intraday_ts)
    exp_str = _fmt_expected(exp)

    if skip_if_fresh and ticker_is_fresh(mode, out_path, now_ist, holidays, intraday_ts):
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
        if mode == "5min":
            fetched = fetch_historical_5min_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
        elif mode == "15min":
            fetched = fetch_historical_15min_df(kite, token, inc_start, end_dt_ist, logger, intraday_ts)
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
        merged = _downcast_numeric_columns(merged)
        _log_indicator_quality(logger, ticker, mode, merged)

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

def run_mode(
    mode: str,
    max_workers: int,
    skip_if_fresh: bool,
    intraday_ts: str,
    holidays: set[date],
    refresh_tokens: bool,
    report_dir: str,
    print_missing_rows: bool,
    print_missing_rows_max: int
):
    logger = logging.getLogger("stocks_fetcher")
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected: {', '.join(VALID_MODES)}")

    now_ist = datetime.now(IST_TZ)
    start_dt = get_start_date(mode, now_ist)

    step = 5 if mode == "5min" else 15
    end_dt = last_completed_intraday_end(now_ist, step, holidays)

    logger.info("=== MODE=%s | intraday_ts=%s | Window: %s → %s (IST) ===",
                mode, intraday_ts, start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M"))

    if end_dt <= start_dt:
        logger.info("End cutoff <= start. Nothing to fetch for %s.", mode)
        return

    syms, pre_token_map = load_stocks_universe(logger)

    missing_files: list[str] = []
    missing_rows: list[str] = []
    fresh: list[str] = []

    if skip_if_fresh:
        for t in syms:
            t = t.upper()
            out_path = os.path.join(DIRS[mode]["out"], f"{t}_stocks_indicators_{mode}.parquet")
            ms = missing_spec(mode, out_path, now_ist, holidays, intraday_ts)
            if ms["kind"] == "fresh":
                fresh.append(t)
            elif ms["kind"] == "file_missing":
                missing_files.append(t)
                missing_rows.append(t)
            else:
                missing_rows.append(t)
    else:
        missing_rows = [t.upper() for t in syms]

    if skip_if_fresh:
        logger.info("[%s] Missing files: %d", mode.upper(), len(missing_files))
        if missing_files:
            rep_dir = os.path.join(report_dir, "missing_files")
            _safe_mkdir(rep_dir)
            miss_file_path = os.path.join(rep_dir, f"missing_files_{mode}.txt")
            Path(miss_file_path).write_text("\n".join(missing_files), encoding="utf-8")
            logger.info("[%s] Missing files list saved: %s", mode.upper(), miss_file_path)
            logger.info("[%s] Missing files sample: %s", mode.upper(), ", ".join(missing_files[:50]))

        logger.info("[%s] Missing evaluation rows (stale symbols): %d", mode.upper(), len(missing_rows))
    else:
        logger.info("[%s] no-skip enabled => processing all symbols: %d", mode.upper(), len(missing_rows))

    if not missing_rows:
        logger.info("[%s] Nothing missing — all symbols fresh.", mode.upper())
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
        logger.info("No valid symbols with tokens.")
        return

    logger.info("[%s] Processing ONLY missing symbols=%d with max_workers=%d ...", mode.upper(), len(work_items), max_workers)

    updated_reports: list[UpdateReport] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_ticker,
                mode, tkr, tok, kite, start_dt, end_dt,
                logger, holidays,
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
        logger.info("[%s] Updated symbols: %d", mode.upper(), len(updated_reports))
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
        logger.warning("[%s] Failed symbols: %d (see stocks_fetcher_run.log)", mode.upper(), failed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="all", help="5min|15min|all")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
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

    p.add_argument("--report-dir", default="reports/stocks_missing_reports",
                   help="Directory to write missing-files and missing-rows reports")
    p.add_argument("--print-missing-rows", action="store_true",
                   help="Print a small preview of newly appended rows per symbol")
    p.add_argument("--print-missing-rows-max", type=int, default=5,
                   help="Max rows to print per symbol when --print-missing-rows is enabled")

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
                max_workers=args.max_workers,
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
            max_workers=args.max_workers,
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
