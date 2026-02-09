# -*- coding: utf-8 -*-
"""
COPY LIVE SIGNAL PARQUETS -> DATEWISE CSV (etf_signals_live)
ONE ROW PER TICKER PER DAY (KEEP FIRST)

Reads:
  out_live_signals/YYYYMMDD/signals_*.parquet

Writes:
  etf_signals_live/YYYYMMDD/signals_YYYYMMDD.csv

DEDUP RULE:
- If a ticker is already present in the day's CSV, DO NOT append it again.
- Keep ONLY the first occurrence for that ticker (earliest bar_time_ist).

CRITICAL FIX:
- Do NOT allow yesterday bars into today's CSV.
- Filter STRICTLY by bar_time_ist IST date == YYYYMMDD
  (checked_at_ist is used ONLY if bar_time_ist is missing)
"""

from __future__ import annotations

import os
import glob
from datetime import datetime, date
from typing import List, Optional

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")

OUT_SIGNALS_DIR = "out_live_signals"
CSV_OUT_ROOT    = "etf_signals_live"
DEDUP_KEY       = "ticker"


# =============================================================================
# Helpers
# =============================================================================
def reorder_columns_ticker_second(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make 'ticker' the 2nd column.
    Keeps first column as-is (usually bar_time_ist),
    preserves order of remaining columns.
    """
    if "ticker" not in df.columns or df.shape[1] < 2:
        return df

    cols = list(df.columns)

    # Remove ticker from current position
    cols.remove("ticker")

    # Insert ticker as second column
    cols.insert(1, "ticker")

    return df[cols]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def yyyymmdd_to_date(day_folder: str) -> date:
    return datetime.strptime(day_folder, "%Y%m%d").date()


def parse_dt(series: pd.Series) -> pd.Series:
    """
    Parse datetime robustly.
    Keeps tz if present; if tz-naive, localizes to IST (assume IST).
    """
    s = pd.to_datetime(series, errors="coerce")

    # tz-aware already?
    try:
        if getattr(s.dtype, "tz", None) is not None:
            return s
    except Exception:
        pass

    # tz-naive -> localize to IST
    try:
        return s.dt.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        # last resort (object dtype)
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(
            IST, nonexistent="shift_forward", ambiguous="NaT"
        )


def ist_date_of(ts: pd.Series) -> pd.Series:
    """Return IST date for tz-aware timestamps."""
    try:
        if getattr(ts.dtype, "tz", None) is not None:
            ts = ts.dt.tz_convert(IST)
    except Exception:
        pass
    return ts.dt.date


def list_signal_parquets_for_day(day_folder: str) -> List[str]:
    pattern = os.path.join(OUT_SIGNALS_DIR, day_folder, "signals_*.parquet")
    return sorted(glob.glob(pattern))


def read_and_concat_parquets(files: List[str]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    # keep only signal_triggered True if present
    if "signal_triggered" in out.columns:
        out = out[out["signal_triggered"] == True].copy()

    # normalize datetime cols
    for c in ["bar_time_ist", "checked_at_ist"]:
        if c in out.columns:
            out[c] = parse_dt(out[c])

    # normalize ticker
    if DEDUP_KEY in out.columns:
        out[DEDUP_KEY] = out[DEDUP_KEY].astype(str).str.strip()

    return out


def filter_signals_strictly_for_day(df: pd.DataFrame, day_folder: str) -> pd.DataFrame:
    """
    STRICT filter:
    - If bar_time_ist exists, keep ONLY rows whose bar_time_ist IST date == day_folder date.
    - If bar_time_ist is missing/NaT, then (and only then) use checked_at_ist date.
    """
    if df.empty:
        return df

    day = yyyymmdd_to_date(day_folder)

    has_bar = "bar_time_ist" in df.columns
    has_chk = "checked_at_ist" in df.columns

    if not has_bar and not has_chk:
        # nothing to filter on; safest is return empty
        return df.iloc[0:0].copy()

    keep = pd.Series(False, index=df.index)

    if has_bar:
        bt = parse_dt(df["bar_time_ist"])
        bt_date = ist_date_of(bt)
        bt_ok = (bt_date == day)
        keep = bt_ok

        # Only fallback to checked_at when bar_time is missing (NaT)
        if has_chk:
            ct = parse_dt(df["checked_at_ist"])
            ct_date = ist_date_of(ct)
            fallback_ok = bt.isna() & (ct_date == day)
            keep = keep | fallback_ok
    else:
        # no bar_time_ist => fallback to checked_at_ist only
        ct = parse_dt(df["checked_at_ist"])
        keep = (ist_date_of(ct) == day)

    return df[keep].copy()


def append_unique_tickers_to_daily_csv(day_folder: str, df_new: pd.DataFrame) -> Optional[str]:
    if df_new.empty:
        return None

    if DEDUP_KEY not in df_new.columns:
        raise ValueError(f"Missing required column '{DEDUP_KEY}' in new signals data.")

    # 1) STRICT day filter to prevent yesterday bar_time_ist
    df_new = filter_signals_strictly_for_day(df_new, day_folder)
    if df_new.empty:
        print(f"[INFO] No signals belong to {day_folder} after strict bar_time_ist filter.")
        return None

    out_dir = os.path.join(CSV_OUT_ROOT, day_folder)
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, f"signals_{day_folder}.csv")

    df_new = df_new.copy()
    df_new["_ticker_norm"] = df_new[DEDUP_KEY].astype(str).str.upper().str.strip()

    # 2) Dedup inside new batch: keep earliest bar_time_ist (or checked_at_ist)
    if "bar_time_ist" in df_new.columns:
        df_new["bar_time_ist"] = parse_dt(df_new["bar_time_ist"])
        df_new = df_new.sort_values(["bar_time_ist"], kind="stable")
    elif "checked_at_ist" in df_new.columns:
        df_new["checked_at_ist"] = parse_dt(df_new["checked_at_ist"])
        df_new = df_new.sort_values(["checked_at_ist"], kind="stable")

    df_new = df_new.drop_duplicates(subset=["_ticker_norm"], keep="first")

    # 3) If CSV exists, never append a ticker already present
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv, low_memory=False)

        existing = set()
        if DEDUP_KEY in df_old.columns:
            existing = set(df_old[DEDUP_KEY].astype(str).str.upper().str.strip().tolist())

        df_add = df_new[~df_new["_ticker_norm"].isin(existing)].copy()
        df_add = df_add.drop(columns=["_ticker_norm"], errors="ignore")

        if df_add.empty:
            print(f"[INFO] No new tickers to add for {day_folder}.")
            return out_csv

        # union columns (keep all columns)
        all_cols = sorted(set(df_old.columns).union(set(df_add.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_add = df_add.reindex(columns=all_cols)

        df_all = pd.concat([df_old, df_add], ignore_index=True)
    else:
        df_all = df_new.drop(columns=["_ticker_norm"], errors="ignore").copy()

    # 4) FINAL SAFETY: only first row per ticker in entire CSV
    df_all["_ticker_norm"] = df_all[DEDUP_KEY].astype(str).str.upper().str.strip()

    if "bar_time_ist" in df_all.columns:
        df_all["bar_time_ist"] = parse_dt(df_all["bar_time_ist"])
        df_all = df_all.sort_values(["bar_time_ist"], kind="stable")
    elif "checked_at_ist" in df_all.columns:
        df_all["checked_at_ist"] = parse_dt(df_all["checked_at_ist"])
        df_all = df_all.sort_values(["checked_at_ist"], kind="stable")

    df_all = df_all.drop_duplicates(subset=["_ticker_norm"], keep="first")
    df_all = df_all.drop(columns=["_ticker_norm"], errors="ignore")
    df_all = reorder_columns_ticker_second(df_all)
    df_all.to_csv(out_csv, index=False)

    df_all.to_csv(out_csv, index=False)
    return out_csv


def copy_signals_daywise(day_folder: str) -> None:
    files = list_signal_parquets_for_day(day_folder)
    if not files:
        print(f"[INFO] No signal parquets found for {day_folder} in {OUT_SIGNALS_DIR}/{day_folder}")
        return

    df_new = read_and_concat_parquets(files)
    if df_new.empty:
        print(f"[INFO] No signal rows found in parquet files for {day_folder}")
        return

    out_csv = append_unique_tickers_to_daily_csv(day_folder, df_new)
    if out_csv:
        print(
            f"[OK] CSV updated (1 row per ticker): {out_csv} "
            f"| scanned_files={len(files)} | scanned_rows={len(df_new)}"
        )


def detect_latest_day_folder() -> Optional[str]:
    if not os.path.isdir(OUT_SIGNALS_DIR):
        return None
    folders = [d for d in os.listdir(OUT_SIGNALS_DIR) if os.path.isdir(os.path.join(OUT_SIGNALS_DIR, d))]
    folders = [d for d in folders if len(d) == 8 and d.isdigit()]
    if not folders:
        return None
    return sorted(folders)[-1]


def main() -> None:
    today = datetime.now(IST).strftime("%Y%m%d")
    if os.path.isdir(os.path.join(OUT_SIGNALS_DIR, today)):
        day_folder = today
    else:
        day_folder = detect_latest_day_folder()

    if not day_folder:
        print("[ERROR] No day folders found under out_live_signals.")
        return

    copy_signals_daywise(day_folder)


if __name__ == "__main__":
    main()
