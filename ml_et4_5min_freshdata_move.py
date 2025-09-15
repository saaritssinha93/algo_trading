# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 14:07:58 2025

@author: Saarit
"""
import os
import glob
import shutil
from datetime import datetime
import pandas as pd
import pytz

def move_indicators_range_all_tickers(
    start_date_str="2025-09-01",
    src_dir="main_indicators_August_5min",
    dst_dir="main_indicators_july_5min",
    pattern="*_main_indicators.csv",
    backup=True,
    dry_run=False
):
    india_tz = pytz.timezone("Asia/Kolkata")
    start_dt_ist = india_tz.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))

    os.makedirs(dst_dir, exist_ok=True)

    def _load_csv_smart(path):
        """Load CSV, coalesce any duplicate 'date*' columns into tz-aware IST 'date' (no warnings)."""
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, low_memory=False)

        # pick any columns that look like date/timestamp
        cand = [c for c in df.columns if c.lower() in ("date","timestamp") or c.lower().startswith("date")]
        if not cand:
            raise ValueError(f"'date' column not found in {path}")

        # Prefer 'date' first, then other 'date.*', then 'timestamp'
        def _rank(c):
            cl = c.lower()
            if cl == "date": return (0, c)
            if cl.startswith("date"): return (1, c)
            if cl == "timestamp": return (2, c)
            return (3, c)
        cand = sorted(set(cand), key=_rank)

        # Parse each candidate to tz-aware IST, then take first non-null across columns (bfill)
        parsed = []
        for c in cand:
            s = pd.to_datetime(df[c], errors="coerce")
            # unify to tz-aware IST
            if getattr(s.dt, "tz", None) is None:
                s = s.dt.tz_localize(india_tz)
            else:
                s = s.dt.tz_convert(india_tz)
            parsed.append(s)

        # Row-wise first non-null without using combine_first (avoids FutureWarning)
        dt_series = parsed[0]
        for s in parsed[1:]:
            dt_series = dt_series.where(dt_series.notna(), s)

        # Drop original date-like columns, insert canonical 'date'
        keep_cols = [c for c in df.columns if c not in cand]
        df = df[keep_cols]
        df.insert(0, "date", dt_series)

        # Clean
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def _backup(path):
        if backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")

    total_moved = 0
    files = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if not files:
        print(f"No files found in {src_dir} matching {pattern}.")
        return

    print(f"Moving rows from {start_date_str} (IST) onwards:")
    print(f"  Source:      {src_dir}")
    print(f"  Destination: {dst_dir}")
    print(f"  Dry-run:     {dry_run}")
    print("-" * 72)

    for src_path in files:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)

        src_df = _load_csv_smart(src_path)
        if src_df is None or src_df.empty:
            print(f"{fname}: source empty or missing → skip")
            continue

        mask_move = src_df["date"] >= start_dt_ist
        move_df = src_df.loc[mask_move].copy()
        keep_df = src_df.loc[~mask_move].copy()

        if move_df.empty:
            max_d = src_df["date"].max()
            max_d_str = max_d.tz_convert(india_tz).strftime("%Y-%m-%d %H:%M %Z") if pd.notna(max_d) else "N/A"
            print(f"{fname}: nothing to move (0 rows) — latest in src: {max_d_str}")
            continue

        # Load/create dest and align columns
        if os.path.exists(dst_path):
            dst_df = _load_csv_smart(dst_path)
            all_cols = list(dict.fromkeys(["date"] + [c for c in dst_df.columns if c != "date"] + [c for c in move_df.columns if c != "date"]))
            dst_df = dst_df.reindex(columns=all_cols)
            move_df = move_df.reindex(columns=all_cols)
        else:
            dst_df = pd.DataFrame(columns=move_df.columns)

        before_dst = len(dst_df)
        before_src = len(src_df)

        merged_dst = (
            pd.concat([dst_df, move_df], ignore_index=True)
              .drop_duplicates(subset=["date"], keep="last")
              .sort_values("date")
              .reset_index(drop=True)
        )

        moved_n = len(merged_dst) - before_dst
        remain_src = len(keep_df)

        if dry_run:
            print(f"{fname}: would move {len(move_df)} rows | src {before_src}→{remain_src} | dst {before_dst}→{len(merged_dst)} (net +{moved_n})")
        else:
            _backup(src_path)
            _backup(dst_path)
            merged_dst.to_csv(dst_path, index=False)
            keep_df.to_csv(src_path, index=False)
            print(f"{fname}: moved {len(move_df)} rows | src {before_src}→{remain_src} | dst {before_dst}→{len(merged_dst)} (net +{moved_n})")

        total_moved += len(move_df)

    print("-" * 72)
    print(f"Done. Total rows moved: {total_moved} {'(dry-run)' if dry_run else ''}")


def print_src_date_ranges(src_dir="main_indicators_August_5min", pattern="*_main_indicators.csv"):
    india_tz = pytz.timezone("Asia/Kolkata")
    for p in sorted(glob.glob(os.path.join(src_dir, pattern))):
        try:
            df = pd.read_csv(p, usecols=["date"])
        except Exception:
            df = pd.read_csv(p)
        # try to parse a few likely date cols
        cand = [c for c in df.columns if c.lower().startswith("date") or c.lower() in ("date","timestamp")]
        s = None
        for c in cand:
            s_try = pd.to_datetime(df[c], errors="coerce")
            if s is None: s = s_try
            else: s = s.where(s.notna(), s_try)
        s = s.dropna()
        if s.empty:
            print(f"{os.path.basename(p)}: no parsable dates")
            continue
        # localize/convert to IST for display
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(india_tz)
        else:
            s = s.dt.tz_convert(india_tz)
        print(f"{os.path.basename(p)}: {s.min().strftime('%Y-%m-%d')} → {s.max().strftime('%Y-%m-%d')}  (rows: {len(s)})")



if __name__ == "__main__":
    move_indicators_range_all_tickers(
        start_date_str="2025-07-11",
        src_dir="main_indicators_August_5min",
        dst_dir="main_indicators_july_5min2",
        backup=True,
        dry_run=False  # set True first to preview
    )
