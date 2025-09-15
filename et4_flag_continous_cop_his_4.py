# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:03:25 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
refresh_flags_once_first_yes_by_dates.py
─────────────────────────────────────────────────────────────
Rebuilds the   flag   column in *_smlitvl.csv files, then
collects the FIRST “YES” row per <ticker, calendar-date>.

Changes in this version
─────────────────────────────────────────────────────────────
• Highest_Daily_Change is now the most extreme Daily_Change
  occurring AFTER (and including) the entry timestamp.
  Delta = |Highest_Daily_Change − entry Daily_Change|
"""

import os, glob, logging, pandas as pd, numpy as np
from datetime import datetime

# ───────── USER: trading days of interest ─────────
DATES = [
    "02-05-2025", "05-05-2025", "06-05-2025", "07-05-2025", "08-05-2025",
    "09-05-2025", "12-05-2025", "13-05-2025", "14-05-2025", "15-05-2025",
    "16-05-2025", "19-05-2025", "20-05-2025", "21-05-2025", "22-05-2025",
    "23-05-2025", "26-05-2025", "27-05-2025", "28-05-2025", "29-05-2025",
    "30-05-2025", "02-06-2025", "03-06-2025", "04-06-2025", "05-06-2025",
    "06-06-2025", "09-06-2025", "10-06-2025", "11-06-2025", "12-06-2025",
    "13-06-2025", "16-06-2025", "17-06-2025",
]
DATE_OBJS = [datetime.strptime(d, "%d-%m-%Y").date() for d in DATES]

# ───────── folders ─────────
OUT_DIR  = "main_indicators_smlitvl_his"
LOG_FILE = "logs/refresh_flags_once.log"
os.makedirs("logs", exist_ok=True)

log = logging.getLogger("flag_once")
log.setLevel(logging.INFO)
_h  = logging.FileHandler(LOG_FILE, encoding="utf-8")
_s  = logging.StreamHandler()
for h in (_h, _s):
    h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s: %(message)s"))
    log.addHandler(h)

# ─────────────────── FLAG REBUILD ────────────────────
def rebuild_flag(csv_path: str) -> None:
    fn = os.path.basename(csv_path)
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], engine="python", on_bad_lines="skip")
        if df.empty:
            log.warning("skip %-40s (empty file)", fn); return

        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

        needed = ["Daily_Change", "Change_2min", "Force_Index",
                  "RSI_14", "ADX_14", "MFI_14"]
        for col in needed:
            if col not in df.columns:
                log.error("%s missing '%s' – rebuild skipped", fn, col)
                return
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---- core YES/NO screens ----
        up_core = (
            (df["Daily_Change"] >= 1) &
            df["Change_2min"].between(0.50, 2.00, inclusive="neither") &
            (df["Change_2min"].shift(1) > 0) &
            df["RSI_14"].between(30, 85) &
            df["MFI_14"].between(30, 85) &
            (df["ADX_14"] >= 25)
        )
        dn_core = (
            (df["Daily_Change"] <= -1) &
            df["Change_2min"].between(-2.00, -0.50, inclusive="neither") &
            (df["Change_2min"].shift(1) < 0) &
            df["RSI_14"].between(15, 50) &
            df["MFI_14"].between(15, 50) &
            (df["ADX_14"] >= 25)
        )

        # ---- Force-Index 2-EMA filter ----
        fi_ema  = df["Force_Index"].ewm(span=2, adjust=False).mean()
        long_ok = (fi_ema > 0)  & (fi_ema.shift(1) <= 0) & (fi_ema > fi_ema.shift(1))
        shrt_ok = (fi_ema < 0)  & (fi_ema.shift(1) >= 0) & (fi_ema < fi_ema.shift(1))

        df["flag"] = "NO"
        df.loc[(up_core & long_ok) | (dn_core & shrt_ok), "flag"] = "YES"

        base  = ["date","open","high","low","close","volume",
                 "Daily_Change","Change_2min","flag",
                 "RSI_14","ADX_14","MFI_14","OBV","Force_Index"]
        extra = [c for c in df.columns if c not in base]
        df.to_csv(csv_path, index=False, columns=base + extra)
        log.info("updated %-40s rows=%d", fn, len(df))

    except Exception as e:
        log.error("FAILED %-40s  %s", fn, e)

# ───────────── COLLECT FIRST-YES ROWS ──────────────
def collect_yes(csv_list: list[str]) -> None:
    pos, neg = {d: [] for d in DATE_OBJS}, {d: [] for d in DATE_OBJS}

    for fp in csv_list:
        try:
            df = pd.read_csv(fp, parse_dates=["date"], engine="python", on_bad_lines="skip")
            if "flag" not in df.columns:
                continue
            df["Daily_Change"] = pd.to_numeric(df["Daily_Change"], errors="coerce")
            df = df[df["date"].dt.date.isin(DATE_OBJS)]
            if df.empty:
                continue

            for d in DATE_OBJS:
                yes_rows = df[(df["flag"] == "YES") & (df["date"].dt.date == d)]
                if yes_rows.empty:
                    continue

                first = yes_rows.sort_values("date").iloc[0].copy()
                first["ticker"] = os.path.basename(fp).split("_")[0]

                # ——— NEW: extreme AFTER the entry ———
                later = df[(df["date"].dt.date == d) & (df["date"] >= first["date"])]
                if later.empty:
                    later = yes_rows.iloc[[0]]  # safety fallback

                if first["Daily_Change"] >= 0:
                    extreme = later["Daily_Change"].max()
                else:
                    extreme = later["Daily_Change"].min()

                first["Highest_Daily_Change"] = extreme
                first["Delta"] = abs(extreme - first["Daily_Change"])

                bucket = pos if first["Daily_Change"] >= 0 else neg
                bucket[d].append(first)

        except Exception as e:
            log.warning("collect skip %s (%s)", os.path.basename(fp), e)

    # -------- emit daily CSVs --------
    def _emit(bkt: dict, label: str) -> None:
        for d, rows in bkt.items():
            if not rows:
                continue
            fname = f"{d.strftime('%d-%m-%Y')}_yes_{label}_flag.csv"
            pd.DataFrame(rows).to_csv(fname, index=False)
            log.info("%s  →  %s  (%d tickers)", label, fname, len(rows))

    _emit(pos, "positive")
    _emit(neg, "negative")

# ───────────────────────── DRIVER ─────────────────────────
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(OUT_DIR, "*_smlitvl.csv"))
    if not csv_files:
        log.error("no *_smlitvl.csv files found in %s", OUT_DIR)
        raise SystemExit

    for fp in csv_files:
        rebuild_flag(fp)

    collect_yes(csv_files)



from pathlib import Path
import shutil
import logging

def move_yes_csv_files(src_dir: str | Path, dest_dir: str | Path = "YES") -> int:
    """
    Move every CSV that matches *_yes_*_flag.csv from `src_dir`
    (non-recursively) into `dest_dir`.

    Parameters
    ----------
    src_dir : str or pathlib.Path
        Folder where the *_yes_*_flag.csv files were generated.
    dest_dir : str or pathlib.Path, default "YES"
        Destination folder (created if it doesn’t exist).

    Returns
    -------
    int
        Number of files successfully moved.
    """
    src_dir  = Path(src_dir).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    pattern = "*_yes_*_flag.csv"
    files   = list(src_dir.glob(pattern))

    if not files:
        logging.warning("No *_yes_*_flag.csv files found in %s", src_dir)
        return 0

    moved = 0
    for fp in files:
        try:
            # If a file with the same name already exists, overwrite it
            target = dest_dir / fp.name
            if target.exists():
                target.unlink()
            shutil.move(fp, target)
            moved += 1
        except Exception as e:
            logging.error("Failed to move %s → %s  (%s)", fp.name, dest_dir, e)

    logging.info("Moved %d file(s) to %s", moved, dest_dir)
    return moved


# ───── example usage ─────
if __name__ == "__main__":
    SOURCE_FOLDER = "."   # folder where *_yes_*_flag.csv files currently live
    TARGET_FOLDER = "YES" # destination folder
    move_yes_csv_files(SOURCE_FOLDER, TARGET_FOLDER)


import os
import glob
import pandas as pd

def summarize_flag_files(dir_path: str = ".") -> None:
    """
    Print a per-file summary of YES-flag CSVs.

    Columns shown
    -------------
    File              : basename of the CSV
    Records           : total rows in that CSV
    Delta > 1         : rows where the 'Delta' column > 1
    % > 1             : share of rows whose Delta > 1   (rounded 1-dp)
    """
    pattern = os.path.join(dir_path, "*_yes_positive_flag.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print("No *_yes_*_flag.csv files found in", dir_path)
        return

    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"⚠️  Skipping {os.path.basename(fp)}  ({e})")
            continue

        total = len(df)
        if total == 0:
            rows.append([os.path.basename(fp), 0, 0, "0.0 %"])
            continue

        # Delta column may be missing if generation failed; guard for it.
        delta_series = pd.to_numeric(df.get("Delta"), errors="coerce")
        gt1          = (delta_series > 1).sum()
        pct          = f"{(gt1 / total) * 100:4.1f} %"

        rows.append([os.path.basename(fp), total, gt1, pct])

    # Build a DataFrame purely for pretty, column-aligned printing.
    summary = pd.DataFrame(
        rows, columns=["File", "Records", "Delta > 1", "% > 1"]
    )

    # Add a grand-total row
    total_row = pd.DataFrame(
        [["— TOTAL —",
          summary["Records"].sum(),
          summary["Delta > 1"].sum(),
          f"{(summary['Delta > 1'].sum() / summary['Records'].sum()) * 100:4.1f} %"]],
        columns=summary.columns
    )
    summary = pd.concat([summary, total_row], ignore_index=True)

    print("\n" + summary.to_string(index=False) + "\n")


# Example usage
if __name__ == "__main__":
    summarize_flag_files(dir_path="./YES")  # or point to another folder

