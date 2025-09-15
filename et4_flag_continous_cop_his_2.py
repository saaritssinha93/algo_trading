# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:25:07 2025

@author: Saarit
"""

# refresh_flags_once_first_yes_by_dates.py
# ─────────────────────────────────────────────────────────────
# One-shot helper
#
#   • Rebuild column  flag  in every  *_smlitvl.csv  under  OUT_DIR
#   • Take the *earliest* YES row for each <ticker, calendar-date> pair
#   • Split those rows into bullish / bearish buckets **per date**
#       ⇒  <DD-MM-YYYY>_yes_positive_flag.csv
#       ⇒  <DD-MM-YYYY>_yes_negative_flag.csv
#   • Add two convenience columns
#         Highest_Daily_Change   (extreme value in that ticker file)
#         Delta                  = |Highest_Daily_Change − Daily_Change|
# ─────────────────────────────────────────────────────────────
import os, glob, logging, pandas as pd, numpy as np
from datetime import datetime

# ------------ USER: list the dates you care about (DD-MM-YYYY) ------------
DATES = [
    "02-05-2025",  # Fri
    "05-05-2025",  # Mon
    "06-05-2025",  # Tue
    "07-05-2025",  # Wed
    "08-05-2025",  # Thu
    "09-05-2025",  # Fri
    "12-05-2025",  # Mon
    "13-05-2025",  # Tue
    "14-05-2025",  # Wed
    "15-05-2025",  # Thu
    "16-05-2025",  # Fri
    "19-05-2025",  # Mon
    "20-05-2025",  # Tue
    "21-05-2025",  # Wed
    "22-05-2025",  # Thu
    "23-05-2025",  # Fri
    "26-05-2025",  # Mon
    "27-05-2025",  # Tue
    "28-05-2025",  # Wed
    "29-05-2025",  # Thu
    "30-05-2025",  # Fri
    "02-06-2025",  # Mon
    "03-06-2025",  # Tue
    "04-06-2025",  # Wed
    "05-06-2025",  # Thu
    "06-06-2025",  # Fri
    "09-06-2025",  # Mon
    "10-06-2025",  # Tue
    "11-06-2025",  # Wed
    "12-06-2025",  # Thu
    "13-06-2025",  # Fri
    "16-06-2025",  # Mon
    "17-06-2025",  # Tue
]

DATE_OBJS  = [datetime.strptime(d, "%d-%m-%Y").date() for d in DATES]

# ------------ folders ------------
OUT_DIR = "main_indicators_smlitvl_his_main"
LOG_FILE = "logs/refresh_flags_once.log"
os.makedirs("logs", exist_ok=True)

log = logging.getLogger("flag_once")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(levelname)s: %(message)s")
log.addHandler(logging.FileHandler(LOG_FILE, encoding="utf-8"))
log.addHandler(logging.StreamHandler())

# ───────────────────────── FLAG RULE ─────────────────────────
def rebuild_flag(csv_path: str) -> None:
    fn = os.path.basename(csv_path)
    try:
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
        except pd.errors.ParserError:
            df = pd.read_csv(csv_path, parse_dates=["date"],
                             engine="python", on_bad_lines="skip")

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

        # ---- core screens (same as before) ----
        up_core = (
            (df["Daily_Change"] >= 1) &
            df["Change_2min"].between(0.50, 2.00, inclusive="neither") &
            (df["Change_2min"].shift(1) > 0) 
            & df["RSI_14"].between(50, 85)
            & df["MFI_14"].between(50, 85)
            & (df["ADX_14"] >= 25)
        )
        dn_core = (
            (df["Daily_Change"] <= -1) &
            df["Change_2min"].between(-2.00, -0.50, inclusive="neither") &
            (df["Change_2min"].shift(1) < 0) 
            & df["RSI_14"].between(15, 50) 
            & df["MFI_14"].between(15, 50)
            & (df["ADX_14"] >= 25)
        )

        #1
        # ---- Force-Index 2-period EMA cross/confirm ----
        fi_ema = df["Force_Index"].ewm(span=2, adjust=False).mean()
        long_ok  = (fi_ema > 0) & (fi_ema.shift(1) <= 0) & (fi_ema > fi_ema.shift(1))
        short_ok = (fi_ema < 0) & (fi_ema.shift(1) >= 0) & (fi_ema < fi_ema.shift(1))
        # ---- Force-Index 2-period EMA (relaxed filter) -----------------
       
        
        #2
        #fi_ema   = df["Force_Index"].ewm(span=2, adjust=False).mean()

        # A bar is eligible if the smoothed Force Index is simply on the
        # correct side of zero (no cross-over or acceleration checks).
        #long_ok  = fi_ema > 0          # bullish bias
        #short_ok = fi_ema < 0          # bearish bias

        #3

        #fi_ema = df["Force_Index"].ewm(span=2, adjust=False).mean()

        #median5 = fi_ema.rolling(5).median()
        #long_ok  = (fi_ema > 0) & (fi_ema >= 3 * median5)
        #short_ok = (fi_ema < 0) & (abs(fi_ema) >= 3 * abs(median5))
    

        df["flag"] = "NO"
        df.loc[(up_core & long_ok) | (dn_core & short_ok), "flag"] = "YES"

        base = ["date","open","high","low","close","volume",
                "Daily_Change","Change_2min","flag",
                "RSI_14","ADX_14","MFI_14","OBV","Force_Index"]
        extra = [c for c in df.columns if c not in base]
        df.to_csv(csv_path, index=False, columns=base+extra)
        log.info("updated %-40s rows=%d", fn, len(df))

    except Exception as e:
        log.error("FAILED %-40s  %s", fn, e)

# ───────────────────── COLLECT BY DATE ─────────────────────
def collect_yes(csv_list:list[str]) -> None:
    # date-key → list[row]
    pos, neg = {d: [] for d in DATE_OBJS}, {d: [] for d in DATE_OBJS}

    for fp in csv_list:
        try:
            df = pd.read_csv(fp, parse_dates=["date"],
                             engine="python", on_bad_lines="skip")
            if "flag" not in df.columns:
                continue
            df["Daily_Change"] = pd.to_numeric(df["Daily_Change"], errors="coerce")

            # keep only requested dates
            df = df[df["date"].dt.date.isin(DATE_OBJS)]
            if df.empty:
                continue

            for d in DATE_OBJS:
                day_rows = df[(df["flag"] == "YES") & (df["date"].dt.date == d)]
                if day_rows.empty:
                    continue
                first = day_rows.sort_values("date").iloc[0].copy()
                first["ticker"] = os.path.basename(fp).split("_")[0]

                # add extreme + delta
                max_c = df["Daily_Change"].max()
                min_c = df["Daily_Change"].min()
                first["Highest_Daily_Change"] = max_c if first["Daily_Change"] >= 0 else min_c
                first["Delta"] = abs(first["Highest_Daily_Change"] - first["Daily_Change"])

                bucket = pos if first["Daily_Change"] >= 0 else neg
                bucket[d].append(first)

        except Exception as e:
            log.warning("collect skip %s (%s)", os.path.basename(fp), e)

    # ------- emit files -------
    def _emit(bkt:dict, label:str):
        for d, rows in bkt.items():
            if not rows:
                continue
            fname = f"{d.strftime('%d-%m-%Y')}_yes_{label}_flag.csv"
            out_path = os.path.join(fname)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            log.info("%s  →  %s  (%d tickers)", label, fname, len(rows))

    _emit(pos, "positive")
    _emit(neg, "negative")

# ─────────────────────────── DRIVER ───────────────────────────
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(OUT_DIR, "*_smlitvl.csv"))
    if not csv_files:
        log.error("no *_smlitvl.csv files found in %s", OUT_DIR)
        raise SystemExit

    for fp in csv_files:
        rebuild_flag(fp)

    collect_yes(csv_files)


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
    Delta > 2         : rows where the 'Delta' column > 2
    % > 2             : share of rows whose Delta > 2   (rounded 1-dp)
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
            rows.append(
                [os.path.basename(fp), 0, 0, "0.0 %"]
            )
            continue

        # Delta column may be missing if generation failed; guard for it.
        delta_series = pd.to_numeric(df.get("Delta"), errors="coerce")
        gt2          = (delta_series > 2).sum()
        pct          = f"{(gt2 / total) * 100:4.1f} %"

        rows.append([os.path.basename(fp), total, gt2, pct])

    # Build a DataFrame purely for pretty, column-aligned printing.
    summary = pd.DataFrame(
        rows, columns=["File", "Records", "Delta > 2", "% > 2"]
    )

    # Add a grand total row
    total_row = pd.DataFrame(
        [["— TOTAL —",
          summary["Records"].sum(),
          summary["Delta > 2"].sum(),
          f"{(summary['Delta > 2'].sum()/summary['Records'].sum())*100:4.1f} %"]],
        columns=summary.columns
    )
    summary = pd.concat([summary, total_row], ignore_index=True)

    print("\n" + summary.to_string(index=False) + "\n")


# Example usage
if __name__ == "__main__":
    summarize_flag_files(dir_path=".")       # or point to another folder



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
