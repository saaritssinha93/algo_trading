# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:43:59 2025

@author: Saarit
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd

# === CONFIG ===
CSV_DIR = "."                 # folder where your papertrade CSVs live
BACKUP_BEFORE_OVERWRITE = True
LIMIT_TO_DATES = True         # set False to update ALL papertrade_5min_v3_*.csv

dates = [
    # August 2024
    "2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09",
    "2024-08-12","2024-08-13","2024-08-14","2024-08-16",
    "2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23",
    "2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30",

    # September 2024
    "2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06",
    "2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13",
    "2024-09-16","2024-09-17","2024-09-18","2024-09-19","2024-09-20",
    "2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27",
    "2024-09-30",

    # October 2024 (Oct 2 holiday)
    "2024-10-01","2024-10-03","2024-10-04","2024-10-07","2024-10-08",
    "2024-10-09","2024-10-10","2024-10-11","2024-10-14","2024-10-15",
    "2024-10-16","2024-10-17","2024-10-18","2024-10-21","2024-10-22",
    "2024-10-23","2024-10-24","2024-10-25","2024-10-28","2024-10-29",
    "2024-10-30","2024-10-31",

    # November 2024 (Nov 1 & Nov 15 holidays)
    "2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08",
    "2024-11-11","2024-11-12","2024-11-13","2024-11-14",
    "2024-11-18","2024-11-19","2024-11-20","2024-11-21","2024-11-22",
    "2024-11-25","2024-11-26","2024-11-27","2024-11-28","2024-11-29",

    # December 2024 (Dec 25 holiday)
    "2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06",
    "2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13",
    "2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20",
    "2024-12-23","2024-12-24","2024-12-26","2024-12-27","2024-12-30",
    "2024-12-31",

    # January 2025
    "2025-01-01","2025-01-02","2025-01-03",
    "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
    "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
    "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
    "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

    # February 2025
    "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
    "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
    "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
    "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

    # March 2025
    "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
    "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
    "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
    "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
    "2025-03-31",

    # April 2025
    "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
    "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
    "2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18",
    "2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25",
    "2025-04-28","2025-04-29","2025-04-30",

    # May 2025
    "2025-05-02",
    "2025-05-05","2025-05-06","2025-05-07","2025-05-08","2025-05-09",
    "2025-05-12","2025-05-13","2025-05-14","2025-05-15","2025-05-16",
    "2025-05-19","2025-05-20","2025-05-21","2025-05-22","2025-05-23",
    "2025-05-26","2025-05-27","2025-05-28","2025-05-29","2025-05-30",

    # June 2025
    "2025-06-02","2025-06-03","2025-06-04","2025-06-05","2025-06-06",
    "2025-06-09","2025-06-10","2025-06-11","2025-06-12","2025-06-13",
    "2025-06-16","2025-06-17","2025-06-18","2025-06-19","2025-06-20",
    "2025-06-23","2025-06-24","2025-06-25","2025-06-26","2025-06-27",
    "2025-06-30",

    # July 2025
    "2025-07-01","2025-07-02","2025-07-03","2025-07-04",
    "2025-07-07","2025-07-08","2025-07-09","2025-07-10"
]

def _price_column(df):
    """Find the 'Price' column (case-insensitive)."""
    for c in df.columns:
        if c.strip().lower() == "price":
            return c
    return None

def _quantity_column_name(df):
    """Use existing Quantity/Qty if present; else 'Quantity'."""
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("quantity", "qty"):
            return c
    return "Quantity"

def _total_value_column_name(df):
    """Use existing Total Value naming if present; else 'Total Value'."""
    candidates = ("Total Value", "Total_Value", "total value", "total_value")
    for c in df.columns:
        if c in candidates or c.strip().lower().replace("_", " ") == "total value":
            return c
    return "Total Value"

def _matches_listed_dates(path, date_list):
    base = os.path.basename(path)
    return any(d in base for d in date_list)

def update_file(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] {path} -> read error: {e}")
        return

    if df.empty:
        print(f"[SKIP] {path} -> empty file")
        return

    price_col = _price_column(df)
    if price_col is None:
        print(f"[SKIP] {path} -> no 'Price' column")
        return

    # Coerce to numeric prices
    price = pd.to_numeric(df[price_col], errors="coerce")

    # qty = int(100000 / price) if price > 0 else 0  (vectorized)
    qty = np.where(price > 0, np.floor(100000.0 / price).astype(int), 0)

    total_val = (qty * price).round(2)

    # Install/update quantity column
    qty_col = _quantity_column_name(df)
    if qty_col in df.columns:
        df[qty_col] = qty
    else:
        # insert right after Price column
        insert_at = list(df.columns).index(price_col) + 1
        df.insert(insert_at, qty_col, qty)

    # Install/update total value column
    tv_col = _total_value_column_name(df)
    if tv_col in df.columns:
        df[tv_col] = total_val
    else:
        insert_at = list(df.columns).index(qty_col) + 1
        df.insert(insert_at, tv_col, total_val)

    # Backup then write
    if BACKUP_BEFORE_OVERWRITE:
        bdir = os.path.join(os.path.dirname(path), "_backup_papertrade_5min_v3")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(path, os.path.join(bdir, os.path.basename(path)))

    df.to_csv(path, index=False)
    print(f"[OK]    {path} -> updated {len(df)} rows "
          f"(qty='{qty_col}', total='{tv_col}', price='{price_col}')")

def main():
    pattern = os.path.join(CSV_DIR, "papertrade_5min_v3_*.csv")
    files = sorted(glob.glob(pattern))
    if LIMIT_TO_DATES:
        files = [f for f in files if _matches_listed_dates(f, dates)]

    if not files:
        print("[INFO] No matching files found.")
        return

    print(f"[INFO] Processing {len(files)} file(s)...")
    for f in files:
        update_file(f)

if __name__ == "__main__":
    main()
