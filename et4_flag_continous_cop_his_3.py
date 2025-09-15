#!/usr/bin/env python3
# refresh_flags_once_first_yes_by_dates.py
# ─────────────────────────────────────────────────────────────
# One-shot helper
#
#   • Rebuild column  flag  in every  *_smlitvl.csv  under  OUT_DIR
#   • Take the *earliest* YES row for each <ticker, calendar-date> pair
#   • Split those rows into bullish / bearish buckets **per date**
#       →  <DD-MM-YYYY>_yes_positive_flag.csv
#       →  <DD-MM-YYYY>_yes_negative_flag.csv
#   • Add convenience columns
#         Highest_Daily_Change   (extreme value in that ticker file)
#         Delta                  = |Highest_Daily_Change − Daily_Change|
#
# Everything is defensive against bad data:
#   – “date” is force-parsed with errors='coerce'
#   – non-numeric strings in indicator columns are coerced to NaN
#   – rows with a NaT ‘date’ never reach any .dt accessor
# ─────────────────────────────────────────────────────────────
import os, glob, logging, pandas as pd, numpy as np
from datetime import datetime, date

# ─────────── USER CONFIG ───────────
DATES = ["05-06-2025"]                     # <— add more as needed (DD-MM-YYYY)
OUT_DIR = "main_indicators_smlitvl"

# ─────────── CONSTANTS ───────────
LOG_FILE = "logs/refresh_flags_once.log"
os.makedirs("logs", exist_ok=True)

DATE_OBJS = [datetime.strptime(d, "%d-%m-%Y").date() for d in DATES]
DATE_SET  = set(DATE_OBJS)                 # faster look-ups

REQUIRED_NUMERIC = [
    "Daily_Change", "Change_2min", "Force_Index",
    "RSI_14", "ADX_14", "MFI_14"
]

SUMMARY_BASE_COLS = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "Daily_Change", "Change_2min",
    "RSI_14", "MFI_14", "ADX_14",
    "OBV", "Force_Index",
    "Highest_Daily_Change", "Delta"
]

# ─────────── LOGGER ───────────
log = logging.getLogger("flag_once")
log.setLevel(logging.INFO)
h_fmt = logging.Formatter("%(asctime)s  %(levelname)s: %(message)s")
log.addHandler(logging.FileHandler(LOG_FILE, encoding="utf-8"))
log.addHandler(logging.StreamHandler())

# ───────────────── FLAG REBUILDER ─────────────────
def rebuild_flag(csv_path: str) -> None:
    """Re-compute the flag column for one *_smlitvl.csv (in-place)."""
    fn = os.path.basename(csv_path)

    # ---------- read ----------
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        log.error("read FAIL  %-40s  %s", fn, e)
        return
    if df.empty:
        log.warning("skip       %-40s (empty file)", fn)
        return

    # ---------- clean ----------
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # date → datetime64[ns]
    if "date" not in df.columns:
        log.error("%s missing 'date' column – skipped", fn)
        return
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])          # drop rows that failed to parse

    # ensure numerics
    for col in REQUIRED_NUMERIC:
        if col not in df.columns:
            log.error("%s missing '%s' – skipped", fn, col)
            return
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------- rules ----------
    up_core = (
        (df["Daily_Change"] >= 1) &
        df["Change_2min"].between(0.50, 2.00, inclusive="neither") &
        (df["Change_2min"].shift(1) > 0) &
        df["RSI_14"].between(50, 85) &
        df["MFI_14"].between(50, 85) &
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

    fi_ema  = df["Force_Index"].ewm(span=2, adjust=False).mean()
    long_ok  = (fi_ema > 0) & (fi_ema.shift(1) <= 0) & (fi_ema > fi_ema.shift(1))
    short_ok = (fi_ema < 0) & (fi_ema.shift(1) >= 0) & (fi_ema < fi_ema.shift(1))

    # ---------- assign ----------
    old_flag = df["flag"].copy() if "flag" in df.columns else pd.Series([pd.NA]*len(df))
    df["flag"] = "NO"
    df.loc[(up_core & long_ok) | (dn_core & short_ok), "flag"] = "YES"

    if old_flag.equals(df["flag"]):
        log.info("unchanged    %-40s", fn)
        return

    # ---------- persist ----------
    base_cols = [
        "date","open","high","low","close","volume",
        "Daily_Change","Change_2min","flag",
        "RSI_14","ADX_14","MFI_14","OBV","Force_Index"
    ]
    extra = [c for c in df.columns if c not in base_cols]
    try:
        df.to_csv(csv_path, index=False, columns=base_cols+extra)
        log.info("updated      %-40s rows=%d", fn, len(df))
    except Exception as e:
        log.error("write FAIL   %-40s  %s", fn, e)


# ───────────────── COLLECT YES ROWS ─────────────────
def collect_yes(csv_files: list[str]) -> None:
    """
    For each target date, take the *earliest* YES bar per ticker and
    write *_yes_positive_flag.csv / *_yes_negative_flag.csv files.
    """
    pos_bucket = {d: [] for d in DATE_OBJS}
    neg_bucket = {d: [] for d in DATE_OBJS}

    for fp in csv_files:
        try:
            df = pd.read_csv(
                fp,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e:
            log.warning("collect skip %-40s (%s)", os.path.basename(fp), e)
            continue
        if "flag" not in df.columns:
            continue

        # ---- date parsing & filtering ----
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            continue
        df["date_only"] = df["date"].dt.date
        df = df[df["date_only"].isin(DATE_SET)]
        if df.empty:
            continue

        # numeric safety
        df["Daily_Change"] = pd.to_numeric(df["Daily_Change"], errors="coerce")

        # ---- earliest YES per calendar date ----
        for d in DATE_OBJS:
            day_yes = df[(df["flag"] == "YES") & (df["date_only"] == d)]
            if day_yes.empty:
                continue
            first = day_yes.sort_values("date").iloc[0].copy()

            first_dict = first.to_dict()
            first_dict["ticker"] = os.path.basename(fp).split("_")[0]

            # Highest/Lowest Daily_Change in *entire file* for context
            max_c = df["Daily_Change"].max()
            min_c = df["Daily_Change"].min()
            first_dict["Highest_Daily_Change"] = max_c if first_dict["Daily_Change"] >= 0 else min_c
            first_dict["Delta"] = abs(first_dict["Highest_Daily_Change"] - first_dict["Daily_Change"])

            bucket = pos_bucket if first_dict["Daily_Change"] >= 0 else neg_bucket
            bucket[d].append(first_dict)

    # ---- emit CSVs ----
    def _emit(bucket: dict[date, list[dict]], tag: str):
        for d, rows in bucket.items():
            if not rows:
                continue
            out_name = f"{d.strftime('%d-%m-%Y')}_yes_{tag}_flag.csv"
            pd.DataFrame(rows)[SUMMARY_BASE_COLS].to_csv(out_name, index=False)
            log.info("%s  →  %s  (%d tickers)", tag.upper(), out_name, len(rows))

    _emit(pos_bucket, "positive")
    _emit(neg_bucket, "negative")


# ─────────────────────── SUMMARISER (optional) ───────────────────────
def summarize_flag_files(dir_path: str = ".") -> None:
    """Pretty print per-file stats for *_yes_positive_flag.csv files."""
    pattern = os.path.join(dir_path, "*_yes_positive_flag.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print("No *_yes_positive_flag.csv files found in", dir_path)
        return

    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"⚠️  Skipping {os.path.basename(fp)}  ({e})")
            continue
        total = len(df)
        delta = pd.to_numeric(df.get("Delta"), errors="coerce")
        gt2   = (delta > 2).sum()
        pct   = f"{(gt2/total)*100:4.1f} %" if total else "0.0 %"
        rows.append([os.path.basename(fp), total, gt2, pct])

    summary = pd.DataFrame(rows, columns=["File", "Records", "Delta > 2", "% > 2"])
    summary.loc["— TOTAL —"] = [
        "— TOTAL —",
        summary["Records"].sum(),
        summary["Delta > 2"].sum(),
        f"{(summary['Delta > 2'].sum()/summary['Records'].sum())*100:4.1f} %"
        if summary["Records"].sum() else "0.0 %"
    ]
    print("\n" + summary.to_string(index=False) + "\n")


# ─────────────────────── FILE MOVER (optional) ───────────────────────
from pathlib import Path, PurePath
import shutil

def move_yes_csv_files(src_dir: str | PurePath, dest_dir: str | PurePath = "YES") -> int:
    """
    Move all *_yes_*_flag.csv (non-recursive) from *src_dir* → *dest_dir*.
    Existing files at destination are silently overwritten.
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
        tgt = dest_dir / fp.name
        try:
            if tgt.exists():
                tgt.unlink()
            shutil.move(fp, tgt)
            moved += 1
        except Exception as e:
            logging.error("Failed to move %s → %s  (%s)", fp.name, dest_dir, e)

    logging.info("Moved %d file(s) to %s", moved, dest_dir)
    return moved


# ──────────────────────────── MAIN ────────────────────────────
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(OUT_DIR, "*_smlitvl.csv"))
    if not csv_files:
        log.error("No *_smlitvl.csv files found in %s", OUT_DIR)
        raise SystemExit

    for fp in csv_files:
        rebuild_flag(fp)

    collect_yes(csv_files)

    # Uncomment for a quick console summary immediately afterwards
    # summarize_flag_files()

    # Uncomment to auto-move new *_yes_* files into ./YES
    # move_yes_csv_files(".", "YES")
