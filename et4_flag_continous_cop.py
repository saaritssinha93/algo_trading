#!/usr/bin/env python3
# refresh_flags_loop.py
# ─────────────────────────────────────────────────────────────
#   every 15 s
#     • recompute “flag” in each *_smlitvl.csv
#       – write back only if at least one flag value changed
#     • build yes_positive_flag.csv / yes_negative_flag.csv
#       – overwrite only if *content* changed
# ─────────────────────────────────────────────────────────────
import os, time, glob, logging, pandas as pd

# ───────── CONFIG ─────────
OUT_DIR   = "main_indicators_smlitvl"
POS_FILE  = "yes_positive_flag.csv"
NEG_FILE  = "yes_negative_flag.csv"
SLEEP_SEC = 15

SUMMARY_COLS = [
    "ticker", "date",                       # identifiers
    "open", "high", "low", "close", "volume",
    "Daily_Change", "Change_2min",
    "RSI_14", "MFI_14", "ADX_14",
    "OBV", "Force_Index"
]

os.makedirs("logs", exist_ok=True)
log = logging.getLogger("flag_loop")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
log.addHandler(logging.FileHandler("logs/refresh_flags_loop.log", encoding="utf-8"))
log.addHandler(logging.StreamHandler())


# ───────── INTERNAL UTILS ─────────
def _dataframe_equal(df_new: pd.DataFrame, csv_path: str) -> bool:
    """Return True if *csv_path* exists and has identical data/ordering."""
    if not os.path.exists(csv_path):
        return False
    try:
        df_old = pd.read_csv(csv_path, parse_dates=["date"] if "date" in df_new.columns else None)
    except Exception:
        return False
    # force identical column order + index before compare
    df_old  = df_old.loc[:, df_new.columns].reset_index(drop=True)
    df_new2 = df_new.reset_index(drop=True)
    return df_old.equals(df_new2)


def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """In-place numeric coercion (non-numeric → NaN)"""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# ───────── FLAG RECOMPUTE ─────────
def recompute_flag(csv_path: str) -> None:
    fn = os.path.basename(csv_path)

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], engine="python", on_bad_lines="skip")
    except Exception as e:
        log.error("read FAIL  %-40s  %s", fn, e)
        return

    if df.empty:
        log.warning("skip       %-40s (empty file)", fn)
        return

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    old_flag = df["flag"].copy() if "flag" in df.columns else pd.Series([pd.NA] * len(df))

    # sanity-check required columns
    req = ["Daily_Change", "Change_2min", "Force_Index", "RSI_14", "ADX_14", "MFI_14"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        log.error("%s missing %s – skip", fn, ", ".join(missing))
        return
    _safe_to_numeric(df, req)

    # ── filters ──
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

    fi_ema = df["Force_Index"].ewm(span=2, adjust=False).mean()
    long_ok  = (fi_ema > 0) & (fi_ema.shift(1) <= 0) & (fi_ema > fi_ema.shift(1))
    short_ok = (fi_ema < 0) & (fi_ema.shift(1) >= 0) & (fi_ema < fi_ema.shift(1))

    # assign flags
    df["flag"] = "NO"
    df.loc[(up_core & long_ok) | (dn_core & short_ok), "flag"] = "YES"

    # write back only if anything changed
    if old_flag.equals(df["flag"]):
        log.info("unchanged    %-40s", fn)
        return

    base = [
        "date", "open", "high", "low", "close", "volume",
        "Daily_Change", "Change_2min", "flag",
        "RSI_14", "ADX_14", "MFI_14", "OBV", "Force_Index"
    ]
    extra = [c for c in df.columns if c not in base]
    try:
        df.to_csv(csv_path, index=False, columns=base + extra)
        log.info("updated      %-40s rows=%d", fn, len(df))
    except Exception as e:
        log.error("write FAIL   %-40s  %s", fn, e)


# ───────── SUMMARY BUILDERS ─────────
def _write_summary(df_new: pd.DataFrame, out_path: str, label: str) -> None:
    if df_new.empty:
        if os.path.exists(out_path):
            os.remove(out_path)
            log.info("%s → none (file removed)", label)
        else:
            log.info("%s → none", label)
        return

    # column order guarantee
    df_ordered = df_new.reindex(columns=[c for c in SUMMARY_COLS if c in df_new.columns])

    if _dataframe_equal(df_ordered, out_path):
        log.info("%s unchanged – not writing %s", label, out_path)
        return

    try:
        df_ordered.to_csv(out_path, index=False)
        log.info("%s rows → %s (tickers=%d)", label, out_path, len(df_ordered))
    except Exception as e:
        log.error("%s write FAIL  %s", label, e)


def collect_yes_rows(csv_files: list[str]) -> None:
    pos_rows, neg_rows = [], []

    for fp in csv_files:
        try:
            df = pd.read_csv(fp, parse_dates=["date"], engine="python", on_bad_lines="skip")
        except Exception as e:
            log.warning("collect skip %-40s (%s)", os.path.basename(fp), e)
            continue
        if "flag" not in df.columns:
            continue

        yes = df[df["flag"] == "YES"]
        if yes.empty:
            continue

        latest = yes.sort_values("date").tail(1).copy()
        latest.insert(0, "ticker", os.path.basename(fp).split("_")[0])

        if latest["Daily_Change"].iloc[0] >= 0:
            pos_rows.append(latest)
        else:
            neg_rows.append(latest)

    # concat & sort (stable ordering -> reliable comparisons)
    pos_df = (pd.concat(pos_rows, ignore_index=True, sort=False)
                .sort_values(["ticker", "date"])
                .reset_index(drop=True)) if pos_rows else pd.DataFrame()
    neg_df = (pd.concat(neg_rows, ignore_index=True, sort=False)
                .sort_values(["ticker", "date"])
                .reset_index(drop=True)) if neg_rows else pd.DataFrame()

    _write_summary(pos_df, POS_FILE, "POS")
    _write_summary(neg_df, NEG_FILE, "NEG")


# ───────── MAIN LOOP ─────────
def run_loop() -> None:
    pattern = os.path.join(OUT_DIR, "*_smlitvl.csv")
    while True:
        csv_files = glob.glob(pattern)
        if not csv_files:
            log.warning("no *_smlitvl.csv files found – sleeping")
            time.sleep(SLEEP_SEC)
            continue

        for fp in csv_files:
            recompute_flag(fp)

        collect_yes_rows(csv_files)
        time.sleep(SLEEP_SEC)


# ───────── ENTRY ─────────
if __name__ == "__main__":
    log.info("flag loop started – refresh every %ds", SLEEP_SEC)
    try:
        run_loop()
    except KeyboardInterrupt:
        log.info("stopped by user")
