# refresh_flags_once_first_yes.py
# ─────────────────────────────────────────────────────────────
# One-shot:
#   • rebuild “flag” in every *_smlitvl.csv inside OUT_DIR
#   • export the *first* YES row per ticker to
#         yes_positive_flag.csv   (bullish)
#         yes_negative_flag.csv   (bearish)
#   • add Highest_Daily_Change  (per-ticker extreme)
#   • add Delta = |Highest_Daily_Change − Daily_Change|
# ─────────────────────────────────────────────────────────────
import os, glob, logging, pandas as pd, numpy as np

OUT_DIR  = "main_indicators_smlitvl_his"
POS_FILE = "yes_positive_flag.csv"
NEG_FILE = "yes_negative_flag.csv"

os.makedirs("logs", exist_ok=True)
log = logging.getLogger("flag_once"); log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
log.addHandler(logging.FileHandler("logs/refresh_flags_once.log", encoding="utf-8"))
log.addHandler(logging.StreamHandler())

# ───────────────────────── flag rebuild ─────────────────────────
def recompute_flag(path: str) -> None:
    try:
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except pd.errors.ParserError:         # lenient fallback
            df = pd.read_csv(path, parse_dates=["date"],
                             engine="python", on_bad_lines="skip")

        if df.empty:
            log.warning("skip %-35s (empty)", os.path.basename(path)); return

        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

        # numeric conversion
        for c in ("Daily_Change", "Change_2min", "Force_Index",
                  "RSI_14", "ADX_14"):
            if c not in df.columns:
                log.error("%s missing '%s' – skipped",
                          os.path.basename(path), c)
                return
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # ------------- build conditions -------------
        up_core = (
            (df["Daily_Change"] >= 3.0) &
            df["Change_2min"].between(0.75, 1.25, inclusive="neither") &
            (df["Change_2min"].shift(1) > 0)
            &(df["RSI_14"] >= 30) & (df["RSI_14"] <= 85)
            &(df["MFI_14"] >= 40) & (df["MFI_14"] <= 85)
            &(df["ADX_14"] >= 18)
        )

        down_core = (
            (df["Daily_Change"] <= -3.0) &
            df["Change_2min"].between(-1.25, -0.75, inclusive="neither") &
            (df["Change_2min"].shift(1) < 0)
            &(df["RSI_14"] >= 15) & (df["RSI_14"] <= 70)
            &(df["MFI_14"] >= 15) & (df["MFI_14"] <= 60)
            &(df["ADX_14"] >= 18)
        )

        fi_cur   = df["Force_Index"]
        fi_prev  = fi_cur.shift(1)
        fi_prev2 = fi_cur.shift(2)
        """
        up_fi   = (
            #(fi_prev  > 0) &
            #(fi_prev2 > 0) &
            #(fi_prev  > 1.10 * fi_prev2) &
            (fi_cur   > 1 * fi_prev)
        )

        down_fi = (
            #(fi_prev  < 0) &
            #(fi_prev2 < 0) &
            #(np.abs(fi_prev)  > 1.10 * np.abs(fi_prev2)) &
            (np.abs(fi_cur)   > 1 * np.abs(fi_prev))
        )
        """
        import numpy as np

      

        # 2) smooth with a 2-period EMA
        df['FI_EMA2'] = df['Force_Index'].ewm(span=2, adjust=False).mean()

        # 3) bullish / bearish signals
        df['FI_up_cross'] = (df['FI_EMA2'] > 0) & (df['FI_EMA2'].shift(1) <= 0)
        df['FI_dn_cross'] = (df['FI_EMA2'] < 0) & (df['FI_EMA2'].shift(1) >= 0)

        # 4) two-bar confirmation (optional)
        df['FI_up_confirm'] = df['FI_EMA2'] > df['FI_EMA2'].shift(1)
        df['FI_dn_confirm'] = df['FI_EMA2'] < df['FI_EMA2'].shift(1)

        # Final entry signals
        df['long_signal']  = df['FI_up_cross']  & df['FI_up_confirm']
        df['short_signal'] = df['FI_dn_cross'] & df['FI_dn_confirm']

        df["flag"] = "NO"
        df.loc[(up_core & df['long_signal']) | (down_core &  df['short_signal']), "flag"] = "YES"
        #df.loc[(up_core) | (down_core), "flag"] = "YES"

        base_cols = ["date","open","high","low","close","volume",
                     "Daily_Change","Change_2min","flag",
                     "RSI_14","ADX_14","MFI_14","OBV","Force_Index"]
        extra = [c for c in df.columns if c not in base_cols]
        df.to_csv(path, index=False, columns=base_cols + extra)
        log.info("updated %-35s rows=%d", os.path.basename(path), len(df))

    except Exception as e:
        log.error("FAILED %-35s  %s", os.path.basename(path), e)

# ───────────────────── collect first-YES rows ─────────────────────
def collect_yes_rows(files: list[str]) -> None:
    pos_rows, neg_rows = [], []

    for fp in files:
        try:
            df = pd.read_csv(fp, parse_dates=["date"],
                             engine="python", on_bad_lines="skip")
            if "flag" not in df.columns:
                continue

            df["Daily_Change"] = pd.to_numeric(df["Daily_Change"],
                                               errors="coerce")

            yes = df[df["flag"] == "YES"]
            if yes.empty:
                continue

            # earliest YES row
            row = yes.sort_values("date").iloc[0].copy()
            row["ticker"] = os.path.basename(fp).split("_")[0]

            # highest (abs) daily change in full file
            max_change = df["Daily_Change"].max(skipna=True)
            min_change = df["Daily_Change"].min(skipna=True)
            row["Highest_Daily_Change"] = (max_change if row["Daily_Change"] >= 0
                                           else min_change)
            row["Delta"] = abs(row["Highest_Daily_Change"] - row["Daily_Change"])

            (pos_rows if row["Daily_Change"] >= 0 else neg_rows).append(row)

        except Exception as e:
            log.warning("collect skip %s (%s)", os.path.basename(fp), e)

    def _emit(rows, outfile, tag):
        if rows:
            pd.DataFrame(rows).to_csv(outfile, index=False)
            log.info("%s rows → %s  (tickers=%d)", tag, outfile, len(rows))
        else:
            log.info("%s rows → none", tag)

    _emit(pos_rows, POS_FILE, "POS")
    _emit(neg_rows, NEG_FILE, "NEG")

# ─────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    csvs = glob.glob(os.path.join(OUT_DIR, "*_smlitvl.csv"))
    if not csvs:
        log.error("no CSVs found in %s", OUT_DIR)
        raise SystemExit

    for fp in csvs:
        recompute_flag(fp)

    collect_yes_rows(csvs)
