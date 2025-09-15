from pathlib import Path
import os, logging
import pandas as pd

# —————————————————— CONFIG ——————————————————
BASE_DIR = Path(r"C:\Users\Saarit\OneDrive\Desktop\Trading\et4\trading_strategies_algo")
SRC_DIR  = BASE_DIR / "main_indicators_smlitvl_his"       # <— L  before .csv
DEST_DIR = BASE_DIR / "main_indicators_smlitvl_his_main"

PATTERN  = "*_smlitvl.csv"            # adjust here if your naming is different

COLS = [
    "date", "open", "high", "low", "close", "volume",
    "Daily_Change", "Change_2min", "flag",
    "RSI_14", "ADX_14", "MFI_14", "OBV", "Force_Index",
    "Prev_Close", "EMA_5", "EMA_20",
    "MACD_Line", "MACD_Signal", "MACD_Hist",
    "Stoch_%K", "Stoch_%D", "VWAP"
]

DEST_DIR.mkdir(parents=True, exist_ok=True)

# —————————————————— LOGGING ——————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
log = logging.getLogger("copy_smlitvl")

def read_csv_safe(path: Path) -> pd.DataFrame:
    """Robust CSV loader that swallows bad rows and drops phantom columns."""
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip"          # ignore malformed rows
    )
    # strip any unnamed junk
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df

# —————————————————— MAIN ——————————————————
files = list(SRC_DIR.glob(PATTERN))
if not files:
    log.error("No files matching %s in %s", PATTERN, SRC_DIR)
    raise SystemExit(1)

for src in files:
    dest = DEST_DIR / src.name
    try:
        df_src  = read_csv_safe(src)
        df_dest = read_csv_safe(dest) if dest.exists() else pd.DataFrame()

        df   = pd.concat([df_dest, df_src], ignore_index=True).drop_duplicates()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)

        # keep canonical order, tack on any extras to the end
        ordered = [c for c in COLS if c in df.columns]
        extras  = [c for c in df.columns if c not in ordered]
        df = df[ordered + extras]

        df.to_csv(dest, index=False)
        log.info("✔ %s → appended %d new rows (total %d)",
                 src.name, len(df_src), len(df))

    except Exception as e:
        log.error("✖ %s failed: %s", src.name, e)

log.info("Done! %d file(s) processed.", len(files))
