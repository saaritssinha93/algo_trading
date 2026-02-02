# build_algo_watchlist.py

import os
import glob
from datetime import datetime
from typing import List, Set

import pandas as pd
import pytz

# --- CONFIG ---

IST = pytz.timezone("Asia/Kolkata")

SIGNALS_TRADE_DATE_DIR = "signals_trade_date"
SIGNALS_TODAY_DIR      = "signals_today"

TICKER_TXT_PATH = "ticker.txt"   # output file


def get_today_ymd_ist() -> str:
    """
    Get today's date in IST as YYYYMMDD string.
    """
    now_ist = datetime.now(IST)
    today_start = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    return today_start.strftime("%Y%m%d")


def _collect_tickers_from_files(file_patterns: list[str]) -> Set[str]:
    """
    Given a list of glob patterns, read all matching CSV files and
    collect 'ticker' values into a set (uppercased, stripped).
    """
    tickers: Set[str] = set()

    for pattern in file_patterns:
        paths = sorted(glob.glob(pattern))
        if not paths:
            # Not fatal – maybe one side has no files for the day
            print(f"[INFO] No files found for pattern: {pattern}")
            continue

        for path in paths:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
                continue

            if "ticker" not in df.columns:
                print(f"[WARN] No 'ticker' column in {path}, skipping.")
                continue

            # Normalize tickers
            col = (
                df["ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace("", pd.NA)
                .dropna()
            )

            tickers.update(col.tolist())

            print(f"[INFO] Read {len(col)} tickers from {os.path.basename(path)}")

    return tickers


def build_algo_watchlist() -> List[str]:
    """
    Main function:
      - determines today's ymd based on IST
      - reads:
            signals_trade_date/signals_{ymd}_*.csv
            signals_today/signals_today_{ymd}_*.csv
      - collects unique tickers
      - writes them to ticker.txt (one per line)
      - returns the sorted list of tickers
    """
    ymd = get_today_ymd_ist()
    print(f"[INFO] Using trade date (IST) ymd = {ymd}")

    patterns = [
        os.path.join(SIGNALS_TRADE_DATE_DIR, f"signals_{ymd}_*.csv"),
        os.path.join(SIGNALS_TODAY_DIR,      f"signals_today_{ymd}_*.csv"),
    ]

    tickers_set = _collect_tickers_from_files(patterns)

    if not tickers_set:
        print("[WARN] No tickers found from today's signal files.")
        # Still write an empty file to keep things predictable
        open(TICKER_TXT_PATH, "w").close()
        return []

    # Sort for consistency
    tickers_list = sorted(tickers_set)

    # Write to ticker.txt
    with open(TICKER_TXT_PATH, "w", encoding="utf-8") as f:
        for t in tickers_list:
            f.write(f"{t}\n")

    print(f"[DONE] Wrote {len(tickers_list)} tickers to {TICKER_TXT_PATH}")
    return tickers_list


if __name__ == "__main__":
    algo_watchlist_symbols = build_algo_watchlist()
    print("\nAlgo watchlist symbols:")
    print(", ".join(algo_watchlist_symbols))
