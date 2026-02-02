# -*- coding: utf-8 -*-
"""
ONE-SHOT (TODAY ONLY, ROW-BY-ROW):
  1) (optional) update 15m parquet once
  2) build merged 15m+daily+weekly like backtest
  3) filter to TODAY (IST)
  4) evaluate mask ROW-BY-ROW
  5) keep ONLY FIRST signal per ticker
  6) print + save CSV (ticker is 2nd column)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
import pytz
import pandas as pd

from kiteconnect import KiteConnect

import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_etfsonly as core
import algosm1_trading_signal_profit_analysis_etf_15m_v7_parquet as bt

IST = pytz.timezone("Asia/Kolkata")

UPDATE_15M_FIRST = True
ONLY_FIRST_SIGNAL_PER_TICKER = True

ROOT = Path(__file__).resolve().parent
API_KEY_FILE = ROOT / "api_key.txt"
ACCESS_TOKEN_FILE = ROOT / "access_token.txt"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def read_first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise ValueError(f"{path} is empty")


def setup_kite_session_fixed() -> KiteConnect:
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE).split()[0]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


core.setup_kite_session = setup_kite_session_fixed


def now_ist() -> datetime:
    return datetime.now(IST)


def run_update_15m_once(holidays: set) -> None:
    core.run_mode(
        mode="15min",
        context="live",
        max_workers=4,
        force_today_daily=False,
        skip_if_fresh=True,
        intraday_ts="end",
        holidays=holidays,
        refresh_tokens=False,
        report_dir=str(REPORTS_DIR),
        print_missing_rows=False,
        print_missing_rows_max=200,
    )


def list_tickers_intersection() -> list[str]:
    def tickers_in(dirpath: Path, suffix: str) -> set[str]:
        if not dirpath.exists():
            return set()
        return {p.name.replace(suffix, "").upper() for p in dirpath.glob(f"*{suffix}")}

    d15 = ROOT / bt.DIR_15M
    dd  = ROOT / bt.DIR_D
    dw  = ROOT / bt.DIR_W

    return sorted(
        tickers_in(d15, "_etf_indicators_15min.parquet")
        & tickers_in(dd, "_etf_indicators_daily.parquet")
        & tickers_in(dw, "_etf_indicators_weekly.parquet")
    )


def build_merged_df(ticker: str) -> pd.DataFrame:
    p15 = ROOT / bt.DIR_15M / f"{ticker}_etf_indicators_15min.parquet"
    pd_ = ROOT / bt.DIR_D   / f"{ticker}_etf_indicators_daily.parquet"
    pw  = ROOT / bt.DIR_W   / f"{ticker}_etf_indicators_weekly.parquet"

    df_i_raw = bt.read_tf_parquet(str(p15))
    df_d_raw = bt.read_tf_parquet(str(pd_))
    df_w_raw = bt.read_tf_parquet(str(pw))

    if df_i_raw.empty or df_d_raw.empty or df_w_raw.empty:
        return pd.DataFrame()

    if "Daily_Change" in df_d_raw.columns:
        df_d_raw = df_d_raw.copy()
        df_d_raw["Daily_Change_prev"] = df_d_raw["Daily_Change"].shift(1)

    df_i = bt.suffix_columns(df_i_raw, "_I")
    df_d = bt.suffix_columns(df_d_raw, "_D")
    df_w = bt.suffix_columns(df_w_raw, "_W")

    df_i = bt.add_intraday_features(df_i)

    df_idw = bt.merge_daily_and_weekly_onto_intraday(df_i, df_d, df_w)
    return df_idw


def scan_today_row_by_row(ticker: str, today: date) -> pd.DataFrame:
    df_idw = build_merged_df(ticker)
    if df_idw.empty or "date" not in df_idw.columns:
        return pd.DataFrame()

    # Ensure tz-aware IST for filtering
    df_idw = df_idw.copy()
    df_idw["date"] = pd.to_datetime(df_idw["date"], errors="coerce")
    if getattr(df_idw["date"].dt, "tz", None) is None:
        df_idw["date"] = df_idw["date"].dt.tz_localize(IST)
    else:
        df_idw["date"] = df_idw["date"].dt.tz_convert(IST)

    df_today = df_idw[df_idw["date"].dt.date == today].copy()
    if df_today.empty:
        return pd.DataFrame()

    df_today = df_today.sort_values("date").reset_index(drop=True)

    hits = []
    for i in range(len(df_today)):
        row_df = df_today.iloc[[i]]
        ok = bool(bt.compute_long_signal_mask(row_df).iloc[0])
        if ok:
            r = row_df.iloc[0]
            hits.append({
                "signal_time_ist": r["date"],
                "ticker": ticker,               # ticker as 2nd column
                "entry_price": r.get("close_I"),
                "ingested_at_ist": now_ist(),
            })
            if ONLY_FIRST_SIGNAL_PER_TICKER:
                break

    return pd.DataFrame(hits)


def main() -> None:
    today = now_ist().date()
    holidays = core._read_holidays(core.HOLIDAYS_FILE_DEFAULT)

    if UPDATE_15M_FIRST:
        run_update_15m_once(holidays)

    tickers = list_tickers_intersection()

    out_all = []
    for t in tickers:
        try:
            df = scan_today_row_by_row(t, today)
            if not df.empty:
                out_all.append(df)
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not out_all:
        print("[DONE] No signals today (row-by-row).")
        return

    out = pd.concat(out_all, ignore_index=True)
    out = out.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    out_csv = REPORTS_DIR / f"today_signals_rowbyrow_{ts}.csv"
    out.to_csv(out_csv, index=False)

    print(out.to_string(index=False))
    print(f"\n[FILE SAVED] {out_csv}")


if __name__ == "__main__":
    main()
