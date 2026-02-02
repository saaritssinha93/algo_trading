# etf_run_once_update15m_and_print_all_signals.py
# One-shot:
#   1) (optional) update 15m parquet once (appends missing rows)
#   2) scan ALL available data and print ALL signals (backtest logic)
#   3) also save a CSV dump to reports/

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import pytz
import pandas as pd

from kiteconnect import KiteConnect

import algosm1_trading_data_continous_run_historical_alltf_v3_parquet_etfsonly as core
import algosm1_trading_signal_profit_analysis_etf_15m_v7_parquet as bt


IST = pytz.timezone("Asia/Kolkata")

# ---- set this True if you want to run the 15m updater first ----
UPDATE_15M_FIRST = True

# --- paths relative to this file (so Spyder working dir doesn't break it) ---
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
    raise ValueError(f"{path} is empty or whitespace-only")


def setup_kite_session_fixed() -> KiteConnect:
    """
    FIX for InvalidHeader:
    - api_key: take first non-empty token only
    - access_token: take first non-empty line only
    """
    api_key = read_first_nonempty_line(API_KEY_FILE).split()[0]
    access_token = read_first_nonempty_line(ACCESS_TOKEN_FILE).split()[0]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# Monkeypatch the core script so it uses the fixed session creator
core.setup_kite_session = setup_kite_session_fixed


def now_ist() -> datetime:
    return datetime.now(IST)


def run_update_15m_once(holidays: set) -> None:
    """
    Calls your existing updater. It appends missing 15m rows and writes parquet.
    """
    core.run_mode(
        mode="15min",
        context="live",
        max_workers=4,              # tune 2-6 depending on rate limits
        force_today_daily=False,
        skip_if_fresh=True,         # avoids rewriting if already up-to-date
        intraday_ts="end",          # candle END timestamps
        holidays=holidays,
        refresh_tokens=False,       # uses cached tokens unless forced
        report_dir=str(REPORTS_DIR),
        print_missing_rows=False,
        print_missing_rows_max=200,
    )


def list_tickers_intersection() -> list[str]:
    """
    Use backtest module’s configured dirs:
      etf_indicators_15min_pq / daily_pq / weekly_pq
    and return tickers present in all 3.
    """
    d15 = Path(ROOT / bt.DIR_15M)
    dd  = Path(ROOT / bt.DIR_D)
    dw  = Path(ROOT / bt.DIR_W)

    def tickers_in(dirpath: Path, suffix: str) -> set[str]:
        if not dirpath.exists():
            return set()
        out = set()
        for p in dirpath.glob(f"*{suffix}"):
            name = p.name
            t = name.replace(suffix, "")
            out.add(t.upper())
        return out

    s15 = tickers_in(d15, "_etf_indicators_15min.parquet")
    sd  = tickers_in(dd,  "_etf_indicators_daily.parquet")
    sw  = tickers_in(dw,  "_etf_indicators_weekly.parquet")

    tickers = sorted(list(s15 & sd & sw))
    return tickers


def signals_for_ticker_all_rows(ticker: str) -> pd.DataFrame:
    """
    Build the merged 15m+daily+weekly dataframe the SAME WAY as backtest,
    apply the SAME LONG mask, and return ALL rows where signal=True.
    """
    p15 = ROOT / bt.DIR_15M / f"{ticker}_etf_indicators_15min.parquet"
    pd_ = ROOT / bt.DIR_D   / f"{ticker}_etf_indicators_daily.parquet"
    pw  = ROOT / bt.DIR_W   / f"{ticker}_etf_indicators_weekly.parquet"

    df_i_raw = bt.read_tf_parquet(str(p15))
    df_d_raw = bt.read_tf_parquet(str(pd_))
    df_w_raw = bt.read_tf_parquet(str(pw))

    if df_i_raw.empty or df_d_raw.empty or df_w_raw.empty:
        return pd.DataFrame()

    # backtest expects Daily_Change_prev on daily, then suffixing
    if "Daily_Change" in df_d_raw.columns:
        df_d_raw["Daily_Change_prev"] = df_d_raw["Daily_Change"].shift(1)

    # suffix first (so add_intraday_features sees *_I columns)
    df_i = bt.suffix_columns(df_i_raw, "_I")
    df_d = bt.suffix_columns(df_d_raw, "_D")
    df_w = bt.suffix_columns(df_w_raw, "_W")

    # add helper intraday features exactly like backtest module
    df_i = bt.add_intraday_features(df_i)

    # merge daily + weekly onto intraday
    df_idw = bt.merge_daily_and_weekly_onto_intraday(df_i, df_d, df_w)
    if df_idw.empty:
        return pd.DataFrame()

    # apply SAME signal mask as backtest
    mask = bt.compute_long_signal_mask(df_idw)
    sig = df_idw.loc[mask].copy()
    if sig.empty:
        return pd.DataFrame()

    # add meta columns
    sig.insert(0, "ticker", ticker)
    sig["ingested_at_ist"] = now_ist()

    # convenience renames if you want
    if "date" in sig.columns:
        sig = sig.rename(columns={"date": "signal_time_ist"})
    if "close_I" in sig.columns:
        sig["entry_price"] = sig["close_I"]

    # keep key columns first (but keep ALL columns too)
    front = [c for c in ["ticker", "signal_time_ist", "entry_price", "ingested_at_ist"] if c in sig.columns]
    rest = [c for c in sig.columns if c not in front]
    sig = sig[front + rest]

    return sig


def main() -> None:
    print("[ONE-SHOT] 15m update + ALL signals scan")

    holidays = core._read_holidays(core.HOLIDAYS_FILE_DEFAULT)

    if UPDATE_15M_FIRST:
        print(f"[STEP] Updating 15m parquet once at {now_ist().strftime('%Y-%m-%d %H:%M:%S%z')}")
        run_update_15m_once(holidays)

    tickers = list_tickers_intersection()
    print(f"[STEP] Tickers with 15m+daily+weekly present: {len(tickers)}")

    all_sig = []
    for t in tickers:
        try:
            df_sig = signals_for_ticker_all_rows(t)
            if not df_sig.empty:
                all_sig.append(df_sig)
        except Exception as e:
            print(f"[WARN] {t}: failed signal build: {e}")

    if not all_sig:
        print("[DONE] No signals found in available data.")
        return

    out = pd.concat(all_sig, ignore_index=True)
    # sort if we have time column
    if "signal_time_ist" in out.columns:
        out = out.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    # save a CSV dump so you can view everything even if console is long
    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    out_csv = REPORTS_DIR / f"all_signals_dump_{ts}.csv"
    out.to_csv(out_csv, index=False)

    # print ALL signals (rows) but not necessarily all columns in console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 240)

    show_cols = [c for c in ["signal_time_ist", "ticker", "entry_price", "ingested_at_ist"] if c in out.columns]
    # if these exist, they help debugging
    debug_cols = [c for c in [
        "RSI_I", "MACD_Hist_I", "EMA_50_I", "EMA_200_I", "ADX_I",
        "VWAP_I", "ATR_I", "Recent_High_5_I", "Daily_Change_prev_D",
        "EMA_50_W", "EMA_200_W"
    ] if c in out.columns]
    cols = show_cols + [c for c in debug_cols if c not in show_cols]

    print("\n================= ALL SIGNALS =================")
    print(f"Total signals: {len(out)}")
    if cols:
        print(out[cols].to_string(index=False))
    else:
        print(out.to_string(index=False))

    print(f"\n[FILE] Full dump saved: {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
