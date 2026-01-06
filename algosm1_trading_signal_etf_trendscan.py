# -*- coding: utf-8 -*-
"""
Multi-timeframe UP trend scan for ETFs.

For each ETF ticker that has:
    • Monthly indicators in main_indicators_etf_monthly
    • Weekly  indicators in main_indicators_etf_weekly
    • Daily   indicators in main_indicators_etf_daily
    • 15-min  indicators in main_indicators_etf_15min

This script:
    1) Reads the latest row from MONTHLY, WEEKLY, DAILY,
       and the last ~10 rows from 15-MIN.
    2) Applies "upward trend" logic on each timeframe:
         - MONTHLY_UP: structural long-term uptrend
               · close > EMA_50 > EMA_200
               · RSI >= 50
               · Monthly_Change >= +1.0% (if available)
               · (optional) ADX >= 15 if present
         - WEEKLY_UP: medium-term uptrend
               · close > EMA_50 > EMA_200
               · RSI >= 50
               · (optional) ADX >= 15 if present
         - DAILY_UP: shorter-term uptrend
               · close > EMA_50 > EMA_200
               · RSI >= 52
               · Daily_Change >= +0.5% (if available)
         - INTRADAY_15M_UP: recent 15m move up
               · last 10 bars: close_end vs close_start slope >= +0.3%
               · last close above 50 EMA and above 200 EMA (if both exist)
               · RSI_15m >= 55
    3) Writes CSVs:
         (a) All tickers:
             trend_scans/etf_multi_tf_trend_scan_<YYYYMMDD_HHMM>_IST.csv
         (b) ALL 4 TFs up (M+W+D+15m):
             trend_scans/etf_multi_tf_trend_scan_UPONLY_<YYYYMMDD_HHMM>_IST.csv
         (c) MONTHLY + WEEKLY + DAILY up:
             trend_scans/etf_multi_tf_trend_scan_MONTHLY_WEEKLY_DAILY_UP_<YYYYMMDD_HHMM>_IST.csv
         (d) MONTHLY + WEEKLY up:
             trend_scans/etf_multi_tf_trend_scan_MONTHLY_WEEKLY_UP_<YYYYMMDD_HHMM>_IST.csv
         (e) WEEKLY + DAILY up:
             trend_scans/etf_multi_tf_trend_scan_WEEKLY_DAILY_UP_<YYYYMMDD_HHMM>_IST.csv
         (f) WEEKLY + DAILY + 15m up:
             trend_scans/etf_multi_tf_trend_scan_WEEKLY_DAILY_15M_UP_<YYYYMMDD_HHMM>_IST.csv
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_M   = "main_indicators_etf_monthly"
DIR_W   = "main_indicators_etf_weekly"
DIR_D   = "main_indicators_etf_daily"
DIR_15  = "main_indicators_etf_15min"   # assumes files like <TICKER>_main_indicators.csv
OUT_DIR = "trend_scans"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------- HELPERS -----------------------
def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """
    Read a timeframe CSV, parse 'date' as tz-aware IST, sort.
    """
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(tz)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()


def _list_tickers_from_dir(directory: str, ending: str) -> set[str]:
    """
    List tickers in a directory by stripping a known suffix from filenames.
    Example:
        directory = main_indicators_etf_weekly
        ending    = "_main_indicators_etf_weekly.csv"
    """
    pattern = os.path.join(directory, f"*{ending}")
    files = glob.glob(pattern)
    tickers = set()
    for f in files:
        base = os.path.basename(f)
        if base.endswith(ending):
            ticker = base[: -len(ending)]
            if ticker:
                tickers.add(ticker)
    return tickers


def _safe_get(row, col):
    """
    Safely get a float value from a row (Series); returns np.nan if missing.
    """
    try:
        return float(row.get(col, np.nan))
    except Exception:
        return np.nan


# ----------------------- TREND CHECK LOGIC -----------------------
def is_monthly_up(last_m: pd.Series) -> bool:
    """
    Monthly UP definition:
        • close > EMA_50 > EMA_200
        • RSI >= 50
        • Monthly_Change >= +1.0% (if present)
        • ADX >= 15 (if ADX column exists)
    """
    close = _safe_get(last_m, "close")
    ema50 = _safe_get(last_m, "EMA_50")
    ema200 = _safe_get(last_m, "EMA_200")
    rsi = _safe_get(last_m, "RSI")
    mchg = _safe_get(last_m, "Monthly_Change")
    adx  = _safe_get(last_m, "ADX")

    if not np.isfinite(close) or not np.isfinite(ema50) or not np.isfinite(ema200) or not np.isfinite(rsi):
        return False

    cond_trend = (close > ema50) and (ema50 > ema200)
    cond_rsi   = rsi >= 50.0

    cond_mchg = True
    if np.isfinite(mchg):
        cond_mchg = mchg >= 1.0   # at least +1% vs previous month

    cond_adx = True
    if np.isfinite(adx):
        cond_adx = adx >= 15.0

    return cond_trend and cond_rsi and cond_mchg and cond_adx


def is_weekly_up(last_w: pd.Series) -> bool:
    """
    Weekly UP definition:
        • close > EMA_50 > EMA_200
        • RSI >= 50
        • ADX >= 15 (if ADX column exists)
    """
    close = _safe_get(last_w, "close")
    ema50 = _safe_get(last_w, "EMA_50")
    ema200 = _safe_get(last_w, "EMA_200")
    rsi = _safe_get(last_w, "RSI")
    adx = _safe_get(last_w, "ADX")

    if not np.isfinite(close) or not np.isfinite(ema50) or not np.isfinite(ema200) or not np.isfinite(rsi):
        return False

    cond_trend = (close > ema50) and (ema50 > ema200)
    cond_rsi = rsi >= 50.0
    cond_adx = True
    if np.isfinite(adx):
        cond_adx = adx >= 15.0

    return cond_trend and cond_rsi and cond_adx


def is_daily_up(last_d: pd.Series) -> bool:
    """
    Daily UP definition:
        • close > EMA_50 > EMA_200
        • RSI >= 52
        • Daily_Change >= +0.5% (if present)
    """
    close = _safe_get(last_d, "close")
    ema50 = _safe_get(last_d, "EMA_50")
    ema200 = _safe_get(last_d, "EMA_200")
    rsi = _safe_get(last_d, "RSI")
    dchg = _safe_get(last_d, "Daily_Change")

    if not np.isfinite(close) or not np.isfinite(ema50) or not np.isfinite(ema200) or not np.isfinite(rsi):
        return False

    cond_trend = (close > ema50) and (ema50 > ema200)
    cond_rsi   = rsi >= 52.0

    cond_dchg = True
    if np.isfinite(dchg):
        cond_dchg = dchg >= 0.5  # at least +0.5% previous day

    return cond_trend and cond_rsi and cond_dchg


def is_15min_up(df_15: pd.DataFrame) -> tuple[bool, float]:
    """
    15-minute UP definition:
        • Use last up-to-10 bars
        • slope = (last_close / first_close - 1) * 100 >= +0.3%
        • last_close above EMA_50
        • last_close above EMA_200 (if present)
        • last RSI >= 55

    Returns:
        (bool_is_up, slope_pct_last_10_bars)
    """
    if df_15.empty:
        return False, np.nan

    tail = df_15.tail(10).copy()
    if len(tail) < 3:
        # Not enough bars to judge intraday trend
        return False, np.nan

    first = tail.iloc[0]
    last  = tail.iloc[-1]

    start_close = _safe_get(first, "close")
    end_close   = _safe_get(last, "close")
    if not (np.isfinite(start_close) and np.isfinite(end_close) and start_close > 0):
        return False, np.nan

    slope_pct = (end_close / start_close - 1.0) * 100.0

    ema50 = _safe_get(last, "EMA_50")
    ema200 = _safe_get(last, "EMA_200")
    rsi   = _safe_get(last, "RSI")

    cond_slope = slope_pct >= 0.3
    cond_ema   = True
    if np.isfinite(ema50):
        cond_ema = cond_ema and (end_close >= ema50)
    if np.isfinite(ema200):
        cond_ema = cond_ema and (end_close >= ema200)

    cond_rsi = np.isfinite(rsi) and (rsi >= 55.0)

    is_up = cond_slope and cond_ema and cond_rsi
    return is_up, slope_pct


# ----------------------- MAIN SCAN -----------------------
def main():
    # 1) Find common tickers present in MONTHLY, WEEKLY, DAILY, and 15-MIN dirs
    tickers_m  = _list_tickers_from_dir(DIR_M,  "_main_indicators_etf_monthly.csv")
    tickers_w  = _list_tickers_from_dir(DIR_W,  "_main_indicators_etf_weekly.csv")
    tickers_d  = _list_tickers_from_dir(DIR_D,  "_main_indicators_etf_daily.csv")
    tickers_15 = _list_tickers_from_dir(DIR_15, "_main_indicators.csv")  # from ETF 15min fetcher

    tickers = tickers_m & tickers_w & tickers_d & tickers_15
    if not tickers:
        print("No common tickers found across Monthly, Weekly, Daily, and 15-min directories.")
        return

    print(f"Found {len(tickers)} tickers with Monthly + Weekly + Daily + 15-min data.")

    rows = []

    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Checking trends for {ticker} ...")

        path_m  = os.path.join(DIR_M,  f"{ticker}_main_indicators_etf_monthly.csv")
        path_w  = os.path.join(DIR_W,  f"{ticker}_main_indicators_etf_weekly.csv")
        path_d  = os.path.join(DIR_D,  f"{ticker}_main_indicators_etf_daily.csv")
        path_15 = os.path.join(DIR_15, f"{ticker}_main_indicators.csv")

        df_m  = _safe_read_tf(path_m)
        df_w  = _safe_read_tf(path_w)
        df_d  = _safe_read_tf(path_d)
        df_15 = _safe_read_tf(path_15)

        if df_m.empty or df_w.empty or df_d.empty or df_15.empty:
            print(f"  - Skipping {ticker}: missing one or more TF files.")
            continue

        last_m = df_m.iloc[-1]
        last_w = df_w.iloc[-1]
        last_d = df_d.iloc[-1]

        monthly_up  = is_monthly_up(last_m)
        weekly_up   = is_weekly_up(last_w)
        daily_up    = is_daily_up(last_d)
        intraday_15m_up, slope_15m = is_15min_up(df_15)

        rows.append(
            {
                "ticker": ticker,

                # MONTHLY snapshot
                "monthly_last_date": last_m["date"],
                "monthly_up": monthly_up,
                "monthly_close": _safe_get(last_m, "close"),
                "monthly_rsi": _safe_get(last_m, "RSI"),
                "monthly_ema50": _safe_get(last_m, "EMA_50"),
                "monthly_ema200": _safe_get(last_m, "EMA_200"),
                "monthly_adx": _safe_get(last_m, "ADX"),
                "monthly_change_pct": _safe_get(last_m, "Monthly_Change"),

                # WEEKLY snapshot
                "weekly_last_date": last_w["date"],
                "weekly_up": weekly_up,
                "weekly_close": _safe_get(last_w, "close"),
                "weekly_rsi": _safe_get(last_w, "RSI"),
                "weekly_ema50": _safe_get(last_w, "EMA_50"),
                "weekly_ema200": _safe_get(last_w, "EMA_200"),
                "weekly_adx": _safe_get(last_w, "ADX"),

                # DAILY snapshot
                "daily_last_date": last_d["date"],
                "daily_up": daily_up,
                "daily_close": _safe_get(last_d, "close"),
                "daily_rsi": _safe_get(last_d, "RSI"),
                "daily_ema50": _safe_get(last_d, "EMA_50"),
                "daily_ema200": _safe_get(last_d, "EMA_200"),
                "daily_change_pct": _safe_get(last_d, "Daily_Change"),

                # 15-min snapshot
                "min15_last_time": df_15.iloc[-1]["date"],
                "intraday_15m_up": intraday_15m_up,
                "min15_last_close": _safe_get(df_15.iloc[-1], "close"),
                "min15_last_rsi": _safe_get(df_15.iloc[-1], "RSI"),
                "min15_last_ema50": _safe_get(df_15.iloc[-1], "EMA_50"),
                "min15_last_ema200": _safe_get(df_15.iloc[-1], "EMA_200"),
                "min15_slope_last_10bars_pct": slope_15m,

                # Combined flags
                "all_timeframes_up": monthly_up and weekly_up and daily_up and intraday_15m_up,
                "higher_tfs_up":     monthly_up and weekly_up and daily_up,  # ignore 15m
            }
        )

    if not rows:
        print("No usable tickers processed.")
        return

    out_df = pd.DataFrame(rows)
    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")

    # (a) Full scan
    out_path_all = os.path.join(OUT_DIR, f"etf_multi_tf_trend_scan_{ts}_IST.csv")
    out_df.to_csv(out_path_all, index=False)
    print(f"\nSaved full trend scan for {len(out_df)} tickers to: {out_path_all}")

    # (b) ALL 4 (M+W+D+15m) up
    up_df = out_df[out_df["all_timeframes_up"]].copy()
    if not up_df.empty:
        out_path_up = os.path.join(
            OUT_DIR, f"etf_multi_tf_trend_scan_UPONLY_{ts}_IST.csv"
        )
        up_df.to_csv(out_path_up, index=False)
        print(
            f"Saved UP-only trend scan (Monthly+Weekly+Daily+15m up) "
            f"for {len(up_df)} tickers to: {out_path_up}"
        )
    else:
        print("No tickers with all_timeframes_up == True, UP-only CSV not created.")

    # (c) MONTHLY + WEEKLY + DAILY up
    mwd_df = out_df[out_df["higher_tfs_up"]].copy()
    if not mwd_df.empty:
        out_path_mwd = os.path.join(
            OUT_DIR, f"etf_multi_tf_trend_scan_MONTHLY_WEEKLY_DAILY_UP_{ts}_IST.csv"
        )
        mwd_df.to_csv(out_path_mwd, index=False)
        print(
            f"Saved MONTHLY+WEEKLY+DAILY-up scan for {len(mwd_df)} tickers to: {out_path_mwd}"
        )
    else:
        print("No tickers with monthly_up & weekly_up & daily_up all True, MONTHLY_WEEKLY_DAILY_UP CSV not created.")

    # (d) MONTHLY + WEEKLY up
    mw_df = out_df[(out_df["monthly_up"]) & (out_df["weekly_up"])].copy()
    if not mw_df.empty:
        out_path_mw = os.path.join(
            OUT_DIR, f"etf_multi_tf_trend_scan_MONTHLY_WEEKLY_UP_{ts}_IST.csv"
        )
        mw_df.to_csv(out_path_mw, index=False)
        print(
            f"Saved MONTHLY+WEEKLY-up scan for {len(mw_df)} tickers to: {out_path_mw}"
        )
    else:
        print("No tickers with monthly_up & weekly_up both True, MONTHLY_WEEKLY_UP CSV not created.")

    # (e) WEEKLY + DAILY up
    wd_df = out_df[(out_df["weekly_up"]) & (out_df["daily_up"])].copy()
    if not wd_df.empty:
        out_path_wd = os.path.join(
            OUT_DIR, f"etf_multi_tf_trend_scan_WEEKLY_DAILY_UP_{ts}_IST.csv"
        )
        wd_df.to_csv(out_path_wd, index=False)
        print(
            f"Saved WEEKLY+DAILY-up scan for {len(wd_df)} tickers to: {out_path_wd}"
        )
    else:
        print("No tickers with weekly_up & daily_up both True, WEEKLY_DAILY_UP CSV not created.")

    # (f) WEEKLY + DAILY + 15m up
    wd15_df = out_df[(out_df["weekly_up"]) & (out_df["daily_up"]) & (out_df["intraday_15m_up"])].copy()
    if not wd15_df.empty:
        out_path_wd15 = os.path.join(
            OUT_DIR, f"etf_multi_tf_trend_scan_WEEKLY_DAILY_15M_UP_{ts}_IST.csv"
        )
        wd15_df.to_csv(out_path_wd15, index=False)
        print(
            f"Saved WEEKLY+DAILY+15m-up scan for {len(wd15_df)} tickers to: {out_path_wd15}"
        )
    else:
        print("No tickers with weekly_up & daily_up & 15m_up all True, WEEKLY_DAILY_15M_UP CSV not created.")

    # Some console samples
    print("\nSample (first 20 rows from full scan):")
    print(out_df.head(20).to_string(index=False))

    if not up_df.empty:
        print("\nSample (all_timeframes_up == True):")
        print(up_df.head(20).to_string(index=False))

    if not mwd_df.empty:
        print("\nSample (monthly_up & weekly_up & daily_up == True):")
        print(mwd_df.head(20).to_string(index=False))

    if not mw_df.empty:
        print("\nSample (monthly_up & weekly_up == True):")
        print(mw_df.head(20).to_string(index=False))

    if not wd_df.empty:
        print("\nSample (weekly_up & daily_up == True):")
        print(wd_df.head(20).to_string(index=False))

    if not wd15_df.empty:
        print("\nSample (weekly_up & daily_up & intraday_15m_up == True):")
        print(wd15_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
