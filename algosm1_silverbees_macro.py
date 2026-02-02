# -*- coding: utf-8 -*-
"""
Creates the missing macro_inputs files for your strategy:

- macro_inputs/MCX_SILVER_RS_KG.csv
- macro_inputs/MCX_GOLD_RS_10G.csv
- macro_inputs/USDINR.csv
- macro_inputs/DXY.csv
- macro_inputs/US_REAL_YIELD.csv
- macro_inputs/EVENT_RISK.csv

Data sources (robust):
- Stooq: XAUUSD, XAGUSD (spot proxies)
- FRED:  DTWEXBGS (broad USD index proxy for DXY)
- FRED:  DFII10   (10Y real yield proxy)

No yfinance used (avoids JSONDecodeError / empty Yahoo responses).
"""

import os
from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np

MACRO_DIR = Path("macro_inputs")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

# Conversions
TROY_OZ_TO_G = 31.1034768

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def _get_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def _stooq_url(symbol: str) -> str:
    # Stooq download endpoint
    # Example: https://stooq.com/q/d/l/?s=xauusd&i=d
    return f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"

def download_stooq_daily(symbol: str) -> pd.DataFrame:
    """
    Returns DF with columns: date, close (daily)
    """
    url = _stooq_url(symbol)
    text = _get_text(url)
    if not text or text.strip() == "":
        raise RuntimeError(f"Stooq empty response for {symbol}")

    df = pd.read_csv(pd.io.common.StringIO(text))
    if df.empty:
        raise RuntimeError(f"Stooq returned empty CSV for {symbol}")

    # Stooq usually: Date,Open,High,Low,Close,Volume
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols:
        date_col = cols["date"]
    elif "data" in cols:  # sometimes localized
        date_col = cols["data"]
    else:
        date_col = df.columns[0]

    close_col = cols.get("close", None)
    if close_col is None:
        raise RuntimeError(f"Stooq unexpected columns for {symbol}: {df.columns.tolist()}")

    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"])
    out = out.set_index("date")
    return out

def _fred_graph_csv_url(series_id: str) -> str:
    # Works without API key when series exists
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def download_fred_series(series_id: str) -> pd.Series:
    """
    Returns Series indexed by date with float values.
    Handles both formats:
    - DATE,VALUE
    - observation_date,SERIESID
    """
    url = _fred_graph_csv_url(series_id)
    text = _get_text(url)

    df = pd.read_csv(pd.io.common.StringIO(text))
    if df.empty or df.shape[1] < 2:
        raise RuntimeError(f"FRED returned empty/unexpected CSV for {series_id}")

    # Normalize date column
    if "DATE" in df.columns:
        dcol = "DATE"
    elif "observation_date" in df.columns:
        dcol = "observation_date"
    else:
        dcol = df.columns[0]

    # Value column: either 'VALUE' or series_id
    if "VALUE" in df.columns:
        vcol = "VALUE"
    elif series_id in df.columns:
        vcol = series_id
    else:
        vcol = df.columns[1]

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)

    s = pd.Series(df[vcol].values, index=df[dcol].values, name=series_id)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def write_macro_csv(name: str, s: pd.Series):
    """
    Writes: macro_inputs/{name}.csv with columns date,value
    date as YYYY-MM-DD (daily). Your strategy localizes to IST anyway.
    """
    s = s.dropna()
    df = pd.DataFrame({"date": pd.to_datetime(s.index).date.astype(str), "value": s.values})
    out_path = MACRO_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(df)}")

def build_usdinr() -> pd.Series:
    """
    Prefer Stooq if available (often works well for USDINR),
    fallback to FRED DEXINUS if needed.
    """
    # Try Stooq
    for sym in ["usdinr", "inrusd"]:
        try:
            df = download_stooq_daily(sym)
            # Heuristic: USDINR should be around 70-90; INRUSD around 0.01-0.02
            s = df["close"].astype(float)
            if s.median() < 1.0:  # likely INRUSD -> invert
                s = 1.0 / (s + 1e-12)
            s.name = "USDINR"
            return s
        except Exception:
            pass

    # Fallback: FRED USD to INR (DEXINUS)
    s = download_fred_series("DEXINUS").rename("USDINR")
    return s

def build_dxy_proxy() -> pd.Series:
    """
    Use FRED DTWEXBGS as a DXY-style broad USD proxy.
    """
    return download_fred_series("DTWEXBGS").rename("DXY")

def build_real_yield() -> pd.Series:
    """
    Use FRED DFII10 (10Y TIPS real yield) as US real yield proxy.
    """
    return download_fred_series("DFII10").rename("US_REAL_YIELD")

def build_mcx_proxies_from_spot(usdinr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Build:
    - MCX_SILVER_RS_KG from XAGUSD (USD/oz) * USDINR * (1000g/kg / 31.103g/oz)
    - MCX_GOLD_RS_10G from XAUUSD (USD/oz) * USDINR * (10g / 31.103g/oz)
    """
    xag = download_stooq_daily("xagusd")["close"].astype(float).rename("XAGUSD")
    xau = download_stooq_daily("xauusd")["close"].astype(float).rename("XAUUSD")

    # align on common dates
    df = pd.concat([usdinr, xag, xau], axis=1).dropna()
    if df.empty:
        raise RuntimeError("No overlapping dates between USDINR and XAU/XAG data")

    mcx_silver_rs_kg = df["XAGUSD"] * df["USDINR"] * (1000.0 / TROY_OZ_TO_G)
    mcx_gold_rs_10g  = df["XAUUSD"] * df["USDINR"] * (10.0   / TROY_OZ_TO_G)

    mcx_silver_rs_kg.name = "MCX_SILVER_RS_KG"
    mcx_gold_rs_10g.name  = "MCX_GOLD_RS_10G"
    return mcx_silver_rs_kg, mcx_gold_rs_10g

def build_event_risk_template(date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Create a default EVENT_RISK series = 0 for all days.
    You can later edit macro_inputs/EVENT_RISK.csv (0..9).
    """
    dates = pd.to_datetime(pd.Series(date_index.date).unique())
    s = pd.Series(0, index=dates, name="EVENT_RISK").sort_index()
    return s

def main():
    print("[MACRO] Building macro_inputs files...")

    # USDINR
    usdinr = build_usdinr()
    write_macro_csv("USDINR", usdinr)

    # DXY proxy
    try:
        dxy = build_dxy_proxy()
        write_macro_csv("DXY", dxy)
    except Exception as e:
        print(f"[WARN] DXY proxy failed: {e}")

    # US real yield
    try:
        realy = build_real_yield()
        write_macro_csv("US_REAL_YIELD", realy)
    except Exception as e:
        print(f"[WARN] US_REAL_YIELD failed: {e}")

    # MCX proxies from spot
    try:
        mcx_silver, mcx_gold = build_mcx_proxies_from_spot(usdinr)
        write_macro_csv("MCX_SILVER_RS_KG", mcx_silver)
        write_macro_csv("MCX_GOLD_RS_10G", mcx_gold)
    except Exception as e:
        print(f"[WARN] MCX proxy build failed: {e}")

    # EVENT_RISK template
    try:
        base_idx = usdinr.index
        event = build_event_risk_template(base_idx)
        write_macro_csv("EVENT_RISK", event)
    except Exception as e:
        print(f"[WARN] EVENT_RISK template failed: {e}")

    print("[DONE] macro_inputs ready.")

if __name__ == "__main__":
    # small polite delay can reduce temporary blocks
    time.sleep(0.2)
    main()
