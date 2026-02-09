# -*- coding: utf-8 -*-
"""
Macro builder for SILVER + GOLD strategies.

WRITES (daily, columns: date,value):
- macro_inputs/USDINR.csv
- macro_inputs/DXY.csv                  (proxy via FRED DTWEXBGS)
- macro_inputs/US_REAL_YIELD.csv        (FRED DFII10)
- macro_inputs/MCX_GOLD_RS_10G.csv      (spot proxy from XAUUSD * USDINR)
- macro_inputs/MCX_SILVER_RS_KG.csv     (spot proxy from XAGUSD * USDINR)
- macro_inputs/GOLD_SPOT_INR_10G.csv    (alias of MCX_GOLD_RS_10G, clearer naming)
- macro_inputs/SILVER_SPOT_INR_KG.csv   (alias of MCX_SILVER_RS_KG, clearer naming)
- macro_inputs/XAUUSD.csv               (optional diagnostic)
- macro_inputs/XAGUSD.csv               (optional diagnostic)
- macro_inputs/EVENT_RISK.csv           (template 0..9, default 0)

SOURCES (robust, no yfinance):
- Stooq: XAUUSD, XAGUSD, USDINR (fallback to INRUSD then invert)
- FRED:  DTWEXBGS (broad USD index proxy), DFII10 (10Y real yield)

NOTES:
- These are proxies (spot-based). They won't exactly match MCX futures settlement.
- USDINR fallback from FRED can be unreliable depending on series availability; script includes sanity checks.
"""

from __future__ import annotations

from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np
from io import StringIO

MACRO_DIR = Path("macro_inputs")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

TROY_OZ_TO_G = 31.1034768

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# ---------------- IO + DOWNLOAD HELPERS ----------------

def _get_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def _stooq_url(symbol: str, interval: str = "d") -> str:
    # Example: https://stooq.com/q/d/l/?s=xauusd&i=d
    return f"https://stooq.com/q/d/l/?s={symbol.lower()}&i={interval}"

def download_stooq_daily(symbol: str) -> pd.DataFrame:
    """
    Returns DF indexed by date with column: close
    """
    url = _stooq_url(symbol, "d")
    text = _get_text(url)
    if not text or not text.strip():
        raise RuntimeError(f"Stooq empty response for {symbol}")

    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise RuntimeError(f"Stooq returned empty CSV for {symbol}")

    # Stooq usually: Date,Open,High,Low,Close,Volume
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", None) or cols.get("data", None) or df.columns[0]
    close_col = cols.get("close", None)
    if close_col is None:
        raise RuntimeError(f"Stooq unexpected columns for {symbol}: {df.columns.tolist()}")

    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    out = out.set_index("date")
    return out

def _fred_graph_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def download_fred_series(series_id: str) -> pd.Series:
    """
    Returns Series indexed by date with float values.
    Handles formats:
    - DATE,VALUE
    - observation_date,SERIESID
    """
    url = _fred_graph_csv_url(series_id)
    text = _get_text(url)

    df = pd.read_csv(StringIO(text))
    if df.empty or df.shape[1] < 2:
        raise RuntimeError(f"FRED returned empty/unexpected CSV for {series_id}")

    # Normalize date column
    if "DATE" in df.columns:
        dcol = "DATE"
    elif "observation_date" in df.columns:
        dcol = "observation_date"
    else:
        dcol = df.columns[0]

    # Value column
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
    Writes macro_inputs/{name}.csv with columns date,value (YYYY-MM-DD)
    """
    s = s.dropna()
    if s.empty:
        print(f"[WARN] {name}: empty series, not writing.")
        return

    df = pd.DataFrame({"date": pd.to_datetime(s.index).date.astype(str), "value": s.values})
    out_path = MACRO_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(df)}")

# ---------------- SERIES BUILDERS ----------------

def _sanity_check_usdinr(s: pd.Series) -> pd.Series:
    s = s.dropna().astype(float)
    if s.empty:
        raise RuntimeError("USDINR series is empty")

    med = float(s.median())
    # rough sanity
    if med < 10 or med > 200:
        raise RuntimeError(f"USDINR sanity check failed (median={med:.2f}). Source likely wrong.")
    return s

def build_usdinr() -> pd.Series:
    """
    Prefer Stooq if available (often works well for USDINR),
    fallback to other if needed.
    """
    # Try Stooq (sometimes usdinr exists, sometimes inrusd exists)
    for sym in ["usdinr", "inrusd"]:
        try:
            df = download_stooq_daily(sym)
            s = df["close"].astype(float)

            # If INRUSD (~0.01-0.02), invert
            if float(s.median()) < 1.0:
                s = 1.0 / (s + 1e-12)

            s = s.rename("USDINR")
            s = _sanity_check_usdinr(s)
            return s
        except Exception as e:
            print(f"[WARN] USDINR via Stooq '{sym}' failed: {e}")

    # Fallback: try a couple of FRED IDs (availability may vary)
    for fred_id in ["DEXINUS", "DEXINRUS"]:
        try:
            s = download_fred_series(fred_id).rename("USDINR")
            # Some FRED series might be INR per USD already; if it's < 1 -> invert
            if float(s.dropna().median()) < 1.0:
                s = 1.0 / (s + 1e-12)
            s = _sanity_check_usdinr(s)
            return s
        except Exception as e:
            print(f"[WARN] USDINR via FRED '{fred_id}' failed: {e}")

    raise RuntimeError("Failed to build USDINR from Stooq and FRED fallbacks.")

def build_dxy_proxy() -> pd.Series:
    """
    Use FRED DTWEXBGS as a broad USD index proxy (DXY-style).
    """
    return download_fred_series("DTWEXBGS").rename("DXY")

def build_real_yield() -> pd.Series:
    """
    Use FRED DFII10 (10Y TIPS real yield) as US real yield proxy.
    """
    return download_fred_series("DFII10").rename("US_REAL_YIELD")

def build_spot_usd_metals() -> tuple[pd.Series, pd.Series]:
    """
    XAUUSD, XAGUSD (USD/oz)
    """
    xau = download_stooq_daily("xauusd")["close"].astype(float).rename("XAUUSD")
    xag = download_stooq_daily("xagusd")["close"].astype(float).rename("XAGUSD")
    return xau, xag

def build_inr_spot_proxies(usdinr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Build INR proxies using spot prices:

    SILVER_SPOT_INR_KG = XAGUSD (USD/oz) * USDINR * (1000g/kg / 31.103g/oz)
    GOLD_SPOT_INR_10G  = XAUUSD (USD/oz) * USDINR * (10g / 31.103g/oz)

    We also write them as MCX_* names because your strategy expects that naming.
    """
    xau, xag = build_spot_usd_metals()

    df = pd.concat([usdinr.rename("USDINR"), xau, xag], axis=1).dropna()
    if df.empty:
        raise RuntimeError("No overlapping dates between USDINR and XAU/XAG data")

    silver_inr_kg = df["XAGUSD"] * df["USDINR"] * (1000.0 / TROY_OZ_TO_G)
    gold_inr_10g = df["XAUUSD"] * df["USDINR"] * (10.0 / TROY_OZ_TO_G)

    silver_inr_kg.name = "SILVER_SPOT_INR_KG"
    gold_inr_10g.name = "GOLD_SPOT_INR_10G"

    return silver_inr_kg, gold_inr_10g

def build_event_risk_template(date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Default EVENT_RISK series = 0 for all days. (Edit later 0..9)
    """
    dates = pd.to_datetime(pd.Series(date_index.date).unique())
    return pd.Series(0, index=dates, name="EVENT_RISK").sort_index()

# ---------------- MAIN ----------------

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

    # Metals spot USD (optional diagnostics)
    try:
        xau, xag = build_spot_usd_metals()
        write_macro_csv("XAUUSD", xau)
        write_macro_csv("XAGUSD", xag)
    except Exception as e:
        print(f"[WARN] XAUUSD/XAGUSD download failed: {e}")

    # INR proxies for strategy (gold + silver)
    try:
        silver_inr_kg, gold_inr_10g = build_inr_spot_proxies(usdinr)

        # Strategy-expected names
        write_macro_csv("MCX_SILVER_RS_KG", silver_inr_kg.rename("MCX_SILVER_RS_KG"))
        write_macro_csv("MCX_GOLD_RS_10G", gold_inr_10g.rename("MCX_GOLD_RS_10G"))

        # Clear aliases (recommended)
        write_macro_csv("SILVER_SPOT_INR_KG", silver_inr_kg)
        write_macro_csv("GOLD_SPOT_INR_10G", gold_inr_10g)

    except Exception as e:
        print(f"[WARN] INR proxy build failed: {e}")

    # EVENT_RISK template
    try:
        base_idx = usdinr.index
        event = build_event_risk_template(base_idx)
        write_macro_csv("EVENT_RISK", event)
    except Exception as e:
        print(f"[WARN] EVENT_RISK template failed: {e}")

    print("[DONE] macro_inputs ready.")

if __name__ == "__main__":
    time.sleep(0.2)
    main()
