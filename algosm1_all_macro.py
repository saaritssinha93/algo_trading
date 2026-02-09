# -*- coding: utf-8 -*-
"""
Unified MACRO builder for ETF strategies (Silver + Gold + Global regime).

WRITES (daily, columns: date,value) into:  macro_inputs/

Core:
- USDINR.csv
- DXY.csv                  (proxy via FRED DTWEXBGS)
- US_REAL_YIELD.csv        (FRED DFII10)

Metals (spot proxies -> INR proxies):
- XAUUSD.csv               (optional diagnostic, USD/oz)
- XAGUSD.csv               (optional diagnostic, USD/oz)
- MCX_GOLD_RS_10G.csv       = XAUUSD * USDINR * (10g / 31.1034768)
- MCX_SILVER_RS_KG.csv      = XAGUSD * USDINR * (1000g / 31.1034768)
- GOLD_SPOT_INR_10G.csv     (alias of MCX_GOLD_RS_10G)
- SILVER_SPOT_INR_KG.csv    (alias of MCX_SILVER_RS_KG)

Extras (global regime library; extend as you like):
- US10Y.csv, CURVE_T10Y3M.csv, VIX.csv, WTI.csv (via FRED)

Template:
- EVENT_RISK.csv           (0..9, default 0)

Sources (robust, NO yfinance):
- Stooq CSV endpoint: XAUUSD, XAGUSD, USDINR (fallback to INRUSD then invert)
- FRED graph CSV endpoint: DTWEXBGS, DFII10, etc. (no API key required)

NOTES:
- These are proxies (spot-based). They won’t exactly match MCX futures settlements.
- Many India-local macros (CPI/WPI/IIP/FII flows/India VIX) are not reliably scrapable for free.
  Best practice: drop those as manual CSVs into macro_inputs/ and your strategy can asof-merge them.
"""

from __future__ import annotations

from pathlib import Path
import time
import random
import requests
import pandas as pd
import numpy as np
from io import StringIO

# ---------------- CONFIG ----------------

MACRO_DIR = Path("macro_inputs")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

TROY_OZ_TO_G = 31.1034768

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
WRITE_DIAGNOSTIC_USD_METALS = True

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# FRED library (add/remove as desired)
FRED_SERIES = {
#    "DXY": "DTWEXBGS",             # broad trade-weighted USD proxy (DXY-style)
    "US_REAL_YIELD": "DFII10",     # 10Y TIPS real yield
    "US10Y": "DGS10",              # nominal 10Y
    "CURVE_T10Y3M": "T10Y3M",      # curve slope
    "VIX": "VIXCLS",               # VIX close
    "WTI": "DCOILWTICO",           # WTI oil
    # Optional examples:
    # "SPX": "SP500",
}

    
    # --- Classic DXY components from FRED (H.10 daily rates) ---
# Note: these are "noon buying rates" style series from the Fed.
DXY_FRED_FX = {
    "EURUSD": "DEXUSEU",  # U.S. Dollars to One Euro  (USD per EUR)
    "USDJPY": "DEXJPUS",  # Japanese Yen to One U.S. Dollar (JPY per USD)
    "GBPUSD": "DEXUSUK",  # U.S. Dollars to One U.K. Pound (USD per GBP)
    "USDCAD": "DEXCAUS",  # Canadian Dollars to One U.S. Dollar (CAD per USD)
    "USDSEK": "DEXSDUS",  # Swedish Kronor to One U.S. Dollar (SEK per USD)
    "USDCHF": "DEXSZUS",  # Swiss Francs to One U.S. Dollar (CHF per USD)
}

# ---------------- HTTP HELPERS ----------------
# ---------------- DXY (Stooq DX.F) ----------------

def build_dxy_synthetic_from_stooq_fx() -> pd.Series:
    """
    Build synthetic DXY from the classic basket using Stooq FX pairs.
    DXY = 50.14348112 *
          EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119 *
          USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
    """
    eurusd = download_stooq_daily_close("eurusd").rename("EURUSD").astype(float)
    usdjpy = download_stooq_daily_close("usdjpy").rename("USDJPY").astype(float)
    gbpusd = download_stooq_daily_close("gbpusd").rename("GBPUSD").astype(float)
    usdcad = download_stooq_daily_close("usdcad").rename("USDCAD").astype(float)
    usdsek = download_stooq_daily_close("usdsek").rename("USDSEK").astype(float)
    usdchf = download_stooq_daily_close("usdchf").rename("USDCHF").astype(float)

    df = pd.concat([eurusd, usdjpy, gbpusd, usdcad, usdsek, usdchf], axis=1).dropna()
    if df.empty:
        raise RuntimeError("No overlapping dates across DXY FX legs")

    dxy = (
        50.14348112 *
        (df["EURUSD"] ** -0.576) *
        (df["USDJPY"] **  0.136) *
        (df["GBPUSD"] ** -0.119) *
        (df["USDCAD"] **  0.091) *
        (df["USDSEK"] **  0.042) *
        (df["USDCHF"] **  0.036)
    )
    dxy.name = "DXY"
    return dxy.sort_index()




def _sleep_jitter(base: float = 0.2) -> None:
    time.sleep(base + random.random() * 0.25)

def _get_text(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    last_err = None
    for k in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            txt = r.text
            if not txt or not txt.strip():
                raise RuntimeError("Empty response")
            return txt
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (2 ** k))
    raise RuntimeError(f"Failed GET: {url} | last_err={last_err}")

# ---------------- STOOQ ----------------

def _stooq_url(symbol: str, interval: str = "d") -> str:
    # Example: https://stooq.com/q/d/l/?s=xauusd&i=d
    return f"https://stooq.com/q/d/l/?s={symbol.lower()}&i={interval}"

def download_stooq_daily_close(symbol: str) -> pd.Series:
    """
    Returns Series indexed by date with float close.
    """
    url = _stooq_url(symbol, "d")
    text = _get_text(url)
    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise RuntimeError(f"Stooq returned empty CSV for {symbol}")

    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("data") or df.columns[0]
    close_col = cols.get("close", None)
    if close_col is None:
        raise RuntimeError(f"Stooq unexpected columns for {symbol}: {df.columns.tolist()}")

    out = df[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])

    s = pd.Series(out["close"].values, index=out["date"].values, name=symbol.upper()).dropna()
    s = s[~pd.Index(s.index).duplicated(keep="last")].sort_index()
    return s

# ---------------- FRED ----------------

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

    # date column
    if "DATE" in df.columns:
        dcol = "DATE"
    elif "observation_date" in df.columns:
        dcol = "observation_date"
    else:
        dcol = df.columns[0]

    # value column
    if "VALUE" in df.columns:
        vcol = "VALUE"
    elif series_id in df.columns:
        vcol = series_id
    else:
        vcol = df.columns[1]

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)

    s = pd.Series(df[vcol].values, index=df[dcol].values, name=series_id).dropna()
    s = s[~pd.Index(s.index).duplicated(keep="last")].sort_index()
    return s

# ---------------- WRITER ----------------

def write_macro_csv(name: str, s: pd.Series) -> Path | None:
    """
    Writes macro_inputs/{name}.csv with columns date,value (YYYY-MM-DD)
    """
    s = s.dropna()
    if s.empty:
        print(f"[WARN] {name}: empty series, not writing.")
        return None
    df = pd.DataFrame({"date": pd.to_datetime(s.index).date.astype(str), "value": s.values})
    out_path = MACRO_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(df)}")
    return out_path

# ---------------- BUILDERS ----------------

def _sanity_check_usdinr(s: pd.Series) -> pd.Series:
    s = s.dropna().astype(float)
    if s.empty:
        raise RuntimeError("USDINR series is empty")
    med = float(s.median())
    if med < 10 or med > 200:
        raise RuntimeError(f"USDINR sanity check failed (median={med:.2f}). Source likely wrong.")
    return s

def build_usdinr() -> pd.Series:
    """
    Prefer Stooq (usdinr or inrusd), then FRED fallbacks.
    """
    for sym in ["usdinr", "inrusd"]:
        try:
            s = download_stooq_daily_close(sym).astype(float)
            if float(s.median()) < 1.0:  # likely INRUSD
                s = 1.0 / (s + 1e-12)
            s = s.rename("USDINR")
            return _sanity_check_usdinr(s)
        except Exception as e:
            print(f"[WARN] USDINR via Stooq '{sym}' failed: {e}")

    for fred_id in ["DEXINUS", "DEXINRUS"]:
        try:
            s = download_fred_series(fred_id).rename("USDINR").astype(float)
            if float(s.dropna().median()) < 1.0:
                s = 1.0 / (s + 1e-12)
            return _sanity_check_usdinr(s)
        except Exception as e:
            print(f"[WARN] USDINR via FRED '{fred_id}' failed: {e}")

    raise RuntimeError("Failed to build USDINR from Stooq and FRED fallbacks.")

def build_spot_usd_metals() -> tuple[pd.Series, pd.Series]:
    """
    XAUUSD, XAGUSD in USD/oz (spot proxies)
    """
    xau = download_stooq_daily_close("xauusd").rename("XAUUSD").astype(float)
    xag = download_stooq_daily_close("xagusd").rename("XAGUSD").astype(float)
    return xau, xag

def build_inr_spot_proxies(usdinr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    SILVER_SPOT_INR_KG = XAGUSD * USDINR * (1000g/kg / 31.103g/oz)
    GOLD_SPOT_INR_10G  = XAUUSD * USDINR * (10g / 31.103g/oz)
    """
    xau, xag = build_spot_usd_metals()
    df = pd.concat([usdinr.rename("USDINR"), xau, xag], axis=1).dropna()
    if df.empty:
        raise RuntimeError("No overlapping dates between USDINR and XAU/XAG data")

    silver_inr_kg = df["XAGUSD"] * df["USDINR"] * (1000.0 / TROY_OZ_TO_G)
    gold_inr_10g  = df["XAUUSD"] * df["USDINR"] * (10.0   / TROY_OZ_TO_G)

    silver_inr_kg.name = "SILVER_SPOT_INR_KG"
    gold_inr_10g.name  = "GOLD_SPOT_INR_10G"
    return silver_inr_kg, gold_inr_10g

def build_event_risk_template(date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Default EVENT_RISK series = 0 for all days. (Edit later 0..9)
    """
    dates = pd.to_datetime(pd.Series(date_index.date).unique())
    return pd.Series(0, index=dates, name="EVENT_RISK").sort_index()

# ---------------- MAIN ----------------

def main():
    print(f"[MACRO] Writing into: {MACRO_DIR.resolve()}")
    print("[MACRO] Building macro_inputs files...")

    _sleep_jitter()

    # 1) USDINR
    usdinr = build_usdinr()
    write_macro_csv("USDINR", usdinr)

    # 2) DXY (FIXED SOURCE)
    try:
        dxy = build_dxy_synthetic_from_stooq_fx()
        write_macro_csv("DXY", dxy)
    except Exception as e:
        print(f"[WARN] DXY build failed: {e}")

    # 3) FRED macro library (everything except DXY)
    for out_name, series_id in FRED_SERIES.items():
        try:
            s = download_fred_series(series_id).rename(out_name)
            write_macro_csv(out_name, s)
            _sleep_jitter(0.12)
        except Exception as e:
            print(f"[WARN] FRED {out_name} ({series_id}) failed: {e}")

    # 4) Metals diagnostics
    if WRITE_DIAGNOSTIC_USD_METALS:
        try:
            xau, xag = build_spot_usd_metals()
            write_macro_csv("XAUUSD", xau)
            write_macro_csv("XAGUSD", xag)
        except Exception as e:
            print(f"[WARN] XAUUSD/XAGUSD download failed: {e}")

    # 5) INR proxies
    try:
        silver_inr_kg, gold_inr_10g = build_inr_spot_proxies(usdinr)

        write_macro_csv("MCX_SILVER_RS_KG", silver_inr_kg.rename("MCX_SILVER_RS_KG"))
        write_macro_csv("MCX_GOLD_RS_10G",  gold_inr_10g.rename("MCX_GOLD_RS_10G"))

        write_macro_csv("SILVER_SPOT_INR_KG", silver_inr_kg)
        write_macro_csv("GOLD_SPOT_INR_10G",  gold_inr_10g)

    except Exception as e:
        print(f"[WARN] INR proxy build failed: {e}")

    # 6) EVENT_RISK template
    try:
        event = build_event_risk_template(usdinr.index)
        write_macro_csv("EVENT_RISK", event)
    except Exception as e:
        print(f"[WARN] EVENT_RISK template failed: {e}")

    print("[DONE] macro_inputs ready.")


if __name__ == "__main__":
    time.sleep(0.2)
    main()
