import os
import io
from datetime import datetime
from typing import Dict, Iterable, Tuple, Set

import pandas as pd
import pytz
import requests
from kiteconnect import KiteConnect

IST = pytz.timezone("Asia/Kolkata")

API_KEY_TXT = "api_key.txt"
ACCESS_TOKEN_TXT = "access_token.txt"
OUT_DIR = "kite_exports"

# Official NSE ETF master list (authoritative)
NSE_ETF_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"


# ----------------- Helpers: read keys -----------------
def read_api_key(path=API_KEY_TXT) -> str:
    """
    Supports:
      A) KEY=VALUE format
      B) 5-line format (API_KEY is line1)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} next to this script.")

    lines = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]

    kv = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip().upper()] = v.strip()

    if kv.get("API_KEY"):
        return kv["API_KEY"]

    if len(lines) >= 1:
        return lines[0]

    raise ValueError("api_key.txt is empty or invalid.")


def read_access_token(path=ACCESS_TOKEN_TXT) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Generate it first.")
    token = open(path, "r", encoding="utf-8").read().strip()
    if not token:
        raise ValueError("access_token.txt is empty.")
    return token


# ----------------- Official ETF master -----------------
def download_nse_etf_master(url: str = NSE_ETF_MASTER_URL, timeout: int = 25) -> pd.DataFrame:
    """
    Downloads NSE official ETF list CSV and returns cleaned DataFrame.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]

    # Normalize expected cols
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    if "ISINNumber" in df.columns:
        df["ISINNumber"] = df["ISINNumber"].astype(str).str.upper().str.strip()

    # Keep only valid rows
    if "Symbol" in df.columns and "ISINNumber" in df.columns:
        df = df[(df["Symbol"] != "") & (df["ISINNumber"] != "")]
    return df


def build_etf_sets(etf_master_df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    etf_isins = set()
    etf_symbols = set()

    if "ISINNumber" in etf_master_df.columns:
        etf_isins = set(
            etf_master_df["ISINNumber"].dropna().astype(str).str.upper().str.strip()
        )
    if "Symbol" in etf_master_df.columns:
        etf_symbols = set(
            etf_master_df["Symbol"].dropna().astype(str).str.upper().str.strip()
        )

    etf_isins.discard("")
    etf_symbols.discard("")
    return etf_isins, etf_symbols


# ----------------- Utility -----------------
def symbols_by_exchange_from_df(df: pd.DataFrame) -> Dict[str, Set[str]]:
    out = {"NSE": set(), "BSE": set()}
    if df is None or df.empty:
        return out

    # Ensure columns exist
    ex_series = df["exchange"] if "exchange" in df.columns else pd.Series(["NSE"] * len(df))
    ts_series = df["tradingsymbol"] if "tradingsymbol" in df.columns else pd.Series([""] * len(df))

    ex_series = ex_series.astype(str).str.upper().str.strip()
    ts_series = ts_series.astype(str).str.upper().str.strip()

    for e, s in zip(ex_series, ts_series):
        if e in out and s:
            out[e].add(s)

    return out


# ----------------- Instruments -> ISIN lookup (FIXED) -----------------
def build_isin_lookup_for_symbols(
    kite: KiteConnect,
    symbols_by_exchange: Dict[str, Set[str]],
    exchanges: Iterable[str] = ("NSE", "BSE"),
) -> Dict[Tuple[str, str], str]:
    """
    Returns mapping: (exchange, tradingsymbol) -> isin
    Loads only the symbols you actually hold/position (fast).
    """
    lookup: Dict[Tuple[str, str], str] = {}

    for ex in exchanges:
        symset = set(symbols_by_exchange.get(ex, set()))
        if not symset:
            continue

        inst = kite.instruments(ex)  # list[dict]
        df = pd.DataFrame(inst)
        if df.empty:
            continue

        # Ensure required cols exist as SERIES (not strings)
        if "exchange" not in df.columns:
            df["exchange"] = ex
        if "tradingsymbol" not in df.columns:
            df["tradingsymbol"] = ""

        if "isin" not in df.columns:
            df["isin"] = ""   # <-- key fix: create a column, not df.get(...)

        df["exchange"] = df["exchange"].astype(str).str.upper().str.strip()
        df["tradingsymbol"] = df["tradingsymbol"].astype(str).str.upper().str.strip()
        df["isin"] = df["isin"].astype(str).str.upper().str.strip()

        # Filter only symbols needed
        df = df[(df["exchange"] == ex) & (df["tradingsymbol"].isin(symset))]

        for row in df[["exchange", "tradingsymbol", "isin"]].itertuples(index=False):
            isin = str(row.isin).upper().strip()
            if isin and isin != "NAN":
                lookup[(row.exchange, row.tradingsymbol)] = isin

    return lookup


# ----------------- Classification (100% via NSE ETF ISIN list) -----------------
def classify_df_equity_vs_etf(
    df: pd.DataFrame,
    etf_isins: Set[str],
    etf_symbols: Set[str],
    isin_lookup: Dict[Tuple[str, str], str],
) -> pd.DataFrame:
    """
    Adds:
      - isin_final
      - class: ETF / EQUITY_STOCK / UNKNOWN
    ETF identification is authoritative by ISIN in NSE ETF list.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    # Ensure cols
    if "exchange" not in out.columns:
        out["exchange"] = "NSE"
    if "tradingsymbol" not in out.columns:
        out["tradingsymbol"] = ""

    out["exchange"] = out["exchange"].astype(str).str.upper().str.strip()
    out["tradingsymbol"] = out["tradingsymbol"].astype(str).str.upper().str.strip()

    # Pick ISIN column if present in this df (holdings usually has it)
    isin_col = None
    for c in ("isin", "ISIN", "isin_number", "isinNumber"):
        if c in out.columns:
            isin_col = c
            break

    if isin_col:
        out["isin_final"] = out[isin_col].astype(str).str.upper().str.strip()
    else:
        out["isin_final"] = ""

    # Enrich missing ISIN from instruments dump
    def _fill_isin(row):
        isin = (row["isin_final"] or "").upper().strip()
        if isin and isin != "NAN":
            return isin
        return isin_lookup.get((row["exchange"], row["tradingsymbol"]), "")

    out["isin_final"] = out.apply(_fill_isin, axis=1)

    # Classify
    def _cls(row):
        isin = (row["isin_final"] or "").upper().strip()
        sym = (row["tradingsymbol"] or "").upper().strip()

        if isin in etf_isins:
            return "ETF"
        # symbol fallback (still official list, but ISIN is the real authority)
        if sym in etf_symbols:
            return "ETF"

        # Equity shares usually INE... ETFs usually INF...
        if isin.startswith("INE"):
            return "EQUITY_STOCK"

        if isin:
            return "UNKNOWN"
        return "UNKNOWN"

    out["class"] = out.apply(_cls, axis=1)
    return out


def save_split_csvs(df: pd.DataFrame, prefix: str, ymd: str):
    path_all = os.path.join(OUT_DIR, f"{prefix}_{ymd}.csv")
    df.to_csv(path_all, index=False)

    eq = df[df["class"] == "EQUITY_STOCK"]
    etf = df[df["class"] == "ETF"]
    unk = df[df["class"] == "UNKNOWN"]

    path_eq = os.path.join(OUT_DIR, f"{prefix}_equity_stocks_{ymd}.csv")
    path_etf = os.path.join(OUT_DIR, f"{prefix}_etfs_{ymd}.csv")
    path_unk = os.path.join(OUT_DIR, f"{prefix}_unknown_{ymd}.csv")

    eq.to_csv(path_eq, index=False)
    etf.to_csv(path_etf, index=False)
    unk.to_csv(path_unk, index=False)

    print(f"✅ Saved: {path_all}  (rows={len(df)})")
    print(f"   ├─ Equity stocks: {path_eq}  (rows={len(eq)})")
    print(f"   ├─ ETFs:          {path_etf} (rows={len(etf)})")
    print(f"   └─ Unknown:       {path_unk} (rows={len(unk)})")


# ----------------- Main export -----------------
def export_portfolio_100pct_etf():
    os.makedirs(OUT_DIR, exist_ok=True)

    api_key = read_api_key()
    access_token = read_access_token()

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    ymd = datetime.now(IST).strftime("%Y%m%d")

    # 1) Official ETF master
    print("Downloading official NSE ETF master...")
    etf_master = download_nse_etf_master()
    etf_isins, etf_symbols = build_etf_sets(etf_master)

    etf_master_path = os.path.join(OUT_DIR, f"nse_etf_master_{ymd}.csv")
    etf_master.to_csv(etf_master_path, index=False)
    print(f"✅ Saved NSE ETF master: {etf_master_path} (ETFs={len(etf_isins)})")

    # 2) Fetch holdings + positions
    holdings = kite.holdings()
    df_holdings = pd.json_normalize(holdings) if holdings else pd.DataFrame()

    pos = kite.positions()
    df_pos_day = pd.json_normalize(pos.get("day", [])) if pos else pd.DataFrame()
    df_pos_net = pd.json_normalize(pos.get("net", [])) if pos else pd.DataFrame()

    # 3) Build ISIN lookup for all symbols we have (positions often need this)
    symbols_needed = {"NSE": set(), "BSE": set()}
    for dfx in [df_holdings, df_pos_day, df_pos_net]:
        sbe = symbols_by_exchange_from_df(dfx)
        symbols_needed["NSE"].update(sbe["NSE"])
        symbols_needed["BSE"].update(sbe["BSE"])

    isin_lookup = build_isin_lookup_for_symbols(kite, symbols_needed, exchanges=("NSE", "BSE"))
    print(f"Instrument ISIN lookup built: {len(isin_lookup)} symbols")

    # 4) Classify + save
    if not df_holdings.empty:
        df_holdings_c = classify_df_equity_vs_etf(df_holdings, etf_isins, etf_symbols, isin_lookup)
        save_split_csvs(df_holdings_c, "holdings", ymd)
    else:
        print("ℹ️ holdings empty.")

    if not df_pos_day.empty:
        df_pos_day_c = classify_df_equity_vs_etf(df_pos_day, etf_isins, etf_symbols, isin_lookup)
        save_split_csvs(df_pos_day_c, "positions_day", ymd)
    else:
        print("ℹ️ positions_day empty.")

    if not df_pos_net.empty:
        df_pos_net_c = classify_df_equity_vs_etf(df_pos_net, etf_isins, etf_symbols, isin_lookup)
        save_split_csvs(df_pos_net_c, "positions_net", ymd)
    else:
        print("ℹ️ positions_net empty.")

    # 5) Margins snapshot (optional)
    try:
        margins = kite.margins()
        df_margins = pd.json_normalize(margins)
        margins_path = os.path.join(OUT_DIR, f"margins_{ymd}.csv")
        df_margins.to_csv(margins_path, index=False)
        print(f"✅ Saved margins: {margins_path} (rows={len(df_margins)})")
    except Exception as e:
        print(f"⚠️ Could not fetch margins: {e}")

    print("\nDONE.")


if __name__ == "__main__":
    export_portfolio_100pct_etf()
