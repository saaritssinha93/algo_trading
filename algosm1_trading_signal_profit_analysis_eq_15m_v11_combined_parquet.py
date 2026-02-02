# -*- coding: utf-8 -*-
"""
ALGO-SM1 | 15m AVWAP INTRADAY (v11) — COMBINED LONG + SHORT runner (FIXED LOADER)
===============================================================================

Fix for Spyder/Windows dataclass import error:
--------------------------------------------
If you load a script via importlib without inserting it into sys.modules first,
dataclasses may fail with:
    AttributeError: 'NoneType' object has no attribute '__dict__'

This runner inserts the module into sys.modules BEFORE exec_module().

REQUIREMENTS (place this file in the SAME folder as these two files):
- algosm1_trading_signal_profit_analysis_eq_15m_v11_intraday_parquet.py            (SHORT)
- algosm1_trading_signal_profit_analysis_eq_15m_v11_long_intraday_parquet.py      (LONG)

Output:
- reports/avwap_longshort_trades_ALL_DAYS_<timestamp>.csv   (ONLY ONE CSV)

"""

from __future__ import annotations

import os
import sys
import heapq
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import pytz


# =============================================================================
# USER SETTINGS (safe: does NOT affect trade generation)
# =============================================================================
IST = pytz.timezone("Asia/Kolkata")

# Notional amount per trade for ₹ P&L calculations (separate for long vs short)
POSITION_SIZE_RS_SHORT = 50_000
POSITION_SIZE_RS_LONG  = 50_000

# Optional cash-constrained portfolio simulation (chronological)
ENABLE_CASH_CONSTRAINED_PORTFOLIO_SIM = False
PORTFOLIO_START_CAPITAL_RS = 1_000_000

# If True, portfolio sim prevents taking BOTH long and short for same ticker on same day
DISALLOW_BOTH_SIDES_SAME_TICKER_DAY = False


# =============================================================================
# LOAD YOUR TWO SCRIPTS AS MODULES (NO LOGIC CHANGED)
# =============================================================================
def _load_module_from_file(module_name: str, file_path: Path):
    """
    Robust loader for Spyder/Windows:
    - Creates module from spec
    - Inserts into sys.modules BEFORE executing
    - Executes module
    """
    import importlib.util

    file_path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {file_path}")

    mod = importlib.util.module_from_spec(spec)

    # ✅ critical for dataclasses/type-checking that consult sys.modules
    sys.modules[module_name] = mod

    # Keep file metadata consistent
    mod.__file__ = str(file_path)

    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _here() -> Path:
    return Path(__file__).resolve().parent


SHORT_FILE = _here() / "algosm1_trading_signal_profit_analysis_eq_15m_v11_intraday_parquet.py"
LONG_FILE  = _here() / "algosm1_trading_signal_profit_analysis_eq_15m_v11_long_intraday_parquet.py"

if not SHORT_FILE.exists():
    raise FileNotFoundError(f"Missing SHORT file: {SHORT_FILE}")
if not LONG_FILE.exists():
    raise FileNotFoundError(f"Missing LONG file:  {LONG_FILE}")

S = _load_module_from_file("avwap_short_v11_module", SHORT_FILE)
L = _load_module_from_file("avwap_long_v11_module",  LONG_FILE)


# =============================================================================
# HELPERS
# =============================================================================
def now_ist() -> datetime:
    return datetime.now(IST)


def _to_df(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        try:
            rows.append(asdict(t))
        except Exception:
            rows.append(dict(t.__dict__))
    return pd.DataFrame(rows)


def _ensure_dt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def _apply_topn_like_original(df: pd.DataFrame, enable: bool, topn: int) -> pd.DataFrame:
    """
    Apply Top-N per day PER SIDE exactly like typical v11 implementation:
      sort by trade_date asc, quality_score desc, ticker asc, entry_time asc
      groupby(trade_date).head(topn)
    """
    if df.empty or (not enable) or (topn is None) or (int(topn) <= 0):
        return df

    req = {"trade_date", "quality_score", "ticker", "entry_time_ist"}
    if not req.issubset(df.columns):
        return df  # don't alter if missing columns

    d = df.copy()
    d["quality_score"] = pd.to_numeric(d["quality_score"], errors="coerce").fillna(0.0)
    d = _ensure_dt(d, ["entry_time_ist", "exit_time_ist", "signal_time_ist"])
    d = d.sort_values(
        ["trade_date", "quality_score", "ticker", "entry_time_ist"],
        ascending=[True, False, True, True],
    )
    d = d.groupby("trade_date", sort=False, as_index=False).head(int(topn)).reset_index(drop=True)
    return d


def _summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "trades": 0, "days": 0,
            "target": 0, "sl": 0, "be": 0, "eod": 0,
            "avg_pnl_pct": 0.0, "sum_pnl_pct": 0.0,
            "hit_rate": 0.0, "sl_rate": 0.0, "be_rate": 0.0, "eod_rate": 0.0,
        }

    d = df.copy()
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce")
    trades = int(len(d))
    days = int(d["trade_date"].nunique()) if "trade_date" in d.columns else 0

    target = int((d["outcome"] == "TARGET").sum()) if "outcome" in d.columns else 0
    sl     = int((d["outcome"] == "SL").sum()) if "outcome" in d.columns else 0
    be     = int((d["outcome"] == "BE").sum()) if "outcome" in d.columns else 0
    eod    = int((d["outcome"] == "EOD").sum()) if "outcome" in d.columns else 0

    return {
        "trades": trades,
        "days": days,
        "target": target, "sl": sl, "be": be, "eod": eod,
        "avg_pnl_pct": float(d["pnl_pct"].mean(skipna=True)),
        "sum_pnl_pct": float(d["pnl_pct"].sum(skipna=True)),
        "hit_rate": (target / trades * 100.0) if trades else 0.0,
        "sl_rate": (sl / trades * 100.0) if trades else 0.0,
        "be_rate": (be / trades * 100.0) if trades else 0.0,
        "eod_rate": (eod / trades * 100.0) if trades else 0.0,
    }


def _print_summary(title: str, s: Dict[str, Any]) -> None:
    print(f"\n================ {title} =================")
    print(f"Total trades                 : {s['trades']}")
    print(f"Unique trade days            : {s['days']}")
    print(f"Trades that hit TARGET       : {s['target']}  | hit-rate={s['hit_rate']:.2f}%")
    print(f"Trades that hit SL           : {s['sl']}      | sl-rate ={s['sl_rate']:.2f}%")
    print(f"Breakeven exits (BE)         : {s['be']}      | be-rate ={s['be_rate']:.2f}%")
    print(f"EOD exits (no SL/Target)     : {s['eod']}     | eod-rate={s['eod_rate']:.2f}%")
    print(f"Average % PnL (per trade)    : {s['avg_pnl_pct']:.4f}%")
    print(f"Sum of all % PnL (all trades): {s['sum_pnl_pct']:.4f}%")
    print("=========================================\n")


def _add_notional_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["position_size_rs"] = d["side"].map(lambda x: POSITION_SIZE_RS_SHORT if str(x).upper() == "SHORT" else POSITION_SIZE_RS_LONG)
    d["pnl_rs"] = (d["pnl_pct"] / 100.0) * pd.to_numeric(d["position_size_rs"], errors="coerce").fillna(0.0)
    return d


# =============================================================================
# OPTIONAL CASH-CONSTRAINED PORTFOLIO SIM
# =============================================================================
def _simulate_cash_constrained(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return df, {
            "start_capital": PORTFOLIO_START_CAPITAL_RS,
            "taken": 0, "skipped": 0,
            "net_pnl_rs": 0.0,
            "final_equity": float(PORTFOLIO_START_CAPITAL_RS),
            "roi_pct": 0.0,
            "max_concurrent": 0,
            "min_cash": float(PORTFOLIO_START_CAPITAL_RS),
        }

    d = _ensure_dt(df, ["entry_time_ist", "exit_time_ist"])
    d = d.sort_values(["entry_time_ist", "exit_time_ist", "ticker", "side"]).reset_index(drop=True)

    cash = float(PORTFOLIO_START_CAPITAL_RS)
    open_heap = []  # (exit_time, size, pnl_rs)
    seen_ticker_day = set()

    taken_flags, cash_before, cash_after, pnl_rs_sim, pos_sizes = [], [], [], [], []
    taken = 0
    skipped = 0
    max_conc = 0
    min_cash = cash

    for _, row in d.iterrows():
        entry_ts = row["entry_time_ist"]
        exit_ts = row["exit_time_ist"]

        while open_heap and open_heap[0][0] <= entry_ts:
            ex_time, size, pnl_rs = heapq.heappop(open_heap)
            cash += size + pnl_rs

        cb = cash
        side = str(row["side"]).upper()
        ticker = str(row["ticker"])
        day = str(row["trade_date"])

        pos = float(POSITION_SIZE_RS_SHORT if side == "SHORT" else POSITION_SIZE_RS_LONG)
        pnl = float(row.get("pnl_rs", 0.0))

        take = True
        if DISALLOW_BOTH_SIDES_SAME_TICKER_DAY:
            key = (ticker, day)
            if key in seen_ticker_day:
                take = False

        if cash < pos:
            take = False

        if take:
            cash -= pos
            heapq.heappush(open_heap, (exit_ts, pos, pnl))
            taken += 1
            seen_ticker_day.add((ticker, day))
        else:
            skipped += 1
            pos = 0.0
            pnl = 0.0

        taken_flags.append(bool(take))
        cash_before.append(float(cb))
        cash_after.append(float(cash))
        pos_sizes.append(float(pos))
        pnl_rs_sim.append(float(pnl))

        max_conc = max(max_conc, len(open_heap))
        min_cash = min(min_cash, cash)

    while open_heap:
        ex_time, size, pnl_rs = heapq.heappop(open_heap)
        cash += size + pnl_rs

    final_equity = cash
    net_pnl = final_equity - float(PORTFOLIO_START_CAPITAL_RS)
    roi = (net_pnl / float(PORTFOLIO_START_CAPITAL_RS) * 100.0) if PORTFOLIO_START_CAPITAL_RS > 0 else 0.0

    d = d.copy()
    d["taken"] = taken_flags
    d["cash_before"] = cash_before
    d["cash_after"] = cash_after
    d["position_size_rs_sim"] = pos_sizes
    d["pnl_rs_sim"] = pnl_rs_sim

    stats = {
        "start_capital": float(PORTFOLIO_START_CAPITAL_RS),
        "taken": int(taken),
        "skipped": int(skipped),
        "net_pnl_rs": float(net_pnl),
        "final_equity": float(final_equity),
        "roi_pct": float(roi),
        "max_concurrent": int(max_conc),
        "min_cash": float(min_cash),
    }
    return d, stats


def _print_portfolio(stats: Dict[str, Any]) -> None:
    print("\n================ PORTFOLIO SUMMARY (cash-constrained) ================")
    print(f"Start capital (₹)            : ₹{stats['start_capital']:,.2f}")
    print(f"Taken trades                 : {stats['taken']}")
    print(f"Skipped trades               : {stats['skipped']}")
    print(f"Net P&L (₹)                  : ₹{stats['net_pnl_rs']:,.2f}")
    print(f"Final equity (₹)             : ₹{stats['final_equity']:,.2f}")
    print(f"ROI on start capital          : {stats['roi_pct']:.2f}%")
    print(f"Max concurrent positions      : {stats['max_concurrent']}")
    print(f"Minimum cash during run (₹)   : ₹{stats['min_cash']:,.2f}")
    print("======================================================================\n")


# =============================================================================
# RUN ONE SIDE (calls your module's exact scan function)
# =============================================================================
def _run_one_side(mod, side_name: str) -> pd.DataFrame:
    print(f"[RUN] {side_name} scanner (using your v11 {side_name} file exactly)")
    tickers = mod.list_tickers_15m()
    print(f"[STEP] Tickers found: {len(tickers)}")

    trades_all = []
    for k, t in enumerate(tickers, start=1):
        path = os.path.join(mod.DIR_15M, f"{t}{mod.END_15M}")
        df = mod.read_15m_parquet(path)
        if df is None or len(df) == 0:
            continue

        trades = mod.scan_all_days_for_ticker(t, df)
        if trades:
            trades_all.extend(trades)

        if k % 50 == 0:
            print(f"  scanned {k}/{len(tickers)} | {side_name.lower()}_trades_so_far={len(trades_all)}")

    out_raw = _to_df(trades_all)
    if out_raw.empty:
        return out_raw

    # Apply Top-N per day PER SIDE (as in individual scripts)
    enable_topn = bool(getattr(mod, "ENABLE_TOPN_PER_DAY", False))
    topn = int(getattr(mod, "TOPN_PER_DAY", 0) or 0)
    out = _apply_topn_like_original(out_raw, enable_topn, topn)

    out["side"] = side_name.upper()
    out = _ensure_dt(out, ["signal_time_ist", "entry_time_ist", "exit_time_ist"])

    sort_cols = [c for c in ["trade_date", "ticker", "entry_time_ist"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def main() -> None:
    print("[RUN] AVWAP v11 COMBINED runner — LONG + SHORT (one CSV) [FIXED]")
    print(f"[INFO] Notional ₹ P&L: SHORT=₹{POSITION_SIZE_RS_SHORT:,.0f} per trade | LONG=₹{POSITION_SIZE_RS_LONG:,.0f} per trade")
    print("[INFO] Top-N per day is applied PER SIDE exactly as in the individual scripts.")
    print("--------------------------------------------------------------")

    short_df = _run_one_side(S, "SHORT")
    long_df  = _run_one_side(L, "LONG")

    if short_df.empty and long_df.empty:
        print("[DONE] No trades found.")
        return

    combined = pd.concat([short_df, long_df], ignore_index=True)
    combined = _add_notional_pnl(combined)

    _print_summary("SHORT SUMMARY (post-filter)", _summary(short_df))
    _print_summary("LONG  SUMMARY (post-filter)", _summary(long_df))
    _print_summary("OVERALL SUMMARY (post-filter)", _summary(combined))

    pnl_short = float(combined.loc[combined["side"].eq("SHORT"), "pnl_rs"].sum()) if "pnl_rs" in combined.columns else 0.0
    pnl_long  = float(combined.loc[combined["side"].eq("LONG"),  "pnl_rs"].sum()) if "pnl_rs" in combined.columns else 0.0
    pnl_all   = float(combined["pnl_rs"].sum()) if "pnl_rs" in combined.columns else 0.0

    print("\n================ NOTIONAL P&L SUMMARY (₹) ================")
    print(f"SHORT notional P&L (₹)        : ₹{pnl_short:,.2f}")
    print(f"LONG  notional P&L (₹)        : ₹{pnl_long:,.2f}")
    print(f"TOTAL notional P&L (₹)        : ₹{pnl_all:,.2f}")
    print("==========================================================\n")

    if ENABLE_CASH_CONSTRAINED_PORTFOLIO_SIM:
        sim_df, pstats = _simulate_cash_constrained(combined)
        _print_portfolio(pstats)
        combined = sim_df  # include sim columns in output CSV

    reports_dir = Path(getattr(S, "REPORTS_DIR", _here() / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    out_csv = reports_dir / f"avwap_longshort_trades_ALL_DAYS_{ts}.csv"
    combined.to_csv(out_csv, index=False)

    cols = [c for c in ["trade_date","ticker","side","setup","impulse_type","quality_score","entry_price","exit_price","outcome","pnl_pct","position_size_rs","pnl_rs"] if c in combined.columns]
    print("=============== SAMPLE (first 30 rows) ===============")
    print(combined.head(30)[cols].to_string(index=False))
    print(f"\n[FILE SAVED] {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
