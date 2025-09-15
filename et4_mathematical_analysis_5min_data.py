# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:45:15 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Intraday 2% target backtest + winners-only fitted indicator ranges.
Adds:
  • Per-ticker winners-only fit with shrinkage to global winners
  • Incremental updater for fitted_indicator_ranges_by_ticker.json

Assumptions:
- Directory: main_indicators_july_5min1
- Files: {TICKER}_main_indicators.csv
- Columns present at least:
  date, open, high, low, close, volume,
  RSI, ATR, EMA_50, EMA_200, 20_SMA, VWAP,
  CCI, MFI, OBV, MACD, MACD_Signal, MACD_Hist,
  Upper_Band, Lower_Band, Recent_High, Recent_Low,
  Stoch_%K, Stoch_%D, ADX, date_only, Daily_Change, Intra_Change

Outputs:
- entries_2pct_results.csv (all entries & outcomes)
- fitted_indicator_ranges_by_ticker.json/.csv (per-ticker winners-only ranges)
"""

import os
import json
import time
from datetime import time as dt_time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from collections import defaultdict

# ============================
# CONFIG
# ============================
INDICATORS_DIR = "main_indicators_july_5min1"

# Date window (training universe)
START_DATE = "2024-08-05"
END_DATE   = "2025-07-10"

# --- LIGHT LOGGING CONFIG ---
LOG_PRINT_EVERY_FILES = 25
LOG_TICKERS_WITH_ENTRIES_MAX = 40

# Entry rules (kept simple, tunable)
ENTRY_PARAMS_DEFAULT = {
    "target_pct": 0.020,          # 2% target
    "stop_pct": None,             # optional fixed SL (e.g., 0.010 for 1%). None => no SL, exit at close if target not hit
    "min_bars_after_entry": 1,    # require >= 1 forward bar (safety)
    "min_bars_left_in_session": 4,# NEW: require at least 4 bars (~20m) left in session

    # Directional bias
    "daily_change_min_bull":  0.0,
    "daily_change_max_bear":  0.0,

    # Trend filters
    "ema_trend_bull": True,       # EMA_50 > EMA_200
    "ema_trend_bear": True,       # EMA_50 < EMA_200

    # Strength/momentum windows
    "adx_min": 20.0, "adx_max": 55.0,
    "rsi_bull_low": 45.0, "rsi_bull_high": 75.0,
    "rsi_bear_low": 25.0, "rsi_bear_high": 55.0,
    "stochd_bull_min": 45.0,
    "stochd_bear_max": 55.0,

    # VWAP relation (distance in fraction; 0.001 = 0.1%)
    "vwap_bull_min": 0.0005,       # NEW default: require ≥ +0.10% above VWAP
    "vwap_bear_min": 0.0005,       # NEW default: require ≤ −0.10% below VWAP

    # Volume participation (VolRel = vol / rolling_vol_10)
    "roll_vol_window": 10,
    "vol_rel_min": 1.10,          # NEW: global participation brake (raise to 1.22 to cut more)

    # Volatility floor
    "atr_pct_min": 0.0025,         # NEW: >= 0.30% ATR/Close to avoid dead bars

    # Intra streak
    "streak_n": 1,
    "streak_thr_bull": 0.00,
    "streak_thr_bear": 0.00,

    # Trading day window (5-min bars)
    "session_start": dt_time(9, 25),
    "session_end":   dt_time(15, 10),
    "no_new_entries_after": dt_time(14, 45),  # NEW: stop taking fresh entries after 14:45
}

# Columns we actually need to keep memory/speed sane
NEEDED_COLS = [
    "date","open","high","low","close","volume","date_only",
    "RSI","ATR","EMA_50","EMA_200","20_SMA","VWAP","CCI","MFI",
    "OBV","MACD","MACD_Signal","MACD_Hist","Upper_Band","Lower_Band",
    "Recent_High","Recent_Low","Stoch_%K","Stoch_%D","ADX",
    "Daily_Change","Intra_Change"
]

# Indicators used in step-1 snapshots & step-2 fits (shared)
INDICATOR_SPECS = {
    "range_both": [
        "RSI", "ADX", "CCI", "MFI",
        "BBP", "BandWidth", "ATR_Pct",
    ],
    "min_bull_max_bear": [
        "StochD", "StochK", "MACD_Hist",
        "VWAP_Dist", "Daily_Change",
        "EMA50_Dist", "EMA200_Dist", "SMA20_Dist",
        "OBV_Slope10",
    ],
    "min_both": ["VolRel"],
}

# ============================
# UTILITIES
# ============================

def _safe_read_csv(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if usecols:
        for c in usecols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[usecols]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    mask = (df["date"].dt.date >= pd.to_datetime(START_DATE).date()) & \
           (df["date"].dt.date <= pd.to_datetime(END_DATE).date())
    df = df.loc[mask].copy()
    if "date_only" not in df.columns:
        df["date_only"] = df["date"].dt.date
    return df.sort_values("date").reset_index(drop=True)

def _add_rolling_vol(df: pd.DataFrame, w: int) -> pd.Series:
    return df["volume"].rolling(w, min_periods=w).mean()

def _in_time_window(ts: pd.Timestamp, start_t: dt_time, end_t: dt_time) -> bool:
    t = ts.time()
    return (t >= start_t) and (t <= end_t)

def _intra_streak_ok(series_intra: pd.Series, idx: int, n: int, thr: float, bullish: bool) -> bool:
    if n <= 0 or idx - (n - 1) < 0:
        return False
    seg = series_intra.iloc[idx - (n - 1):idx + 1]
    if seg.isna().any():
        return False
    return (seg >= thr).all() if bullish else (seg <= -abs(thr)).all()

# ============================
# ENTRY LOGIC (Step 1)
# ============================

def _row_is_bull_entry(df: pd.DataFrame, i: int, p: Dict[str, Any]) -> bool:
    r = df.iloc[i]

    # time gates
    if not _in_time_window(r["date"], p["session_start"], p["session_end"]):
        return False
    if r["date"].time() > p["no_new_entries_after"]:
        return False

    # required cols present
    req = ["close","VWAP","RSI","ADX","Stoch_%D","Daily_Change","EMA_50","EMA_200",
           "volume","rolling_vol10","Intra_Change","ATR_Pct"]
    if any(pd.isna(r.get(c)) for c in req):
        return False

    # enough bars left in session
    # (assumes df is a single-day slice in generate_intraday_entries_for_file)
    last_idx = df.index[df["date"].apply(lambda ts: _in_time_window(ts, p["session_start"], p["session_end"]))].max()
    if (last_idx - i) < p["min_bars_left_in_session"]:
        return False

    # global gates
    if r["Daily_Change"] < p["daily_change_min_bull"]:
        return False
    if p["ema_trend_bull"] and not (r["EMA_50"] >= r["EMA_200"]):
        return False
    if not (p["adx_min"] <= r["ADX"] <= p["adx_max"]):
        return False
    if not (p["rsi_bull_low"] <= r["RSI"] <= p["rsi_bull_high"]):
        return False
    if r["Stoch_%D"] < p["stochd_bull_min"]:
        return False
    if r["close"] < r["VWAP"] * (1 + p["vwap_bull_min"]):
        return False
    if r["volume"] < p["vol_rel_min"] * r["rolling_vol10"]:
        return False
    if r["ATR_Pct"] < p["atr_pct_min"]:
        return False
    if not _intra_streak_ok(df["Intra_Change"], i, p["streak_n"], p["streak_thr_bull"], bullish=True):
        return False
    return True


def _row_is_bear_entry(df: pd.DataFrame, i: int, p: Dict[str, Any]) -> bool:
    r = df.iloc[i]

    # time gates
    if not _in_time_window(r["date"], p["session_start"], p["session_end"]):
        return False
    if r["date"].time() > p["no_new_entries_after"]:
        return False

    # required cols present
    req = ["close","VWAP","RSI","ADX","Stoch_%D","Daily_Change","EMA_50","EMA_200",
           "volume","rolling_vol10","Intra_Change","ATR_Pct"]
    if any(pd.isna(r.get(c)) for c in req):
        return False

    # enough bars left in session
    last_idx = df.index[df["date"].apply(lambda ts: _in_time_window(ts, p["session_start"], p["session_end"]))].max()
    if (last_idx - i) < p["min_bars_left_in_session"]:
        return False

    # global gates
    if r["Daily_Change"] > p["daily_change_max_bear"]:
        return False
    if p["ema_trend_bear"] and not (r["EMA_50"] <= r["EMA_200"]):
        return False
    if not (p["adx_min"] <= r["ADX"] <= p["adx_max"]):
        return False
    if not (p["rsi_bear_low"] <= r["RSI"] <= p["rsi_bear_high"]):
        return False
    if r["Stoch_%D"] > p["stochd_bear_max"]:
        return False
    # force below VWAP by a margin when vwap_bear_min > 0
    if p["vwap_bear_min"] > 0 and r["close"] > r["VWAP"] * (1 - p["vwap_bear_min"]):
        return False
    if r["volume"] < p["vol_rel_min"] * r["rolling_vol10"]:
        return False
    if r["ATR_Pct"] < p["atr_pct_min"]:
        return False
    if not _intra_streak_ok(df["Intra_Change"], i, p["streak_n"], p["streak_thr_bear"], bullish=False):
        return False
    return True


def _simulate_exit_same_day(df_day: pd.DataFrame, entry_idx: int, side: str, target_pct: float,
                            stop_pct: Optional[float]) -> Tuple[bool, pd.Timestamp, float, float]:
    entry_row = df_day.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    if side == "bull":
        target = entry_price * (1 + target_pct)
        stop   = entry_price * (1 - stop_pct) if stop_pct else None
    else:
        target = entry_price * (1 - target_pct)
        stop   = entry_price * (1 + stop_pct) if stop_pct else None
    fwd = df_day.iloc[entry_idx+1:].copy()
    if fwd.empty:
        return (False, entry_row["date"], entry_price, 0.0)
    hit_target = False
    exit_px = None
    exit_time = None
    max_fav = 0.0
    for _, r in fwd.iterrows():
        if side == "bull":
            max_fav = max(max_fav, (r["high"] - entry_price) / entry_price)
            if stop is not None and r["low"] <= stop:
                hit_target = False; exit_px = stop; exit_time = r["date"]; break
            if r["high"] >= target:
                hit_target = True; exit_px = target; exit_time = r["date"]; break
        else:
            max_fav = max(max_fav, (entry_price - r["low"]) / entry_price)
            if stop is not None and r["high"] >= stop:
                hit_target = False; exit_px = stop; exit_time = r["date"]; break
            if r["low"] <= target:
                hit_target = True; exit_px = target; exit_time = r["date"]; break
    if exit_px is None:
        last = fwd.iloc[-1]
        exit_px = float(last["close"]); exit_time = last["date"]
    return (hit_target, exit_time, float(exit_px), float(max_fav))

# ============================
# STEP 1: ENTRY GENERATION
# ============================

def generate_intraday_entries_for_file(ticker: str,
                                       path: str,
                                       params: Dict[str, Any]) -> pd.DataFrame:
    df = _safe_read_csv(path, usecols=NEEDED_COLS)
    if df.empty:
        return pd.DataFrame()

    # ---------- Derived features (computed once on the full DF) ----------
    eps = 1e-9
    df["rolling_vol10"] = _add_rolling_vol(df, params["roll_vol_window"])

    # distances to reference curves
    df["VWAP_Dist"]   = df["close"] / df["VWAP"]    - 1.0
    df["EMA50_Dist"]  = df["close"] / df["EMA_50"]  - 1.0
    df["EMA200_Dist"] = df["close"] / df["EMA_200"] - 1.0
    df["SMA20_Dist"]  = df["close"] / df["20_SMA"]  - 1.0

    # normalized ATR
    df["ATR_Pct"] = df["ATR"] / (df["close"] + eps)

    # Bollinger features
    bbw = (df["Upper_Band"] - df["Lower_Band"])
    df["BandWidth"] = bbw / (df["close"] + eps)
    df["BBP"] = (df["close"] - df["Lower_Band"]) / (bbw + eps)

    # Momentum & volume
    df["VolRel"] = df["volume"] / (df["rolling_vol10"] + eps)
    df["StochK"] = df["Stoch_%K"]
    df["StochD"] = df["Stoch_%D"]  # alias

    # Scale-invariant OBV slope (10 bars)
    obv_prev = df["OBV"].shift(10)
    df["OBV_Slope10"] = (df["OBV"] - obv_prev) / (obv_prev.abs() + eps)

    rows = []
    for d, g in df.groupby("date_only", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        eligible = g["date"].apply(lambda ts: _in_time_window(ts, params["session_start"], params["session_end"]))
        idxs = list(np.where(eligible.values)[0])

        for i in idxs:
            if i >= len(g) - params["min_bars_after_entry"]:
                continue

            # ---- Bull entry ----
            if _row_is_bull_entry(g, i, params):
                hit, ex_t, ex_px, mfe = _simulate_exit_same_day(g, i, "bull", params["target_pct"], params["stop_pct"])
                r = g.iloc[i]
                rows.append({
                    "Ticker": ticker, "Side": "Bullish",
                    "EntryTime": r["date"], "EntryPrice": float(r["close"]),
                    "ExitTime": ex_t, "ExitPrice": ex_px,
                    "HitTarget": bool(hit), "ReturnPct": (ex_px / float(r["close"]) - 1.0) * 100.0,
                    "MFE": mfe * 100.0,

                    # --- snapshots for Step-2 fitting ---
                    "RSI": r["RSI"], "ADX": r["ADX"],
                    "StochD": r["StochD"], "StochK": r["StochK"],
                    "VWAP_Dist": r["VWAP_Dist"], "VolRel": r["VolRel"],
                    "Daily_Change": r["Daily_Change"], "Intra_Change": r["Intra_Change"],
                    "EMA50_Dist": r["EMA50_Dist"], "EMA200_Dist": r["EMA200_Dist"], "SMA20_Dist": r["SMA20_Dist"],
                    "ATR_Pct": r["ATR_Pct"],
                    "BBP": r["BBP"], "BandWidth": r["BandWidth"],
                    "MACD_Hist": r["MACD_Hist"],
                    "CCI": r["CCI"], "MFI": r["MFI"],
                    "OBV_Slope10": r["OBV_Slope10"],

                    "EMA_50": r["EMA_50"], "EMA_200": r["EMA_200"], "DateOnly": d
                })

            # ---- Bear entry ----
            if _row_is_bear_entry(g, i, params):
                hit, ex_t, ex_px, mfe = _simulate_exit_same_day(g, i, "bear", params["target_pct"], params["stop_pct"])
                r = g.iloc[i]
                rows.append({
                    "Ticker": ticker, "Side": "Bearish",
                    "EntryTime": r["date"], "EntryPrice": float(r["close"]),
                    "ExitTime": ex_t, "ExitPrice": ex_px,
                    "HitTarget": bool(hit), "ReturnPct": (ex_px / float(r["close"]) - 1.0) * 100.0 if ex_px else np.nan,
                    "MFE": mfe * 100.0,

                    "RSI": r["RSI"], "ADX": r["ADX"],
                    "StochD": r["StochD"], "StochK": r["StochK"],
                    "VWAP_Dist": r["VWAP_Dist"], "VolRel": r["VolRel"],
                    "Daily_Change": r["Daily_Change"], "Intra_Change": r["Intra_Change"],
                    "EMA50_Dist": r["EMA50_Dist"], "EMA200_Dist": r["EMA200_Dist"], "SMA20_Dist": r["SMA20_Dist"],
                    "ATR_Pct": r["ATR_Pct"],
                    "BBP": r["BBP"], "BandWidth": r["BandWidth"],
                    "MACD_Hist": r["MACD_Hist"],
                    "CCI": r["CCI"], "MFI": r["MFI"],
                    "OBV_Slope10": r["OBV_Slope10"],

                    "EMA_50": r["EMA_50"], "EMA_200": r["EMA_200"], "DateOnly": d
                })

    return pd.DataFrame(rows)

def run_step1_scan(indicators_dir: str = INDICATORS_DIR,
                   params: Dict[str, Any] = ENTRY_PARAMS_DEFAULT,
                   out_csv: str = "entries_2pct_results.csv") -> pd.DataFrame:
    all_rows = []
    files = [f for f in os.listdir(indicators_dir) if f.endswith("_main_indicators.csv")]
    files.sort()
    nfiles = len(files)

    print(f"\n[Step1] Scanning {nfiles} files in '{indicators_dir}' for window {START_DATE} → {END_DATE}")
    print(f"[Step1] Target={params['target_pct']*100:.1f}%, Stop={'None' if params['stop_pct'] is None else f'{params['stop_pct']*100:.1f}%'}")

    side_counts = defaultdict(int)
    hit_counts = defaultdict(int)
    total_entries = 0
    tickers_logged = 0

    for k, fname in enumerate(files, start=1):
        ticker = fname.replace("_main_indicators.csv","")
        path = os.path.join(indicators_dir, fname)
        try:
            df_entries = generate_intraday_entries_for_file(ticker, path, params)
            if not df_entries.empty:
                all_rows.append(df_entries)

                c_by_side = df_entries.groupby("Side").size().to_dict()
                h_by_side = df_entries.groupby(["Side","HitTarget"]).size().to_dict()

                for s, c in c_by_side.items():
                    side_counts[s] += int(c)
                    total_entries += int(c)

                for (s, h), c in h_by_side.items():
                    if h:
                        hit_counts[s] += int(c)

                if tickers_logged < LOG_TICKERS_WITH_ENTRIES_MAX:
                    t0 = df_entries["EntryTime"].min()
                    t1 = df_entries["EntryTime"].max()
                    bull = int(c_by_side.get("Bullish", 0))
                    bear = int(c_by_side.get("Bearish", 0))
                    bull_hit = int(h_by_side.get(("Bullish", True), 0))
                    bear_hit = int(h_by_side.get(("Bearish", True), 0))
                    print(f"  [+] {ticker}: entries={bull+bear} (B:{bull}/{bull_hit} hit, R:{bear}/{bear_hit} hit) "
                          f"time=[{str(t0)[11:16]}→{str(t1)[11:16]}]")
                    tickers_logged += 1

        except Exception as e:
            print(f"  [!] {ticker}: {e}")

        if (k % LOG_PRINT_EVERY_FILES) == 0 or (k == nfiles):
            bull = side_counts.get("Bullish", 0)
            bear = side_counts.get("Bearish", 0)
            bull_hit = hit_counts.get("Bullish", 0)
            bear_hit = hit_counts.get("Bearish", 0)
            hit_total = bull_hit + bear_hit
            wr = (hit_total / total_entries * 100.0) if total_entries else 0.0
            print(f"[{k}/{nfiles}] files | entries so far={total_entries} (Bull:{bull}, Bear:{bear}) | hit-rate={wr:.1f}%")

    if not all_rows:
        print("[Step1] No entries found.")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True).sort_values("EntryTime")
    result.to_csv(out_csv, index=False)

    final_side = result.groupby("Side").size().to_dict()
    final_hit  = result.groupby(["Side","HitTarget"]).size().to_dict()
    bull = int(final_side.get("Bullish", 0))
    bear = int(final_side.get("Bearish", 0))
    bull_hit = int(final_hit.get(("Bullish", True), 0))
    bear_hit = int(final_hit.get(("Bearish", True), 0))
    wr = ((bull_hit + bear_hit) / len(result) * 100.0) if len(result) else 0.0

    print(f"[Step1] Saved entries → {out_csv} (rows={len(result)})")
    print(f"[Step1] Totals: Bull={bull} (hits {bull_hit}), Bear={bear} (hits {bear_hit}), overall hit-rate={wr:.1f}%")
    return result

# ============================
# STEP 2b: FIT PER-TICKER RANGES (full recompute, winners-only)
# ============================

def _qband(series: pd.Series, lo: float, hi: float) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    lo_v, hi_v = np.quantile(s, [lo, hi])
    return float(lo_v), float(hi_v)

def fit_indicator_ranges_by_ticker(entries: pd.DataFrame,
                                   out_json: str = "fitted_indicator_ranges_by_ticker.json",
                                   out_csv: str = "fitted_indicator_ranges_by_ticker.csv",
                                   q_low: float = 0.20,     # was 0.10
                                   q_high: float = 0.80,    # was 0.90
                                   shrink_k: int = 200) -> pd.DataFrame:
    """
    Winners-only, per-ticker fitting with shrinkage to global winners.
    shrink_k controls pull toward global: weight_ticker = n / (n + shrink_k)
    """

    if entries.empty:
        print("[Fit/ByTicker] No entries provided.")
        return pd.DataFrame()

    def _build_side_cfg(sub: pd.DataFrame,
                        side: str,
                        global_side_cfg: dict,
                        ql: float, qh: float,
                        shrink_k_val: int) -> Tuple[dict, int]:
        n = len(sub)
        w = n / (n + shrink_k_val) if shrink_k_val > 0 else 1.0

        def _blend(val_t, val_g):
            if (val_t is None) or (isinstance(val_t, float) and np.isnan(val_t)):
                return val_g
            if (val_g is None) or (isinstance(val_g, float) and np.isnan(val_g)):
                return val_t
            return float(w * val_t + (1 - w) * val_g)

        side_cfg = {"Samples_wins": int(n), "Quantiles_used": [ql, qh]}

        # Range indicators
        for name in INDICATOR_SPECS["range_both"]:
            if name in sub.columns:
                lo_t, hi_t = _qband(sub[name], ql, qh)
                lo_g = hi_g = None
                g = global_side_cfg.get(f"{name}_range")
                if isinstance(g, (list, tuple)) and len(g) == 2:
                    lo_g, hi_g = float(g[0]), float(g[1])
                side_cfg[f"{name}_range"] = [_blend(lo_t, lo_g), _blend(hi_t, hi_g)]

        # One-sided (bull=min, bear=max)
        for name in INDICATOR_SPECS["min_bull_max_bear"]:
            if name not in sub.columns:
                continue
            lo_t, hi_t = _qband(sub[name], ql, qh)
            if side == "Bullish":
                base_g = global_side_cfg.get(f"{name}_min")
                side_cfg[f"{name}_min"] = _blend(lo_t, base_g)
            else:
                base_g = global_side_cfg.get(f"{name}_max")
                side_cfg[f"{name}_max"] = _blend(hi_t, base_g)

        # Participation floor (both)
        for name in INDICATOR_SPECS["min_both"]:
            if name not in sub.columns:
                continue
            lo_t, _ = _qband(sub[name], ql, qh)
            base_g = global_side_cfg.get(f"{name}_min")
            side_cfg[f"{name}_min"] = _blend(lo_t, base_g)

        return side_cfg, n

    # Global winners (baseline)
    cfg_global = {}
    for side in ["Bullish", "Bearish"]:
        gsub = entries[(entries["Side"] == side) & (entries["HitTarget"] == True)].copy()
        side_cfg = {}
        if len(gsub):
            # build global with no shrink
            side_cfg, _ = _build_side_cfg(gsub, side, {}, q_low, q_high, shrink_k_val=0)
        cfg_global[side] = side_cfg

    # Per-ticker
    cfg_all = {}
    out_rows = []
    for ticker, df_t in entries.groupby("Ticker"):
        cfg_all[ticker] = {}
        for side in ["Bullish", "Bearish"]:
            sub = df_t[(df_t["Side"] == side) & (df_t["HitTarget"] == True)].copy()
            side_cfg, n = _build_side_cfg(sub, side, cfg_global.get(side, {}), q_low, q_high, shrink_k)
            cfg_all[ticker][side] = side_cfg

            row = {"Ticker": ticker, "Side": side, "Samples_wins": n, "Quantiles_used": f"[{q_low:.2f},{q_high:.2f}]"}
            for k, v in side_cfg.items():
                if k in ("Samples_wins", "Quantiles_used"): continue
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    row[k] = f"[{v[0]:.6g},{v[1]:.6g}]"
                else:
                    row[k] = v
            out_rows.append(row)

        if len(out_rows) % 200 == 0:
            print(f"[Fit/ByTicker] {len(out_rows)//2} tickers processed...")

    with open(out_json, "w") as f:
        json.dump(cfg_all, f, indent=2)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(out_csv, index=False)
    print(f"[Fit/ByTicker] Saved per-ticker ranges → {out_json}, {out_csv} (tickers={len(cfg_all)})")
    return df_out

# ============================
# INCREMENTAL UPDATER (winners-only)
# ============================

def _compute_new_cfg(entries_new: pd.DataFrame,
                     q_low: float = 0.10,
                     q_high: float = 0.90) -> Dict[str, Dict]:
    """
    Winners-only, per-ticker ranges on NEW data only.
    Returns: {ticker: {Side: { ... cfg ... , Samples_wins:int, Quantiles_used:[ql,qh]}}}
    """
    cfg = {}
    e = entries_new[entries_new["HitTarget"] == True].copy()
    if e.empty:
        return cfg

    for ticker, g_t in e.groupby("Ticker"):
        cfg[ticker] = {}
        for side in ["Bullish", "Bearish"]:
            sub = g_t[g_t["Side"] == side]
            n = len(sub)
            side_cfg = {"Samples_wins": int(n), "Quantiles_used": [q_low, q_high]}
            if n == 0:
                cfg[ticker][side] = side_cfg
                continue

            # ranges
            for name in INDICATOR_SPECS["range_both"]:
                if name in sub.columns:
                    lo, hi = _qband(sub[name], q_low, q_high)
                    side_cfg[f"{name}_range"] = [lo, hi]

            # one-sided (bull=min, bear=max)
            for name in INDICATOR_SPECS["min_bull_max_bear"]:
                if name in sub.columns:
                    lo, hi = _qband(sub[name], q_low, q_high)
                    if side == "Bullish":
                        side_cfg[f"{name}_min"] = lo
                    else:
                        side_cfg[f"{name}_max"] = hi

            # min both
            for name in INDICATOR_SPECS["min_both"]:
                if name in sub.columns:
                    lo, _ = _qband(sub[name], q_low, q_high)
                    side_cfg[f"{name}_min"] = lo

            cfg[ticker][side] = side_cfg
    return cfg

def _blend_scalar(old_val, new_val, w_new: float):
    o = None if old_val is None else float(old_val)
    n = None if new_val is None else float(new_val)
    if (n is None) or (isinstance(n, float) and np.isnan(n)):
        return o
    if (o is None) or (isinstance(o, float) and np.isnan(o)):
        return n
    return float((1 - w_new) * o + w_new * n)

def _merge_side(old_side: Dict, new_side: Dict) -> Dict:
    """Merge one ticker-side dict using count-weighted blending."""
    old_n = int(old_side.get("Samples_wins", 0)) if old_side else 0
    new_n = int(new_side.get("Samples_wins", 0)) if new_side else 0
    tot_n = old_n + new_n
    out = {}
    out["Samples_wins"] = tot_n
    out["Quantiles_used"] = new_side.get("Quantiles_used", old_side.get("Quantiles_used", [0.10, 0.90]))

    if tot_n == 0:
        return out

    w_new = new_n / tot_n

    # merge ranges
    for name in INDICATOR_SPECS["range_both"]:
        k = f"{name}_range"
        lo_old, hi_old = (old_side.get(k, [np.nan, np.nan]) if old_side else [np.nan, np.nan])
        lo_new, hi_new = (new_side.get(k, [np.nan, np.nan]) if new_side else [np.nan, np.nan])
        lo = _blend_scalar(lo_old, lo_new, w_new)
        hi = _blend_scalar(hi_old, hi_new, w_new)
        out[k] = [lo, hi]

    # merge one-sided + min_both
    keys = set()
    if old_side: keys |= set(old_side.keys())
    if new_side: keys |= set(new_side.keys())
    for k in keys:
        if k.endswith("_min") or k.endswith("_max"):
            v_old = old_side.get(k) if old_side else None
            v_new = new_side.get(k) if new_side else None
            out[k] = _blend_scalar(v_old, v_new, w_new)

    return out

def update_fitted_ranges_by_ticker(existing_json_path: str,
                                   entries_new: pd.DataFrame,
                                   out_json_path: Optional[str] = None,
                                   q_low: float = 0.10,
                                   q_high: float = 0.90,
                                   make_backup: bool = True) -> Dict:
    """
    Incrementally update fitted_indicator_ranges_by_ticker.json with new winners.
    - Reads existing JSON (if any)
    - Computes new per-ticker winners-only ranges on 'entries_new'
    - Merges using count-weighted blending
    - Writes atomically; optional timestamped backup
    """
    if os.path.exists(existing_json_path):
        with open(existing_json_path, "r") as f:
            old_cfg = json.load(f)
    else:
        old_cfg = {}

    new_cfg = _compute_new_cfg(entries_new, q_low=q_low, q_high=q_high)

    merged = {}
    tickers = set(old_cfg.keys()) | set(new_cfg.keys())
    for ticker in sorted(tickers):
        merged[ticker] = {}
        for side in ["Bullish", "Bearish"]:
            old_side = (old_cfg.get(ticker, {}) or {}).get(side, {})
            new_side = (new_cfg.get(ticker, {}) or {}).get(side, {})
            merged[ticker][side] = _merge_side(old_side, new_side)

    # meta
    merged_with_meta = {
        **merged,
        "_meta": {
            "last_updated_epoch": int(time.time()),
            "q_low": q_low, "q_high": q_high,
            "note": "Weighted incremental merge of winners-only per-ticker ranges."
        }
    }

    target = out_json_path or existing_json_path
    # backup
    if make_backup and os.path.exists(existing_json_path):
        ts = time.strftime("%Y%m%d-%H%M%S")
        bk = existing_json_path.replace(".json", f".bak.{ts}.json")
        try:
            with open(bk, "w") as f:
                json.dump(old_cfg, f, indent=2)
            print(f"[Update] Backup saved → {bk}")
        except Exception as e:
            print(f"[Update] Backup failed: {e}")

    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged_with_meta, f, indent=2)
    os.replace(tmp, target)
    print(f"[Update] Merged ranges written → {target}")
    return merged_with_meta

def update_fitted_ranges_by_ticker_from_entries_csv(entries_csv_path: str,
                                                    json_path: str,
                                                    out_json_path: Optional[str] = None,
                                                    time_col: str = "EntryTime",
                                                    q_low: float = 0.10,
                                                    q_high: float = 0.90,
                                                    make_backup: bool = True) -> Dict:
    """
    Convenience: read entries CSV, auto-filter only rows newer than last update (if _meta present),
    and incrementally update the JSON.
    """
    if not os.path.exists(entries_csv_path):
        print(f"[Update] Entries CSV not found: {entries_csv_path}")
        return {}

    df = pd.read_csv(entries_csv_path, parse_dates=[time_col])
    if df.empty:
        print("[Update] Entries CSV is empty.")
        return {}

    last_epoch = 0
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                old = json.load(f)
            last_epoch = int(((old or {}).get("_meta") or {}).get("last_updated_epoch", 0))
        except Exception:
            last_epoch = 0

    if last_epoch > 0:
        cutoff = pd.to_datetime(pd.Timestamp.utcfromtimestamp(last_epoch))
        # keep strictly newer than last update
        entries_new = df[df[time_col] > cutoff]
    else:
        entries_new = df.copy()

    if entries_new.empty:
        print("[Update] No new entries since last update; nothing to merge.")
        return {}

    return update_fitted_ranges_by_ticker(
        existing_json_path=json_path,
        entries_new=entries_new,
        out_json_path=out_json_path,
        q_low=q_low, q_high=q_high,
        make_backup=make_backup
    )

def run_step2_from_entries_csv(entries_csv: str,
                               q_low: float = 0.20,
                               q_high: float = 0.80,
                               shrink_k: int = 200,
                               out_json="fitted_indicator_ranges_by_ticker.json",
                               out_csv="fitted_indicator_ranges_by_ticker.csv"):
    if not os.path.exists(entries_csv):
        print(f"[Step2-only] entries CSV not found: {entries_csv}")
        return
    entries = pd.read_csv(entries_csv, parse_dates=["EntryTime"]) if "EntryTime" in pd.read_csv(entries_csv, nrows=1).columns else pd.read_csv(entries_csv)
    if entries.empty:
        print("[Step2-only] entries CSV is empty.")
        return
    fit_indicator_ranges_by_ticker(entries,
                                   out_json=out_json,
                                   out_csv=out_csv,
                                   q_low=q_low,
                                   q_high=q_high,
                                   shrink_k=shrink_k)


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    # Step 1: scan & simulate 2% target
    entries = run_step1_scan(INDICATORS_DIR, ENTRY_PARAMS_DEFAULT, out_csv="entries_2pct_results.csv")

    if not entries.empty:
        # ---- FULL RECOMPUTE (per-ticker winners-only with shrinkage to global) ----
        fit_indicator_ranges_by_ticker(entries,
    out_json="fitted_indicator_ranges_by_ticker15_85.json",
    out_csv="fitted_indicator_ranges_by_ticker15_85.csv",
    q_low=0.20, q_high=0.80,   # tightened
    shrink_k=200)


if __name__ == "__main__":
    run_step2_from_entries_csv(
        entries_csv="entries_2pct_results.csv",
        q_low=0.10, q_high=0.90, shrink_k=200,
        out_json="fitted_indicator_ranges_by_ticke10_90.json",
        out_csv="fitted_indicator_ranges_by_ticker10_90.csv"
    )
