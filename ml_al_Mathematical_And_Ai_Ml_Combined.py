# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:24:18 2025

@author: Saarit

Combined pipeline: Intraday 2% backtest (Step-1) + Per-ticker ML scorer (Step-2)

Order of execution (main):
  1) Step-1 scans CSVs in INDICATORS_DIR and writes entries_2pct_results.csv
  2) Fits winners-only indicator ranges (multiple quantile bands)
  3) Step-2 trains per-ticker models from entries_2pct_results.csv,
     saves OOF predictions and serialized models.

This file merges:
- et4_mathematical_analysis_5min_data_1 (scan + ranges)
- et4_ai_ml_model_fortickers (per-ticker ML)

Parameters/thresholds preserved exactly from the originals provided.
"""

# ============================
# Common imports
# ============================
import os
import json
import time
import warnings
from dataclasses import dataclass
from datetime import time as dt_time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from collections import defaultdict

# ML imports (used in Step-2)
try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold
    try:
        # Present in sklearn >= 1.1
        from sklearn.model_selection import StratifiedGroupKFold
        HAS_SGK = True
    except Exception:
        StratifiedGroupKFold = None
        HAS_SGK = False
    from sklearn.metrics import roc_auc_score, brier_score_loss
    import joblib
    _HAS_SKLEARN = True
except Exception as _e:
    HAS_SGK = False
    _HAS_SKLEARN = False

# ============================
# --------- STEP 1 ----------
# Intraday 2% target backtest + winners-only fitted indicator ranges
# (Values unchanged)
# ============================

INDICATORS_DIR = "main_indicators_july_5min2"

# Date window (training universe)
START_DATE = "2024-08-05"
END_DATE   = "2025-08-22"

# --- LIGHT LOGGING CONFIG ---
LOG_PRINT_EVERY_FILES = 25
LOG_TICKERS_WITH_ENTRIES_MAX = 40

# Entry rules (kept simple, tunable)
ENTRY_PARAMS_DEFAULT = {
    "target_pct": 0.020,          # 2% target
    "stop_pct": None,             # None => no SL
    "min_bars_after_entry": 1,

    # Directional bias
    "daily_change_min_bull":  0.0,
    "daily_change_max_bear":  0.0,

    # Trend filters
    "ema_trend_bull": True,
    "ema_trend_bear": True,

    # Strength/momentum windows
    "adx_min": 20.0, "adx_max": 55.0,
    "rsi_bull_low": 45.0, "rsi_bull_high": 75.0,
    "rsi_bear_low": 25.0, "rsi_bear_high": 55.0,
    "stochd_bull_min": 45.0,
    "stochd_bear_max": 55.0,

    # VWAP relation (distance frac)
    "vwap_bull_min": 0.000,
    "vwap_bear_min": 0.000,

    # Volume participation
    "roll_vol_window": 10,
    "vol_rel_min": 1.10,

    # Intra streak
    "streak_n": 1,
    "streak_thr_bull": 0.00,
    "streak_thr_bear": 0.00,

    # Trading day time window (5-min bars)
    "session_start": dt_time(9, 25),
    "session_end":   dt_time(15, 10),
}

# Columns we need
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

# ---------- Utilities ----------

def _coalesce_date_columns(df: pd.DataFrame) -> pd.Series:
    """Coalesce any date-like columns into a single tz-aware IST datetime Series."""
    india_tz = "Asia/Kolkata"
    cand = [c for c in df.columns if c.lower() in ("date", "timestamp") or c.lower().startswith("date")]
    if not cand:
        raise ValueError("No date-like column found.")
    def _rank(c):
        cl = c.lower()
        if cl == "date": return (0, c)
        if cl.startswith("date"): return (1, c)
        if cl == "timestamp": return (2, c)
        return (3, c)
    cand = sorted(dict.fromkeys(cand), key=_rank)

    parsed = []
    for c in cand:
        s = pd.to_datetime(df[c], errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(india_tz)
        else:
            s = s.dt.tz_convert(india_tz)
        parsed.append(s)

    dt_series = parsed[0]
    for s in parsed[1:]:
        dt_series = dt_series.where(dt_series.notna(), s)
    return dt_series


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_read_csv(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    raw = pd.read_csv(path, low_memory=False)

    # Build canonical 'date'
    try:
        dt_series = _coalesce_date_columns(raw)
    except Exception:
        dt_series = pd.to_datetime(raw.get("date", pd.Series([], dtype="object")), errors="coerce")
        if getattr(dt_series.dt, "tz", None) is None:
            dt_series = dt_series.dt.tz_localize("Asia/Kolkata")

    date_like = [c for c in raw.columns if c.lower() in ("date","timestamp") or c.lower().startswith("date")]
    keep_cols = [c for c in raw.columns if c not in date_like]
    df = raw[keep_cols].copy()
    df.insert(0, "date", dt_series)

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    mask = (df["date"].dt.date >= pd.to_datetime(START_DATE).date()) & \
           (df["date"].dt.date <= pd.to_datetime(END_DATE).date())
    df = df.loc[mask].copy()
    if df.empty:
        return df

    if "date_only" not in df.columns:
        df["date_only"] = df["date"].dt.date

    df = _ensure_columns(df, NEEDED_COLS)
    num_cols = [c for c in NEEDED_COLS if c not in ("date", "date_only")]
    df = _coerce_numeric(df, num_cols)

    return df


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

# ---------- Entry logic ----------

def _row_is_bull_entry(df: pd.DataFrame, i: int, p: Dict[str, Any]) -> bool:
    r = df.iloc[i]
    if not _in_time_window(r["date"], p["session_start"], p["session_end"]):
        return False
    req = ["close","VWAP","RSI","ADX","Stoch_%D","Daily_Change","EMA_50","EMA_200","volume","rolling_vol10","Intra_Change"]
    if any(pd.isna(r.get(c)) for c in req):
        return False
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
    if not _intra_streak_ok(df["Intra_Change"], i, p["streak_n"], p["streak_thr_bull"], bullish=True):
        return False
    return True


def _row_is_bear_entry(df: pd.DataFrame, i: int, p: Dict[str, Any]) -> bool:
    r = df.iloc[i]
    if not _in_time_window(r["date"], p["session_start"], p["session_end"]):
        return False
    req = ["close","VWAP","RSI","ADX","Stoch_%D","Daily_Change","EMA_50","EMA_200","volume","rolling_vol10","Intra_Change"]
    if any(pd.isna(r.get(c)) for c in req):
        return False
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
    if r["close"] > r["VWAP"] * (1 - p["vwap_bear_min"]) and p["vwap_bear_min"] > 0:
        return False
    if r["volume"] < p["vol_rel_min"] * r["rolling_vol10"]:
        return False
    if not _intra_streak_ok(df["Intra_Change"], i, p["streak_n"], p["streak_thr_bear"], bullish=False):
        return False
    return True


def _simulate_exit_same_day(df_day: pd.DataFrame, entry_idx: int, side: str, target_pct: float,
                            stop_pct: Optional[float]) -> Tuple[bool, pd.Timestamp, float, float]:
    """Return (hit_anywhere, exit_time, exit_price, mfe_fraction)."""
    entry_row   = df_day.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    fwd = df_day.iloc[entry_idx + 1:].copy()
    if fwd.empty:
        return (False, entry_row["date"], entry_price, 0.0)

    if side == "bull":
        day_high = float(fwd["high"].max())
        mfe = (day_high - entry_price) / entry_price
        hit_anywhere = (mfe >= target_pct)
        target = entry_price * (1 + target_pct)
        stop   = entry_price * (1 - stop_pct) if stop_pct else None
    else:
        day_low = float(fwd["low"].min())
        mfe = (entry_price - day_low) / entry_price
        hit_anywhere = (mfe >= target_pct)
        target = entry_price * (1 - target_pct)
        stop   = entry_price * (1 + stop_pct) if stop_pct else None

    exit_px = None
    exit_time = None

    for _, r in fwd.iterrows():
        if side == "bull":
            if stop is not None and r["low"] <= stop:
                exit_px = stop; exit_time = r["date"]; break
            if r["high"] >= target:
                exit_px = target; exit_time = r["date"]; break
        else:
            if stop is not None and r["high"] >= stop:
                exit_px = stop; exit_time = r["date"]; break
            if r["low"] <= target:
                exit_px = target; exit_time = r["date"]; break

    if exit_px is None:
        last = fwd.iloc[-1]
        exit_px = float(last["close"]); exit_time = last["date"]

    return (bool(hit_anywhere), exit_time, float(exit_px), float(mfe))


# ---------- Entry generation ----------

def generate_intraday_entries_for_file(ticker: str,
                                       path: str,
                                       params: Dict[str, Any]) -> pd.DataFrame:
    df = _safe_read_csv(path, usecols=NEEDED_COLS)
    if df.empty:
        return pd.DataFrame()

    # Derived features
    eps = 1e-9
    df["rolling_vol10"] = _add_rolling_vol(df, params["roll_vol_window"])

    df["VWAP_Dist"]   = df["close"] / (df["VWAP"] + eps)    - 1.0
    df["EMA50_Dist"]  = df["close"] / (df["EMA_50"] + eps)  - 1.0
    df["EMA200_Dist"] = df["close"] / (df["EMA_200"] + eps) - 1.0
    df["SMA20_Dist"]  = df["close"] / (df["20_SMA"] + eps)  - 1.0

    df["ATR_Pct"] = df["ATR"] / (df["close"] + eps)

    bbw = (df["Upper_Band"] - df["Lower_Band"])
    df["BandWidth"] = bbw / (df["close"] + eps)
    df["BBP"] = (df["close"] - df["Lower_Band"]) / (bbw + eps)

    df["VolRel"] = df["volume"] / (df["rolling_vol10"] + eps)
    df["StochK"] = df.get("Stoch_%K")
    df["StochD"] = df.get("Stoch_%D")

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

            if _row_is_bull_entry(g, i, params):
                hit, ex_t, ex_px, mfe = _simulate_exit_same_day(g, i, "bull", params["target_pct"], params["stop_pct"])
                r = g.iloc[i]
                rows.append({
                    "Ticker": ticker, "Side": "Bullish",
                    "EntryTime": r["date"], "EntryPrice": float(r["close"]),
                    "ExitTime": ex_t, "ExitPrice": ex_px,
                    "HitTarget": bool(hit), "ReturnPct": (ex_px / float(r["close"]) - 1.0) * 100.0,
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


# ---------- Winners-only per-ticker ranges (with shrinkage) ----------

def _qband(series: pd.Series, lo: float, hi: float) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    lo_v, hi_v = np.quantile(s, [lo, hi])
    return float(lo_v), float(hi_v)


def fit_indicator_ranges_by_ticker(entries: pd.DataFrame,
                                   out_json: str = "fitted_indicator_ranges_by_ticker.json",
                                   out_csv: str = "fitted_indicator_ranges_by_ticker.csv",
                                   q_low: float = 0.10,
                                   q_high: float = 0.90,
                                   shrink_k: int = 200) -> pd.DataFrame:
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

        for name in INDICATOR_SPECS["range_both"]:
            if name in sub.columns:
                lo_t, hi_t = _qband(sub[name], ql, qh)
                lo_g = hi_g = None
                g = global_side_cfg.get(f"{name}_range")
                if isinstance(g, (list, tuple)) and len(g) == 2:
                    lo_g, hi_g = float(g[0]), float(g[1])
                side_cfg[f"{name}_range"] = [_blend(lo_t, lo_g), _blend(hi_t, hi_g)]

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

        for name in INDICATOR_SPECS["min_both"]:
            if name not in sub.columns:
                continue
            lo_t, _ = _qband(sub[name], ql, qh)
            base_g = global_side_cfg.get(f"{name}_min")
            side_cfg[f"{name}_min"] = _blend(lo_t, base_g)

        return side_cfg, n

    cfg_global = {}
    for side in ["Bullish", "Bearish"]:
        gsub = entries[(entries["Side"] == side) & (entries["HitTarget"] == True)].copy()
        side_cfg = {}
        if len(gsub):
            side_cfg, _ = _build_side_cfg(gsub, side, {}, q_low, q_high, shrink_k_val=0)
        cfg_global[side] = side_cfg

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
                if k in ("Samples_wins", "Quantiles_used"):
                    continue
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


# ---------- Incremental updater ----------

def _compute_new_cfg(entries_new: pd.DataFrame,
                     q_low: float = 0.10,
                     q_high: float = 0.90) -> Dict[str, Dict]:
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

            for name in INDICATOR_SPECS["range_both"]:
                if name in sub.columns:
                    lo, hi = _qband(sub[name], q_low, q_high)
                    side_cfg[f"{name}_range"] = [lo, hi]

            for name in INDICATOR_SPECS["min_bull_max_bear"]:
                if name in sub.columns:
                    lo, hi = _qband(sub[name], q_low, q_high)
                    if side == "Bullish":
                        side_cfg[f"{name}_min"] = lo
                    else:
                        side_cfg[f"{name}_max"] = hi

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
    old_n = int(old_side.get("Samples_wins", 0)) if old_side else 0
    new_n = int(new_side.get("Samples_wins", 0)) if new_side else 0
    tot_n = old_n + new_n
    out = {}
    out["Samples_wins"] = tot_n
    out["Quantiles_used"] = new_side.get("Quantiles_used", old_side.get("Quantiles_used", [0.10, 0.90]))

    if tot_n == 0:
        return out

    w_new = new_n / tot_n

    for name in INDICATOR_SPECS["range_both"]:
        k = f"{name}_range"
        lo_old, hi_old = (old_side.get(k, [np.nan, np.nan]) if old_side else [np.nan, np.nan])
        lo_new, hi_new = (new_side.get(k, [np.nan, np.nan]) if new_side else [np.nan, np.nan])
        lo = _blend_scalar(lo_old, lo_new, w_new)
        hi = _blend_scalar(hi_old, hi_new, w_new)
        out[k] = [lo, hi]

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

    merged_with_meta = {
        **merged,
        "_meta": {
            "last_updated_epoch": int(time.time()),
            "q_low": q_low, "q_high": q_high,
            "note": "Weighted incremental merge of winners-only per-ticker ranges."
        }
    }

    target = out_json_path or existing_json_path
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


def run_step2_from_entries_csv(entries_csv: str,
                               q_low: float = 0.20,
                               q_high: float = 0.80,
                               shrink_k: int = 200,
                               out_json="fitted_indicator_ranges_by_ticker.json",
                               out_csv="fitted_indicator_ranges_by_ticker.csv"):
    if not os.path.exists(entries_csv):
        print(f"[Step2-only] entries CSV not found: {entries_csv}")
        return
    head = pd.read_csv(entries_csv, nrows=1)
    parse_cols = ["EntryTime"] if "EntryTime" in head.columns else None
    entries = pd.read_csv(entries_csv, parse_dates=parse_cols)
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
# --------- STEP 2 ----------
# Per-ticker ML scorer for 'target hit' with robust CV
# ============================

# Default paths for Step-2
DATA_PATH_DEFAULT = "entries_2pct_results.csv"
SAVE_MODELS_DEFAULT = True
MODELS_PATH_DEFAULT = "per_ticker_models.joblib"
OOF_PATH_DEFAULT = "oof_predictions.csv"

# Feature set (as in original)
FEATURES_DEFAULT = [
    "RSI","ADX","StochD","StochK","VWAP_Dist","VolRel","Daily_Change","Intra_Change",
    "EMA50_Dist","EMA200_Dist","SMA20_Dist","ATR_Pct","BBP","BandWidth",
    "MACD_Hist","CCI","MFI","OBV_Slope10"
]

# CV settings
MAX_SPLITS = 5
MIN_SPLITS = 2
MAX_NEIGHBORS = 25
MIN_NEIGHBORS = 3


def _to_float_pct(x):
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip().rstrip("%")
        return float(s)
    except Exception:
        return np.nan


def recompute_hit(df: pd.DataFrame) -> pd.Series:
    rp = df["ReturnPct"].apply(_to_float_pct)
    side = df["Side"].astype(str).str.strip().str.lower()
    hit = ( (side == "bullish") & (rp >= 2.0) ) | ( (side == "bearish") & (rp <= -2.0) )
    return hit.astype(int)


def pick_cv_splits(X, y, groups, n_splits: int):
    if _HAS_SKLEARN and HAS_SGK and len(np.unique(y)) >= 2 and len(np.unique(groups)) >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cv.split(X, y, groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(X, y, groups)


def safe_k(n_train: int) -> int:
    if n_train <= 0:
        return MIN_NEIGHBORS
    k = int(np.sqrt(n_train))
    k = max(MIN_NEIGHBORS, min(k, MAX_NEIGHBORS, n_train))
    return k


@dataclass
class PerTickerModel:
    kind: str                   # "knn" or "constant"
    pipeline: Pipeline = None   # for "knn"
    const_p: float = None       # for "constant"
    features: List[str] = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].copy()
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.features]
        if self.kind == "constant":
            return np.full((len(X),), float(self.const_p), dtype=float)
        return np.clip(self.pipeline.predict(X), 0.0, 1.0)


def run_ai_ml_pipeline(data_path: str = DATA_PATH_DEFAULT,
                       save_models: bool = SAVE_MODELS_DEFAULT,
                       models_path: str = MODELS_PATH_DEFAULT,
                       oof_path: str = OOF_PATH_DEFAULT,
                       features: Optional[List[str]] = None):
    if not _HAS_SKLEARN:
        print("[ML] scikit-learn not available; skipping Step-2.")
        return None, None

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    if "EntryTime" in df.columns:
        df["EntryTime"] = pd.to_datetime(df["EntryTime"], errors="coerce")
    if "DateOnly" in df.columns:
        df["DateOnly"] = df["DateOnly"].astype(str)
    else:
        if "EntryTime" in df.columns:
            df["DateOnly"] = df["EntryTime"].dt.date.astype(str)
        else:
            df["DateOnly"] = "NA"

    df["HitTarget_bin"] = recompute_hit(df)

    feats = features if features is not None else FEATURES_DEFAULT
    avail_feats = [c for c in feats if c in df.columns]
    if not avail_feats:
        raise ValueError("None of the requested FEATURES are present in the CSV.")
    feats = avail_feats

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Side"] = df["Side"].astype(str).str.strip().str.capitalize()
    df = df.dropna(subset=["Ticker","Side","DateOnly"]).reset_index(drop=True)

    all_oof = []
    models: Dict[str, PerTickerModel] = {}

    tickers = sorted(df["Ticker"].unique())
    print(f"Tickers to model: {len(tickers)}")

    for tkr in tickers:
        dft = df[df["Ticker"] == tkr].reset_index(drop=True)
        y = dft["HitTarget_bin"].to_numpy(dtype=int)
        groups = dft["DateOnly"].to_numpy()

        Xt = dft[feats].copy()
        for c in feats:
            if c not in Xt.columns:
                Xt[c] = np.nan
        Xt = Xt[feats].to_numpy(dtype=float)

        n = len(dft)
        uniq_dates = np.unique(groups).size
        n_splits = max(MIN_SPLITS, min(MAX_SPLITS, uniq_dates))

        if np.unique(y).size < 2 or n < 10 or uniq_dates < 2:
            p_const = float((y.mean() * n + 0.5) / (n + 1.0))
            models[tkr] = PerTickerModel(kind="constant", const_p=p_const, features=feats)
            oof_pred = np.full(n, p_const, dtype=float)
            all_oof.append(pd.DataFrame({
                "Ticker": tkr, "EntryTime": dft.get("EntryTime", pd.NaT),
                "Side": dft["Side"], "HitTarget": y, "PredProb": oof_pred, "Fold": -1
            }))
            print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  -> CONSTANT p={p_const:.3f} (degenerate or tiny)")
            continue

        oof_pred = np.zeros(n, dtype=float)
        fold_ids = np.full(n, -1, dtype=int)

        splits = list(pick_cv_splits(Xt, y, groups, n_splits=n_splits))
        if len(splits) < 2:
            p_const = float((y.mean() * n + 0.5) / (n + 1.0))
            models[tkr] = PerTickerModel(kind="constant", const_p=p_const, features=feats)
            oof_pred[:] = p_const
            fold_ids[:] = -1
            print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  -> CONSTANT (couldn't split CV)")
        else:
            for fold, (tr, te) in enumerate(splits, start=1):
                Xtr, Xte = Xt[tr], Xt[te]
                ytr = y[tr]

                k = safe_k(len(tr))
                pipe = Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler(with_mean=True, with_std=True)),
                    ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance", p=2)),
                ])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xtr, ytr.astype(float))

                preds = np.clip(pipe.predict(Xte), 0.0, 1.0)
                oof_pred[te] = preds
                fold_ids[te] = fold

            k_full = safe_k(n)
            final_pipe = Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True, with_std=True)),
                ("knn", KNeighborsRegressor(n_neighbors=k_full, weights="distance", p=2)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_pipe.fit(Xt, y.astype(float))

            models[tkr] = PerTickerModel(kind="knn", pipeline=final_pipe, features=feats)

            try:
                if len(np.unique(y)) == 2 and len(np.unique(oof_pred)) > 1:
                    auc = roc_auc_score(y, oof_pred)
                else:
                    auc = np.nan
            except Exception:
                auc = np.nan
            try:
                brier = brier_score_loss(y, oof_pred)
            except Exception:
                brier = np.nan

            topk = max(1, int(0.2 * n))
            idx_top = np.argsort(-oof_pred)[:topk]
            hr_top = y[idx_top].mean() if topk > 0 else np.nan

            print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  folds={len(splits)} | "
                  f"AUC={auc:.3f}  Brier={brier:.3f}  Top20%HR={hr_top:.3f}")

        all_oof.append(pd.DataFrame({
            "Ticker": tkr,
            "EntryTime": dft.get("EntryTime", pd.NaT),
            "Side": dft["Side"],
            "HitTarget": y,
            "PredProb": oof_pred,
            "Fold": fold_ids
        }))

    oof_df = pd.concat(all_oof, ignore_index=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved OOF predictions → {oof_path} (rows={len(oof_df)})")

    if save_models:
        joblib.dump({"models": models, "features": feats}, models_path)
        print(f"Saved per-ticker models → {models_path}")

    return models, oof_df


# ============================
# MAIN: Step-1 then Step-2
# ============================
if __name__ == "__main__":
    # ---- Step 1: scan & simulate 2% target ----
    entries = run_step1_scan(INDICATORS_DIR, ENTRY_PARAMS_DEFAULT, out_csv=DATA_PATH_DEFAULT)

    if not entries.empty:
        # Full recompute of winners-only ranges (baseline 10–90)
        fit_indicator_ranges_by_ticker(entries,
                                       out_json="fitted_indicator_ranges_by_ticker.json",
                                       out_csv="fitted_indicator_ranges_by_ticker.csv",
                                       q_low=0.10, q_high=0.90,
                                       shrink_k=200)

        # Additional bands (as per provided script)
        run_step2_from_entries_csv(
            entries_csv=DATA_PATH_DEFAULT,
            q_low=0.10, q_high=0.90, shrink_k=200,
            out_json="fitted_indicator_ranges_by_ticker10_90.json",
            out_csv="fitted_indicator_ranges_by_ticker10_90.csv"
        )
        run_step2_from_entries_csv(
            entries_csv=DATA_PATH_DEFAULT,
            q_low=0.15, q_high=0.85, shrink_k=200,
            out_json="fitted_indicator_ranges_by_ticker15_85.json",
            out_csv="fitted_indicator_ranges_by_ticker15_85.csv"
        )
        run_step2_from_entries_csv(
            entries_csv=DATA_PATH_DEFAULT,
            q_low=0.20, q_high=0.80, shrink_k=200,
            out_json="fitted_indicator_ranges_by_ticker20_80.json",
            out_csv="fitted_indicator_ranges_by_ticker20_80.csv"
        )
        run_step2_from_entries_csv(
            entries_csv=DATA_PATH_DEFAULT,
            q_low=0.25, q_high=0.75, shrink_k=200,
            out_json="fitted_indicator_ranges_by_ticker25_75.json",
            out_csv="fitted_indicator_ranges_by_ticker25_75.csv"
        )

        # ---- Step 2: per-ticker ML training ----
        run_ai_ml_pipeline(
            data_path=DATA_PATH_DEFAULT,
            save_models=SAVE_MODELS_DEFAULT,
            models_path=MODELS_PATH_DEFAULT,
            oof_path=OOF_PATH_DEFAULT,
            features=FEATURES_DEFAULT,
        )
    else:
        print("[Main] No entries generated in Step-1; skipping Step-2/ML and ranges fit.")
