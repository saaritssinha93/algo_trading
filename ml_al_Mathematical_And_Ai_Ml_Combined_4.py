# -*- coding: utf-8 -*-
"""
Weekly-frozen trading bundle with stacked ML, EV thresholds, regime gate, and drift controls.

Pipeline:
  Step-1  Intraday 2% backtest → winners-only per-ticker indicator ranges
  Step-2  Per-ticker KNN + Global HGB + Stacker (OOF), isotonic calibration, per-ticker thresholds (Youden + EV)
  Extras  Regime gate features/model, PSI drift reference

Outputs under models/ (STAMP = END_DATE yyyymmdd):
  - entries_2pct_results_{STAMP}.csv
  - fitted_indicator_ranges_by_ticker_{STAMP}.json/.csv (+ extra bands)
  - per_ticker_models_{STAMP}.joblib  (all models, calibrators, thresholds, drift, regime)
  - oof_predictions_{STAMP}.csv
  - manifest_{STAMP}.json
"""

# =============== Common imports ===============
import os, json, time, warnings
from dataclasses import dataclass
from datetime import time as dt_time, timedelta
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# =============== ML stack ===============
try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        HAS_SGK = True
    except Exception:
        StratifiedGroupKFold = None
        HAS_SGK = False
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.isotonic import IsotonicRegression
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import joblib
    _HAS_SKLEARN = True
except Exception:
    HAS_SGK = False
    _HAS_SKLEARN = False

# =============== STEP 1: Backtest & ranges ===============
INDICATORS_DIR = "main_indicators_july_5min2"

# Training window
START_DATE = "2024-08-10"   # inclusive
END_DATE   = "2025-08-22"   # inclusive

# Validity/future-proofing
GRACE_DAYS = 7
LIVE_DEFENSIVE_BUMP = 0.02
THRESH_FLOOR = 0.50
THRESH_CEIL  = 0.90

# Drift/Regime defaults
PSI_CUTOFF = 0.20           # bump/gate if drift > this
PSI_BUMP_PP = 0.02
KEY_DRIFT_FEATURES = ["VWAP_Dist","VolRel","ATR_Pct"]

# PnL assumptions for EV thresholds (set realistic, net of costs/slippage)
EV_R_HIT  = +0.018   # +1.8% net when target fires
EV_R_MISS = -0.010   # -1.0% net when it doesn't

# Logging cadence
LOG_PRINT_EVERY_FILES = 25
LOG_TICKERS_WITH_ENTRIES_MAX = 40

ENTRY_PARAMS_DEFAULT = {
    "target_pct": 0.020,
    "stop_pct": None,
    "min_bars_after_entry": 1,
    "daily_change_min_bull":  0.0,
    "daily_change_max_bear":  0.0,
    "ema_trend_bull": True,
    "ema_trend_bear": True,
    "adx_min": 20.0, "adx_max": 55.0,
    "rsi_bull_low": 45.0, "rsi_bull_high": 75.0,
    "rsi_bear_low": 25.0, "rsi_bear_high": 55.0,
    "stochd_bull_min": 45.0,
    "stochd_bear_max": 55.0,
    "vwap_bull_min": 0.000,
    "vwap_bear_min": 0.000,
    "roll_vol_window": 10,
    "vol_rel_min": 1.10,
    "streak_n": 1,
    "streak_thr_bull": 0.00,
    "streak_thr_bear": 0.00,
    "session_start": dt_time(9, 25),
    "session_end":   dt_time(15, 10),
}

NEEDED_COLS = [
    "date","open","high","low","close","volume","date_only",
    "RSI","ATR","EMA_50","EMA_200","20_SMA","VWAP","CCI","MFI",
    "OBV","MACD","MACD_Signal","MACD_Hist","Upper_Band","Lower_Band",
    "Recent_High","Recent_Low","Stoch_%K","Stoch_%D","ADX",
    "Daily_Change","Intra_Change"
]

INDICATOR_SPECS = {
    "range_both": ["RSI","ADX","CCI","MFI","BBP","BandWidth","ATR_Pct"],
    "min_bull_max_bear": [
        "StochD","StochK","MACD_Hist","VWAP_Dist","Daily_Change",
        "EMA50_Dist","EMA200_Dist","SMA20_Dist","OBV_Slope10"
    ],
    "min_both": ["VolRel"],
}

def _coalesce_date_columns(df: pd.DataFrame) -> pd.Series:
    india_tz = "Asia/Kolkata"
    cand = [c for c in df.columns if c.lower() in ("date","timestamp") or c.lower().startswith("date")]
    if not cand:
        raise ValueError("No date-like column found.")
    def _rank(c):
        cl = c.lower()
        if cl == "date": return (0,c)
        if cl.startswith("date"): return (1,c)
        if cl == "timestamp": return (2,c)
        return (3,c)
    cand = sorted(dict.fromkeys(cand), key=_rank)
    parsed = []
    for c in cand:
        s = pd.to_datetime(df[c], errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(india_tz)
        else:
            s = s.dt.tz_convert(india_tz)
        parsed.append(s)
    out = parsed[0]
    for s in parsed[1:]:
        out = out.where(out.notna(), s)
    return out

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

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    raw = pd.read_csv(path, low_memory=False)
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
    if df.empty: return df
    if "date_only" not in df.columns:
        df["date_only"] = df["date"].dt.date
    df = _ensure_columns(df, NEEDED_COLS)
    num_cols = [c for c in NEEDED_COLS if c not in ("date","date_only")]
    df = _coerce_numeric(df, num_cols)
    return df

def _add_rolling_vol(df: pd.DataFrame, w: int) -> pd.Series:
    return df["volume"].rolling(w, min_periods=w).mean()

def _in_time_window(ts: pd.Timestamp, start_t: dt_time, end_t: dt_time) -> bool:
    t = ts.time()
    return (t >= start_t) and (t <= end_t)

def _intra_streak_ok(series_intra: pd.Series, idx: int, n: int, thr: float, bullish: bool) -> bool:
    if n <= 0 or idx - (n - 1) < 0: return False
    seg = series_intra.iloc[idx - (n - 1):idx + 1]
    if seg.isna().any(): return False
    return (seg >= thr).all() if bullish else (seg <= -abs(thr)).all()

def _row_is_bull_entry(df: pd.DataFrame, i: int, p: Dict[str, Any]) -> bool:
    r = df.iloc[i]
    if not _in_time_window(r["date"], p["session_start"], p["session_end"]): return False
    req = ["close","VWAP","RSI","ADX","Stoch_%D","Daily_Change","EMA_50","EMA_200","volume","rolling_vol10","Intra_Change"]
    if any(pd.isna(r.get(c)) for c in req): return False
    if r["Daily_Change"] < p["daily_change_min_bull"]: return False
    if p["ema_trend_bull"] and not (r["EMA_50"] >= r["EMA_200"]): return False
    if not (p["adx_min"] <= r["ADX"] <= p["adx_max"]): return False
    if not (p["rsi_bull_low"] <= r["RSI"] <= p["rsi_bull_high"]): return False
    if r["Stoch_%D"] < p["stochd_bull_min"]: return False
    if r["close"] < r["VWAP"] * (1 + p["vwap_bull_min"]): return False
    if r["volume"] < p["vol_rel_min"] * r["rolling_vol10"]: return False
    if not _intra_streak_ok(df["Intra_Change"], i, p["streak_n"], p["streak_thr_bull"], bullish=True): return False
    return True

def _row_is_bear_entry(df: pd.DataFrame, i: int, p: Dict[str, Any]) -> bool:
    r = df.iloc[i]
    if not _in_time_window(r["date"], p["session_start"], p["session_end"]): return False
    req = ["close","VWAP","RSI","ADX","Stoch_%D","Daily_Change","EMA_50","EMA_200","volume","rolling_vol10","Intra_Change"]
    if any(pd.isna(r.get(c)) for c in req): return False
    if r["Daily_Change"] > p["daily_change_max_bear"]: return False
    if p["ema_trend_bear"] and not (r["EMA_50"] <= r["EMA_200"]): return False
    if not (p["adx_min"] <= r["ADX"] <= p["adx_max"]): return False
    if not (p["rsi_bear_low"] <= r["RSI"] <= p["rsi_bear_high"]): return False
    if r["Stoch_%D"] > p["stochd_bear_max"]: return False
    if r["close"] > r["VWAP"] * (1 - p["vwap_bear_min"]) and p["vwap_bear_min"] > 0: return False
    if r["volume"] < p["vol_rel_min"] * r["rolling_vol10"]: return False
    if not _intra_streak_ok(df["Intra_Change"], i, p["streak_n"], p["streak_thr_bear"], bullish=False): return False
    return True

def _simulate_exit_same_day(df_day: pd.DataFrame, entry_idx: int, side: str, target_pct: float,
                            stop_pct: Optional[float]) -> Tuple[bool, pd.Timestamp, float, float]:
    entry_row   = df_day.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    fwd = df_day.iloc[entry_idx + 1:].copy()
    if fwd.empty: return (False, entry_row["date"], entry_price, 0.0)
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
    exit_px = None; exit_time = None
    for _, r in fwd.iterrows():
        if side == "bull":
            if stop is not None and r["low"] <= stop:  exit_px = stop;   exit_time = r["date"]; break
            if r["high"] >= target:                    exit_px = target; exit_time = r["date"]; break
        else:
            if stop is not None and r["high"] >= stop: exit_px = stop;   exit_time = r["date"]; break
            if r["low"]  <= target:                    exit_px = target; exit_time = r["date"]; break
    if exit_px is None:
        last = fwd.iloc[-1]
        exit_px = float(last["close"]); exit_time = last["date"]
    return (bool(hit_anywhere), exit_time, float(exit_px), float(mfe))

def generate_intraday_entries_for_file(ticker: str, path: str, params: Dict[str, Any]) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty: return pd.DataFrame()
    eps = 1e-9
    df["rolling_vol10"] = _add_rolling_vol(df, params["roll_vol_window"])
    df["VWAP_Dist"]   = df["close"] / (df["VWAP"] + eps) - 1.0
    df["EMA50_Dist"]  = df["close"] / (df["EMA_50"] + eps) - 1.0
    df["EMA200_Dist"] = df["close"] / (df["EMA_200"] + eps) - 1.0
    df["SMA20_Dist"]  = df["close"] / (df["20_SMA"] + eps) - 1.0
    df["ATR_Pct"]     = df["ATR"]   / (df["close"] + eps)
    bbw = (df["Upper_Band"] - df["Lower_Band"])
    df["BandWidth"] = bbw / (df["close"] + eps)
    df["BBP"] = (df["close"] - df["Lower_Band"]) / (bbw + eps)
    df["VolRel"] = df["volume"] / (df["rolling_vol10"] + eps)
    df["StochK"] = df.get("Stoch_%K"); df["StochD"] = df.get("Stoch_%D")
    obv_prev = df["OBV"].shift(10)
    df["OBV_Slope10"] = (df["OBV"] - obv_prev) / (obv_prev.abs() + eps)

    rows = []
    for d, g in df.groupby("date_only", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 3: continue
        eligible = g["date"].apply(lambda ts: _in_time_window(ts, params["session_start"], params["session_end"]))
        idxs = list(np.where(eligible.values)[0])
        for i in idxs:
            if i >= len(g) - params["min_bars_after_entry"]: continue
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
                    "ATR_Pct": r["ATR_Pct"], "BBP": r["BBP"], "BandWidth": r["BandWidth"],
                    "MACD_Hist": r["MACD_Hist"], "CCI": r["CCI"], "MFI": r["MFI"],
                    "OBV_Slope10": r["OBV_Slope10"], "EMA_50": r["EMA_50"], "EMA_200": r["EMA_200"], "DateOnly": d
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
                    "ATR_Pct": r["ATR_Pct"], "BBP": r["BBP"], "BandWidth": r["BandWidth"],
                    "MACD_Hist": r["MACD_Hist"], "CCI": r["CCI"], "MFI": r["MFI"],
                    "OBV_Slope10": r["OBV_Slope10"], "EMA_50": r["EMA_50"], "EMA_200": r["EMA_200"], "DateOnly": d
                })
    return pd.DataFrame(rows)

def run_step1_scan(indicators_dir: str = INDICATORS_DIR,
                   params: Dict[str, Any] = ENTRY_PARAMS_DEFAULT,
                   out_csv: str = "entries_2pct_results.csv") -> pd.DataFrame:
    all_rows = []
    files = [f for f in os.listdir(indicators_dir) if f.endswith("_main_indicators.csv")]
    files.sort()
    nfiles = len(files)
    print(f"\n[Step1] Scanning {nfiles} files in '{indicators_dir}' for {START_DATE} → {END_DATE}")
    stop_val = params.get("stop_pct", None)
    stop_str = "None" if stop_val is None else f"{stop_val*100:.1f}%"
    print(f"[Step1] Target={params['target_pct']*100:.1f}%, Stop={stop_str}")

    side_counts = defaultdict(int); hit_counts = defaultdict(int)
    total_entries = 0; tickers_logged = 0

    for k, fname in enumerate(files, start=1):
        ticker = fname.replace("_main_indicators.csv",""); path = os.path.join(indicators_dir, fname)
        try:
            df_entries = generate_intraday_entries_for_file(ticker, path, params)
            if not df_entries.empty:
                all_rows.append(df_entries)
                c_by_side = df_entries.groupby("Side").size().to_dict()
                h_by_side = df_entries.groupby(["Side","HitTarget"]).size().to_dict()
                for s, c in c_by_side.items():
                    side_counts[s] += int(c); total_entries += int(c)
                for (s, h), c in h_by_side.items():
                    if h: hit_counts[s] += int(c)
                if tickers_logged < LOG_TICKERS_WITH_ENTRIES_MAX:
                    t0 = df_entries["EntryTime"].min(); t1 = df_entries["EntryTime"].max()
                    bull = int(c_by_side.get("Bullish", 0)); bear = int(c_by_side.get("Bearish", 0))
                    bull_hit = int(h_by_side.get(("Bullish", True), 0)); bear_hit = int(h_by_side.get(("Bearish", True), 0))
                    print(f"  [+] {ticker}: entries={bull+bear} (B:{bull}/{bull_hit} hit, R:{bear}/{bear_hit} hit) "
                          f"time=[{str(t0)[11:16]}→{str(t1)[11:16]}]")
                    tickers_logged += 1
        except Exception as e:
            print(f"  [!] {ticker}: {e}")

        if (k % LOG_PRINT_EVERY_FILES) == 0 or (k == nfiles):
            bull = side_counts.get("Bullish", 0); bear = side_counts.get("Bearish", 0)
            bull_hit = hit_counts.get("Bullish", 0); bear_hit = hit_counts.get("Bearish", 0)
            hit_total = bull_hit + bear_hit
            wr = (hit_total / total_entries * 100.0) if total_entries else 0.0
            print(f"[{k}/{nfiles}] files | entries so far={total_entries} (Bull:{bull}, Bear:{bear}) | hit-rate={wr:.1f}%")

    if not all_rows:
        print("[Step1] No entries found."); return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True).sort_values("EntryTime")
    result.to_csv(out_csv, index=False)

    final_side = result.groupby("Side").size().to_dict()
    final_hit  = result.groupby(["Side","HitTarget"]).size().to_dict()
    bull = int(final_side.get("Bullish", 0)); bear = int(final_side.get("Bearish", 0))
    bull_hit = int(final_hit.get(("Bullish", True), 0)); bear_hit = int(final_hit.get(("Bearish", True), 0))
    wr = ((bull_hit + bear_hit) / len(result) * 100.0) if len(result) else 0.0
    print(f"[Step1] Saved entries → {out_csv} (rows={len(result)})")
    print(f"[Step1] Totals: Bull={bull} (hits {bull_hit}), Bear={bear} (hits {bear_hit}), overall hit-rate={wr:.1f}%")
    return result

# ------- Winners-only ranges -------
def _qband(series: pd.Series, lo: float, hi: float) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return (float("nan"), float("nan"))
    lo_v, hi_v = np.quantile(s, [lo, hi])
    return float(lo_v), float(hi_v)

def fit_indicator_ranges_by_ticker(entries: pd.DataFrame,
                                   out_json: str = "fitted_indicator_ranges_by_ticker.json",
                                   out_csv: str = "fitted_indicator_ranges_by_ticker.csv",
                                   q_low: float = 0.10,
                                   q_high: float = 0.90,
                                   shrink_k: int = 200) -> pd.DataFrame:
    if entries.empty:
        print("[Fit/ByTicker] No entries provided."); return pd.DataFrame()

    def _build_side_cfg(sub: pd.DataFrame, side: str, global_side_cfg: dict,
                        ql: float, qh: float, shrink_k_val: int) -> Tuple[dict, int]:
        n = len(sub); w = n / (n + shrink_k_val) if shrink_k_val > 0 else 1.0
        def _blend(val_t, val_g):
            if (val_t is None) or (isinstance(val_t, float) and np.isnan(val_t)): return val_g
            if (val_g is None) or (isinstance(val_g, float) and np.isnan(val_g)): return val_t
            return float(w * val_t + (1 - w) * val_g)
        side_cfg = {"Samples_wins": int(n), "Quantiles_used": [ql, qh]}
        for name in INDICATOR_SPECS["range_both"]:
            if name in sub.columns:
                lo_t, hi_t = _qband(sub[name], ql, qh)
                g = global_side_cfg.get(f"{name}_range")
                lo_g, hi_g = (g if isinstance(g,(list,tuple)) and len(g)==2 else [None, None])
                side_cfg[f"{name}_range"] = [_blend(lo_t, lo_g), _blend(hi_t, hi_g)]
        for name in INDICATOR_SPECS["min_bull_max_bear"]:
            if name not in sub.columns: continue
            lo_t, hi_t = _qband(sub[name], ql, qh)
            if side == "Bullish":
                base_g = global_side_cfg.get(f"{name}_min"); side_cfg[f"{name}_min"] = _blend(lo_t, base_g)
            else:
                base_g = global_side_cfg.get(f"{name}_max"); side_cfg[f"{name}_max"] = _blend(hi_t, base_g)
        for name in INDICATOR_SPECS["min_both"]:
            if name not in sub.columns: continue
            lo_t, _ = _qband(sub[name], ql, qh)
            base_g = global_side_cfg.get(f"{name}_min"); side_cfg[f"{name}_min"] = _blend(lo_t, base_g)
        return side_cfg, n

    cfg_global = {}
    for side in ["Bullish","Bearish"]:
        gsub = entries[(entries["Side"]==side) & (entries["HitTarget"]==True)].copy()
        cfg_global[side] = _build_side_cfg(gsub, side, {}, q_low, q_high, 0)[0] if len(gsub) else {}

    cfg_all = {}; out_rows = []
    for ticker, df_t in entries.groupby("Ticker"):
        cfg_all[ticker] = {}
        for side in ["Bullish","Bearish"]:
            sub = df_t[(df_t["Side"]==side) & (df_t["HitTarget"]==True)].copy()
            side_cfg, n = _build_side_cfg(sub, side, cfg_global.get(side, {}), q_low, q_high, shrink_k)
            cfg_all[ticker][side] = side_cfg
            row = {"Ticker": ticker, "Side": side, "Samples_wins": n, "Quantiles_used": f"[{q_low:.2f},{q_high:.2f}]"}
            for k, v in side_cfg.items():
                if k in ("Samples_wins","Quantiles_used"): continue
                row[k] = f"[{v[0]:.6g},{v[1]:.6g}]" if (isinstance(v,(list,tuple)) and len(v)==2) else v
            out_rows.append(row)
        if len(out_rows) % 200 == 0:
            print(f"[Fit/ByTicker] {len(out_rows)//2} tickers processed...")

    with open(out_json, "w") as f:
        json.dump(cfg_all, f, indent=2)
    df_out = pd.DataFrame(out_rows); df_out.to_csv(out_csv, index=False)
    print(f"[Fit/ByTicker] Saved per-ticker ranges → {out_json}, {out_csv} (tickers={len(cfg_all)})")
    return df_out

# =============== STEP 2: Modeling (Per-ticker + Global + Stacker) ===============
ASSET_DIR = Path("models"); ASSET_DIR.mkdir(exist_ok=True)
STAMP = pd.to_datetime(END_DATE).strftime("%Y%m%d")
DATA_PATH_DEFAULT   = f"models/entries_2pct_results_{STAMP}.csv"
SAVE_MODELS_DEFAULT = True
MODELS_PATH_DEFAULT = f"models/per_ticker_models_{STAMP}.joblib"
OOF_PATH_DEFAULT    = f"models/oof_predictions_{STAMP}.csv"

FEATURES_DEFAULT = [
    "RSI","ADX","StochD","StochK","VWAP_Dist","VolRel","Daily_Change","Intra_Change",
    "EMA50_Dist","EMA200_Dist","SMA20_Dist","ATR_Pct","BBP","BandWidth",
    "MACD_Hist","CCI","MFI","OBV_Slope10"
]
REGIME_FEATURES = ["Regime_DayRetMean","Regime_DayIntraStd","Regime_BullShare"]

MAX_SPLITS = 5; MIN_SPLITS = 2
MAX_NEIGHBORS = 25; MIN_NEIGHBORS = 3

def _to_float_pct(x):
    if pd.isna(x): return np.nan
    try: return float(str(x).strip().rstrip("%"))
    except Exception: return np.nan

def recompute_hit(df: pd.DataFrame) -> pd.Series:
    rp = df["ReturnPct"].apply(_to_float_pct)
    side = df["Side"].astype(str).str.strip().str.lower()
    hit = ((side=="bullish") & (rp >= 2.0)) | ((side=="bearish") & (rp <= -2.0))
    return hit.astype(int)

def pick_cv_splits(X, y, groups, n_splits: int):
    if _HAS_SKLEARN and HAS_SGK and len(np.unique(y))>=2 and len(np.unique(groups))>=n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cv.split(X, y, groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(X, y, groups)

def safe_k(n_train: int) -> int:
    if n_train <= 0: return MIN_NEIGHBORS
    k = int(np.sqrt(n_train))
    return max(MIN_NEIGHBORS, min(k, MAX_NEIGHBORS, n_train))

@dataclass
class PerTickerModel:
    kind: str                   # "knn" or "constant"
    pipeline: Pipeline = None   # for "knn"
    const_p: float = None       # for "constant"
    features: List[str] = None
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].copy()
        for c in self.features:
            if c not in X.columns: X[c] = np.nan
        X = X[self.features]
        if self.kind == "constant": return np.full((len(X),), float(self.const_p), dtype=float)
        return np.clip(self.pipeline.predict(X), 0.0, 1.0)

def _youden_threshold(y_true: np.ndarray, p_pred: np.ndarray,
                      lo: float=0.10, hi: float=0.90, steps: int=81) -> float:
    if y_true.size==0 or np.unique(y_true).size<2:
        base = float(np.nanmean(p_pred)) if p_pred.size else THRESH_FLOOR
        return float(np.clip(base, THRESH_FLOOR, THRESH_CEIL))
    best_t, best_j = 0.50, -1.0
    for t in np.linspace(lo, hi, steps):
        pred = (p_pred >= t).astype(int)
        tp = int(((pred==1) & (y_true==1)).sum()); fp = int(((pred==1) & (y_true==0)).sum())
        fn = int(((pred==0) & (y_true==1)).sum()); tn = int(((pred==0) & (y_true==0)).sum())
        tpr = tp / (tp+fn) if (tp+fn) else 0.0; fpr = fp / (fp+tn) if (fp+tn) else 0.0
        j = tpr - fpr
        if j > best_j: best_j, best_t = j, t
    return float(np.clip(best_t, THRESH_FLOOR, THRESH_CEIL))

def _ev_threshold(y: np.ndarray, p: np.ndarray,
                  r_hit: float=EV_R_HIT, r_miss: float=EV_R_MISS,
                  steps: int=81, lo: float=0.10, hi: float=0.90) -> float:
    best_t, best_ev = 0.50, -1e9
    for t in np.linspace(lo, hi, steps):
        take = p >= t
        if not np.any(take): continue
        ph = p[take]
        ev = np.mean(ph * r_hit + (1.0 - ph) * r_miss)
        if ev > best_ev: best_ev, best_t = ev, t
    return float(np.clip(best_t, THRESH_FLOOR, THRESH_CEIL))

def _make_regime_features(entries_df: pd.DataFrame) -> pd.DataFrame:
    df = entries_df.copy()
    # Day-level aggregates as simple regime proxies
    day_grp = df.groupby("DateOnly")
    df["Regime_DayRetMean"]   = day_grp["Daily_Change"].transform("mean")
    df["Regime_DayIntraStd"]  = day_grp["Intra_Change"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=3).std())
    # Bullish share among entries for the day (proxy breadth)
    df["SideBin"] = (df["Side"].astype(str).str.lower()=="bullish").astype(int)
    df["Regime_BullShare"] = day_grp["SideBin"].transform("mean")
    return df

def _psi_reference_bins(values: np.ndarray, bins: int=10) -> np.ndarray:
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return np.linspace(0, 1, bins+1)
    return np.quantile(vals, np.linspace(0,1,bins+1))

def run_ai_ml_pipeline(data_path: str = DATA_PATH_DEFAULT,
                       save_models: bool = SAVE_MODELS_DEFAULT,
                       models_path: str = MODELS_PATH_DEFAULT,
                       oof_path: str = OOF_PATH_DEFAULT,
                       features: Optional[List[str]] = None):
    if not _HAS_SKLEARN:
        print("[ML] scikit-learn not available; skipping Step-2."); return None, None
    if not os.path.exists(data_path): raise FileNotFoundError(f"CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty: raise ValueError("CSV is empty.")

    if "EntryTime" in df.columns: df["EntryTime"] = pd.to_datetime(df["EntryTime"], errors="coerce")
    if "DateOnly" not in df.columns:
        df["DateOnly"] = df["EntryTime"].dt.date.astype(str) if "EntryTime" in df.columns else "NA"
    else:
        df["DateOnly"] = df["DateOnly"].astype(str)

    df["HitTarget_bin"] = recompute_hit(df)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Side"] = df["Side"].astype(str).str.strip().str.capitalize()
    df = df.dropna(subset=["Ticker","Side","DateOnly"]).reset_index(drop=True)
    df["RowId"] = np.arange(len(df))

    # Add regime features
    df = _make_regime_features(df)

    # Feature set
    feats_base = features if features is not None else FEATURES_DEFAULT
    feats = [c for c in feats_base if c in df.columns]
    if not feats: raise ValueError("None of the requested FEATURES are present.")
    feats_all = feats + [c for c in REGIME_FEATURES if c in df.columns]

    N = len(df)
    knn_oof = np.full(N, np.nan, dtype=float)
    knn_fold = np.full(N, -1, dtype=int)

    # ----- Per-ticker KNN with OOF -----
    all_oof_rows = []
    models_local: Dict[str, PerTickerModel] = {}
    tickers = sorted(df["Ticker"].unique())
    print(f"Tickers to model (per-ticker KNN): {len(tickers)}")

    for tkr in tickers:
        dft = df[df["Ticker"] == tkr].reset_index(drop=True)
        y = dft["HitTarget_bin"].to_numpy(dtype=int)
        groups = dft["DateOnly"].to_numpy()
        Xt = dft[feats].copy()
        for c in feats:
            if c not in Xt.columns: Xt[c] = np.nan
        Xt = Xt[feats].to_numpy(dtype=float)
        n = len(dft); uniq_dates = np.unique(groups).size
        n_splits = max(MIN_SPLITS, min(MAX_SPLITS, uniq_dates))

        if np.unique(y).size < 2 or n < 10 or uniq_dates < 2:
            p_const = float((y.mean() * n + 0.5) / (n + 1.0))
            models_local[tkr] = PerTickerModel(kind="constant", const_p=p_const, features=feats)
            oof_pred = np.full(n, p_const, dtype=float); fold_ids = np.full(n, -1, dtype=int)
            print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  -> CONSTANT p={p_const:.3f}")
        else:
            oof_pred = np.zeros(n, dtype=float); fold_ids = np.full(n, -1, dtype=int)
            splits = list(pick_cv_splits(Xt, y, groups, n_splits=n_splits))
            for fold, (tr, te) in enumerate(splits, start=1):
                Xtr, Xte = Xt[tr], Xt[te]; ytr = y[tr]
                k = safe_k(len(tr))
                pipe = Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler(with_mean=True, with_std=True)),
                    ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance", p=2)),
                ])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xtr, ytr.astype(float))
                oof_pred[te] = np.clip(pipe.predict(Xte), 0.0, 1.0); fold_ids[te] = fold
            k_full = safe_k(n)
            final_pipe = Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True, with_std=True)),
                ("knn", KNeighborsRegressor(n_neighbors=k_full, weights="distance", p=2)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_pipe.fit(Xt, y.astype(float))
            models_local[tkr] = PerTickerModel(kind="knn", pipeline=final_pipe, features=feats)

            try:
                auc = roc_auc_score(y, oof_pred) if len(np.unique(y)) == 2 and len(np.unique(oof_pred)) > 1 else np.nan
            except Exception:
                auc = np.nan
            try:
                brier = brier_score_loss(y, oof_pred)
            except Exception:
                brier = np.nan
            topk = max(1, int(0.2 * n))
            idx_top = np.argsort(-oof_pred)[:topk]
            hr_top = y[idx_top].mean() if topk > 0 else np.nan
            print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d} | AUC={auc:.3f}  Brier={brier:.3f}  Top20%HR={hr_top:.3f}")

        # stitch OOF to global arrays
        rows = dft["RowId"].to_numpy()
        knn_oof[rows] = oof_pred
        knn_fold[rows] = fold_ids

        # keep rows for OOF CSV
        all_oof_rows.append(pd.DataFrame({
            "RowId": rows, "Ticker": tkr,
            "EntryTime": dft.get("EntryTime", pd.NaT),
            "Side": dft["Side"], "HitTarget": y,
            "P_local": oof_pred, "FoldLocal": fold_ids
        }))

    # ----- Global HGB with OOF across dates -----
    X_all = df[feats_all].to_numpy()  # HGB can handle NaNs
    y_all = df["HitTarget_bin"].to_numpy(dtype=int)
    groups_all = df["DateOnly"].to_numpy()

    if HAS_SGK and len(np.unique(y_all)) >= 2 and len(np.unique(groups_all)) >= 3:
        splitter = StratifiedGroupKFold(n_splits=min(MAX_SPLITS, max(MIN_SPLITS, len(np.unique(groups_all)))), shuffle=True, random_state=42)
        splits = list(splitter.split(X_all, y_all, groups_all))
    else:
        splitter = GroupKFold(n_splits=min(MAX_SPLITS, max(MIN_SPLITS, len(np.unique(groups_all)))))
        splits = list(splitter.split(X_all, y_all, groups_all))

    global_oof = np.zeros(N, dtype=float)
    global_fold = np.full(N, -1, dtype=int)
    for fold, (tr, te) in enumerate(splits, start=1):
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, l2_regularization=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_all[tr], y_all[tr])
        proba = model.predict_proba(X_all[te])[:,1]
        global_oof[te] = proba
        global_fold[te] = fold

    # Fit final global model on full data
    global_model = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, l2_regularization=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_model.fit(X_all, y_all)

    # ----- Stacker (LogReg) trained on OOF predictions -----
    S = np.vstack([knn_oof, global_oof]).T
    stacker = LogisticRegression(max_iter=500, C=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacker.fit(S, y_all)
    p_stack_oof = stacker.predict_proba(S)[:,1]

    # ----- Calibrators (per ticker) + thresholds (Youden + EV) -----
    calibrators: Dict[str, IsotonicRegression] = {}
    thresholds: Dict[str, Dict[str, float]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    # Add calibrated stacked prediction for OOF CSV
    p_stack_oof_cal = np.copy(p_stack_oof)

    for tkr, g in df.groupby("Ticker"):
        idx = g.index.to_numpy()
        y = y_all[idx]
        p = p_stack_oof[idx]

        # Isotonic per ticker if enough data
        if len(idx) >= 20 and len(np.unique(y)) >= 2:
            try:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(p, y)
                calibrators[tkr] = ir
                p_cal = ir.predict(p)
            except Exception:
                p_cal = p
        else:
            p_cal = p

        p_stack_oof_cal[idx] = p_cal

        # Thresholds
        t_y = _youden_threshold(y, p_cal, lo=0.10, hi=0.90, steps=81)
        t_ev = _ev_threshold(y, p_cal, r_hit=EV_R_HIT, r_miss=EV_R_MISS, steps=81, lo=0.10, hi=0.90)
        thresholds[tkr] = {
            "youden": float(t_y),
            "ev": float(t_ev),
            "chosen": float(np.clip(t_ev, THRESH_FLOOR, THRESH_CEIL))
        }

        # Quick metrics on calibrated OOF
        try:
            auc = roc_auc_score(y, p_cal) if len(np.unique(y))==2 else np.nan
        except Exception:
            auc = np.nan
        try:
            brier = brier_score_loss(y, p_cal)
        except Exception:
            brier = np.nan
        topk = max(1, int(0.2 * len(idx)))
        hr_top = float(y[np.argsort(-p_cal)[:topk]].mean()) if topk>0 else np.nan
        metrics[tkr] = {"auc": float(auc) if auc==auc else np.nan,
                        "brier": float(brier) if brier==brier else np.nan,
                        "top20_hr": float(hr_top) if hr_top==hr_top else np.nan}

    # ----- Build OOF CSV -----
    oof_df_base = pd.concat(all_oof_rows, ignore_index=True).drop(columns=["HitTarget"])  # local info
    oof_df_full = df[["RowId","Ticker","EntryTime","Side","HitTarget_bin"]].rename(columns={"HitTarget_bin":"HitTarget"}).copy()
    oof_df_full["P_local"] = knn_oof
    oof_df_full["FoldLocal"] = knn_fold
    oof_df_full["P_global"] = global_oof
    oof_df_full["FoldGlobal"] = global_fold
    oof_df_full["P_stack"] = p_stack_oof
    oof_df_full["P_stack_cal"] = p_stack_oof_cal

    # Save OOF
    oof_df_full.sort_values(["Ticker","EntryTime","RowId"]).to_csv(oof_path, index=False)
    print(f"Saved OOF predictions → {oof_path} (rows={len(oof_df_full)})")

    # ----- Prepare live-time components -----
    # 1) Precompute per-row full predictions for stacker inputs (fitting stacker used OOF; for live we need producers)
    # local full preds:
    p_local_full = np.zeros(N, dtype=float)
    for tkr, model in models_local.items():
        idx = df.index[df["Ticker"]==tkr].to_numpy()
        if len(idx)==0: continue
        Xsub = df.loc[idx, model.features]
        p_local_full[idx] = model.predict_proba(Xsub)

    # global full preds:
    p_global_full = global_model.predict_proba(X_all)[:,1]

    # Fit stacker on producers' FULL preds (so it can be used live)
    S_full = np.vstack([p_local_full, p_global_full]).T
    stacker_live = LogisticRegression(max_iter=500, C=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacker_live.fit(S_full, y_all)

    # ----- PSI reference bins for drift monitoring -----
    psi_ref_bins = {}
    for ft in KEY_DRIFT_FEATURES:
        if ft in df.columns:
            psi_ref_bins[ft] = _psi_reference_bins(pd.to_numeric(df[ft], errors="coerce").to_numpy(), bins=10).tolist()

    payload = {
        "local_models": models_local,
        "global_model": global_model,
        "stacker": stacker_live,                 # meta on FULL preds (for live)
        "stacker_training_note": "Trained on full producers' preds; calibration/thresholds from OOF.",
        "features_base": feats,
        "features_regime": [f for f in REGIME_FEATURES if f in df.columns],
        "calibrators": calibrators,              # per-ticker isotonic on OOF stacker preds
        "thresholds": thresholds,                # per-ticker {'youden','ev','chosen'}
        "metrics_oof": metrics,                  # quick diagnostics per ticker
        "psi_ref_bins": psi_ref_bins,            # for live PSI
        "config": {
            "ev_r_hit": EV_R_HIT, "ev_r_miss": EV_R_MISS,
            "psi_cutoff": PSI_CUTOFF, "psi_bump_pp": PSI_BUMP_PP,
            "thresholds_floor": THRESH_FLOOR, "thresholds_ceil": THRESH_CEIL,
            "live_defensive_bump": LIVE_DEFENSIVE_BUMP
        },
        "stamp": STAMP, "start_date": START_DATE, "end_date": END_DATE
    }

    if save_models:
        joblib.dump(payload, models_path)
        print(f"Saved per-ticker models bundle → {models_path}")

    return payload, oof_df_full

# =============== MAIN ===============
if __name__ == "__main__":
    # Validate dates
    try:
        _ = pd.to_datetime(START_DATE); _ = pd.to_datetime(END_DATE)
    except Exception as _e:
        raise ValueError(f"Invalid START_DATE/END_DATE: {START_DATE}, {END_DATE}") from _e

    ASSET_DIR = Path("models"); ASSET_DIR.mkdir(exist_ok=True)
    entries_csv_path = f"models/entries_2pct_results_{STAMP}.csv"

    # ---- Step 1: backtest & entries CSV ----
    entries = run_step1_scan(INDICATORS_DIR, ENTRY_PARAMS_DEFAULT, out_csv=entries_csv_path)

    if not entries.empty:
        # Baseline and extra bands
        def _emit_ranges(lo, hi, tag=""):
            jpath = f"models/fitted_indicator_ranges_by_ticker{tag}_{STAMP}.json" if tag else f"models/fitted_indicator_ranges_by_ticker_{STAMP}.json"
            cpath = f"models/fitted_indicator_ranges_by_ticker{tag}_{STAMP}.csv"  if tag else f"models/fitted_indicator_ranges_by_ticker_{STAMP}.csv"
            fit_indicator_ranges_by_ticker(entries, out_json=jpath, out_csv=cpath, q_low=lo, q_high=hi, shrink_k=200)
            return jpath, cpath

        main_ranges_json, main_ranges_csv = _emit_ranges(0.10, 0.90, "")
        _emit_ranges(0.10, 0.90, "10_90")
        _emit_ranges(0.15, 0.85, "15_85")
        _emit_ranges(0.20, 0.80, "20_80")
        _emit_ranges(0.25, 0.75, "25_75")

        # ---- Step 2: ML & thresholds ----
        models_path = f"models/per_ticker_models_{STAMP}.joblib"
        oof_path    = f"models/oof_predictions_{STAMP}.csv"
        payload, oof_df = run_ai_ml_pipeline(
            data_path=entries_csv_path,
            save_models=True,
            models_path=models_path,
            oof_path=oof_path,
            features=FEATURES_DEFAULT,
        )

        # ---- Manifest ----
        end_dt = pd.to_datetime(END_DATE).date()
        valid_from = end_dt
        valid_thru = (end_dt + timedelta(days=GRACE_DAYS))
        manifest = {
            "stamp": STAMP,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "valid_from": str(valid_from),
            "valid_thru": str(valid_thru),
            "grace_days": GRACE_DAYS,
            "live_defensive_bump": LIVE_DEFENSIVE_BUMP,
            "thresholds_floor": THRESH_FLOOR,
            "thresholds_ceil": THRESH_CEIL,
            "psi_cutoff": PSI_CUTOFF,
            "psi_bump_pp": PSI_BUMP_PP,
            # Artifacts
            "entries_csv": entries_csv_path,
            "ranges_csv": main_ranges_csv,
            "ranges_json": main_ranges_json,
            "models_path": models_path,
            "oof_path": oof_path,
            "features_base": FEATURES_DEFAULT,
            "features_regime": [f for f in REGIME_FEATURES if f in entries.columns],
            "quantiles": [0.10, 0.90],
            "calibration": "isotonic",
            # Convenience: thresholds copied for quick access
            "per_ticker_thresholds": (payload or {}).get("thresholds", {}),
        }
        man_path = f"models/manifest_{STAMP}.json"
        Path(man_path).write_text(json.dumps(manifest, indent=2))
        print(f"[Main] Wrote manifest → {man_path}")
    else:
        print("[Main] No entries generated in Step-1; skipping Step-2/ML and ranges fit.")
