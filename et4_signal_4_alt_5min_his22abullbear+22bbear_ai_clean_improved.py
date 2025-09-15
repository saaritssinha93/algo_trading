# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:04:53 2025

Author: Saarit

Fast + Parity (CLEAN) detector for ~1,038 tickers.

This version runs in CLEAN-parity mode by default to reproduce clean.py counts
while keeping the faster I/O and vectorized detector.

Switch mode by editing COMPAT_MODE_DEFAULT below ("clean" | "fast").
"""

import os
import sys
import glob
import json
import signal as _signal
import pytz
import logging
import warnings
import traceback
import threading
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
from tqdm import tqdm
from filelock import FileLock, Timeout
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
from functools import lru_cache

# ---------------------------------------------
# 0) Mode toggle (default: "clean" for parity with clean.py)
# ---------------------------------------------
COMPAT_MODE_DEFAULT = "clean"   # "clean" => identical counts to clean.py ; "fast" => speed

# ---------------------------------------------
# 1) Configuration & setup
# ---------------------------------------------
INDICATORS_DIR = "main_indicators_july_5min2"
SIGNALS_DB     = "generated_signals_historical5minv4a.json"
ENTRIES_DIR    = "main_indicators_history_entries_5minv4a"
LOG_DIR        = "logs"

# ML assets
OOF_PATH       = "oof_predictions.csv"
MODELS_PATH    = "per_ticker_models.joblib"
USE_ML_FILTER  = True
GLOBAL_MIN_PROB = 0.50
GLOBAL_MAX_PROB = 0.90
ML_SOFT_MARGIN_BEAR = 0.03  # bear soft admit

# Fitted constraints
FITTED_CSV_MAIN     = "fitted_indicator_ranges_by_ticker.csv"          # 22a
FITTED_CSV_BEAR_ALT = "fitted_indicator_ranges_by_ticker20_80.csv"     # 22b (bear union)
DETECTOR_BEAR_ALT_VOLREL_MIN = 0.95  # relaxed participation for 22b-bear

# Ensure dirs
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(ENTRIES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

india_tz = pytz.timezone("Asia/Kolkata")

# Logging (keep light for speed)
logger = logging.getLogger("fast_detector")
logger.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "signalnewn5minv4a.log"),
    when="M", interval=30, backupCount=5, delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.ERROR)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

print(f"Script start (vectorized, COMPAT_MODE={COMPAT_MODE_DEFAULT.upper()})...")

# Optional: set a working directory
try:
    cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

# ---------------------------------------------
# 1b) ML model wrapper & assets
# ---------------------------------------------
@dataclass
class PerTickerModel:
    kind: str                   # "knn" or "constant"
    pipeline: Any = None        # sklearn Pipeline for "knn"
    const_p: float = None       # for "constant"
    features: List[str] = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        if self.features is None:
            return np.full((len(X),), float(self.const_p or 0.0), dtype=float)
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.features]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.pipeline.predict(X) if self.kind != "constant" else np.full((len(X),), float(self.const_p or 0.0))
        return np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)

ML_MODELS: Dict[str, PerTickerModel] = {}
ML_FEATURES: List[str] = []
ML_THRESHOLDS: Dict[str, float] = {}

def _best_threshold_from_oof(df_tkr: pd.DataFrame) -> float:
    if df_tkr.empty:
        return GLOBAL_MIN_PROB
    y = pd.to_numeric(df_tkr["HitTarget"], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(df_tkr["PredProb"], errors="coerce").fillna(0.0).to_numpy()
    if np.unique(y).size < 2:
        thr = float(np.clip(np.nanmean(p) if len(p) else GLOBAL_MIN_PROB,
                            GLOBAL_MIN_PROB, GLOBAL_MAX_PROB))
        return thr
    qs = np.unique(np.quantile(p, np.linspace(0.10, 0.90, 17)))
    grid = np.unique(np.concatenate([qs, np.linspace(0.3, 0.8, 11)]))
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        pred = (p >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_t = float(np.clip(best_t, GLOBAL_MIN_PROB, GLOBAL_MAX_PROB))
    return best_t

def load_ml_assets():
    global ML_MODELS, ML_FEATURES, ML_THRESHOLDS
    if not USE_ML_FILTER:
        print("[ML] ML filter disabled by config.")
        return
    # Models
    if os.path.exists(MODELS_PATH):
        try:
            payload = joblib.load(MODELS_PATH)
            ML_MODELS = payload.get("models", {})
            ML_FEATURES = payload.get("features", [])
            print(f"[ML] Loaded models for {len(ML_MODELS)} tickers. Features: {len(ML_FEATURES)}")
        except Exception as e:
            print(f"[ML] Could not load models: {e}")
            ML_MODELS = {}
            ML_FEATURES = []
    else:
        print(f"[ML] Models file not found: {MODELS_PATH}")

    # OOF thresholds
    if os.path.exists(OOF_PATH):
        try:
            oof = pd.read_csv(OOF_PATH)
            req = {"Ticker","PredProb","HitTarget"}
            if req.issubset(oof.columns):
                oof["TickerKey"] = oof["Ticker"].astype(str).str.strip().str.upper()
                for tkr, g in oof.groupby("TickerKey"):
                    ML_THRESHOLDS[tkr] = _best_threshold_from_oof(g)
                if ML_THRESHOLDS:
                    mn, mx = min(ML_THRESHOLDS.values()), max(ML_THRESHOLDS.values())
                    print(f"[ML] Derived thresholds for {len(ML_THRESHOLDS)} tickers (range: {mn:.2f}-{mx:.2f})")
            else:
                print("[ML] OOF missing required columns; skipping thresholds.")
        except Exception as e:
            print(f"[ML] Could not read OOF: {e}")
    else:
        print(f"[ML] OOF file not found: {OOF_PATH}")

load_ml_assets()

# ---------------------------------------------
# 2) Signal DB utilities
# ---------------------------------------------
def load_generated_signals() -> set:
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, "r") as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logger.error("Signals DB JSON is corrupted. Starting with empty set.")
    return set()

def save_generated_signals(generated_signals: set) -> None:
    with open(SIGNALS_DB, "w") as f:
        json.dump(list(generated_signals), f, indent=2)

# ---------------------------------------------
# 3) Fitted constraints preload & caching
# ---------------------------------------------
COLMAP = {
    "RSI":"RSI","ADX":"ADX","CCI":"CCI","MFI":"MFI","BBP":"BBP","BandWidth":"BandWidth","ATR_Pct":"ATR_Pct",
    "StochD":"StochD","StochK":"StochK","MACD_Hist":"MACD_Hist","VWAP_Dist":"VWAP_Dist",
    "Daily_Change":"Daily_Change","EMA50_Dist":"EMA50_Dist","EMA200_Dist":"EMA200_Dist",
    "SMA20_Dist":"SMA20_Dist","OBV_Slope10":"OBV_Slope10","VolRel":"VolRel",
}

def _parse_range_cell_fast(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return (np.nan, np.nan)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return float(x[0]), float(x[1])
        except Exception:
            return (np.nan, np.nan)
    s = str(x).strip()
    if not s:
        return (np.nan, np.nan)
    try:
        lo, hi = eval(s, {}, {})  # CSV we control; expect tuples/lists
        return float(lo), float(hi)
    except Exception:
        s = s.strip("[]")
        parts = [p for p in s.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except Exception:
                pass
        return (np.nan, np.nan)

def _to_float_fast(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return np.nan
    try: return float(x)
    except Exception:
        try: return float(str(x).strip())
        except Exception: return np.nan

def _read_fitted_csv_to_dict(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Ticker" not in df.columns or "Side" not in df.columns:
        return {}
    df["TickerKey"] = df["Ticker"].astype(str).str.strip().str.upper()
    out = {}
    for (tkey, side), g in df.groupby(["TickerKey","Side"]):
        merged = {}
        for _, row in g.iterrows():
            for col, val in row.items():
                if col in ("Ticker","TickerKey","Side","Samples_wins","Quantiles_used"):
                    continue
                if col.endswith("_range"):
                    rng = _parse_range_cell_fast(val)
                    if not (np.isnan(rng[0]) or np.isnan(rng[1])):
                        merged[col] = rng
                elif col.endswith("_min") or col.endswith("_max"):
                    fv = _to_float_fast(val)
                    if not np.isnan(fv):
                        merged[col] = fv
        if merged:
            out.setdefault(tkey, {})[side] = merged
    return out

@lru_cache(maxsize=1)
def get_fitted_constraints_main():
    return _read_fitted_csv_to_dict(FITTED_CSV_MAIN)

@lru_cache(maxsize=1)
def get_fitted_constraints_22b():
    return _read_fitted_csv_to_dict(FITTED_CSV_BEAR_ALT)

# ---------------------------------------------
# 4) Fast CSV loader (warm-start)
# ---------------------------------------------
NEEDED_COLS = [
    "date","open","high","low","close","volume",
    "RSI","ADX","CCI","MFI","MACD_Hist","Stoch_%K","Stoch_%D",
    "Upper_Band","Lower_Band","ATR","20_SMA","EMA_50","EMA_200","OBV",
    "Daily_Change","Signal_ID","Entry Signal","CalcFlag","logtime"
]

def fast_load_csv_for_day(
    file_path: str,
    day_start,
    day_end,
    volrel_window: int = 10,
    volrel_min_periods: int = 10,
    warmup_days: int = 1,
) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=NEEDED_COLS)

    target_label = day_start.strftime("%Y-%m-%d")
    prev_label   = (day_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    keep_labels  = {target_label, prev_label}

    df = pd.read_csv(
        file_path,
        usecols=lambda c: (c in NEEDED_COLS),
        dtype={"date": "string"},
        engine="c",
        memory_map=True,
        low_memory=False,
        on_bad_lines="skip",
    )
    if "date" not in df.columns or df.empty:
        return pd.DataFrame(columns=NEEDED_COLS)

    pref = df["date"].str.slice(0, 10)
    m_pref = pref.isin(keep_labels)
    df2 = df.loc[m_pref].copy()

    if df2.empty:
        df["__dt"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df.dropna(subset=["__dt"], inplace=True)
        if df["__dt"].dt.tz is None:
            df["__dt"] = df["__dt"].dt.tz_localize(india_tz)
        else:
            df["__dt"] = df["__dt"].dt.tz_convert(india_tz)
        dts = df["__dt"].dt.date
        day_set = {day_start.date(), (day_start - timedelta(days=warmup_days)).date()}
        df2 = df.loc[dts.isin(day_set)].copy()
        df2.rename(columns={"__dt": "date"}, inplace=True)
    else:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce", utc=False)
        df2.dropna(subset=["date"], inplace=True)
        if df2["date"].dt.tz is None:
            df2["date"] = df2["date"].dt.tz_localize(india_tz)
        else:
            df2["date"] = df2["date"].dt.tz_convert(india_tz)

    if df2.empty:
        return df2

    df2.sort_values("date", inplace=True)
    df2.reset_index(drop=True, inplace=True)

    for c in ["open","high","low","close","volume","ATR","20_SMA","EMA_50","EMA_200",
              "Upper_Band","Lower_Band","RSI","ADX","CCI","MFI","MACD_Hist","Stoch_%K","Stoch_%D","OBV"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    sess = df2["date"].dt.date
    tp = (df2.get("high", np.nan) + df2.get("low", np.nan) + df2.get("close", np.nan)) / 3.0
    tp_vol = tp * df2.get("volume", np.nan)
    df2["_cum_tp_vol"] = tp_vol.groupby(sess).cumsum()
    df2["_cum_vol"]    = df2["volume"].groupby(sess).cumsum()
    df2["VWAP"]        = df2["_cum_tp_vol"] / df2["_cum_vol"]
    df2.drop(columns=["_cum_tp_vol","_cum_vol"], inplace=True)

    eps = 1e-9
    if "Upper_Band" in df2.columns and "Lower_Band" in df2.columns:
        bbw = (df2["Upper_Band"] - df2["Lower_Band"])
        df2["BandWidth"] = bbw / (df2["close"] + eps)
        df2["BBP"] = (df2["close"] - df2["Lower_Band"]) / (bbw + eps)

    if "OBV" in df2.columns:
        obv_prev = df2["OBV"].shift(10)
        df2["OBV_Slope10"] = (df2["OBV"] - obv_prev) / (obv_prev.abs() + eps)

    df2["rolling_vol_10"] = df2["volume"].rolling(volrel_window, min_periods=volrel_min_periods).mean()
    df2["VolRel"] = df2["volume"] / (df2["rolling_vol_10"] + eps)

    df2["StochD"] = df2.get("Stoch_%D", np.nan)
    df2["StochK"] = df2.get("Stoch_%K", np.nan)
    df2["VWAP_Dist"]   = df2["close"] / df2["VWAP"] - 1.0
    df2["EMA50_Dist"]  = (df2["close"]/df2["EMA_50"]  - 1.0) if "EMA_50"  in df2.columns else np.nan
    df2["EMA200_Dist"] = (df2["close"]/df2["EMA_200"] - 1.0) if "EMA_200" in df2.columns else np.nan
    df2["SMA20_Dist"]  = (df2["close"]/df2["20_SMA"]  - 1.0) if "20_SMA"  in df2.columns else np.nan

    is_today = df2["date"].dt.date == day_start.date()
    df_today = df2.loc[is_today].copy()
    if df_today.empty:
        return df_today

    t = df_today["date"].dt.time
    intraday = (t >= dt_time(9, 25)) & (t <= dt_time(15, 10))
    df_today = df_today.loc[intraday].copy()
    df_today.sort_values("date", inplace=True)
    df_today.reset_index(drop=True, inplace=True)
    return df_today

# ---------------------------------------------
# 5) Vectorized signal detector (with compat_mode)
# ---------------------------------------------
def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set,
    fitted_csv_path: str = FITTED_CSV_MAIN,
    volrel_window: int = 10,
    volrel_min_periods: int = 10,
    debug: bool = False,
    volrel_gate_min: float = 1.10,
    use_vwap_side_gate: bool = False,
    vwap_long_min: float = 0.001,
    vwap_short_max: float = -0.001,
    use_ml_filter: bool = USE_ML_FILTER,
    ml_models: Dict[str, PerTickerModel] = ML_MODELS,
    ml_features: List[str] = ML_FEATURES,
    ml_thresholds: Dict[str, float] = ML_THRESHOLDS,
    compat_mode: str = COMPAT_MODE_DEFAULT,   # <-- default to global
) -> list:
    """
    Vectorized detector with a 'clean' compatibility mode.
    - fast: operate on today's slice directly (speed).
    - clean: reproduce original semantics by computing features on
             combined(history + today) and aligning to the last bar <= ts
             in the same day (merge_asof), matching clean.py counts.
    """
    signals = []
    if df_for_detection.empty:
        return signals

    eps = 1e-9
    tkey = str(ticker).strip().upper()

    def _numify(df):
        num_cols = [
            "open","high","low","close","volume","ATR","20_SMA","EMA_50","EMA_200",
            "Upper_Band","Lower_Band","RSI","ADX","CCI","MFI","MACD_Hist","Stoch_%K","Stoch_%D","OBV"
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    if compat_mode == "clean":
        combined = pd.concat([df_for_rolling, df_for_detection], ignore_index=True)
        if "date" not in combined.columns:
            return signals
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined.dropna(subset=["date"], inplace=True)
        if combined["date"].dt.tz is None:
            combined["date"] = combined["date"].dt.tz_localize(india_tz)
        else:
            combined["date"] = combined["date"].dt.tz_convert(india_tz)
        combined.sort_values("date", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined["__day"] = combined["date"].dt.date
        combined = _numify(combined)

        tp = (combined.get("high", np.nan) + combined.get("low", np.nan) + combined.get("close", np.nan)) / 3.0
        tp_vol = tp * combined.get("volume", np.nan)
        combined["_cum_tp_vol"] = tp_vol.groupby(combined["__day"]).cumsum()
        combined["_cum_vol"]    = combined["volume"].groupby(combined["__day"]).cumsum()
        combined["VWAP"]        = combined["_cum_tp_vol"] / combined["_cum_vol"]
        combined.drop(columns=["_cum_tp_vol","_cum_vol"], inplace=True)

        if "Upper_Band" in combined.columns and "Lower_Band" in combined.columns:
            bbw = (combined["Upper_Band"] - combined["Lower_Band"])
            combined["BandWidth"] = bbw / (combined["close"] + eps)
            combined["BBP"] = (combined["close"] - combined["Lower_Band"]) / (bbw + eps)

        if "OBV" in combined.columns:
            obv_prev = combined["OBV"].shift(10)
            combined["OBV_Slope10"] = (combined["OBV"] - obv_prev) / (obv_prev.abs() + eps)

        combined["rolling_vol_10"] = combined["volume"].rolling(volrel_window, min_periods=volrel_min_periods).mean()
        combined["VolRel"]         = combined["volume"] / (combined["rolling_vol_10"] + eps)

        combined["StochD"] = combined.get("Stoch_%D", np.nan)
        combined["StochK"] = combined.get("Stoch_%K", np.nan)
        combined["VWAP_Dist"]   = combined["close"]/combined["VWAP"] - 1.0
        combined["EMA50_Dist"]  = (combined["close"]/combined["EMA_50"]  - 1.0) if "EMA_50"  in combined.columns else np.nan
        combined["EMA200_Dist"] = (combined["close"]/combined["EMA_200"] - 1.0) if "EMA_200" in combined.columns else np.nan
        combined["SMA20_Dist"]  = (combined["close"]/combined["20_SMA"]  - 1.0) if "20_SMA"  in combined.columns else np.nan

        d = df_for_detection.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d.dropna(subset=["date"], inplace=True)
        if d["date"].dt.tz is None:
            d["date"] = d["date"].dt.tz_localize(india_tz)
        else:
            d["date"] = d["date"].dt.tz_convert(india_tz)
        d.sort_values("date", inplace=True)
        d.reset_index(drop=True, inplace=True)
        d["__day"] = d["date"].dt.date

        feat_cols = [
            "date","__day","close","VWAP","VolRel","VWAP_Dist","StochD","StochK",
            "EMA50_Dist","EMA200_Dist","SMA20_Dist","BBP","BandWidth","OBV_Slope10",
            "RSI","ADX","CCI","MFI","MACD_Hist","ATR","20_SMA","EMA_50","EMA_200",
            "Upper_Band","Lower_Band","volume","rolling_vol_10"
        ]
        feat_cols = [c for c in feat_cols if c in combined.columns]
        cf = combined[feat_cols].sort_values("date").copy()

        d_aligned = pd.merge_asof(
            d[["date","__day"]].sort_values("date"),
            cf.sort_values("date"),
            on="date",
            by="__day",
            direction="backward",
            allow_exact_matches=True
        ).set_index(d.index)

        r = d_aligned.copy()
        d_work = d.copy()
        for col in r.columns:
            if col not in d_work.columns:
                d_work[col] = r[col]
            else:
                m = d_work[col].isna()
                d_work.loc[m, col] = r.loc[m, col]

    else:
        d_work = df_for_detection.copy()
        d_work["date"] = pd.to_datetime(d_work["date"], errors="coerce")
        d_work.dropna(subset=["date"], inplace=True)
        if d_work["date"].dt.tz is None:
            d_work["date"] = d_work["date"].dt.tz_localize(india_tz)
        else:
            d_work["date"] = d_work["date"].dt.tz_convert(india_tz)
        d_work.sort_values("date", inplace=True)
        d_work.reset_index(drop=True, inplace=True)

        for c in ["open","high","low","close","volume","ATR","20_SMA","EMA_50","EMA_200",
                  "Upper_Band","Lower_Band","RSI","ADX","CCI","MFI","MACD_Hist","Stoch_%K","Stoch_%D","OBV"]:
            if c in d_work.columns:
                d_work[c] = pd.to_numeric(d_work[c], errors="coerce")

        if "VWAP" not in d_work.columns or d_work["VWAP"].isna().all():
            tp = (d_work.get("high", np.nan) + d_work.get("low", np.nan) + d_work.get("close", np.nan)) / 3.0
            tp_vol = tp * d_work.get("volume", np.nan)
            d_work["_cum_tp_vol"] = tp_vol.cumsum()
            d_work["_cum_vol"] = d_work["volume"].cumsum()
            d_work["VWAP"] = d_work["_cum_tp_vol"] / d_work["_cum_vol"]
            d_work.drop(columns=["_cum_tp_vol","_cum_vol"], inplace=True)

        if "rolling_vol_10" not in d_work.columns or d_work["rolling_vol_10"].isna().all():
            d_work["rolling_vol_10"] = d_work["volume"].rolling(volrel_window, min_periods=volrel_min_periods).mean()

        if "VolRel" not in d_work.columns or d_work["VolRel"].isna().all():
            d_work["VolRel"] = d_work["volume"] / (d_work["rolling_vol_10"] + eps)

        if (("BandWidth" not in d_work.columns) or d_work["BandWidth"].isna().all() or
            ("BBP" not in d_work.columns) or d_work["BBP"].isna().all()):
            if "Upper_Band" in d_work.columns and "Lower_Band" in d_work.columns:
                bbw = (d_work["Upper_Band"] - d_work["Lower_Band"])
                d_work["BandWidth"] = bbw / (d_work["close"] + eps)
                d_work["BBP"] = (d_work["close"] - d_work["Lower_Band"]) / (bbw + eps)

        if "OBV_Slope10" not in d_work.columns or d_work["OBV_Slope10"].isna().all():
            if "OBV" in d_work.columns:
                obv_prev = d_work["OBV"].shift(10)
                d_work["OBV_Slope10"] = (d_work["OBV"] - obv_prev) / (obv_prev.abs() + eps)

        d_work["StochD"] = d_work.get("Stoch_%D", d_work.get("StochD", np.nan))
        d_work["StochK"] = d_work.get("Stoch_%K", d_work.get("StochK", np.nan))
        d_work["VWAP_Dist"]   = d_work["close"]/d_work["VWAP"] - 1.0
        d_work["EMA50_Dist"]  = (d_work["close"]/d_work["EMA_50"]  - 1.0) if "EMA_50"  in d_work.columns else np.nan
        d_work["EMA200_Dist"] = (d_work["close"]/d_work["EMA_200"] - 1.0) if "EMA_200" in d_work.columns else np.nan
        d_work["SMA20_Dist"]  = (d_work["close"]/d_work["20_SMA"]  - 1.0) if "20_SMA"  in d_work.columns else np.nan

    available_cols = set(d_work.columns)

    def _to_float_fast(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
        try: return float(x)
        except Exception:
            try: return float(str(x).strip())
            except Exception: return np.nan

    def _clean_constraints_for_df(raw: dict) -> dict:
        if not raw: return {}
        cleaned = {}
        for k, v in raw.items():
            if k.endswith("_range"):
                base = k[:-6]; col = COLMAP.get(base, base)
                if col in available_cols:
                    lo, hi = v if isinstance(v, (tuple, list)) else (np.nan, np.nan)
                    if not (np.isnan(lo) or np.isnan(hi)):
                        cleaned[(col, "range")] = (float(lo), float(hi))
            elif k.endswith("_min"):
                base = k[:-4]; col = COLMAP.get(base, base)
                if col in available_cols:
                    fv = _to_float_fast(v)
                    if not np.isnan(fv):
                        cleaned[(col, "min")] = float(fv)
            elif k.endswith("_max"):
                base = k[:-4]; col = COLMAP.get(base, base)
                if col in available_cols:
                    fv = _to_float_fast(v)
                    if not np.isnan(fv):
                        cleaned[(col, "max")] = float(fv)
        return cleaned

    main_constraints = get_fitted_constraints_main().get(tkey, {})
    alt_constraints  = get_fitted_constraints_22b().get(tkey, {})

    bull_cfg = _clean_constraints_for_df(main_constraints.get("Bullish", {}))
    bear_cfg = _clean_constraints_for_df(main_constraints.get("Bearish", {}))
    bear_alt = _clean_constraints_for_df(alt_constraints.get("Bearish", {}))

    allow_bull = pd.Series(True, index=d_work.index)
    allow_bear = pd.Series(True, index=d_work.index)
    if use_vwap_side_gate:
        vdist = d_work["VWAP_Dist"]
        allow_bull &= vdist >= vwap_long_min
        allow_bear &= vdist <= vwap_short_max

    volrel_ok_global = d_work["VolRel"] >= volrel_gate_min
    volrel_relaxed   = d_work["VolRel"].isna() | (d_work["VolRel"] >= DETECTOR_BEAR_ALT_VOLREL_MIN)

    have_ml = use_ml_filter and (tkey in ml_models) and (ml_features is not None and len(ml_features) > 0)
    ml_prob = None
    p_thresh = ml_thresholds.get(tkey, GLOBAL_MIN_PROB)
    if have_ml:
        feat_df = pd.DataFrame({f: d_work.get(f, np.nan) for f in ml_features})
        try:
            ml_prob = pd.Series(ml_models[tkey].predict_proba(feat_df), index=d_work.index)
        except Exception:
            ml_prob = None

    def build_mask(cfg: dict) -> pd.Series:
        if not cfg:
            return pd.Series(False, index=d_work.index)
        m = pd.Series(True, index=d_work.index)
        for (col, kind), val in cfg.items():
            if kind == "range":
                lo, hi = val
                m &= d_work[col].between(lo, hi, inclusive="both")
            elif kind == "min":
                m &= d_work[col] >= val
            elif kind == "max":
                m &= d_work[col] <= val
        return m

    mask_bull_22a = build_mask(bull_cfg) & allow_bull & volrel_ok_global
    mask_bear_22a = build_mask(bear_cfg) & allow_bear & volrel_ok_global
    mask_bear_22b = build_mask(bear_alt) & allow_bear & volrel_relaxed if bear_alt else pd.Series(False, index=d_work.index)

    if have_ml and ml_prob is not None:
        mask_ml_bull = ml_prob >= p_thresh
        mask_ml_bear = (ml_prob + ML_SOFT_MARGIN_BEAR) >= p_thresh
    else:
        mask_ml_bull = pd.Series(True, index=d_work.index)
        mask_ml_bear = pd.Series(True, index=d_work.index)

    bull_hits     = d_work.index[mask_bull_22a & mask_ml_bull]
    bear_hits_22a = d_work.index[mask_bear_22a & mask_ml_bear]
    bear_hits_22b = d_work.index[mask_bear_22b & mask_ml_bear]

    now_iso = datetime.now(india_tz).isoformat()

    def _emit_rows(id_suffix: str, entry_label: str, trend: str, idxs, with_ml: bool):
        if len(idxs) == 0:
            return
        rows = d_work.loc[idxs]
        for r_idx, r in rows.iterrows():
            sid = f"{ticker}-{pd.to_datetime(r['date']).isoformat()}-{id_suffix}"
            if sid in existing_signal_ids:
                continue
            rec = {
                "Ticker": ticker,
                "date": r["date"],
                "Entry Type": entry_label,
                "Trend Type": trend,
                "Price": round(float(r.get("close", np.nan)), 2),
                "Signal_ID": sid,
                "logtime": now_iso,
                "Entry Signal": "Yes",
            }
            if with_ml and (ml_prob is not None):
                rec["ML_Prob"] = round(float(ml_prob.loc[r_idx]), 4)
                rec["ML_Thresh"] = round(float(p_thresh), 4)
            signals.append(rec)
            existing_signal_ids.add(sid)

    if have_ml and ml_prob is not None:
        _emit_rows("BULL_CSV_FITTED_ML", "CSVFittedFilter+ML", "Bullish", bull_hits, True)
        _emit_rows("BEAR_CSV_FITTED_ML", "CSVFittedFilter+ML", "Bearish", bear_hits_22a, True)
        _emit_rows("BEAR_CSV_FITTED_PERM_ML", "CSVFittedFilter_PERMISSIVE+ML", "Bearish", bear_hits_22b, True)
    else:
        _emit_rows("BULL_CSV_FITTED", "CSVFittedFilter", "Bullish", bull_hits, False)
        _emit_rows("BEAR_CSV_FITTED", "CSVFittedFilter", "Bearish", bear_hits_22a, False)
        _emit_rows("BEAR_CSV_FITTED_PERM", "CSVFittedFilter_PERMISSIVE", "Bearish", bear_hits_22b, False)

    if debug:
        print(f"[{ticker}] hits => bull22a:{len(bull_hits)} bear22a:{len(bear_hits_22a)} bear22b:{len(bear_hits_22b)} (mode={compat_mode})")

    return signals

# ---------------------------------------------
# 6) Update main CSV (CalcFlag/Entry Signal)
# ---------------------------------------------
def normalize_time(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        raise KeyError("DataFrame missing 'date' column.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d.dropna(subset=["date"], inplace=True)
    if d["date"].dt.tz is None:
        d["date"] = d["date"].dt.tz_localize(tz)
    else:
        d["date"] = d["date"].dt.tz_convert(tz)
    d.sort_values("date", inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d

def mark_signals_in_main_csv(ticker: str, signals_list: list, main_path: str, tz_obj) -> None:
    if not os.path.exists(main_path):
        logger.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return
    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)
    try:
        with lock:
            df_main = pd.read_csv(main_path)
            if "date" in df_main.columns:
                df_main = normalize_time(df_main, tz="Asia/Kolkata")
            if df_main.empty:
                logger.warning(f"[{ticker}] => main_indicators empty. Skipping.")
                return
            for col in ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]:
                if col not in df_main.columns:
                    df_main[col] = ""
            signal_ids = [sig["Signal_ID"] for sig in signals_list if "Signal_ID" in sig]
            if not signal_ids:
                return
            mask = df_main["Signal_ID"].isin(signal_ids)
            if mask.any():
                df_main.loc[mask, "Entry Signal"] = "Yes"
                df_main.loc[mask, "CalcFlag"] = "Yes"
                current_time_iso = datetime.now(tz_obj).isoformat()
                df_main.loc[mask & (df_main["logtime"] == ""), "logtime"] = current_time_iso
                df_main.sort_values("date", inplace=True)
                df_main.to_csv(main_path, index=False)
    except Timeout:
        logger.error(f"[{ticker}] => FileLock timeout for {main_path}.")
    except Exception as e:
        logger.error(f"[{ticker}] => Error marking signals: {e}")

# ---------------------------------------------
# 7) Detect signals for one trading date across all CSVs (PARITY by default)
# ---------------------------------------------
def find_price_action_entries_for_date(
    target_date: datetime.date,
    compat_mode: str = COMPAT_MODE_DEFAULT,   # default to CLEAN parity
    warmup_days: int = 1,
    max_workers: int | None = None,
) -> pd.DataFrame:
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(15, 10)))

    DETECTOR_VOLREL_WINDOW = 10
    DETECTOR_VOLREL_MIN_PERIODS = 10

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logger.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals: List[dict] = []
    lock = threading.Lock()

    def _usecols_pred(c: str) -> bool:
        if c == "date":
            return True
        if c in NEEDED_COLS:
            return True
        try:
            if ML_FEATURES and c in ML_FEATURES:
                return True
        except Exception:
            pass
        return False

    def process_file(file_path: str) -> list:
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()

        df_today = fast_load_csv_for_day(
            file_path,
            st, et,
            volrel_window=DETECTOR_VOLREL_WINDOW,
            volrel_min_periods=DETECTOR_VOLREL_MIN_PERIODS,
            warmup_days=warmup_days,
        )
        if df_today.empty:
            return []

        if compat_mode == "clean":
            try:
                df_full = pd.read_csv(
                    file_path,
                    usecols=_usecols_pred,
                    engine="c",
                    low_memory=False,
                    on_bad_lines="skip",
                )
                df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
                df_full.dropna(subset=["date"], inplace=True)
                if df_full["date"].dt.tz is None:
                    df_full["date"] = df_full["date"].dt.tz_localize(india_tz)
                else:
                    df_full["date"] = df_full["date"].dt.tz_convert(india_tz)
                df_full.sort_values("date", inplace=True)
                df_full.reset_index(drop=True, inplace=True)
            except Exception:
                df_full = pd.DataFrame()
        else:
            df_full = pd.DataFrame()

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,
            df_for_detection=df_today,
            existing_signal_ids=existing_ids,
            debug=False,
            use_ml_filter=USE_ML_FILTER,
            ml_models=ML_MODELS,
            ml_features=ML_FEATURES,
            ml_thresholds=ML_THRESHOLDS,
            volrel_window=DETECTOR_VOLREL_WINDOW,
            volrel_min_periods=DETECTOR_VOLREL_MIN_PERIODS,
            compat_mode=compat_mode,
        )

        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
        return new_signals

    if max_workers is None:
        max_workers = min(12, max(6, (os.cpu_count() or 8)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {target_date}"):
            try:
                sigs = fut.result()
                if sigs:
                    all_signals.extend(sigs)
            except Exception as e:
                logger.error(f"Error processing file {futures[fut]}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logger.info(f"No new signals detected for {target_date}.")
        return pd.DataFrame()

    return pd.DataFrame(all_signals).sort_values("date").reset_index(drop=True)

# ---------------------------------------------
# 8) Papertrade builder (first per ticker; prefers highest ML_Prob)
# ---------------------------------------------
def create_papertrade_df(df_signals: pd.DataFrame, output_file: str, target_date) -> None:
    if df_signals.empty:
        print("[Papertrade] DataFrame is empty => no rows to add.")
        return
    df = normalize_time(df_signals, tz="Asia/Kolkata").sort_values("date")
    start_dt = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    end_dt   = india_tz.localize(datetime.combine(target_date, dt_time(15, 30)))
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    if df.empty:
        print("[Papertrade] No rows in [09:25 - 15:30].")
        return

    for col in ["Target Price", "Quantity", "Total value"]:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price <= 0:
            continue
        if trend == "Bullish":
            target = price * 1.01
        elif trend == "Bearish":
            target = price * 0.99
        else:
            target = price
        qty = int(100000 / price) if price > 0 else 0
        total_val = qty * price
        df.at[idx, "Target Price"] = round(target, 2)
        df.at[idx, "Quantity"] = qty
        df.at[idx, "Total value"] = round(total_val, 2)

    if "time" not in df.columns:
        df["time"] = datetime.now(india_tz).strftime("%H:%M:%S")
    else:
        blank_mask = (df["time"].isna()) | (df["time"] == "")
        if blank_mask.any():
            df.loc[blank_mask, "time"] = datetime.now(india_tz).strftime("%H:%M:%S")

    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        if os.path.exists(output_file):
            existing = normalize_time(pd.read_csv(output_file), tz="Asia/Kolkata")
            existing_keys = set((erow["Ticker"], erow["date"].date()) for _, erow in existing.iterrows())
            new_rows = []
            if "ML_Prob" in df.columns:
                df = df.sort_values(["Ticker","ML_Prob","date"], ascending=[True, False, True])
            for tkr, g in df.groupby("Ticker"):
                g = g.sort_values("date")
                row = g.iloc[0]
                key = (row["Ticker"], row["date"].date())
                if key not in existing_keys:
                    new_rows.append(row)
            if not new_rows:
                print("[Papertrade] No brand-new rows to append. All existing.")
                return
            new_df = pd.DataFrame(new_rows, columns=df.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["Ticker", "date"], keep="first", inplace=True)
            combined.sort_values("date", inplace=True)
            combined.to_csv(output_file, index=False)
            print(f"[Papertrade] Appended => {output_file} with {len(new_rows)} new row(s).")
        else:
            if "ML_Prob" in df.columns:
                df = df.sort_values(["Ticker","ML_Prob","date"], ascending=[True, False, True])
                df = df.groupby("Ticker", as_index=False).head(1)
            else:
                df.drop_duplicates(subset=["Ticker"], keep="first", inplace=True)
            df.sort_values("date", inplace=True)
            df.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df)} rows.")

# ---------------------------------------------
# 9) Orchestrator for a single date (uses CLEAN parity by default)
# ---------------------------------------------
def run_for_date(date_str: str, compat_mode: str = COMPAT_MODE_DEFAULT) -> None:
    try:
        target_day = datetime.strptime(date_str, "%Y-%m-%d").date()
        date_label = target_day.strftime("%Y-%m-%d")
        entries_file    = os.path.join(ENTRIES_DIR, f"{date_label}_entries.csv")
        papertrade_file = f"papertrade_5min_v4aa_{date_label}.csv"

        signals_df = find_price_action_entries_for_date(target_day, compat_mode=compat_mode)
        if signals_df.empty:
            pd.DataFrame().to_csv(entries_file, index=False)
            print(f"[ENTRIES] No signals for {date_label}. Wrote empty CSV.")
            return

        signals_df.to_csv(entries_file, index=False)
        print(f"[ENTRIES] Wrote {len(signals_df)} signals to {entries_file}")

        for ticker, group in signals_df.groupby("Ticker"):
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, group.to_dict("records"), main_csv_path, india_tz)

        create_papertrade_df(signals_df, papertrade_file, target_day)
        print(f"[PAPERTRADE] Entries written to {papertrade_file}.")
    except Exception as e:
        logger.error(f"Error in run_for_date({date_str}): {e}")
        logger.error(traceback.format_exc())

# ---------------------------------------------
# 10) Graceful shutdown handler
# ---------------------------------------------
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

_signal.signal(_signal.SIGINT, signal_handler)
_signal.signal(_signal.SIGTERM, signal_handler)

# ---------------------------------------------
# 11) Entry Point (Multiple Dates)
# ---------------------------------------------
if __name__ == "__main__":
    date_args = sys.argv[1:]
    if not date_args:
        date_args = [
            "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
            "2025-03-10","2025-03-11","2025-03-12","2025-03-13","2025-03-17",
            "2025-03-18","2025-03-19","2025-03-20","2025-03-21","2025-03-24",
            "2025-03-25","2025-03-26","2025-03-27","2025-03-28","2025-03-31",
            "2025-04-01","2025-04-02","2025-04-03","2025-04-04","2025-04-07",
            "2025-04-08","2025-04-09","2025-04-10","2025-04-14","2025-04-15",
            "2025-04-16","2025-04-17","2025-04-18","2025-04-21","2025-04-22",
            "2025-04-23","2025-04-24","2025-04-25","2025-04-28","2025-04-29",
            "2025-04-30","2025-05-02","2025-05-05","2025-05-06","2025-05-07",
            "2025-05-08","2025-05-09","2025-05-12","2025-05-13","2025-05-14",
            "2025-05-15","2025-05-16","2025-05-19","2025-05-20","2025-05-21",
            "2025-05-22","2025-05-23","2025-05-26","2025-05-27","2025-05-28",
            "2025-05-29","2025-05-30","2025-06-02","2025-06-03","2025-06-04",
            "2025-06-05","2025-06-06","2025-06-09","2025-06-10","2025-06-11",
            "2025-06-12","2025-06-13","2025-06-16","2025-06-17","2025-06-18",
            "2025-06-19","2025-06-20","2025-06-23","2025-06-24","2025-06-25",
            "2025-06-26","2025-06-27","2025-06-30",
            "2025-07-01","2025-07-02","2025-07-03","2025-07-04","2025-07-07","2025-07-08","2025-07-09","2025-07-10",
            "2025-07-11","2025-07-14","2025-07-15","2025-07-16","2025-07-17","2025-07-18","2025-07-21","2025-07-22",
            "2025-07-23","2025-07-24","2025-07-25","2025-07-28","2025-07-29","2025-07-30","2025-07-31",
            "2025-08-01","2025-08-04","2025-08-05","2025-08-06","2025-08-07","2025-08-08","2025-08-11","2025-08-12",
            "2025-08-13","2025-08-14","2025-08-18","2025-08-19","2025-08-20","2025-08-21","2025-08-22"
        ]
    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr, compat_mode=COMPAT_MODE_DEFAULT)
