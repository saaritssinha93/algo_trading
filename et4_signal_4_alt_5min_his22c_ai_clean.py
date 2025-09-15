# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 23:17:25 2025

@author: Saarit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:02 2025

Author: Saarit

Clean version + ML filter (RELAXED):
- Detects intraday signals using CSV-fitted indicator ranges (padded).
- Loads per-ticker ML models (per_ticker_models.joblib) and OOF (oof_predictions.csv).
- Computes a per-ticker probability threshold from OOF (max F1), clamped to 0.35–0.70
  and uses a small soft margin so near-misses pass.
- Participation gate: VolRel min_periods=1, threshold=0.95; NaNs filled with 1.0.

Also:
- Robust CSV loader that coalesces duplicate 'date' columns.
- Proper probability extraction (predict_proba/decision_function).
"""

import os
import sys
import glob
import json
import signal
import pytz
import logging
import traceback
import threading
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
from tqdm import tqdm
from filelock import FileLock, Timeout
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
import joblib
from dataclasses import dataclass
from typing import Dict, Any, List

# ---------------------------------------------
# 1) Configuration & setup (relaxed)
# ---------------------------------------------
INDICATORS_DIR = "main_indicators_july_5min1"
SIGNALS_DB     = "generated_signals_historical5minv5.json"
ENTRIES_DIR    = "main_indicators_history_entries_5minv5"
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(ENTRIES_DIR, exist_ok=True)

OOF_PATH     = "oof_predictions.csv"
MODELS_PATH  = "per_ticker_models.joblib"

USE_ML_FILTER    = True      # flip to False to see raw CSV hits
GLOBAL_MIN_PROB  = 0.40      # relaxed floor
GLOBAL_MAX_PROB  = 0.80     # relaxed ceiling
ML_SOFT_MARGIN   = 0.01      # lets near-threshold signals pass

RANGE_PADDING_FRAC = 0.03    # widen each numeric constraint by ±5% (range/min/max)

india_tz = pytz.timezone("Asia/Kolkata")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = TimedRotatingFileHandler(
    "logs\\signalnewn5minv5.log", when="M", interval=30, backupCount=5, delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.warning("Logging with TimedRotatingFileHandler.")
print("Script start...")

# Optional: change working directory if needed
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
try:
    os.chdir(cwd)
except Exception as e:
    logger.error(f"Error changing directory: {e}")

# ---------------------------------------------
# 1b) Model wrapper
# ---------------------------------------------
@dataclass
class PerTickerModel:
    kind: str                   # "knn" or "constant"
    pipeline: Any = None        # sklearn Pipeline / estimator
    const_p: float = None       # float for "constant"
    features: List[str] = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.features]
        if self.kind == "constant":
            return np.full((len(X),), float(self.const_p), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba(X)[:, 1]
            elif hasattr(self.pipeline, "decision_function"):
                z = self.pipeline.decision_function(X)
                probs = 1.0 / (1.0 + np.exp(-z))
            else:
                preds = self.pipeline.predict(X)
                probs = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
        return np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)

# ---------------------------------------------
# 1c) Load ML assets & thresholds
# ---------------------------------------------
ML_MODELS: Dict[str, PerTickerModel] = {}
ML_FEATURES: List[str] = []   # optional, models hold their own
ML_THRESHOLDS: Dict[str, float] = {}

def _best_threshold_from_oof(df_tkr: pd.DataFrame) -> float:
    if df_tkr.empty:
        return GLOBAL_MIN_PROB
    y = pd.to_numeric(df_tkr["HitTarget"], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(df_tkr["PredProb"], errors="coerce").fillna(0.0).to_numpy()
    if np.unique(y).size < 2:
        thr = float(np.clip(np.nanmean(p) if len(p) else GLOBAL_MIN_PROB, GLOBAL_MIN_PROB, GLOBAL_MAX_PROB))
        return thr
    qs = np.unique(np.quantile(p, np.linspace(0.10, 0.90, 17)))
    grid = np.unique(np.concatenate([qs, np.linspace(0.25, 0.85, 13)]))
    best_f1, best_t = -1.0, (GLOBAL_MIN_PROB + GLOBAL_MAX_PROB) / 2
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
    return float(np.clip(best_t, GLOBAL_MIN_PROB, GLOBAL_MAX_PROB))

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
            if {"Ticker","PredProb","HitTarget"}.issubset(oof.columns):
                oof["TickerKey"] = oof["Ticker"].astype(str).str.strip().str.upper()
                for tkr, g in oof.groupby("TickerKey"):
                    ML_THRESHOLDS[tkr] = _best_threshold_from_oof(g)
                if ML_THRESHOLDS:
                    vals = list(ML_THRESHOLDS.values())
                    print(f"[ML] Thresholds for {len(ML_THRESHOLDS)} tickers "
                          f"(range: {min(vals):.2f}-{max(vals):.2f})")
            else:
                print("[ML] OOF missing required columns; skipping thresholds.")
        except Exception as e:
            print(f"[ML] Could not read OOF: {e}")
    else:
        print(f"[ML] OOF file not found: {OOF_PATH}")

load_ml_assets()

# ---------------------------------------------
# 2) Utilities: signal DB + CSV/time
# ---------------------------------------------
def load_generated_signals() -> set:
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, "r") as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB JSON is corrupted. Starting with empty set.")
    return set()

def save_generated_signals(generated_signals: set) -> None:
    with open(SIGNALS_DB, "w") as f:
        json.dump(list(generated_signals), f, indent=2)

def _read_csv_fast(path: str, usecols=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine="python", usecols=usecols)
    except Exception:
        try:
            return pd.read_csv(path, engine="pyarrow", usecols=usecols)
        except Exception:
            return pd.read_csv(path, usecols=usecols)

def load_and_normalize_csv(file_path: str, expected_cols=None, tz: str = "Asia/Kolkata", usecols=None) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])
    df = _read_csv_fast(file_path, usecols=usecols)
    if df.empty:
        return df
    # coalesce duplicate date columns
    if df.columns.duplicated().any():
        seen = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
    date_like = [c for c in df.columns if c.lower() == "date" or c.lower().startswith("date.")]
    if date_like:
        primary = "date" if "date" in df.columns else date_like[0]
        s = df[primary].copy()
        for c in date_like:
            if c == primary: 
                continue
            s = s.where(s.notna(), df[c])
        df.drop(columns=[c for c in date_like if c != primary], inplace=True, errors="ignore")
        df.rename(columns={primary: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.dropna(subset=["date"], inplace=True)
        df["date"] = df["date"].dt.tz_convert(tz)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    elif expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = "" if c != "date" else pd.NaT
        df = df[expected_cols]
    return df

# ---------------------------------------------
# 3) Session VWAP (needed cols only)
# ---------------------------------------------
def calculate_session_vwap(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vol_col: str = "volume",
    tz: str = "Asia/Kolkata",
) -> pd.DataFrame:
    d = df.copy()
    dt = pd.to_datetime(d["date"], errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(tz)
    else:
        dt = dt.dt.tz_convert(tz)
    d["_session"] = dt.dt.date
    tp = (d[high_col] + d[low_col] + d[price_col]) / 3.0
    tp_vol = tp * d[vol_col]
    d["_cum_tp_vol"] = tp_vol.groupby(d["_session"]).cumsum()
    d["_cum_vol"] = d[vol_col].groupby(d["_session"]).cumsum()
    d["VWAP"] = d["_cum_tp_vol"] / d["_cum_vol"]
    d.drop(columns=["_session", "_cum_tp_vol", "_cum_vol"], inplace=True)
    return d

# ---------------------------------------------
# 4) Signal detection (ranges + optional ML)
# ---------------------------------------------
def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set,
    fitted_csv_path: str = "fitted_indicator_ranges_by_ticker.csv",  # permissive
    volrel_window: int = 10,
    volrel_min_periods: int = 5,      # relaxed
    debug: bool = False,
    volrel_gate_min: float = 1.05,    # relaxed
    use_vwap_side_gate: bool = False,
    vwap_long_min: float = 0.001,
    vwap_short_max: float = -0.001,
    use_ml_filter: bool = USE_ML_FILTER,
    ml_models: Dict[str, PerTickerModel] = ML_MODELS,
    ml_features: List[str] = None,    # model.features will be used
    ml_thresholds: Dict[str, float] = ML_THRESHOLDS,
) -> list:

    import ast
    from datetime import datetime

    signals = []
    if df_for_detection.empty:
        return signals

    # ---- Load per-ticker fitted constraints ----
    def _parse_range_cell(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return (np.nan, np.nan)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            try: return (float(x[0]), float(x[1]))
            except: return (np.nan, np.nan)
        s = str(x).strip()
        if not s: return (np.nan, np.nan)
        try:
            lo, hi = ast.literal_eval(s)
            return (float(lo), float(hi))
        except Exception:
            s = s.strip("[]")
            parts = [p for p in s.split(",") if p.strip() != ""]
            if len(parts) == 2:
                try: return (float(parts[0]), float(parts[1]))
                except: pass
            return (np.nan, np.nan)

    def _to_float(x):
        if x is None or (isinstance(x, float) and pd.isna(x)): return np.nan
        try: return float(x)
        except Exception:
            try: return float(str(x).strip())
            except Exception: return np.nan

    try:
        cfg_df = pd.read_csv(fitted_csv_path)
    except Exception as e:
        print(f"[detect] Could not read '{fitted_csv_path}': {e}")
        return signals

    tkey = str(ticker).strip().upper()
    if "Ticker" not in cfg_df.columns:
        return signals
    cfg_df = cfg_df.copy()
    cfg_df["__TickerKey"] = cfg_df["Ticker"].astype(str).str.strip().str.upper()
    cfg_df = cfg_df[cfg_df["__TickerKey"] == tkey]
    if cfg_df.empty:
        if debug: print(f"[detect:{ticker}] no per-ticker rows in fitted CSV.")
        return signals

    sides_raw = {}
    for _, row in cfg_df.iterrows():
        side = str(row.get("Side", "")).strip()
        if side not in ("Bullish", "Bearish"): continue
        constraints = {}
        for col in row.index:
            if col in ("Ticker", "__TickerKey", "Side", "Samples_wins", "Quantiles_used"): 
                continue
            if col.endswith("_range"):
                constraints[col] = _parse_range_cell(row[col])
            elif col.endswith("_min") or col.endswith("_max"):
                constraints[col] = _to_float(row[col])
        if constraints:
            sides_raw[side] = constraints
    if not sides_raw:
        if debug: print(f"[detect:{ticker}] no constraints present after parse.")
        return signals

    # ---- Derived columns ----
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined.dropna(subset=["date"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["date_only"] = combined["date"].dt.date

    if "VWAP" not in combined.columns:
        combined = calculate_session_vwap(combined)

    if "rolling_vol_10" in combined.columns:
        rv = combined["rolling_vol_10"]
    elif "rolling_vol10" in combined.columns:
        rv = combined["rolling_vol10"]; combined["rolling_vol_10"] = rv
    else:
        rv = combined["volume"].rolling(volrel_window, min_periods=volrel_min_periods).mean()
        combined["rolling_vol_10"] = rv

    eps = 1e-9
    combined["StochD"] = combined.get("Stoch_%D", np.nan)
    combined["StochK"] = combined.get("Stoch_%K", np.nan)
    combined["VWAP_Dist"] = combined["close"] / combined["VWAP"] - 1.0
    combined["EMA50_Dist"]  = combined["close"] / combined["EMA_50"]  - 1.0 if "EMA_50"  in combined.columns else np.nan
    combined["EMA200_Dist"] = combined["close"] / combined["EMA_200"] - 1.0 if "EMA_200" in combined.columns else np.nan
    combined["SMA20_Dist"]  = combined["close"] / combined["20_SMA"]  - 1.0 if "20_SMA"  in combined.columns else np.nan
    combined["VolRel"] = combined["volume"] / (combined["rolling_vol_10"] + eps)
    combined.loc[:, "VolRel"] = combined["VolRel"].fillna(1.0)   # safe assignment; keeps early bars
    combined["ATR_Pct"] = (combined["ATR"] / (combined["close"] + eps)) if "ATR" in combined.columns else np.nan
    if {"Upper_Band","Lower_Band"}.issubset(combined.columns):
        bbw = (combined["Upper_Band"] - combined["Lower_Band"])
        combined["BandWidth"] = bbw / (combined["close"] + eps)
        combined["BBP"] = (combined["close"] - combined["Lower_Band"]) / (bbw + eps)
    else:
        combined["BandWidth"] = np.nan
        combined["BBP"] = np.nan
    if "OBV" in combined.columns:
        obv_prev = combined["OBV"].shift(10)
        combined["OBV_Slope10"] = (combined["OBV"] - obv_prev) / (obv_prev.abs() + eps)
    else:
        combined["OBV_Slope10"] = np.nan

    colmap = {
        "RSI":"RSI","ADX":"ADX","CCI":"CCI","MFI":"MFI","BBP":"BBP","BandWidth":"BandWidth","ATR_Pct":"ATR_Pct",
        "StochD":"StochD","StochK":"StochK","MACD_Hist":"MACD_Hist","VWAP_Dist":"VWAP_Dist",
        "Daily_Change":"Daily_Change","EMA50_Dist":"EMA50_Dist","EMA200_Dist":"EMA200_Dist",
        "SMA20_Dist":"SMA20_Dist","OBV_Slope10":"OBV_Slope10","VolRel":"VolRel",
    }
    available_cols = set(combined.columns)

    # ----- Relax: pad constraints -----
    def _pad_range(lo, hi):
        if np.isnan(lo) or np.isnan(hi): return (np.nan, np.nan)
        span = hi - lo
        pad  = abs(span) * RANGE_PADDING_FRAC
        return (lo - pad, hi + pad)

    def _pad_min(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
        return v * (1.0 - RANGE_PADDING_FRAC if v >= 0 else 1.0 + RANGE_PADDING_FRAC)

    def _pad_max(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
        return v * (1.0 + RANGE_PADDING_FRAC if v >= 0 else 1.0 - RANGE_PADDING_FRAC)

    def _clean_constraints(raw: dict) -> dict:
        cleaned = {}
        for k, v in raw.items():
            if k.endswith("_range"):
                base = k[:-6]; col = colmap.get(base, base)
                if col not in available_cols: continue
                lo, hi = v if isinstance(v, (list, tuple)) and len(v) == 2 else (np.nan, np.nan)
                lo, hi = _pad_range(lo, hi)
                if np.isnan(lo) or np.isnan(hi): continue
                cleaned[k] = (float(lo), float(hi))
            elif k.endswith("_min"):
                base = k[:-4]; col = colmap.get(base, base)
                if col not in available_cols: continue
                vv = _pad_min(v)
                if vv is None or (isinstance(vv, float) and np.isnan(vv)): continue
                cleaned[k] = float(vv)
            elif k.endswith("_max"):
                base = k[:-4]; col = colmap.get(base, base)
                if col not in available_cols: continue
                vv = _pad_max(v)
                if vv is None or (isinstance(vv, float) and np.isnan(vv)): continue
                cleaned[k] = float(vv)
        return cleaned

    sides = {side: _clean_constraints(raw) for side, raw in sides_raw.items()}
    if all(len(cfg) == 0 for cfg in sides.values()):
        if debug: print(f"[detect:{ticker}] All constraints dropped.")
        return signals

    def find_cidx(ts):
        ts = pd.to_datetime(ts)
        exact = combined.index[combined["date"] == ts]
        if len(exact): return int(exact[0])
        d = ts.date()
        mask = (combined["date_only"] == d) & (combined["date"] <= ts)
        if mask.any(): return int(combined.index[mask][-1])
        return None

    def _check_side_constraints(row: pd.Series, side_constraints: dict) -> bool:
        for k, rng in side_constraints.items():
            if k.endswith("_range"):
                base = k[:-6]; col = colmap.get(base, base)
                x = row.get(col, np.nan); lo, hi = rng
                if pd.isna(x) or not (lo <= x <= hi): return False
        for k, thr in side_constraints.items():
            if k.endswith("_min"):
                base = k[:-4]; col = colmap.get(base, base)
                x = row.get(col, np.nan)
                if pd.isna(x) or not (x >= thr): return False
        for k, thr in side_constraints.items():
            if k.endswith("_max"):
                base = k[:-4]; col = colmap.get(base, base)
                x = row.get(col, np.nan)
                if pd.isna(x) or not (x <= thr): return False
        return True

    # --- ML helpers ---
    tkey = str(ticker).strip().upper()
    model = ml_models.get(tkey)
    model_features = getattr(model, "features", []) if model is not None else []
    p_thresh = ml_thresholds.get(tkey, GLOBAL_MIN_PROB)
    have_ml = use_ml_filter and (model is not None) and (len(model_features) > 0)

    # Evaluate detection slice
    for _, row_det in df_for_detection.iterrows():
        dt = row_det.get("date", None)
        if pd.isna(dt): continue

        cidx = find_cidx(dt)
        if cidx is None: continue

        r = combined.loc[cidx]

        # Participation brake (relaxed)
        volrel_val = r.get("VolRel", np.nan)
        if pd.isna(volrel_val) or volrel_val < volrel_gate_min:
            continue

        # Optional VWAP side gate (off by default)
        allow_bull = allow_bear = True
        if use_vwap_side_gate:
            vdist = r.get("VWAP_Dist", np.nan)
            if pd.isna(vdist): continue
            if vdist < vwap_long_min:  allow_bull = False
            if vdist > vwap_short_max: allow_bear = False

        # ML prob (if enabled)
        ml_prob = None
        if have_ml:
            one_row = pd.DataFrame([{f: r.get(f, np.nan) for f in model_features}])
            try:
                ml_prob = float(model.predict_proba(one_row)[0])
            except Exception:
                ml_prob = None

        # ---- Bullish ----
        bull_cfg = sides.get("Bullish")
        if bull_cfg and allow_bull and _check_side_constraints(r, bull_cfg):
            ok_ml = True
            if have_ml:
                if (ml_prob is None) or (ml_prob + ML_SOFT_MARGIN < p_thresh):
                    ok_ml = False
            if ok_ml:
                sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_CSV_FITTED" + ("_ML" if have_ml else "")
                if sid not in existing_signal_ids:
                    signals.append({
                        "Ticker": ticker, "date": dt,
                        "Entry Type": "CSVFittedFilter" + ("+ML" if have_ml else ""),
                        "Trend Type": "Bullish",
                        "Price": round(float(r.get("close", np.nan)), 2),
                        "Signal_ID": sid,
                        "logtime": row_det.get("logtime", datetime.now().isoformat()),
                        "Entry Signal": "Yes",
                        "ML_Prob": (round(ml_prob, 4) if ml_prob is not None else np.nan),
                        "ML_Thresh": round(p_thresh, 4) if have_ml else np.nan,
                    })
                    existing_signal_ids.add(sid)

        # ---- Bearish ----
        bear_cfg = sides.get("Bearish")
        if bear_cfg and allow_bear and _check_side_constraints(r, bear_cfg):
            ok_ml = True
            if have_ml:
                if (ml_prob is None) or (ml_prob + ML_SOFT_MARGIN < p_thresh):
                    ok_ml = False
            if ok_ml:
                sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_CSV_FITTED" + ("_ML" if have_ml else "")
                if sid not in existing_signal_ids:
                    signals.append({
                        "Ticker": ticker, "date": dt,
                        "Entry Type": "CSVFittedFilter" + ("+ML" if have_ml else ""),
                        "Trend Type": "Bearish",
                        "Price": round(float(r.get("close", np.nan)), 2),
                        "Signal_ID": sid,
                        "logtime": row_det.get("logtime", datetime.now().isoformat()),
                        "Entry Signal": "Yes",
                        "ML_Prob": (round(ml_prob, 4) if ml_prob is not None else np.nan),
                        "ML_Thresh": round(p_thresh, 4) if have_ml else np.nan,
                    })
                    existing_signal_ids.add(sid)

    return signals

# ---------------------------------------------
# 5) Mark signals in main CSV => CalcFlag='Yes'
# ---------------------------------------------
def mark_signals_in_main_csv(ticker: str, signals_list: list, main_path: str, tz_obj) -> None:
    if not os.path.exists(main_path):
        logging.warning(f"[{ticker}] => No file found: {main_path}. Skipping.")
        return
    lock_path = main_path + ".lock"
    lock = FileLock(lock_path, timeout=10)
    try:
        with lock:
            df_main = load_and_normalize_csv(main_path, expected_cols=["Signal_ID"], tz="Asia/Kolkata")
            if df_main.empty:
                logging.warning(f"[{ticker}] => main_indicators empty. Skipping.")
                return
            for col in ["Signal_ID", "Entry Signal", "CalcFlag", "logtime"]:
                if col not in df_main.columns:
                    df_main[col] = ""
            signal_ids = [sig["Signal_ID"] for sig in signals_list if "Signal_ID" in sig]
            if not signal_ids:
                logging.info(f"[{ticker}] => No valid Signal_IDs to update.")
                return
            mask = df_main["Signal_ID"].isin(signal_ids)
            if mask.any():
                df_main.loc[mask, ["Entry Signal", "CalcFlag"]] = "Yes"
                current_time_iso = datetime.now(tz_obj).isoformat()
                df_main.loc[mask & (df_main["logtime"] == ""), "logtime"] = current_time_iso
                df_main.sort_values("date", inplace=True)
                df_main.to_csv(main_path, index=False)
                logging.info(f"[{ticker}] => Set CalcFlag='Yes' for {mask.sum()} row(s).")
            else:
                logging.info(f"[{ticker}] => No matching Signal_IDs found to update.")
    except Timeout:
        logging.error(f"[{ticker}] => FileLock timeout for {main_path}.")
    except Exception as e:
        logging.error(f"[{ticker}] => Error marking signals: {e}")

# ---------------------------------------------
# 6) Detect signals for one trading date across all CSVs
# ---------------------------------------------
def find_price_action_entries_for_date(target_date: datetime.date) -> pd.DataFrame:
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(15, 10)))

    DETECTOR_VOLREL_WINDOW = 10
    DETECTOR_VOLREL_MIN_PERIODS = 5   # relaxed
    DETECTOR_DEBUG = False

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def _header_cols(path: str):
        try:
            return list(pd.read_csv(path, nrows=0, engine="python").columns)
        except Exception:
            try:
                return list(pd.read_csv(path, nrows=0, engine="pyarrow").columns)
            except Exception:
                return list(pd.read_csv(path, nrows=0).columns)

    def process_file(file_path: str) -> list:
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()
        header = set(_header_cols(file_path))
        base_cols = {"date","open","high","low","close","volume",
                     "Upper_Band","Lower_Band","MACD_Hist","RSI","ADX","CCI","MFI",
                     "ATR","OBV","Stoch_%D","Stoch_%K","EMA_50","EMA_200","20_SMA"}
        usecols = list(base_cols & header)

        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata", usecols=usecols)
        if df_full.empty:
            return []
        df_today = df_full[(df_full["date"] >= st) & (df_full["date"] <= et)].copy()
        if df_today.empty:
            return []
        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,
            df_for_detection=df_today,
            existing_signal_ids=existing_ids,
            fitted_csv_path="fitted_indicator_ranges_by_ticker20_80.csv",
            volrel_window=DETECTOR_VOLREL_WINDOW,
            volrel_min_periods=DETECTOR_VOLREL_MIN_PERIODS,
            debug=DETECTOR_DEBUG,
            use_ml_filter=USE_ML_FILTER,
            ml_models=ML_MODELS,
            ml_thresholds=ML_THRESHOLDS,
        )
        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
            return new_signals
        return []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {target_date}"):
            try:
                signals = fut.result()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logging.error(f"Error processing file {futures[fut]}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logging.info(f"No new signals detected for {target_date}.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(all_signals).sort_values("date").reset_index(drop=True)
    logging.info(f"Total new signals detected for {target_date}: {len(signals_df)}")
    return signals_df

# ---------------------------------------------
# 7) Papertrade builder (first signal per ticker)
# ---------------------------------------------
def normalize_time(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        raise KeyError("DataFrame missing 'date' column.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_convert(tz)
    d.dropna(subset=["date"], inplace=True)
    d.sort_values("date", inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d

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
        target = price * (1.01 if trend == "Bullish" else 0.99 if trend == "Bearish" else 1.0)
        qty = int(100000 / price)
        df.at[idx, "Target Price"] = round(target, 2)
        df.at[idx, "Quantity"] = qty
        df.at[idx, "Total value"] = round(qty * price, 2)

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
            existing = load_and_normalize_csv(output_file, tz="Asia/Kolkata")
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
# 8) Orchestrator for a single date
# ---------------------------------------------
def run_for_date(date_str: str) -> None:
    try:
        target_day = datetime.strptime(date_str, "%Y-%m-%d").date()
        date_label = target_day.strftime("%Y-%m-%d")
        entries_file    = os.path.join(ENTRIES_DIR, f"{date_label}_entries.csv")
        papertrade_file = f"papertrade_5min_v5_{date_label}.csv"

        signals_df = find_price_action_entries_for_date(target_day)
        if signals_df.empty:
            logging.info(f"No signals found for {date_label}.")
            pd.DataFrame().to_csv(entries_file, index=False)
            return

        signals_df.to_csv(entries_file, index=False)
        print(f"[ENTRIES] Wrote {len(signals_df)} signals to {entries_file}")

        for ticker, group in signals_df.groupby("Ticker"):
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, group.to_dict("records"), main_csv_path, india_tz)

        create_papertrade_df(signals_df, papertrade_file, target_day)
        logging.info(f"Papertrade entries written to {papertrade_file}.")
    except Exception as e:
        logging.error(f"Error in run_for_date({date_str}): {e}")
        logging.error(traceback.format_exc())

# ---------------------------------------------
# 9) Graceful shutdown handler
# ---------------------------------------------
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------------------------------------------
# 10) Entry Point (Multiple Dates)
# ---------------------------------------------
if __name__ == "__main__":
    date_args = sys.argv[1:]
    if not date_args:
        # Backfill calendar (example)
        date_args = [
            "2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09",
            "2024-08-12","2024-08-13","2024-08-14","2024-08-16",
            "2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23",
            "2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30",
            "2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06",
            "2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13",
            "2024-09-16","2024-09-17","2024-09-18","2024-09-19","2024-09-20",
            "2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27",
            "2024-09-30","2024-10-01","2024-10-03","2024-10-04","2024-10-07",
            "2024-10-08","2024-10-09","2024-10-10","2024-10-11","2024-10-14",
            "2024-10-15","2024-10-16","2024-10-17","2024-10-18","2024-10-21",
            "2024-10-22","2024-10-23","2024-10-24","2024-10-25","2024-10-28",
            "2024-10-29","2024-10-30","2024-10-31",
            "2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08",
            "2024-11-11","2024-11-12","2024-11-13","2024-11-14","2024-11-18",
            "2024-11-19","2024-11-20","2024-11-21","2024-11-22","2024-11-25",
            "2024-11-26","2024-11-27","2024-11-28","2024-11-29",
            "2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06",
            "2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13",
            "2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20",
            "2024-12-23","2024-12-24","2024-12-26","2024-12-27","2024-12-30",
            "2024-12-31","2025-01-01","2025-01-02","2025-01-03","2025-01-06",
            "2025-01-07","2025-01-08","2025-01-09","2025-01-10","2025-01-13",
            "2025-01-14","2025-01-15","2025-01-16","2025-01-17","2025-01-20",
            "2025-01-21","2025-01-22","2025-01-23","2025-01-24","2025-01-27",
            "2025-01-28","2025-01-29","2025-01-30","2025-01-31","2025-02-03",
            "2025-02-04","2025-02-05","2025-02-06","2025-02-07","2025-02-10",
            "2025-02-11","2025-02-12","2025-02-13","2025-02-14","2025-02-17",
            "2025-02-18","2025-02-19","2025-02-20","2025-02-21","2025-02-24",
            "2025-02-25","2025-02-26","2025-02-27","2025-02-28",
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
            # July 2025
            "2025-07-01","2025-07-02","2025-07-03","2025-07-04","2025-07-07","2025-07-08","2025-07-09","2025-07-10",
            "2025-07-11","2025-07-14","2025-07-15","2025-07-16","2025-07-17","2025-07-18","2025-07-21","2025-07-22",
            "2025-07-23","2025-07-24","2025-07-25","2025-07-28","2025-07-29","2025-07-30","2025-07-31"

            # August 2025
            "2025-08-01","2025-08-04","2025-08-05","2025-08-06","2025-08-07","2025-08-08","2025-08-11","2025-08-12",
            "2025-08-13","2025-08-14","2025-08-18","2025-08-19","2025-08-20","2025-08-21","2025-08-22","2025-08-25",
            "2025-08-26","2025-08-28","2025-08-29"
        ]
    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
