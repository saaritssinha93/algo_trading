# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:02 2025

Author: Saarit (relaxed rewrite v7)

Clean version + ML filter (relaxed & synced) — **More Entries, Still Sensible**

Aligned with the weekly-frozen bundle:
- Uses local per-ticker KNN + global HGB + stacker (live producers) from models bundle
- Applies per-ticker isotonic calibrators and uses saved per-ticker thresholds ('chosen')
- Falls back to OOF only if needed
- Pulls ranges_csv and live_defensive_bump from manifest; derives permissive 20_80 ranges

RELAX KNOBS (vs your last v6):
- VolRel gate ↓ 0.85 → **0.75** (global), alt-union gate also **0.75**
- Range widening ±10% → **±20%**
- VWAP side gates widened: Bull allowed down to **−0.75%**, Bear up to **+0.75%**
- ML threshold aggressive shift: −0.02 → **−0.04**; Bear gets +0.01 soft margin
- New **K-of-N** for *_range* constraints (relaxed): require **≥75%** to pass (alt-union uses **65%**)
- New **near-miss ML admit zone**: if within **2pp** below thr and tape is favorable (good VolRel or VWAP proximity), admit
- Quality floors softer: ATR% ≥ **0.12%** (was 0.15%), BandWidth ≥ **0.15%** (was 0.18%)
- Early bars: keep min_periods=1 for rolling volume (no zeroing)

This version aims for ~5–20 entries/day depending on universe & day. Tighten by raising
VOLREL_GATE_MIN_DEFAULT, reducing WIDEN_PCT, or setting RANGE_PASS_RATIO to 0.85.
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
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
from tqdm import tqdm
from filelock import FileLock, Timeout
from logging.handlers import TimedRotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# --- joblib presence flag ---
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# ---------------------------------------------
# 1) Configuration & setup
# ---------------------------------------------
from pathlib import Path
import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Base / absolute paths (prevents accidental empty scans)
try:
    BASE_DIR = Path(__file__).resolve().parent
except Exception:
    BASE_DIR = Path.cwd()

ASSET_DIR = (BASE_DIR / "models"); ASSET_DIR.mkdir(exist_ok=True)
LOGS_DIR = (BASE_DIR / "logs"); LOGS_DIR.mkdir(exist_ok=True)

# Toggle for sanity checks during bring-up
RELAX_EARLY_VOLREL_MIN_PERIODS = True  # True: min_periods=1 so early bars aren't zeroed
VOLREL_GATE_MIN_DEFAULT = 0.75         # relaxed from 0.85 => more entries
DETECTOR_DEBUG_GLOBAL = False          # lightweight debug prints

INDICATORS_DIR = str((BASE_DIR / "main_indicators_5min").resolve())
SIGNALS_DB     = str((BASE_DIR / "generated_signals_historical5minv7_relaxed.json").resolve())
ENTRIES_DIR    = str((BASE_DIR / "main_indicators_history_entries_5minv7_relaxed").resolve())
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(ENTRIES_DIR, exist_ok=True)

# Defaults (overridden by manifest if present)
OOF_PATH        = str((BASE_DIR / "oof_predictions.csv").resolve())
MODELS_PATH     = str((BASE_DIR / "per_ticker_models.joblib").resolve())
FITTED_CSV_MAIN = str((BASE_DIR / "fitted_indicator_ranges_by_ticker.csv").resolve())  # primary
FITTED_CSV_ALT  = str((BASE_DIR / "fitted_indicator_ranges_by_ticker20_80.csv").resolve())  # permissive/union

# --- ML filter switches / thresholds ---
USE_ML_FILTER = False
GLOBAL_MIN_PROB = 0.50          # floor if per-ticker threshold missing
GLOBAL_MAX_PROB = 0.90          # ceiling to avoid over-aggressive gates
LIVE_DEFENSIVE_BUMP = 0.00      # overridden by manifest if present
ML_SOFT_MARGIN_BEAR = 0.01      # slight bear easing

# Additional dials for more entries (light touch)
WIDEN_PCT = 0.20                 # widen fitted ranges ±20%
AGGR_PROB_SHIFT = -0.04          # lower ML threshold by 4pp (more admits)
VWAP_LONG_MIN = -0.0075          # Bull allowed as low as -0.75% under VWAP
VWAP_SHORT_MAX = 0.0075          # Bear allowed as high as +0.75% above VWAP

# K-of-N relaxers for range constraints
RANGE_PASS_RATIO = 0.75          # require at least 75% of *_range checks to pass
ALT_RANGE_PASS_RATIO = 0.65      # for permissive union CSV

# Near-miss ML soft zone
ML_NEAR_MISS = 0.02              # admit within 2pp if tape is favorable

# union (permissive) VolRel floor
DETECTOR_ALT_VOLREL_MIN = 0.75

# quality floors (avoid dead/noise bars)
MIN_ATR_PCT = 0.0012             # 0.12%
MIN_BB_WIDTH = 0.0015            # 0.15%

india_tz = pytz.timezone("Asia/Kolkata")

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = TimedRotatingFileHandler(
    str(LOGS_DIR / "signal_relaxed_5minv7.log"), when="M", interval=30, backupCount=5, delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logging.info("Logging with TimedRotatingFileHandler.")
print("Script start...")

# ---------------------------------------------
# 1a) Manifest (robust weekly freeze picker)
# ---------------------------------------------

def _pick_latest_manifest(cutoff_date=None, max_age_days=30):
    """
    Pick newest manifest within [-7, +max_age_days] relative to cutoff_date.
    Accept a bundle up to 7 days ahead (weekly freeze) and up to 30 days old.
    """
    if cutoff_date is None:
        cutoff_date = _dt.date.today()
    cands = []
    for p in ASSET_DIR.glob("manifest_*.json"):
        try:
            man = json.loads(p.read_text())
            end = _dt.date.fromisoformat(man.get("end_date"))
            age = (cutoff_date - end).days
            if -7 <= age <= max_age_days:
                cands.append((end, man))
        except Exception:
            pass
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]

_manifest = _pick_latest_manifest(cutoff_date=_dt.date.today())

def _derive_alt_ranges_path(primary_csv: str) -> str:
    """Given the primary ranges CSV path, derive the permissive 20_80 CSV next to it."""
    try:
        p = Path(primary_csv)
        name = p.name
        if "20_80" in name:
            return str(p)
        if "fitted_indicator_ranges_by_ticker_" in name:
            new = name.replace("fitted_indicator_ranges_by_ticker_", "fitted_indicator_ranges_by_ticker20_80_")
            return str(p.with_name(new))
        if name == "fitted_indicator_ranges_by_ticker.csv":
            return str(p.with_name("fitted_indicator_ranges_by_ticker20_80.csv"))
    except Exception:
        pass
    return primary_csv  # fallback

if _manifest:
    OOF_PATH        = str((BASE_DIR / _manifest.get("oof_path", OOF_PATH)).resolve())
    MODELS_PATH     = str((BASE_DIR / _manifest.get("models_path", MODELS_PATH)).resolve())
    FITTED_CSV_MAIN = str((BASE_DIR / _manifest.get("ranges_csv", FITTED_CSV_MAIN)).resolve())
    # live defensive bump from manifest (preferred)
    try:
        LIVE_DEFENSIVE_BUMP = float(_manifest.get("live_defensive_bump", LIVE_DEFENSIVE_BUMP))
    except Exception:
        pass
    # derive alt 20_80 path next to primary
    _alt_guess = _derive_alt_ranges_path(FITTED_CSV_MAIN)
    if os.path.exists(_alt_guess):
        FITTED_CSV_ALT = _alt_guess
    print(f"[Manifest] Using bundle {_manifest.get('stamp')} end={_manifest.get('end_date')}")
else:
    print("[Manifest] No valid manifest found; using default paths.")

print(f"[Detector] Using paths:\n"
      f"  Indicators Dir : {INDICATORS_DIR}\n"
      f"  Ranges CSV     : {FITTED_CSV_MAIN}\n"
      f"  Ranges CSV Alt : {FITTED_CSV_ALT}\n"
      f"  Models         : {MODELS_PATH}\n"
      f"  OOF            : {OOF_PATH}\n"
      f"  Live bump      : {LIVE_DEFENSIVE_BUMP:+.2%}")

# ---------------------------------------------
# 1b) Shims for per-ticker models (compat)
# ---------------------------------------------
try:
    from sklearn.pipeline import Pipeline  # type hints
except Exception:
    Pipeline = Any  # noqa: F821

@dataclass
class PerTickerModel:
    kind: str = "knn"                 # "knn" or "constant"
    pipeline: Optional[Any] = None    # sklearn Pipeline
    const_p: Optional[float] = None   # for constant model
    features: Optional[List[str]] = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        feats = self.features or []
        X = df.copy()
        for c in feats:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feats] if feats else X
        if self.kind == "constant":
            p = float(self.const_p if self.const_p is not None else 0.5)
            return np.full((len(X),), p, dtype=float)
        if self.pipeline is None:
            return np.full((len(X),), 0.5, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.pipeline.predict(X)
        return np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)

@dataclass
class PerTickerModelShim:
    obj: Any
    features: List[str]
    kind: str = "knn"
    const_p: Optional[float] = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.features]
        if hasattr(self.obj, "kind") and hasattr(self.obj, "predict_proba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = self.obj.predict_proba(X)
            return np.clip(np.asarray(p, float), 0.0, 1.0)
        if self.kind == "constant" or getattr(self.obj, "kind", None) == "constant":
            p = self.const_p if self.const_p is not None else getattr(self.obj, "const_p", 0.5)
            return np.full((len(X),), float(p), dtype=float)
        if hasattr(self.obj, "predict_proba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = self.obj.predict_proba(X)
            if isinstance(p, np.ndarray) and p.ndim == 2:
                p = p[:, -1]
            return np.clip(np.asarray(p, float), 0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = self.obj.predict(X)
        return np.clip(np.asarray(p, float), 0.0, 1.0)

# ---------------------------------------------
# 1c) Load ML assets & build thresholds (new bundle schema)
# ---------------------------------------------
ML_MODELS: Dict[str, PerTickerModelShim] = {}   # per-ticker local producers
ML_FEATURES_BASE: List[str] = []                # per-ticker features_base
ML_FEATURES_REGIME: List[str] = []              # regime features (for global)
ML_FEATURES_GLOBAL: List[str] = []              # base + regime (for global)
ML_THRESHOLDS: Dict[str, float] = {}            # per-ticker chosen thresholds
ML_CALIBRATORS: Dict[str, Any] = {}             # per-ticker isotonic
GLOBAL_MODEL_OBJ: Optional[Any] = None          # HGB classifier
STACKER_OBJ: Optional[Any] = None               # logistic regression stacker

def _best_threshold_from_oof_you_dens_j(df_tkr: pd.DataFrame) -> float:
    """Threshold that maximizes Youden's J = TPR - FPR. Uses P_stack_cal if available, else PredProb."""
    if df_tkr.empty:
        return GLOBAL_MIN_PROB
    # prefer stack-calibrated pred from new OOF
    prob_col = "P_stack_cal" if "P_stack_cal" in df_tkr.columns else ("PredProb" if "PredProb" in df_tkr.columns else None)
    if prob_col is None:
        return GLOBAL_MIN_PROB
    y = pd.to_numeric(df_tkr.get("HitTarget"), errors="coerce").fillna(0).astype(int).to_numpy()
    p = np.clip(pd.to_numeric(df_tkr[prob_col], errors="coerce").fillna(0.0).to_numpy(), 0, 1)
    if np.unique(y).size < 2:
        base = float(np.nanmean(p)) if len(p) else GLOBAL_MIN_PROB
        return float(np.clip(base, GLOBAL_MIN_PROB, GLOBAL_MAX_PROB))
    thrs = np.linspace(0.10, 0.90, 81)
    best_t, best_j = 0.50, -1.0
    for t in thrs:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return float(np.clip(best_t, GLOBAL_MIN_PROB, GLOBAL_MAX_PROB))

def load_ml_assets():
    """
    Loads new bundle:
      - local_models (per-ticker), global_model, stacker
      - features_base, features_regime
      - calibrators (per-ticker) and thresholds[tkr]['chosen']
    Falls back to OOF thresholds if per-ticker thresholds missing.
    """
    global ML_MODELS, ML_FEATURES_BASE, ML_FEATURES_REGIME, ML_FEATURES_GLOBAL, \
           ML_THRESHOLDS, ML_CALIBRATORS, GLOBAL_MODEL_OBJ, STACKER_OBJ

    ML_MODELS, ML_FEATURES_BASE, ML_FEATURES_REGIME, ML_FEATURES_GLOBAL = {}, [], [], []
    ML_THRESHOLDS, ML_CALIBRATORS = {}, {}
    GLOBAL_MODEL_OBJ = None
    STACKER_OBJ = None

    if not _HAS_JOBLIB or not os.path.exists(MODELS_PATH):
        print(f"[ML] Models not available at {MODELS_PATH}")
        return

    payload = joblib.load(MODELS_PATH)

    # New schema (preferred)
    local_models = payload.get("local_models") or payload.get("models") or {}
    ML_FEATURES_BASE = payload.get("features_base") or payload.get("features") or []
    ML_FEATURES_REGIME = payload.get("features_regime") or []
    ML_FEATURES_GLOBAL = list(dict.fromkeys(list(ML_FEATURES_BASE) + list(ML_FEATURES_REGIME)))

    GLOBAL_MODEL_OBJ = payload.get("global_model")
    STACKER_OBJ = payload.get("stacker")
    ML_CALIBRATORS = {str(k).strip().upper(): v for k, v in (payload.get("calibrators") or {}).items()}

    # thresholds: use 'chosen' when dict, else float
    th = payload.get("thresholds") or {}
    for k, v in th.items():
        if isinstance(v, dict) and "chosen" in v:
            ML_THRESHOLDS[str(k).strip().upper()] = float(v["chosen"])
        else:
            try:
                ML_THRESHOLDS[str(k).strip().upper()] = float(v)
            except Exception:
                pass

    # wrap local models
    wrapped = {}
    for key, m in (local_models or {}).items():
        tkey = str(key).strip().upper()
        knd = getattr(m, "kind", None)
        const_p = getattr(m, "const_p", None)
        wrapped[tkey] = PerTickerModelShim(
            obj=m,
            features=ML_FEATURES_BASE,
            kind=(knd or "knn"),
            const_p=const_p
        )
    ML_MODELS = wrapped

    print(
        f"[ML] Loaded local_models={len(ML_MODELS)}, base_feats={len(ML_FEATURES_BASE)}, "
        f"regime_feats={len(ML_FEATURES_REGIME)}, calibrators={len(ML_CALIBRATORS)}, "
        f"thresholds={len(ML_THRESHOLDS)}, global={'yes' if GLOBAL_MODEL_OBJ else 'no'}, "
        f"stacker={'yes' if STACKER_OBJ else 'no'}"
    )

    # thresholds from OOF (fallback)
    if not ML_THRESHOLDS and os.path.exists(OOF_PATH):
        try:
            oof = pd.read_csv(OOF_PATH, parse_dates=["EntryTime"])
            for tkr, g in oof.groupby("Ticker"):
                ML_THRESHOLDS[str(tkr).strip().upper()] = _best_threshold_from_oof_you_dens_j(g)
            print(f"[ML] Backfilled thresholds from OOF for {len(ML_THRESHOLDS)} tickers.")
        except Exception as e:
            print(f"[ML] Failed to compute thresholds from OOF: {e}")


# Load ML on import
load_ml_assets()

# ---------------------------------------------
# 2) Small utilities: signal DB, time parsing, CSV loading
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

def normalize_time(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        raise KeyError("DataFrame missing 'date' column.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d.dropna(subset=["date"], inplace=True)
    if getattr(d["date"].dt, "tz", None) is None:
        d["date"] = d["date"].dt.tz_localize(tz)
    else:
        d["date"] = d["date"].dt.tz_convert(tz)
    d.sort_values("date", inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d

def load_and_normalize_csv(file_path: str, expected_cols=None, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])
    df = pd.read_csv(file_path, low_memory=False)
    if "date" in df.columns:
        df = normalize_time(df, tz)
    else:
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[expected_cols]
    return df

# ---------------------------------------------
# 3) Indicators (only what's needed): Session VWAP
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
    if getattr(dt.dt, "tz", None) is None:
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
# 4) Signal detection (CSV ranges + ML stack)
# ---------------------------------------------

def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set,
    fitted_csv_path: str = FITTED_CSV_MAIN,
    volrel_window: int = 8,
    volrel_min_periods: int = 10,
    debug: bool = DETECTOR_DEBUG_GLOBAL,
    volrel_gate_min: float = VOLREL_GATE_MIN_DEFAULT,
    use_vwap_side_gate: bool = True,
    vwap_long_min: float = VWAP_LONG_MIN,
    vwap_short_max: float = VWAP_SHORT_MAX,
    # --- ML gating ---
    use_ml_filter: bool = USE_ML_FILTER,
    ml_models: Dict[str, PerTickerModelShim] = ML_MODELS,
    ml_features: List[str] = None,  # per-ticker base features
    ml_thresholds: Dict[str, float] = ML_THRESHOLDS,
) -> list:
    import ast

    signals = []
    if df_for_detection.empty:
        return signals

    # ---- Load per-ticker fitted constraints from CSV ----
    def _parse_range_cell(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return (np.nan, np.nan)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            try:
                return (float(x[0]), float(x[1]))
            except Exception:
                return (np.nan, np.nan)
        s = str(x).strip()
        if not s:
            return (np.nan, np.nan)
        try:
            lo, hi = ast.literal_eval(s)
            return (float(lo), float(hi))
        except Exception:
            s = s.strip("[]")
            parts = [p for p in s.split(",") if p.strip() != ""]
            if len(parts) == 2:
                try:
                    return (float(parts[0]), float(parts[1]))
                except Exception:
                    pass
            return (np.nan, np.nan)

    def _to_float(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return np.nan
        try:
            return float(x)
        except Exception:
            try:
                return float(str(x).strip())
            except Exception:
                return np.nan

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
        if side not in ("Bullish", "Bearish"):
            continue
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

    # ---- Derived columns & rolling stats ----
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined.dropna(subset=["date"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["date_only"] = combined["date"].dt.date

    if "VWAP" not in combined.columns:
        combined = calculate_session_vwap(combined)

    # compatibility: if MACD histogram is named 'Histogram' (writer side), map it
    if "MACD_Hist" not in combined.columns and "Histogram" in combined.columns:
        combined["MACD_Hist"] = combined["Histogram"]

    if "rolling_vol_10" in combined.columns:
        rv = combined["rolling_vol_10"]
    elif "rolling_vol10" in combined.columns:
        rv = combined["rolling_vol10"]; combined["rolling_vol_10"] = rv
    else:
        minp = 1 if RELAX_EARLY_VOLREL_MIN_PERIODS else volrel_min_periods
        rv = combined["volume"].rolling(volrel_window, min_periods=minp).mean()
        combined["rolling_vol_10"] = rv

    eps = 1e-9
    combined["StochD"] = combined.get("Stoch_%D", np.nan)
    combined["StochK"] = combined.get("Stoch_%K", np.nan)
    combined["VWAP_Dist"] = combined["close"] / combined["VWAP"] - 1.0
    combined["EMA50_Dist"]  = (combined["close"] / combined["EMA_50"]  - 1.0) if "EMA_50"  in combined.columns else np.nan
    combined["EMA200_Dist"] = (combined["close"] / combined["EMA_200"] - 1.0) if "EMA_200" in combined.columns else np.nan
    combined["SMA20_Dist"]  = (combined["close"] / combined.get("20_SMA", np.nan) - 1.0) if "20_SMA" in combined.columns else np.nan
    combined["VolRel"] = combined["volume"] / (combined["rolling_vol_10"] + eps)
    combined["ATR_Pct"] = (combined["ATR"] / (combined["close"] + eps)) if "ATR" in combined.columns else np.nan
    if {"Upper_Band", "Lower_Band"}.issubset(combined.columns):
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

    # Column map (CSV constraint name → live column)
    colmap = {
        "RSI":"RSI","ADX":"ADX","CCI":"CCI","MFI":"MFI","BBP":"BBP","BandWidth":"BandWidth","ATR_Pct":"ATR_Pct",
        "StochD":"StochD","StochK":"StochK","MACD_Hist":"MACD_Hist","VWAP_Dist":"VWAP_Dist",
        "Daily_Change":"Daily_Change","EMA50_Dist":"EMA50_Dist","EMA200_Dist":"EMA200_Dist",
        "SMA20_Dist":"SMA20_Dist","OBV_Slope10":"OBV_Slope10","VolRel":"VolRel",
    }
    available_cols = set(combined.columns)

    # ---- Range widening helper ----
    def _widen(lo: float, hi: float, key: str) -> (float, float):
        if np.isnan(lo) or np.isnan(hi):
            return (lo, hi)
        width = hi - lo
        if width <= 0:
            pad = abs(lo) * WIDEN_PCT if lo != 0 else WIDEN_PCT
        else:
            pad = width * WIDEN_PCT
        lo2, hi2 = lo - pad, hi + pad
        # clamp some bounded indicators
        if key.startswith("RSI"):
            lo2, hi2 = max(0.0, lo2), min(100.0, hi2)
        if key.startswith("BBP"):
            lo2, hi2 = max(0.0, lo2), min(1.0, hi2)
        return (lo2, hi2)

    def _clean_constraints(raw: dict) -> dict:
        cleaned = {}
        for k, v in raw.items():
            if k.endswith("_range"):
                base = k[:-6]; col = colmap.get(base, base)
                if col not in available_cols:
                    continue
                lo, hi = v if isinstance(v, (list, tuple)) and len(v) == 2 else (np.nan, np.nan)
                if np.isnan(lo) or np.isnan(hi):
                    continue
                lo2, hi2 = _widen(float(lo), float(hi), base)
                cleaned[k] = (lo2, hi2)
            elif k.endswith("_min") or k.endswith("_max"):
                base = k[:-4]; col = colmap.get(base, base)
                if col not in available_cols:
                    continue
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                cleaned[k] = float(v)
        return cleaned

    sides = {side: _clean_constraints(raw) for side, raw in sides_raw.items()}

    # --- Optional alt (permissive union) constraints (both sides) ---
    sides_alt = {}
    try:
        if os.path.exists(FITTED_CSV_ALT):
            cfg_df_b = pd.read_csv(FITTED_CSV_ALT)
            if "Ticker" in cfg_df_b.columns:
                cfg_df_b["__TickerKey"] = cfg_df_b["Ticker"].astype(str).str.strip().str.upper()
                cfg_df_b = cfg_df_b[cfg_df_b["__TickerKey"] == tkey]
                raw_b = {}
                for _, row_b in cfg_df_b.iterrows():
                    side_b = str(row_b.get("Side", "")).strip()
                    if side_b not in ("Bullish", "Bearish"):
                        continue
                    for colb in row_b.index:
                        if colb in ("Ticker", "__TickerKey", "Side", "Samples_wins", "Quantiles_used"):
                            continue
                        if colb.endswith("_range"):
                            raw_b.setdefault(side_b, {})[colb] = _parse_range_cell(row_b[colb])
                        elif colb.endswith("_min") or colb.endswith("_max"):
                            raw_b.setdefault(side_b, {})[colb] = _to_float(row_b[colb])
                sides_alt = {side: _clean_constraints(rc) for side, rc in raw_b.items()}
    except Exception as _e_alt:
        if debug:
            print(f"[detect:{ticker}] alt constraints load failed: {_e_alt}")

    if all(len(cfg) == 0 for cfg in sides.values()):
        if debug: print(f"[detect:{ticker}] All constraints dropped.")
        return signals

    # --- ML helpers for this ticker (new bundle: local + global + stacker + calibrator) ---
    model_local = ml_models.get(tkey)
    base_feats = ml_features if ml_features is not None else ML_FEATURES_BASE
    have_ml = use_ml_filter and (model_local is not None) and bool(base_feats) and (GLOBAL_MODEL_OBJ is not None) and (STACKER_OBJ is not None)

    # Per-ticker threshold (chosen) with freshness decay & aggressive shift
    p_thresh = ml_thresholds.get(tkey, GLOBAL_MIN_PROB)

    if debug:
        print(f"[{ticker}] have_ml={have_ml} base_thr={p_thresh:.3f} ranges: "
              f"Bull={len(sides.get('Bullish',{}))} Bear={len(sides.get('Bearish',{}))}")

    # index finder (align ts to nearest ≤ ts within same day)
    def find_cidx(ts):
        ts = pd.to_datetime(ts)
        exact = combined.index[combined["date"] == ts]
        if len(exact):
            return int(exact[0])
        d = ts.date()
        mask = (combined["date_only"] == d) & (combined["date"] <= ts)
        if mask.any():
            return int(combined.index[mask][-1])
        return None

    def _check_side_constraints(row: pd.Series, side_constraints: dict, *, range_pass_ratio: float) -> bool:
        # quality floors first
        if pd.notna(row.get("ATR_Pct", np.nan)) and row.get("ATR_Pct", 0) < MIN_ATR_PCT:
            return False
        if pd.notna(row.get("BandWidth", np.nan)) and row.get("BandWidth", 0) < MIN_BB_WIDTH:
            return False
        # ranges (relaxed K-of-N)
        total_ranges = 0
        pass_ranges = 0
        for k, rng in side_constraints.items():
            if not k.endswith("_range"):
                continue
            base = k[:-6]; col = colmap.get(base, base)
            x = row.get(col, np.nan); lo, hi = rng
            if np.isnan(x) or np.isnan(lo) or np.isnan(hi):
                continue
            total_ranges += 1
            if lo <= x <= hi:
                pass_ranges += 1
        needed = int(np.ceil(range_pass_ratio * total_ranges)) if total_ranges > 0 else 0
        if total_ranges > 0 and pass_ranges < needed:
            return False
        # mins (strict)
        for k, thr in side_constraints.items():
            if not k.endswith("_min"):
                continue
            base = k[:-4]; col = colmap.get(base, base)
            x = row.get(col, np.nan)
            if pd.isna(x) or not (x >= thr):
                return False
        # maxs (strict)
        for k, thr in side_constraints.items():
            if not k.endswith("_max"):
                continue
            base = k[:-4]; col = colmap.get(base, base)
            x = row.get(col, np.nan)
            if pd.isna(x) or not (x <= thr):
                return False
        return True

    for _, row_det in df_for_detection.iterrows():
        dt = row_det.get("date", None)
        if pd.isna(dt):
            continue

        # --- freshness relaxer: adjust *per row* based on manifest end_date
        p_thresh_adj = p_thresh
        try:
            man_end = _dt.date.fromisoformat(_manifest.get("end_date")) if _manifest else None
            this_day = pd.to_datetime(dt).date()
            if man_end:
                fresh_days = max(0, (this_day - man_end).days)
                # soften up to 2pp (0.020) at 0.5pp/day
                p_thresh_adj = max(GLOBAL_MIN_PROB, p_thresh - min(0.02, 0.005 * fresh_days))
        except Exception:
            pass

        cidx = find_cidx(dt)
        if cidx is None:
            continue

        r = combined.loc[cidx]

        # global participation brake (slightly easier very early session)
        volrel_val = r.get("VolRel", np.nan)
        t_local = r["date"].tz_convert("Asia/Kolkata").time() if hasattr(r["date"], "tzinfo") else r["date"].time()
        vr_gate = volrel_gate_min - 0.05 if t_local <= dt_time(10, 0) else volrel_gate_min
        volrel_ok_global = not (pd.isna(volrel_val) or volrel_val < vr_gate)
        if debug and not volrel_ok_global:
            print(f"[{ticker}] {dt} reject: VolRel={volrel_val:.2f} < {vr_gate:.2f}")

        # side-aware VWAP gate
        allow_bull = allow_bear = True
        vdist = r.get("VWAP_Dist", np.nan)
        if use_vwap_side_gate:
            if pd.isna(vdist):
                continue
            if vdist < vwap_long_min:
                allow_bull = False
            if vdist > vwap_short_max:
                allow_bear = False

        # Build one-row features for ML scoring (if enabled): local (base) + global (base+regime)
        ml_prob: Optional[float] = None
        _have_ml_this_row = have_ml
        if _have_ml_this_row:
            try:
                row_base = {f: r.get(f, np.nan) for f in base_feats}
                p_local = float(model_local.predict_proba(pd.DataFrame([row_base]))[0])

                if GLOBAL_MODEL_OBJ is not None:
                    row_global = {f: r.get(f, np.nan) for f in ML_FEATURES_GLOBAL}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if hasattr(GLOBAL_MODEL_OBJ, "predict_proba"):
                            p_global = float(GLOBAL_MODEL_OBJ.predict_proba(pd.DataFrame([row_global]))[:,1][0])
                        else:
                            p_global = float(np.clip(GLOBAL_MODEL_OBJ.predict(pd.DataFrame([row_global]))[0], 0.0, 1.0))
                else:
                    p_global = p_local  # fallback symmetry

                # Stacker combines local & global producers
                if STACKER_OBJ is not None:
                    S = np.array([[p_local, p_global]], dtype=float)
                    if hasattr(STACKER_OBJ, "predict_proba"):
                        ml_raw = float(STACKER_OBJ.predict_proba(S)[:,1][0])
                    else:
                        ml_raw = float(np.clip(STACKER_OBJ.predict(S)[0], 0.0, 1.0))
                else:
                    ml_raw = p_local

                # Per-ticker isotonic calibration if available
                cal = ML_CALIBRATORS.get(tkey)
                ml_prob = float(np.clip(cal.predict([ml_raw])[0], 0.0, 1.0)) if cal else ml_raw

                if debug:
                    print(f"[{ticker}] {dt} p_loc={p_local:.3f} p_glb={p_global:.3f} p_stk={ml_raw:.3f} p_cal={ml_prob:.3f} "+
                          f"vs thr={p_thresh_adj + LIVE_DEFENSIVE_BUMP + AGGR_PROB_SHIFT:.3f}")
            except Exception as ex:
                _have_ml_this_row = False
                if debug:
                    print(f"[{ticker}] ML failed on row: {ex}")

        # effective threshold after aggressive shift + live bump
        eff_thr = p_thresh_adj + LIVE_DEFENSIVE_BUMP + AGGR_PROB_SHIFT

        def _ml_pass_ok(prob: Optional[float], eff_t: float, *, side: str) -> bool:
            if prob is None:
                return False
            if prob >= eff_t:
                return True
            # near-miss zone with favorable tape
            if prob >= (eff_t - ML_NEAR_MISS):
                favorable = False
                if not pd.isna(volrel_val) and volrel_val >= (vr_gate + 0.10):
                    favorable = True
                # VWAP proximity lenient within ±0.20%
                if not pd.isna(vdist) and (-0.002 <= vdist <= 0.002):
                    favorable = True
                return favorable
            return False

        # ---- Bullish (relaxed K-of-N)
        bull_cfg = sides.get("Bullish")
        if bull_cfg and allow_bull and _check_side_constraints(r, bull_cfg, range_pass_ratio=RANGE_PASS_RATIO):
            if volrel_ok_global:
                if _have_ml_this_row:
                    if _ml_pass_ok(ml_prob, eff_thr, side="Bullish"):
                        sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_CSV_FITTED_ML"
                        if sid not in existing_signal_ids:
                            signals.append({
                                "Ticker": ticker, "date": dt, "Entry Type": "CSVFittedFilter+ML",
                                "Trend Type": "Bullish", "Price": round(float(r.get("close", np.nan)), 2),
                                "Signal_ID": sid, "logtime": row_det.get("logtime", datetime.now().isoformat()),
                                "Entry Signal": "Yes", "ML_Prob": round(ml_prob, 4), "ML_Thresh": round(eff_thr, 4),
                            })
                            existing_signal_ids.add(sid)
                else:
                    sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_CSV_FITTED"
                    if sid not in existing_signal_ids:
                        signals.append({
                            "Ticker": ticker, "date": dt, "Entry Type": "CSVFittedFilter",
                            "Trend Type": "Bullish", "Price": round(float(r.get("close", np.nan)), 2),
                            "Signal_ID": sid, "logtime": row_det.get("logtime", datetime.now().isoformat()),
                            "Entry Signal": "Yes",
                        })
                        existing_signal_ids.add(sid)

        # ---- Bearish (relaxed K-of-N)
        bear_cfg = sides.get("Bearish")
        if bear_cfg and allow_bear and _check_side_constraints(r, bear_cfg, range_pass_ratio=RANGE_PASS_RATIO):
            if volrel_ok_global:
                if _have_ml_this_row:
                    if _ml_pass_ok(ml_prob + ML_SOFT_MARGIN_BEAR if ml_prob is not None else None, eff_thr, side="Bearish"):
                        sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_CSV_FITTED_ML"
                        if sid not in existing_signal_ids:
                            signals.append({
                                "Ticker": ticker, "date": dt, "Entry Type": "CSVFittedFilter+ML",
                                "Trend Type": "Bearish", "Price": round(float(r.get("close", np.nan)), 2),
                                "Signal_ID": sid, "logtime": row_det.get("logtime", datetime.now().isoformat()),
                                "Entry Signal": "Yes", "ML_Prob": round(ml_prob, 4) if ml_prob is not None else np.nan,
                                "ML_Thresh": round(eff_thr, 4),
                            })
                            existing_signal_ids.add(sid)
                else:
                    sid = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_CSV_FITTED"
                    if sid not in existing_signal_ids:
                        signals.append({
                            "Ticker": ticker, "date": dt, "Entry Type": "CSVFittedFilter",
                            "Trend Type": "Bearish", "Price": round(float(r.get("close", np.nan)), 2),
                            "Signal_ID": sid, "logtime": row_det.get("logtime", datetime.now().isoformat()),
                            "Entry Signal": "Yes",
                        })
                        existing_signal_ids.add(sid)

        # ---- Permissive UNION (alt file) — Bear
        bear_cfg_b = sides_alt.get("Bearish") if isinstance(sides_alt, dict) else None
        if bear_cfg_b and allow_bear:
            if pd.isna(volrel_val) or (volrel_val >= DETECTOR_ALT_VOLREL_MIN):
                if _check_side_constraints(r, bear_cfg_b, range_pass_ratio=ALT_RANGE_PASS_RATIO):
                    ok_ml_b = True
                    if _have_ml_this_row:
                        ok_ml_b = _ml_pass_ok(ml_prob + ML_SOFT_MARGIN_BEAR if ml_prob is not None else None, eff_thr, side="Bearish")
                    if ok_ml_b:
                        sid_b = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_CSV_FITTED_PERM" + ("_ML" if _have_ml_this_row else "")
                        if sid_b not in existing_signal_ids:
                            signals.append({
                                "Ticker": ticker, "date": dt,
                                "Entry Type": "CSVFittedFilter_PERMISSIVE" + ("+ML" if _have_ml_this_row else ""),
                                "Trend Type": "Bearish",
                                "Price": round(float(r.get("close", np.nan)), 2),
                                "Signal_ID": sid_b,
                                "logtime": row_det.get("logtime", datetime.now().isoformat()),
                                "Entry Signal": "Yes",
                                "ML_Prob": (round(ml_prob, 4) if ml_prob is not None else np.nan),
                                "ML_Thresh": round(eff_thr, 4) if _have_ml_this_row else np.nan,
                            })
                            existing_signal_ids.add(sid_b)

        # ---- Permissive UNION (alt file) — Bull
        bull_cfg_b = sides_alt.get("Bullish") if isinstance(sides_alt, dict) else None
        if bull_cfg_b and allow_bull:
            if pd.isna(volrel_val) or (volrel_val >= DETECTOR_ALT_VOLREL_MIN):
                if _check_side_constraints(r, bull_cfg_b, range_pass_ratio=ALT_RANGE_PASS_RATIO):
                    ok_ml_bu = True
                    if _have_ml_this_row:
                        ok_ml_bu = _ml_pass_ok(ml_prob, eff_thr, side="Bullish")
                    if ok_ml_bu:
                        sid_bu = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_CSV_FITTED_PERM" + ("_ML" if _have_ml_this_row else "")
                        if sid_bu not in existing_signal_ids:
                            signals.append({
                                "Ticker": ticker, "date": dt,
                                "Entry Type": "CSVFittedFilter_PERMISSIVE" + ("+ML" if _have_ml_this_row else ""),
                                "Trend Type": "Bullish",
                                "Price": round(float(r.get("close", np.nan)), 2),
                                "Signal_ID": sid_bu,
                                "logtime": row_det.get("logtime", datetime.now().isoformat()),
                                "Entry Signal": "Yes",
                                "ML_Prob": (round(ml_prob, 4) if ml_prob is not None else np.nan),
                                "ML_Thresh": round(eff_thr, 4) if _have_ml_this_row else np.nan,
                            })
                            existing_signal_ids.add(sid_bu)

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
                df_main.loc[mask, "Entry Signal"] = "Yes"
                df_main.loc[mask, "CalcFlag"] = "Yes"
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
    DETECTOR_VOLREL_MIN_PERIODS = 10
    DETECTOR_DEBUG = DETECTOR_DEBUG_GLOBAL

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    print(f"[Detector] Found {len(files)} CSVs under {INDICATORS_DIR}")
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    # Path sanity
    if not os.path.exists(FITTED_CSV_MAIN) and os.path.exists(FITTED_CSV_ALT):
        print(f"[Detector] Primary ranges CSV missing; falling back to: {FITTED_CSV_ALT}")

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def process_file(file_path: str) -> list:
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()
        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df_full.empty:
            if DETECTOR_DEBUG:
                print(f"[{ticker}] file loaded but empty.")
            return []
        df_today = df_full[(df_full["date"] >= st) & (df_full["date"] <= et)].copy()
        if df_today.empty:
            if DETECTOR_DEBUG:
                print(f"[{ticker}] no rows between {st.time()}–{et.time()} (IST).")
            return []
        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,
            df_for_detection=df_today,
            existing_signal_ids=existing_ids,
            fitted_csv_path=FITTED_CSV_MAIN if os.path.exists(FITTED_CSV_MAIN) else FITTED_CSV_ALT,
            volrel_window=DETECTOR_VOLREL_WINDOW,
            volrel_min_periods=DETECTOR_VOLREL_MIN_PERIODS,
            debug=DETECTOR_DEBUG,
            use_ml_filter=USE_ML_FILTER,
            ml_models=ML_MODELS,
            ml_features=ML_FEATURES_BASE,
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
# 7) Build papertrade CSV for a date (first signal per ticker)
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
        qty = int(100000 / price)
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
                df = df.sort_values(["Ticker","ML_Prob","date"], ascending=[True, False, True]).groupby("Ticker", as_index=False).head(1)
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
        papertrade_file = os.path.join(BASE_DIR, f"papertrade_5min_v7_relaxed_{date_label}.csv")

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
    # Quick visibility on files present
    file_count = len(glob.glob(os.path.join(INDICATORS_DIR, "*_main_indicators.csv")))
    print(f"[Startup] Indicators CSVs present: {file_count}")

    # If primary ranges CSV missing but manifest exists, shout
    if not os.path.exists(FITTED_CSV_MAIN):
        print(f"[Startup] WARNING: Ranges CSV not found at {FITTED_CSV_MAIN}")

    date_args = sys.argv[1:]
    if not date_args:
        # Example: backfill last few dates when run without args
        date_args = [
            "2025-09-04"
        ]
    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
