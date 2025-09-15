# -*- coding: utf-8 -*-
"""
Per-ticker ML scorer for 'target hit' with robust CV.

- Recomputes HitTarget from ReturnPct & Side:
    Bullish  => hit if ReturnPct >= +2.0
    Bearish  => hit if ReturnPct <= -2.0
  (ReturnPct treated as numeric percent; '2.3' == +2.3%)

- Builds a separate model per Ticker:
    * Pipeline: SimpleImputer(median) -> StandardScaler -> KNN Regressor
    * Grouped CV by DateOnly (no same-day leakage)
    * Handles single-class folds & tiny samples gracefully
    * Produces OOF predictions and refits a final model per ticker

Outputs (in working directory):
    - oof_predictions.csv
    - per_ticker_models.joblib   (optional; comment out if not needed)
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

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

# -----------------------
# Config
# -----------------------
DATA_PATH = "entries_2pct_results.csv"   # <- change if needed
SAVE_MODELS = True
MODELS_PATH = "per_ticker_models.joblib"
OOF_PATH = "oof_predictions.csv"

# Features to use (pick any available subset from your CSV)
FEATURES = [
    "RSI","ADX","StochD","StochK","VWAP_Dist","VolRel","Daily_Change","Intra_Change",
    "EMA50_Dist","EMA200_Dist","SMA20_Dist","ATR_Pct","BBP","BandWidth",
    "MACD_Hist","CCI","MFI","OBV_Slope10"
]

# CV settings
MAX_SPLITS = 5           # up to 5 date-based folds (will shrink if fewer dates)
MIN_SPLITS = 2
MAX_NEIGHBORS = 25       # cap for K in KNN
MIN_NEIGHBORS = 3        # floor for K in KNN

# -----------------------
# Utilities
# -----------------------
def _to_float_pct(x):
    """Parse ReturnPct robustly; accepts raw floats/strings with or without %."""
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip().rstrip("%")
        return float(s)
    except Exception:
        return np.nan

def recompute_hit(df: pd.DataFrame) -> pd.Series:
    """HitTarget rule: Bullish >= +2.0, Bearish <= -2.0 (percent units)."""
    rp = df["ReturnPct"].apply(_to_float_pct)
    side = df["Side"].astype(str).str.strip().str.lower()
    hit = ( (side == "bullish") & (rp >= 2.0) ) | ( (side == "bearish") & (rp <= -2.0) )
    return hit.astype(int)  # 1/0 labels

def pick_cv_splits(X, y, groups, n_splits: int):
    """Prefer StratifiedGroupKFold (if available & both classes present), else GroupKFold."""
    if HAS_SGK and len(np.unique(y)) >= 2 and len(np.unique(groups)) >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return cv.split(X, y, groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(X, y, groups)

def safe_k(n_train: int) -> int:
    """Choose a KNN neighborhood size safe for the fold size."""
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
        # ensure consistent column order & presence
        for c in self.features:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.features]
        if self.kind == "constant":
            return np.full((len(X),), float(self.const_p), dtype=float)
        return np.clip(self.pipeline.predict(X), 0.0, 1.0)

# -----------------------
# Load & prep data
# -----------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if df.empty:
    raise ValueError("CSV is empty.")

# Basic hygiene
if "EntryTime" in df.columns:
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], errors="coerce")
if "DateOnly" in df.columns:
    # keep as string-like grouping key for stability
    df["DateOnly"] = df["DateOnly"].astype(str)
else:
    # derive DateOnly if missing
    if "EntryTime" in df.columns:
        df["DateOnly"] = df["EntryTime"].dt.date.astype(str)
    else:
        df["DateOnly"] = "NA"

# Recompute target per your rule
df["HitTarget_bin"] = recompute_hit(df)

# Keep only features that exist
avail_feats = [c for c in FEATURES if c in df.columns]
if not avail_feats:
    raise ValueError("None of the requested FEATURES are present in the CSV.")
FEATURES = avail_feats

# Filter out rows missing Side/Ticker or groups
df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
df["Side"] = df["Side"].astype(str).str.strip().str.capitalize()
df = df.dropna(subset=["Ticker","Side","DateOnly"]).reset_index(drop=True)

# -----------------------
# Train per-ticker
# -----------------------
all_oof = []  # collect OOF predictions for evaluation/export
models: Dict[str, PerTickerModel] = {}

tickers = sorted(df["Ticker"].unique())
print(f"Tickers to model: {len(tickers)}")

for tkr in tickers:
    dft = df[df["Ticker"] == tkr].reset_index(drop=True)
    y = dft["HitTarget_bin"].to_numpy(dtype=int)
    groups = dft["DateOnly"].to_numpy()

    # Feature matrix (ensure column presence/order)
    Xt = dft[FEATURES].copy()
    for c in FEATURES:
        if c not in Xt.columns:
            Xt[c] = np.nan
    Xt = Xt[FEATURES].to_numpy(dtype=float)

    n = len(dft)
    uniq_dates = np.unique(groups).size
    n_splits = max(MIN_SPLITS, min(MAX_SPLITS, uniq_dates))

    # Degenerate ticker: no variance in label at all
    if np.unique(y).size < 2 or n < 10 or uniq_dates < 2:
        # Smoothed constant prob
        p_const = float((y.mean() * n + 0.5) / (n + 1.0))
        models[tkr] = PerTickerModel(kind="constant", const_p=p_const, features=FEATURES)
        oof_pred = np.full(n, p_const, dtype=float)
        all_oof.append(pd.DataFrame({
            "Ticker": tkr, "EntryTime": dft.get("EntryTime", pd.NaT),
            "Side": dft["Side"], "HitTarget": y, "PredProb": oof_pred, "Fold": -1
        }))
        print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  -> CONSTANT p={p_const:.3f} (degenerate or tiny)")
        continue

    # Out-of-fold predictions container
    oof_pred = np.zeros(n, dtype=float)
    fold_ids = np.full(n, -1, dtype=int)

    # Build CV splits
    splits = list(pick_cv_splits(Xt, y, groups, n_splits=n_splits))
    if len(splits) < 2:
        # safety: use constant if we somehow can't split
        p_const = float((y.mean() * n + 0.5) / (n + 1.0))
        models[tkr] = PerTickerModel(kind="constant", const_p=p_const, features=FEATURES)
        oof_pred[:] = p_const
        fold_ids[:] = -1
        print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  -> CONSTANT (couldn't split CV)")
    else:
        # Manual CV loop with per-fold K and pipeline
        for fold, (tr, te) in enumerate(splits, start=1):
            Xtr, Xte = Xt[tr], Xt[te]
            ytr = y[tr]

            # Build pipeline fresh per fold
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

        # Final refit on ALL data with data-dependent K
        k_full = safe_k(n)
        final_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("knn", KNeighborsRegressor(n_neighbors=k_full, weights="distance", p=2)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipe.fit(Xt, y.astype(float))

        models[tkr] = PerTickerModel(kind="knn", pipeline=final_pipe, features=FEATURES)

        # Metrics (only meaningful if both classes in OOF)
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

        # Top-20% hit rate (ranking utility)
        topk = max(1, int(0.2 * n))
        idx_top = np.argsort(-oof_pred)[:topk]
        hr_top = y[idx_top].mean() if topk > 0 else np.nan

        print(f"  {tkr:<12} n={n:4d} dates={uniq_dates:3d}  folds={len(splits)} | "
              f"AUC={auc:.3f}  Brier={brier:.3f}  Top20%HR={hr_top:.3f}")

    # collect OOF for this ticker
    all_oof.append(pd.DataFrame({
        "Ticker": tkr,
        "EntryTime": dft.get("EntryTime", pd.NaT),
        "Side": dft["Side"],
        "HitTarget": y,
        "PredProb": oof_pred,
        "Fold": fold_ids
    }))

# -----------------------
# Save outputs
# -----------------------
oof_df = pd.concat(all_oof, ignore_index=True)
oof_df.to_csv(OOF_PATH, index=False)
print(f"\nSaved OOF predictions → {OOF_PATH} (rows={len(oof_df)})")

if SAVE_MODELS:
    joblib.dump({"models": models, "features": FEATURES}, MODELS_PATH)
    print(f"Saved per-ticker models → {MODELS_PATH}")

# -----------------------
# Example: score a few rows (optional)
# -----------------------
# To score new data for a given ticker:
# new_df must contain the same feature columns; missing ones are allowed (imputed).
# tkr = tickers[0]
# sample = df[df["Ticker"] == tkr].head(5)[FEATURES]
# preds = models[tkr].predict_proba(sample)
# print(f"\nSample preds for {tkr}:\n", preds)
