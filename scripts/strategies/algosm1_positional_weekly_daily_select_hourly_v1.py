# -*- coding: utf-8 -*-
"""
Multi-timeframe positional LONG backtest framework (HOURLY execution, WEEKLY context).

CHANGES (as requested):
  - REMOVED all daily data usage (no main_indicators_daily anywhere).
  - Uses 1H data from: main_indicators_1h/{ticker}_main_indicators_1h.csv
  - All functions updated to operate on HOURLY bars (suffix _H).
  - Weekly context kept as before (suffix _W) and merged via as-of join (with +1 day shift to avoid lookahead).
  - Evaluation is now on HOURLY data (no daily evaluation).

Keeps your added features:
  1) Prints date range of analysis/evaluation
  2) Prints time-to-target stats (hits only) -> computed in DAYS (fractional) from hourly timestamps
  3) Prints signals-per-day stats
  4) De-duplicated per ticker/day signals (keeps earliest per IST date)
  5) Position sizing rule (qty=1 if entry_price > CAPITAL_PER_SIGNAL_RS else floor(capital/price))
"""

import os
import glob
import sys
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pytz


# ----------------------- CONFIG -----------------------
DIR_H   = "main_indicators_1h"       # <-- HOURLY ONLY
DIR_W   = "main_indicators_weekly"   # weekly context stays
OUT_DIR = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- AS-OF CUT (IMPORTANT) ----
NOW_IST = datetime.now(IST)

# last completed "hour end" timestamp bucket:
# Example: at 16:07, cutoff_end = 16:00; we include bars with date <= 16:00.
CUTOFF_END_IST = NOW_IST.replace(minute=0, second=0, microsecond=0)

# Target configuration
TARGET_PCT: float = 8.0
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0

def _pct_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")

TARGET_TAG   = _pct_tag(TARGET_PCT)
TARGET_LABEL = f"{TARGET_PCT:.2f}%"

HIT_COL      = f"hit_{TARGET_TAG}"
TIME_TD_COL  = f"time_to_{TARGET_TAG}_days"
TIME_CAL_COL = f"days_to_{TARGET_TAG}_calendar"
EVAL_SUFFIX  = f"{TARGET_TAG}_eval"

# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 15000.0
MAX_CAPITAL_RS: float = 10000000.0
N_RECENT_SKIP = 500


# --- Separate TP/SL analysis globals ---
target_bew: float = 6.0       # take-profit %
stoploss_new: float = -4.0    # stop-loss % (negative for long)

# ----------------------- STRATEGY SELECTION -----------------------
DEFAULT_STRATEGY_KEYS: list[str] = [
    "custom10", "custom09", "custom08", "custom07", "custom06",
    "custom05", "custom04", "custom03", "custom02", "custom01",
]


# ----------------------- HELPERS (COMMON) -----------------------
def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """Read CSV, parse 'date' as tz-aware IST, sort ascending."""
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")

        # Your files typically store UTC timestamps; keep same pattern:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(tz)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()


def _prepare_tf_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Append suffix to all columns except 'date'."""
    if df.empty:
        return df
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)


def _list_tickers_from_dir(directory: str) -> set[str]:
    files = glob.glob(os.path.join(directory, "*_main_indicators*.csv"))
    tickers = set()
    for f in files:
        base = os.path.basename(f)
        if "_main_indicators" in base:
            t = base.split("_main_indicators")[0].strip()
            if t:
                tickers.add(t)
    return tickers


def _find_file_hourly(directory: str, ticker: str) -> Optional[str]:
    """Strictly use {ticker}_main_indicators_1h.csv if present, else fallback to pattern."""
    strict = os.path.join(directory, f"{ticker}_main_indicators_1h.csv")
    if os.path.exists(strict):
        return strict
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators_1h*.csv"))
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None


def _find_file_weekly(directory: str, ticker: str) -> Optional[str]:
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None


# ======================================================================
#                 COMMON MTF LOADER (HOURLY + WEEKLY)
# ======================================================================
def _apply_asof_cut_inclusive(df: pd.DataFrame, cutoff_dt_ist: datetime) -> pd.DataFrame:
    """Keep rows <= cutoff_dt_ist (IST). Intended for completed-hour bucket end timestamps."""
    if df.empty or "date" not in df.columns:
        return df
    return df[df["date"] <= cutoff_dt_ist].copy()


def _load_mtf_context_hourly_weekly(ticker: str) -> pd.DataFrame:
    """
    Load Hourly (1H) + Weekly, suffix columns (_H, _W), merge Weekly onto Hourly via as-of join.

    AS-OF LOGIC:
      - Hourly: rows <= CUTOFF_END_IST (up to last completed hour bucket)
      - Weekly: rows <= CUTOFF_END_IST, shift weekly date +1 day,
        then merge_asof(direction="backward") to avoid same-week lookahead.

    Hourly prev-change:
      create Bar_Change_prev = Bar_Change.shift(1) BEFORE suffixing => Bar_Change_prev_H
    """
    path_h = _find_file_hourly(DIR_H, ticker)
    path_w = _find_file_weekly(DIR_W, ticker)

    if not path_h or not path_w:
        return pd.DataFrame()

    df_h = _safe_read_tf(path_h)
    df_w = _safe_read_tf(path_w)
    if df_h.empty or df_w.empty:
        return pd.DataFrame()

    df_h = _apply_asof_cut_inclusive(df_h, CUTOFF_END_IST)
    df_w = _apply_asof_cut_inclusive(df_w, CUTOFF_END_IST)
    if df_h.empty or df_w.empty:
        return pd.DataFrame()

    # Build a robust "bar change" series for hourly:
    # Preference: Intra_Change (if present) else Daily_Change (if present) else computed from close.
    if "Intra_Change" in df_h.columns:
        df_h["Bar_Change"] = pd.to_numeric(df_h["Intra_Change"], errors="coerce")
    elif "Daily_Change" in df_h.columns:
        df_h["Bar_Change"] = pd.to_numeric(df_h["Daily_Change"], errors="coerce")
    elif "close" in df_h.columns:
        c = pd.to_numeric(df_h["close"], errors="coerce")
        df_h["Bar_Change"] = (c / c.shift(1) - 1.0) * 100.0
    else:
        df_h["Bar_Change"] = np.nan

    df_h["Bar_Change_prev"] = df_h["Bar_Change"].shift(1)

    df_h = _prepare_tf_with_suffix(df_h, "_H")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Avoid same-week lookahead: weekly becomes "available" next day
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(days=1)

    df_hw = pd.merge_asof(
        df_h.sort_values("date"),
        df_w_sorted.sort_values("date"),
        on="date",
        direction="backward",
    )
    return df_hw.reset_index(drop=True)


# ======================================================================
#                 STRATEGY TEMPLATE (PARAMETRIC)
# ======================================================================
def _build_signals_positional_template(ticker: str, P: dict) -> list[dict]:
    df = _load_mtf_context_hourly_weekly(ticker)
    if df.empty:
        return []

    cols = set(df.columns)
    need = {"date", "open_H", "high_H", "low_H", "close_H"}
    if not need.issubset(cols):
        return []

    def _col(name: str, default=np.nan):
        return df[name] if name in cols else pd.Series(default, index=df.index)

    close = pd.to_numeric(df["close_H"], errors="coerce")

    # ---- tradability filters ----
    if P.get("min_price") is not None:
        tradable_ok = (close >= float(P["min_price"])).fillna(False)
    else:
        tradable_ok = pd.Series(True, index=df.index)

    vol = pd.to_numeric(_col("volume_H"), errors="coerce")
    if ("volume_H" in cols) and (P.get("min_vol_sma20") is not None):
        vol_sma20 = vol.rolling(20, min_periods=20).mean()
        tradable_ok &= (vol_sma20 >= float(P["min_vol_sma20"])).fillna(False)

    # ---- weekly filter ----
    weekly_ok = pd.Series(True, index=df.index)

    if P.get("wk_close_above_ema200", True) and {"close_W", "EMA_200_W"}.issubset(cols):
        weekly_ok &= (pd.to_numeric(df["close_W"], errors="coerce") > pd.to_numeric(df["EMA_200_W"], errors="coerce"))

    if P.get("wk_close_above_ema50", False) and {"close_W", "EMA_50_W"}.issubset(cols):
        weekly_ok &= (pd.to_numeric(df["close_W"], errors="coerce") > pd.to_numeric(df["EMA_50_W"], errors="coerce"))

    if P.get("wk_ema50_over_ema200_min") is not None and {"EMA_50_W", "EMA_200_W"}.issubset(cols):
        weekly_ok &= (
            pd.to_numeric(df["EMA_50_W"], errors="coerce")
            >= pd.to_numeric(df["EMA_200_W"], errors="coerce") * float(P["wk_ema50_over_ema200_min"])
        )

    if P.get("wk_rsi_min") is not None and ("RSI_W" in cols):
        weekly_ok &= (pd.to_numeric(df["RSI_W"], errors="coerce") >= float(P["wk_rsi_min"]))

    if P.get("wk_adx_min") is not None and ("ADX_W" in cols):
        weekly_ok &= (pd.to_numeric(df["ADX_W"], errors="coerce") >= float(P["wk_adx_min"]))

    weekly_ok = weekly_ok.fillna(False)

    # ---- previous BAR change filter (replaces "daily prev") ----
    # Uses Bar_Change_prev_H (built in loader)
    bprev = pd.to_numeric(_col("Bar_Change_prev_H"), errors="coerce")
    prevbar_ok = pd.Series(True, index=df.index)

    if P.get("dprev_min") is not None:
        prevbar_ok &= (bprev >= float(P["dprev_min"]))
    if P.get("dprev_max") is not None:
        prevbar_ok &= (bprev <= float(P["dprev_max"]))

    prev_close = close.shift(1)
    ema20 = pd.to_numeric(_col("EMA_20_H"), errors="coerce")
    ema50 = pd.to_numeric(_col("EMA_50_H"), errors="coerce")

    if P.get("prev_close_above_ema20", False) and ("EMA_20_H" in cols):
        prevbar_ok &= (prev_close >= ema20.shift(1))

    if P.get("prev_close_above_ema50", False) and ("EMA_50_H" in cols):
        prevbar_ok &= (prev_close >= ema50.shift(1))

    prevbar_ok = prevbar_ok.fillna(False)

    # ---- breakout (now on HOURLY highs) ----
    high = pd.to_numeric(df["high_H"], errors="coerce")
    breakout_windows = P.get("breakout_windows", [20, 50, 100])
    prior_levels = [high.rolling(int(w), min_periods=int(w)).max().shift(1) for w in breakout_windows]

    breakout_level = prior_levels[0].copy()
    name_parts = [f"RH{breakout_windows[0]}"]
    for i in range(1, len(prior_levels)):
        fb = breakout_level.isna()
        breakout_level[fb] = prior_levels[i][fb]
        name_parts.append(f"RH{breakout_windows[i]}")
    breakout_name = "PRIOR_" + "_or_".join(name_parts)

    close_mult = float(P.get("breakout_close_mult", 1.0015))
    wick_mult_hi = float(P.get("breakout_wick_mult_hi", 1.0010))
    wick_mult_close = float(P.get("breakout_wick_mult_close", 1.0005))

    close_break = close >= breakout_level * close_mult
    wick_break  = (high >= breakout_level * wick_mult_hi) & (close >= breakout_level * wick_mult_close)
    breakout_ok = (close_break | wick_break).fillna(False)

    # ---- candle quality ----
    open_ = pd.to_numeric(df["open_H"], errors="coerce")
    green = (close > open_).fillna(False)
    body_min_pct = float(P.get("body_min_pct", 0.0040))
    body_ok = ((close - open_) >= (open_ * body_min_pct)).fillna(False)

    # ---- momentum boosters (HIGH-MOMENTUM: fewer, stronger) ----
    boosters_required = int(P.get("boosters_required", 5))

    # RSI: must be high AND rising (no OR)
    rsi = pd.to_numeric(_col("RSI_H"), errors="coerce")
    rsi_prev = rsi.shift(1)
    rsi_min = float(P.get("rsi_min", 60.0))
    rsi_rise_min = float(P.get("rsi_rise_min", 0.20))  # min rise vs prev 1H bar
    rsi_boost = ((rsi >= rsi_min) & ((rsi - rsi_prev) >= rsi_rise_min)).fillna(False)

    # MACD Hist: must be positive AND rising/accelerating
    macd = pd.to_numeric(_col("MACD_Hist_H"), errors="coerce")
    macd_prev = macd.shift(1)
    macd_hist_min = float(P.get("macd_hist_min", 0.0))
    macd_hist_mult = float(P.get("macd_hist_mult", 1.02))
    macd_rise_min = float(P.get("macd_rise_min", 0.0))
    macd_boost = (
        (macd >= macd_hist_min)
        & (macd > 0.0)
        & (macd_prev.notna())
        & (macd >= macd_prev * macd_hist_mult)
        & ((macd - macd_prev) >= macd_rise_min)
    ).fillna(False)

    # EMA20 slope: require real upward slope over last N hourly bars
    ema20 = pd.to_numeric(_col("EMA_20_H"), errors="coerce")
    slope_lb = int(P.get("ema20_slope_lookback", 6))
    slope_min_pct = float(P.get("ema20_slope_min_pct", 0.25))  # % over lookback
    ema20_slope_pct = (ema20 / ema20.shift(slope_lb) - 1.0) * 100.0
    ema_slope_boost = (ema20_slope_pct >= slope_min_pct).fillna(False)

    # EMA stack: prefer 20 > 50 > 200 (strong trend quality)
    ema_stack_boost = pd.Series(False, index=df.index)
    if P.get("require_ema_stack", True) and {"EMA_20_H", "EMA_50_H", "EMA_200_H"}.issubset(cols):
        ema50 = pd.to_numeric(_col("EMA_50_H"), errors="coerce")
        ema200 = pd.to_numeric(_col("EMA_200_H"), errors="coerce")
        ema_stack_boost = ((ema20 > ema50) & (ema50 > ema200)).fillna(False)
    elif P.get("require_ema_stack_20_50", False) and {"EMA_20_H", "EMA_50_H"}.issubset(cols):
        ema50 = pd.to_numeric(_col("EMA_50_H"), errors="coerce")
        ema_stack_boost = (ema20 > ema50).fillna(False)

    # Continuation: two higher closes (filters noisy single-bar spikes)
    higher_close_boost = ((close > close.shift(1)) & (close.shift(1) > close.shift(2))).fillna(False)

    # Volume expansion
    vol_mult = float(P.get("vol_mult", 1.6))
    if "volume_H" in cols:
        vol_mean_20 = vol.rolling(20, min_periods=20).mean()
        vol_boost = (vol >= (vol_mean_20 * vol_mult)).fillna(False)
    else:
        vol_boost = pd.Series(False, index=df.index)

    # Strong 1H bar % move (your "day_change_min" concept but hourly)
    bar_chg = pd.to_numeric(_col("Bar_Change_H"), errors="coerce").fillna(0.0)
    bar_change_min = float(P.get("bar_change_min", P.get("day_change_min", 1.5)))
    bar_boost = (bar_chg >= bar_change_min).fillna(False)

    boosters = (
        rsi_boost.astype(int)
        + macd_boost.astype(int)
        + ema_slope_boost.astype(int)
        + ema_stack_boost.astype(int)
        + higher_close_boost.astype(int)
        + vol_boost.astype(int)
        + bar_boost.astype(int)
    )
    momentum_ok = boosters >= boosters_required


    # ---- trend + anti-chase ----
    above_ema20 = pd.Series(True, index=df.index)
    above_ema50 = pd.Series(True, index=df.index)
    not_too_far = pd.Series(True, index=df.index)

    if P.get("require_close_above_ema20", True) and ("EMA_20_H" in cols):
        above_ema20 = (close >= ema20).fillna(False)
        anti_mult = float(P.get("anti_chase_ema20_mult", 1.05))
        not_too_far = (close <= ema20 * anti_mult).fillna(False)

    if P.get("require_close_above_ema50", False) and ("EMA_50_H" in cols):
        above_ema50 = (close >= ema50).fillna(False)

    if P.get("d_adx_min") is not None and ("ADX_H" in cols):
        adx_h_ok = (pd.to_numeric(df["ADX_H"], errors="coerce") >= float(P["d_adx_min"])).fillna(False)
    else:
        adx_h_ok = pd.Series(True, index=df.index)

    exec_ok = (
        tradable_ok
        & breakout_ok
        & green & body_ok
        & momentum_ok
        & above_ema20 & above_ema50
        & not_too_far
        & adx_h_ok
    )

    long_mask = weekly_ok & prevbar_ok & exec_ok
    if not long_mask.any():
        return []

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_H"]),
            "strategy": P.get("name", "CUSTOMXX"),
            "sub_strategy": (
                f"W(prev_week)_strict + PrevBarChange + {breakout_name} + "
                f"boosters>={boosters_required} + anti_chase"
            ),
        })
    return signals

def make_strategies_high_momentum(params: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for k, P in params.items():
        Q = dict(P)

        # Require more confirmations (since we added more boosters)
        base_req = int(Q.get("boosters_required", 4))
        Q["boosters_required"] = max(base_req + 1, 5)

        # Stronger momentum thresholds
        Q["rsi_min"] = max(float(Q.get("rsi_min", 58.0)), 60.0)
        Q["rsi_rise_min"] = max(float(Q.get("rsi_rise_min", 0.20)), 0.20)

        Q["macd_hist_mult"] = max(float(Q.get("macd_hist_mult", 1.01)), 1.02)
        Q["macd_rise_min"] = max(float(Q.get("macd_rise_min", 0.0)), 0.0)

        Q["vol_mult"] = max(float(Q.get("vol_mult", 1.5)), 1.6)
        Q["bar_change_min"] = max(float(Q.get("bar_change_min", Q.get("day_change_min", 1.3))), 1.5)

        # Trend quality
        Q["ema20_slope_lookback"] = int(Q.get("ema20_slope_lookback", 6))
        Q["ema20_slope_min_pct"] = max(float(Q.get("ema20_slope_min_pct", 0.25)), 0.25)
        Q["require_ema_stack"] = True

        # Keep anti-chase tight (avoid late entries)
        if Q.get("anti_chase_ema20_mult") is not None:
            Q["anti_chase_ema20_mult"] = min(float(Q["anti_chase_ema20_mult"]), 1.045)

        # Prefer price above EMA50 as well
        Q["require_close_above_ema50"] = True

        out[k] = Q
    return out

def make_strategies_more_strict(params: dict[str, dict]) -> dict[str, dict]:
    """
    Tighten thresholds to reduce signals.
    """
    out = {}
    for k, P in params.items():
        Q = dict(P)

        # Require more confirmations
        if Q.get("boosters_required") is not None:
            Q["boosters_required"] = min(5, int(Q["boosters_required"]) + 1)

        # Stronger momentum
        if Q.get("rsi_min") is not None:
            Q["rsi_min"] = float(Q["rsi_min"]) + 2.0
        if Q.get("rsi_delta_min") is not None:
            Q["rsi_delta_min"] = float(Q["rsi_delta_min"]) + 0.2

        # Stronger breakout + candle body
        if Q.get("breakout_close_mult") is not None:
            Q["breakout_close_mult"] = float(Q["breakout_close_mult"]) + 0.0005
        if Q.get("body_min_pct") is not None:
            Q["body_min_pct"] = float(Q["body_min_pct"]) + 0.0007

        # Volume stricter
        if Q.get("vol_mult") is not None:
            Q["vol_mult"] = float(Q["vol_mult"]) + 0.20

        # Hourly bar-change stricter
        if Q.get("day_change_min") is not None:
            Q["day_change_min"] = float(Q["day_change_min"]) + 0.30

        # Weekly trend stricter
        if Q.get("wk_rsi_min") is not None:
            Q["wk_rsi_min"] = float(Q["wk_rsi_min"]) + 2.0
        if Q.get("wk_adx_min") is not None:
            Q["wk_adx_min"] = float(Q["wk_adx_min"]) + 2.0
        else:
            # If not set, add a baseline weekly ADX filter
            Q["wk_adx_min"] = 18

        # Force EMA50 alignment on hourly (cuts many weak signals)
        Q["require_close_above_ema50"] = True
        Q["prev_close_above_ema50"] = True
        Q["wk_close_above_ema50"] = True

        # Stronger "previous bar change" (avoid weak setups)
        if Q.get("dprev_min") is not None:
            Q["dprev_min"] = float(Q["dprev_min"]) + 0.15
        if Q.get("dprev_max") is not None:
            Q["dprev_max"] = float(Q["dprev_max"]) - 0.10  # narrows allowed band

        # Anti-chase tighter (price must be closer to EMA20)
        if Q.get("anti_chase_ema20_mult") is not None:
            Q["anti_chase_ema20_mult"] = max(1.02, float(Q["anti_chase_ema20_mult"]) - 0.01)

        out[k] = Q
    return out






# ======================================================================
#                     10 POSITIONAL STRATEGIES (UNCHANGED PARAMS)
# ======================================================================
STRATEGY_PARAMS: dict[str, dict] = {
    "custom01": dict(
        name="CUSTOM01",
        wk_close_above_ema200=True,
        wk_rsi_min=45, wk_adx_min=None,
        dprev_min=0.0, dprev_max=6.0,  # now applies to PrevBarChange (%)
        prev_close_above_ema20=False, prev_close_above_ema50=False,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0010,
        body_min_pct=0.0030,
        require_close_above_ema20=True,
        require_close_above_ema50=False,
        boosters_required=2,
        rsi_min=54, rsi_delta_min=0.8,
        macd_hist_min=0.0, macd_hist_mult=1.00,
        vol_mult=1.25,
        day_change_min=0.8,           # now applies to Bar_Change (%)
        anti_chase_ema20_mult=1.07,
        min_price=20.0,
    ),
    "custom02": dict(
        name="CUSTOM02",
        wk_close_above_ema200=True,
        wk_rsi_min=48, wk_adx_min=16,
        dprev_min=0.0, dprev_max=5.5,
        prev_close_above_ema20=True, prev_close_above_ema50=False,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0012,
        body_min_pct=0.0035,
        require_close_above_ema20=True,
        require_close_above_ema50=False,
        boosters_required=2,
        rsi_min=56, rsi_delta_min=1.0,
        macd_hist_min=0.0, macd_hist_mult=1.005,
        vol_mult=1.30,
        day_change_min=1.0,
        anti_chase_ema20_mult=1.065,
        min_price=25.0,
    ),
    "custom03": dict(
        name="CUSTOM03",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.00,
        wk_rsi_min=50, wk_adx_min=18,
        dprev_min=0.2, dprev_max=5.0,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0015,
        body_min_pct=0.0040,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=2,
        rsi_min=57, rsi_delta_min=1.1,
        macd_hist_min=0.0, macd_hist_mult=1.01,
        vol_mult=1.35,
        day_change_min=1.2,
        anti_chase_ema20_mult=1.06,
        min_price=30.0,
    ),
    "custom04": dict(
        name="CUSTOM04",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.005,
        wk_rsi_min=52, wk_adx_min=20,
        dprev_min=0.3, dprev_max=4.8,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0018,
        body_min_pct=0.0045,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=3,
        rsi_min=58.5, rsi_delta_min=1.2,
        macd_hist_min=0.0, macd_hist_mult=1.015,
        vol_mult=1.45,
        day_change_min=1.4,
        anti_chase_ema20_mult=1.055,
        min_price=40.0,
    ),
    "custom05": dict(
        name="CUSTOM05",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.01,
        wk_rsi_min=54, wk_adx_min=22,
        dprev_min=0.4, dprev_max=4.5,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 50, 120],
        breakout_close_mult=1.0020,
        breakout_wick_mult_hi=1.0012,
        breakout_wick_mult_close=1.0008,
        body_min_pct=0.0050,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=3,
        rsi_min=60.0, rsi_delta_min=1.3,
        macd_hist_min=0.0, macd_hist_mult=1.02,
        vol_mult=1.55,
        day_change_min=1.6,
        anti_chase_ema20_mult=1.05,
        d_adx_min=18,
        min_price=50.0,
    ),
    "custom06": dict(
        name="CUSTOM06",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.012,
        wk_rsi_min=55, wk_adx_min=23,
        dprev_min=0.5, dprev_max=4.2,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 60, 120],
        breakout_close_mult=1.0022,
        body_min_pct=0.0052,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=3,
        rsi_min=60.5, rsi_delta_min=1.4,
        macd_hist_min=0.0, macd_hist_mult=1.02,
        vol_mult=1.60,
        day_change_min=1.7,
        anti_chase_ema20_mult=1.048,
        d_adx_min=19,
        min_price=70.0,
        min_vol_sma20=200000,
    ),
    "custom07": dict(
        name="CUSTOM07",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.02,
        wk_rsi_min=57, wk_adx_min=25,
        dprev_min=0.6, dprev_max=4.0,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 50, 150],
        breakout_close_mult=1.0025,
        body_min_pct=0.0055,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=3,
        rsi_min=61.5, rsi_delta_min=1.5,
        macd_hist_min=0.0, macd_hist_mult=1.025,
        vol_mult=1.70,
        day_change_min=1.8,
        anti_chase_ema20_mult=1.045,
        d_adx_min=20,
        min_price=80.0,
        min_vol_sma20=300000,
    ),
    "custom08": dict(
        name="CUSTOM08",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.02,
        wk_rsi_min=58, wk_adx_min=26,
        dprev_min=0.7, dprev_max=3.8,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0028,
        body_min_pct=0.0060,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=4,
        rsi_min=62.0, rsi_delta_min=1.6,
        macd_hist_min=0.0, macd_hist_mult=1.03,
        vol_mult=1.80,
        day_change_min=2.0,
        anti_chase_ema20_mult=1.042,
        d_adx_min=21,
        min_price=100.0,
        min_vol_sma20=300000,
    ),
    "custom09": dict(
        name="CUSTOM09",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.03,
        wk_rsi_min=59, wk_adx_min=27,
        dprev_min=0.8, dprev_max=3.6,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 60, 120],
        breakout_close_mult=1.0030,
        body_min_pct=0.0062,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=4,
        rsi_min=63.0, rsi_delta_min=1.7,
        macd_hist_min=0.0, macd_hist_mult=1.035,
        vol_mult=1.90,
        day_change_min=2.2,
        anti_chase_ema20_mult=1.038,
        d_adx_min=22,
        min_price=120.0,
        min_vol_sma20=400000,
    ),
    "custom10": dict(
        name="CUSTOM10",
        wk_close_above_ema200=True,
        wk_close_above_ema50=True,
        wk_ema50_over_ema200_min=1.03,
        wk_rsi_min=60, wk_adx_min=28,
        dprev_min=0.8, dprev_max=3.5,
        prev_close_above_ema20=True, prev_close_above_ema50=True,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0032,
        breakout_wick_mult_hi=1.0015,
        breakout_wick_mult_close=1.0010,
        body_min_pct=0.0065,
        require_close_above_ema20=True,
        require_close_above_ema50=True,
        boosters_required=4,
        rsi_min=64.0, rsi_delta_min=1.8,
        macd_hist_min=0.0, macd_hist_mult=1.04,
        vol_mult=2.0,
        day_change_min=2.3,
        anti_chase_ema20_mult=1.035,
        d_adx_min=23,
        min_price=150.0,
        min_vol_sma20=500000,
    ),
}

# STRICT tweak (reduces signals)
STRATEGY_PARAMS = make_strategies_more_strict(STRATEGY_PARAMS)
STRATEGY_PARAMS = make_strategies_high_momentum(STRATEGY_PARAMS)



def build_signals_for_strategies(ticker: str, strategy_keys: list[str]) -> list[dict]:
    all_sigs: list[dict] = []
    for key in strategy_keys:
        key_lower = key.lower().strip()
        if key_lower not in STRATEGY_PARAMS:
            print(f"! Unknown strategy key '{key}', skipping for ticker {ticker}")
            continue
        P = STRATEGY_PARAMS[key_lower]
        all_sigs.extend(_build_signals_positional_template(ticker, P))
    return all_sigs


# ----------------------- HOURLY READER -----------------------
def _safe_read_hourly(path: str) -> pd.DataFrame:
    return _safe_read_tf(path, tz=IST)


# ----------------------- POSITION SIZING -----------------------
def attach_position_sizing(sig_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - qty
      - invested_amount

    Rule:
      - If entry_price > CAPITAL_PER_SIGNAL_RS => qty = 1
      - Else qty = floor(CAPITAL_PER_SIGNAL_RS / entry_price)
      - Always >= 1
    """
    if sig_df is None or sig_df.empty:
        return sig_df

    out = sig_df.copy()
    ep = pd.to_numeric(out.get("entry_price", np.nan), errors="coerce")

    qty = np.floor(CAPITAL_PER_SIGNAL_RS / ep).astype("float")
    qty = qty.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    qty = qty.clip(lower=1.0).astype(int)

    invested = (qty.astype(float) * ep.astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["qty"] = qty
    out["invested_amount"] = invested
    return out


# ----------------------- EVALUATION (HOURLY) -----------------------
def evaluate_signal_on_hourly_long_only(row: pd.Series, df_h: pd.DataFrame) -> dict:
    """
    Positional evaluation on HOURLY bars:
      - evaluates strictly AFTER signal bar (no same-bar evaluation)
      - if future high hits target => exit at target on first hit bar
      - else exit at last available future close

    pnl_rs computed using invested_amount (qty * entry_price).
    time-to-target reported in DAYS (fractional) from hourly timestamps.
    """
    entry_time = pd.to_datetime(row["signal_time_ist"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize(IST)
    else:
        entry_time = entry_time.tz_convert(IST)

    entry_price = float(row["entry_price"])
    if (not np.isfinite(entry_price)) or entry_price <= 0 or df_h is None or df_h.empty:
        return {
            HIT_COL: False, TIME_TD_COL: np.nan, TIME_CAL_COL: np.nan,
            "exit_time": entry_time, "exit_price": entry_price,
            "pnl_pct": 0.0, "pnl_rs": 0.0,
            "max_favorable_pct": np.nan, "max_adverse_pct": np.nan,
        }

    # sizing (safe fallback)
    qty = row.get("qty", None)
    try:
        qty = int(qty) if qty is not None else int(max(1, np.floor(CAPITAL_PER_SIGNAL_RS / entry_price)))
    except Exception:
        qty = int(max(1, np.floor(CAPITAL_PER_SIGNAL_RS / entry_price)))

    invested_amount = row.get("invested_amount", None)
    try:
        invested_amount = float(invested_amount) if invested_amount is not None else float(qty * entry_price)
    except Exception:
        invested_amount = float(qty * entry_price)

    dfx = df_h.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], utc=True, errors="coerce").dt.tz_convert(IST)
    dfx = dfx.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    future = dfx[dfx["date"] > entry_time].copy()
    if future.empty or (not {"high", "low", "close"}.issubset(future.columns)):
        return {
            HIT_COL: False, TIME_TD_COL: np.nan, TIME_CAL_COL: np.nan,
            "exit_time": entry_time, "exit_price": entry_price,
            "pnl_pct": 0.0, "pnl_rs": 0.0,
            "max_favorable_pct": np.nan, "max_adverse_pct": np.nan,
        }

    target_price = entry_price * TARGET_MULT
    hit_mask = pd.to_numeric(future["high"], errors="coerce") >= target_price

    max_fav = ((pd.to_numeric(future["high"], errors="coerce") - entry_price) / entry_price * 100.0).max()
    max_adv = ((pd.to_numeric(future["low"], errors="coerce") - entry_price) / entry_price * 100.0).min()

    if hit_mask.any():
        first_hit_pos = int(np.argmax(hit_mask.values))
        hit_row = future.iloc[first_hit_pos]
        exit_time = hit_row["date"]
        exit_price = float(target_price)

        dt_days = (exit_time - entry_time).total_seconds() / 86400.0
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        pnl_rs  = invested_amount * (pnl_pct / 100.0)

        return {
            HIT_COL: True,
            TIME_TD_COL: float(dt_days),
            TIME_CAL_COL: float(dt_days),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl_pct": float(pnl_pct),
            "pnl_rs": float(pnl_rs),
            "qty": int(qty),
            "invested_amount": float(invested_amount),
            "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
            "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
        }
    else:
        return {
            HIT_COL: False,
            "status": "OPEN",
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "pnl_pct": np.nan,
            "pnl_rs": 0.0,  # keep 0 for easy sums
            "qty": int(qty),
            "invested_amount": float(invested_amount),
            "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
            "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
        }


def evaluate_signal_hourly_tp_sl(
    row: pd.Series,
    df_h: pd.DataFrame,
    target_pct: float,
    stoploss_pct: float,
    tie_break: str = "SL",   # if TP and SL both happen in same hourly bar, choose "SL" (conservative) or "TP"
) -> dict:
    """
    Separate evaluation:
      - Exit at first occurrence of either:
          TP: future high >= entry * (1 + target_pct/100)
          SL: future low  <= entry * (1 + stoploss_pct/100)   (stoploss_pct negative)
      - If neither occurs, trade remains OPEN (no pnl, no exit_time)
      - Uses invested_amount (qty * entry_price) like your main eval.

    NOTE: In the same-bar TP+SL case, we can't know intrabar order from OHLC.
          tie_break="SL" is conservative for LONGs.
    """
    entry_time = pd.to_datetime(row["signal_time_ist"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize(IST)
    else:
        entry_time = entry_time.tz_convert(IST)

    entry_price = float(row["entry_price"])
    if (not np.isfinite(entry_price)) or entry_price <= 0 or df_h is None or df_h.empty:
        return {
            "tp_hit": False,
            "sl_hit": False,
            "status_tp_sl": "BAD_DATA",
            "exit_time_tp_sl": pd.NaT,
            "exit_price_tp_sl": np.nan,
            "pnl_pct_tp_sl": np.nan,
            "pnl_rs_tp_sl": 0.0,
            "days_to_exit_tp_sl": np.nan,
        }

    # sizing (use same logic as your main)
    try:
        qty = int(row.get("qty", 0)) if row.get("qty", None) is not None else 0
    except Exception:
        qty = 0
    if qty <= 0:
        qty = int(max(1, np.floor(CAPITAL_PER_SIGNAL_RS / entry_price)))

    invested_amount = row.get("invested_amount", None)
    try:
        invested_amount = float(invested_amount) if invested_amount is not None else float(qty * entry_price)
    except Exception:
        invested_amount = float(qty * entry_price)

    dfx = df_h.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], utc=True, errors="coerce").dt.tz_convert(IST)
    dfx = dfx.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    future = dfx[dfx["date"] > entry_time].copy()
    if future.empty or (not {"high", "low", "close"}.issubset(future.columns)):
        return {
            "tp_hit": False,
            "sl_hit": False,
            "status_tp_sl": "OPEN",
            "exit_time_tp_sl": pd.NaT,
            "exit_price_tp_sl": np.nan,
            "pnl_pct_tp_sl": np.nan,
            "pnl_rs_tp_sl": 0.0,
            "days_to_exit_tp_sl": np.nan,
            "qty_tp_sl": int(qty),
            "invested_amount_tp_sl": float(invested_amount),
        }

    tp_price = entry_price * (1.0 + target_pct / 100.0)
    sl_price = entry_price * (1.0 + stoploss_pct / 100.0)

    hi = pd.to_numeric(future["high"], errors="coerce")
    lo = pd.to_numeric(future["low"], errors="coerce")

    tp_mask = (hi >= tp_price).fillna(False)
    sl_mask = (lo <= sl_price).fillna(False)

    if not (tp_mask.any() or sl_mask.any()):
        return {
            "tp_hit": False,
            "sl_hit": False,
            "status_tp_sl": "OPEN",
            "exit_time_tp_sl": pd.NaT,
            "exit_price_tp_sl": np.nan,
            "pnl_pct_tp_sl": np.nan,
            "pnl_rs_tp_sl": 0.0,
            "days_to_exit_tp_sl": np.nan,
            "qty_tp_sl": int(qty),
            "invested_amount_tp_sl": float(invested_amount),
        }

    # first occurrence indexes (if exists)
    tp_idx = int(np.argmax(tp_mask.values)) if tp_mask.any() else None
    sl_idx = int(np.argmax(sl_mask.values)) if sl_mask.any() else None

    # decide which happens first
    exit_kind = None
    if tp_idx is not None and sl_idx is not None:
        if tp_idx < sl_idx:
            exit_kind = "TP"
        elif sl_idx < tp_idx:
            exit_kind = "SL"
        else:
            # same bar: tie-break
            exit_kind = "SL" if str(tie_break).upper() == "SL" else "TP"
    elif tp_idx is not None:
        exit_kind = "TP"
    else:
        exit_kind = "SL"

    if exit_kind == "TP":
        exit_row = future.iloc[tp_idx]
        exit_time = exit_row["date"]
        pnl_pct = float(target_pct)
        pnl_rs = invested_amount * (pnl_pct / 100.0)
        return {
            "tp_hit": True,
            "sl_hit": False,
            "status_tp_sl": "TP",
            "exit_time_tp_sl": exit_time,
            "exit_price_tp_sl": float(tp_price),
            "pnl_pct_tp_sl": pnl_pct,
            "pnl_rs_tp_sl": float(pnl_rs),
            "days_to_exit_tp_sl": float((exit_time - entry_time).total_seconds() / 86400.0),
            "qty_tp_sl": int(qty),
            "invested_amount_tp_sl": float(invested_amount),
        }

    # SL
    exit_row = future.iloc[sl_idx]
    exit_time = exit_row["date"]
    pnl_pct = float(stoploss_pct)  # negative
    pnl_rs = invested_amount * (pnl_pct / 100.0)
    return {
        "tp_hit": False,
        "sl_hit": True,
        "status_tp_sl": "SL",
        "exit_time_tp_sl": exit_time,
        "exit_price_tp_sl": float(sl_price),
        "pnl_pct_tp_sl": pnl_pct,
        "pnl_rs_tp_sl": float(pnl_rs),
        "days_to_exit_tp_sl": float((exit_time - entry_time).total_seconds() / 86400.0),
        "qty_tp_sl": int(qty),
        "invested_amount_tp_sl": float(invested_amount),
    }


def run_tp_sl_profit_loss_analysis(
    sig_df: pd.DataFrame,
    data_cache: Dict[str, pd.DataFrame],
    strategy_label: str,
    ts: str,
) -> pd.DataFrame:
    """
    Separate analysis printout:
      - TP = target_bew%
      - SL = stoploss_new%
      - Only CLOSED trades (TP or SL) are used for capital P&L.
      - OPEN trades (neither TP nor SL hit) are ignored (no exit, no pnl).
    """
    print("\n" + "=" * 72)
    print(f"--- TP/SL PROFIT-LOSS ANALYSIS (TP={target_bew:.2f}%, SL={stoploss_new:.2f}%) ---")
    print("=" * 72)

    if sig_df is None or sig_df.empty:
        print("No signals available.")
        return pd.DataFrame()

    # Ensure sizing exists
    if ("qty" not in sig_df.columns) or ("invested_amount" not in sig_df.columns):
        sig_df = attach_position_sizing(sig_df)

    rows = []
    for _, row in sig_df.iterrows():
        tkr = row["ticker"]
        df_h = data_cache.get(tkr, pd.DataFrame())
        info = evaluate_signal_hourly_tp_sl(
            row=row,
            df_h=df_h,
            target_pct=target_bew,
            stoploss_pct=stoploss_new,
            tie_break="SL",
        )
        rr = row.to_dict()
        rr.update(info)
        rows.append(rr)

    out_tp_sl = pd.DataFrame(rows).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    total = len(out_tp_sl)
    tp_cnt = int((out_tp_sl["status_tp_sl"] == "TP").sum())
    sl_cnt = int((out_tp_sl["status_tp_sl"] == "SL").sum())
    open_cnt = int((out_tp_sl["status_tp_sl"] == "OPEN").sum())
    closed_cnt = tp_cnt + sl_cnt
    win_rate = (tp_cnt / closed_cnt * 100.0) if closed_cnt else 0.0

    print(f"Signals (total)        : {total}")
    print(f"Closed (TP or SL)      : {closed_cnt}")
    print(f"  TP hits              : {tp_cnt}")
    print(f"  SL hits              : {sl_cnt}")
    print(f"Open (ignored in P&L)  : {open_cnt}")
    print(f"Win rate (TP/(TP+SL))  : {win_rate:.1f}%")

    # time stats
    if closed_cnt:
        days = pd.to_numeric(out_tp_sl.loc[out_tp_sl["status_tp_sl"].isin(["TP","SL"]), "days_to_exit_tp_sl"], errors="coerce").dropna()
        if not days.empty:
            print(f"Avg days to exit (closed): {days.mean():.2f}")
            print(f"Median days to exit      : {days.median():.2f}")

    # Save full TP/SL result separately
    tag = f"tp{_pct_tag(target_bew)}_sl{_pct_tag(abs(stoploss_new))}"
    tp_sl_path = os.path.join(OUT_DIR, f"tp_sl_analysis_1h_{strategy_label}_{tag}_{ts}_IST.csv")
    out_tp_sl.to_csv(tp_sl_path, index=False)
    print(f"Saved TP/SL analysis CSV: {tp_sl_path}")

    # Capital simulation ONLY on closed trades
    closed = out_tp_sl[out_tp_sl["status_tp_sl"].isin(["TP", "SL"])].copy()
    if closed.empty:
        print("\nNo closed trades to compute realized P&L under TP/SL.")
        return out_tp_sl

    # Map into apply_capital_constraint expected columns
    closed["exit_time"] = closed["exit_time_tp_sl"]
    closed["pnl_rs"] = pd.to_numeric(closed["pnl_rs_tp_sl"], errors="coerce").fillna(0.0)
    closed["pnl_pct"] = pd.to_numeric(closed["pnl_pct_tp_sl"], errors="coerce").fillna(0.0)

    # IMPORTANT: Keep your existing invested_amount/qty as-is (per-signal investment rule)
    # Then run capital constraint
    closed_constrained, cap = apply_capital_constraint(closed)

    print("\n--- TP/SL CAPITAL-CONSTRAINED (CLOSED TRADES ONLY) ---")
    print(f"Signals closed     : {cap['total_signals']}")
    print(f"Signals taken      : {cap['signals_taken']}")
    print(f"Final value        : ₹{cap['final_portfolio_value']:,.2f}")
    print(f"Net P&L            : ₹{cap['net_pnl_rs']:,.2f} ({cap['net_pnl_pct']:.2f}%)")

    # Extra: realized P&L breakdown
    realized_pnl = float(closed_constrained.loc[closed_constrained["taken"] == True, "pnl_rs"].sum()) if "taken" in closed_constrained.columns else float(closed_constrained["pnl_rs"].sum())
    print(f"Realized P&L (taken): ₹{realized_pnl:,.2f}")

    return out_tp_sl


# ----------------------- PRINTS -----------------------
def _hourly_date_range_from_cache(
    data_cache: Dict[str, pd.DataFrame],
    cutoff_dt_ist: Optional[datetime] = None,
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Returns (min_dt, max_dt) from HOURLY candle data across all tickers in cache.
    If cutoff_dt_ist is provided -> uses only rows <= cutoff_dt_ist (IST).
    """
    mins = []
    maxs = []

    for tkr, df in data_cache.items():
        if df is None or df.empty or "date" not in df.columns:
            continue

        dt = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
        if dt.empty:
            continue

        dt = dt.dt.tz_convert(IST)

        if cutoff_dt_ist is not None:
            dt = dt[dt <= cutoff_dt_ist]

        if dt.empty:
            continue

        mins.append(dt.min())
        maxs.append(dt.max())

    if not mins:
        return None, None

    return min(mins), max(maxs)


def print_analysis_date_range_from_hourly(data_cache: Dict[str, pd.DataFrame]) -> None:
    """
    Prints ranges derived ONLY from hourly candle data:
      - Analysis range: hourly data used for signal generation (<= CUTOFF_END_IST)
      - Eval range    : full hourly data available in files (no cutoff)
    """
    print("\n--- ANALYSIS DATE RANGE (FROM HOURLY DATA) ---")

    mn_a, mx_a = _hourly_date_range_from_cache(data_cache, cutoff_dt_ist=CUTOFF_END_IST)
    if mn_a is None or mx_a is None:
        print("No hourly data in cache to infer range.")
    else:
        print(f"Hourly analysis range (IST): {mn_a.strftime('%Y-%m-%d')}  →  {mx_a.strftime('%Y-%m-%d')}")

    mn_e, mx_e = _hourly_date_range_from_cache(data_cache, cutoff_dt_ist=None)
    if mn_e is None or mx_e is None:
        print("No hourly data in cache to infer eval range.")
    else:
        print(f"Hourly eval range (IST)    : {mn_e.strftime('%Y-%m-%d')}  →  {mx_e.strftime('%Y-%m-%d')}")



def print_target_hit_time_stats(out_df: pd.DataFrame) -> None:
    if out_df is None or out_df.empty:
        print("\n--- TIME-TO-TARGET STATS ---")
        print("No evaluation rows available.")
        return

    if HIT_COL not in out_df.columns:
        print("\n--- TIME-TO-TARGET STATS ---")
        print(f"Missing column: {HIT_COL}")
        return

    hit_df = out_df[out_df[HIT_COL].astype(bool)].copy()
    if hit_df.empty:
        print("\n--- TIME-TO-TARGET STATS ---")
        print(f"No target hits for {TARGET_LABEL}.")
        return

    if TIME_CAL_COL not in hit_df.columns:
        print("\n--- TIME-TO-TARGET STATS ---")
        print(f"Missing column: {TIME_CAL_COL}")
        return

    tt = pd.to_numeric(hit_df[TIME_CAL_COL], errors="coerce").dropna()

    print("\n--- TIME-TO-TARGET STATS (hits only) ---")
    print(f"Target: {TARGET_LABEL}")
    print(f"Hit count: {len(tt)}")
    if tt.empty:
        print("No valid time-to-target values to compute stats.")
        return
    print(f"Average days to hit target : {float(tt.mean()):.2f}")
    print(f"Median days to hit target  : {float(tt.median()):.2f}")


def print_signals_per_day_stats(sig_df: pd.DataFrame) -> None:
    print("\n--- SIGNALS PER DAY STATS (generated signals) ---")
    if sig_df is None or sig_df.empty or "signal_time_ist" not in sig_df.columns:
        print("No signals to compute per-day stats.")
        return

    st = pd.to_datetime(sig_df["signal_time_ist"], utc=True, errors="coerce").dropna()
    if st.empty:
        print("No valid signal timestamps.")
        return

    st = st.dt.tz_convert(IST)
    counts = pd.Series(st.dt.date).value_counts().sort_index()
    if counts.empty:
        print("No per-day counts available.")
        return

    print(f"Days covered               : {len(counts)}")
    print(f"Average signals per day    : {float(counts.mean()):.2f}")
    print(f"Median signals per day     : {float(counts.median()):.2f}")


# ----------------------- DEDUPE -----------------------
def dedupe_signals_per_ticker_cooldown(sig_df: pd.DataFrame, cooldown_hours: int = 24) -> pd.DataFrame:
    """Keep earliest signal per ticker, then enforce cooldown in hours."""
    if sig_df is None or sig_df.empty:
        return sig_df
    if "ticker" not in sig_df.columns or "signal_time_ist" not in sig_df.columns:
        return sig_df

    df = sig_df.copy()
    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"], utc=True, errors="coerce").dt.tz_convert(IST)
    df = df.dropna(subset=["signal_time_ist"]).sort_values(["ticker", "signal_time_ist"]).reset_index(drop=True)

    kept_rows = []
    last_time = {}

    cooldown = pd.Timedelta(hours=int(cooldown_hours))

    for _, r in df.iterrows():
        tkr = r["ticker"]
        t = r["signal_time_ist"]
        if (tkr not in last_time) or ((t - last_time[tkr]) >= cooldown):
            kept_rows.append(r)
            last_time[tkr] = t

    out = pd.DataFrame(kept_rows).reset_index(drop=True)

    print("\n--- DEDUPE (per ticker cooldown) ---")
    print(f"Cooldown hours: {cooldown_hours}")
    print(f"Before: {len(df)} signals")
    print(f"After : {len(out)} signals")
    print(f"Removed: {len(df) - len(out)}")
    return out

# ----------------------- DEDUPE -----------------------
def dedupe_signals_per_ticker_day(sig_df: pd.DataFrame) -> pd.DataFrame:
    """Keep earliest signal per (ticker, IST calendar date)."""
    if sig_df is None or sig_df.empty:
        return sig_df
    if "ticker" not in sig_df.columns or "signal_time_ist" not in sig_df.columns:
        return sig_df

    df = sig_df.copy()
    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"], utc=True, errors="coerce").dt.tz_convert(IST)
    df = df.dropna(subset=["signal_time_ist"])

    df["_sig_date_ist"] = df["signal_time_ist"].dt.date
    before = len(df)

    df = df.sort_values(["ticker", "_sig_date_ist", "signal_time_ist"]).drop_duplicates(
        subset=["ticker", "_sig_date_ist"],
        keep="first"
    ).reset_index(drop=True)

    after = len(df)
    df = df.drop(columns=["_sig_date_ist"], errors="ignore")

    print("\n--- DEDUPE (per ticker/day) ---")
    print(f"Before: {before} signals")
    print(f"After : {after} signals")
    print(f"Removed duplicates: {before - after}")
    return df


# ----------------------- CAPITAL-CONSTRAINED PORTFOLIO SIM -----------------------
def apply_capital_constraint(out_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Uses per-row invested_amount (qty * entry_price), not fixed CAPITAL_PER_SIGNAL_RS."""
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"], utc=True, errors="coerce").dt.tz_convert(IST)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    if df["exit_time"].notna().any():
        df.loc[df["exit_time"].notna(), "exit_time"] = df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)

    df["pnl_rs"]  = pd.to_numeric(df.get("pnl_rs", 0.0), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0.0), errors="coerce").fillna(0.0)

    if ("invested_amount" not in df.columns) or ("qty" not in df.columns):
        df = attach_position_sizing(df)

    df["invested_amount"] = pd.to_numeric(df["invested_amount"], errors="coerce").fillna(0.0)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(1).astype(int).clip(lower=1)

    df["final_value_per_signal"] = df["invested_amount"] + df["pnl_rs"]

    available_capital = float(MAX_CAPITAL_RS)
    open_trades: list[dict] = []
    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        # Release capital from closed trades
        still_open = []
        for tr in open_trades:
            if pd.isna(tr["exit_time"]) or tr["exit_time"] <= entry_time:
                available_capital += tr["value"]
            else:
                still_open.append(tr)
        open_trades = still_open

        need_cap = float(row["invested_amount"])
        if need_cap <= 0:
            continue

        if available_capital >= need_cap:
            df.at[idx, "taken"] = True
            available_capital -= need_cap

            exit_time = row["exit_time"]
            if pd.isna(exit_time):
                exit_time = entry_time

            open_trades.append({"exit_time": exit_time, "value": float(row["final_value_per_signal"])})

    # Close remaining
    for tr in open_trades:
        available_capital += tr["value"]

    df["invested_amount_constrained"] = np.where(df["taken"], df["invested_amount"], 0.0)
    df["final_value_constrained"] = np.where(df["taken"], df["final_value_per_signal"], 0.0)

    summary = {
        "total_signals": int(len(df)),
        "signals_taken": int(df["taken"].sum()),
        "gross_invested": float(df["invested_amount_constrained"].sum()),
        "final_portfolio_value": float(available_capital),
        "net_pnl_rs": float(available_capital - MAX_CAPITAL_RS),
        "net_pnl_pct": float((available_capital - MAX_CAPITAL_RS) / MAX_CAPITAL_RS * 100.0) if MAX_CAPITAL_RS else 0.0,
    }
    return df, summary


def build_daily_summary(out_df: pd.DataFrame) -> pd.DataFrame:
    dfx = out_df.copy()
    dfx["signal_time_ist"] = pd.to_datetime(dfx["signal_time_ist"], utc=True, errors="coerce").dt.tz_convert(IST)
    dfx = dfx.dropna(subset=["signal_time_ist"])
    dfx["date"] = dfx["signal_time_ist"].dt.date
    if "taken" not in dfx.columns:
        dfx["taken"] = True
    dfx["pnl_rs"] = pd.to_numeric(dfx.get("pnl_rs", 0.0), errors="coerce").fillna(0.0)

    rows = []
    for day, g in dfx.groupby("date", dropna=False):
        taken = g["taken"].astype(bool)
        rows.append({
            "date": day,
            "signals_generated": int(len(g)),
            "signals_taken": int(taken.sum()),
            "pnl_rs_sum": float(g["pnl_rs"].sum()),
            "pnl_rs_taken_sum": float(g.loc[taken, "pnl_rs"].sum()) if taken.any() else 0.0,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ----------------------- CLI STRATEGY SELECTION -----------------------
def _parse_strategy_keys_from_argv() -> list[str]:
    args = [a.lower().strip() for a in sys.argv[1:] if a.strip()]
    if not args:
        return DEFAULT_STRATEGY_KEYS

    if len(args) == 1 and args[0] == "all":
        return sorted(STRATEGY_PARAMS.keys())

    selected: list[str] = []
    for a in args:
        if a in STRATEGY_PARAMS:
            selected.append(a)
        else:
            print(f"! Unknown strategy key '{a}' – valid keys: {', '.join(sorted(STRATEGY_PARAMS.keys()))}")
    return selected or DEFAULT_STRATEGY_KEYS


# ----------------------- MAIN -----------------------
def main():
    print(f"\nAS-OF RUN TIME (IST): {NOW_IST.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Using ONLY HOURLY data up to: {CUTOFF_END_IST.strftime('%Y-%m-%d %H:%M:%S %Z')} (last completed hour bucket)\n")

    tickers_h = _list_tickers_from_dir(DIR_H)
    tickers_w = _list_tickers_from_dir(DIR_W)
    tickers = tickers_h & tickers_w

    if not tickers:
        print("No common tickers found across HOURLY and WEEKLY directories.")
        return

    strategy_keys = _parse_strategy_keys_from_argv()
    strategy_label = "+".join(strategy_keys)

    print(f"Found {len(tickers)} tickers with both 1H+Weekly.")
    print(f"Running strategies: {strategy_label}\n")

    all_signals: list[dict] = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        try:
            all_signals.extend(build_signals_for_strategies(ticker, strategy_keys))
            if i % 100 == 0:
                print(f"[{i}/{len(tickers)}] ... processed")
        except Exception as e:
            print(f"! {ticker}: {e}")

    if not all_signals:
        print("No LONG signals generated.")
        return

    sig_df = pd.DataFrame(all_signals)
    sig_df["signal_time_ist"] = pd.to_datetime(sig_df["signal_time_ist"], utc=True, errors="coerce").dt.tz_convert(IST)
    sig_df = sig_df.dropna(subset=["signal_time_ist"]).reset_index(drop=True)

    # De-dupe
    sig_df = dedupe_signals_per_ticker_cooldown(sig_df, cooldown_hours=36)
    
    # De-dupe: 1 signal per ticker per IST day
    #sig_df = dedupe_signals_per_ticker_day(sig_df)

    # Attach qty/invested_amount BEFORE eval + capital sim
    sig_df = attach_position_sizing(sig_df)

    # Stats AFTER dedupe (and before eval)
    print_signals_per_day_stats(sig_df)

    sig_df["itr"] = np.arange(1, len(sig_df) + 1)

    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")
    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_1h_{strategy_label}_{ts}_IST.csv")
    sig_df.to_csv(signals_path, index=False)
    print(f"Saved signals: {signals_path}")

    # ----------------------- EVALUATE ON HOURLY -----------------------
    data_cache: Dict[str, pd.DataFrame] = {}
    for tkr in sig_df["ticker"].unique():
        p = _find_file_hourly(DIR_H, tkr)
        data_cache[tkr] = _safe_read_hourly(p) if p else pd.DataFrame()

    results = []
    for _, row in sig_df.iterrows():
        tkr = row["ticker"]
        df_h = data_cache.get(tkr, pd.DataFrame())

        new_info = (
            evaluate_signal_on_hourly_long_only(row, df_h)
            if not df_h.empty else {
                HIT_COL: False, TIME_TD_COL: np.nan, TIME_CAL_COL: np.nan,
                "exit_time": row["signal_time_ist"], "exit_price": row["entry_price"],
                "pnl_pct": 0.0, "pnl_rs": 0.0,
                "max_favorable_pct": np.nan, "max_adverse_pct": np.nan,
                "qty": int(row.get("qty", 1)),
                "invested_amount": float(row.get("invested_amount", 0.0)),
            }
        )

        rr = row.to_dict()
        rr.update(new_info)
        results.append(rr)

    out_df = pd.DataFrame(results).sort_values(["signal_time_ist", "ticker", "strategy"]).reset_index(drop=True)

    # Print analysis range (signals + exits)
    print_analysis_date_range_from_hourly(data_cache)


    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_1h_{strategy_label}_{EVAL_SUFFIX}_{ts}_IST.csv")
    out_df.to_csv(eval_path, index=False)
    print(f"\nSaved eval: {eval_path}")

    total = len(out_df)
    hits = int(out_df[HIT_COL].sum()) if HIT_COL in out_df.columns else 0
    hit_pct = (hits / total * 100.0) if total else 0.0
    print(f"\n[ALL STRATEGIES] Total={total}, hit {TARGET_LABEL}={hits} ({hit_pct:.1f}%)")

    if total > N_RECENT_SKIP:
        nr = out_df.iloc[:-N_RECENT_SKIP]
        hits_nr = int(nr[HIT_COL].sum()) if HIT_COL in nr.columns else 0
        nr_total = len(nr)
        hit_pct_nr = (hits_nr / nr_total * 100.0) if nr_total else 0.0
        print(f"[NON-RECENT excl last {N_RECENT_SKIP}] Total={nr_total}, hits={hits_nr} ({hit_pct_nr:.1f}%)")

    # Avg + median time-to-target
    print_target_hit_time_stats(out_df)

    # Capital constrained (uses invested_amount)
    out_df_constrained, cap = apply_capital_constraint(out_df)

    # Save constrained back into SAME eval path (your previous behavior)
    out_df_constrained.to_csv(eval_path, index=False)

    print("\n--- CAPITAL-CONSTRAINED (ALL STRATEGIES COMBINED) ---")
    print(f"Signals generated: {cap['total_signals']}")
    print(f"Signals taken    : {cap['signals_taken']}")
    print(f"Final value      : ₹{cap['final_portfolio_value']:,.2f}")
    print(f"Net P&L          : ₹{cap['net_pnl_rs']:,.2f} ({cap['net_pnl_pct']:.2f}%)")

    daily_summary = build_daily_summary(out_df_constrained)
    daily_summary_path = os.path.join(
        OUT_DIR,
        f"multi_tf_daily_summary_{strategy_label}_{EVAL_SUFFIX}_{ts}_IST.csv",
    )
    daily_summary.to_csv(daily_summary_path, index=False)
    print(f"\nSaved daily summary: {daily_summary_path}")

    # --- Separate TP/SL analysis printout (TP=6%, SL=-4%) ---
    run_tp_sl_profit_loss_analysis(sig_df=sig_df, data_cache=data_cache, strategy_label=strategy_label, ts=ts)



if __name__ == "__main__":
    main()
