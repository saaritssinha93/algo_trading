# -*- coding: utf-8 -*-
"""
Multi-timeframe positional LONG backtest framework (DAILY execution, WEEKLY context).

Execution TF:
    • Daily bars from DIR_D = "main_indicators_daily"

Context TF:
    • Weekly from DIR_W with NO same-week lookahead:
        - Weekly 'date' is shifted by +1 day
        - merge_asof(direction="backward")
        - So current week weekly bar becomes visible only from NEXT Monday

Key no-lookahead rules:
    • Weekly context: previous fully completed week only
    • Daily "prev": any prev-day filters use D-1 via shift(1)

What changed vs your old single-strategy file:
    ✅ NO intraday (only Daily + Weekly)
    ✅ 10 selectable strategies via STRATEGY_PARAMS (custom01..custom10)
    ✅ You can run ONE or MANY strategies in a single run via CLI or DEFAULT_STRATEGY_KEYS
"""

import os
import glob
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Target configuration ----
TARGET_PCT: float = 6.0
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
CAPITAL_PER_SIGNAL_RS: float = 50000.0
MAX_CAPITAL_RS: float = 20000000.0
N_RECENT_SKIP = 100

# ----------------------- STRATEGY SELECTION -----------------------
# Default strategies if you DON'T pass anything on CLI.
# You can change this default list to whatever mix you want.
DEFAULT_STRATEGY_KEYS: list[str] = ["custom10", "custom09", "custom08", "custom07", "custom06", "custom05", "custom04", "custom03", "custom02", "custom01"]  # for strongest "continuously rising" bias


# ----------------------- HELPERS (COMMON) -----------------------

def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """Read CSV, parse 'date' as tz-aware IST, sort ascending."""
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")

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


def _find_file(directory: str, ticker: str):
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None


# ======================================================================
#                 COMMON MTF LOADER (DAILY + WEEKLY)
# ======================================================================

def _load_mtf_context_daily_weekly(ticker: str) -> pd.DataFrame:
    """
    Load Daily + Weekly, suffix columns (_D, _W), merge Weekly onto Daily via as-of join.

    Weekly NO lookahead:
      shift weekly 'date' by +1 day and merge_asof(direction="backward")

    Daily prev:
      create Daily_Change_prev = Daily_Change.shift(1) BEFORE suffixing => Daily_Change_prev_D
    """
    path_d = _find_file(DIR_D, ticker)
    path_w = _find_file(DIR_W, ticker)

    if not path_d or not path_w:
        return pd.DataFrame()

    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)
    if df_d.empty or df_w.empty:
        return pd.DataFrame()

    # Daily prev-day change (D-1)
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Weekly NO same-week lookahead
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(days=1)

    df_dw = pd.merge_asof(
        df_d.sort_values("date"),
        df_w_sorted.sort_values("date"),
        on="date",
        direction="backward",
    )
    return df_dw.reset_index(drop=True)


# ======================================================================
#                 STRATEGY TEMPLATE (PARAMETRIC)
# ======================================================================

def _build_signals_positional_template(ticker: str, P: dict) -> list[dict]:
    """
    Generic positional LONG builder using only Daily+Weekly.

    P keys (with typical meaning):
      Weekly:
        wk_rsi_min, wk_adx_min, wk_close_above_ema200 (bool),
        wk_close_above_ema50 (bool), wk_ema50_over_ema200_min (float)
      Daily prev (D-1):
        dprev_min, dprev_max, prev_close_above_ema20, prev_close_above_ema50
      Entry day (D):
        breakout_windows: list[int], breakout_close_mult, breakout_wick_mult_hi, breakout_wick_mult_close
        body_min_pct, anti_chase_ema20_mult, require_close_above_ema20, require_close_above_ema50
        boosters_required (int)
        booster thresholds:
          rsi_min, rsi_delta_min, macd_hist_min, macd_hist_mult,
          vol_mult, day_change_min
        optional:
          d_adx_min
      Extra quality:
        min_price (optional), min_vol_sma20 (optional)
    """
    df = _load_mtf_context_daily_weekly(ticker)
    if df.empty:
        return []

    cols = set(df.columns)
    need = {"date", "open_D", "high_D", "low_D", "close_D"}
    if not need.issubset(cols):
        return []

    def _col(name: str, default=np.nan):
        return df[name] if name in cols else pd.Series(default, index=df.index)

    # ------------------- OPTIONAL "quality / tradability" filters -------------------
    close = df["close_D"]

    if P.get("min_price") is not None:
        tradable_ok = (close >= float(P["min_price"])).fillna(False)
    else:
        tradable_ok = pd.Series(True, index=df.index)

    # Liquidity proxy: require volume SMA20 above threshold (if volume exists)
    vol = _col("volume_D")
    if ("volume_D" in cols) and (P.get("min_vol_sma20") is not None):
        vol_sma20 = vol.rolling(20, min_periods=20).mean()
        tradable_ok &= (vol_sma20 >= float(P["min_vol_sma20"])).fillna(False)

    # ------------------- WEEKLY FILTER (prev fully completed week) -------------------
    weekly_ok = pd.Series(True, index=df.index)

    if P.get("wk_close_above_ema200", True) and {"close_W", "EMA_200_W"}.issubset(cols):
        weekly_ok &= (df["close_W"] > df["EMA_200_W"])

    if P.get("wk_close_above_ema50", False) and ("EMA_50_W" in cols) and ("close_W" in cols):
        weekly_ok &= (df["close_W"] > df["EMA_50_W"])

    if P.get("wk_ema50_over_ema200_min") is not None and {"EMA_50_W", "EMA_200_W"}.issubset(cols):
        weekly_ok &= (df["EMA_50_W"] >= df["EMA_200_W"] * float(P["wk_ema50_over_ema200_min"]))

    if P.get("wk_rsi_min") is not None and ("RSI_W" in cols):
        weekly_ok &= (df["RSI_W"] >= float(P["wk_rsi_min"]))

    if P.get("wk_adx_min") is not None and ("ADX_W" in cols):
        weekly_ok &= (df["ADX_W"] >= float(P["wk_adx_min"]))

    weekly_ok = weekly_ok.fillna(False)

    # ------------------- DAILY PREV FILTER (D-1) -------------------
    dprev = _col("Daily_Change_prev_D")
    daily_prev_ok = pd.Series(True, index=df.index)

    if P.get("dprev_min") is not None:
        daily_prev_ok &= (dprev >= float(P["dprev_min"]))
    if P.get("dprev_max") is not None:
        daily_prev_ok &= (dprev <= float(P["dprev_max"]))

    # Align previous close to MAs (use shift(1) consistently)
    prev_close = close.shift(1)
    ema20 = _col("EMA_20_D")
    ema50 = _col("EMA_50_D")

    if P.get("prev_close_above_ema20", False) and ("EMA_20_D" in cols):
        daily_prev_ok &= (prev_close >= ema20.shift(1))

    if P.get("prev_close_above_ema50", False) and ("EMA_50_D" in cols):
        daily_prev_ok &= (prev_close >= ema50.shift(1))

    daily_prev_ok = daily_prev_ok.fillna(False)

    # ------------------- BREAKOUT (NO LOOKAHEAD) -------------------
    high = df["high_D"]

    breakout_windows = P.get("breakout_windows", [20, 50, 100])
    prior_levels = []
    for w in breakout_windows:
        prior_levels.append(high.rolling(int(w), min_periods=int(w)).max().shift(1))

    breakout_level = prior_levels[0].copy()
    name_parts = [f"RH{breakout_windows[0]}"]
    for i in range(1, len(prior_levels)):
        fb = breakout_level.isna()
        breakout_level[fb] = prior_levels[i][fb]
        name_parts.append(f"RH{breakout_windows[i]}")
    # e.g. PRIOR_RH20_or_RH50_or_RH100
    breakout_name = "PRIOR_" + "_or_".join(name_parts)

    close_mult = float(P.get("breakout_close_mult", 1.0015))
    wick_mult_hi = float(P.get("breakout_wick_mult_hi", 1.0010))
    wick_mult_close = float(P.get("breakout_wick_mult_close", 1.0005))

    close_break = close >= breakout_level * close_mult
    wick_break  = (high >= breakout_level * wick_mult_hi) & (close >= breakout_level * wick_mult_close)
    breakout_ok = (close_break | wick_break).fillna(False)

    # ------------------- CANDLE QUALITY -------------------
    open_ = df["open_D"]
    green = (close > open_).fillna(False)
    body_min_pct = float(P.get("body_min_pct", 0.0040))  # 0.40% default
    body_ok = ((close - open_) >= (open_ * body_min_pct)).fillna(False)

    # ------------------- MOMENTUM BOOSTERS (STRICT) -------------------
    boosters_required = int(P.get("boosters_required", 3))

    # RSI booster
    rsi = _col("RSI_D")
    rsi_prev = rsi.shift(1)
    rsi_min = float(P.get("rsi_min", 58.0))
    rsi_delta_min = float(P.get("rsi_delta_min", 1.2))
    rsi_boost = ((rsi >= rsi_min) | (rsi >= (rsi_prev + rsi_delta_min))).fillna(False)

    # MACD Hist booster
    macd = _col("MACD_Hist_D")
    macd_prev = macd.shift(1)
    macd_hist_min = float(P.get("macd_hist_min", 0.0))
    macd_hist_mult = float(P.get("macd_hist_mult", 1.01))
    if "MACD_Hist_D" in cols:
        macd_boost = ((macd >= macd_hist_min) & (macd >= macd_prev * macd_hist_mult)).fillna(False)
    else:
        macd_boost = pd.Series(False, index=df.index)

    # Volume booster
    vol_mult = float(P.get("vol_mult", 1.5))
    if "volume_D" in cols:
        vol_mean_20 = vol.rolling(20, min_periods=20).mean()
        vol_boost = (vol >= (vol_mean_20 * vol_mult)).fillna(False)
    else:
        vol_boost = pd.Series(False, index=df.index)

    # Day-change booster
    day_chg = _col("Daily_Change_D", 0.0).fillna(0.0)
    day_change_min = float(P.get("day_change_min", 1.3))
    day_boost = (day_chg >= day_change_min).fillna(False)

    boosters = (
        rsi_boost.astype(int)
        + macd_boost.astype(int)
        + vol_boost.astype(int)
        + day_boost.astype(int)
    )
    momentum_ok = boosters >= boosters_required

    # ------------------- TREND ALIGNMENT + ANTI-CHASE -------------------
    above_ema20 = pd.Series(True, index=df.index)
    above_ema50 = pd.Series(True, index=df.index)
    not_too_far = pd.Series(True, index=df.index)

    if P.get("require_close_above_ema20", True) and ("EMA_20_D" in cols):
        above_ema20 = (close >= ema20).fillna(False)
        anti_mult = float(P.get("anti_chase_ema20_mult", 1.05))
        not_too_far = (close <= ema20 * anti_mult).fillna(False)

    if P.get("require_close_above_ema50", False) and ("EMA_50_D" in cols):
        above_ema50 = (close >= ema50).fillna(False)

    # Optional ADX daily
    if P.get("d_adx_min") is not None and ("ADX_D" in cols):
        adx_d_ok = (df["ADX_D"] >= float(P["d_adx_min"])).fillna(False)
    else:
        adx_d_ok = pd.Series(True, index=df.index)

    # ------------------- FINAL ENTRY MASK -------------------
    exec_ok = (
        tradable_ok
        & breakout_ok
        & green & body_ok
        & momentum_ok
        & above_ema20 & above_ema50
        & not_too_far
        & adx_d_ok
    )

    long_mask = weekly_ok & daily_prev_ok & exec_ok
    if not long_mask.any():
        return []

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_D"]),
            "strategy": P.get("name", "CUSTOMXX"),
            "sub_strategy": (
                f"W(prev_week)_strict + Dprev(D-1) + {breakout_name} + "
                f"boosters>={boosters_required} + anti_chase"
            ),
        })
    return signals


# ======================================================================
#                     10 POSITIONAL STRATEGIES
# ======================================================================

STRATEGY_PARAMS: dict[str, dict] = {
    # Softer continuation (still trend + breakout, but fewer boosters)
    "custom01": dict(
        name="CUSTOM01",
        wk_close_above_ema200=True,
        wk_rsi_min=45, wk_adx_min=None,
        dprev_min=0.0, dprev_max=6.0,
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
        day_change_min=0.8,
        anti_chase_ema20_mult=1.07,
        min_price=20.0,
    ),

    # More trend alignment
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

    # Adds weekly MA structure
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

    # Momentum first (3 boosters)
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

    # Stricter breakout + anti-chase
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

    # Adds liquidity proxy (volume SMA20 threshold)
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
        min_vol_sma20=200000,  # adjust for your universe
    ),

    # “Very strong trend” weekly
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

    # Requires 4 boosters (max momentum)
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

    # Tighter anti-chase + stricter weekly separation
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

    # CUSTOM10: “continuously rising / max strict continuation”
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


def build_signals_for_strategies(ticker: str, strategy_keys: list[str]) -> list[dict]:
    """Run one or many strategies for a single ticker and return combined signals."""
    all_sigs: list[dict] = []
    for key in strategy_keys:
        key_lower = key.lower()
        if key_lower not in STRATEGY_PARAMS:
            print(f"! Unknown strategy key '{key}', skipping for ticker {ticker}")
            continue
        P = STRATEGY_PARAMS[key_lower]
        sigs = _build_signals_positional_template(ticker, P)
        all_sigs.extend(sigs)
    return all_sigs


def build_signals_selected(ticker: str) -> list[dict]:
    """
    Backward-compatible wrapper:
    uses DEFAULT_STRATEGY_KEYS when called from external code.
    """
    return build_signals_for_strategies(ticker, DEFAULT_STRATEGY_KEYS)


# ----------------------- EVALUATION ON DAILY -----------------------

def _safe_read_daily(path: str) -> pd.DataFrame:
    return _safe_read_tf(path, tz=IST)


def evaluate_signal_on_daily_long_only(row: pd.Series, df_d: pd.DataFrame) -> dict:
    """
    Positional evaluation:
      - looks strictly AFTER entry day (no same-bar evaluation)
      - if future high hits target => exit at target on first hit day
      - else exit at last available future close
    """
    entry_time = pd.to_datetime(row["signal_time_ist"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize(IST)
    else:
        entry_time = entry_time.tz_convert(IST)

    entry_price = float(row["entry_price"])
    if (not np.isfinite(entry_price)) or entry_price <= 0 or df_d is None or df_d.empty:
        return {
            HIT_COL: False, TIME_TD_COL: np.nan, TIME_CAL_COL: np.nan,
            "exit_time": entry_time, "exit_price": entry_price,
            "pnl_pct": 0.0, "pnl_rs": 0.0,
            "max_favorable_pct": np.nan, "max_adverse_pct": np.nan,
        }

    dfx = df_d.copy()
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
    hit_mask = future["high"] >= target_price

    max_fav = ((future["high"] - entry_price) / entry_price * 100.0).max()
    max_adv = ((future["low"] - entry_price) / entry_price * 100.0).min()

    if hit_mask.any():
        first_hit_pos = int(np.argmax(hit_mask.values))
        hit_row = future.iloc[first_hit_pos]
        exit_time = hit_row["date"]
        exit_price = float(target_price)
        dt_days = (exit_time - entry_time).days
        return {
            HIT_COL: True,
            TIME_TD_COL: float(dt_days),
            TIME_CAL_COL: float(dt_days),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl_pct": float((exit_price - entry_price) / entry_price * 100.0),
            "pnl_rs": float(CAPITAL_PER_SIGNAL_RS * ((exit_price - entry_price) / entry_price)),
            "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
            "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
        }
    else:
        last_row = future.iloc[-1]
        exit_time = last_row["date"]
        exit_price = float(last_row["close"])
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        return {
            HIT_COL: False,
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl_pct": float(pnl_pct),
            "pnl_rs": float(CAPITAL_PER_SIGNAL_RS * (pnl_pct / 100.0)),
            "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
            "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
        }


# ----------------------- CAPITAL-CONSTRAINED PORTFOLIO SIM -----------------------

def apply_capital_constraint(out_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"], utc=True, errors="coerce").dt.tz_convert(IST)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    if df["exit_time"].notna().any():
        df.loc[df["exit_time"].notna(), "exit_time"] = df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)

    df["pnl_rs"]  = pd.to_numeric(df.get("pnl_rs", 0.0), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0.0), errors="coerce").fillna(0.0)

    df["invested_amount"] = CAPITAL_PER_SIGNAL_RS
    df["final_value_per_signal"] = df["invested_amount"] + df["pnl_rs"]

    available_capital = MAX_CAPITAL_RS
    open_trades: list[dict] = []
    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        still_open = []
        for tr in open_trades:
            if pd.isna(tr["exit_time"]) or tr["exit_time"] <= entry_time:
                available_capital += tr["value"]
            else:
                still_open.append(tr)
        open_trades = still_open

        if available_capital >= CAPITAL_PER_SIGNAL_RS:
            df.at[idx, "taken"] = True
            available_capital -= CAPITAL_PER_SIGNAL_RS
            exit_time = row["exit_time"]
            if pd.isna(exit_time):
                exit_time = entry_time
            open_trades.append({"exit_time": exit_time, "value": float(row["final_value_per_signal"])})
        else:
            df.at[idx, "taken"] = False

    for tr in open_trades:
        available_capital += tr["value"]

    df["invested_amount_constrained"] = np.where(df["taken"], CAPITAL_PER_SIGNAL_RS, 0.0)
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
    """
    Parse strategy keys from CLI args.

    Usage examples:
        python script.py                 -> uses DEFAULT_STRATEGY_KEYS
        python script.py custom04        -> only custom04
        python script.py custom04 custom10
        python script.py all             -> all 10 strategies
    """
    args = [a.lower().strip() for a in sys.argv[1:] if a.strip()]
    if not args:
        return DEFAULT_STRATEGY_KEYS

    # If single 'all' -> expand to all strategies
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
    tickers_d = _list_tickers_from_dir(DIR_D)
    tickers_w = _list_tickers_from_dir(DIR_W)
    tickers = tickers_d & tickers_w

    if not tickers:
        print("No common tickers found across Daily and Weekly directories.")
        return

    strategy_keys = _parse_strategy_keys_from_argv()
    strategy_label = "+".join(strategy_keys)

    print(f"Found {len(tickers)} tickers with both Daily+Weekly.")
    print(f"Running strategies: {strategy_label}\n")

    all_signals: list[dict] = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        try:
            sigs = build_signals_for_strategies(ticker, strategy_keys)
            all_signals.extend(sigs)
            if i % 100 == 0:
                print(f"[{i}/{len(tickers)}] ... processed")
        except Exception as e:
            print(f"! {ticker}: {e}")

    if not all_signals:
        print("No LONG signals generated.")
        return

    sig_df = (
        pd.DataFrame(all_signals)
        .sort_values(["signal_time_ist", "ticker", "strategy"])
        .reset_index(drop=True)
    )
    sig_df["itr"] = np.arange(1, len(sig_df) + 1)
    sig_df["signal_time_ist"] = pd.to_datetime(
        sig_df["signal_time_ist"], utc=True, errors="coerce"
    ).dt.tz_convert(IST)
    sig_df["signal_date"] = sig_df["signal_time_ist"].dt.date

    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")
    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_{strategy_label}_{ts}_IST.csv")
    sig_df.to_csv(signals_path, index=False)
    print(f"\nSaved signals: {signals_path} (rows={len(sig_df)})")

    # Evaluate
    data_cache: dict[str, pd.DataFrame] = {}
    for tkr in sig_df["ticker"].unique():
        p = _find_file(DIR_D, tkr)
        data_cache[tkr] = _safe_read_daily(p) if p else pd.DataFrame()

    results = []
    for _, row in sig_df.iterrows():
        tkr = row["ticker"]
        df_d = data_cache.get(tkr, pd.DataFrame())
        new_info = (
            evaluate_signal_on_daily_long_only(row, df_d)
            if not df_d.empty else {
                HIT_COL: False, TIME_TD_COL: np.nan, TIME_CAL_COL: np.nan,
                "exit_time": row["signal_time_ist"], "entry_price": row["entry_price"],
                "pnl_pct": 0.0, "pnl_rs": 0.0,
                "max_favorable_pct": np.nan, "max_adverse_pct": np.nan,
            }
        )
        rr = row.to_dict()
        rr.update(new_info)
        results.append(rr)

    out_df = (
        pd.DataFrame(results)
        .sort_values(["signal_time_ist", "ticker", "strategy"])
        .reset_index(drop=True)
    )

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_{strategy_label}_{EVAL_SUFFIX}_{ts}_IST.csv")
    out_df.to_csv(eval_path, index=False)
    print(f"Saved eval: {eval_path}")

    # Stats (overall across strategies)
    total = len(out_df)
    hits = int(out_df[HIT_COL].sum())
    hit_pct = (hits / total * 100.0) if total else 0.0
    print(f"\n[ALL STRATEGIES] Total={total}, hit {TARGET_LABEL}={hits} ({hit_pct:.1f}%)")

    if total > N_RECENT_SKIP:
        nr = out_df.iloc[:-N_RECENT_SKIP]
        hits_nr = int(nr[HIT_COL].sum())
        nr_total = len(nr)
        hit_pct_nr = (hits_nr / nr_total * 100.0) if nr_total else 0.0
        print(f"[NON-RECENT excl last {N_RECENT_SKIP}] Total={nr_total}, hits={hits_nr} ({hit_pct_nr:.1f}%)")

    # Capital constrained
    out_df_constrained, cap = apply_capital_constraint(out_df)
    out_df_constrained.to_csv(eval_path, index=False)
    print("\n--- CAPITAL-CONSTRAINED (ALL STRATEGIES COMBINED) ---")
    print(f"Signals generated: {cap['total_signals']}")
    print(f"Signals taken    : {cap['signals_taken']}")
    print(f"Final value      : ₹{cap['final_portfolio_value']:,.2f}")
    print(f"Net P&L          : ₹{cap['net_pnl_rs']:,.2f} ({cap['net_pnl_pct']:.2f}%)")

    # Daily summary
    daily_summary = build_daily_summary(out_df_constrained)
    daily_summary_path = os.path.join(
        OUT_DIR,
        f"multi_tf_daily_summary_{strategy_label}_{EVAL_SUFFIX}_{ts}_IST.csv",
    )
    daily_summary.to_csv(daily_summary_path, index=False)
    print(f"\nSaved daily summary: {daily_summary_path}")


if __name__ == "__main__":
    main()
