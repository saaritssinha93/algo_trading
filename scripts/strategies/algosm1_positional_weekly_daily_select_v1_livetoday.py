# -*- coding: utf-8 -*-
"""
TODAY-ONLY positional LONG signal generator (DAILY execution, WEEKLY context).

AS-OF RULES (IST):
  - Daily data used: strictly BEFORE today's 00:00 IST (=> up to yesterday close)
  - Weekly context: previous fully completed week only (no same-week lookahead):
        weekly_date_shifted = weekly_date + 1 day
        merge_asof(direction="backward") onto Daily timeline

OUTPUT:
  - Generates entries ONLY for "today" (run date in IST)
  - Uses whole dataset history to compute rolling levels, but with AS-OF cut for context
  - One signal per ticker for today
  - If a "today output" already exists in OUT_DIR, it reuses it and exits.
"""

import os
import glob
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals_today"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

NOW_IST = datetime.now(IST)
TODAY_START_IST = NOW_IST.replace(hour=0, minute=0, second=0, microsecond=0)
TODAY_DATE = TODAY_START_IST.date()

# ----------------------- STRATEGY SELECTION -----------------------
DEFAULT_STRATEGY_KEYS: list[str] = [
    "custom10", "custom09", "custom08", "custom07", "custom06",
    "custom02", "custom01", "custom05", "custom04", "custom03",
]

# ----------------------- HELPERS -----------------------
def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """Read CSV, parse 'date' as tz-aware IST, sort ascending."""
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(tz)
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

def _apply_asof_cut(df: pd.DataFrame, cutoff_dt_ist: datetime) -> pd.DataFrame:
    """Keep rows strictly before cutoff_dt_ist (IST)."""
    if df.empty or "date" not in df.columns:
        return df
    return df[df["date"] < cutoff_dt_ist].copy()

# ======================================================================
#                 MTF LOADER (DAILY + WEEKLY) AS-OF
# ======================================================================
def _load_mtf_context_daily_weekly_asof(ticker: str) -> pd.DataFrame:
    """
    Returns a DAILY-indexed dataframe with WEEKLY columns merged in (as-of),
    using AS-OF CUT:
      - Daily: < TODAY_START_IST
      - Weekly: < TODAY_START_IST
    Then:
      - Daily prev fields computed (shift(1))
      - Weekly 'date' shifted by +1 day and merge_asof(backward)
    """
    path_d = _find_file(DIR_D, ticker)
    path_w = _find_file(DIR_W, ticker)
    if not path_d or not path_w:
        return pd.DataFrame()

    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)
    if df_d.empty or df_w.empty:
        return pd.DataFrame()

    # AS-OF CUTS (previous daily / previous fully completed weekly context)
    df_d = _apply_asof_cut(df_d, TODAY_START_IST)
    df_w = _apply_asof_cut(df_w, TODAY_START_IST)
    if df_d.empty or df_w.empty:
        return pd.DataFrame()

    # Daily prev-day change (D-1)
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = pd.to_numeric(df_d["Daily_Change"], errors="coerce").shift(1)
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
def _build_today_signal_for_ticker(ticker: str, P: dict) -> dict | None:
    """
    Build a TODAY-ONLY signal (single row) for a ticker.

    We evaluate conditions on the latest available DAILY bar (yesterday close),
    while weekly context is previous completed week (as-of merge).
    Output signal_time_ist is TODAY_START_IST (i.e., today).
    """
    df = _load_mtf_context_daily_weekly_asof(ticker)
    if df.empty:
        return None

    cols = set(df.columns)
    need = {"date", "open_D", "high_D", "low_D", "close_D"}
    if not need.issubset(cols):
        return None

    # Use the last available daily bar BEFORE today (i.e., yesterday bar in df)
    r = df.iloc[-1].copy()

    def _get(name: str, default=np.nan):
        return r[name] if name in cols else default

    close = float(_get("close_D", np.nan))
    open_ = float(_get("open_D", np.nan))
    high = float(_get("high_D", np.nan))
    if (not np.isfinite(close)) or close <= 0:
        return None

    # ------------------- TRADABILITY -------------------
    if P.get("min_price") is not None:
        if close < float(P["min_price"]):
            return None

    # ------------------- WEEKLY FILTER -------------------
    # (already previous-week as-of merged)
    if P.get("wk_close_above_ema200", True) and {"close_W", "EMA_200_W"}.issubset(cols):
        if not (_get("close_W") > _get("EMA_200_W")):
            return None

    if P.get("wk_close_above_ema50", False) and {"close_W", "EMA_50_W"}.issubset(cols):
        if not (_get("close_W") > _get("EMA_50_W")):
            return None

    if P.get("wk_ema50_over_ema200_min") is not None and {"EMA_50_W", "EMA_200_W"}.issubset(cols):
        if not (_get("EMA_50_W") >= _get("EMA_200_W") * float(P["wk_ema50_over_ema200_min"])):
            return None

    if P.get("wk_rsi_min") is not None and ("RSI_W" in cols):
        if not (_get("RSI_W") >= float(P["wk_rsi_min"])):
            return None

    if P.get("wk_adx_min") is not None and ("ADX_W" in cols):
        if not (_get("ADX_W") >= float(P["wk_adx_min"])):
            return None

    # ------------------- DAILY PREV FILTER (D-1) -------------------
    dprev = _get("Daily_Change_prev_D", np.nan)
    if P.get("dprev_min") is not None and (not np.isfinite(dprev) or dprev < float(P["dprev_min"])):
        return None
    if P.get("dprev_max") is not None and (np.isfinite(dprev) and dprev > float(P["dprev_max"])):
        return None

    # ------------------- BREAKOUT LEVELS (NO LOOKAHEAD) -------------------
    # We compute breakout from full df history (already cut as-of today).
    high_s = pd.to_numeric(df["high_D"], errors="coerce")

    breakout_windows = P.get("breakout_windows", [20, 50, 100])
    prior_levels = [high_s.rolling(int(w), min_periods=int(w)).max().shift(1) for w in breakout_windows]

    breakout_level = prior_levels[0].iloc[-1]
    for i in range(1, len(prior_levels)):
        if pd.isna(breakout_level):
            breakout_level = prior_levels[i].iloc[-1]

    if not np.isfinite(breakout_level) or breakout_level <= 0:
        return None

    close_mult = float(P.get("breakout_close_mult", 1.0015))
    wick_mult_hi = float(P.get("breakout_wick_mult_hi", 1.0010))
    wick_mult_close = float(P.get("breakout_wick_mult_close", 1.0005))

    close_break = close >= breakout_level * close_mult
    wick_break  = (high >= breakout_level * wick_mult_hi) and (close >= breakout_level * wick_mult_close)
    if not (close_break or wick_break):
        return None

    # ------------------- CANDLE QUALITY -------------------
    if not (close > open_):
        return None
    body_min_pct = float(P.get("body_min_pct", 0.0040))
    if not ((close - open_) >= (open_ * body_min_pct)):
        return None

    # ------------------- MOMENTUM BOOSTERS -------------------
    boosters_required = int(P.get("boosters_required", 3))
    boosters = 0

    # RSI booster
    if "RSI_D" in cols:
        rsi_s = pd.to_numeric(df["RSI_D"], errors="coerce")
        rsi = float(rsi_s.iloc[-1])
        rsi_prev = float(rsi_s.shift(1).iloc[-1]) if len(rsi_s) >= 2 else np.nan
        rsi_min = float(P.get("rsi_min", 58.0))
        rsi_delta_min = float(P.get("rsi_delta_min", 1.2))
        if np.isfinite(rsi) and (rsi >= rsi_min or (np.isfinite(rsi_prev) and rsi >= rsi_prev + rsi_delta_min)):
            boosters += 1

    # MACD Hist booster
    if "MACD_Hist_D" in cols:
        m_s = pd.to_numeric(df["MACD_Hist_D"], errors="coerce")
        macd = float(m_s.iloc[-1])
        macd_prev = float(m_s.shift(1).iloc[-1]) if len(m_s) >= 2 else np.nan
        macd_hist_min = float(P.get("macd_hist_min", 0.0))
        macd_hist_mult = float(P.get("macd_hist_mult", 1.01))
        if np.isfinite(macd) and np.isfinite(macd_prev) and (macd >= macd_hist_min) and (macd >= macd_prev * macd_hist_mult):
            boosters += 1

    # Volume booster
    if "volume_D" in cols:
        vol_s = pd.to_numeric(df["volume_D"], errors="coerce")
        vol = float(vol_s.iloc[-1])
        vol_mult = float(P.get("vol_mult", 1.5))
        vol_mean_20 = float(vol_s.rolling(20, min_periods=20).mean().iloc[-1])
        if np.isfinite(vol) and np.isfinite(vol_mean_20) and (vol >= vol_mean_20 * vol_mult):
            boosters += 1

    # Day change booster
    if "Daily_Change_D" in cols:
        chg = float(pd.to_numeric(df["Daily_Change_D"], errors="coerce").iloc[-1])
        day_change_min = float(P.get("day_change_min", 1.3))
        if np.isfinite(chg) and chg >= day_change_min:
            boosters += 1

    if boosters < boosters_required:
        return None

    # ------------------- TREND ALIGNMENT + ANTI-CHASE -------------------
    if P.get("require_close_above_ema20", True) and ("EMA_20_D" in cols):
        ema20 = float(pd.to_numeric(df["EMA_20_D"], errors="coerce").iloc[-1])
        if not (np.isfinite(ema20) and close >= ema20):
            return None
        anti_mult = float(P.get("anti_chase_ema20_mult", 1.05))
        if not (close <= ema20 * anti_mult):
            return None

    if P.get("require_close_above_ema50", False) and ("EMA_50_D" in cols):
        ema50 = float(pd.to_numeric(df["EMA_50_D"], errors="coerce").iloc[-1])
        if not (np.isfinite(ema50) and close >= ema50):
            return None

    if P.get("d_adx_min") is not None and ("ADX_D" in cols):
        adx = float(pd.to_numeric(df["ADX_D"], errors="coerce").iloc[-1])
        if not (np.isfinite(adx) and adx >= float(P["d_adx_min"])):
            return None

    # PASSED: emit TODAY signal
    return {
        "signal_time_ist": TODAY_START_IST,   # today-only output
        "asof_daily_bar": r["date"],          # yesterday daily bar timestamp
        "ticker": ticker,
        "signal_side": "LONG",
        "entry_price": float(close),          # using yesterday close as reference entry
        "strategy": P.get("name", "CUSTOMXX"),
        "note": "TODAY signal computed using yesterday Daily + previous completed Weekly (no lookahead).",
    }

# ======================================================================
#                     10 POSITIONAL STRATEGIES
# ======================================================================
STRATEGY_PARAMS: dict[str, dict] = {
    "custom01": dict(
        name="CUSTOM01",
        wk_close_above_ema200=True,
        wk_rsi_min=45, wk_adx_min=None,
        dprev_min=0.0, dprev_max=6.0,
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
    "custom02": dict(
        name="CUSTOM02",
        wk_close_above_ema200=True,
        wk_rsi_min=48, wk_adx_min=16,
        dprev_min=0.0, dprev_max=5.5,
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

# ----------------------- CLI STRATEGY SELECTION -----------------------
def _parse_strategy_keys_from_argv() -> list[str]:
    args = [a.lower().strip() for a in sys.argv[1:] if a.strip()]
    if not args:
        return DEFAULT_STRATEGY_KEYS
    if len(args) == 1 and args[0] == "all":
        return sorted(STRATEGY_PARAMS.keys())
    selected = [a for a in args if a in STRATEGY_PARAMS]
    if not selected:
        print("! No valid strategy keys from argv; using defaults.")
        return DEFAULT_STRATEGY_KEYS
    return selected

# ----------------------- OUTPUT REUSE -----------------------
def _latest_today_output_if_exists(strategy_label: str) -> str | None:
    """
    If an output file for TODAY exists, return its path.
    We look for: signals_today_<strategies>_YYYYMMDD_*.csv
    """
    ymd = TODAY_START_IST.strftime("%Y%m%d")
    patt = os.path.join(OUT_DIR, f"signals_today_{strategy_label}_{ymd}_*.csv")
    hits = sorted(glob.glob(patt), key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0] if hits else None

# ----------------------- MAIN -----------------------
def main():
    strategy_keys = _parse_strategy_keys_from_argv()
    strategy_label = "+".join(strategy_keys)

    print(f"\nRUN TIME (IST): {NOW_IST.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"TODAY (IST):    {TODAY_DATE}")
    print(f"AS-OF CUT:      using ONLY data before {TODAY_START_IST.strftime('%Y-%m-%d %H:%M:%S %Z')} (yesterday daily, prev completed weekly)\n")

    # reuse if today's file exists
    existing = _latest_today_output_if_exists(strategy_label)
    if existing:
        print(f"Found existing TODAY output. Reusing: {existing}")
        try:
            df_existing = pd.read_csv(existing)
            print(f"Rows: {len(df_existing)}")
        except Exception as e:
            print(f"! Could not read existing file, will regenerate. Error: {e}")
        else:
            return

    tickers_d = _list_tickers_from_dir(DIR_D)
    tickers_w = _list_tickers_from_dir(DIR_W)
    tickers = sorted(tickers_d & tickers_w)
    if not tickers:
        print("No common tickers found across Daily and Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with both Daily+Weekly.")
    print(f"Running strategies: {strategy_label}\n")

    # Build TODAY signals
    rows = []
    for i, ticker in enumerate(tickers, start=1):
        best_row = None

        # Try strategies in order; first match wins (so you can control strictness priority)
        for key in strategy_keys:
            P = STRATEGY_PARAMS.get(key.lower())
            if not P:
                continue
            sig = _build_today_signal_for_ticker(ticker, P)
            if sig is not None:
                best_row = sig
                break

        if best_row is not None:
            rows.append(best_row)

        if i % 200 == 0:
            print(f"[{i}/{len(tickers)}] processed... today_signals={len(rows)}")

    if not rows:
        print("No TODAY signals generated.")
        return

    out_df = pd.DataFrame(rows)
    out_df["signal_time_ist"] = pd.to_datetime(out_df["signal_time_ist"])
    out_df["signal_date"] = pd.to_datetime(out_df["signal_time_ist"]).dt.date

    # Ensure only TODAY rows (hard guarantee)
    out_df = out_df[out_df["signal_date"] == TODAY_DATE].copy()

    # One row per ticker for today (already 1, but enforce)
    out_df = out_df.sort_values(["ticker", "strategy"]).drop_duplicates(subset=["ticker", "signal_date"], keep="first")

    ymd = TODAY_START_IST.strftime("%Y%m%d")
    ts = NOW_IST.strftime("%H%M%S")
    out_path = os.path.join(OUT_DIR, f"signals_today_{ymd}_{ts}_IST.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved TODAY signals: {out_path}")
    print(f"Rows: {len(out_df)}")

if __name__ == "__main__":
    main()
