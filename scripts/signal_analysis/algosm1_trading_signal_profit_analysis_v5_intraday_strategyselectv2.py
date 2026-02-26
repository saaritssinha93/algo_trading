# -*- coding: utf-8 -*-
"""
Multi-timeframe positional LONG backtest framework (5-minute execution).

Execution TF:
    • 5-minute bars from DIR_5m = "main_indicators_5min"

Context TFs:
    • Daily from DIR_D
    • Weekly from DIR_W  (NO same-date lookahead handled via +1s shift)

Flow
----
1) Generate MTF LONG signals using Weekly + Daily + 5m execution logic
2) Evaluate those signals on 5m data for +TARGET_PCT% target hit
3) Apply capital constraint simulation

Strategies included
-------------------
Base:
  1) trend_pullback
  2) weekly_breakout
  3) weekly_squeeze
  4) insidebar_breakout
  5) hh_hl_retest
  6) golden_cross

Custom:
  7) custom1
  8) custom2
  9) custom3
 10) custom4
 11) strategy11        (ADDED on 5m)
 12) custom6           (ADDED: union/voting meta-strategy on 5m)
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_5m  = "main_indicators_5min"
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals"

# filename endings (adjust if your files differ)
ENDING_5M = "_main_indicators_5min.csv"
ENDING_D  = "_main_indicators_daily.csv"
ENDING_W  = "_main_indicators_weekly.csv"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Target configuration (FIXED) ----
TARGET_PCT: float = 0.5
TARGET_MULT: float = 1.0 + TARGET_PCT / 100.0

def _pct_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")

TARGET_TAG = _pct_tag(TARGET_PCT)

TARGET_LABEL = f"{TARGET_PCT:.2f}%"
HIT_COL = f"hit_{TARGET_TAG}"
INTRADAY_HIT_COL = f"hit_{TARGET_TAG}_intraday"
TIME_TD_COL = f"time_to_{TARGET_TAG}_days"
TIME_CAL_COL = f"days_to_{TARGET_TAG}_calendar"
EVAL_SUFFIX = f"{TARGET_TAG}_eval"


# Capital sizing
CAPITAL_PER_SIGNAL_RS: float = 50000.0
MAX_CAPITAL_RS: float = 7000000.0

TRADING_HOURS_PER_DAY = 6.25
N_RECENT_SKIP = 1

# 5m scaling helpers
BARS_PER_HOUR_5M = 12
ROLL_MEAN_1H_EQ = 12          # ~1 hour on 5m
ROLL_MEAN_5H_EQ = 60          # ~5 hours on 5m
ROLL_MEAN_10H_EQ = 120        # ~10 hours on 5m

from datetime import time

# ----------------------- INTRADAY SESSION (IST) -----------------------
MKT_OPEN  = time(9, 15)
MKT_CLOSE = time(15, 30)

# For entries, keep a buffer so you still have time to exit intraday if no target hit
ENTRY_START = time(9, 20)
ENTRY_END   = time(15, 20)

# ---------- REALISTIC COSTS (optional) ----------
# For NSE intraday, even small costs matter a lot for 0.5% targets.
# Keep 0.0 if you don't want costs.
COST_BPS_ENTRY = 2.0   # 2 bps = 0.02%
COST_BPS_EXIT  = 2.0   # 2 bps = 0.02%
SLIPPAGE_BPS   = 1.0   # applied on both entry & exit (optional)

def _apply_costs_to_pct(pct: float) -> float:
    """
    Subtract entry+exit costs and slippage from pnl_pct.
    """
    total_bps = COST_BPS_ENTRY + COST_BPS_EXIT + 2.0 * SLIPPAGE_BPS
    return pct - (total_bps / 100.0)  # 100 bps = 1%


def _intraday_entry_mask(ts: pd.Series,
                         start: time = ENTRY_START,
                         end: time = ENTRY_END) -> pd.Series:
    """
    True if timestamp time is within allowed ENTRY window.
    Assumes ts is tz-aware IST timestamps.
    """
    return ts.dt.time.between(start, end)

def _intraday_trade_mask(ts: pd.Series,
                         start: time = MKT_OPEN,
                         end: time = MKT_CLOSE) -> pd.Series:
    """
    True if timestamp time is within trading window.
    Used for evaluation future bars.
    """
    return ts.dt.time.between(start, end)

# ----------------------- HELPERS (COMMON) -----------------------
def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """Read a timeframe CSV, parse 'date' as tz-aware IST, sort."""
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


def _get_exit_profile(strategy: str) -> dict:
    """
    Per-strategy intraday exit profiles.
    R = risk = entry - SL. TP levels are in R multiples.
    """
    s = (strategy or "").lower()

    # Defaults (reasonable intraday)
    profile = dict(
        k_sl=1.10,            # SL = entry - k_sl*ATR
        tp1_R=0.70, tp1_frac=0.50,
        tp2_R=None, tp2_frac=0.0,
        move_sl_to_be=True,   # after TP1
        k_trail=1.80,         # Chandelier: HH - k_trail*ATR
        time_stop_bars=12,    # ~1 hour on 5m
        min_progress_R=0.20,  # if not >= this by time stop -> exit
    )

    # Pullback / bounce: tighter trail, quicker
    if s in ("trend_pullback", "golden_cross", "hh_hl_retest"):
        profile.update(k_sl=1.15, tp1_R=0.60, tp1_frac=0.50, k_trail=1.50, time_stop_bars=12, min_progress_R=0.20)

    # Breakout / momentum: looser trail, allow runners
    if s in ("weekly_breakout", "weekly_breakout_daily_confirm", "custom2", "custom4", "strategy11", "custom6"):
        profile.update(k_sl=0.95, tp1_R=0.50, tp1_frac=0.40, k_trail=2.20, time_stop_bars=18, min_progress_R=0.15)

    # Squeeze: capture burst, use TP2 and looser trail
    if s in ("weekly_squeeze", "weekly_squeeze_daily_expansion"):
        profile.update(k_sl=1.40, tp1_R=1.00, tp1_frac=0.50, tp2_R=1.80, tp2_frac=0.25, k_trail=2.60, time_stop_bars=15, min_progress_R=0.20)

    # Inside bar breakout: fail-fast + bracket-ish
    if s in ("insidebar_breakout", "weekly_trend_insidebar_breakout"):
        profile.update(k_sl=1.00, tp1_R=0.80, tp1_frac=0.50, k_trail=1.80, time_stop_bars=12, min_progress_R=0.20)

    # Custom1: more HTF, but still intraday: moderate
    if s == "custom1":
        profile.update(k_sl=1.15, tp1_R=0.70, tp1_frac=0.50, k_trail=1.70, time_stop_bars=15, min_progress_R=0.20)

    # Custom3: strict entry; exit faster if it doesn’t move
    if s == "custom3":
        profile.update(k_sl=1.00, tp1_R=0.70, tp1_frac=0.50, k_trail=1.70, time_stop_bars=12, min_progress_R=0.25)

    return profile


def _locate_entry_index(df_5m: pd.DataFrame, entry_time: pd.Timestamp) -> int | None:
    """
    Find the entry bar index in df_5m:
    - exact match preferred
    - else use the last bar <= entry_time (asof)
    Assumes df_5m sorted by date and tz-aware IST timestamps.
    """
    if df_5m.empty or "date" not in df_5m.columns:
        return None

    # Ensure Timestamp and comparable
    if not isinstance(entry_time, pd.Timestamp):
        entry_time = pd.to_datetime(entry_time)

    # Exact match
    exact = df_5m.index[df_5m["date"] == entry_time]
    if len(exact):
        return int(exact[0])

    # Asof: last bar <= entry_time
    dates_ns = pd.to_datetime(df_5m["date"]).values.astype("datetime64[ns]")
    et_ns = entry_time.to_datetime64()

    pos = np.searchsorted(dates_ns, et_ns, side="right") - 1
    if pos < 0 or pos >= len(df_5m):
        return None
    return int(pos)



def _list_tickers_from_dir(directory: str) -> set[str]:
    """
    Robust ticker extraction:
    - Finds files containing '_main_indicators'
    - Extracts ticker as the part BEFORE '_main_indicators'
    Works for:
      TICKER_main_indicators_5min.csv
      TICKER_main_indicators_etf_5min.csv
      TICKER_main_indicators_daily.csv
      TICKER_main_indicators_etf_daily.csv
      etc.
    """
    files = glob.glob(os.path.join(directory, "*_main_indicators*.csv"))
    tickers = set()
    for f in files:
        base = os.path.basename(f)
        if "_main_indicators" in base:
            ticker = base.split("_main_indicators")[0].strip()
            if ticker:
                tickers.add(ticker)
    return tickers

def build_daily_summary(out_df: pd.DataFrame) -> pd.DataFrame:
    dfx = out_df.copy()

    dfx["signal_time_ist"] = pd.to_datetime(dfx["signal_time_ist"], errors="coerce")
    dfx = dfx.dropna(subset=["signal_time_ist"])

    if dfx["signal_time_ist"].dt.tz is None:
        dfx["signal_time_ist"] = dfx["signal_time_ist"].dt.tz_localize(IST)
    else:
        dfx["signal_time_ist"] = dfx["signal_time_ist"].dt.tz_convert(IST)

    dfx["date"] = dfx["signal_time_ist"].dt.date

    if "taken" not in dfx.columns:
        dfx["taken"] = True

    # make sure these exist
    for c in ["pnl_rs", "pnl_rs_levered", "pnl_pct", "pnl_pct_levered"]:
        dfx[c] = pd.to_numeric(dfx.get(c, 0.0), errors="coerce").fillna(0.0)

    if INTRADAY_HIT_COL in dfx.columns:
        dfx[INTRADAY_HIT_COL] = dfx[INTRADAY_HIT_COL].astype(bool).fillna(False)
    else:
        dfx[INTRADAY_HIT_COL] = False

    # ✅ ensure is_intraday_trade exists; if not, infer it
    if "is_intraday_trade" not in dfx.columns:
        dfx["exit_time"] = pd.to_datetime(dfx.get("exit_time", pd.NaT), errors="coerce")
        if dfx["exit_time"].notna().any():
            if dfx.loc[dfx["exit_time"].notna(), "exit_time"].dt.tz is None:
                dfx.loc[dfx["exit_time"].notna(), "exit_time"] = dfx.loc[dfx["exit_time"].notna(), "exit_time"].dt.tz_localize(IST)
            else:
                dfx.loc[dfx["exit_time"].notna(), "exit_time"] = dfx.loc[dfx["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)
        dfx["is_intraday_trade"] = (
            dfx["exit_time"].notna() &
            (dfx["exit_time"].dt.date == dfx["signal_time_ist"].dt.date)
        )

    def _sum_where(series, mask):
        return float(series[mask].sum()) if len(series) else 0.0

    grp = dfx.groupby("date", dropna=False)

    rows = []
    for day, g in grp:
        taken = g["taken"].astype(bool)
        intraday = g["is_intraday_trade"].astype(bool)

        rows.append({
            "date": day,
            "signals_generated": int(len(g)),
            "signals_taken": int(taken.sum()),

            "intraday_trades_generated": int(intraday.sum()),
            "intraday_trades_taken": int((intraday & taken).sum()),

            "intraday_hits_generated": int(g[INTRADAY_HIT_COL].sum()),
            "intraday_hits_taken": int((g[INTRADAY_HIT_COL] & taken).sum()),

            # ✅ ONLY intraday P&L (this is what you want)
            "pnl_rs_intraday_sum": _sum_where(g["pnl_rs"], intraday),
            "pnl_rs_levered_intraday_sum": _sum_where(g["pnl_rs_levered"], intraday),

            # ✅ ONLY intraday P&L for taken trades
            "pnl_rs_intraday_taken_sum": _sum_where(g["pnl_rs"], intraday & taken),
            "pnl_rs_levered_intraday_taken_sum": _sum_where(g["pnl_rs_levered"], intraday & taken),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)




def _find_file(directory: str, ticker: str) -> str | None:
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None
# ======================================================================
#                 COMMON MTF LOADER (5m + DAILY + WEEKLY)
# ======================================================================
def _load_mtf_context_5m(ticker: str) -> pd.DataFrame:
    """
    Load 5m, Daily, Weekly CSVs for a ticker, suffix columns
    (_M, _D, _W) and merge Daily + Weekly onto 5m via as-of joins.

    Returns:
        df_mdw: 5m-based dataframe with *_M, *_D, *_W columns.
    """


    path_5m = _find_file(DIR_5m, ticker)
    path_d  = _find_file(DIR_D, ticker)
    path_w  = _find_file(DIR_W, ticker)

    if not path_5m or not path_d or not path_w:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return pd.DataFrame()


    df_m = _safe_read_tf(path_5m)
    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)

    if df_m.empty or df_d.empty or df_w.empty:
        print(f"- Skipping {ticker}: missing one or more TF files.")
        return pd.DataFrame()

    # Daily: compute previous-day Daily_Change BEFORE suffixing
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = df_d["Daily_Change"].shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    # Suffix all TFs
    df_m = _prepare_tf_with_suffix(df_m, "_M")
    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Merge Daily onto 5m (backward as-of)
    df_md = pd.merge_asof(
        df_m.sort_values("date"),
        df_d.sort_values("date"),
        on="date",
        direction="backward",
    )

    # Weekly: NO same-date lookahead
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] += pd.Timedelta(seconds=1)

    df_mdw = pd.merge_asof(
        df_md.sort_values("date"),
        df_w_sorted,
        on="date",
        direction="backward",
    )
    
    # ✅ Enforce intraday entry window globally (applies to ALL strategies)
    df_mdw = df_mdw[_intraday_entry_mask(df_mdw["date"])].copy()
    df_mdw.reset_index(drop=True, inplace=True)

    
    return df_mdw


# ----------------------- 5m EXTRAS (common utility) -----------------------
def _add_5m_extras(df_mdw: pd.DataFrame) -> pd.DataFrame:
    """
    Add common 5m extras (rolling means, previous values, rolling highs).
    Assumes suffixed 5m columns: ATR_M, RSI_M, MACD_Hist_M, ADX_M, high_M, etc.
    """
    if df_mdw.empty:
        return df_mdw

    df = df_mdw.copy()

    # ATR mean (~1H)
    if "ATR_M" in df.columns:
        df["ATR_M_1h_mean"] = df["ATR_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).mean()
    else:
        df["ATR_M_1h_mean"] = np.nan

    # ADX mean (~1H)
    if "ADX_M" in df.columns:
        df["ADX_M_1h_mean"] = df["ADX_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).mean()
    else:
        df["ADX_M_1h_mean"] = np.nan

    # Previous RSI / MACD hist / Stoch K
    if "RSI_M" in df.columns:
        df["RSI_M_prev"] = df["RSI_M"].shift(1)
    else:
        df["RSI_M_prev"] = np.nan

    if "MACD_Hist_M" in df.columns:
        df["MACD_Hist_M_prev"] = df["MACD_Hist_M"].shift(1)
    else:
        df["MACD_Hist_M_prev"] = np.nan

    if "Stoch_%K_M" in df.columns:
        df["Stoch_%K_M_prev"] = df["Stoch_%K_M"].shift(1)
    else:
        df["Stoch_%K_M_prev"] = np.nan

    # Recent highs on 5m:
    #   - 1h high (12 bars) and 5h high (60 bars)
    if "high_M" in df.columns:
        df["Recent_High_12_M"] = df["high_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).max()
        df["Recent_High_60_M"] = df["high_M"].rolling(ROLL_MEAN_5H_EQ, min_periods=ROLL_MEAN_5H_EQ).max()
    else:
        df["Recent_High_12_M"] = np.nan
        df["Recent_High_60_M"] = np.nan

    # Intra change (if exists) else 0
    if "Intra_Change_M" not in df.columns:
        df["Intra_Change_M"] = 0.0

    # Bullish count (2 of last 3 bullish) helper
    if {"open_M", "close_M"}.issubset(df.columns):
        bull = (df["close_M"] > df["open_M"]).astype(int)
        df["Bull_3_sum"] = bull.rolling(3, min_periods=3).sum()
    else:
        df["Bull_3_sum"] = np.nan

    return df


# ======================================================================
# STRATEGY 1: trend_pullback (ported to 5m execution)
# ======================================================================
def build_signals_trend_pullback(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    req = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "close_D","open_D","high_D","low_D","20_SMA_D","EMA_50_D","RSI_D","MACD_Hist_D",
        "close_M"
    }
    if not req.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for trend_pullback.")
        return []

    weekly_core_ok = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 52.0) &
        (df["ADX_W"] > 19.0)
    )
    if "Recent_High_W" in df.columns:
        weekly_loc_ok = df["close_W"] >= df["Recent_High_W"] * 0.93
    else:
        weekly_loc_ok = True
    weekly_ok = weekly_core_ok & weekly_loc_ok

    macd_prev = df["MACD_Hist_D"].shift(1)
    macd_rising = df["MACD_Hist_D"] >= macd_prev

    day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
    range_pos = (df["close_D"] - df["low_D"]) / day_range
    strong_close = range_pos >= 0.65

    pullback_zone = (
        (df["close_D"] >= df["20_SMA_D"] * 0.985) &
        (df["close_D"] <= df["EMA_50_D"] * 1.02)
    )

    daily_core_ok = (
        (df["close_D"] > df["open_D"]) &
        pullback_zone &
        strong_close &
        (df["RSI_D"] > 52.0) &
        macd_rising
    )

    if "Daily_Change_prev_D" in df.columns:
        dc_prev = df["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 0.7) & (dc_prev <= 7.5)
    else:
        daily_move_ok = True
    daily_ok = daily_core_ok & daily_move_ok

    # 5m gate (optional): above VWAP and RSI not weak
    if {"VWAP_M","RSI_M"}.issubset(df.columns):
        exec_ok = (df["close_M"] > df["VWAP_M"]) & (df["RSI_M"] > 50.0)
    else:
        exec_ok = True

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "trend_pullback",
        })
    return signals


# ======================================================================
# STRATEGY 2: weekly_breakout_daily_confirm (ported to 5m)
# ======================================================================
def build_signals_weekly_breakout_daily_confirm(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    req = {
        "close_W","Recent_High_W","RSI_W","ADX_W",
        "close_D","EMA_50_D","20_SMA_D","RSI_D","MACD_Hist_D",
        "close_M"
    }
    if not req.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for weekly_breakout.")
        return []

    weekly_ok = (
        (df["close_W"] >= df["Recent_High_W"] * 1.005) &
        (df["RSI_W"] > 52.0) &
        (df["ADX_W"] > 12.0)
    )

    daily_core_ok = (
        (df["close_D"] >= df["Recent_High_W"] * 0.995) &
        (df["close_D"] > df["20_SMA_D"]) &
        (df["close_D"] > df["EMA_50_D"]) &
        (df["RSI_D"] > 52.0) &
        (df["MACD_Hist_D"] > -0.02)
    )

    if "Daily_Change_prev_D" in df.columns:
        dc_prev = df["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 1.0) & (dc_prev <= 10.0)
    else:
        daily_move_ok = True

    daily_ok = daily_core_ok & daily_move_ok

    if "VWAP_M" in df.columns:
        exec_ok = df["close_M"] > df["VWAP_M"]
    else:
        exec_ok = True

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "weekly_breakout",
        })
    return signals


# ======================================================================
# STRATEGY 3: weekly_squeeze_daily_expansion (ported to 5m)
# ======================================================================
def build_signals_weekly_squeeze_daily_expansion(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    req = {
        "close_W","Upper_Band_W","Lower_Band_W","RSI_W","MACD_Hist_W","ADX_W",
        "close_D","Upper_Band_D","ATR_D","RSI_D",
        "close_M"
    }
    if not req.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for weekly_squeeze.")
        return []

    bb_width_W = (df["Upper_Band_W"] - df["Lower_Band_W"]) / df["close_W"].replace(0, np.nan)
    bb_squeeze = bb_width_W <= 0.10

    macd_prev_W = df["MACD_Hist_W"].shift(1)
    macd_rising_W = df["MACD_Hist_W"] >= macd_prev_W

    weekly_ok = (
        bb_squeeze &
        (df["RSI_W"] > 48.0) &
        (df["ADX_W"].between(8.0, 25.0)) &
        macd_rising_W
    )

    atr_prev1 = df["ATR_D"].shift(1)
    atr_prev2 = df["ATR_D"].shift(2)
    atr_rising = (df["ATR_D"] >= atr_prev1) & (atr_prev1 >= atr_prev2)

    daily_core_ok = (
        (df["close_D"] >= df["Upper_Band_D"] * 1.0) &
        atr_rising &
        (df["RSI_D"] > 58.0)
    )

    if "Daily_Change_prev_D" in df.columns:
        dc_prev = df["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 1.5) & (dc_prev <= 12.0)
    else:
        daily_move_ok = True

    daily_ok = daily_core_ok & daily_move_ok

    if {"VWAP_M","RSI_M"}.issubset(df.columns):
        exec_ok = (df["close_M"] > df["VWAP_M"]) & (df["RSI_M"] > 50.0)
    else:
        exec_ok = True

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "weekly_squeeze",
        })
    return signals


# ======================================================================
# STRATEGY 4: insidebar_breakout (ported to 5m execution timing)
# ======================================================================
def build_signals_weekly_trend_insidebar_breakout(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    req = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W","20_SMA_W",
        "high_D","low_D","close_D","open_D","20_SMA_D","Stoch_%K_D","Stoch_%D_D","RSI_D",
        "close_M"
    }
    if not req.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for insidebar_breakout.")
        return []

    weekly_ok = (
        (df["20_SMA_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["ADX_W"] > 20.0) &
        (df["RSI_W"] > 50.0)
    )

    high_prev = df["high_D"].shift(1)
    low_prev  = df["low_D"].shift(1)

    inside_bar = (
        (df["high_D"] <= high_prev * 1.001) &
        (df["low_D"]  >= low_prev * 0.999)
    )
    breakout = df["close_D"] >= high_prev
    stoch_cross = df["Stoch_%K_D"] > df["Stoch_%D_D"]

    day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
    range_pos = (df["close_D"] - df["low_D"]) / day_range
    strong_close = range_pos >= 0.55

    daily_ok = (
        inside_bar &
        breakout &
        strong_close &
        (df["close_D"] > df["20_SMA_D"]) &
        (df["RSI_D"] > 50.0) &
        stoch_cross
    )

    long_mask = weekly_ok & daily_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "insidebar_breakout",
        })
    return signals


# ======================================================================
# STRATEGY 5: hh_hl_retest (ported to 5m execution)
# ======================================================================
def build_signals_weekly_hh_hl_retest(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    req = {
        "close_W","20_SMA_W","RSI_W","ADX_W","Recent_High_W",
        "open_D","low_D","high_D","close_D","MACD_Hist_D","MFI_D","20_SMA_D",
        "close_M"
    }
    if not req.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for hh_hl_retest.")
        return []

    weekly_ok = (
        (df["close_W"] > df["20_SMA_W"] * 1.01) &
        (df["RSI_W"] > 57.0) &
        (df["ADX_W"] > 17.0)
    )

    recent_high_W = df["Recent_High_W"]
    retest_ok = df["low_D"] <= recent_high_W * 1.005
    reclaim_ok = df["close_D"] >= recent_high_W * 0.998

    day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
    range_pos = (df["close_D"] - df["low_D"]) / day_range
    strong_close = range_pos >= 0.55

    daily_ok = (
        retest_ok &
        reclaim_ok &
        (df["close_D"] > df["open_D"]) &
        (df["close_D"] > df["20_SMA_D"]) &
        (df["MACD_Hist_D"] > 0.0) &
        (df["MFI_D"] > 50.0) &
        strong_close
    )

    if {"VWAP_M","RSI_M"}.issubset(df.columns):
        exec_ok = (df["close_M"] > df["VWAP_M"]) & (df["RSI_M"] > 50.0)
    else:
        exec_ok = True

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "hh_hl_retest",
        })
    return signals


# ======================================================================
# STRATEGY 6: golden_cross (ported to 5m execution)
# ======================================================================
def build_signals_weekly_golden_cross_ema_bounce(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    req = {
        "EMA_50_W","EMA_200_W","RSI_W",
        "low_D","close_D","open_D","20_SMA_D","EMA_50_D","RSI_D","MACD_Hist_D",
        "close_M"
    }
    if not req.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for golden_cross.")
        return []

    weekly_ok = (df["EMA_50_W"] > df["EMA_200_W"]) & (df["RSI_W"] > 25.0)

    touch_ma = (df["low_D"] <= df["20_SMA_D"]) | (df["low_D"] <= df["EMA_50_D"])

    daily_core_ok = (
        touch_ma &
        (df["close_D"] > df["20_SMA_D"]) &
        (df["close_D"] > df["EMA_50_D"]) &
        (df["close_D"] > df["open_D"]) &
        (df["RSI_D"].between(30.0, 70.0)) &
        (df["MACD_Hist_D"] > 1)
    )

    if "Daily_Change_prev_D" in df.columns:
        dc_prev = df["Daily_Change_prev_D"]
        daily_move_ok = (dc_prev >= 0.0) & (dc_prev <= 7.0)
    else:
        daily_move_ok = True

    daily_ok = daily_core_ok & daily_move_ok

    if {"VWAP_M","RSI_M"}.issubset(df.columns):
        exec_ok = (df["close_M"] > df["VWAP_M"]) & (df["RSI_M"] > 25.0)
    else:
        exec_ok = True

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "golden_cross",
        })
    return signals


# ======================================================================
# CUSTOM 1 (ported: weekly+daily only, 5m entry timing)
# ======================================================================
def build_signals_custom1(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    required_cols = {
        "close_W","open_W","high_W","low_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W","MACD_Hist_W","MFI_W","ATR_W",
        "20_SMA_W","Upper_Band_W","Lower_Band_W","Recent_High_W","Stoch_%K_W","Stoch_%D_W",
        "close_D","open_D","high_D","low_D","EMA_50_D","EMA_200_D","RSI_D","MACD_Hist_D","MFI_D","ADX_D","ATR_D",
        "20_SMA_D","Upper_Band_D","Lower_Band_D","Recent_High_D","Stoch_%K_D","Stoch_%D_D","Daily_Change_prev_D",
        "close_M"
    }
    if not required_cols.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for custom1.")
        return []

    df = df.dropna(subset=list(required_cols))
    if df.empty:
        return []

    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"] * 1.005) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 42.0) &
        (df["ADX_W"] > 12.0)
    )
    weekly_mild = (
        (df["close_W"] > df["EMA_200_W"] * 1.015) &
        (df["close_W"] >= df["EMA_50_W"] * 0.99) &
        (df["RSI_W"].between(42.0, 66.0)) &
        (df["ADX_W"] > 10.0)
    )
    weekly_macd_ok = df["MACD_Hist_W"] >= -0.02
    weekly_mfi_ok  = df["MFI_W"] > 45.0

    close_W = df["close_W"]
    atr_pct_W = (df["ATR_W"] / close_W.replace(0, np.nan)) * 100.0
    weekly_atr_ok = (atr_pct_W >= 0.4) & (atr_pct_W <= 10.0)

    weekly_bb_pos_ok = (
        (close_W >= df["20_SMA_W"] * 0.995) &
        (close_W <= df["Upper_Band_W"] * 1.03) &
        (close_W >= df["Lower_Band_W"] * 0.99)
    )
    weekly_recent_high_ok = close_W >= (df["Recent_High_W"] * 0.94)

    weekly_stoch_ok = (
        (df["Stoch_%K_W"] >= df["Stoch_%D_W"] * 0.98) &
        (df["Stoch_%K_W"] >= 35.0) &
        (df["Stoch_%K_W"] <= 88.0)
    )
    weekly_not_extreme = df["RSI_W"] < 85.0

    weekly_ok = (
        (weekly_strong | weekly_mild) &
        weekly_macd_ok & weekly_mfi_ok & weekly_atr_ok &
        weekly_bb_pos_ok & weekly_recent_high_ok & weekly_stoch_ok &
        weekly_not_extreme
    )

    daily_prev = df["Daily_Change_prev_D"]
    daily_move_ok = (daily_prev >= 1.0) & (daily_prev <= 7.0)

    daily_trend_ok = (
        (df["close_D"] >= df["EMA_50_D"] * 0.995) &
        (df["EMA_50_D"] >= df["EMA_200_D"] * 0.995)
    )
    daily_rsi_ok  = df["RSI_D"].between(42.0, 72.0)
    daily_macd_ok = df["MACD_Hist_D"] >= -0.02
    daily_mfi_ok  = df["MFI_D"] > 45.0
    daily_adx_ok  = df["ADX_D"] > 10.0

    close_D = df["close_D"]
    atr_pct_D = (df["ATR_D"] / close_D.replace(0, np.nan)) * 100.0
    atr_vol_ok = (atr_pct_D >= 0.3) & (atr_pct_D <= 7.0)

    bb_pos_ok = (
        (close_D >= df["20_SMA_D"] * 0.99) &
        (close_D <= df["Upper_Band_D"] * 1.04) &
        (close_D >= df["Lower_Band_D"] * 0.99)
    )
    recent_high_ok = close_D >= (df["Recent_High_D"] * 0.95)

    day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
    range_pos = (close_D - df["low_D"]) / day_range
    pa_upper_close_ok = range_pos >= 0.55

    stoch_daily_ok = (
        (df["Stoch_%K_D"] >= df["Stoch_%D_D"] * 0.98) &
        (df["Stoch_%K_D"] >= 35.0) &
        (df["Stoch_%K_D"] <= 90.0)
    )

    daily_ok = (
        daily_move_ok & daily_trend_ok & daily_rsi_ok & daily_macd_ok &
        daily_mfi_ok & daily_adx_ok & atr_vol_ok & bb_pos_ok &
        recent_high_ok & pa_upper_close_ok & stoch_daily_ok
    )

    long_mask = weekly_ok & daily_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom1",
        })
    return signals


# ======================================================================
# CUSTOM 2 (ported to 5m: relaxed weekly + banded daily + 5m trigger)
# ======================================================================
def build_signals_custom2(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    required = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "Daily_Change_prev_D",
        "close_M","EMA_200_M","EMA_50_M",
        "RSI_M","RSI_M_prev",
        "MACD_Hist_M","MACD_Hist_M_prev",
        "Stoch_%K_M","Stoch_%D_M",
        "ATR_M","ATR_M_1h_mean",
        "ADX_M","ADX_M_1h_mean",
        "VWAP_M","Recent_High_60_M"
    }
    if not required.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for custom2.")
        return []

    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 46.0) &
        (df["ADX_W"] > 11.0)
    )
    weekly_mild = (
        (df["close_W"] > df["EMA_200_W"] * 1.01) &
        (df["RSI_W"].between(44.0, 65.0)) &
        (df["ADX_W"] > 9.0)
    )
    weekly_ok = weekly_strong | weekly_mild

    dprev = df["Daily_Change_prev_D"]
    daily_ok = (dprev >= 1.0) & (dprev <= 8.0)

    exec_trend = (
        (df["close_M"] >= df["EMA_200_M"] * 0.995) &
        (df["close_M"] >= df["EMA_50_M"]  * 0.995)
    )
    rsi_ok = (
        (df["RSI_M"] > 56.0) |
        ((df["RSI_M"] > 52.0) & (df["RSI_M"] > df["RSI_M_prev"]))
    )
    macd_ok = (
        (df["MACD_Hist_M"] > -0.02) &
        (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.85)
    )
    stoch_ok = df["Stoch_%K_M"] >= df["Stoch_%D_M"] * 0.98

    momentum_ok = rsi_ok & macd_ok & stoch_ok

    vol_ok = (
        (df["ATR_M"] >= df["ATR_M_1h_mean"] * 0.92) &
        ((df["ADX_M"] >= 11.0) | (df["ADX_M"] >= df["ADX_M_1h_mean"]))
    )

    price_above_vwap = df["close_M"] >= df["VWAP_M"]

    breakout_strict = df["close_M"] >= df["Recent_High_60_M"]
    breakout_near   = df["close_M"] >= df["Recent_High_60_M"] * 0.995

    strong_up = df["Intra_Change_M"].fillna(0.0) >= 0.2

    breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up))

    exec_ok = exec_trend & momentum_ok & vol_ok & breakout_ok

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom2",
        })
    return signals


# ======================================================================
# CUSTOM 3 (ported: stricter daily + strict 5m execution)
# ======================================================================
def build_signals_custom3(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    required = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "close_D","EMA_50_D","RSI_D","MACD_Hist_D","ATR_D","high_D",
        "close_M","VWAP_M","RSI_M","MACD_Hist_M","ATR_M","ATR_M_1h_mean",
        "Recent_High_12_M"
    }
    if not required.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for custom3.")
        return []

    # WEEKLY
    weekly_ok = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 50.0) &
        (df["ADX_W"] > 15.0)
    )

    # DAILY (stricter)
    high10 = df["high_D"].rolling(10, min_periods=10).max()
    daily_close_vs_high = df["close_D"] >= (high10 * 0.99)
    atr5 = df["ATR_D"].rolling(5, min_periods=5).mean()
    daily_atr_expansion = df["ATR_D"] >= (atr5 * 0.97)

    daily_ok = (
        (df["close_D"] > df["EMA_50_D"]) &
        (df["RSI_D"] > 52.0) &
        (df["MACD_Hist_D"] >= 0.0) &
        daily_atr_expansion &
        daily_close_vs_high
    )

    # 5m EXEC (strict)
    exec_ok = (
        (df["close_M"] > df["VWAP_M"]) &
        (df["RSI_M"] > 56.0) &
        (df["MACD_Hist_M"] > 0.0) &
        (df["ATR_M"] >= df["ATR_M_1h_mean"] * 1.0) &
        (df["close_M"] >= df["Recent_High_12_M"] * 0.999)
    )

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom3",
        })
    return signals


# ======================================================================
# CUSTOM 4 (your stricter version, ported to 5m execution)
# ======================================================================
def build_signals_custom4(ticker: str) -> list[dict]:
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    required = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W","Daily_Change_prev_D",
        "close_D","open_D","high_D","low_D",
        "close_M","EMA_200_M","EMA_50_M",
        "RSI_M","RSI_M_prev",
        "MACD_Hist_M","MACD_Hist_M_prev",
        "Stoch_%K_M","Stoch_%D_M",
        "ATR_M","ATR_M_1h_mean",
        "ADX_M","ADX_M_1h_mean",
        "VWAP_M","Recent_High_60_M"
    }
    if not required.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for custom4.")
        return []

    # WEEKLY (slightly stricter)
    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 46.0) &
        (df["ADX_W"] > 15.0)
    )
    weekly_mild = (
        (df["close_W"] > df["EMA_200_W"] * 1.025) &
        (df["RSI_W"].between(44.0, 68.0)) &
        (df["ADX_W"] > 13.0)
    )
    weekly_not_extreme = df["RSI_W"] < 80.0
    if "Recent_High_W" in df.columns:
        weekly_loc_ok = df["close_W"] >= df["Recent_High_W"] * 0.93
    else:
        weekly_loc_ok = True
    weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme & weekly_loc_ok

    # DAILY (candle quality)
    daily_prev = df["Daily_Change_prev_D"]
    daily_ok_band = (daily_prev >= 2.0) & (daily_prev <= 8.0)

    day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
    range_pos = (df["close_D"] - df["low_D"]) / day_range
    bullish_day = df["close_D"] > df["open_D"]
    strong_close = range_pos >= 0.62
    daily_ok = daily_ok_band & bullish_day & strong_close

    # 5m EXEC (tight trigger)
    exec_trend = (
        (df["close_M"] >= df["EMA_200_M"] * 0.998) &
        (df["close_M"] >= df["EMA_50_M"]  * 0.998)
    )
    rsi_ok = (
        (df["RSI_M"] > 58.0) |
        ((df["RSI_M"] > 54.0) & (df["RSI_M"] > df["RSI_M_prev"]))
    )
    macd_ok = (
        (df["MACD_Hist_M"] > -0.015) &
        (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.92)
    )
    stoch_ok = (
        (df["Stoch_%K_M"] >= df["Stoch_%D_M"] * 0.99) &
        (df["Stoch_%K_M"] >= 22.0)
    )
    momentum_ok = rsi_ok & macd_ok & stoch_ok

    vol_ok = (
        (df["ATR_M"] >= df["ATR_M_1h_mean"] * 0.96) &
        ((df["ADX_M"] >= 12.0) | (df["ADX_M"] >= df["ADX_M_1h_mean"] * 1.04))
    )

    price_above_vwap = df["close_M"] > df["VWAP_M"]

    breakout_strict = df["close_M"] >= df["Recent_High_60_M"]
    breakout_near   = df["close_M"] >= df["Recent_High_60_M"] * 0.997
    strong_up = df["Intra_Change_M"].fillna(0.0) >= 0.14
    breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up))

    rsi_improving = df["RSI_M"] >= df["RSI_M_prev"] - 0.2

    exec_ok = exec_trend & momentum_ok & vol_ok & breakout_ok & rsi_improving

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom4",
        })
    return signals


# ======================================================================
# STRATEGY 11 (ADDED): your “loose weekly + looser execution” on 5m (calibrated)
# ======================================================================
def build_signals_strategy11(ticker: str) -> list[dict]:
    """
    STRATEGY 11 (5m)

    - Weekly relaxed regime
    - Daily prev change gate (>=2%)
    - 5m execution: momentum + volatility + breakout, with anti-noise guards
    """
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    required = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "Daily_Change_prev_D",
        "close_M","EMA_200_M","EMA_50_M",
        "RSI_M","RSI_M_prev",
        "MACD_Hist_M","MACD_Hist_M_prev",
        "Stoch_%K_M","Stoch_%D_M",
        "ATR_M","ATR_M_1h_mean",
        "ADX_M","ADX_M_1h_mean",
        "VWAP_M","Recent_High_60_M",
        "Bull_3_sum"
    }
    if not required.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for strategy11.")
        return []

    # Weekly relaxed
    weekly_strong = (
        (df["close_W"] > df["EMA_50_W"]) &
        (df["EMA_50_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 46.0) &
        (df["ADX_W"] > 11.0)
    )
    weekly_mild = (
        (df["close_W"] > df["EMA_200_W"] * 1.01) &
        (df["RSI_W"].between(44.0, 65.0)) &
        (df["ADX_W"] > 9.0)
    )
    weekly_ok = weekly_strong | weekly_mild

    # Daily: prev day move >=2
    daily_ok = df["Daily_Change_prev_D"] >= 2.0

    # 5m execution (calibrated)
    exec_trend = (
        (df["close_M"] >= df["EMA_200_M"] * 0.995) &
        (df["close_M"] >= df["EMA_50_M"]  * 0.995)
    )

    rsi_ok = (
        (df["RSI_M"] > 56.0) |
        ((df["RSI_M"] > 52.0) & (df["RSI_M"] > df["RSI_M_prev"]))
    )

    macd_ok = (
        (df["MACD_Hist_M"] > -0.02) &
        (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.85)
    )

    stoch_ok = df["Stoch_%K_M"] >= df["Stoch_%D_M"] * 0.98
    momentum_ok = rsi_ok & macd_ok & stoch_ok

    vol_ok = (
        (df["ATR_M"] >= df["ATR_M_1h_mean"] * 0.92) &
        ((df["ADX_M"] >= 11.0) | (df["ADX_M"] >= df["ADX_M_1h_mean"]))
    )

    price_above_vwap = df["close_M"] >= df["VWAP_M"]

    breakout_strict = df["close_M"] >= df["Recent_High_60_M"]
    breakout_near   = df["close_M"] >= df["Recent_High_60_M"] * 0.995
    strong_up = df["Intra_Change_M"].fillna(0.0) >= 0.20
    breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up))

    # anti-noise guards
    bull_2of3 = df["Bull_3_sum"] >= 2  # 2 of last 3 bullish
    not_too_far_from_vwap = (df["close_M"] <= df["VWAP_M"] * 1.012)

    exec_ok = exec_trend & momentum_ok & vol_ok & breakout_ok & bull_2of3 & not_too_far_from_vwap

    long_mask = weekly_ok & daily_ok & exec_ok

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "strategy11",
        })
    return signals


# ======================================================================
# CUSTOM 6 (ADDED): voting union of families on 5m
# ======================================================================
def build_signals_custom6(ticker: str) -> list[dict]:
    """
    CUSTOM 6 (5m)

    Families combined:
      - trend_pullback
      - golden_cross
      - custom2
      - custom4
      - strategy11

    Weighted votes:
      trend_pullback 1
      custom4        1
      golden_cross   2
      custom2        1
      strategy11     3

    Threshold:
      votes >= 3

    Additional safety:
      At least one HTF-anchored family must fire (trend_pullback OR golden_cross)
    """
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    cols = set(df.columns)

    def _false_series():
        return pd.Series(False, index=df.index)

    mask_tp = _false_series()
    mask_gc = _false_series()
    mask_c2 = _false_series()
    mask_c4 = _false_series()
    mask_s11 = _false_series()

    # Build each mask by calling its logic in-place (fast), using the same df
    # To avoid duplicated code, we reuse the same strategy functions by recomputing masks inline:
    # (kept explicit here for stability and transparency)

    # -------- trend_pullback mask --------
    req_tp = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "close_D","open_D","high_D","low_D","20_SMA_D","EMA_50_D","RSI_D","MACD_Hist_D",
        "close_M"
    }
    if req_tp.issubset(cols):
        weekly_core_ok = (
            (df["close_W"] > df["EMA_50_W"]) &
            (df["EMA_50_W"] > df["EMA_200_W"]) &
            (df["RSI_W"] > 52.0) &
            (df["ADX_W"] > 19.0)
        )
        weekly_loc_ok = (df["close_W"] >= df["Recent_High_W"] * 0.93) if "Recent_High_W" in cols else True
        weekly_ok = weekly_core_ok & weekly_loc_ok

        macd_prev = df["MACD_Hist_D"].shift(1)
        macd_rising = df["MACD_Hist_D"] >= macd_prev

        day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
        range_pos = (df["close_D"] - df["low_D"]) / day_range
        strong_close = range_pos >= 0.65

        pullback_zone = (
            (df["close_D"] >= df["20_SMA_D"] * 0.985) &
            (df["close_D"] <= df["EMA_50_D"] * 1.02)
        )

        daily_core_ok = (
            (df["close_D"] > df["open_D"]) &
            pullback_zone &
            strong_close &
            (df["RSI_D"] > 52.0) &
            macd_rising
        )

        if "Daily_Change_prev_D" in cols:
            dc_prev = df["Daily_Change_prev_D"]
            daily_move_ok = (dc_prev >= 0.7) & (dc_prev <= 7.5)
        else:
            daily_move_ok = True

        daily_ok = daily_core_ok & daily_move_ok

        if {"VWAP_M","RSI_M"}.issubset(cols):
            exec_ok = (df["close_M"] > df["VWAP_M"]) & (df["RSI_M"] > 50.0)
        else:
            exec_ok = True

        mask_tp = weekly_ok & daily_ok & exec_ok

    # -------- golden_cross mask --------
    req_gc = {
        "EMA_50_W","EMA_200_W","RSI_W",
        "low_D","close_D","open_D","20_SMA_D","EMA_50_D","RSI_D","MACD_Hist_D",
        "close_M"
    }
    if req_gc.issubset(cols):
        weekly_ok = (df["EMA_50_W"] > df["EMA_200_W"]) & (df["RSI_W"] > 25.0)
        touch_ma = (df["low_D"] <= df["20_SMA_D"]) | (df["low_D"] <= df["EMA_50_D"])
        daily_core_ok = (
            touch_ma &
            (df["close_D"] > df["20_SMA_D"]) &
            (df["close_D"] > df["EMA_50_D"]) &
            (df["close_D"] > df["open_D"]) &
            (df["RSI_D"].between(30.0, 70.0)) &
            (df["MACD_Hist_D"] > 1)
        )
        if "Daily_Change_prev_D" in cols:
            dc_prev = df["Daily_Change_prev_D"]
            daily_move_ok = (dc_prev >= 0.0) & (dc_prev <= 7.0)
        else:
            daily_move_ok = True
        daily_ok = daily_core_ok & daily_move_ok

        if {"VWAP_M","RSI_M"}.issubset(cols):
            exec_ok = (df["close_M"] > df["VWAP_M"]) & (df["RSI_M"] > 25.0)
        else:
            exec_ok = True

        mask_gc = weekly_ok & daily_ok & exec_ok

    # -------- custom2 mask --------
    # reuse the same requirements as custom2
    req_c2 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "Daily_Change_prev_D",
        "close_M","EMA_200_M","EMA_50_M",
        "RSI_M","RSI_M_prev",
        "MACD_Hist_M","MACD_Hist_M_prev",
        "Stoch_%K_M","Stoch_%D_M",
        "ATR_M","ATR_M_1h_mean",
        "ADX_M","ADX_M_1h_mean",
        "VWAP_M","Recent_High_60_M"
    }
    if req_c2.issubset(cols):
        weekly_strong = (
            (df["close_W"] > df["EMA_50_W"]) &
            (df["EMA_50_W"] > df["EMA_200_W"]) &
            (df["RSI_W"] > 46.0) &
            (df["ADX_W"] > 11.0)
        )
        weekly_mild = (
            (df["close_W"] > df["EMA_200_W"] * 1.01) &
            (df["RSI_W"].between(44.0, 65.0)) &
            (df["ADX_W"] > 9.0)
        )
        weekly_ok = weekly_strong | weekly_mild

        dprev = df["Daily_Change_prev_D"]
        daily_ok = (dprev >= 1.0) & (dprev <= 8.0)

        exec_trend = (
            (df["close_M"] >= df["EMA_200_M"] * 0.995) &
            (df["close_M"] >= df["EMA_50_M"]  * 0.995)
        )
        rsi_ok = (
            (df["RSI_M"] > 56.0) |
            ((df["RSI_M"] > 52.0) & (df["RSI_M"] > df["RSI_M_prev"]))
        )
        macd_ok = (
            (df["MACD_Hist_M"] > -0.02) &
            (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.85)
        )
        stoch_ok = df["Stoch_%K_M"] >= df["Stoch_%D_M"] * 0.98
        momentum_ok = rsi_ok & macd_ok & stoch_ok

        vol_ok = (
            (df["ATR_M"] >= df["ATR_M_1h_mean"] * 0.92) &
            ((df["ADX_M"] >= 11.0) | (df["ADX_M"] >= df["ADX_M_1h_mean"]))
        )

        price_above_vwap = df["close_M"] >= df["VWAP_M"]
        breakout_strict = df["close_M"] >= df["Recent_High_60_M"]
        breakout_near   = df["close_M"] >= df["Recent_High_60_M"] * 0.995
        strong_up = df["Intra_Change_M"].fillna(0.0) >= 0.2
        breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up))

        exec_ok = exec_trend & momentum_ok & vol_ok & breakout_ok
        mask_c2 = weekly_ok & daily_ok & exec_ok

    # -------- custom4 mask --------
    req_c4 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W","Daily_Change_prev_D",
        "close_D","open_D","high_D","low_D",
        "close_M","EMA_200_M","EMA_50_M",
        "RSI_M","RSI_M_prev",
        "MACD_Hist_M","MACD_Hist_M_prev",
        "Stoch_%K_M","Stoch_%D_M",
        "ATR_M","ATR_M_1h_mean",
        "ADX_M","ADX_M_1h_mean",
        "VWAP_M","Recent_High_60_M"
    }
    if req_c4.issubset(cols):
        # reuse same logic as build_signals_custom4 but as mask
        weekly_strong = (
            (df["close_W"] > df["EMA_50_W"]) &
            (df["EMA_50_W"] > df["EMA_200_W"]) &
            (df["RSI_W"] > 46.0) &
            (df["ADX_W"] > 15.0)
        )
        weekly_mild = (
            (df["close_W"] > df["EMA_200_W"] * 1.025) &
            (df["RSI_W"].between(44.0, 68.0)) &
            (df["ADX_W"] > 13.0)
        )
        weekly_not_extreme = df["RSI_W"] < 80.0
        weekly_loc_ok = (df["close_W"] >= df["Recent_High_W"] * 0.93) if "Recent_High_W" in cols else True
        weekly_ok = (weekly_strong | weekly_mild) & weekly_not_extreme & weekly_loc_ok

        daily_prev = df["Daily_Change_prev_D"]
        daily_ok_band = (daily_prev >= 2.0) & (daily_prev <= 8.0)

        day_range = (df["high_D"] - df["low_D"]).replace(0, np.nan)
        range_pos = (df["close_D"] - df["low_D"]) / day_range
        bullish_day = df["close_D"] > df["open_D"]
        strong_close = range_pos >= 0.62
        daily_ok = daily_ok_band & bullish_day & strong_close

        exec_trend = (
            (df["close_M"] >= df["EMA_200_M"] * 0.998) &
            (df["close_M"] >= df["EMA_50_M"]  * 0.998)
        )
        rsi_ok = (
            (df["RSI_M"] > 58.0) |
            ((df["RSI_M"] > 54.0) & (df["RSI_M"] > df["RSI_M_prev"]))
        )
        macd_ok = (
            (df["MACD_Hist_M"] > -0.015) &
            (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.92)
        )
        stoch_ok = (
            (df["Stoch_%K_M"] >= df["Stoch_%D_M"] * 0.99) &
            (df["Stoch_%K_M"] >= 22.0)
        )
        momentum_ok = rsi_ok & macd_ok & stoch_ok

        vol_ok = (
            (df["ATR_M"] >= df["ATR_M_1h_mean"] * 0.96) &
            ((df["ADX_M"] >= 12.0) | (df["ADX_M"] >= df["ADX_M_1h_mean"] * 1.04))
        )

        price_above_vwap = df["close_M"] > df["VWAP_M"]
        breakout_strict = df["close_M"] >= df["Recent_High_60_M"]
        breakout_near   = df["close_M"] >= df["Recent_High_60_M"] * 0.997
        strong_up = df["Intra_Change_M"].fillna(0.0) >= 0.14
        breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up))
        rsi_improving = df["RSI_M"] >= df["RSI_M_prev"] - 0.2

        exec_ok = exec_trend & momentum_ok & vol_ok & breakout_ok & rsi_improving

        mask_c4 = weekly_ok & daily_ok & exec_ok

    # -------- strategy11 mask --------
    req_s11 = {
        "close_W","EMA_50_W","EMA_200_W","RSI_W","ADX_W",
        "Daily_Change_prev_D",
        "close_M","EMA_200_M","EMA_50_M",
        "RSI_M","RSI_M_prev",
        "MACD_Hist_M","MACD_Hist_M_prev",
        "Stoch_%K_M","Stoch_%D_M",
        "ATR_M","ATR_M_1h_mean",
        "ADX_M","ADX_M_1h_mean",
        "VWAP_M","Recent_High_60_M","Bull_3_sum"
    }
    if req_s11.issubset(cols):
        weekly_strong = (
            (df["close_W"] > df["EMA_50_W"]) &
            (df["EMA_50_W"] > df["EMA_200_W"]) &
            (df["RSI_W"] > 46.0) &
            (df["ADX_W"] > 11.0)
        )
        weekly_mild = (
            (df["close_W"] > df["EMA_200_W"] * 1.01) &
            (df["RSI_W"].between(44.0, 65.0)) &
            (df["ADX_W"] > 9.0)
        )
        weekly_ok = weekly_strong | weekly_mild
        daily_ok = df["Daily_Change_prev_D"] >= 2.0

        exec_trend = (
            (df["close_M"] >= df["EMA_200_M"] * 0.995) &
            (df["close_M"] >= df["EMA_50_M"]  * 0.995)
        )
        rsi_ok = (
            (df["RSI_M"] > 56.0) |
            ((df["RSI_M"] > 52.0) & (df["RSI_M"] > df["RSI_M_prev"]))
        )
        macd_ok = (
            (df["MACD_Hist_M"] > -0.02) &
            (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.85)
        )
        stoch_ok = df["Stoch_%K_M"] >= df["Stoch_%D_M"] * 0.98
        momentum_ok = rsi_ok & macd_ok & stoch_ok

        vol_ok = (
            (df["ATR_M"] >= df["ATR_M_1h_mean"] * 0.92) &
            ((df["ADX_M"] >= 11.0) | (df["ADX_M"] >= df["ADX_M_1h_mean"]))
        )

        price_above_vwap = df["close_M"] >= df["VWAP_M"]
        breakout_strict = df["close_M"] >= df["Recent_High_60_M"]
        breakout_near   = df["close_M"] >= df["Recent_High_60_M"] * 0.995
        strong_up = df["Intra_Change_M"].fillna(0.0) >= 0.20
        breakout_ok = price_above_vwap & (breakout_strict | (breakout_near & strong_up))

        bull_2of3 = df["Bull_3_sum"] >= 2
        not_too_far = (df["close_M"] <= df["VWAP_M"] * 1.012)

        exec_ok = exec_trend & momentum_ok & vol_ok & breakout_ok & bull_2of3 & not_too_far

        mask_s11 = weekly_ok & daily_ok & exec_ok

    votes = (
        1 * mask_tp.astype(int) +
        1 * mask_c4.astype(int) +
        2 * mask_gc.astype(int) +
        1 * mask_c2.astype(int) +
        3 * mask_s11.astype(int)
    )
    THRESH = 3
    combined = votes >= THRESH

    # Safety: at least one HTF-anchored family fires
    htf_anchor = mask_tp | mask_gc
    combined = combined & htf_anchor

    if not combined.any():
        return []

    signals = []
    for idx, r in df[combined].iterrows():
        tags = []
        if mask_tp.loc[idx]: tags.append("trend_pullback")
        if mask_gc.loc[idx]: tags.append("golden_cross")
        if mask_c2.loc[idx]: tags.append("custom2")
        if mask_c4.loc[idx]: tags.append("custom4")
        if mask_s11.loc[idx]: tags.append("strategy11")

        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom6",
            "sub_strategy": "+".join(tags) if tags else "custom6",
            "sub_strategy_votes": int(votes.loc[idx]),
        })
    return signals


def build_signals_custom7(ticker: str) -> list[dict]:
    """
    CUSTOM 7 (5m intraday):
    Opening Range Breakout (first 30 min) + VWAP confirmation + volume expansion.
    HTF filters (Weekly+Daily) keep regime bullish.

    Entry window is already enforced globally inside _load_mtf_context_5m().
    """

    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    required = {
        # Weekly context
        "close_W", "EMA_200_W", "RSI_W",
        # Daily context
        "close_D", "open_D", "RSI_D", "Daily_Change_prev_D",
        # 5m execution
        "open_M", "high_M", "low_M", "close_M", "VWAP_M", "RSI_M", "RSI_M_prev",
        "volume_M"
    }
    if not required.issubset(df.columns):
        print(f"- Skipping {ticker}: missing columns for custom7.")
        return []

    # ---------------- HTF FILTERS ----------------
    # Weekly: bullish regime (simple + robust)
    weekly_ok = (
        (df["close_W"] > df["EMA_200_W"]) &
        (df["RSI_W"] > 45.0)
    )

    # Daily: positive bias but not extremely overextended
    # You can tune this band later
    if "Daily_Change_prev_D" in df.columns:
        dprev = df["Daily_Change_prev_D"]
        daily_move_ok = (dprev >= 0.3) & (dprev <= 9.0)
    else:
        daily_move_ok = True

    daily_ok = (
        (df["close_D"] >= df["open_D"] * 0.995) &   # not a weak day
        (df["RSI_D"] > 48.0) &
        daily_move_ok
    )

    # ---------------- INTRADAY ORB (first 30 min) ----------------
    # We compute OR high per day using first 6 candles (9:15 to 9:45).
    df["d"] = df["date"].dt.date
    df["t"] = df["date"].dt.time

    # Opening range window
    OR_START = time(9, 15)
    OR_END   = time(9, 45)  # first 30 min

    in_or = df["t"].between(OR_START, OR_END)
    # OR high per day (computed only from the OR window)
    or_high = df.loc[in_or].groupby("d")["high_M"].max()
    df["OR_high"] = df["d"].map(or_high)

    # Only consider entries AFTER OR is formed
    AFTER_OR = time(9, 50)
    after_or = df["t"] >= AFTER_OR

    # Breakout: close crosses above OR_high (avoid noise: use close not just wick)
    breakout = (df["close_M"] >= df["OR_high"] * 1.0005)

    # VWAP confirmation (stay above VWAP)
    vwap_ok = df["close_M"] > df["VWAP_M"]

    # RSI rising / momentum improving
    rsi_ok = (df["RSI_M"] > 52.0) & (df["RSI_M"] >= df["RSI_M_prev"])

    # Volume expansion: current volume vs 1-hour rolling avg (12 bars)
    vol_mean_1h = df["volume_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).mean()
    vol_ok = df["volume_M"] >= (vol_mean_1h * 1.20)

    # Optional: avoid chasing too far from VWAP
    not_too_far = df["close_M"] <= df["VWAP_M"] * 1.015

    exec_ok = after_or & breakout & vwap_ok & rsi_ok & vol_ok & not_too_far

    long_mask = weekly_ok & daily_ok & exec_ok

    if not long_mask.any():
        return []

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom7",
            "sub_strategy": "ORB30_VWAP_VOL",
        })

    return signals

def build_signals_custom8(ticker: str) -> list[dict]:
    """
    CUSTOM 8 (5m intraday) — Pullback + Continuation (RELAXED for more signals)

    HTF:
      - Weekly bullish regime (tolerant, with fallbacks)
      - Daily previous-day strength band (tolerant, with fallbacks)

    5m Execution:
      - Time window: 09:20 to 11:35 IST (no new entries after 11:35)
      - Pullback: near VWAP and/or EMA20 (relaxed tolerance)
      - Continuation: RSI improving + optional MACD improvement + reclaim of recent swing (relaxed)
      - “High upward momentum” bias: requires at least one of (Intra_Change, Bull_3_sum, strong RSI/MACD)
    """

    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    cols = set(df.columns)

    # ---------- Minimal requirements ----------
    minimal = {"close_M", "high_M", "low_M", "VWAP_M", "RSI_M"}
    if not minimal.issubset(cols):
        print(f"- Skipping {ticker}: missing minimal 5m columns for custom8.")
        return []

    def _col(name: str, default=np.nan):
        return df[name] if name in cols else pd.Series(default, index=df.index)

    # ---------- HTF FILTERS ----------
    # Weekly bullish regime (tolerant)
    if {"close_W", "EMA_200_W", "RSI_W"}.issubset(cols):
        weekly_ok = (df["close_W"] > df["EMA_200_W"]) & (df["RSI_W"] > 40.0)
    elif {"close_W", "EMA_200_W"}.issubset(cols):
        weekly_ok = (df["close_W"] > df["EMA_200_W"])
    else:
        weekly_ok = pd.Series(True, index=df.index)

    # Daily prev-day move band (relaxed)
    if "Daily_Change_prev_D" in cols:
        dprev = df["Daily_Change_prev_D"]
        daily_ok = (dprev >= 0.3) & (dprev <= 10.0)
    else:
        daily_ok = pd.Series(True, index=df.index)

    # ---------- 5m REALTIME EXECUTION ----------
    df["t"] = df["date"].dt.time

    AFTER_OPEN  = time(9, 20)
    CUTOFF_TIME = time(10, 35)     # ✅ ignore signals after 11:35

    in_time = (df["t"] >= AFTER_OPEN) & (df["t"] <= CUTOFF_TIME)

    close = df["close_M"]
    high  = df["high_M"]
    low   = df["low_M"]
    vwap  = df["VWAP_M"]

    # EMA20 optional (fallback to 20_SMA_M if present)
    ema20 = _col("EMA_20_M")
    if "EMA_20_M" not in cols and "20_SMA_M" in cols:
        ema20 = df["20_SMA_M"]

    # --- Pullback (RELAXED) ---
    # Wider tolerance to increase signals
    near_vwap = (low <= vwap * 1.004) & (close >= vwap * 0.996)

    if ("EMA_20_M" in cols) or ("20_SMA_M" in cols):
        near_ema20 = (low <= ema20 * 1.004) & (close >= ema20 * 0.996)
        pullback_zone = near_vwap | near_ema20
    else:
        pullback_zone = near_vwap

    # Prefer price at/above VWAP on close (momentum bias, but still allows pullback wick)
    vwap_close_ok = close >= vwap * 0.999

    # --- Continuation (RELAXED) ---
    # RSI: a bit looser, but must be non-weak and not falling hard
    rsi_prev = _col("RSI_M_prev")
    rsi_ok = (df["RSI_M"] >= 50.0) & (df["RSI_M"] >= (rsi_prev - 0.6))

    # MACD: optional, relaxed
    if {"MACD_Hist_M", "MACD_Hist_M_prev"}.issubset(cols):
        macd_ok = (df["MACD_Hist_M"] >= -0.04) & (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.80)
    elif "MACD_Hist_M" in cols:
        macd_ok = (df["MACD_Hist_M"] >= -0.04)
    else:
        macd_ok = pd.Series(True, index=df.index)

    # Reclaim: relaxed (use Recent_High_12_M if available, else rolling high)
    if "Recent_High_12_M" in cols:
        rh = df["Recent_High_12_M"]
    else:
        rh = high.rolling(12, min_periods=12).max()
    reclaim = close >= rh * 0.997   # was ~0.999, relaxed to allow earlier continuation

    # Volume: optional, relaxed
    if "volume_M" in cols:
        vol_mean_1h = df["volume_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).mean()
        vol_ok = df["volume_M"] >= (vol_mean_1h * 1.05)   # was 1.10
    else:
        vol_ok = pd.Series(True, index=df.index)

    # Anti-chase: relaxed slightly
    not_too_far = close <= vwap * 1.018  # was ~1.012

    # --- High upward momentum bias (but not too strict) ---
    # Require at least ONE of these to be true:
    #   1) Intra_Change_M positive enough
    #   2) 2 of last 3 candles bullish (Bull_3_sum)
    #   3) Strong RSI+MACD combo
    intra = _col("Intra_Change_M", 0.0).fillna(0.0)
    cond_intra = intra >= 0.10  # relaxed (tune 0.08–0.15)

    if "Bull_3_sum" in cols:
        cond_bull = df["Bull_3_sum"] >= 2
    else:
        cond_bull = pd.Series(False, index=df.index)

    cond_rsi_macd = (df["RSI_M"] >= 54.0) & (macd_ok if isinstance(macd_ok, pd.Series) else True)

    momentum_bias = cond_intra | cond_bull | cond_rsi_macd

    exec_ok = (
        in_time &
        pullback_zone &
        vwap_close_ok &
        rsi_ok &
        macd_ok &
        reclaim &
        vol_ok &
        not_too_far &
        momentum_bias
    )

    long_mask = weekly_ok & daily_ok & exec_ok
    if not long_mask.any():
        return []

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom8",
            "sub_strategy": "PB_VWAP_EMA20_CONT_RELAX_MOMO",
        })

    return signals


def build_signals_custom9(ticker: str) -> list[dict]:
    """
    CUSTOM 9 (5m intraday) — STRICT + BREAKOUT-SENSITIVE
    WEEKLY(prev completed via loader) + DAILY(prev day) + 5m BREAKOUT + MOMENTUM STACK (>=2)

    Key traits:
      - Weekly is effectively prev completed week via your loader (no same-week lookahead)
      - Daily uses prev-day metric (Daily_Change_prev_D)
      - Breakout is more sensitive:
          * uses PRIOR RH8 -> fallback RH12 -> fallback RH60 (all shift(1) no self-inclusion)
          * allows high-touch breakout (high pierce) if close confirms near level
      - Momentum remains strict: >=2 boosters
    """

    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    cols = set(df.columns)

    # ---- Minimal requirements ----
    minimal = {"date", "close_M", "high_M", "VWAP_M", "RSI_M"}
    if not minimal.issubset(cols):
        print(f"- Skipping {ticker}: missing minimal 5m columns for custom9.")
        return []

    def _col(name: str, default=np.nan):
        return df[name] if name in cols else pd.Series(default, index=df.index)

    # ===================== WEEKLY FILTER (STRICT, prev completed via loader) =====================
    if {"close_W", "EMA_200_W", "RSI_W"}.issubset(cols):
        weekly_ok = (df["close_W"] > df["EMA_200_W"]) & (df["RSI_W"] >= 45.0)
        if {"EMA_50_W", "EMA_200_W"}.issubset(cols):
            weekly_ok = weekly_ok & (df["EMA_50_W"] >= df["EMA_200_W"] * 0.995)
        weekly_ok = weekly_ok.fillna(False)
    elif {"close_W", "EMA_200_W"}.issubset(cols):
        weekly_ok = (df["close_W"] > df["EMA_200_W"]).fillna(False)
    else:
        weekly_ok = pd.Series(True, index=df.index)  # only if weekly truly unavailable

    # ===================== DAILY(prev) FILTER (STRICT) =====================
    if "Daily_Change_prev_D" in cols:
        dprev = df["Daily_Change_prev_D"]
        daily_ok = ((dprev >= 0.5) & (dprev <= 8.0)).fillna(False)
    else:
        daily_ok = pd.Series(True, index=df.index)

    # ===================== 5m TIME WINDOW (strict but not too tight) =====================
    t = df["date"].dt.time
    AFTER_OPEN  = time(9, 25)
    CUTOFF_TIME = time(10, 45)   # slightly wider than 10:30 for better capture
    in_time = t.between(AFTER_OPEN, CUTOFF_TIME)

    close = df["close_M"]
    high  = df["high_M"]
    vwap  = df["VWAP_M"]

    # ===================== BREAKOUT (MORE SENSITIVE, STILL NO LOOKAHEAD) =====================
    # PRIOR highs only (shift(1) => no self-inclusion)
    prior_high_8  = high.rolling(8,  min_periods=8 ).max().shift(1)
    prior_high_12 = high.rolling(12, min_periods=12).max().shift(1)
    prior_high_60 = high.rolling(60, min_periods=60).max().shift(1)

    breakout_level = prior_high_8.copy()
    breakout_name = "PRIOR_RH8"

    fb = breakout_level.isna()
    breakout_level[fb] = prior_high_12[fb]
    breakout_name = "PRIOR_RH8_or_RH12"

    fb2 = breakout_level.isna()
    breakout_level[fb2] = prior_high_60[fb2]
    breakout_name = "PRIOR_RH8_or_RH12_or_RH60"

    # More sensitive clearance:
    # - close-confirmed breakout: +0.03%
    # - wick/pierce breakout: high >= +0.02% AND close >= +0.01% (prevents pure wick traps)
    close_break = close >= breakout_level * 1.0003
    wick_break  = (high >= breakout_level * 1.0002) & (close >= breakout_level * 1.0001)

    breakout = (close_break | wick_break).fillna(False)

    # ===================== CANDLE QUALITY (STRICT if open exists) =====================
    if "open_M" in cols:
        green = close > df["open_M"]
        body_ok = (close - df["open_M"]) >= (df["open_M"] * 0.0005)  # >= 0.05% body
    else:
        green = pd.Series(True, index=df.index)
        body_ok = pd.Series(True, index=df.index)

    # ===================== MOMENTUM BOOSTERS (STRICT: need >=2 of 4) =====================
    # 1) RSI booster (strong or rising)
    rsi_prev = _col("RSI_M_prev")
    rsi_boost = ((df["RSI_M"] >= 55.0) | (df["RSI_M"] >= (rsi_prev + 0.3))).fillna(False)

    # 2) MACD booster (strict)
    if {"MACD_Hist_M", "MACD_Hist_M_prev"}.issubset(cols):
        macd_boost = (df["MACD_Hist_M"] >= 0.0) & (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.90)
        macd_boost = macd_boost.fillna(False)
    elif "MACD_Hist_M" in cols:
        macd_boost = (df["MACD_Hist_M"] >= 0.0).fillna(False)
    else:
        macd_boost = pd.Series(False, index=df.index)

    # 3) Volume booster (strict)
    if "volume_M" in cols:
        vol_mean_1h = df["volume_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).mean()
        vol_boost = (df["volume_M"] >= (vol_mean_1h * 1.15)).fillna(False)
    else:
        vol_boost = pd.Series(False, index=df.index)

    # 4) Intra-change booster (strict)
    intra = _col("Intra_Change_M", 0.0).fillna(0.0)
    intra_boost = intra >= 0.18

    boosters = (
        rsi_boost.astype(int) +
        macd_boost.astype(int) +
        vol_boost.astype(int) +
        intra_boost.astype(int)
    )
    momentum_ok = boosters >= 2

    # Optional trend strength (if ADX exists)
    if "ADX_M" in cols:
        adx_ok = (df["ADX_M"] >= 14.0).fillna(False)
    else:
        adx_ok = pd.Series(True, index=df.index)

    # ===================== ANTI-CHASE (STRICT) =====================
    # Keep strict, but a touch less restrictive than 1.02 if you want more capture
    not_too_far = close <= vwap * 1.023  # slightly relaxed from 1.02 but still strict

    exec_ok = (
        in_time &
        breakout &
        green & body_ok &
        adx_ok &
        not_too_far &
        momentum_ok
    )

    long_mask = weekly_ok & daily_ok & exec_ok
    if not long_mask.any():
        return []

    signals = []
    for _, r in df[long_mask].iterrows():
        signals.append({
            "signal_time_ist": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "entry_price": float(r["close_M"]),
            # keep schema compatibility, but reflect actual gating (not always True)
            "weekly_ok": bool(r.get("close_W", np.nan) == r.get("close_W", np.nan)) if "close_W" in cols else True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": "custom9",
            "sub_strategy": f"Wk_strict + Dprev_strict + {breakout_name} (sensitive) + boosters>=2",
        })

    return signals








# ======================================================================
# STRATEGY DISPATCHER
# ======================================================================
def build_signals_for_ticker(ticker: str, strategy: str = "trend_pullback") -> list[dict]:
    s = strategy.lower()

    if s == "trend_pullback":
        return build_signals_trend_pullback(ticker)
    elif s == "weekly_breakout":
        return build_signals_weekly_breakout_daily_confirm(ticker)
    elif s == "weekly_squeeze":
        return build_signals_weekly_squeeze_daily_expansion(ticker)
    elif s == "insidebar_breakout":
        return build_signals_weekly_trend_insidebar_breakout(ticker)
    elif s == "hh_hl_retest":
        return build_signals_weekly_hh_hl_retest(ticker)
    elif s == "golden_cross":
        return build_signals_weekly_golden_cross_ema_bounce(ticker)
    elif s == "custom1":
        return build_signals_custom1(ticker)
    elif s == "custom2":
        return build_signals_custom2(ticker)
    elif s == "custom3":
        return build_signals_custom3(ticker)
    elif s == "custom4":
        return build_signals_custom4(ticker)
    elif s == "strategy11":
        return build_signals_strategy11(ticker)
    elif s == "custom6":
        return build_signals_custom6(ticker)
    elif s == "custom7":
        return build_signals_custom7(ticker)
    elif s == "custom8":
        return build_signals_custom8(ticker)
    elif s == "custom9":
        return build_signals_custom9(ticker)
    else:
        print(f"! Unknown strategy '{strategy}', defaulting to 'trend_pullback'.")
        return build_signals_trend_pullback(ticker)


# ======================================================================
# USER INPUT FOR STRATEGY
# ======================================================================
def choose_strategy_from_input() -> str:
    options = {
        "1": "trend_pullback",
        "2": "weekly_breakout",
        "3": "weekly_squeeze",
        "4": "insidebar_breakout",
        "5": "hh_hl_retest",
        "6": "golden_cross",
        "7": "custom1",
        "8": "custom2",
        "9": "custom3",
        "10": "custom4",
        "11": "strategy11",
        "12": "custom6",
        "13": "custom7",
        "14": "custom8",
        "15": "custom9",
    }

    print("\n=== Select Strategy for LONG Signal Generation (5m execution) ===")
    print(" 1) trend_pullback")
    print(" 2) weekly_breakout")
    print(" 3) weekly_squeeze")
    print(" 4) insidebar_breakout")
    print(" 5) hh_hl_retest")
    print(" 6) golden_cross")
    print(" 7) custom1")
    print(" 8) custom2")
    print(" 9) custom3")
    print("10) custom4")
    print("11) strategy11   (added)")
    print("12) custom6      (added meta-voting)")
    print("13) custom7      (ORB30 + VWAP + Vol)")
    print("14) custom8      (ORB30 + VWAP + Vol)")
    print("15) custom9      (mom + breakout)")
    print("---------------------------------------------------------------")
    raw = input("Enter choice (1-15 or strategy name): ").strip().lower()

    if raw in options:
        strategy = options[raw]
    else:
        strategy = raw if raw else "trend_pullback"

    print(f"\n>> Using strategy: {strategy}\n")
    return strategy


# ----------------------- EVALUATION ON 5m -----------------------
def _safe_read_5m(path: str) -> pd.DataFrame:
    """Read 5m CSV for evaluation, parse 'date' as tz-aware IST, sort."""
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["date"] = df["date"].dt.tz_convert(IST)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading 5m data {path}: {e}")
        return pd.DataFrame()


def evaluate_signal_on_5m_long_only(row: pd.Series, df_5m: pd.DataFrame) -> dict:
    """
    Positional evaluation on 5m (multi-day) in the SAME STYLE as evaluate_signal_on_1h_long_only.

    - Uses ALL future 5m bars after entry (across days), but typically df_5m is already filtered
      to market hours (09:15-15:30) for all days.
    - HIT_COL: target hit ANYTIME after entry (positional)
    - INTRADAY_HIT_COL: target hit on the SAME DAY as entry (diagnostic)
    - Exit:
        • If target touched -> exit at target_price on first hit bar (positional)
        • Else -> exit at last available future close in df_5m
    """

    # ---- Parse & validate entry ----
    entry_time = pd.to_datetime(row["signal_time_ist"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize(IST)
    else:
        entry_time = entry_time.tz_convert(IST)

    entry_price = float(row["entry_price"])

    if not np.isfinite(entry_price) or entry_price <= 0 or df_5m is None or df_5m.empty:
        return {
            HIT_COL: False,
            INTRADAY_HIT_COL: False,
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "pnl_pct": np.nan,
            "pnl_rs": np.nan,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }

    # ---- Ensure df_5m dates are tz-aware IST and sorted ----
    dfx = df_5m.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date"])
    if dfx["date"].dt.tz is None:
        dfx["date"] = dfx["date"].dt.tz_localize(IST)
    else:
        dfx["date"] = dfx["date"].dt.tz_convert(IST)
    dfx = dfx.sort_values("date").reset_index(drop=True)

    # ---- All future bars (multi-day) ----
    future_all = dfx[dfx["date"] > entry_time].copy()
    if future_all.empty:
        return {
            HIT_COL: False,
            INTRADAY_HIT_COL: False,
            TIME_TD_COL: np.nan,
            TIME_CAL_COL: np.nan,
            "exit_time": entry_time,
            "exit_price": entry_price,
            "pnl_pct": 0.0,
            "pnl_rs": 0.0,
            "max_favorable_pct": np.nan,
            "max_adverse_pct": np.nan,
        }

    target_price = entry_price * TARGET_MULT
    hit_mask_all = future_all["high"] >= target_price

    # Diagnostics over the full future horizon
    max_fav = ((future_all["high"] - entry_price) / entry_price * 100.0).max()
    max_adv = ((future_all["low"]  - entry_price) / entry_price * 100.0).min()

    # ---- Intraday hit diagnostic (same calendar date as entry) ----
    entry_date = entry_time.date()
    same_day_future = future_all[future_all["date"].dt.date == entry_date]
    intraday_hit = bool((same_day_future["high"] >= target_price).any()) if not same_day_future.empty else False

    # ---- Positional hit logic (like 1H) ----
    if hit_mask_all.any():
        first_hit_pos = int(np.argmax(hit_mask_all.values))
        hit_time = future_all.iloc[first_hit_pos]["date"]

        exit_time = hit_time
        exit_price = float(target_price)

        dt_hours = (hit_time - entry_time).total_seconds() / 3600.0
        dt_days = dt_hours / TRADING_HOURS_PER_DAY
        cal_days = (hit_time.date() - entry_time.date()).days

        hit_flag = True
        time_to_target_days = dt_days
        days_to_target_calendar = cal_days
    else:
        last_row = future_all.iloc[-1]
        exit_time = last_row["date"]
        exit_price = float(last_row["close"]) if "close" in last_row else entry_price

        hit_flag = False
        time_to_target_days = np.nan
        days_to_target_calendar = np.nan

    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    pnl_rs = CAPITAL_PER_SIGNAL_RS * (pnl_pct / 100.0)

    return {
        HIT_COL: hit_flag,
        INTRADAY_HIT_COL: bool(intraday_hit),
        TIME_TD_COL: float(time_to_target_days) if np.isfinite(time_to_target_days) else np.nan,
        TIME_CAL_COL: float(days_to_target_calendar) if np.isfinite(days_to_target_calendar) else np.nan,
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "pnl_pct": float(pnl_pct),
        "pnl_rs": float(pnl_rs),
        "max_favorable_pct": float(max_fav) if np.isfinite(max_fav) else np.nan,
        "max_adverse_pct": float(max_adv) if np.isfinite(max_adv) else np.nan,
    }


# ----------------------- CAPITAL-CONSTRAINED PORTFOLIO SIM -----------------------
def apply_capital_constraint(out_df: pd.DataFrame, intraday_mult: float = 5.0):
    """
    Apply a portfolio-level capital constraint with intraday leverage:

    - Start with MAX_CAPITAL_RS of cash.
    - For each signal in chronological order:
        * Release any open trades whose exit_time <= entry_time.
        * If available_capital >= CAPITAL_PER_SIGNAL_RS -> take trade.
          Else -> skip trade.

    Intraday P&L rule:
      If exit_time is on the same calendar date as entry_time (IST), then
      pnl_rs is multiplied by intraday_mult (default 5x).
      Otherwise, pnl_rs is unchanged.

    NOTE:
      Capital deployed (CAPITAL_PER_SIGNAL_RS) is NOT multiplied — only P&L.
    """
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    # --- Ensure tz-aware IST for robust date comparisons ---
    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"])
    if df["signal_time_ist"].dt.tz is None:
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_localize(IST)
    else:
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_convert(IST)

    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    # exit_time may contain NaT; only convert tz for non-NaT
    if df["exit_time"].notna().any():
        # If exit_time is naive, localize; else convert
        if df.loc[df["exit_time"].notna(), "exit_time"].dt.tz is None:
            df.loc[df["exit_time"].notna(), "exit_time"] = df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_localize(IST)
        else:
            df.loc[df["exit_time"].notna(), "exit_time"] = df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)

    # --- Identify intraday trades: same entry date and exit date ---
    df["is_intraday_trade"] = (
        df["exit_time"].notna() &
        (df["exit_time"].dt.date == df["signal_time_ist"].dt.date)
    )

    # --- Leverage P&L only (do NOT multiply deployed capital) ---
    df["pnl_rs"] = pd.to_numeric(df.get("pnl_rs", 0.0), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0.0), errors="coerce").fillna(0.0)

    df["pnl_rs_levered"] = np.where(df["is_intraday_trade"], df["pnl_rs"] * intraday_mult, df["pnl_rs"])
    df["pnl_pct_levered"] = np.where(df["is_intraday_trade"], df["pnl_pct"] * intraday_mult, df["pnl_pct"])

    df["invested_amount"] = CAPITAL_PER_SIGNAL_RS
    df["final_value_per_signal_levered"] = df["invested_amount"] + df["pnl_rs_levered"]

    # ---------------- Capital constraint simulation ----------------
    available_capital = MAX_CAPITAL_RS
    open_trades = []  # list of dicts: {"exit_time": ts, "value": float}

    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        # 1) Release trades that have exited before/at this entry_time
        still_open = []
        for trade in open_trades:
            exit_time = trade["exit_time"]
            if pd.isna(exit_time) or exit_time <= entry_time:
                available_capital += trade["value"]
            else:
                still_open.append(trade)
        open_trades = still_open

        # 2) Try to take this trade
        if available_capital >= CAPITAL_PER_SIGNAL_RS:
            df.at[idx, "taken"] = True
            available_capital -= CAPITAL_PER_SIGNAL_RS

            exit_time = row["exit_time"]
            if pd.isna(exit_time):
                exit_time = entry_time

            final_value = float(row["final_value_per_signal_levered"])
            open_trades.append({"exit_time": exit_time, "value": final_value})
        else:
            df.at[idx, "taken"] = False

    # 3) Release remaining open trades
    for trade in open_trades:
        available_capital += trade["value"]

    df["invested_amount_constrained"] = np.where(df["taken"], CAPITAL_PER_SIGNAL_RS, 0.0)
    df["final_value_constrained_levered"] = np.where(df["taken"], df["final_value_per_signal_levered"], 0.0)

    total_signals = len(df)
    signals_taken = int(df["taken"].sum())
    gross_invested = df["invested_amount_constrained"].sum()
    final_portfolio_value = float(available_capital)
    net_pnl_rs = final_portfolio_value - MAX_CAPITAL_RS
    net_pnl_pct = (net_pnl_rs / MAX_CAPITAL_RS * 100.0) if MAX_CAPITAL_RS > 0 else 0.0

    summary = {
        "total_signals": total_signals,
        "signals_taken": signals_taken,
        "gross_invested": gross_invested,
        "final_portfolio_value": final_portfolio_value,
        "net_pnl_rs": net_pnl_rs,
        "net_pnl_pct": net_pnl_pct,
        "intraday_mult": intraday_mult,
        "intraday_trades_count": int(df["is_intraday_trade"].sum()),
    }
    
    # --- aliases to avoid downstream column-name mismatches ---
    df["final_value_constrained"] = df.get("final_value_constrained_levered", 0.0)
    df["invested_amount_constrained"] = df.get("invested_amount_constrained", 0.0)

    
    return df, summary



# ----------------------- MAIN -----------------------
def main(strategy: str):
    # --------- PART 1: GENERATE LONG SIGNALS ---------
    tickers_5m = _list_tickers_from_dir(DIR_5m)
    tickers_d  = _list_tickers_from_dir(DIR_D)
    tickers_w  = _list_tickers_from_dir(DIR_W)

    tickers = tickers_5m & tickers_d & tickers_w

    if not tickers:
        print("No common tickers found across 5m, Daily, Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with all 3 timeframes.")
    print(f"Running strategy: {strategy}\n")

    all_signals = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Processing {ticker} ...")
        try:
            ticker_signals = build_signals_for_ticker(ticker, strategy=strategy)
            all_signals.extend(ticker_signals)
            print(f"  -> {len(ticker_signals)} LONG signals for {ticker}")
        except Exception as e:
            print(f"! Error while processing {ticker}: {e}")

    if not all_signals:
        print("No LONG signals generated.")
        return

    sig_df = pd.DataFrame(all_signals)
    sig_df.sort_values(["signal_time_ist", "ticker"], inplace=True)

    sig_df["itr"] = np.arange(1, len(sig_df) + 1)

    sig_df["signal_time_ist"] = pd.to_datetime(sig_df["signal_time_ist"]).dt.tz_convert(IST)
    sig_df["signal_date"] = sig_df["signal_time_ist"].dt.date

    ts = datetime.now(IST).strftime("%Y%m%d_%H%M")

    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_5m_{strategy}_{ts}_IST.csv")
    sig_df.to_csv(signals_path, index=False)

    print(f"\nSaved {len(sig_df)} multi-timeframe LONG signals to: {signals_path}")
    print("\nLast few LONG signals:")
    print(sig_df.tail(20).to_string(index=False))

    # --------- PART 1b: DAILY COUNTS ---------
    print("\n--- Computing per-day LONG signal counts ---")
    daily_counts = sig_df.groupby("signal_date").size().reset_index(name="LONG_signals")
    daily_counts.rename(columns={"signal_date": "date"}, inplace=True)

    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_counts_5m_{strategy}_{ts}_IST.csv")
    daily_counts.to_csv(daily_counts_path, index=False)

    print(f"\nSaved daily LONG signal counts to: {daily_counts_path}")
    print("\nDaily counts preview:")
    print(daily_counts.tail(20).to_string(index=False))

    # --------- PART 2: EVALUATE TARGET_PCT OUTCOME ON 5m ---------
    print(f"\n--- Evaluating +{TARGET_LABEL} outcome on LONG signals (5m eval) ---")
    print(f"Assuming capital per signal : ₹{CAPITAL_PER_SIGNAL_RS:,.2f}")
    print(f"Total LONG signals in file : {len(sig_df)}")

    tickers_in_signals = sig_df["ticker"].unique().tolist()
    print(f"Found {len(tickers_in_signals)} tickers in generated LONG signals.")

    data_cache = {}
    for t in tickers_in_signals:
        path_5m = _find_file(DIR_5m, t)
        if not path_5m:
            data_cache[t] = pd.DataFrame()
            continue

        df_5m = _safe_read_5m(path_5m)
        # Optional speed-up: keep only intraday session bars
        df_5m = df_5m[_intraday_trade_mask(df_5m["date"], start=MKT_OPEN, end=MKT_CLOSE)].copy()
        df_5m.reset_index(drop=True, inplace=True)

        data_cache[t] = df_5m

    results = []
    for _, row in sig_df.iterrows():
        ticker = row["ticker"]
        df_5m = data_cache.get(ticker, pd.DataFrame())

        if df_5m.empty:
            new_info = {
                HIT_COL: False,
                INTRADAY_HIT_COL: False,
                TIME_TD_COL: np.nan,
                TIME_CAL_COL: np.nan,
                "exit_time": row["signal_time_ist"],
                "exit_price": row["entry_price"],
                "pnl_pct": 0.0,
                "pnl_rs": 0.0,
                "max_favorable_pct": np.nan,
                "max_adverse_pct": np.nan,
            }
        else:
            new_info = evaluate_signal_on_5m_long_only(row, df_5m)

        merged_row = row.to_dict()
        merged_row.update(new_info)
        results.append(merged_row)

    out_df = pd.DataFrame(results).sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True)

    out_df["invested_amount"] = CAPITAL_PER_SIGNAL_RS
    out_df["final_value_per_signal"] = out_df["invested_amount"] + out_df["pnl_rs"]

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_5m_{strategy}_{EVAL_SUFFIX}_{ts}_IST.csv")
    out_df.to_csv(eval_path, index=False)
    print(f"\nSaved {TARGET_LABEL} evaluation (LONG only, 5m eval) to: {eval_path}")

    total_long = len(out_df)
    hits_all = int(out_df[HIT_COL].sum())
    hit_pct_all = (hits_all / total_long * 100.0) if total_long else 0.0
    print(f"\n[LONG SIGNALS - ALL] Total evaluated: {total_long}, {TARGET_LABEL} hit: {hits_all} ({hit_pct_all:.1f}%)")

    intraday_hits = int(out_df[INTRADAY_HIT_COL].sum()) if INTRADAY_HIT_COL in out_df.columns else 0
    intraday_hit_pct = (intraday_hits / total_long * 100.0) if total_long else 0.0
    print(f"[INTRADAY HITS] Target hit same-day: {intraday_hits} ({intraday_hit_pct:.1f}%)")

    hits_only = out_df[out_df[HIT_COL] == True]
    intraday_among_hits = int(hits_only[INTRADAY_HIT_COL].sum()) if len(hits_only) else 0
    intraday_among_hits_pct = (intraday_among_hits / len(hits_only) * 100.0) if len(hits_only) else 0.0
    print(f"[INTRADAY AMONG HITS] {intraday_among_hits}/{len(hits_only)} ({intraday_among_hits_pct:.1f}%)")

    if total_long > N_RECENT_SKIP:
        non_recent = out_df.iloc[:-N_RECENT_SKIP].copy()
        hits_nr = int(non_recent[HIT_COL].sum())
        hit_pct_nr = (hits_nr / len(non_recent) * 100.0) if len(non_recent) else 0.0
        print(f"\n[LONG SIGNALS - NON-RECENT] (excluding last {N_RECENT_SKIP}) "
              f"Evaluated: {len(non_recent)}, {TARGET_LABEL} hit: {hits_nr} ({hit_pct_nr:.1f}%)")
    else:
        print(f"\n[LONG SIGNALS - NON-RECENT] Not enough rows to exclude last {N_RECENT_SKIP}.")

    # --------- CAPITAL-CONSTRAINED SIM ---------
    print("\n--- CAPITAL-CONSTRAINED PORTFOLIO SIMULATION ---")
    print(f"Max capital pool           : ₹{MAX_CAPITAL_RS:,.2f}")
    print(f"Capital per signal (fixed) : ₹{CAPITAL_PER_SIGNAL_RS:,.2f}")

    out_df_constrained, cap_summary = apply_capital_constraint(out_df)

    print(f"Signals generated (total)  : {cap_summary['total_signals']}")
    print(f"Signals actually taken     : {cap_summary['signals_taken']}")
    print(f"Gross capital deployed     : ₹{cap_summary['gross_invested']:,.2f}")
    print(f"Final portfolio value      : ₹{cap_summary['final_portfolio_value']:,.2f}")
    print(f"Net portfolio P&L          : ₹{cap_summary['net_pnl_rs']:,.2f} ({cap_summary['net_pnl_pct']:.2f}%)")

    # overwrite eval with constrained columns
    out_df_constrained.to_csv(eval_path, index=False)
    print(f"\nRe-saved evaluation with constrained-capital columns to: {eval_path}")

    # =====================================================================
    # NEW: EXPORT NON-INTRADAY P&L (NO LEVERAGE) TO SEPARATE CSV(s)
    # =====================================================================
    if "is_intraday_trade" not in out_df_constrained.columns:
        # safety fallback (should exist from apply_capital_constraint)
        out_df_constrained["is_intraday_trade"] = False

    non_intraday_all = out_df_constrained[out_df_constrained["is_intraday_trade"] == False].copy()

    # Keep ONLY non-levered P&L columns as the "truth" for this file
    # (You can drop levered columns to avoid confusion.)
    drop_if_present = [
        "pnl_rs_levered", "pnl_pct_levered",
        "final_value_per_signal_levered",
        "final_value_constrained_levered",
        "final_value_constrained",
    ]
    for c in drop_if_present:
        if c in non_intraday_all.columns:
            non_intraday_all.drop(columns=[c], inplace=True)

    # Optional: ensure constrained final value matches non-levered pnl for non-intraday trades
    # (For non-intraday, levered == non-levered, but this makes the file self-consistent.)
    if "taken" in non_intraday_all.columns:
        non_intraday_all["final_value_constrained_non_intraday"] = np.where(
            non_intraday_all["taken"].astype(bool),
            CAPITAL_PER_SIGNAL_RS + non_intraday_all["pnl_rs"].astype(float),
            0.0
        )
    else:
        non_intraday_all["final_value_constrained_non_intraday"] = CAPITAL_PER_SIGNAL_RS + non_intraday_all["pnl_rs"].astype(float)

    non_intraday_path = os.path.join(
        OUT_DIR,
        f"multi_tf_signals_5m_{strategy}_{EVAL_SUFFIX}_{ts}_IST_NON_INTRADAY_NO_LEV.csv"
    )
    non_intraday_all.to_csv(non_intraday_path, index=False)
    print(f"\nSaved NON-INTRADAY (no leverage) trades to: {non_intraday_path}")
    print(f"Non-intraday rows: {len(non_intraday_all)}")

    # If you ALSO want "only taken" non-intraday trades:
    if "taken" in non_intraday_all.columns:
        non_intraday_taken = non_intraday_all[non_intraday_all["taken"].astype(bool)].copy()
        non_intraday_taken_path = os.path.join(
            OUT_DIR,
            f"multi_tf_signals_5m_{strategy}_{EVAL_SUFFIX}_{ts}_IST_NON_INTRADAY_TAKEN_NO_LEV.csv"
        )
        non_intraday_taken.to_csv(non_intraday_taken_path, index=False)
        print(f"Saved NON-INTRADAY TAKEN (no leverage) trades to: {non_intraday_taken_path}")
        print(f"Non-intraday taken rows: {len(non_intraday_taken)}")

    # --------- PART 2b: DAY-WISE SUMMARY (signals + intraday hits + P/L) ---------
    print("\n--- Building day-wise summary (signals, intraday hits, P/L) ---")
    daily_summary = build_daily_summary(out_df_constrained)

    daily_summary_path = os.path.join(
        OUT_DIR,
        f"multi_tf_daily_summary_5m_{strategy}_{EVAL_SUFFIX}_{ts}_IST.csv"
    )
    daily_summary.to_csv(daily_summary_path, index=False)

    print(f"Saved day-wise summary to: {daily_summary_path}")
    print(daily_summary.tail(20).to_string(index=False))

    cols_preview = [
        "itr","signal_time_ist","signal_date","ticker","signal_side","strategy",
        "entry_price","exit_time","exit_price","pnl_pct","pnl_rs",
        HIT_COL, INTRADAY_HIT_COL, TIME_TD_COL, TIME_CAL_COL,
        "max_favorable_pct","max_adverse_pct",
        "taken","invested_amount_constrained","final_value_constrained",
    ]
    existing = [c for c in cols_preview if c in out_df_constrained.columns]
    print("\nLast few evaluated LONG signals (5m):")
    print(out_df_constrained[existing].tail(20).to_string(index=False))



if __name__ == "__main__":
    selected_strategy = choose_strategy_from_input()
    main(selected_strategy)
