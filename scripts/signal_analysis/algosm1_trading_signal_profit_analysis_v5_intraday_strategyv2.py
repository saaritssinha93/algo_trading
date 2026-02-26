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
1) Generate MTF LONG signals using ONLY CUSTOM9 (Weekly + Daily + 5m execution logic)
2) Evaluate those signals on 5m data for +TARGET_PCT% target hit
3) Apply capital constraint simulation
4) Export signals + eval + constrained eval + non-intraday pnl + day-wise summary

NOTE:
- Removed ALL other strategies and the select/dispatcher option.
- Everything else is kept aligned in logic and functioning.
"""

import os
import glob
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_5m  = "main_indicators_5min"
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals"

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
MAX_CAPITAL_RS: float = 1500000.0

TRADING_HOURS_PER_DAY = 6.25
N_RECENT_SKIP = 1

# 5m scaling helpers
ROLL_MEAN_1H_EQ = 12          # ~1 hour on 5m

# ----------------------- INTRADAY SESSION (IST) -----------------------
MKT_OPEN  = time(9, 15)
MKT_CLOSE = time(15, 30)

# For entries, keep a buffer so you still have time to exit intraday if no target hit
ENTRY_START = time(9, 20)
ENTRY_END   = time(15, 20)

# Hardwired single strategy name
STRATEGY_NAME = "custom9"


# ----------------------- INTRADAY MASKS -----------------------
def _intraday_entry_mask(ts: pd.Series,
                         start: time = ENTRY_START,
                         end: time = ENTRY_END) -> pd.Series:
    """True if timestamp time is within allowed ENTRY window. Assumes tz-aware IST."""
    return ts.dt.time.between(start, end)

def _intraday_trade_mask(ts: pd.Series,
                         start: time = MKT_OPEN,
                         end: time = MKT_CLOSE) -> pd.Series:
    """True if timestamp time is within trading window. Used for evaluation future bars."""
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

def _list_tickers_from_dir(directory: str) -> set[str]:
    """
    Robust ticker extraction:
    - Finds files containing '_main_indicators'
    - Extracts ticker as the part BEFORE '_main_indicators'
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

    Weekly: NO same-date lookahead is enforced by +1s shift.

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
        print(f"- Skipping {ticker}: empty one or more TF dataframes.")
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

    # ✅ Enforce intraday entry window globally
    df_mdw = df_mdw[_intraday_entry_mask(df_mdw["date"])].copy()
    df_mdw.reset_index(drop=True, inplace=True)
    return df_mdw


# ----------------------- 5m EXTRAS (common utility) -----------------------
def _add_5m_extras(df_mdw: pd.DataFrame) -> pd.DataFrame:
    """
    Add common 5m extras used by custom9:
      - RSI_M_prev
      - MACD_Hist_M_prev
    """
    if df_mdw.empty:
        return df_mdw

    df = df_mdw.copy()

    if "RSI_M" in df.columns:
        df["RSI_M_prev"] = df["RSI_M"].shift(1)
    else:
        df["RSI_M_prev"] = np.nan

    if "MACD_Hist_M" in df.columns:
        df["MACD_Hist_M_prev"] = df["MACD_Hist_M"].shift(1)
    else:
        df["MACD_Hist_M_prev"] = np.nan

    return df


# ======================================================================
# ONLY STRATEGY: CUSTOM 9
# ======================================================================
def build_signals_custom9(ticker: str) -> list[dict]:
    """
    CUSTOM 9 (5m intraday) — STRICT + BREAKOUT-SENSITIVE

    - Weekly strict: close_W > EMA_200_W and RSI_W >= 46; EMA_50_W >= EMA_200_W (if available)
    - Daily(prev) band: 0.8 to 6.5 (if Daily_Change_prev_D exists)
    - Time window: 09:25 to 10:25
    - Breakout uses PRIOR rolling highs (8/12/60) with shift(1) (no lookahead)
    - Requires close > VWAP (confirmation)
    - Candle quality: green + min body
    - Momentum boosters >= 2 of: RSI boost, MACD boost, Vol boost, Intra_Change boost
    - ADX >= 16 (if available)
    - Anti-chase: close <= VWAP * 1.018
    """
    df = _load_mtf_context_5m(ticker)
    if df.empty:
        return []
    df = _add_5m_extras(df)

    cols = set(df.columns)

    minimal = {"date", "close_M", "high_M", "VWAP_M", "RSI_M"}
    if not minimal.issubset(cols):
        print(f"- Skipping {ticker}: missing minimal 5m columns for custom9.")
        return []

    def _col(name: str, default=np.nan):
        return df[name] if name in cols else pd.Series(default, index=df.index)

    # ===================== WEEKLY FILTER (strict) =====================
    if {"close_W", "EMA_200_W", "RSI_W"}.issubset(cols):
        weekly_ok = (df["close_W"] > df["EMA_200_W"]) & (df["RSI_W"] >= 46.0)
        if {"EMA_50_W", "EMA_200_W"}.issubset(cols):
            weekly_ok = weekly_ok & (df["EMA_50_W"] >= df["EMA_200_W"] * 1.000)
        weekly_ok = weekly_ok.fillna(False)
    elif {"close_W", "EMA_200_W"}.issubset(cols):
        weekly_ok = (df["close_W"] > df["EMA_200_W"]).fillna(False)
    else:
        weekly_ok = pd.Series(True, index=df.index)

    # ===================== DAILY(prev) FILTER (band) =====================
    if "Daily_Change_prev_D" in cols:
        dprev = df["Daily_Change_prev_D"]
        daily_ok = ((dprev >= 0.8) & (dprev <= 7)).fillna(False)
    else:
        daily_ok = pd.Series(True, index=df.index)

    # ===================== 5m TIME WINDOW (tighter) =====================
    t = df["date"].dt.time
    AFTER_OPEN  = time(9, 20)
    CUTOFF_TIME = time(10, 45)
    in_time = t.between(AFTER_OPEN, CUTOFF_TIME)

    close = df["close_M"]
    high  = df["high_M"]
    vwap  = df["VWAP_M"]

    # ===================== BREAKOUT (no lookahead) =====================
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

    close_break = close >= breakout_level * 1.0005
    wick_break  = (high >= breakout_level * 1.0004) & (close >= breakout_level * 1.0003)
    breakout = (close_break | wick_break).fillna(False)

    # ===================== VWAP CONFIRMATION =====================
    vwap_confirm = (close > vwap).fillna(False)

    # ===================== CANDLE QUALITY =====================
    if "open_M" in cols:
        green = close > df["open_M"]
        body_ok = (close - df["open_M"]) >= (df["open_M"] * 0.0009)  # >= 0.09% body
    else:
        green = pd.Series(True, index=df.index)
        body_ok = pd.Series(True, index=df.index)

    # ===================== MOMENTUM BOOSTERS (>=2) =====================
    rsi_prev = _col("RSI_M_prev")
    rsi_boost = ((df["RSI_M"] >= 57.0) | (df["RSI_M"] >= (rsi_prev + 0.7))).fillna(False)

    if {"MACD_Hist_M", "MACD_Hist_M_prev"}.issubset(cols):
        macd_boost = (df["MACD_Hist_M"] >= 0.0) & (df["MACD_Hist_M"] >= df["MACD_Hist_M_prev"] * 0.97)
        macd_boost = macd_boost.fillna(False)
    elif "MACD_Hist_M" in cols:
        macd_boost = (df["MACD_Hist_M"] >= 0.0).fillna(False)
    else:
        macd_boost = pd.Series(False, index=df.index)

    if "volume_M" in cols:
        vol_mean_1h = df["volume_M"].rolling(ROLL_MEAN_1H_EQ, min_periods=ROLL_MEAN_1H_EQ).mean()
        vol_boost = (df["volume_M"] >= (vol_mean_1h * 1.25)).fillna(False)
    else:
        vol_boost = pd.Series(False, index=df.index)

    intra = _col("Intra_Change_M", 0.0).fillna(0.0)
    intra_boost = intra >= 0.22

    boosters = (
        rsi_boost.astype(int) +
        macd_boost.astype(int) +
        vol_boost.astype(int) +
        intra_boost.astype(int)
    )
    momentum_ok = boosters >= 2

    # Trend strength
    if "ADX_M" in cols:
        adx_ok = (df["ADX_M"] >= 16.0).fillna(False)
    else:
        adx_ok = pd.Series(True, index=df.index)

    # ===================== ANTI-CHASE =====================
    not_too_far = close <= vwap * 1.018

    exec_ok = (
        in_time &
        breakout &
        vwap_confirm &
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
            "weekly_ok": True,
            "daily_ok": True,
            "exec_ok": True,
            "strategy": STRATEGY_NAME,
            "sub_strategy": f"Wk_strict + Dprev(0.8-7) + {breakout_name} + VWAP_confirm + boosters>=2",
        })
    return signals


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
    Positional evaluation on 5m (multi-day).

    - HIT_COL: target hit ANYTIME after entry (positional)
    - INTRADAY_HIT_COL: target hit on SAME DAY as entry (diagnostic)
    - Exit:
        • If target touched -> exit at target_price on first hit bar
        • Else -> exit at last available future close
    """
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

    dfx = df_5m.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date"])
    if dfx["date"].dt.tz is None:
        dfx["date"] = dfx["date"].dt.tz_localize(IST)
    else:
        dfx["date"] = dfx["date"].dt.tz_convert(IST)
    dfx = dfx.sort_values("date").reset_index(drop=True)

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

    max_fav = ((future_all["high"] - entry_price) / entry_price * 100.0).max()
    max_adv = ((future_all["low"]  - entry_price) / entry_price * 100.0).min()

    entry_date = entry_time.date()
    same_day_future = future_all[future_all["date"].dt.date == entry_date]
    intraday_hit = bool((same_day_future["high"] >= target_price).any()) if not same_day_future.empty else False

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

    - Start with MAX_CAPITAL_RS cash.
    - For each signal chronologically:
        * Release any open trades whose exit_time <= entry_time.
        * If available_capital >= CAPITAL_PER_SIGNAL_RS -> take trade else skip.

    Intraday P&L rule:
      If exit_time is on same calendar date as entry_time (IST),
      pnl_rs is multiplied by intraday_mult (default 5x).
      Otherwise unchanged.

    NOTE: Deployed capital is NOT multiplied — only P&L.
    """
    df = out_df.sort_values(["signal_time_ist", "ticker"]).reset_index(drop=True).copy()

    df["signal_time_ist"] = pd.to_datetime(df["signal_time_ist"])
    if df["signal_time_ist"].dt.tz is None:
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_localize(IST)
    else:
        df["signal_time_ist"] = df["signal_time_ist"].dt.tz_convert(IST)

    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    if df["exit_time"].notna().any():
        if df.loc[df["exit_time"].notna(), "exit_time"].dt.tz is None:
            df.loc[df["exit_time"].notna(), "exit_time"] = df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_localize(IST)
        else:
            df.loc[df["exit_time"].notna(), "exit_time"] = df.loc[df["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)

    df["is_intraday_trade"] = (
        df["exit_time"].notna() &
        (df["exit_time"].dt.date == df["signal_time_ist"].dt.date)
    )

    df["pnl_rs"] = pd.to_numeric(df.get("pnl_rs", 0.0), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0.0), errors="coerce").fillna(0.0)

    df["pnl_rs_levered"] = np.where(df["is_intraday_trade"], df["pnl_rs"] * intraday_mult, df["pnl_rs"])
    df["pnl_pct_levered"] = np.where(df["is_intraday_trade"], df["pnl_pct"] * intraday_mult, df["pnl_pct"])

    df["invested_amount"] = CAPITAL_PER_SIGNAL_RS
    df["final_value_per_signal_levered"] = df["invested_amount"] + df["pnl_rs_levered"]

    available_capital = MAX_CAPITAL_RS
    open_trades = []
    df["taken"] = False

    for idx, row in df.iterrows():
        entry_time = row["signal_time_ist"]

        still_open = []
        for trade in open_trades:
            exit_time = trade["exit_time"]
            if pd.isna(exit_time) or exit_time <= entry_time:
                available_capital += trade["value"]
            else:
                still_open.append(trade)
        open_trades = still_open

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

    # aliases to avoid downstream mismatches
    df["final_value_constrained"] = df.get("final_value_constrained_levered", 0.0)
    df["invested_amount_constrained"] = df.get("invested_amount_constrained", 0.0)

    return df, summary


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

    for c in ["pnl_rs", "pnl_rs_levered", "pnl_pct", "pnl_pct_levered"]:
        dfx[c] = pd.to_numeric(dfx.get(c, 0.0), errors="coerce").fillna(0.0)

    if INTRADAY_HIT_COL in dfx.columns:
        dfx[INTRADAY_HIT_COL] = dfx[INTRADAY_HIT_COL].astype(bool).fillna(False)
    else:
        dfx[INTRADAY_HIT_COL] = False

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

    rows = []
    for day, g in dfx.groupby("date", dropna=False):
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

            "pnl_rs_intraday_sum": _sum_where(g["pnl_rs"], intraday),
            "pnl_rs_levered_intraday_sum": _sum_where(g["pnl_rs_levered"], intraday),

            "pnl_rs_intraday_taken_sum": _sum_where(g["pnl_rs"], intraday & taken),
            "pnl_rs_levered_intraday_taken_sum": _sum_where(g["pnl_rs_levered"], intraday & taken),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ----------------------- MAIN -----------------------
def main():
    # --------- PART 1: GENERATE LONG SIGNALS (CUSTOM9 ONLY) ---------
    tickers_5m = _list_tickers_from_dir(DIR_5m)
    tickers_d  = _list_tickers_from_dir(DIR_D)
    tickers_w  = _list_tickers_from_dir(DIR_W)

    tickers = tickers_5m & tickers_d & tickers_w
    if not tickers:
        print("No common tickers found across 5m, Daily, Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with all 3 timeframes.")
    print(f"Running strategy: {STRATEGY_NAME}\n")

    all_signals = []
    for i, ticker in enumerate(sorted(tickers), start=1):
        print(f"[{i}/{len(tickers)}] Processing {ticker} ...")
        try:
            ticker_signals = build_signals_custom9(ticker)
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

    signals_path = os.path.join(OUT_DIR, f"multi_tf_signals_5m_{STRATEGY_NAME}_{ts}_IST.csv")
    sig_df.to_csv(signals_path, index=False)

    print(f"\nSaved {len(sig_df)} multi-timeframe LONG signals to: {signals_path}")
    print("\nLast few LONG signals:")
    print(sig_df.tail(20).to_string(index=False))

    # --------- PART 1b: DAILY COUNTS ---------
    print("\n--- Computing per-day LONG signal counts ---")
    daily_counts = sig_df.groupby("signal_date").size().reset_index(name="LONG_signals")
    daily_counts.rename(columns={"signal_date": "date"}, inplace=True)

    daily_counts_path = os.path.join(OUT_DIR, f"multi_tf_signals_daily_counts_5m_{STRATEGY_NAME}_{ts}_IST.csv")
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
    for tkr in tickers_in_signals:
        path_5m = _find_file(DIR_5m, tkr)
        if not path_5m:
            data_cache[tkr] = pd.DataFrame()
            continue

        df_5m = _safe_read_5m(path_5m)
        df_5m = df_5m[_intraday_trade_mask(df_5m["date"], start=MKT_OPEN, end=MKT_CLOSE)].copy()
        df_5m.reset_index(drop=True, inplace=True)
        data_cache[tkr] = df_5m

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

    eval_path = os.path.join(OUT_DIR, f"multi_tf_signals_5m_{STRATEGY_NAME}_{EVAL_SUFFIX}_{ts}_IST.csv")
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

    # --------- Export NON-INTRADAY P&L (NO LEVERAGE) ---------
    dfx = out_df_constrained.copy()

    dfx["signal_time_ist"] = pd.to_datetime(dfx["signal_time_ist"])
    if dfx["signal_time_ist"].dt.tz is None:
        dfx["signal_time_ist"] = dfx["signal_time_ist"].dt.tz_localize(IST)
    else:
        dfx["signal_time_ist"] = dfx["signal_time_ist"].dt.tz_convert(IST)

    dfx["exit_time"] = pd.to_datetime(dfx["exit_time"], errors="coerce")
    if dfx["exit_time"].notna().any():
        if dfx.loc[dfx["exit_time"].notna(), "exit_time"].dt.tz is None:
            dfx.loc[dfx["exit_time"].notna(), "exit_time"] = dfx.loc[dfx["exit_time"].notna(), "exit_time"].dt.tz_localize(IST)
        else:
            dfx.loc[dfx["exit_time"].notna(), "exit_time"] = dfx.loc[dfx["exit_time"].notna(), "exit_time"].dt.tz_convert(IST)

    dfx["is_intraday_trade"] = (
        dfx["exit_time"].notna() &
        (dfx["exit_time"].dt.date == dfx["signal_time_ist"].dt.date)
    )

    non_intraday = dfx[~dfx["is_intraday_trade"]].copy()

    non_intraday_cols = [
        "itr","signal_time_ist","signal_date","ticker","signal_side","strategy",
        "entry_price","exit_time","exit_price",
        HIT_COL, INTRADAY_HIT_COL, TIME_TD_COL, TIME_CAL_COL,
        "pnl_pct","pnl_rs",
        "taken",
        "max_favorable_pct","max_adverse_pct",
    ]
    non_intraday_cols = [c for c in non_intraday_cols if c in non_intraday.columns]

    non_intraday_path = os.path.join(
        OUT_DIR,
        f"multi_tf_non_intraday_pnl_5m_{STRATEGY_NAME}_{EVAL_SUFFIX}_{ts}_IST.csv"
    )
    non_intraday[non_intraday_cols].to_csv(non_intraday_path, index=False)
    print(f"\nSaved NON-INTRADAY (no leverage) P&L CSV to: {non_intraday_path}")
    print(f"Non-intraday rows: {len(non_intraday)}")

    # --------- Day-wise summary ---------
    print("\n--- Building day-wise summary (signals, intraday hits, P/L) ---")
    daily_summary = build_daily_summary(out_df_constrained)

    daily_summary_path = os.path.join(
        OUT_DIR,
        f"multi_tf_daily_summary_5m_{STRATEGY_NAME}_{EVAL_SUFFIX}_{ts}_IST.csv"
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
    main()
