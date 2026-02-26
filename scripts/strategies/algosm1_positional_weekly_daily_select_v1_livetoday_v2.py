# -*- coding: utf-8 -*-
"""
TRADE-DATE positional LONG signal generator (DAILY execution, WEEKLY context).

Key idea to avoid "late breakout zone" picks:
  1) FIRST-BREAKOUT ONLY:
        - Signal only on the first breakout day (no "already broke earlier" repeats).
  2) MAX-EXTENSION (ANTI-CHASE):
        - Reject if close is too extended above breakout level.
  3) CUSP / WATCHLIST:
        - If price is just BELOW breakout level (within a small band),
          emit WATCH signal with a stop-buy plan slightly above breakout level.

AS-OF RULES (IST) relative to TRADE DATE:
  - Daily data used: strictly BEFORE trade_date 00:00 IST  (=> up to D-1 close)
  - Weekly context: previous fully completed week only (no same-week lookahead):
        weekly_date_shifted = weekly_date + 1 day
        merge_asof(direction="backward") onto Daily timeline

OUTPUT:
  - Generates signals ONLY for trade_date (in IST)
  - One signal per ticker for trade_date (breakout preferred over watch)
"""

import os
import glob
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals_trade_date"

IST = pytz.timezone("Asia/Kolkata")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------- DEFAULT STRATEGY ORDER -----------------------
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

def _pick_first_available(series_list: list[pd.Series]) -> pd.Series:
    """Pick first series that yields a finite value at the last index."""
    for s in series_list:
        try:
            v = s.iloc[-1]
            if np.isfinite(v):
                return s
        except Exception:
            continue
    return series_list[0]

def _get_bbwidth_series(df: pd.DataFrame, cols: set[str]) -> pd.Series | None:
    """
    Your indicator CSVs sometimes use BandWidth or BB_Width.
    After suffix, they become BandWidth_D / BB_Width_D.
    """
    for c in ("BB_Width_D", "BandWidth_D"):
        if c in cols:
            return pd.to_numeric(df[c], errors="coerce")
    return None

# ======================================================================
#                 MTF LOADER (DAILY + WEEKLY) AS-OF
# ======================================================================
def _load_mtf_context_daily_weekly_asof(ticker: str, trade_start_ist: datetime) -> pd.DataFrame:
    """
    Returns a DAILY-indexed dataframe with WEEKLY columns merged in (as-of),
    using AS-OF CUT relative to trade_start_ist:
      - Daily: < trade_start_ist
      - Weekly: < trade_start_ist
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

    # AS-OF CUTS
    df_d = _apply_asof_cut(df_d, trade_start_ist)
    df_w = _apply_asof_cut(df_w, trade_start_ist)
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
#                 SIGNAL BUILDER (BREAKOUT + CUSP WATCH)
# ======================================================================
def _build_trade_date_signal_for_ticker(
    ticker: str,
    P: dict,
    trade_start_ist: datetime,
    mode: str,  # "breakout", "watch", "both"
) -> dict | None:
    """
    Evaluate on last available daily bar BEFORE trade date (i.e., D-1 bar),
    with weekly context as previous completed week.

    Returns either:
      - BREAKOUT signal (first breakout only, not extended)
      - WATCH signal (cusp: close just below breakout level)
    """
    df = _load_mtf_context_daily_weekly_asof(ticker, trade_start_ist)
    if df.empty:
        return None

    cols = set(df.columns)
    need = {"date", "open_D", "high_D", "low_D", "close_D"}
    if not need.issubset(cols):
        return None

    # last daily bar before trade date (D-1)
    r = df.iloc[-1].copy()

    def _get(name: str, default=np.nan):
        return r[name] if name in cols else default

    close = float(_get("close_D", np.nan))
    open_ = float(_get("open_D", np.nan))
    high  = float(_get("high_D", np.nan))
    low   = float(_get("low_D", np.nan))

    if (not np.isfinite(close)) or close <= 0:
        return None

    # ------------------- TRADABILITY -------------------
    if P.get("min_price") is not None and close < float(P["min_price"]):
        return None

    # ------------------- WEEKLY FILTER -------------------
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

    # ------------------- DAILY PREV FILTER (D-1 shift) -------------------
    dprev = _get("Daily_Change_prev_D", np.nan)
    if P.get("dprev_min") is not None and (not np.isfinite(dprev) or dprev < float(P["dprev_min"])):
        return None
    if P.get("dprev_max") is not None and (np.isfinite(dprev) and dprev > float(P["dprev_max"])):
        return None

    # ------------------- TREND ALIGNMENT + EMA ANTI-CHASE -------------------
    if P.get("require_close_above_ema20", True) and ("EMA_20_D" in cols):
        ema20 = float(pd.to_numeric(df["EMA_20_D"], errors="coerce").iloc[-1])
        if not (np.isfinite(ema20) and close >= ema20):
            return None
        # keep this, but ALSO add breakout-level extension below
        anti_mult = float(P.get("anti_chase_ema20_mult", 1.05))
        if not (close <= ema20 * anti_mult):
            return None

    if P.get("require_close_above_ema50", False) and ("EMA_50_D" in cols):
        ema50 = float(pd.to_numeric(df["EMA_50_D"], errors="coerce").iloc[-1])
        if not (np.isfinite(ema50) and close >= ema50):
            return None

    # ADX min + optional rising ADX
    if P.get("d_adx_min") is not None and ("ADX_D" in cols):
        adx_s = pd.to_numeric(df["ADX_D"], errors="coerce")
        adx = float(adx_s.iloc[-1])
        if not (np.isfinite(adx) and adx >= float(P["d_adx_min"])):
            return None

        if P.get("d_adx_rising", True):
            look = int(P.get("d_adx_rising_lookback", 3))
            if len(adx_s.dropna()) >= look + 1:
                prev_mean = float(adx_s.iloc[-(look+1):-1].mean())
                if np.isfinite(prev_mean) and not (adx >= prev_mean + float(P.get("d_adx_rising_min", 0.0))):
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

    # For WATCH mode, allow slightly fewer boosters (cusp setups are often BEFORE big day-change)
    if mode in ("watch", "both"):
        boosters_required_watch = int(P.get("watch_boosters_required", max(1, boosters_required - 1)))
    else:
        boosters_required_watch = boosters_required

    # ------------------- BREAKOUT LEVELS (NO LOOKAHEAD) -------------------
    # Use rolling max of highs, shifted by 1 (prior level).
    high_s = pd.to_numeric(df["high_D"], errors="coerce")
    close_s = pd.to_numeric(df["close_D"], errors="coerce")

    breakout_windows = P.get("breakout_windows", [20, 50, 100])
    breakout_windows = [int(w) for w in breakout_windows]

    level_series_list = [high_s.rolling(w, min_periods=w).max().shift(1) for w in breakout_windows]
    level_s = _pick_first_available(level_series_list)  # use first window that is available

    breakout_level = float(level_s.iloc[-1])
    if (not np.isfinite(breakout_level)) or breakout_level <= 0:
        return None

    close_mult = float(P.get("breakout_close_mult", 1.0015))
    wick_mult_hi = float(P.get("breakout_wick_mult_hi", 1.0010))
    wick_mult_close = float(P.get("breakout_wick_mult_close", 1.0005))

    # breakout flags across history for FIRST-breakout enforcement
    breakout_close_flag = close_s >= (level_s * close_mult)
    breakout_wick_flag  = (high_s >= (level_s * wick_mult_hi)) & (close_s >= (level_s * wick_mult_close))
    breakout_flag = (breakout_close_flag | breakout_wick_flag).fillna(False)

    is_breakout_today = bool(breakout_flag.iloc[-1])
    was_breakout_yday = bool(breakout_flag.shift(1).iloc[-1]) if len(breakout_flag) >= 2 else False

    # if it already broke recently, skip (prevents late-stage repeat signals)
    no_repeat_days = int(P.get("no_repeat_breakout_days", 5))
    recent_window = breakout_flag.iloc[-(no_repeat_days+1):-1] if len(breakout_flag) >= (no_repeat_days+1) else breakout_flag.iloc[:-1]
    broke_recently = bool(recent_window.any()) if len(recent_window) else False

    # ------------------- ATR for adaptive extension/SL -------------------
    atr = np.nan
    if "ATR_D" in cols:
        atr = float(pd.to_numeric(df["ATR_D"], errors="coerce").iloc[-1])

    # ------------------- (A) BREAKOUT SIGNAL (FIRST ONLY + NOT EXTENDED) -------------------
    if mode in ("breakout", "both") and is_breakout_today:
        # FIRST breakout only:
        # - must NOT have broken recently
        # - and yesterday (previous bar) must NOT have been a breakout (helps prevent multi-day breakout streak picks)
        if broke_recently or was_breakout_yday:
            return None

        # Candle quality (breakout day should show intent)
        if not (close > open_):
            return None
        body_min_pct = float(P.get("body_min_pct", 0.0040))
        if not ((close - open_) >= (open_ * body_min_pct)):
            return None

        # boosters for breakout must meet normal requirement
        if boosters < boosters_required:
            return None

        # Max-extension (ANTI-CHASE) around breakout level
        max_ext_pct = float(P.get("max_extension_pct", 0.015))  # 1.5% default
        ext_pct = (close / breakout_level) - 1.0
        if ext_pct > max_ext_pct:
            return None

        # Optional ATR-based extension cap (more adaptive)
        if np.isfinite(atr):
            max_ext_atr_mult = float(P.get("max_extension_atr_mult", 0.60))
            if (close - breakout_level) > (atr * max_ext_atr_mult):
                return None

        # Plan levels for next session (trade date)
        entry_buffer_pct = float(P.get("entry_buffer_pct", 0.0010))  # 0.10% above level
        sl_buffer_pct = float(P.get("sl_buffer_pct", 0.0080))         # 0.80% below level
        entry_stop = breakout_level * (1.0 + entry_buffer_pct)
        sl_level = breakout_level * (1.0 - sl_buffer_pct)

        # If ATR present, prefer ATR-based SL if tighter logic needed
        if np.isfinite(atr):
            sl_atr_mult = float(P.get("sl_atr_mult", 1.2))
            sl_level_atr = close - (atr * sl_atr_mult)
            # choose the tighter (higher) SL to reduce risk while still reasonable
            sl_level = max(sl_level, sl_level_atr)

        return {
            "signal_time_ist": trade_start_ist,
            "asof_daily_bar": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "signal_type": "BREAKOUT_FIRST",
            "strategy": P.get("name", "CUSTOMXX"),
            "ref_close": float(close),
            "breakout_level": float(breakout_level),
            "extension_pct": float(ext_pct * 100.0),
            "planned_entry_stop": float(entry_stop),
            "planned_sl": float(sl_level),
            "note": "First-breakout only + anti-chase max-extension. Plan: stop-buy slightly above breakout level on trade date.",
        }

    # ------------------- (B) WATCH / CUSP SIGNAL (JUST BELOW BREAKOUT) -------------------
    if mode in ("watch", "both") and (not is_breakout_today):
        # avoid names that already broke recently (late regime)
        if broke_recently:
            return None

        # cusp distance band (close just under breakout level)
        dist_pct = ((breakout_level - close) / breakout_level) * 100.0  # positive when below
        watch_min_dist = float(P.get("watch_min_dist_pct", 0.00))  # can be 0
        watch_max_dist = float(P.get("watch_max_dist_pct", 0.80))  # within 0.8% of level

        if not (dist_pct >= watch_min_dist and dist_pct <= watch_max_dist):
            return None

        # require price to have "touched" near level intraday on D-1 (optional but helps cusp quality)
        touch_buf_pct = float(P.get("watch_touch_buffer_pct", 0.25))  # within 0.25% of level
        if not (np.isfinite(high) and high >= breakout_level * (1.0 - (touch_buf_pct / 100.0))):
            return None

        # for cusp, prefer not a huge candle (avoid already moving hard)
        if not (close >= open_):  # keep it simple: mild green (setup)
            return None
        watch_body_max = float(P.get("watch_body_max_pct", 0.0120))  # max 1.2% body
        body_pct = (close - open_) / max(open_, 1e-9)
        if body_pct > watch_body_max:
            return None

        # boosters for watch can be slightly relaxed
        if boosters < boosters_required_watch:
            return None

        # optional squeeze filter (low BB width) to catch cusp before expansion
        if P.get("require_bb_squeeze", False):
            bbw = _get_bbwidth_series(df, cols)
            if bbw is not None:
                look = int(P.get("bb_squeeze_lookback", 120))
                q = float(P.get("bb_squeeze_quantile", 0.30))  # bottom 30% width
                if len(bbw.dropna()) >= 30:
                    tail = bbw.dropna().iloc[-look:] if len(bbw.dropna()) > look else bbw.dropna()
                    thr = float(tail.quantile(q))
                    if np.isfinite(thr) and not (float(bbw.iloc[-1]) <= thr):
                        return None

        # Plan: stop-buy above breakout level for trade date
        entry_buffer_pct = float(P.get("entry_buffer_pct", 0.0010))  # 0.10% above level
        sl_buffer_pct = float(P.get("sl_buffer_pct", 0.0080))         # 0.80% below level
        entry_stop = breakout_level * (1.0 + entry_buffer_pct)
        sl_level = breakout_level * (1.0 - sl_buffer_pct)

        # If ATR present, SL can be ATR-adjusted
        if np.isfinite(atr):
            sl_atr_mult = float(P.get("sl_atr_mult", 1.0))
            sl_level_atr = breakout_level - (atr * sl_atr_mult)
            sl_level = max(sl_level, sl_level_atr)

        return {
            "signal_time_ist": trade_start_ist,
            "asof_daily_bar": r["date"],
            "ticker": ticker,
            "signal_side": "LONG",
            "signal_type": "WATCH_CUSP",
            "strategy": P.get("name", "CUSTOMXX"),
            "ref_close": float(close),
            "breakout_level": float(breakout_level),
            "distance_to_breakout_pct": float(dist_pct),
            "planned_entry_stop": float(entry_stop),
            "planned_sl": float(sl_level),
            "note": "Cusp watchlist: close is just below breakout level. Plan: stop-buy slightly above breakout level on trade date.",
        }

    return None

# ======================================================================
#                     10 POSITIONAL STRATEGIES
#   (Only added NEW params; your old logic preserved)
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
        watch_boosters_required=1,

        rsi_min=54, rsi_delta_min=0.8,
        macd_hist_min=0.0, macd_hist_mult=1.00,
        vol_mult=1.25,
        day_change_min=0.8,

        anti_chase_ema20_mult=1.07,
        min_price=20.0,

        # NEW anti-late + cusp params
        no_repeat_breakout_days=5,
        max_extension_pct=0.015,
        max_extension_atr_mult=0.60,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.80,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0120,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0080,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=1,

        rsi_min=56, rsi_delta_min=1.0,
        macd_hist_min=0.0, macd_hist_mult=1.005,
        vol_mult=1.30,
        day_change_min=1.0,

        anti_chase_ema20_mult=1.065,
        min_price=25.0,

        # NEW
        no_repeat_breakout_days=5,
        max_extension_pct=0.014,
        max_extension_atr_mult=0.60,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.75,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0115,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0080,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=1,

        rsi_min=57, rsi_delta_min=1.1,
        macd_hist_min=0.0, macd_hist_mult=1.01,
        vol_mult=1.35,
        day_change_min=1.2,

        anti_chase_ema20_mult=1.06,
        min_price=30.0,

        # NEW
        no_repeat_breakout_days=6,
        max_extension_pct=0.013,
        max_extension_atr_mult=0.55,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.70,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0110,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0075,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=2,

        rsi_min=58.5, rsi_delta_min=1.2,
        macd_hist_min=0.0, macd_hist_mult=1.015,
        vol_mult=1.45,
        day_change_min=1.4,

        anti_chase_ema20_mult=1.055,
        min_price=40.0,

        # NEW
        no_repeat_breakout_days=7,
        max_extension_pct=0.012,
        max_extension_atr_mult=0.50,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.65,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0105,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0070,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=2,

        rsi_min=60.0, rsi_delta_min=1.3,
        macd_hist_min=0.0, macd_hist_mult=1.02,
        vol_mult=1.55,
        day_change_min=1.6,

        anti_chase_ema20_mult=1.05,
        d_adx_min=18,
        min_price=50.0,

        # NEW
        no_repeat_breakout_days=7,
        max_extension_pct=0.011,
        max_extension_atr_mult=0.50,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.60,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0100,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0070,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=2,

        rsi_min=60.5, rsi_delta_min=1.4,
        macd_hist_min=0.0, macd_hist_mult=1.02,
        vol_mult=1.60,
        day_change_min=1.7,

        anti_chase_ema20_mult=1.048,
        d_adx_min=19,
        min_price=70.0,
        min_vol_sma20=200000,

        # NEW
        no_repeat_breakout_days=8,
        max_extension_pct=0.010,
        max_extension_atr_mult=0.45,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.55,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0095,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0065,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=2,

        rsi_min=61.5, rsi_delta_min=1.5,
        macd_hist_min=0.0, macd_hist_mult=1.025,
        vol_mult=1.70,
        day_change_min=1.8,

        anti_chase_ema20_mult=1.045,
        d_adx_min=20,
        min_price=80.0,
        min_vol_sma20=300000,

        # NEW
        no_repeat_breakout_days=9,
        max_extension_pct=0.009,
        max_extension_atr_mult=0.45,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.50,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0090,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0060,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=3,

        rsi_min=62.0, rsi_delta_min=1.6,
        macd_hist_min=0.0, macd_hist_mult=1.03,
        vol_mult=1.80,
        day_change_min=2.0,

        anti_chase_ema20_mult=1.042,
        d_adx_min=21,
        min_price=100.0,
        min_vol_sma20=300000,

        # NEW
        no_repeat_breakout_days=10,
        max_extension_pct=0.008,
        max_extension_atr_mult=0.40,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.45,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0085,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0060,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=3,

        rsi_min=63.0, rsi_delta_min=1.7,
        macd_hist_min=0.0, macd_hist_mult=1.035,
        vol_mult=1.90,
        day_change_min=2.2,

        anti_chase_ema20_mult=1.038,
        d_adx_min=22,
        min_price=120.0,
        min_vol_sma20=400000,

        # NEW
        no_repeat_breakout_days=10,
        max_extension_pct=0.008,
        max_extension_atr_mult=0.40,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.45,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0085,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0055,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
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
        watch_boosters_required=3,

        rsi_min=64.0, rsi_delta_min=1.8,
        macd_hist_min=0.0, macd_hist_mult=1.04,
        vol_mult=2.0,
        day_change_min=2.3,

        anti_chase_ema20_mult=1.035,
        d_adx_min=23,
        min_price=150.0,
        min_vol_sma20=500000,

        # NEW (strictest, tightest extension)
        no_repeat_breakout_days=12,
        max_extension_pct=0.007,
        max_extension_atr_mult=0.35,
        watch_min_dist_pct=0.00,
        watch_max_dist_pct=0.40,
        watch_touch_buffer_pct=0.25,
        watch_body_max_pct=0.0080,
        entry_buffer_pct=0.0010,
        sl_buffer_pct=0.0050,
        sl_atr_mult=1.0,
        d_adx_rising=True,
        d_adx_rising_lookback=3,
        d_adx_rising_min=0.0,
        require_bb_squeeze=False,
    ),
}

# ----------------------- CLI -----------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trade-date", dest="trade_date", default=None,
                   help="Trade date in YYYY-MM-DD (IST). If omitted, uses today IST.")
    p.add_argument("--mode", dest="mode", default="both", choices=["both", "breakout", "watch"],
                   help="Generate breakout signals, watchlist signals, or both.")
    p.add_argument("strategies", nargs="*", help="Strategy keys (e.g., custom10 custom09) or 'all'.")
    return p.parse_args()

def _resolve_strategy_keys(args_strats: list[str]) -> list[str]:
    if not args_strats:
        return DEFAULT_STRATEGY_KEYS
    lowered = [a.lower().strip() for a in args_strats if a.strip()]
    if len(lowered) == 1 and lowered[0] == "all":
        return sorted(STRATEGY_PARAMS.keys())
    selected = [a for a in lowered if a in STRATEGY_PARAMS]
    if not selected:
        print("! No valid strategy keys from argv; using defaults.")
        return DEFAULT_STRATEGY_KEYS
    return selected

def _trade_start_from_date_str(trade_date: str | None) -> datetime:
    now_ist = datetime.now(IST)
    if not trade_date:
        d = now_ist.date()
    else:
        d = pd.to_datetime(trade_date).date()
    return IST.localize(datetime(d.year, d.month, d.day, 0, 0, 0))

def _latest_output_if_exists(strategy_label: str, trade_start_ist: datetime, mode: str) -> str | None:
    ymd = trade_start_ist.strftime("%Y%m%d")
    patt = os.path.join(OUT_DIR, f"signals_{mode}_{strategy_label}_{ymd}_*.csv")
    hits = sorted(glob.glob(patt), key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0] if hits else None

# ----------------------- MAIN -----------------------
def main():
    args = _parse_args()
    trade_start_ist = _trade_start_from_date_str(args.trade_date)
    trade_date = trade_start_ist.date()
    now_ist = datetime.now(IST)

    strategy_keys = _resolve_strategy_keys(args.strategies)
    strategy_label = "+".join(strategy_keys)
    mode = args.mode.lower()

    print(f"\nRUN TIME (IST):   {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"TRADE DATE (IST): {trade_date}")
    print(f"AS-OF CUT:        using ONLY data before {trade_start_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} (D-1 daily, prev completed weekly)")
    print(f"MODE:             {mode}")
    print(f"STRATEGIES:       {strategy_label}\n")

    existing = _latest_output_if_exists(strategy_label, trade_start_ist, mode)
    if existing:
        print(f"Found existing output. Reusing: {existing}")
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

    print(f"Found {len(tickers)} tickers with both Daily+Weekly.\n")

    rows = []
    for i, ticker in enumerate(tickers, start=1):
        best_row = None

        # Try strategies in order; first match wins
        for key in strategy_keys:
            P = STRATEGY_PARAMS.get(key.lower())
            if not P:
                continue

            sig = _build_trade_date_signal_for_ticker(
                ticker=ticker,
                P=P,
                trade_start_ist=trade_start_ist,
                mode=mode,
            )
            if sig is not None:
                best_row = sig
                break

        if best_row is not None:
            rows.append(best_row)

        if i % 200 == 0:
            print(f"[{i}/{len(tickers)}] processed... signals={len(rows)}")

    if not rows:
        print("No signals generated.")
        return

    out_df = pd.DataFrame(rows)
    out_df["signal_time_ist"] = pd.to_datetime(out_df["signal_time_ist"])
    out_df["signal_date"] = out_df["signal_time_ist"].dt.date

    # Ensure only TRADE DATE rows
    out_df = out_df[out_df["signal_date"] == trade_date].copy()

    # Prefer BREAKOUT over WATCH if duplicates somehow happen
    if "signal_type" in out_df.columns:
        out_df["_prio"] = out_df["signal_type"].map({"BREAKOUT_FIRST": 0, "WATCH_CUSP": 1}).fillna(9)
        out_df = out_df.sort_values(["ticker", "_prio", "strategy"]).drop_duplicates(subset=["ticker", "signal_date"], keep="first")
        out_df = out_df.drop(columns=["_prio"], errors="ignore")
    else:
        out_df = out_df.sort_values(["ticker", "strategy"]).drop_duplicates(subset=["ticker", "signal_date"], keep="first")

    ymd = trade_start_ist.strftime("%Y%m%d")
    ts = now_ist.strftime("%H%M%S")
    out_path = os.path.join(OUT_DIR, f"signals_{mode}_{strategy_label}_{ymd}_{ts}_IST.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved signals: {out_path}")
    print(f"Rows: {len(out_df)}")
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
