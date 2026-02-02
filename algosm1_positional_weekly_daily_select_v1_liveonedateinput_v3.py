# -*- coding: utf-8 -*-
"""
TRADE-DATE positional LONG signal generator (DAILY execution, WEEKLY context),
with REAL trading-day handling (weekends + NSE holidays).

Key behavior (IST):
1) You enter a TRADE DATE (intended execution date).
2) If that date is a non-trading day (weekend / holiday), script auto-shifts it to the NEXT trading day
   (can be disabled with --no-shift-nontrading).
3) "AS-OF" data used to generate signals is always BEFORE the trade date:
     - Daily used: strictly < trade_date 00:00 (=> D-1 trading close)
     - Weekly used: strictly < start_of_week(trade_date) 00:00 (=> previous completed week only)

This avoids lookahead from partial-current-week weekly candles and correctly handles holiday weeks.

Outputs:
- One signal per ticker for the resolved trade date
- Signal types:
    BREAKOUT_FIRST (preferred) or WATCH_CUSP

Holiday handling:
- Best: install exchange_calendars (XNSE) OR provide a holiday file.
- If neither is available, only weekends are treated as non-trading.

Holiday file formats supported (any one):
- CSV with a column named 'date' (YYYY-MM-DD)
- CSV/TXT where each line is a date (YYYY-MM-DD)
- The script will also auto-look for: nse_holidays.csv / holidays_nse.csv in cwd or script dir.
"""

from __future__ import annotations

import os
import glob
import argparse
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Set, Tuple

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
DEFAULT_STRATEGY_KEYS = [
    "custom10", "custom09", "custom08", "custom07", "custom06",
    "custom05", "custom04", "custom03", "custom02", "custom01",
]

# ======================================================================
#                       NSE TRADING CALENDAR
# ======================================================================
def _discover_holidays_file(explicit: Optional[str]) -> Optional[str]:
    """Try user path first; else look in cwd / script dir for common names."""
    if explicit:
        return explicit

    candidates = [
        "nse_holidays.csv",
        "holidays_nse.csv",
        "nse_holidays.txt",
        "holidays_nse.txt",
    ]

    # cwd
    for c in candidates:
        p = os.path.join(os.getcwd(), c)
        if os.path.exists(p):
            return p

    # script dir
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for c in candidates:
            p = os.path.join(script_dir, c)
            if os.path.exists(p):
                return p
    except Exception:
        pass

    return None

def _load_holidays_from_file(path: str) -> Set[date]:
    """
    Parse holiday dates from CSV/TXT.
    Accepts:
      - CSV with column 'date'
      - or a one-date-per-line file
    """
    p = path.strip()
    if not os.path.exists(p):
        return set()

    try:
        if p.lower().endswith(".csv"):
            df = pd.read_csv(p)
            if "date" in df.columns:
                s = df["date"]
            else:
                s = df.iloc[:, 0]  # first column
            dts = pd.to_datetime(s, errors="coerce").dt.date
            return set([d for d in dts if pd.notna(d)])
        else:
            lines = [x.strip() for x in open(p, "r", encoding="utf-8").read().splitlines() if x.strip()]
            dts = pd.to_datetime(lines, errors="coerce")
            out: Set[date] = set()
            for x in dts:
                try:
                    out.add(pd.Timestamp(x).date())
                except Exception:
                    pass
            return out
    except Exception:
        # fallback: strict YYYY-MM-DD
        try:
            lines = [x.strip() for x in open(p, "r", encoding="utf-8").read().splitlines() if x.strip()]
            out: Set[date] = set()
            for ln in lines:
                try:
                    out.add(datetime.strptime(ln, "%Y-%m-%d").date())
                except Exception:
                    pass
            return out
        except Exception:
            return set()

def _try_exchange_calendars() -> Optional[object]:
    """Try to load a proper NSE trading calendar from exchange_calendars."""
    try:
        import exchange_calendars as ecals  # type: ignore
    except Exception:
        return None

    for cal_name in ("XNSE", "NSE"):
        try:
            return ecals.get_calendar(cal_name)
        except Exception:
            continue
    return None

def _build_trading_day_helpers(holidays_file: Optional[str]) -> Tuple[str, Set[date], Optional[object]]:
    """
    Returns:
      (calendar_mode, holidays_set, exch_cal_obj)
    calendar_mode in {"exchange_calendars", "file:<path>", "weekend_only"}
    """
    cal = _try_exchange_calendars()
    if cal is not None:
        return ("exchange_calendars", set(), cal)

    hf = _discover_holidays_file(holidays_file)
    if hf:
        hs = _load_holidays_from_file(hf)
        if hs:
            return (f"file:{hf}", hs, None)

    return ("weekend_only", set(), None)

def _is_trading_day(d: date, holidays_set: Set[date], cal_obj: Optional[object]) -> bool:
    if cal_obj is not None:
        try:
            ts = pd.Timestamp(d.isoformat())
            return bool(cal_obj.is_session(ts))
        except Exception:
            pass

    if d.weekday() >= 5:
        return False
    if d in holidays_set:
        return False
    return True

def _next_trading_day(d: date, holidays_set: Set[date], cal_obj: Optional[object]) -> date:
    if cal_obj is not None:
        try:
            ts = pd.Timestamp(d.isoformat())
            if bool(cal_obj.is_session(ts)):
                return d
            nxt = cal_obj.next_session(ts)
            return pd.Timestamp(nxt).date()
        except Exception:
            pass

    cur = d
    for _ in range(30):  # enough for long stretches
        if _is_trading_day(cur, holidays_set, None):
            return cur
        cur = cur + timedelta(days=1)
    return cur

def _start_of_week_monday(d: date) -> date:
    """Calendar Monday of the week containing d."""
    return d - timedelta(days=d.weekday())

# ======================================================================
#                         DATA IO HELPERS
# ======================================================================
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

def _find_file(directory: str, ticker: str) -> Optional[str]:
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None

def _apply_asof_cut(df: pd.DataFrame, cutoff_dt_ist: datetime) -> pd.DataFrame:
    """Keep rows strictly before cutoff_dt_ist (IST)."""
    if df.empty or "date" not in df.columns:
        return df
    return df[df["date"] < cutoff_dt_ist].copy()

def _pick_first_available(series_list: List[pd.Series]) -> pd.Series:
    """Pick first series that yields a finite value at the last index; else return first."""
    for s in series_list:
        try:
            v = s.iloc[-1]
            if np.isfinite(v):
                return s
        except Exception:
            continue
    return series_list[0]

def _first_present_col(cols: set[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

def _series(df: pd.DataFrame, cols: set[str], candidates: List[str]) -> Optional[pd.Series]:
    c = _first_present_col(cols, candidates)
    if not c:
        return None
    return pd.to_numeric(df[c], errors="coerce")

def _get_bbwidth_series(df: pd.DataFrame, cols: set[str]) -> Optional[pd.Series]:
    return _series(df, cols, ["BB_Width_D", "BandWidth_D"])

def _get_ma20_series(df: pd.DataFrame, cols: set[str]) -> Optional[pd.Series]:
    return _series(df, cols, ["EMA_20_D", "20_SMA_D"])

def _get_macd_hist_series(df: pd.DataFrame, cols: set[str]) -> Optional[pd.Series]:
    return _series(df, cols, ["MACD_Hist_D", "Histogram_D", "MACD_Histogram_D"])

# ======================================================================
#                 MTF LOADER (DAILY + WEEKLY) AS-OF
# ======================================================================
def _load_mtf_context_daily_weekly_asof(
    ticker: str,
    daily_cutoff_ist: datetime,
    weekly_cutoff_ist: datetime,
) -> pd.DataFrame:
    """
    Returns a DAILY-indexed dataframe with WEEKLY columns merged in (as-of).

    Cuts:
      - Daily:  < daily_cutoff_ist  (D-1 for trade date)
      - Weekly: < weekly_cutoff_ist (previous completed week only; start_of_week(trade_date) 00:00)
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
    df_d = _apply_asof_cut(df_d, daily_cutoff_ist)
    df_w = _apply_asof_cut(df_w, weekly_cutoff_ist)
    if df_d.empty or df_w.empty:
        return pd.DataFrame()

    # Daily prev-day change (D-1)
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = pd.to_numeric(df_d["Daily_Change"], errors="coerce").shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    df_dw = pd.merge_asof(
        df_d.sort_values("date"),
        df_w.sort_values("date"),
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
    trade_start_ist: datetime,     # resolved trade date 00:00 IST
    weekly_cutoff_ist: datetime,   # start_of_week(trade_date) 00:00 IST
    mode: str,                     # "breakout", "watch", "both"
) -> Optional[dict]:
    df = _load_mtf_context_daily_weekly_asof(
        ticker=ticker,
        daily_cutoff_ist=trade_start_ist,
        weekly_cutoff_ist=weekly_cutoff_ist,
    )
    if df.empty:
        return None

    cols = set(df.columns)
    need = {"date", "open_D", "high_D", "low_D", "close_D"}
    if not need.issubset(cols):
        return None

    r = df.iloc[-1].copy()

    def _get(name: str, default=np.nan):
        return r[name] if name in cols else default

    close = float(_get("close_D", np.nan))
    open_ = float(_get("open_D", np.nan))
    high  = float(_get("high_D", np.nan))

    if (not np.isfinite(close)) or close <= 0:
        return None

    # ------------------- TRADABILITY -------------------
    if P.get("min_price") is not None and close < float(P["min_price"]):
        return None

    # ------------------- WEEKLY FILTERS -------------------
    if P.get("wk_close_above_ema200", False) and {"close_W", "EMA_200_W"}.issubset(cols):
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

    # ------------------- TREND ALIGNMENT + ANTI-CHASE -------------------
    ma20_s = _get_ma20_series(df, cols)
    if P.get("require_close_above_ma20", True) and (ma20_s is not None):
        ma20 = float(ma20_s.iloc[-1])
        if not (np.isfinite(ma20) and close >= ma20):
            return None
        anti_mult = float(P.get("anti_chase_ma20_mult", 1.06))
        if not (close <= ma20 * anti_mult):
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

        if P.get("d_adx_rising", False):
            look = int(P.get("d_adx_rising_lookback", 3))
            if len(adx_s.dropna()) >= look + 1:
                prev_mean = float(adx_s.iloc[-(look+1):-1].mean())
                if np.isfinite(prev_mean) and not (adx >= prev_mean + float(P.get("d_adx_rising_min", 0.0))):
                    return None

    # ------------------- MOMENTUM BOOSTERS -------------------
    boosters_required = int(P.get("boosters_required", 2))
    boosters = 0

    # RSI booster
    if "RSI_D" in cols:
        rsi_s = pd.to_numeric(df["RSI_D"], errors="coerce")
        rsi = float(rsi_s.iloc[-1])
        rsi_prev = float(rsi_s.shift(1).iloc[-1]) if len(rsi_s) >= 2 else np.nan
        rsi_min = float(P.get("rsi_min", 55.0))
        rsi_delta_min = float(P.get("rsi_delta_min", 1.0))
        if np.isfinite(rsi) and (rsi >= rsi_min or (np.isfinite(rsi_prev) and rsi >= rsi_prev + rsi_delta_min)):
            boosters += 1

    # MACD Hist booster
    m_s = _get_macd_hist_series(df, cols)
    if m_s is not None:
        macd = float(m_s.iloc[-1])
        macd_prev = float(m_s.shift(1).iloc[-1]) if len(m_s) >= 2 else np.nan
        macd_hist_min = float(P.get("macd_hist_min", 0.0))
        macd_hist_mult = float(P.get("macd_hist_mult", 1.00))
        if np.isfinite(macd) and np.isfinite(macd_prev) and (macd >= macd_hist_min) and (macd >= macd_prev * macd_hist_mult):
            boosters += 1

    # Volume booster
    if "volume_D" in cols:
        vol_s = pd.to_numeric(df["volume_D"], errors="coerce")
        vol = float(vol_s.iloc[-1])
        vol_mult = float(P.get("vol_mult", 1.25))
        vol_mean_20 = float(vol_s.rolling(20, min_periods=20).mean().iloc[-1])
        if np.isfinite(vol) and np.isfinite(vol_mean_20) and (vol >= vol_mean_20 * vol_mult):
            boosters += 1

    # Day change booster
    if "Daily_Change_D" in cols:
        chg = float(pd.to_numeric(df["Daily_Change_D"], errors="coerce").iloc[-1])
        day_change_min = float(P.get("day_change_min", 0.9))
        if np.isfinite(chg) and chg >= day_change_min:
            boosters += 1

    boosters_required_watch = int(P.get("watch_boosters_required", max(1, boosters_required - 1)))

    # ------------------- BREAKOUT LEVELS (NO LOOKAHEAD) -------------------
    high_s = pd.to_numeric(df["high_D"], errors="coerce")
    close_s = pd.to_numeric(df["close_D"], errors="coerce")

    breakout_windows = [int(w) for w in P.get("breakout_windows", [20, 50, 100]) if int(w) >= 5]

    level_series_list = [high_s.rolling(w, min_periods=w).max().shift(1) for w in breakout_windows]
    level_s = _pick_first_available(level_series_list)

    breakout_level = float(level_s.iloc[-1])
    if (not np.isfinite(breakout_level)) or breakout_level <= 0:
        return None

    close_mult = float(P.get("breakout_close_mult", 1.0012))
    wick_mult_hi = float(P.get("breakout_wick_mult_hi", 1.0008))
    wick_mult_close = float(P.get("breakout_wick_mult_close", 1.0004))

    breakout_close_flag = close_s >= (level_s * close_mult)
    breakout_wick_flag  = (high_s >= (level_s * wick_mult_hi)) & (close_s >= (level_s * wick_mult_close))
    breakout_flag = (breakout_close_flag | breakout_wick_flag).fillna(False)

    is_breakout_today = bool(breakout_flag.iloc[-1])

    no_repeat_days = int(P.get("no_repeat_breakout_days", 5))
    recent_window = breakout_flag.iloc[-(no_repeat_days+1):-1] if len(breakout_flag) >= (no_repeat_days+1) else breakout_flag.iloc[:-1]
    broke_recently = bool(recent_window.any()) if len(recent_window) else False

    # ATR for extension/SL
    atr = np.nan
    if "ATR_D" in cols:
        atr = float(pd.to_numeric(df["ATR_D"], errors="coerce").iloc[-1])

    # ------------------- (A) BREAKOUT SIGNAL -------------------
    if mode in ("breakout", "both") and is_breakout_today:
        if broke_recently:
            return None

        if P.get("require_green_candle", True) and not (close > open_):
            return None

        body_min_pct = float(P.get("body_min_pct", 0.0030))
        if P.get("require_body_min", True) and not ((close - open_) >= (open_ * body_min_pct)):
            return None

        if boosters < boosters_required:
            return None

        max_ext_pct = float(P.get("max_extension_pct", 0.020))
        ext_pct = (close / breakout_level) - 1.0
        if ext_pct > max_ext_pct:
            return None

        if np.isfinite(atr):
            max_ext_atr_mult = float(P.get("max_extension_atr_mult", 0.75))
            if (close - breakout_level) > (atr * max_ext_atr_mult):
                return None

        entry_buffer_pct = float(P.get("entry_buffer_pct", 0.0010))
        sl_buffer_pct = float(P.get("sl_buffer_pct", 0.0080))
        entry_stop = breakout_level * (1.0 + entry_buffer_pct)
        sl_level = breakout_level * (1.0 - sl_buffer_pct)

        if np.isfinite(atr):
            sl_atr_mult = float(P.get("sl_atr_mult", 1.2))
            sl_level_atr = close - (atr * sl_atr_mult)
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
            "boosters": int(boosters),
        }

    # ------------------- (B) WATCH / CUSP SIGNAL -------------------
    if mode in ("watch", "both") and (not is_breakout_today):
        if broke_recently:
            return None

        dist_pct = ((breakout_level - close) / breakout_level) * 100.0
        watch_min_dist = float(P.get("watch_min_dist_pct", 0.00))
        watch_max_dist = float(P.get("watch_max_dist_pct", 1.25))

        if not (dist_pct >= watch_min_dist and dist_pct <= watch_max_dist):
            return None

        touch_buf_pct = float(P.get("watch_touch_buffer_pct", 0.35))
        if not (np.isfinite(high) and high >= breakout_level * (1.0 - (touch_buf_pct / 100.0))):
            return None

        if P.get("watch_require_non_red", True) and not (close >= open_):
            return None

        watch_body_max = float(P.get("watch_body_max_pct", 0.0150))
        body_pct = (close - open_) / max(open_, 1e-9)
        if body_pct > watch_body_max:
            return None

        if boosters < boosters_required_watch:
            return None

        if P.get("require_bb_squeeze", False):
            bbw = _get_bbwidth_series(df, cols)
            if bbw is not None and len(bbw.dropna()) >= 30:
                look = int(P.get("bb_squeeze_lookback", 120))
                q = float(P.get("bb_squeeze_quantile", 0.35))
                tail = bbw.dropna().iloc[-look:] if len(bbw.dropna()) > look else bbw.dropna()
                thr = float(tail.quantile(q))
                if np.isfinite(thr) and not (float(bbw.iloc[-1]) <= thr):
                    return None

        entry_buffer_pct = float(P.get("entry_buffer_pct", 0.0010))
        sl_buffer_pct = float(P.get("sl_buffer_pct", 0.0080))
        entry_stop = breakout_level * (1.0 + entry_buffer_pct)
        sl_level = breakout_level * (1.0 - sl_buffer_pct)

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
            "boosters": int(boosters),
        }

    return None


def make_strategies_more_aggressive(params: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for k, P in params.items():
        Q = dict(P)

        # --- core aggressiveness knobs ---
        # 1) Require fewer boosters (but never below 1)
        if "boosters_required" in Q and Q["boosters_required"] is not None:
            Q["boosters_required"] = max(1, int(Q["boosters_required"]) - 1)

        # 2) Slightly easier momentum gates
        if "rsi_min" in Q and Q["rsi_min"] is not None:
            Q["rsi_min"] = float(Q["rsi_min"]) - 2.0
        if "vol_mult" in Q and Q["vol_mult"] is not None:
            Q["vol_mult"] = max(1.05, float(Q["vol_mult"]) - 0.10)
        if "day_change_min" in Q and Q["day_change_min"] is not None:
            Q["day_change_min"] = float(Q["day_change_min"]) - 0.20

        # 3) Easier breakout confirmation + smaller body requirement
        if "breakout_close_mult" in Q and Q["breakout_close_mult"] is not None:
            Q["breakout_close_mult"] = max(1.0006, float(Q["breakout_close_mult"]) - 0.0002)
        if "body_min_pct" in Q and Q["body_min_pct"] is not None:
            Q["body_min_pct"] = max(0.0020, float(Q["body_min_pct"]) - 0.0005)

        # 4) Relax anti-chase (allow slightly more extension vs MA20/EMA20)
        # NOTE: your code reads anti_chase_ma20_mult but params use anti_chase_ema20_mult.
        # We set BOTH to be safe.
        base_anti = None
        if "anti_chase_ema20_mult" in Q and Q["anti_chase_ema20_mult"] is not None:
            base_anti = float(Q["anti_chase_ema20_mult"]) + 0.01
            Q["anti_chase_ema20_mult"] = base_anti
        if "anti_chase_ma20_mult" in Q and Q["anti_chase_ma20_mult"] is not None:
            Q["anti_chase_ma20_mult"] = float(Q["anti_chase_ma20_mult"]) + 0.01
        elif base_anti is not None:
            Q["anti_chase_ma20_mult"] = base_anti  # align with what the code actually reads

        # 5) Allow a bit more post-breakout extension (more signals)
        Q.setdefault("max_extension_pct", 0.028)         # default in code is 0.020
        Q.setdefault("max_extension_atr_mult", 0.95)     # default in code is 0.75

        # 6) Allow repeats sooner
        Q.setdefault("no_repeat_breakout_days", 3)       # default in code is 5

        # 7) Watchlist slightly easier too (optional but usually desired)
        Q.setdefault("watch_max_dist_pct", 1.60)         # default 1.25
        Q.setdefault("watch_touch_buffer_pct", 0.45)     # default 0.35
        # If watch_boosters_required exists, relax it too
        if "watch_boosters_required" in Q and Q["watch_boosters_required"] is not None:
            Q["watch_boosters_required"] = max(1, int(Q["watch_boosters_required"]) - 1)

        out[k] = Q
    return out


# ======================================================================
#                     10 POSITIONAL STRATEGIES
# ======================================================================
# ======================================================================
#                     10 POSITIONAL STRATEGIES
# ======================================================================

STRATEGY_PARAMS: dict[str, dict] = {
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

STRATEGY_PARAMS = make_strategies_more_aggressive(STRATEGY_PARAMS)

# ======================================================================
#                               CLI
# ======================================================================
def _parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--trade-date", dest="trade_date", default=None,
                   help="Trade date in YYYY-MM-DD (IST). If omitted, you will be prompted.")
    p.add_argument("--mode", dest="mode", default="both", choices=["both", "breakout", "watch"],
                   help="Generate breakout signals, watchlist signals, or both.")
    p.add_argument("--strategies", dest="strategies", default=None,
                   help="Comma-separated keys (e.g., custom10,custom06). If omitted, uses default order.")
    p.add_argument("--max-tickers", dest="max_tickers", type=int, default=None,
                   help="Optional: process only first N tickers (for quick testing).")
    p.add_argument("--holidays-file", dest="holidays_file", default=None,
                   help="Path to NSE holiday file (CSV/TXT). If omitted, tries nse_holidays.csv in cwd/script dir.")
    p.add_argument("--no-shift-nontrading", dest="shift_nontrading", action="store_false",
                   help="Do NOT shift an input trade-date that falls on a holiday/weekend.")
    p.set_defaults(shift_nontrading=True)
    return p.parse_args()

def _resolve_strategy_keys(args_strats: Optional[str]) -> List[str]:
    if not STRATEGY_PARAMS:
        raise RuntimeError("STRATEGY_PARAMS is empty. Add strategies (custom01..custom10) before running.")

    if not args_strats:
        keys = [k for k in DEFAULT_STRATEGY_KEYS if k in STRATEGY_PARAMS]
        if not keys:
            raise RuntimeError(f"No default strategies exist. Available keys: {sorted(STRATEGY_PARAMS.keys())}")
        return keys

    lowered = [a.strip().lower() for a in args_strats.split(",") if a.strip()]
    if len(lowered) == 1 and lowered[0] == "all":
        return sorted(STRATEGY_PARAMS.keys())

    selected = [a for a in lowered if a in STRATEGY_PARAMS]
    missing = [a for a in lowered if a not in STRATEGY_PARAMS]
    if missing:
        raise RuntimeError(f"Unknown strategies: {missing}. Available: {sorted(STRATEGY_PARAMS.keys())}")
    if not selected:
        raise RuntimeError(f"No valid strategies selected. Available: {sorted(STRATEGY_PARAMS.keys())}")
    return selected

def _prompt_trade_date_ist() -> str:
    while True:
        raw = input("Enter TRADE DATE (IST) in YYYY-MM-DD (press ENTER for today): ").strip()
        if raw == "":
            return datetime.now(IST).strftime("%Y-%m-%d")
        try:
            _ = pd.to_datetime(raw, format="%Y-%m-%d", errors="raise").date()
            return raw
        except Exception:
            print("Invalid date. Please enter in YYYY-MM-DD format (example: 2025-12-29).")

def _resolve_trade_start_ist(
    trade_date_str: Optional[str],
    holidays_set: Set[date],
    cal_obj: Optional[object],
    shift_nontrading: bool,
) -> Tuple[datetime, Optional[date]]:
    if not trade_date_str:
        trade_date_str = _prompt_trade_date_ist()

    d = pd.to_datetime(trade_date_str, format="%Y-%m-%d", errors="raise").date()
    original = d

    if shift_nontrading and not _is_trading_day(d, holidays_set, cal_obj):
        d = _next_trading_day(d, holidays_set, cal_obj)

    trade_start = IST.localize(datetime(d.year, d.month, d.day, 0, 0, 0))
    changed = original if original != d else None
    return trade_start, changed

# ======================================================================
#                               MAIN
# ======================================================================
def main():
    args = _parse_args()

    cal_mode, holidays_set, cal_obj = _build_trading_day_helpers(args.holidays_file)

    trade_start_ist, changed_from = _resolve_trade_start_ist(
        trade_date_str=args.trade_date,
        holidays_set=holidays_set,
        cal_obj=cal_obj,
        shift_nontrading=args.shift_nontrading,
    )
    trade_day = trade_start_ist.date()
    now_ist = datetime.now(IST)

    week_start = _start_of_week_monday(trade_day)
    weekly_cutoff_ist = IST.localize(datetime(week_start.year, week_start.month, week_start.day, 0, 0, 0))

    strategy_keys = _resolve_strategy_keys(args.strategies)
    strategy_label = "+".join(strategy_keys)
    mode = args.mode.lower()

    print(f"\nRUN TIME (IST):     {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if changed_from is not None:
        print(f"INPUT TRADE DATE:   {changed_from}  (NON-TRADING)")
        print(f"RESOLVED TRADE DATE:{trade_day}  (NEXT TRADING DAY)")
    else:
        print(f"TRADE DATE (IST):   {trade_day}")

    print(f"CALENDAR MODE:      {cal_mode}")
    if cal_mode == "weekend_only":
        print("! Warning: No NSE holiday source detected (only weekends treated as non-trading).")
        print("  To fully handle NSE holidays, either provide --holidays-file or install exchange_calendars.")

    print(f"AS-OF DAILY CUT:    use data strictly before {trade_start_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} (D-1)")
    print(f"AS-OF WEEKLY CUT:   use data strictly before {weekly_cutoff_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} (prev completed week)")
    print(f"MODE:               {mode}")
    print(f"STRATEGIES:         {strategy_label}\n")

    tickers_d = _list_tickers_from_dir(DIR_D)
    tickers_w = _list_tickers_from_dir(DIR_W)
    tickers = sorted(tickers_d & tickers_w)
    if not tickers:
        print("No common tickers found across Daily and Weekly directories.")
        return

    if args.max_tickers is not None:
        tickers = tickers[: max(1, int(args.max_tickers))]

    print(f"Found {len(tickers)} tickers with both Daily+Weekly.\n")

    rows: List[dict] = []
    for i, ticker in enumerate(tickers, start=1):
        best_breakout = None
        best_watch = None

        for key in strategy_keys:
            P = STRATEGY_PARAMS.get(key.lower())
            if not P:
                continue

            sig = _build_trade_date_signal_for_ticker(
                ticker=ticker,
                P=P,
                trade_start_ist=trade_start_ist,
                weekly_cutoff_ist=weekly_cutoff_ist,
                mode=mode,
            )
            if sig is None:
                continue

            if sig.get("signal_type") == "BREAKOUT_FIRST":
                best_breakout = sig
                break
            if best_watch is None and sig.get("signal_type") == "WATCH_CUSP":
                best_watch = sig

        if best_breakout is not None:
            rows.append(best_breakout)
        elif best_watch is not None:
            rows.append(best_watch)

        if i % 200 == 0:
            print(f"[{i}/{len(tickers)}] processed... signals={len(rows)}")

    if not rows:
        print("No signals generated.")
        return

    out_df = pd.DataFrame(rows)
    out_df["signal_time_ist"] = pd.to_datetime(out_df["signal_time_ist"])
    out_df["signal_date"] = out_df["signal_time_ist"].dt.date

    out_df = out_df[out_df["signal_date"] == trade_day].copy()

    out_df["_prio"] = out_df["signal_type"].map({"BREAKOUT_FIRST": 0, "WATCH_CUSP": 1}).fillna(9)
    out_df = out_df.sort_values(["ticker", "_prio", "strategy"]).drop_duplicates(
        subset=["ticker", "signal_date"], keep="first"
    )
    out_df = out_df.drop(columns=["_prio"], errors="ignore")

    ymd = trade_start_ist.strftime("%Y%m%d")
    ts = now_ist.strftime("%H%M%S")
    out_path = os.path.join(OUT_DIR, f"signals_{ymd}_{ts}_IST.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved signals: {out_path}")
    print(f"Rows: {len(out_df)}")
    show_cols = [c for c in ["ticker", "signal_type", "strategy", "ref_close", "breakout_level", "planned_entry_stop", "planned_sl", "boosters", "asof_daily_bar"] if c in out_df.columns]
    print(out_df[show_cols].head(25).to_string(index=False))

if __name__ == "__main__":
    main()
