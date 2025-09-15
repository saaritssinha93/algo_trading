# -*- coding: utf-8 -*-
"""
Pullback-first intraday detector:
- Bullish: pullback to VWAP (wick touch) then reclaim; trend up; mid-zone momentum.
- Bearish: pop to VWAP (wick touch) then reject; trend down; mid-zone momentum.
- Uses session VWAP (resets daily IST), EMA20/50 slopes, ADX/RSI/Stoch sanity,
  light volume filter, ATR sanity, and prior-day S/R safety.
- Writes all signals per day, and appends the first signal per ticker to papertrade CSV.

Assumed input columns in *_main_indicators.csv:
  date, open, high, low, close, volume,
  Daily_Change, ADX, RSI, Stoch_%K, Stoch_%D
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

# =============================================================================
# 1) Configuration & logging
# =============================================================================
INDICATORS_DIR = "main_indicators_july_5min"            # input folder with *_main_indicators.csv
SIGNALS_DB     = "generated_signals_historical5minpb1.json"
ENTRIES_DIR    = "main_indicators_history_entries_5minpb1"  # output per-day signal dump
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(ENTRIES_DIR, exist_ok=True)

india_tz = pytz.timezone("Asia/Kolkata")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = TimedRotatingFileHandler(
    "logs\\signal_pullback_5min1.log", when="M", interval=30, backupCount=5, delay=True
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

print("Script start...")

# Optional: set working directory if needed
try:
    os.chdir("C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo")
except Exception as e:
    logger.error(f"Error changing directory: {e}")


# =============================================================================
# 2) Utilities: signal DB, time normalization, CSV I/O
# =============================================================================
def load_generated_signals() -> set:
    """Load set of previously emitted Signal_IDs to avoid duplicates."""
    if os.path.exists(SIGNALS_DB):
        try:
            with open(SIGNALS_DB, "r") as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logging.error("Signals DB JSON corrupted; starting empty.")
    return set()


def save_generated_signals(ids: set) -> None:
    with open(SIGNALS_DB, "w") as f:
        json.dump(list(ids), f, indent=2)


def normalize_time(df: pd.DataFrame, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Ensure 'date' is tz-aware in IST and sorted ascending."""
    d = df.copy()
    if "date" not in d.columns:
        raise KeyError("DataFrame missing 'date' column.")

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d.dropna(subset=["date"], inplace=True)

    if d["date"].dt.tz is None:
        d["date"] = d["date"].dt.tz_localize(tz)
    else:
        d["date"] = d["date"].dt.tz_convert(tz)

    d.sort_values("date", inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


def load_and_normalize_csv(file_path: str, expected_cols=None, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Read CSV; normalize 'date' if present; otherwise ensure expected columns."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=expected_cols if expected_cols else [])
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df = normalize_time(df, tz)
    else:
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[expected_cols]
    return df


# =============================================================================
# 3) Indicators: session VWAP (resets daily in IST)
# =============================================================================
def calculate_session_vwap(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vol_col: str = "volume",
    tz: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Compute session VWAP, grouped by IST calendar day."""
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
    d["_cum_vol"]    = d[vol_col].groupby(d["_session"]).cumsum()
    d["VWAP"] = d["_cum_tp_vol"] / d["_cum_vol"]

    d.drop(columns=["_session", "_cum_tp_vol", "_cum_vol"], inplace=True)
    return d


# =============================================================================
# 4) Detection: buy dips / sell pops around VWAP
# =============================================================================
def detect_signals_in_memory(
    ticker: str,
    df_for_rolling: pd.DataFrame,
    df_for_detection: pd.DataFrame,
    existing_signal_ids: set,
) -> list:
    """
    Combined detector — emits up to four independent entry types:
      1) Pullback+VWAPReclaim (Bullish)
      2) Pullback+VWAPReject  (Bearish)
      3) DailyChange+IntraStreak (Bullish)
      4) DailyChange+IntraStreak (Bearish)

    Notes:
    - Uses VWAP, EMA20/EMA50 slopes, ATR14, RSI/Stoch/ADX, rolling volume.
    - Streak logic uses globals INTRA_STREAK_BARS / INTRA_THRESHOLD[_L/_H] / STOCH_TOL if present.
    - Signals are appended independently; a day can get all four for a ticker when conditions match.
    """

    signals = []
    if df_for_detection.empty:
        return signals

    # ---------- Merge & session key ----------
    combined = pd.concat([df_for_rolling, df_for_detection]).drop_duplicates()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined.dropna(subset=["date"], inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # IST day key for intraday grouping
    dt_series = combined["date"]
    try:
        ist = (dt_series.dt.tz_localize("Asia/Kolkata")
               if getattr(dt_series.dt, "tz", None) is None
               else dt_series.dt.tz_convert("Asia/Kolkata"))
    except Exception:
        ist = pd.to_datetime(dt_series, errors="coerce").dt.tz_localize("Asia/Kolkata")
    combined["_date_only"] = ist.dt.date

    # ---------- Ensure VWAP ----------
    if "VWAP" not in combined.columns:
        combined = calculate_session_vwap(combined)

    # ---------- Features used by both blocks ----------
    combined["rolling_vol_10"] = combined["volume"].rolling(10, min_periods=10).mean()
    combined["StochDiff"] = combined["Stoch_%D"].diff()

    # Intra_Change (for streak logic)
    if ("Intra_Change" not in combined.columns) or combined["Intra_Change"].isna().all():
        combined["Intra_Change"] = combined.groupby("_date_only")["close"].pct_change().mul(100.0)
    intra_by_idx = combined["Intra_Change"]

    # EMAs & slopes (for pullback logic)
    if "EMA20" not in combined.columns:
        combined["EMA20"] = combined["close"].ewm(span=20, adjust=False).mean()
    if "EMA50" not in combined.columns:
        combined["EMA50"] = combined["close"].ewm(span=50, adjust=False).mean()
    combined["EMA20_Slope"] = combined["EMA20"].diff()
    combined["EMA50_Slope"] = combined["EMA50"].diff()

    # ATR14 (for pullback safety)
    prev_close = combined["close"].shift(1)
    tr1 = combined["high"] - combined["low"]
    tr2 = (combined["high"] - prev_close).abs()
    tr3 = (combined["low"]  - prev_close).abs()
    combined["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    combined["ATR14"] = combined["TR"].rolling(14, min_periods=14).mean()

    # ---------- Helpers ----------
    def vwap_slope_up(ix, bars=2) -> bool:
        if ix < bars: return False
        seg = combined["VWAP"].iloc[ix - bars : ix + 1]
        return seg.is_monotonic_increasing

    def vwap_slope_down(ix, bars=2) -> bool:
        if ix < bars: return False
        seg = combined["VWAP"].iloc[ix - bars : ix + 1]
        return seg.is_monotonic_decreasing

    def lower_wick_ratio(row) -> float:
        rng = row["high"] - row["low"]
        if rng <= 0: return 0.0
        return (min(row["open"], row["close"]) - row["low"]) / rng

    def upper_wick_ratio(row) -> float:
        rng = row["high"] - row["low"]
        if rng <= 0: return 0.0
        return (row["high"] - max(row["open"], row["close"])) / rng

    def is_midday_ist(ts) -> bool:
        tloc = pd.Timestamp(ts).tz_convert("Asia/Kolkata").time()
        return dt_time(12, 0) <= tloc <= dt_time(13, 15)

    def prev_day_high(df, cdate):
        days = sorted(df["_date_only"].unique())
        y = [d for d in days if d < cdate]
        if not y: return None
        yday = y[-1]
        return float(df.loc[df["_date_only"] == yday, "high"].max())

    def prev_day_low(df, cdate):
        days = sorted(df["_date_only"].unique())
        y = [d for d in days if d < cdate]
        if not y: return None
        yday = y[-1]
        return float(df.loc[df["_date_only"] == yday, "low"].min())

    def find_cidx(ts):
        """Find index of the bar at timestamp ts; fall back to last prior bar in same day."""
        t = pd.Timestamp(ts)
        t = t.tz_localize("Asia/Kolkata") if t.tzinfo is None else t.tz_convert("Asia/Kolkata")
        exact = combined.index[combined["date"] == t]
        if len(exact): return int(exact[0])
        mask = (combined["_date_only"] == t.date()) & (combined["date"] <= t)
        if mask.any(): return int(combined.index[mask][-1])
        mask2 = (combined["_date_only"] == t.date())
        if mask2.any(): return int(combined.index[mask2][0])
        return None

    # --- Streak helpers (from code 2) ---
    def is_adx_increasing(df_, cix, bars_ago=1) -> bool:
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, "ADX"].to_numpy()
        if any(pd.isna(v) for v in vals): return False
        return all(vals[j] < vals[j + 1] for j in range(len(vals) - 1))

    def is_rsi_increasing(df_, cix, bars_ago=1) -> bool:
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, "RSI"].to_numpy()
        if any(pd.isna(v) for v in vals): return False
        return all(vals[j] < vals[j + 1] for j in range(len(vals) - 1))

    def is_rsi_decreasing(df_, cix, bars_ago=1) -> bool:
        si = cix - bars_ago
        if si < 0: return False
        vals = df_.loc[si:cix, "RSI"].to_numpy()
        if any(pd.isna(v) for v in vals): return False
        return all(vals[j] > vals[j + 1] for j in range(len(vals) - 1))

    def has_intra_streak(cidx: int, n: int, thresh, bullish: bool) -> bool:
        """Check the last n bars meet a threshold or band, within the same session."""
        if n <= 0 or cidx is None or cidx - (n - 1) < 0:
            return False
        start = cidx - (n - 1)
        cur_day = combined["_date_only"].iloc[cidx]
        window_days = combined["_date_only"].iloc[start : cidx + 1]
        if window_days.nunique() != 1 or window_days.iloc[0] != cur_day:
            return False
        seg = intra_by_idx.iloc[start : cidx + 1]
        if seg.isna().any():
            return False
        try:
            if isinstance(thresh, (tuple, list)) and len(thresh) == 2:
                lo = float(thresh[0]); hi = float(thresh[1])
                lo, hi = sorted((abs(lo), abs(hi)))
                return ((seg >= lo) & (seg <= hi)).all() if bullish else ((seg <= -lo) & (seg >= -hi)).all()
            else:
                t = abs(float(thresh))
                return (seg > t).all() if bullish else (seg < -t).all()
        except Exception:
            return False

    STOCH_TOL = float(globals().get("STOCH_TOL", 1.0))

    def stoch_bull_cross(df_, idx, tol=STOCH_TOL) -> bool:
        if idx < 1: return False
        k0, d0 = df_.at[idx, "Stoch_%K"], df_.at[idx, "Stoch_%D"]
        k1 = df_.at[idx - 1, "Stoch_%K"]
        if any(pd.isna(x) for x in (k0, d0, k1)): return False
        return (k0 >= d0 - tol) and (k0 > k1)

    def stoch_bear_cross(df_, idx, tol=STOCH_TOL) -> bool:
        if idx < 1: return False
        k0, d0 = df_.at[idx, "Stoch_%K"], df_.at[idx, "Stoch_%D"]
        k1 = df_.at[idx - 1, "Stoch_%K"]
        if any(pd.isna(x) for x in (k0, d0, k1)): return False
        return (k0 <= d0 + tol) and (k0 < k1)

    # ---------- Tunables ----------
    # Pullback (from code 1)
    TOUCH_BUF   = 0.0015
    RECLAIM_BUF = 0.0010
    VOL_MIN_X   = 1.2
    WICK_MIN    = 0.25
    PD_SAFETY   = 0.0030

    # Streak (from code 2)
    DEFAULT_STREAK = int(globals().get("INTRA_STREAK_BARS", 1))
    DEFAULT_THRESH = float(globals().get("INTRA_THRESHOLD", 0.0))

    # ---------- Main pass over today's detection slice ----------
    for _, row_det in df_for_detection.iterrows():
        dt = row_det.get("date", None)
        if pd.isna(dt): 
            continue

        daily_change = row_det.get("Daily_Change", np.nan)
        if pd.isna(daily_change): 
            continue

        # streak bars (N) and threshold/band (if provided)
        n = row_det.get("Intra_Streak_Bars", DEFAULT_STREAK)
        try:
            n = int(n) if pd.notna(n) else DEFAULT_STREAK
        except Exception:
            n = DEFAULT_STREAK

        thresh = None
        tL, tH = row_det.get("Intra_Threshold_L", None), row_det.get("Intra_Threshold_H", None)
        if tL is not None and tH is not None and not (pd.isna(tL) or pd.isna(tH)):
            try: thresh = (float(tL), float(tH))
            except Exception: thresh = None
        if thresh is None:
            gL, gH = globals().get("INTRA_THRESHOLD_L", None), globals().get("INTRA_THRESHOLD_H", None)
            if gL is not None and gH is not None:
                try: thresh = (float(gL), float(gH))
                except Exception: thresh = None
        if thresh is None:
            tv = row_det.get("Intra_Threshold", None)
            try: thresh = float(tv) if (tv is not None and not pd.isna(tv)) else DEFAULT_THRESH
            except Exception: thresh = DEFAULT_THRESH

        cidx = find_cidx(dt)
        if cidx is None: 
            continue

        # snapshot features
        row_c   = combined.loc[cidx]
        price   = row_c["close"]
        vol     = row_c["volume"]
        rvol    = row_c["rolling_vol_10"]
        stoch_d = row_c.get("Stoch_%D", np.nan)
        stoch_k = row_c.get("Stoch_%K", np.nan)
        stoch_diff = row_c.get("StochDiff", np.nan)
        adx_val = row_c.get("ADX", np.nan)
        rsi     = row_c.get("RSI", np.nan)
        vwap    = row_c.get("VWAP", np.nan)
        ema20   = row_c.get("EMA20", np.nan)
        ema50   = row_c.get("EMA50", np.nan)
        atr     = row_c.get("ATR14", np.nan)
        daykey  = row_c["_date_only"]

        # ================================
        # 1) Bullish Pullback (buy near lows)
        # ================================
        bull_pb_ok = False
        if (
            daily_change > 1.0
            and not is_midday_ist(dt)
            and pd.notna(vwap) and pd.notna(rsi) and pd.notna(stoch_k) and pd.notna(stoch_d) and pd.notna(adx_val)
            and vwap_slope_up(cidx, bars=2)
            and (ema20 > ema50) and (row_c["EMA20_Slope"] > 0) and (row_c["EMA50_Slope"] > 0)
            and (40 <= rsi <= 60)
            and (15 <= stoch_k <= 45) and (stoch_d <= 55)
            and (row_c["low"] <= vwap * (1 + TOUCH_BUF))
            and (price >= vwap * (1 + RECLAIM_BUF))
            and (lower_wick_ratio(row_c) >= WICK_MIN)
            and (vol >= VOL_MIN_X * rvol)
            and (25 <= adx_val <= 45)
        ):
            pdh = prev_day_high(combined, daykey)
            near_pdh = (pdh is not None) and (abs(price - pdh) / pdh <= PD_SAFETY)
            atr_bad = (pd.notna(atr) and atr > 0 and (0.02 <= 1.2 * (atr / max(price, 1e-9))))
            if not near_pdh and not atr_bad:
                bull_pb_ok = True

        if bull_pb_ok:
            sig = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_PULLBACK_VWAPRECLAIM"
            if sig not in existing_signal_ids:
                entry_price = round(float(vwap * (1 + RECLAIM_BUF)), 2)
                signals.append({
                    "Ticker": ticker, "date": dt,
                    "Entry Type": "Pullback+VWAPReclaim",
                    "Trend Type": "Bullish",
                    "Price": entry_price,
                    "Daily Change %": round(float(daily_change), 2),
                    "Signal_ID": sig,
                    "logtime": row_det.get("logtime", datetime.now().isoformat()),
                    "Entry Signal": "Yes",
                })
                existing_signal_ids.add(sig)

        # ================================
        # 2) Bearish Pullback (sell near highs)
        # ================================
        bear_pb_ok = False
        if (
            daily_change < -1.0
            and not is_midday_ist(dt)
            and pd.notna(vwap) and pd.notna(rsi) and pd.notna(stoch_k) and pd.notna(stoch_d) and pd.notna(adx_val)
            and vwap_slope_down(cidx, bars=2)
            and (ema20 < ema50) and (row_c["EMA20_Slope"] < 0) and (row_c["EMA50_Slope"] < 0)
            and (45 <= rsi <= 65)
            and (55 <= stoch_k <= 85) and (stoch_d >= 45)
            and (row_c["high"] >= vwap * (1 - TOUCH_BUF))
            and (price <= vwap * (1 - RECLAIM_BUF))
            and (upper_wick_ratio(row_c) >= WICK_MIN)
            and (vol >= VOL_MIN_X * rvol)
            and (25 <= adx_val <= 45)
        ):
            pdl = prev_day_low(combined, daykey)
            near_pdl = (pdl is not None) and (abs(price - pdl) / pdl <= PD_SAFETY)
            atr_bad = (pd.notna(atr) and atr > 0 and (0.02 <= 1.2 * (atr / max(price, 1e-9))))
            if not near_pdl and not atr_bad:
                bear_pb_ok = True

        if bear_pb_ok:
            sig = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_PULLBACK_VWAPREJECT"
            if sig not in existing_signal_ids:
                entry_price = round(float(vwap * (1 - RECLAIM_BUF)), 2)
                signals.append({
                    "Ticker": ticker, "date": dt,
                    "Entry Type": "Pullback+VWAPReject",
                    "Trend Type": "Bearish",
                    "Price": entry_price,
                    "Daily Change %": round(float(daily_change), 2),
                    "Signal_ID": sig,
                    "logtime": row_det.get("logtime", datetime.now().isoformat()),
                    "Entry Signal": "Yes",
                })
                existing_signal_ids.add(sig)

        # ================================
        # 3) Bullish DailyChange + IntraStreak
        # ================================
        bull_streak_ok = False
        if (
            (daily_change > 2)
            and has_intra_streak(cidx, n, thresh, bullish=True)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and (30 < adx_val < 45)
            and is_rsi_increasing(combined, cidx, bars_ago=2)
            and (50 < rsi < 75)
            and (stoch_d > 50)
            and (stoch_diff > 0)
            and stoch_bull_cross(combined, cidx)
            and pd.notna(vwap) and (price > 1.005 * vwap)
            and (vol >= 2.5 * rvol)
        ):
            bull_streak_ok = True

        if bull_streak_ok:
            sig = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BULL_DCHANGE_INTRASTREAK"
            if sig not in existing_signal_ids:
                signals.append({
                    "Ticker": ticker, "date": dt,
                    "Entry Type": "DailyChange+IntraStreak",
                    "Trend Type": "Bullish",
                    "Price": round(float(price), 2),
                    "Daily Change %": round(float(daily_change), 2),
                    "Signal_ID": sig,
                    "logtime": row_det.get("logtime", datetime.now().isoformat()),
                    "Entry Signal": "Yes",
                })
                existing_signal_ids.add(sig)

        # ================================
        # 4) Bearish DailyChange + IntraStreak
        # ================================
        bear_streak_ok = False
        if (
            (daily_change < -2)
            and has_intra_streak(cidx, n, thresh, bullish=False)
            and is_adx_increasing(combined, cidx, bars_ago=1)
            and (30 < adx_val < 45)
            and is_rsi_decreasing(combined, cidx, bars_ago=2)
            and (25 < rsi < 50)
            and (stoch_d < 50)
            and (stoch_diff < 0)
            and stoch_bear_cross(combined, cidx)
            and pd.notna(vwap) and (price < 0.995 * vwap)
            and (vol >= 2.5 * rvol)
        ):
            bear_streak_ok = True

        if bear_streak_ok:
            sig = f"{ticker}-{pd.to_datetime(dt).isoformat()}-BEAR_DCHANGE_INTRASTREAK"
            if sig not in existing_signal_ids:
                signals.append({
                    "Ticker": ticker, "date": dt,
                    "Entry Type": "DailyChange+IntraStreak",
                    "Trend Type": "Bearish",
                    "Price": round(float(price), 2),
                    "Daily Change %": round(float(daily_change), 2),
                    "Signal_ID": sig,
                    "logtime": row_det.get("logtime", datetime.now().isoformat()),
                    "Entry Signal": "Yes",
                })
                existing_signal_ids.add(sig)

    return signals

# =============================================================================
# 5) Mark signals in main CSV => CalcFlag='Yes'
# =============================================================================
def mark_signals_in_main_csv(ticker: str, signals_list: list, main_path: str, tz_obj) -> None:
    """Set 'Entry Signal'='Yes', 'CalcFlag'='Yes' and fill 'logtime' for matched Signal_ID rows."""
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


# =============================================================================
# 6) Detect signals for one trading date across all CSVs
# =============================================================================
def find_price_action_entries_for_date(target_date: datetime.date) -> pd.DataFrame:
    """
    Collect pullback signals for the given date across all *_main_indicators.csv.
    Intraday window considered: 09:25 to 14:04 IST.
    """
    st = india_tz.localize(datetime.combine(target_date, dt_time(9, 25)))
    et = india_tz.localize(datetime.combine(target_date, dt_time(14, 4)))

    pattern = os.path.join(INDICATORS_DIR, "*_main_indicators.csv")
    files = glob.glob(pattern)
    if not files:
        logging.info("No indicator CSV files found.")
        return pd.DataFrame()

    existing_ids = load_generated_signals()
    all_signals = []
    lock = threading.Lock()

    def process_file(file_path: str) -> list:
        ticker = os.path.basename(file_path).replace("_main_indicators.csv", "").upper()
        df_full = load_and_normalize_csv(file_path, tz="Asia/Kolkata")
        if df_full.empty:
            return []

        # slice to the target day
        df_today = df_full[(df_full["date"] >= st) & (df_full["date"] <= et)].copy()

        new_signals = detect_signals_in_memory(
            ticker=ticker,
            df_for_rolling=df_full,
            df_for_detection=df_today,
            existing_signal_ids=existing_ids,
        )
        if new_signals:
            with lock:
                for sig in new_signals:
                    existing_ids.add(sig["Signal_ID"])
        return new_signals or []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {target_date}"):
            try:
                res = fut.result()
                if res:
                    all_signals.extend(res)
            except Exception as e:
                logging.error(f"Error processing file {futures[fut]}: {e}")

    save_generated_signals(existing_ids)

    if not all_signals:
        logging.info(f"No new signals detected for {target_date}.")
        return pd.DataFrame()

    return pd.DataFrame(all_signals).sort_values("date").reset_index(drop=True)


# =============================================================================
# 7) Build papertrade CSV (first signal per ticker)
# =============================================================================
def create_papertrade_df(df_signals: pd.DataFrame, output_file: str, target_date) -> None:
    """
    Append the first signal per ticker for the target day to output_file.
    Adds simple 'Target Price', 'Quantity', 'Total value' if missing.
    """
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

    # simple target/size illustration
    for idx, row in df.iterrows():
        price = float(row.get("Price", 0) or 0)
        trend = row.get("Trend Type", "")
        if price <= 0:
            continue
        target = price * (1.01 if trend == "Bullish" else 0.99 if trend == "Bearish" else 1.0)
        qty = int(20000 / price)
        df.at[idx, "Target Price"] = round(target, 2)
        df.at[idx, "Quantity"] = qty
        df.at[idx, "Total value"] = round(qty * price, 2)

    if "time" not in df.columns:
        df["time"] = datetime.now(india_tz).strftime("%H:%M:%S")
    else:
        blank = (df["time"].isna()) | (df["time"] == "")
        if blank.any():
            df.loc[blank, "time"] = datetime.now(india_tz).strftime("%H:%M:%S")

    lock_path = output_file + ".lock"
    lock = FileLock(lock_path, timeout=10)
    with lock:
        if os.path.exists(output_file):
            existing = normalize_time(pd.read_csv(output_file), tz="Asia/Kolkata")
            existing_keys = set((er["Ticker"], er["date"].date()) for _, er in existing.iterrows())

            new_rows = [row for _, row in df.iterrows()
                        if (row["Ticker"], row["date"].date()) not in existing_keys]

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
            df.drop_duplicates(subset=["Ticker"], keep="first", inplace=True)
            df.sort_values("date", inplace=True)
            df.to_csv(output_file, index=False)
            print(f"[Papertrade] Created => {output_file} with {len(df)} rows.")


# =============================================================================
# 8) Orchestrator for a single date
# =============================================================================
def run_for_date(date_str: str) -> None:
    """
    Run detection + updates for one date (YYYY-MM-DD):
      - Save all signals to ENTRIES_DIR/{date}_entries.csv
      - Mark main CSVs' rows (CalcFlag/Entry Signal)
      - Append first-per-ticker to papertrade CSV
    """
    try:
        target_day = datetime.strptime(date_str, "%Y-%m-%d").date()
        date_label = target_day.strftime("%Y-%m-%d")

        entries_file    = os.path.join(ENTRIES_DIR, f"{date_label}_entries.csv")
        papertrade_file = f"papertrade_5min_pb_{date_label}.csv"

        # Detect
        signals_df = find_price_action_entries_for_date(target_day)
        if signals_df.empty:
            logging.info(f"No signals found for {date_label}.")
            pd.DataFrame().to_csv(entries_file, index=False)
            return

        # Save all signals
        signals_df.to_csv(entries_file, index=False)
        print(f"[ENTRIES] Wrote {len(signals_df)} signals to {entries_file}")

        # Mark CalcFlag in each ticker's main CSV
        for ticker, group in signals_df.groupby("Ticker"):
            main_csv_path = os.path.join(INDICATORS_DIR, f"{ticker}_main_indicators.csv")
            mark_signals_in_main_csv(ticker, group.to_dict("records"), main_csv_path, india_tz)

        # Append to papertrade
        create_papertrade_df(signals_df, papertrade_file, target_day)
        logging.info(f"Papertrade entries written to {papertrade_file}.")
    except Exception as e:
        logging.error(f"Error in run_for_date({date_str}): {e}")
        logging.error(traceback.format_exc())


# =============================================================================
# 9) Graceful shutdown
# =============================================================================
def signal_handler(sig, frame):
    print("Interrupt received, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# 10) Entry point with backfill dates
# =============================================================================
if __name__ == "__main__":
    date_args = sys.argv[1:]
    if not date_args:
        # Backfill calendar (Aug–Dec 2024, skipping known Indian market holidays),
        # then Jan–Jul 2025 (sample list—adjust as needed).
        date_args = [
            # August 2024
            "2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09",
            "2024-08-12","2024-08-13","2024-08-14","2024-08-16",
            "2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23",
            "2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30",

            # September 2024
            "2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06",
            "2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13",
            "2024-09-16","2024-09-17","2024-09-18","2024-09-19","2024-09-20",
            "2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27",
            "2024-09-30",

            # October 2024 (Oct 2 holiday)
            "2024-10-01","2024-10-03","2024-10-04","2024-10-07","2024-10-08",
            "2024-10-09","2024-10-10","2024-10-11","2024-10-14","2024-10-15",
            "2024-10-16","2024-10-17","2024-10-18","2024-10-21","2024-10-22",
            "2024-10-23","2024-10-24","2024-10-25","2024-10-28","2024-10-29",
            "2024-10-30","2024-10-31",

            # November 2024 (Nov 1, Nov 15 holidays)
            "2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08",
            "2024-11-11","2024-11-12","2024-11-13","2024-11-14",
            "2024-11-18","2024-11-19","2024-11-20","2024-11-21","2024-11-22",
            "2024-11-25","2024-11-26","2024-11-27","2024-11-28","2024-11-29",

            # December 2024 (Dec 25 holiday)
            "2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06",
            "2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13",
            "2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20",
            "2024-12-23","2024-12-24","2024-12-26","2024-12-27","2024-12-30",
            "2024-12-31",

            # January 2025
            "2025-01-01","2025-01-02","2025-01-03",
            "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
            "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
            "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
            "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

            # February 2025
            "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
            "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
            "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
            "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

            # March 2025 (Mar 14 holiday)
            "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
            "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
            "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
            "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
            "2025-03-31",

            # April 2025 (Apr 11 holiday)
            "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
            "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
            "2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18",
            "2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25",
            "2025-04-28","2025-04-29","2025-04-30",

            # May 2025
            "2025-05-02",
            "2025-05-05","2025-05-06","2025-05-07","2025-05-08","2025-05-09",
            "2025-05-12","2025-05-13","2025-05-14","2025-05-15","2025-05-16",
            "2025-05-19","2025-05-20","2025-05-21","2025-05-22","2025-05-23",
            "2025-05-26","2025-05-27","2025-05-28","2025-05-29","2025-05-30",

            # June 2025
            "2025-06-02","2025-06-03","2025-06-04","2025-06-05","2025-06-06",
            "2025-06-09","2025-06-10","2025-06-11","2025-06-12","2025-06-13",
            "2025-06-16","2025-06-17","2025-06-18","2025-06-19","2025-06-20",
            "2025-06-23","2025-06-24","2025-06-25","2025-06-26","2025-06-27",
            "2025-06-30",

            # July 2025
            "2025-07-01","2025-07-02","2025-07-03","2025-07-04",
            "2025-07-07","2025-07-08","2025-07-09","2025-07-10"
        ]

    for dstr in date_args:
        print(f"\n=== Processing {dstr} ===")
        run_for_date(dstr)
