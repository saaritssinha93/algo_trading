# -*- coding: utf-8 -*-
"""
SILVERBEES strategy engine (intraday + positional) using saved indicator CSVs
+ adds ENTRY + EXIT flags for BOTH intraday and positional
+ keeps file EXISTS flags (inputs + macros)
+ prints EXIT events to console (optional)
+ adds EXIT as a WORD in output CSV (INTRA_EXIT_WORD / POS_EXIT_WORD)

WHAT YOU GET IN OUTPUT (per 5m row):
- SIG_INTRA (BUY_ORB/SELL_ORB/BUY_TREND/SELL_TREND/BUY_MR/SELL_MR/HOLD)
- INTRA_ENTRY (1 on the bar where an intraday trade is opened)
- INTRA_EXIT  (1 on the bar where that intraday trade is closed)
- INTRA_EXIT_WORD ("EXIT" on exit bar)
- INTRA_DIR   (+1 long, -1 short, 0 flat)
- INTRA_REASON_ENTRY, INTRA_REASON_EXIT (text labels)
- POS_ENTRY, POS_EXIT, POS_EXIT_WORD, POS_DIR (+1/-1/0) for swing regime
- EXISTS_* flags showing which input/macro files existed when you ran the script

INTRADAY EXIT RULES (default, configurable below):
- Exit on ATR stop-loss OR ATR take-profit
- Exit if VWAP flip against your direction (optional)
- Exit if opposite intraday signal appears (optional)
- ALWAYS exit on last bar of day (EOD square-off)

POSITIONAL EXIT RULES (default, configurable below):
- Exit on opposite swing signal
- Exit when daily EMA trend flips (EMA50_1d < EMA200_1d for longs, opposite for shorts)
- Optional ATR-based stop (using ATR_1d if available, else ATR_15m)

INPUTS:
- etf_indicators_5min/{TICKER}_etf_indicators_5min.csv
- etf_indicators_15min/{TICKER}_etf_indicators_15min.csv
- etf_indicators_1h/{TICKER}_etf_indicators_1h.csv
- etf_indicators_daily/{TICKER}_etf_indicators_daily.csv
- etf_indicators_weekly/{TICKER}_etf_indicators_weekly.csv

OPTIONAL MACROS (date,value):
- macro_inputs/MCX_SILVER_RS_KG.csv
- macro_inputs/MCX_GOLD_RS_10G.csv
- macro_inputs/USDINR.csv
- macro_inputs/DXY.csv
- macro_inputs/US_REAL_YIELD.csv
- macro_inputs/EVENT_RISK.csv

OUTPUT:
- etf_signals/{TICKER}_signals_summary_5m.csv
"""

import os
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)
IST_TZ = pytz.timezone("Asia/Kolkata")

# ---------------- CONFIG ----------------

DIRS = {
    "5min":   "etf_indicators_5min",
    "15min":  "etf_indicators_15min",
    "1h":     "etf_indicators_1h",
    "daily":  "etf_indicators_daily",
    "weekly": "etf_indicators_weekly",
}
OUT_DIR = "etf_signals"
MACRO_DIR = "macro_inputs"

TF_COLS = {
    "15min":  ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "VWAP", "RSI"],
    "1h":     ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "RSI"],
    "daily":  ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "Recent_High", "Recent_Low", "RSI"],
    "weekly": ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "RSI"],
}

# --- Premium / macro gates (looser so signals appear more often) ---
PREMIUM_LONG_MAX = 1.2
PREMIUM_SHORT_MIN = -1.2
PREMIUM_EXTREME = 2.5

MACRO_LONG_MIN = 1.0
MACRO_SHORT_MAX = -1.0

# --- ORB sensitivity ---
OR_BARS = 3
OR_BUFFER_RS = 0.005
ORB_USE_WICK_TOUCH = True
ORB_REQUIRE_VWAP = True

# --- Trend day threshold ---
ADX_TREND_MIN_15M = 17

# --- Macro bias zscore window ---
Z_WIN = 288

# --- Mean reversion sensitivity ---
MR_K = 1.2
USE_RSI_FILTER_FOR_MR = True
MR_RSI_LONG_MAX = 45
MR_RSI_SHORT_MIN = 55
VOL_UNIT_FLOOR = 0.12

# --- MR spam control (but allow more signals) ---
MR_MAX_SIGNALS_PER_DAY = 3
MR_MIN_GAP_BARS = 6

# --- New intraday TREND_PULLBACK signal controls ---
ENABLE_TREND_PULLBACK = True
PULLBACK_ATR_MULT = 0.35
TREND_EMA_FAST_CANDIDATES = ["EMA_20", "EMA_20_15m", "EMA_50", "EMA_50_15m"]
TREND_EMA_SLOW_CANDIDATES = ["EMA_50", "EMA_50_15m", "EMA_200", "EMA_200_15m"]

# ---------------- EXIT CONFIG ----------------

# Intraday exit behavior
INTRA_USE_OPPOSITE_SIGNAL_EXIT = True
INTRA_USE_VWAP_FLIP_EXIT = True
VWAP_EXIT_BUFFER_PCT = 0.00  # 0.00 = flip exactly at vwap; set 0.02 etc if you want buffer

INTRA_SL_ATR_MULT = 0.80     # stop-loss = entry +/- 0.80 * ATR_price
INTRA_TP_ATR_MULT = 1.20     # take-profit = entry +/- 1.20 * ATR_price

# Positional exit behavior
POS_USE_TREND_FLIP_EXIT = True
POS_USE_ATR_STOP = True
POS_SL_ATR_MULT = 2.5

# Console printing
PRINT_EXIT_EVENTS = True  # prints a line whenever an INTRA_EXIT or POS_EXIT occurs

# ---------------- HELPERS ----------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_ist_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(IST_TZ)
    else:
        dt = dt.dt.tz_convert(IST_TZ)
    return dt.dt.tz_localize(None)

def _file_exists(path: Path) -> int:
    return int(path.exists())

def _read_etf_file(ticker: str, tf: str) -> pd.DataFrame:
    fp = Path(DIRS[tf]) / f"{ticker}_etf_indicators_{tf}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    df = pd.read_csv(fp)
    if df.empty or "date" not in df.columns:
        raise ValueError(f"Bad/empty CSV: {fp}")

    df["date"] = _to_ist_naive_datetime(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.name = "ts"
    return df

def _read_macro_series(name: str) -> pd.Series | None:
    fp = Path(MACRO_DIR) / f"{name}.csv"
    if not fp.exists():
        return None

    df = pd.read_csv(fp)
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return None

    df["date"] = _to_ist_naive_datetime(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=df["date"].values, name=name).dropna()
    s = s.sort_index()
    s.index.name = "ts"
    return s

def _merge_asof(base: pd.DataFrame, higher: pd.DataFrame, cols: list[str], suffix: str) -> pd.DataFrame:
    cols = [c for c in cols if c in higher.columns]
    if not cols:
        return base

    b = base.reset_index()
    h = higher.reset_index()[["ts"] + cols]

    out = pd.merge_asof(
        b.sort_values("ts"),
        h.sort_values("ts"),
        on="ts",
        direction="backward"
    ).set_index("ts")

    out.index.name = "ts"
    return out.rename(columns={c: f"{c}_{suffix}" for c in cols})

def _add_series_asof(feat: pd.DataFrame, s: pd.Series, colname: str) -> pd.DataFrame:
    b = feat.reset_index()
    m = s.rename(colname).to_frame().reset_index().rename(columns={"ts": "ts", "index": "ts"})

    b["ts"] = pd.to_datetime(b["ts"], errors="coerce")
    m["ts"] = pd.to_datetime(m["ts"], errors="coerce")

    out = pd.merge_asof(
        b.sort_values("ts"),
        m.sort_values("ts"),
        on="ts",
        direction="backward"
    ).set_index("ts")

    out.index.name = "ts"
    return out

def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def _opening_range(df_5m: pd.DataFrame, bars: int) -> pd.DataFrame:
    df = df_5m.copy()
    day = df.index.date
    df["ORH"] = df.groupby(day)["high"].transform(lambda x: x.iloc[:bars].max())
    df["ORL"] = df.groupby(day)["low"].transform(lambda x: x.iloc[:bars].min())
    return df

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None

def _pick_like(df: pd.DataFrame, contains: str) -> str | None:
    key = contains.lower()
    cols = [c for c in df.columns if key in str(c).lower()]
    for c in cols:
        if df[c].notna().any():
            return c
    return None

def _fill_intraday(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    day = df.index.date
    df[cols] = df.groupby(day)[cols].transform(lambda g: g.ffill().bfill())
    return df

def _limit_signals_per_day(sig_mask: pd.Series, max_per_day: int, min_gap_bars: int) -> pd.Series:
    out = pd.Series(False, index=sig_mask.index)
    day = sig_mask.index.date

    for _, idx in pd.Series(sig_mask.index, index=sig_mask.index).groupby(day):
        day_mask = sig_mask.loc[idx]
        true_pos = np.where(day_mask.values)[0]
        if len(true_pos) == 0:
            continue

        chosen = []
        last = -10**9
        for p in true_pos:
            if p - last >= min_gap_bars:
                chosen.append(p)
                last = p
            if len(chosen) >= max_per_day:
                break

        out.loc[idx[chosen]] = True

    return out

def _is_buy_sig(x: str) -> bool:
    return isinstance(x, str) and x.startswith("BUY")

def _is_sell_sig(x: str) -> bool:
    return isinstance(x, str) and x.startswith("SELL")

# ---------------- EXIT ENGINE ----------------

def _build_intraday_entries_exits(feat: pd.DataFrame, has_vwap: bool, vwap_col: str | None, atr_price: pd.Series) -> pd.DataFrame:
    idx = feat.index
    day = idx.date

    close = pd.to_numeric(feat["close"], errors="coerce").values
    vwap = pd.to_numeric(feat[vwap_col], errors="coerce").values if (has_vwap and vwap_col in feat.columns) else None
    atrp = pd.to_numeric(atr_price, errors="coerce").fillna(0).values

    sig = feat["SIG_INTRA"].astype(str).values

    intra_dir = np.zeros(len(feat), dtype=int)
    entry = np.zeros(len(feat), dtype=int)
    exit_ = np.zeros(len(feat), dtype=int)
    reason_entry = np.array([""] * len(feat), dtype=object)
    reason_exit = np.array([""] * len(feat), dtype=object)

    unique_days = pd.Series(day).unique()
    for d in unique_days:
        mask = (day == d)
        pos = 0
        ep = np.nan
        stop = np.nan
        tp = np.nan

        day_ix = np.where(mask)[0]
        if len(day_ix) == 0:
            continue

        last_bar_i = day_ix[-1]

        for _, i in enumerate(day_ix):
            intra_dir[i] = pos

            # EOD forced exit
            if i == last_bar_i and pos != 0:
                exit_[i] = 1
                reason_exit[i] = "EOD_EXIT"
                pos = 0
                ep = np.nan
                stop = np.nan
                tp = np.nan
                intra_dir[i] = 0
                continue

            # entry
            if pos == 0:
                if _is_buy_sig(sig[i]):
                    pos = 1
                    ep = close[i]
                    a = atrp[i] if atrp[i] > 0 else (close[i] * 0.002)
                    stop = ep - INTRA_SL_ATR_MULT * a
                    tp = ep + INTRA_TP_ATR_MULT * a
                    entry[i] = 1
                    reason_entry[i] = sig[i]
                    intra_dir[i] = 1
                elif _is_sell_sig(sig[i]):
                    pos = -1
                    ep = close[i]
                    a = atrp[i] if atrp[i] > 0 else (close[i] * 0.002)
                    stop = ep + INTRA_SL_ATR_MULT * a
                    tp = ep - INTRA_TP_ATR_MULT * a
                    entry[i] = 1
                    reason_entry[i] = sig[i]
                    intra_dir[i] = -1
                continue

            # exits
            if pos == 1:
                if close[i] <= stop:
                    exit_[i] = 1
                    reason_exit[i] = "SL_ATR"
                    pos = 0
                elif close[i] >= tp:
                    exit_[i] = 1
                    reason_exit[i] = "TP_ATR"
                    pos = 0
                else:
                    if INTRA_USE_VWAP_FLIP_EXIT and has_vwap and vwap is not None and not np.isnan(vwap[i]):
                        buf = VWAP_EXIT_BUFFER_PCT / 100.0
                        if close[i] < vwap[i] * (1 - buf):
                            exit_[i] = 1
                            reason_exit[i] = "VWAP_FLIP"
                            pos = 0
                    if (pos != 0) and INTRA_USE_OPPOSITE_SIGNAL_EXIT and _is_sell_sig(sig[i]):
                        exit_[i] = 1
                        reason_exit[i] = "OPPOSITE_SIG"
                        pos = 0

            elif pos == -1:
                if close[i] >= stop:
                    exit_[i] = 1
                    reason_exit[i] = "SL_ATR"
                    pos = 0
                elif close[i] <= tp:
                    exit_[i] = 1
                    reason_exit[i] = "TP_ATR"
                    pos = 0
                else:
                    if INTRA_USE_VWAP_FLIP_EXIT and has_vwap and vwap is not None and not np.isnan(vwap[i]):
                        buf = VWAP_EXIT_BUFFER_PCT / 100.0
                        if close[i] > vwap[i] * (1 + buf):
                            exit_[i] = 1
                            reason_exit[i] = "VWAP_FLIP"
                            pos = 0
                    if (pos != 0) and INTRA_USE_OPPOSITE_SIGNAL_EXIT and _is_buy_sig(sig[i]):
                        exit_[i] = 1
                        reason_exit[i] = "OPPOSITE_SIG"
                        pos = 0

            intra_dir[i] = pos

            if pos == 0 and exit_[i] == 1:
                ep = np.nan
                stop = np.nan
                tp = np.nan

    feat["INTRA_DIR"] = intra_dir
    feat["INTRA_ENTRY"] = entry
    feat["INTRA_EXIT"] = exit_
    feat["INTRA_REASON_ENTRY"] = reason_entry
    feat["INTRA_REASON_EXIT"] = reason_exit
    return feat

def _build_positional_entries_exits(feat: pd.DataFrame, atr_pos_price: pd.Series) -> pd.DataFrame:
    sigp = feat["SIG_POS"].astype(str).values
    close = pd.to_numeric(feat["close"], errors="coerce").values
    atrp = pd.to_numeric(atr_pos_price, errors="coerce").fillna(0).values

    ema50 = pd.to_numeric(feat["EMA_50_1d"], errors="coerce").values if "EMA_50_1d" in feat.columns else None
    ema200 = pd.to_numeric(feat["EMA_200_1d"], errors="coerce").values if "EMA_200_1d" in feat.columns else None

    pos_dir = np.zeros(len(feat), dtype=int)
    entry = np.zeros(len(feat), dtype=int)
    exit_ = np.zeros(len(feat), dtype=int)
    reason_entry = np.array([""] * len(feat), dtype=object)
    reason_exit = np.array([""] * len(feat), dtype=object)

    pos = 0
    ep = np.nan
    stop = np.nan

    for i in range(len(feat)):
        pos_dir[i] = pos

        want_long = (sigp[i] == "BUY_SWING")
        want_short = (sigp[i] == "SELL_SWING")

        # exits
        if pos == 1:
            if want_short:
                exit_[i] = 1
                reason_exit[i] = "OPPOSITE_SWING"
                pos = 0
            elif POS_USE_TREND_FLIP_EXIT and ema50 is not None and ema200 is not None:
                if (not np.isnan(ema50[i])) and (not np.isnan(ema200[i])) and (ema50[i] < ema200[i]):
                    exit_[i] = 1
                    reason_exit[i] = "EMA_TREND_FLIP"
                    pos = 0
            if pos == 1 and POS_USE_ATR_STOP:
                a = atrp[i] if atrp[i] > 0 else (close[i] * 0.01)
                if np.isnan(stop):
                    stop = ep - POS_SL_ATR_MULT * a
                if close[i] <= stop:
                    exit_[i] = 1
                    reason_exit[i] = "ATR_STOP"
                    pos = 0

        elif pos == -1:
            if want_long:
                exit_[i] = 1
                reason_exit[i] = "OPPOSITE_SWING"
                pos = 0
            elif POS_USE_TREND_FLIP_EXIT and ema50 is not None and ema200 is not None:
                if (not np.isnan(ema50[i])) and (not np.isnan(ema200[i])) and (ema50[i] > ema200[i]):
                    exit_[i] = 1
                    reason_exit[i] = "EMA_TREND_FLIP"
                    pos = 0
            if pos == -1 and POS_USE_ATR_STOP:
                a = atrp[i] if atrp[i] > 0 else (close[i] * 0.01)
                if np.isnan(stop):
                    stop = ep + POS_SL_ATR_MULT * a
                if close[i] >= stop:
                    exit_[i] = 1
                    reason_exit[i] = "ATR_STOP"
                    pos = 0

        if exit_[i] == 1 and pos == 0:
            ep = np.nan
            stop = np.nan

        # entries
        if pos == 0:
            if want_long:
                pos = 1
                ep = close[i]
                entry[i] = 1
                reason_entry[i] = "BUY_SWING"
                stop = np.nan
            elif want_short:
                pos = -1
                ep = close[i]
                entry[i] = 1
                reason_entry[i] = "SELL_SWING"
                stop = np.nan

        pos_dir[i] = pos

    feat["POS_DIR"] = pos_dir
    feat["POS_ENTRY"] = entry
    feat["POS_EXIT"] = exit_
    feat["POS_REASON_ENTRY"] = reason_entry
    feat["POS_REASON_EXIT"] = reason_exit
    return feat

# ---------------- CORE ----------------

@dataclass
class StrategyResult:
    ticker: str
    out_path: str
    last_ts: str
    rows: int

def build_features_and_signals(ticker: str) -> StrategyResult:
    _ensure_dir(OUT_DIR)

    # EXISTS flags (inputs)
    exists_5m = _file_exists(Path(DIRS["5min"]) / f"{ticker}_etf_indicators_5min.csv")
    exists_15m = _file_exists(Path(DIRS["15min"]) / f"{ticker}_etf_indicators_15min.csv")
    exists_1h = _file_exists(Path(DIRS["1h"]) / f"{ticker}_etf_indicators_1h.csv")
    exists_1d = _file_exists(Path(DIRS["daily"]) / f"{ticker}_etf_indicators_daily.csv")
    exists_1w = _file_exists(Path(DIRS["weekly"]) / f"{ticker}_etf_indicators_weekly.csv")

    # load
    df5  = _read_etf_file(ticker, "5min")
    df15 = _read_etf_file(ticker, "15min")
    df1h = _read_etf_file(ticker, "1h")
    dfd  = _read_etf_file(ticker, "daily")
    dfw  = _read_etf_file(ticker, "weekly")

    feat = df5.copy()

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in feat.columns:
            raise ValueError(f"[{ticker}] missing required column in 5m: {c}")

    # merge TFs
    feat = _merge_asof(feat, df15, TF_COLS["15min"], "15m")
    feat = _merge_asof(feat, df1h, TF_COLS["1h"], "1h")
    feat = _merge_asof(feat, dfd,  TF_COLS["daily"], "1d")
    feat = _merge_asof(feat, dfw,  TF_COLS["weekly"], "1w")

    feat = _fill_intraday(feat, [c for c in feat.columns if c.endswith("_15m") or c.endswith("_1h")])

    feat = _opening_range(feat, OR_BARS)

    adx_col = _pick_col(feat, ["ADX_15m"]) or _pick_col(feat, ["ADX", "ADX_1h"])
    if adx_col is None:
        feat["TREND_DAY"] = np.nan
    else:
        feat["TREND_DAY"] = (pd.to_numeric(feat[adx_col], errors="coerce") >= ADX_TREND_MIN_15M).astype(int)

    # macros exists flags
    macro_files = {
        "MCX_SILVER_RS_KG": Path(MACRO_DIR) / "MCX_SILVER_RS_KG.csv",
        "MCX_GOLD_RS_10G":  Path(MACRO_DIR) / "MCX_GOLD_RS_10G.csv",
        "USDINR":           Path(MACRO_DIR) / "USDINR.csv",
        "DXY":              Path(MACRO_DIR) / "DXY.csv",
        "US_REAL_YIELD":    Path(MACRO_DIR) / "US_REAL_YIELD.csv",
        "EVENT_RISK":       Path(MACRO_DIR) / "EVENT_RISK.csv",
    }
    for k, p in macro_files.items():
        feat[f"EXISTS_{k}"] = _file_exists(p)

    s_mcx_silver = _read_macro_series("MCX_SILVER_RS_KG")
    s_mcx_gold   = _read_macro_series("MCX_GOLD_RS_10G")
    s_usdinr     = _read_macro_series("USDINR")
    s_dxy        = _read_macro_series("DXY")
    s_realy      = _read_macro_series("US_REAL_YIELD")
    s_event      = _read_macro_series("EVENT_RISK")

    if s_mcx_silver is not None:
        feat = _add_series_asof(feat, s_mcx_silver, "MCX_SILVER_RS_KG")
        feat["FAIR_SILVERBEES"] = feat["MCX_SILVER_RS_KG"] / 1000.0
        feat["PREMIUM_PCT"] = (feat["close"] - feat["FAIR_SILVERBEES"]) / (feat["FAIR_SILVERBEES"] + 1e-12) * 100.0
    else:
        feat["MCX_SILVER_RS_KG"] = np.nan
        feat["FAIR_SILVERBEES"] = np.nan
        feat["PREMIUM_PCT"] = np.nan

    if (s_mcx_gold is not None) and (s_mcx_silver is not None):
        feat = _add_series_asof(feat, s_mcx_gold, "MCX_GOLD_RS_10G")
        gold_per_g = feat["MCX_GOLD_RS_10G"] / 10.0
        silver_per_g = feat["MCX_SILVER_RS_KG"] / 1000.0
        feat["GSR"] = gold_per_g / (silver_per_g + 1e-12)
    else:
        feat["MCX_GOLD_RS_10G"] = np.nan
        feat["GSR"] = np.nan

    for s, nm in [(s_usdinr, "USDINR"), (s_dxy, "DXY"), (s_realy, "US_REAL_YIELD")]:
        if s is not None:
            feat = _add_series_asof(feat, s, nm)
        else:
            feat[nm] = np.nan

    if s_event is not None:
        feat = _add_series_asof(feat, s_event, "EVENT_RISK")
        feat["EVENT_RISK"] = pd.to_numeric(feat["EVENT_RISK"], errors="coerce").fillna(0).clip(0, 9)
    else:
        feat["EVENT_RISK"] = 0

    feat["HAS_MACRO"] = feat[["USDINR", "DXY", "US_REAL_YIELD"]].notna().any(axis=1)

    feat["MACRO_BIAS"] = 0.0
    if feat["USDINR"].notna().any():
        feat["Z_USDINR"] = _zscore(pd.to_numeric(feat["USDINR"], errors="coerce"), Z_WIN)
        feat["MACRO_BIAS"] += feat["Z_USDINR"].fillna(0)
    if feat["DXY"].notna().any():
        feat["Z_DXY"] = _zscore(pd.to_numeric(feat["DXY"], errors="coerce"), Z_WIN)
        feat["MACRO_BIAS"] += -feat["Z_DXY"].fillna(0)
    if feat["US_REAL_YIELD"].notna().any():
        feat["Z_REALY"] = _zscore(pd.to_numeric(feat["US_REAL_YIELD"], errors="coerce"), Z_WIN)
        feat["MACRO_BIAS"] += -feat["Z_REALY"].fillna(0)

    feat["BIAS_DAMP"] = (1.0 - 0.10 * feat["EVENT_RISK"]).clip(0.5, 1.0)
    feat["MACRO_BIAS"] = (feat["MACRO_BIAS"] * feat["BIAS_DAMP"]).clip(-5, 5)

    feat["DISTORTION_DAY"] = 0
    if feat["PREMIUM_PCT"].notna().any():
        feat["DISTORTION_DAY"] = (feat["PREMIUM_PCT"].abs() >= PREMIUM_EXTREME).astype(int)

    feat["GATE_LONG"] = True
    feat["GATE_SHORT"] = True
    if feat["PREMIUM_PCT"].notna().any():
        feat["GATE_LONG"] = feat["PREMIUM_PCT"] <= PREMIUM_LONG_MAX
        feat["GATE_SHORT"] = feat["PREMIUM_PCT"] >= PREMIUM_SHORT_MIN

    macro_long_ok  = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] >= MACRO_LONG_MIN)
    macro_short_ok = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] <= MACRO_SHORT_MAX)
    feat["GATE_LONG"]  = feat["GATE_LONG"]  & macro_long_ok  & (feat["DISTORTION_DAY"] == 0)
    feat["GATE_SHORT"] = feat["GATE_SHORT"] & macro_short_ok & (feat["DISTORTION_DAY"] == 0)

    vwap_col = _pick_col(feat, ["VWAP", "VWAP_15m"]) or _pick_like(feat, "vwap")
    has_vwap = vwap_col is not None and vwap_col in feat.columns and feat[vwap_col].notna().any()

    if not has_vwap:
        feat["ABOVE_VWAP"] = True
        feat["BELOW_VWAP"] = True
        feat["VWAP_DIST_PCT"] = 0.0
    else:
        vwap = pd.to_numeric(feat[vwap_col], errors="coerce")
        feat["ABOVE_VWAP"] = feat["close"] >= vwap
        feat["BELOW_VWAP"] = feat["close"] <= vwap
        feat["VWAP_DIST_PCT"] = (feat["close"] - vwap) / (vwap + 1e-12) * 100.0

    if ORB_USE_WICK_TOUCH:
        orb_long_break  = (feat["high"] >= (feat["ORH"] + OR_BUFFER_RS))
        orb_short_break = (feat["low"]  <= (feat["ORL"] - OR_BUFFER_RS))
    else:
        orb_long_break  = (feat["close"] > (feat["ORH"] + OR_BUFFER_RS))
        orb_short_break = (feat["close"] < (feat["ORL"] - OR_BUFFER_RS))

    if ORB_REQUIRE_VWAP and has_vwap:
        feat["BREAK_LONG"]  = orb_long_break & feat["ABOVE_VWAP"]
        feat["BREAK_SHORT"] = orb_short_break & feat["BELOW_VWAP"]
    else:
        feat["BREAK_LONG"]  = orb_long_break
        feat["BREAK_SHORT"] = orb_short_break

    atr_col = _pick_col(feat, ["ATR", "ATR_15m"]) or _pick_like(feat, "atr")
    if atr_col is not None and atr_col in feat.columns and feat[atr_col].notna().any():
        atr = pd.to_numeric(feat[atr_col], errors="coerce")
        feat["VOL_UNIT"] = (atr / (feat["close"] + 1e-12)) * 100.0
        atr_price_intra = atr
    else:
        feat["VOL_UNIT"] = feat["VWAP_DIST_PCT"].rolling(50).std()
        atr_price_intra = feat["close"] * (feat["VOL_UNIT"] / 100.0)

    feat["VOL_UNIT"] = pd.to_numeric(feat["VOL_UNIT"], errors="coerce").fillna(VOL_UNIT_FLOOR).clip(lower=VOL_UNIT_FLOOR)

    feat["WEAK_BIAS"] = (~feat["HAS_MACRO"]) | feat["MACRO_BIAS"].between(-1.5, 1.5)

    trend_ok = (feat["TREND_DAY"] == 0) | (feat["TREND_DAY"].isna())
    mr_base = feat["WEAK_BIAS"] & trend_ok & (feat["DISTORTION_DAY"] == 0) & has_vwap

    mr_long  = mr_base & (feat["VWAP_DIST_PCT"] <= -MR_K * feat["VOL_UNIT"])
    mr_short = mr_base & (feat["VWAP_DIST_PCT"] >=  MR_K * feat["VOL_UNIT"])

    rsi_col = _pick_col(feat, ["RSI", "RSI_15m"]) or _pick_like(feat, "rsi")
    if USE_RSI_FILTER_FOR_MR and (rsi_col is not None) and (rsi_col in feat.columns) and feat[rsi_col].notna().any():
        rsi = pd.to_numeric(feat[rsi_col], errors="coerce")
        mr_long  = mr_long  & (rsi <= MR_RSI_LONG_MAX)
        mr_short = mr_short & (rsi >= MR_RSI_SHORT_MIN)

    mr_long_entry  = _limit_signals_per_day(mr_long,  MR_MAX_SIGNALS_PER_DAY, MR_MIN_GAP_BARS)
    mr_short_entry = _limit_signals_per_day(mr_short, MR_MAX_SIGNALS_PER_DAY, MR_MIN_GAP_BARS)

    pb_long = pd.Series(False, index=feat.index)
    pb_short = pd.Series(False, index=feat.index)

    if ENABLE_TREND_PULLBACK and has_vwap:
        ema_fast = _pick_col(feat, TREND_EMA_FAST_CANDIDATES) or _pick_like(feat, "ema_20")
        ema_slow = _pick_col(feat, TREND_EMA_SLOW_CANDIDATES) or _pick_like(feat, "ema_50") or _pick_like(feat, "ema_200")

        if ema_fast is not None and ema_slow is not None and ema_fast in feat.columns and ema_slow in feat.columns:
            ef = pd.to_numeric(feat[ema_fast], errors="coerce")
            es = pd.to_numeric(feat[ema_slow], errors="coerce")

            if atr_col is not None and atr_col in feat.columns and feat[atr_col].notna().any():
                atr_price = pd.to_numeric(feat[atr_col], errors="coerce")
            else:
                atr_price = feat["close"] * (feat["VOL_UNIT"] / 100.0)

            near_fast = (feat["close"] - ef).abs() <= (PULLBACK_ATR_MULT * atr_price)
            trend_up = (ef > es)
            trend_dn = (ef < es)
            td = (feat["TREND_DAY"] == 1)

            pb_long = td & trend_up & near_fast & feat["ABOVE_VWAP"] & feat["GATE_LONG"]
            pb_short = td & trend_dn & near_fast & feat["BELOW_VWAP"] & feat["GATE_SHORT"]

            pb_long = _limit_signals_per_day(pb_long,  6, 3)
            pb_short = _limit_signals_per_day(pb_short, 6, 3)

    feat["SIG_INTRA"] = "HOLD"
    feat.loc[feat["GATE_LONG"] & feat["BREAK_LONG"], "SIG_INTRA"] = "BUY_ORB"
    feat.loc[feat["GATE_SHORT"] & feat["BREAK_SHORT"], "SIG_INTRA"] = "SELL_ORB"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & pb_long, "SIG_INTRA"] = "BUY_TREND"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & pb_short, "SIG_INTRA"] = "SELL_TREND"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & mr_long_entry, "SIG_INTRA"] = "BUY_MR"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & mr_short_entry, "SIG_INTRA"] = "SELL_MR"

    feat["SIG_POS"] = "HOLD"
    if all(c in feat.columns for c in ["EMA_50_1d", "EMA_200_1d", "EMA_50_1w", "EMA_200_1w"]):
        up_d = feat["EMA_50_1d"] > feat["EMA_200_1d"]
        up_w = feat["EMA_50_1w"] > feat["EMA_200_1w"]
        dn_d = feat["EMA_50_1d"] < feat["EMA_200_1d"]

        pos_long_gate = up_d & up_w & ((~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] > 0))
        pos_short_gate = dn_d & ((~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] < 0))

        if feat["PREMIUM_PCT"].notna().any():
            pos_long_gate = pos_long_gate & (feat["PREMIUM_PCT"] <= 1.5)
            pos_short_gate = pos_short_gate & (feat["PREMIUM_PCT"] >= -1.5)

        feat.loc[pos_long_gate, "SIG_POS"] = "BUY_SWING"
        feat.loc[pos_short_gate, "SIG_POS"] = "SELL_SWING"

    # Entries/exits
    feat = _build_intraday_entries_exits(feat=feat, has_vwap=has_vwap, vwap_col=vwap_col, atr_price=atr_price_intra)

    atr_pos_col = _pick_col(feat, ["ATR_1d", "ATR", "ATR_15m"]) or _pick_like(feat, "atr_1d")
    if atr_pos_col is not None and atr_pos_col in feat.columns and feat[atr_pos_col].notna().any():
        atr_price_pos = pd.to_numeric(feat[atr_pos_col], errors="coerce")
    else:
        atr_price_pos = pd.to_numeric(atr_price_intra, errors="coerce")

    feat = _build_positional_entries_exits(feat, atr_pos_price=atr_price_pos)

    # EXIT word columns
    feat["INTRA_EXIT_WORD"] = np.where(pd.to_numeric(feat["INTRA_EXIT"], errors="coerce").fillna(0).astype(int) == 1, "EXIT", "")
    feat["POS_EXIT_WORD"] = np.where(pd.to_numeric(feat["POS_EXIT"], errors="coerce").fillna(0).astype(int) == 1, "EXIT", "")

    # Optional: print exit events
    if PRINT_EXIT_EVENTS:
        intra_prev_dir = pd.to_numeric(feat["INTRA_DIR"], errors="coerce").shift(1).fillna(0).astype(int)
        pos_prev_dir = pd.to_numeric(feat["POS_DIR"], errors="coerce").shift(1).fillna(0).astype(int)

        intra_exit_rows = feat.loc[feat["INTRA_EXIT"] == 1, ["INTRA_REASON_EXIT"]].copy()
        for ts, r in intra_exit_rows.iterrows():
            side = "LONG" if intra_prev_dir.loc[ts] == 1 else ("SHORT" if intra_prev_dir.loc[ts] == -1 else "UNK")
            print(f"[EXIT][INTRA][{side}] {ts} reason={r.get('INTRA_REASON_EXIT','')}")

        pos_exit_rows = feat.loc[feat["POS_EXIT"] == 1, ["POS_REASON_EXIT"]].copy()
        for ts, r in pos_exit_rows.iterrows():
            side = "LONG" if pos_prev_dir.loc[ts] == 1 else ("SHORT" if pos_prev_dir.loc[ts] == -1 else "UNK")
            print(f"[EXIT][POS][{side}] {ts} reason={r.get('POS_REASON_EXIT','')}")

    # input exists flags
    feat["EXISTS_5m"] = exists_5m
    feat["EXISTS_15m"] = exists_15m
    feat["EXISTS_1h"] = exists_1h
    feat["EXISTS_1d"] = exists_1d
    feat["EXISTS_1w"] = exists_1w

    summary_cols = [
        "close",
        "SIG_INTRA", "INTRA_ENTRY", "INTRA_EXIT", "INTRA_EXIT_WORD", "INTRA_DIR", "INTRA_REASON_ENTRY", "INTRA_REASON_EXIT",
        "SIG_POS", "POS_ENTRY", "POS_EXIT", "POS_EXIT_WORD", "POS_DIR", "POS_REASON_ENTRY", "POS_REASON_EXIT",
        "MACRO_BIAS", "PREMIUM_PCT", "FAIR_SILVERBEES",
        "MCX_SILVER_RS_KG", "GSR", "EVENT_RISK",
        "DISTORTION_DAY", "TREND_DAY", "ORH", "ORL",
        "EXISTS_5m", "EXISTS_15m", "EXISTS_1h", "EXISTS_1d", "EXISTS_1w",
        "EXISTS_MCX_SILVER_RS_KG", "EXISTS_MCX_GOLD_RS_10G", "EXISTS_USDINR", "EXISTS_DXY", "EXISTS_US_REAL_YIELD", "EXISTS_EVENT_RISK"
    ]

    summary = feat.reset_index().rename(columns={"ts": "date"})
    keep = ["date"] + [c for c in summary_cols if c in summary.columns]
    summary = summary.loc[:, keep].copy()

    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    round_map = {
        "close": 2, "MACRO_BIAS": 2, "PREMIUM_PCT": 2, "FAIR_SILVERBEES": 2,
        "MCX_SILVER_RS_KG": 0, "GSR": 2
    }
    for c, nd in round_map.items():
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").round(nd)

    out_path = Path(OUT_DIR) / f"{ticker}_signals_summary_5m.csv"
    _ensure_dir(str(out_path.parent))
    summary.to_csv(out_path, index=False)

    last_ts = str(feat.index.max())
    return StrategyResult(ticker=ticker, out_path=str(out_path), last_ts=last_ts, rows=len(summary))

def main():
    ticker = "SILVERBEES"
    res = build_features_and_signals(ticker)
    print(f"[OK] {res.ticker} -> {res.out_path} | rows={res.rows} | last_ts={res.last_ts}")

if __name__ == "__main__":
    main()
