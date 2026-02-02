# -*- coding: utf-8 -*-
"""
SILVERBEES strategy engine (intraday + positional) using saved indicator CSVs.

GOAL (more sensitive intraday signals):
- ORB becomes easier to trigger (smaller buffer + “OR touch” option)
- Adds a new TREND-FOLLOW intraday signal (EMA pullback + VWAP)
- Mean-reversion becomes more frequent (smaller MR_K, softer RSI filter, allows limited repeats per day)
- Still avoids MR spam: limits MR signals per day + minimum time gap between MR signals

KEY FIXES KEPT:
1) merge_asof dtype stability: convert ALL timestamps to IST and then tz-naive for merges
2) higher-TF NaNs: intraday day-wise ffill+bfill for merged columns
3) TREND_DAY fix: if ADX_15m exists but is all NaN -> fallback to ADX (5m) / ADX_1h
4) Output: clean datetime string

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

# --- Premium / macro gates (keep, but slightly looser so signals appear more often) ---
PREMIUM_LONG_MAX = 1.2
PREMIUM_SHORT_MIN = -1.2
PREMIUM_EXTREME = 2.5

MACRO_LONG_MIN = 1.0
MACRO_SHORT_MAX = -1.0

# --- ORB sensitivity ---
OR_BARS = 3
OR_BUFFER_RS = 0.005             # reduced from 0.02
ORB_USE_WICK_TOUCH = True        # breakout can trigger if high/low crosses OR levels
ORB_REQUIRE_VWAP = True          # keep vwap confirmation

# --- Trend day threshold slightly lower => more TREND_DAY=1 (enables trend signals) ---
ADX_TREND_MIN_15M = 17

# --- Macro bias zscore window ---
Z_WIN = 288

# --- Mean reversion sensitivity ---
MR_K = 1.2                       # reduced from 1.8 (more MR)
USE_RSI_FILTER_FOR_MR = True
MR_RSI_LONG_MAX = 45             # softer filter
MR_RSI_SHORT_MIN = 55            # softer filter
VOL_UNIT_FLOOR = 0.12            # slightly lower

# --- MR spam control (but allow more than 1 per day) ---
MR_MAX_SIGNALS_PER_DAY = 3        # allow up to 3 MR signals per day
MR_MIN_GAP_BARS = 6               # at least 6*5m = 30 min between MR signals

# --- New intraday TREND_PULLBACK signal controls ---
ENABLE_TREND_PULLBACK = True
PULLBACK_ATR_MULT = 0.35          # how close to EMA to count as pullback (scaled by ATR)
TREND_EMA_FAST_CANDIDATES = ["EMA_20", "EMA_20_15m", "EMA_50", "EMA_50_15m"]
TREND_EMA_SLOW_CANDIDATES = ["EMA_50", "EMA_50_15m", "EMA_200", "EMA_200_15m"]


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

def _read_etf_file(ticker: str, tf: str) -> pd.DataFrame:
    fp = Path(DIRS[tf]) / f"{ticker}_etf_indicators_{tf}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    df = pd.read_csv(fp)
    if df.empty or "date" not in df.columns:
        raise ValueError(f"Bad/empty CSV: {fp}")

    df["date"] = _to_ist_naive_datetime(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df.set_index("date")

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
    return s.sort_index()

def _merge_asof(base: pd.DataFrame, higher: pd.DataFrame, cols: list[str], suffix: str) -> pd.DataFrame:
    cols = [c for c in cols if c in higher.columns]
    if not cols:
        return base
    b = base.reset_index().rename(columns={"date": "ts"})
    h = higher.reset_index().rename(columns={"date": "ts"})[["ts"] + cols]
    out = pd.merge_asof(b.sort_values("ts"), h.sort_values("ts"), on="ts", direction="backward").set_index("ts")
    return out.rename(columns={c: f"{c}_{suffix}" for c in cols})

def _add_series_asof(feat: pd.DataFrame, s: pd.Series, colname: str) -> pd.DataFrame:
    b = feat.reset_index().rename(columns={"date": "ts"})
    m = s.rename(colname).to_frame().reset_index().rename(columns={"index": "ts"})
    b["ts"] = pd.to_datetime(b["ts"], errors="coerce")
    m["ts"] = pd.to_datetime(m["ts"], errors="coerce")
    out = pd.merge_asof(b.sort_values("ts"), m.sort_values("ts"), on="ts", direction="backward").set_index("ts")
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
    """
    Keep at most max_per_day True per day, enforcing a minimum bar-gap between Trues.
    Works on boolean series aligned to the dataframe index.
    """
    out = pd.Series(False, index=sig_mask.index)
    day = sig_mask.index.date

    for d, idx in pd.Series(sig_mask.index, index=sig_mask.index).groupby(day):
        # idx is the timestamps for that day (in original order)
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

        # mark chosen positions
        out.loc[idx[chosen]] = True

    return out


# ---------------- CORE ----------------

@dataclass
class StrategyResult:
    ticker: str
    out_path: str
    last_ts: str
    rows: int

def build_features_and_signals(ticker: str) -> StrategyResult:
    _ensure_dir(OUT_DIR)

    df5  = _read_etf_file(ticker, "5min")
    df15 = _read_etf_file(ticker, "15min")
    df1h = _read_etf_file(ticker, "1h")
    dfd  = _read_etf_file(ticker, "daily")
    dfw  = _read_etf_file(ticker, "weekly")

    feat = df5.copy()

    # required
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in feat.columns:
            raise ValueError(f"[{ticker}] missing required column in 5m: {c}")

    # merge TFs
    feat = _merge_asof(feat, df15, TF_COLS["15min"], "15m")
    feat = _merge_asof(feat, df1h, TF_COLS["1h"], "1h")
    feat = _merge_asof(feat, dfd,  TF_COLS["daily"], "1d")
    feat = _merge_asof(feat, dfw,  TF_COLS["weekly"], "1w")

    # fill higher-TF gaps intraday
    feat = _fill_intraday(feat, [c for c in feat.columns if c.endswith("_15m") or c.endswith("_1h")])

    # opening range
    feat = _opening_range(feat, OR_BARS)

    # TREND_DAY (more sensitive due to lower threshold)
    adx_col = _pick_col(feat, ["ADX_15m"]) or _pick_col(feat, ["ADX", "ADX_1h"])
    if adx_col is None:
        feat["TREND_DAY"] = np.nan
    else:
        feat["TREND_DAY"] = (pd.to_numeric(feat[adx_col], errors="coerce") >= ADX_TREND_MIN_15M).astype(int)

    # macros
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

    # macro flags
    feat["HAS_MACRO"] = feat[["USDINR", "DXY", "US_REAL_YIELD"]].notna().any(axis=1)

    # MACRO_BIAS (looser thresholds above, but keep same construction)
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

    # distortion
    feat["DISTORTION_DAY"] = 0
    if feat["PREMIUM_PCT"].notna().any():
        feat["DISTORTION_DAY"] = (feat["PREMIUM_PCT"].abs() >= PREMIUM_EXTREME).astype(int)

    # gates (premium)
    feat["GATE_LONG"] = True
    feat["GATE_SHORT"] = True
    if feat["PREMIUM_PCT"].notna().any():
        feat["GATE_LONG"] = feat["PREMIUM_PCT"] <= PREMIUM_LONG_MAX
        feat["GATE_SHORT"] = feat["PREMIUM_PCT"] >= PREMIUM_SHORT_MIN

    # macro gates: if macro missing -> don't block
    macro_long_ok  = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] >= MACRO_LONG_MIN)
    macro_short_ok = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] <= MACRO_SHORT_MAX)
    feat["GATE_LONG"]  = feat["GATE_LONG"]  & macro_long_ok  & (feat["DISTORTION_DAY"] == 0)
    feat["GATE_SHORT"] = feat["GATE_SHORT"] & macro_short_ok & (feat["DISTORTION_DAY"] == 0)

    # VWAP
    vwap_col = _pick_col(feat, ["VWAP", "VWAP_15m"]) or _pick_like(feat, "vwap")
    has_vwap = vwap_col is not None and feat[vwap_col].notna().any()

    if not has_vwap:
        feat["ABOVE_VWAP"] = True
        feat["BELOW_VWAP"] = True
        feat["VWAP_DIST_PCT"] = 0.0
    else:
        vwap = pd.to_numeric(feat[vwap_col], errors="coerce")
        feat["ABOVE_VWAP"] = feat["close"] >= vwap
        feat["BELOW_VWAP"] = feat["close"] <= vwap
        feat["VWAP_DIST_PCT"] = (feat["close"] - vwap) / (vwap + 1e-12) * 100.0

    # ORB: more sensitive (wick touch allowed)
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

    # VOL_UNIT
    atr_col = _pick_col(feat, ["ATR", "ATR_15m"]) or _pick_like(feat, "atr")
    if atr_col is not None and feat[atr_col].notna().any():
        atr = pd.to_numeric(feat[atr_col], errors="coerce")
        feat["VOL_UNIT"] = (atr / (feat["close"] + 1e-12)) * 100.0
    else:
        feat["VOL_UNIT"] = feat["VWAP_DIST_PCT"].rolling(50).std()

    feat["VOL_UNIT"] = pd.to_numeric(feat["VOL_UNIT"], errors="coerce").fillna(VOL_UNIT_FLOOR).clip(lower=VOL_UNIT_FLOOR)

    # WEAK_BIAS (slightly wider)
    feat["WEAK_BIAS"] = (~feat["HAS_MACRO"]) | feat["MACRO_BIAS"].between(-1.5, 1.5)

    # MR base (allow MR also when TREND_DAY unknown, but only if VWAP exists)
    trend_ok = (feat["TREND_DAY"] == 0) | (feat["TREND_DAY"].isna())
    mr_base = feat["WEAK_BIAS"] & trend_ok & (feat["DISTORTION_DAY"] == 0) & has_vwap

    mr_long  = mr_base & (feat["VWAP_DIST_PCT"] <= -MR_K * feat["VOL_UNIT"])
    mr_short = mr_base & (feat["VWAP_DIST_PCT"] >=  MR_K * feat["VOL_UNIT"])

    # RSI filter (softer)
    rsi_col = _pick_col(feat, ["RSI", "RSI_15m"]) or _pick_like(feat, "rsi")
    if USE_RSI_FILTER_FOR_MR and (rsi_col is not None) and feat[rsi_col].notna().any():
        rsi = pd.to_numeric(feat[rsi_col], errors="coerce")
        mr_long  = mr_long  & (rsi <= MR_RSI_LONG_MAX)
        mr_short = mr_short & (rsi >= MR_RSI_SHORT_MIN)

    # Limit MR per day (allows more, but controlled)
    mr_long_entry  = _limit_signals_per_day(mr_long,  MR_MAX_SIGNALS_PER_DAY, MR_MIN_GAP_BARS)
    mr_short_entry = _limit_signals_per_day(mr_short, MR_MAX_SIGNALS_PER_DAY, MR_MIN_GAP_BARS)

    # New TREND_PULLBACK intraday signals (more opportunities on trend days)
    pb_long = pd.Series(False, index=feat.index)
    pb_short = pd.Series(False, index=feat.index)

    if ENABLE_TREND_PULLBACK and has_vwap:
        ema_fast = _pick_col(feat, TREND_EMA_FAST_CANDIDATES) or _pick_like(feat, "ema_20") or _pick_like(feat, "ema 20")
        ema_slow = _pick_col(feat, TREND_EMA_SLOW_CANDIDATES) or _pick_like(feat, "ema_50") or _pick_like(feat, "ema_200")

        if ema_fast is not None and ema_slow is not None:
            ef = pd.to_numeric(feat[ema_fast], errors="coerce")
            es = pd.to_numeric(feat[ema_slow], errors="coerce")
            atrv = pd.to_numeric(feat[atr_col], errors="coerce") if (atr_col is not None and feat[atr_col].notna().any()) else np.nan

            # pullback zone width using ATR in price units (ATR is usually in price units)
            # if ATR not present, approximate using close * VOL_UNIT%
            if np.isnan(atrv).all():
                atr_price = feat["close"] * (feat["VOL_UNIT"] / 100.0)
            else:
                atr_price = atrv

            near_fast = (feat["close"] - ef).abs() <= (PULLBACK_ATR_MULT * atr_price)

            trend_up = (ef > es)
            trend_dn = (ef < es)

            # trend-day filter (more sensitive threshold already)
            td = (feat["TREND_DAY"] == 1)

            pb_long = td & trend_up & near_fast & feat["ABOVE_VWAP"] & feat["GATE_LONG"]
            pb_short = td & trend_dn & near_fast & feat["BELOW_VWAP"] & feat["GATE_SHORT"]

            # debounce pullback signals (avoid repeating every bar)
            pb_long = _limit_signals_per_day(pb_long,  6, 3)   # allow more, smaller gap
            pb_short = _limit_signals_per_day(pb_short, 6, 3)

    # Final intraday signals with priority:
    # ORB first, then TREND_PULLBACK, then MR
    feat["SIG_INTRA"] = "HOLD"

    feat.loc[feat["GATE_LONG"] & feat["BREAK_LONG"], "SIG_INTRA"] = "BUY_ORB"
    feat.loc[feat["GATE_SHORT"] & feat["BREAK_SHORT"], "SIG_INTRA"] = "SELL_ORB"

    feat.loc[(feat["SIG_INTRA"] == "HOLD") & pb_long, "SIG_INTRA"] = "BUY_TREND"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & pb_short, "SIG_INTRA"] = "SELL_TREND"

    feat.loc[(feat["SIG_INTRA"] == "HOLD") & mr_long_entry, "SIG_INTRA"] = "BUY_MR"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & mr_short_entry, "SIG_INTRA"] = "SELL_MR"

    # positional
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

    # output summary
    summary_cols = [
        "close", "SIG_INTRA", "SIG_POS",
        "MACRO_BIAS", "PREMIUM_PCT", "FAIR_SILVERBEES",
        "MCX_SILVER_RS_KG", "GSR", "EVENT_RISK",
        "DISTORTION_DAY", "TREND_DAY", "ORH", "ORL"
    ]

    summary = feat.reset_index().rename(columns={"ts": "date", "index": "date"})
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
    summary.to_csv(out_path, index=False)

    last_ts = str(feat.index.max())
    return StrategyResult(ticker=ticker, out_path=str(out_path), last_ts=last_ts, rows=len(summary))

def main():
    ticker = "SILVERBEES"
    res = build_features_and_signals(ticker)
    print(f"[OK] {res.ticker} -> {res.out_path} | rows={res.rows} | last_ts={res.last_ts}")

if __name__ == "__main__":
    main()
