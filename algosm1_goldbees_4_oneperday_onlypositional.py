# -*- coding: utf-8 -*-
"""
GOLDBEES strategy engine (POSITIONAL ONLY) using saved indicator CSVs

WHAT THIS VERSION DOES (as requested)
- REMOVED: all intraday trade logic, intraday signals, intraday P&L, intraday trade CSV.
- Keeps ONLY positional logic:
  * No new entry while a positional position is open.
  * When flat, at most ONE entry per day (even if swing signal repeats).
  * Target = +10% (long) / -10% (short). SL disabled (0% by default).
  * Signals are ENTRY EVENTS (rising-edge within each day).

- Keeps EXISTS_* flags for ETF TF files + macro files.

INPUTS:
- etf_indicators_5min/{TICKER}_etf_indicators_5min.csv   (base bars for output alignment)
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
- etf_signals/{TICKER}_signals_summary_5m.csv        (contains only positional columns + macro/meta)
- etf_signals/{TICKER}_trades_positional.csv        (positional trades only)
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

# ---------------- INVESTMENT / P&L CONFIG ----------------
POS_INVESTMENT_RS = 10_00_000.0

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

# --- Premium / macro gates ---
PREMIUM_LONG_MAX = 1.2
PREMIUM_SHORT_MIN = -1.2
PREMIUM_EXTREME = 2.5

MACRO_LONG_MIN = 1.0
MACRO_SHORT_MAX = -1.0

# --- Macro bias zscore window ---
Z_WIN = 288

# ---------------- POSITIONAL TARGET CONFIG (PERCENT TARGETS) ----------------
POS_TP_PCT = 10.0
POS_SL_PCT = 0.0  # 0 disables SL (kept so you can change later)

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
        direction="backward",
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
        direction="backward",
    ).set_index("ts")

    out.index.name = "ts"
    return out

def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def _fill_intraday(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    day = df.index.date
    df[cols] = df.groupby(day)[cols].transform(lambda g: g.ffill().bfill())
    return df

def _edge_per_day(mask: pd.Series) -> pd.Series:
    """Rising edge within each day: True only on first bar of a new True-run."""
    if mask.empty:
        return mask
    mask = mask.fillna(False).astype(bool)
    d = mask.index.date
    prev = mask.groupby(d).shift(1).fillna(False)
    return mask & (~prev)

# ---------------- POSITIONAL ENTRY/EXIT ENGINE ----------------

def _build_positional_entries_exits_one_entry_per_day_when_flat_pct(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Positional:
    - No new entry while position open.
    - When flat, at most 1 entry per day.
    - Exit on % target. SL is disabled if POS_SL_PCT <= 0.
    """
    if "SIG_POS" not in feat.columns:
        feat["SIG_POS"] = "HOLD"

    sigp = feat["SIG_POS"].astype(str).values
    close = pd.to_numeric(feat["close"], errors="coerce").values

    idx = feat.index
    day = idx.date

    pos_dir = np.zeros(len(feat), dtype=int)
    entry = np.zeros(len(feat), dtype=int)
    exit_ = np.zeros(len(feat), dtype=int)
    reason_entry = np.array([""] * len(feat), dtype=object)
    reason_exit = np.array([""] * len(feat), dtype=object)

    pos = 0
    ep = np.nan
    tp = np.nan
    sl = np.nan

    cur_day = None
    entered_today = False

    for i in range(len(feat)):
        if cur_day is None or day[i] != cur_day:
            cur_day = day[i]
            entered_today = False

        pos_dir[i] = pos

        want_long = (sigp[i] == "BUY_SWING")
        want_short = (sigp[i] == "SELL_SWING")

        # exits
        if pos == 1:
            if (POS_SL_PCT > 0) and (not np.isnan(sl)) and (close[i] <= sl):
                exit_[i] = 1
                reason_exit[i] = "SL_PCT"
                pos = 0
            elif (not np.isnan(tp)) and (close[i] >= tp):
                exit_[i] = 1
                reason_exit[i] = f"TP_{POS_TP_PCT:.2f}PCT"
                pos = 0

        elif pos == -1:
            if (POS_SL_PCT > 0) and (not np.isnan(sl)) and (close[i] >= sl):
                exit_[i] = 1
                reason_exit[i] = "SL_PCT"
                pos = 0
            elif (not np.isnan(tp)) and (close[i] <= tp):
                exit_[i] = 1
                reason_exit[i] = f"TP_{POS_TP_PCT:.2f}PCT"
                pos = 0

        if exit_[i] == 1 and pos == 0:
            ep = np.nan
            tp = np.nan
            sl = np.nan

        # entries (only if flat and not entered today)
        if pos == 0 and (not entered_today):
            if want_long and (not np.isnan(close[i])) and close[i] > 0:
                pos = 1
                entered_today = True
                ep = close[i]
                tp = ep * (1.0 + POS_TP_PCT / 100.0)
                sl = ep * (1.0 - POS_SL_PCT / 100.0) if POS_SL_PCT > 0 else np.nan
                entry[i] = 1
                reason_entry[i] = "BUY_SWING"
            elif want_short and (not np.isnan(close[i])) and close[i] > 0:
                pos = -1
                entered_today = True
                ep = close[i]
                tp = ep * (1.0 - POS_TP_PCT / 100.0)
                sl = ep * (1.0 + POS_SL_PCT / 100.0) if POS_SL_PCT > 0 else np.nan
                entry[i] = 1
                reason_entry[i] = "SELL_SWING"

        pos_dir[i] = pos

    feat["POS_DIR"] = pos_dir
    feat["POS_ENTRY"] = entry
    feat["POS_EXIT"] = exit_
    feat["POS_REASON_ENTRY"] = reason_entry
    feat["POS_REASON_EXIT"] = reason_exit
    return feat

# ---------------- P&L ENGINE ----------------

def _calc_pnl_from_events(
    feat: pd.DataFrame,
    entry_col: str,
    exit_col: str,
    dir_col: str,
    reason_entry_col: str | None,
    reason_exit_col: str | None,
    investment_rs: float,
    prefix: str,
    price_col: str = "close",
    bar_minutes: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes realized P&L from entry/exit event flags.
    - Quantity = investment_rs / entry_price
    - Long pnl  = qty * (exit - entry)
    - Short pnl = qty * (entry - exit)

    Adds bar-level columns:
      {prefix}_ENTRY_PRICE, {prefix}_EXIT_PRICE, {prefix}_QTY,
      {prefix}_TRADE_PNL_RS (realized on exit bar), {prefix}_CUM_PNL_RS

    Returns (feat_with_cols, trades_df).
    """
    n = len(feat)
    if n == 0:
        return feat, pd.DataFrame()

    price = pd.to_numeric(feat[price_col], errors="coerce").values
    entry_flag = pd.to_numeric(feat[entry_col], errors="coerce").fillna(0).astype(int).values
    exit_flag = pd.to_numeric(feat[exit_col], errors="coerce").fillna(0).astype(int).values
    direction = pd.to_numeric(feat[dir_col], errors="coerce").fillna(0).astype(int).values

    entry_price = np.full(n, np.nan, dtype=float)
    exit_price = np.full(n, np.nan, dtype=float)
    qty = np.full(n, np.nan, dtype=float)
    trade_pnl = np.zeros(n, dtype=float)
    cum_pnl = np.zeros(n, dtype=float)
    trade_id = np.full(n, -1, dtype=int)

    open_pos = 0
    ep = np.nan
    q = np.nan
    tid = 0
    entry_i = None
    realized = 0.0

    trades = []

    for i in range(n):
        cum_pnl[i] = realized

        # open
        if entry_flag[i] == 1 and open_pos == 0:
            d = int(direction[i])
            if d != 0 and (not np.isnan(price[i])) and price[i] > 0:
                open_pos = d
                ep = float(price[i])
                q = float(investment_rs) / ep
                tid += 1
                entry_i = i

                entry_price[i] = ep
                qty[i] = q
                trade_id[i] = tid

        if open_pos != 0:
            trade_id[i] = tid

        # close
        if exit_flag[i] == 1 and open_pos != 0:
            xp = float(price[i]) if (not np.isnan(price[i])) else np.nan
            exit_price[i] = xp

            if (not np.isnan(xp)) and (not np.isnan(ep)) and (not np.isnan(q)):
                pnl = q * (xp - ep) if open_pos == 1 else q * (ep - xp)
                trade_pnl[i] = pnl
                realized += pnl
                cum_pnl[i] = realized

                entry_ts = feat.index[entry_i] if entry_i is not None else pd.NaT
                exit_ts = feat.index[i]
                bars = int(i - entry_i) if entry_i is not None else np.nan
                mins = float(bars) * float(bar_minutes) if entry_i is not None else np.nan
                ret = (xp - ep) / ep * open_pos if ep != 0 else np.nan

                trades.append({
                    "trade_id": tid,
                    "dir": "LONG" if open_pos == 1 else "SHORT",
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "entry_price": ep,
                    "exit_price": xp,
                    "qty": q,
                    "investment_rs": investment_rs,
                    "pnl_rs": pnl,
                    "ret_pct": ret * 100.0 if ret is not None and not np.isnan(ret) else np.nan,
                    "holding_bars": bars,
                    "holding_minutes": mins,
                    "reason_entry": (feat[reason_entry_col].iat[entry_i] if (reason_entry_col and entry_i is not None and reason_entry_col in feat.columns) else ""),
                    "reason_exit": (feat[reason_exit_col].iat[i] if (reason_exit_col and reason_exit_col in feat.columns) else ""),
                })

            open_pos = 0
            ep = np.nan
            q = np.nan
            entry_i = None

    feat[f"{prefix}_ENTRY_PRICE"] = entry_price
    feat[f"{prefix}_EXIT_PRICE"] = exit_price
    feat[f"{prefix}_QTY"] = qty
    feat[f"{prefix}_TRADE_PNL_RS"] = np.round(trade_pnl, 2)
    feat[f"{prefix}_CUM_PNL_RS"] = np.round(cum_pnl, 2)
    feat[f"{prefix}_TRADE_ID"] = trade_id

    return feat, pd.DataFrame(trades)

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
    exists_5m  = int((Path(DIRS["5min"])   / f"{ticker}_etf_indicators_5min.csv").exists())
    exists_15m = int((Path(DIRS["15min"])  / f"{ticker}_etf_indicators_15min.csv").exists())
    exists_1h  = int((Path(DIRS["1h"])     / f"{ticker}_etf_indicators_1h.csv").exists())
    exists_1d  = int((Path(DIRS["daily"])  / f"{ticker}_etf_indicators_daily.csv").exists())
    exists_1w  = int((Path(DIRS["weekly"]) / f"{ticker}_etf_indicators_weekly.csv").exists())

    # load
    df5  = _read_etf_file(ticker, "5min")   # base output alignment
    df15 = _read_etf_file(ticker, "15min")
    df1h = _read_etf_file(ticker, "1h")
    dfd  = _read_etf_file(ticker, "daily")
    dfw  = _read_etf_file(ticker, "weekly")

    feat = df5.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in feat.columns:
            raise ValueError(f"[{ticker}] missing required column in 5m: {c}")

    # merge TFs (kept so positional signals can use higher TF EMAs while output stays on 5m index)
    feat = _merge_asof(feat, df15, TF_COLS["15min"], "15m")
    feat = _merge_asof(feat, df1h, TF_COLS["1h"], "1h")
    feat = _merge_asof(feat, dfd,  TF_COLS["daily"], "1d")
    feat = _merge_asof(feat, dfw,  TF_COLS["weekly"], "1w")
    feat = _fill_intraday(feat, [c for c in feat.columns if c.endswith("_15m") or c.endswith("_1h")])

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
        feat[f"EXISTS_{k}"] = int(p.exists())

    # read macro series
    s_mcx_silver = _read_macro_series("MCX_SILVER_RS_KG")
    s_mcx_gold   = _read_macro_series("MCX_GOLD_RS_10G")
    s_usdinr     = _read_macro_series("USDINR")
    s_dxy        = _read_macro_series("DXY")
    s_realy      = _read_macro_series("US_REAL_YIELD")
    s_event      = _read_macro_series("EVENT_RISK")

    if s_mcx_silver is not None:
        feat = _add_series_asof(feat, s_mcx_silver, "MCX_SILVER_RS_KG")
        feat["FAIR_GOLDBEES"] = feat["MCX_SILVER_RS_KG"] / 1000.0
        feat["PREMIUM_PCT"] = (feat["close"] - feat["FAIR_GOLDBEES"]) / (feat["FAIR_GOLDBEES"] + 1e-12) * 100.0
    else:
        feat["MCX_SILVER_RS_KG"] = np.nan
        feat["FAIR_GOLDBEES"] = np.nan
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

    # MACRO_BIAS
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

    # gates (premium + macro)
    feat["GATE_LONG"] = True
    feat["GATE_SHORT"] = True
    if feat["PREMIUM_PCT"].notna().any():
        feat["GATE_LONG"] = feat["PREMIUM_PCT"] <= PREMIUM_LONG_MAX
        feat["GATE_SHORT"] = feat["PREMIUM_PCT"] >= PREMIUM_SHORT_MIN

    macro_long_ok  = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] >= MACRO_LONG_MIN)
    macro_short_ok = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] <= MACRO_SHORT_MAX)
    feat["GATE_LONG"]  = feat["GATE_LONG"]  & macro_long_ok  & (feat["DISTORTION_DAY"] == 0)
    feat["GATE_SHORT"] = feat["GATE_SHORT"] & macro_short_ok & (feat["DISTORTION_DAY"] == 0)

    # ---------------- POSITIONAL SIGNALS (ENTRY EVENTS) ----------------
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

        pos_long_event = _edge_per_day(pos_long_gate)
        pos_short_event = _edge_per_day(pos_short_gate)

        feat.loc[pos_long_event, "SIG_POS"] = "BUY_SWING"
        feat.loc[(feat["SIG_POS"] == "HOLD") & pos_short_event, "SIG_POS"] = "SELL_SWING"

    # Build positional entries/exits
    feat = _build_positional_entries_exits_one_entry_per_day_when_flat_pct(feat)

    # Exit word
    feat["POS_EXIT_WORD"] = np.where(
        pd.to_numeric(feat["POS_EXIT"], errors="coerce").fillna(0).astype(int) == 1,
        "EXIT",
        ""
    )

    # P&L calculations (positional only)
    feat, positional_trades = _calc_pnl_from_events(
        feat=feat,
        entry_col="POS_ENTRY",
        exit_col="POS_EXIT",
        dir_col="POS_DIR",
        reason_entry_col="POS_REASON_ENTRY",
        reason_exit_col="POS_REASON_EXIT",
        investment_rs=POS_INVESTMENT_RS,
        prefix="POS",
        price_col="close",
        bar_minutes=5.0,  # since bars are 5m in this output frame
    )

    # add ETF input exists flags (constant)
    feat["EXISTS_5m"] = exists_5m
    feat["EXISTS_15m"] = exists_15m
    feat["EXISTS_1h"] = exists_1h
    feat["EXISTS_1d"] = exists_1d
    feat["EXISTS_1w"] = exists_1w

    # ---------------- OUTPUT ----------------

    summary_cols = [
        "close",
        "SIG_POS", "POS_ENTRY", "POS_EXIT", "POS_EXIT_WORD", "POS_DIR",
        "POS_REASON_ENTRY", "POS_REASON_EXIT",
        "POS_ENTRY_PRICE", "POS_EXIT_PRICE", "POS_QTY", "POS_TRADE_PNL_RS", "POS_CUM_PNL_RS",
        "MACRO_BIAS", "PREMIUM_PCT", "FAIR_GOLDBEES",
        "MCX_SILVER_RS_KG", "GSR", "EVENT_RISK",
        "DISTORTION_DAY",
        "EXISTS_5m", "EXISTS_15m", "EXISTS_1h", "EXISTS_1d", "EXISTS_1w",
        "EXISTS_MCX_SILVER_RS_KG", "EXISTS_MCX_GOLD_RS_10G", "EXISTS_USDINR", "EXISTS_DXY",
        "EXISTS_US_REAL_YIELD", "EXISTS_EVENT_RISK",
    ]

    summary = feat.reset_index().rename(columns={"ts": "date"})
    keep = ["date"] + [c for c in summary_cols if c in summary.columns]
    summary = summary.loc[:, keep].copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    # rounding
    round_map = {
        "close": 2,
        "MACRO_BIAS": 2,
        "PREMIUM_PCT": 2,
        "FAIR_GOLDBEES": 2,
        "MCX_SILVER_RS_KG": 0,
        "GSR": 2,
        "POS_ENTRY_PRICE": 2,
        "POS_EXIT_PRICE": 2,
        "POS_QTY": 6,
        "POS_TRADE_PNL_RS": 2,
        "POS_CUM_PNL_RS": 2,
    }
    for c, nd in round_map.items():
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").round(nd)

    out_path = Path(OUT_DIR) / f"{ticker}_signals_summary_5m.csv"
    _ensure_dir(str(out_path.parent))
    summary.to_csv(out_path, index=False)

    # per-trade CSV (positional only)
    trades_pos_path = Path(OUT_DIR) / f"{ticker}_trades_positional.csv"
    if not positional_trades.empty:
        positional_trades.to_csv(trades_pos_path, index=False)
    else:
        pd.DataFrame(columns=[
            "trade_id","dir","entry_ts","exit_ts","entry_price","exit_price","qty","investment_rs",
            "pnl_rs","ret_pct","holding_bars","holding_minutes","reason_entry","reason_exit"
        ]).to_csv(trades_pos_path, index=False)

    last_ts = str(feat.index.max())
    return StrategyResult(ticker=ticker, out_path=str(out_path), last_ts=last_ts, rows=len(summary))


def _print_run_summary(ticker: str) -> None:
    """Pretty-print positional trade counts, invested capital, P&L, and max capital at risk (pos only)."""

    pos_fp = Path(OUT_DIR) / f"{ticker}_trades_positional.csv"
    bars_fp = Path(OUT_DIR) / f"{ticker}_signals_summary_5m.csv"

    pos = pd.read_csv(pos_fp) if pos_fp.exists() else pd.DataFrame()

    def _safe_sum(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    n_pos = int(len(pos))
    invested_pos = _safe_sum(pos, "investment_rs")
    pnl_pos = _safe_sum(pos, "pnl_rs")
    ending_pos = invested_pos + pnl_pos

    # Max capital at risk (based on bar-level positions)
    max_at_risk = 0.0
    if bars_fp.exists():
        bars = pd.read_csv(bars_fp)
        pos_dir = pd.to_numeric(bars.get("POS_DIR", 0), errors="coerce").fillna(0).astype(int)
        at_risk = pos_dir.ne(0).astype(int) * POS_INVESTMENT_RS
        max_at_risk = float(at_risk.max()) if len(at_risk) else 0.0

    def _fmt_rs(x: float) -> str:
        return f"₹{x:,.0f}"

    def _fmt_pct(pnl: float, invested: float) -> str:
        if invested == 0:
            return "0.00%"
        return f"{(pnl / invested) * 100.0:.2f}%"

    print("\n================ RUN SUMMARY (POSITIONAL ONLY) ================")
    print("POSITIONAL")
    print(f"  Trades           : {n_pos}")
    print(f"  Invested (sum)   : {_fmt_rs(invested_pos)}  (₹{POS_INVESTMENT_RS:,.0f} per trade)")
    print(f"  Profit/Loss      : {_fmt_rs(pnl_pos)}  | ROI: {_fmt_pct(pnl_pos, invested_pos)}")
    print(f"  Ending Value     : {_fmt_rs(ending_pos)}")
    print(f"  Max capital used : {_fmt_rs(max_at_risk)}")


def main():
    ticker = "GOLDBEES"
    res = build_features_and_signals(ticker)
    print(f"[OK] {res.ticker} -> {res.out_path} | rows={res.rows} | last_ts={res.last_ts}")
    print(f"[OK] Trades written: {Path(OUT_DIR) / (ticker + '_trades_positional.csv')}")
    _print_run_summary(ticker)


if __name__ == "__main__":
    main()
