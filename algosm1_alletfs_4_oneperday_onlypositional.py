# -*- coding: utf-8 -*-
"""
ALGO-SM1 ETF strategy engine (POSITIONAL ONLY) — 15-minute base timeframe.

CHANGES IN THIS REWRITE (as requested)
- BUY_SWING / SELL_SWING entry signals do NOT repeat until the previous position is exited.
  * Signals are "consumed" by the position engine:
    - If flat and eligible -> signal becomes an entry on that bar.
    - If in position -> same-direction signals are suppressed.
    - Opposite signal is allowed ONLY if it actually triggers an exit on that bar.
- Per-ticker run summary prints OPEN/UNEXITED trade separately (unrealized P&L).
- Overall summary prints:
  - total invested (closed trades)
  - total profit (realized)
  - net investment / ending value (realized + open invested + unrealized)

Notes
- Still: single open position at a time.
- Still: at most ONE entry per day WHEN FLAT (entered_today gate).
- Exit: TP / SL (optional) / time-stop (MAX_HOLD_DAYS) / opposite signal (if enabled).
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)
IST_TZ = pytz.timezone("Asia/Kolkata")

# ---------------- BASE TIMEFRAME ----------------
BASE_TF = "15min"
BAR_MINUTES = 15.0

# ---------------- INVESTMENT / P&L ----------------
POS_INVESTMENT_RS = 1_00_000.0  # per trade

# ---------------- DIRECTORIES (must match your indicator pipeline) ----------------
DIRS = {
    "15min":  "etf_indicators_15min",
    "1h":     "etf_indicators_1h",
    "daily":  "etf_indicators_daily",
    "weekly": "etf_indicators_weekly",
}
OUT_DIR = "etf_signals"
MACRO_DIR = "macro_inputs"

# ---------------- INDICATOR COLS EXPECTED IN HIGHER TF FILES ----------------
TF_COLS = {
    "1h":     ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "RSI"],
    "daily":  ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "Recent_High", "Recent_Low", "RSI"],
    "weekly": ["EMA_20", "EMA_50", "EMA_200", "ADX", "ATR", "RSI"],
}

# ---------------- POSITIONAL ENTRY/EXIT CONFIG ----------------
POS_TP_PCT = 4.0
POS_SL_PCT = 0.0  # 0 disables SL

MAX_HOLD_DAYS = 60
BARS_PER_DAY = int(round(390.0 / BAR_MINUTES))  # NSE 09:15-15:30 => 390 min => 26 bars on 15m
MAX_HOLD_BARS = int(MAX_HOLD_DAYS * BARS_PER_DAY)

# If True, opposite signal can exit early (in addition to TP/SL/TIME_STOP)
EXIT_ON_OPPOSITE_SIGNAL = True

# ---------------- GATES ----------------
# Premium gates (metals only, if proxy series is present)
PREMIUM_LONG_MAX = 2.0
PREMIUM_SHORT_MIN = -2.0
PREMIUM_EXTREME = 4.0

# Macro bias gates (soft, only applied when macro exists)
MACRO_LONG_MIN = 0.5
MACRO_SHORT_MAX = -0.25

# Macro zscore window in bars
Z_WIN = 96


USE_MACROS = True

# ---------------- ETF GROUPING + MACRO SELECTION ----------------
METAL_ETFS_GOLD = {"GOLDBEES", "SETFGOLD", "EGOLD", "GOLD1", "BBNPPGOLD", "BSLGOLDETF"}
METAL_ETFS_SILVER = {"SILVERBEES", "SILVER", "AXISILVER", "TATSILV"}

DEFAULT_MACROS_EQUITY = ["USDINR", "DXY", "US_REAL_YIELD", "US10Y", "VIX", "CURVE_T10Y3M", "WTI", "EVENT_RISK"]
DEFAULT_MACROS_METALS = ["USDINR", "DXY", "US_REAL_YIELD", "XAUUSD", "XAGUSD", "MCX_GOLD_RS_10G", "MCX_SILVER_RS_KG", "EVENT_RISK"]


# ---------------- HELPERS ----------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_ist_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    # If tz-naive series, localize to IST; otherwise convert to IST.
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
    df = (
        df.dropna(subset=["date"])
          .drop_duplicates(subset="date")
          .sort_values("date")
          .reset_index(drop=True)
          .set_index("date")
    )
    df.index.name = "ts"
    return df

def _read_macro_series(name: str) -> pd.Series | None:
    fp = Path(MACRO_DIR) / f"{name}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
    m = s.rename(colname).to_frame().reset_index().rename(columns={"index": "ts"})
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

def _fill_intraday(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    day = df.index.date
    df[cols] = df.groupby(day)[cols].transform(lambda g: g.ffill().bfill())
    return df

def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None

def _fmt_rs(x: float) -> str:
    return f"₹{x:,.0f}"

def _fmt_pct(pnl: float, invested: float) -> str:
    if invested == 0:
        return "0.00%"
    return f"{(pnl / invested) * 100.0:.2f}%"

def _choose_macro_list_for_ticker(ticker: str) -> list[str]:
    """
    Returns macro file names to load for a ticker.
    NOTE: If USE_MACROS=False, caller won't use this anyway.
    """
    t = str(ticker).strip().upper()
    if (t in METAL_ETFS_GOLD) or (t in METAL_ETFS_SILVER):
        return DEFAULT_MACROS_METALS
    return DEFAULT_MACROS_EQUITY


def _apply_macro_block_or_pass(feat: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Returns (macro_long_ok, macro_short_ok)
    - If USE_MACROS=False => always True (macros never block trades)
    - If USE_MACROS=True  => same original macro gating based on HAS_MACRO & MACRO_BIAS
    """
    if not USE_MACROS:
        # broadcast True series aligned to feat index
        return (pd.Series(True, index=feat.index), pd.Series(True, index=feat.index))

    macro_long_ok  = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] >= MACRO_LONG_MIN)
    macro_short_ok = (~feat["HAS_MACRO"]) | (feat["MACRO_BIAS"] <= MACRO_SHORT_MAX)
    return (macro_long_ok, macro_short_ok)

# ---------------- POSITIONAL ENGINE (CONSUMES SIGNALS) ----------------
def _build_positional_entries_exits_and_signals(
    feat: pd.DataFrame,
    candidate_col: str,
    max_hold_bars: int,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Positional engine:
    - Only one open position at a time.
    - When flat, at most 1 entry per day.
    - Exit on TP/SL/TimeStop/Opposite signal (optional).
    - Signals are "consumed":
        * SIG_POS shows only when it causes an entry OR when opposite signal causes an exit.
        * No repeated BUY/SELL signals while position is open.
    Returns:
    - feat with POS_* columns + SIG_POS (consumed)
    - open_position_state (for later unrealized reporting), or None
    """
    if candidate_col not in feat.columns:
        feat[candidate_col] = "HOLD"

    cand = feat[candidate_col].astype(str).values
    close = pd.to_numeric(feat["close"], errors="coerce").values
    idx = feat.index
    day = idx.date
    n = len(feat)

    pos_dir = np.zeros(n, dtype=int)
    entry = np.zeros(n, dtype=int)
    exit_ = np.zeros(n, dtype=int)
    reason_entry = np.array([""] * n, dtype=object)
    reason_exit = np.array([""] * n, dtype=object)
    sig_used = np.array(["HOLD"] * n, dtype=object)

    pos = 0  # 1 long, -1 short, 0 flat
    ep = np.nan
    tp = np.nan
    sl = np.nan
    entry_i = None
    entry_ts = None

    cur_day = None
    entered_today = False

    for i in range(n):
        # reset daily entry gate
        if cur_day is None or day[i] != cur_day:
            cur_day = day[i]
            entered_today = False

        price = close[i]
        want_long = (cand[i] == "BUY_SWING")
        want_short = (cand[i] == "SELL_SWING")

        # ---------------- EXITS ----------------
        if pos != 0 and (not np.isnan(price)) and price > 0:
            # SL/TP depend on direction
            if pos == 1:
                if (POS_SL_PCT > 0) and (not np.isnan(sl)) and (price <= sl):
                    exit_[i] = 1
                    reason_exit[i] = "SL_PCT"
                    pos = 0
                elif (not np.isnan(tp)) and (price >= tp):
                    exit_[i] = 1
                    reason_exit[i] = f"TP_{POS_TP_PCT:.2f}PCT"
                    pos = 0
                elif (entry_i is not None) and ((i - entry_i) >= max_hold_bars):
                    exit_[i] = 1
                    reason_exit[i] = "TIME_STOP"
                    pos = 0
                elif EXIT_ON_OPPOSITE_SIGNAL and want_short:
                    exit_[i] = 1
                    reason_exit[i] = "OPPOSITE_SIGNAL"
                    pos = 0
                    sig_used[i] = "SELL_SWING"  # opposite consumed as exit trigger

            elif pos == -1:
                if (POS_SL_PCT > 0) and (not np.isnan(sl)) and (price >= sl):
                    exit_[i] = 1
                    reason_exit[i] = "SL_PCT"
                    pos = 0
                elif (not np.isnan(tp)) and (price <= tp):
                    exit_[i] = 1
                    reason_exit[i] = f"TP_{POS_TP_PCT:.2f}PCT"
                    pos = 0
                elif (entry_i is not None) and ((i - entry_i) >= max_hold_bars):
                    exit_[i] = 1
                    reason_exit[i] = "TIME_STOP"
                    pos = 0
                elif EXIT_ON_OPPOSITE_SIGNAL and want_long:
                    exit_[i] = 1
                    reason_exit[i] = "OPPOSITE_SIGNAL"
                    pos = 0
                    sig_used[i] = "BUY_SWING"   # opposite consumed as exit trigger

            if exit_[i] == 1 and pos == 0:
                # clear position state
                ep = np.nan
                tp = np.nan
                sl = np.nan
                entry_i = None
                entry_ts = None

        # ---------------- ENTRIES (ONLY WHEN FLAT) ----------------
        if pos == 0 and (not entered_today) and (not np.isnan(price)) and price > 0:
            if want_long:
                pos = 1
                entered_today = True
                entry_i = i
                entry_ts = idx[i]
                ep = float(price)
                tp = ep * (1.0 + POS_TP_PCT / 100.0)
                sl = ep * (1.0 - POS_SL_PCT / 100.0) if POS_SL_PCT > 0 else np.nan
                entry[i] = 1
                reason_entry[i] = "BUY_SWING"
                sig_used[i] = "BUY_SWING"  # consumed as entry trigger

            elif want_short:
                pos = -1
                entered_today = True
                entry_i = i
                entry_ts = idx[i]
                ep = float(price)
                tp = ep * (1.0 - POS_TP_PCT / 100.0)
                sl = ep * (1.0 + POS_SL_PCT / 100.0) if POS_SL_PCT > 0 else np.nan
                entry[i] = 1
                reason_entry[i] = "SELL_SWING"
                sig_used[i] = "SELL_SWING"  # consumed as entry trigger

        pos_dir[i] = pos

    feat["SIG_POS"] = sig_used
    feat["POS_DIR"] = pos_dir
    feat["POS_ENTRY"] = entry
    feat["POS_EXIT"] = exit_
    feat["POS_REASON_ENTRY"] = reason_entry
    feat["POS_REASON_EXIT"] = reason_exit

    # build open position state (if still open at end)
    open_state = None
    if pos != 0 and entry_i is not None and entry_ts is not None and (not np.isnan(ep)):
        last_price = float(close[-1]) if (n > 0 and not np.isnan(close[-1])) else np.nan
        qty = float(POS_INVESTMENT_RS) / float(ep) if ep > 0 else np.nan
        unreal = np.nan
        if not np.isnan(last_price) and not np.isnan(qty):
            unreal = qty * (last_price - ep) if pos == 1 else qty * (ep - last_price)

        open_state = {
            "dir": "LONG" if pos == 1 else "SHORT",
            "entry_ts": entry_ts,
            "entry_price": float(ep),
            "last_ts": idx[-1] if n else pd.NaT,
            "last_price": last_price,
            "qty": qty,
            "investment_rs": float(POS_INVESTMENT_RS),
            "unrealized_pnl_rs": float(unreal) if unreal is not None and not np.isnan(unreal) else 0.0,
            "holding_bars": int((n - 1) - entry_i) if n else 0,
            "holding_minutes": float(int((n - 1) - entry_i)) * float(BAR_MINUTES) if n else 0.0,
        }

    return feat, open_state


# ---------------- P&L (REALIZED TRADES ONLY) ----------------
def _calc_realized_trades_from_events(
    feat: pd.DataFrame,
    entry_col: str,
    exit_col: str,
    dir_col: str,
    reason_entry_col: str | None,
    reason_exit_col: str | None,
    investment_rs: float,
    price_col: str = "close",
    bar_minutes: float = 15.0,
) -> pd.DataFrame:
    """
    Computes realized trades from entry/exit flags.
    Quantity = investment_rs / entry_price.
    Long pnl  = qty*(exit-entry), Short pnl = qty*(entry-exit).
    """
    if feat.empty:
        return pd.DataFrame()

    price = pd.to_numeric(feat[price_col], errors="coerce").values
    entry_flag = pd.to_numeric(feat[entry_col], errors="coerce").fillna(0).astype(int).values
    exit_flag = pd.to_numeric(feat[exit_col], errors="coerce").fillna(0).astype(int).values
    direction = pd.to_numeric(feat[dir_col], errors="coerce").fillna(0).astype(int).values

    open_pos = 0
    ep = np.nan
    q = np.nan
    entry_i = None
    tid = 0
    trades = []

    for i in range(len(feat)):
        if entry_flag[i] == 1 and open_pos == 0:
            d = int(direction[i])
            if d != 0 and (not np.isnan(price[i])) and price[i] > 0:
                open_pos = d
                ep = float(price[i])
                q = float(investment_rs) / ep
                entry_i = i
                tid += 1

        if exit_flag[i] == 1 and open_pos != 0 and entry_i is not None:
            xp = float(price[i]) if (not np.isnan(price[i])) else np.nan
            if (not np.isnan(xp)) and (not np.isnan(ep)) and (not np.isnan(q)):
                pnl = q * (xp - ep) if open_pos == 1 else q * (ep - xp)
                bars = int(i - entry_i)
                mins = float(bars) * float(bar_minutes)
                ret = (xp - ep) / ep * open_pos if ep != 0 else np.nan

                trades.append({
                    "trade_id": tid,
                    "dir": "LONG" if open_pos == 1 else "SHORT",
                    "entry_ts": feat.index[entry_i],
                    "exit_ts": feat.index[i],
                    "entry_price": ep,
                    "exit_price": xp,
                    "qty": q,
                    "investment_rs": float(investment_rs),
                    "pnl_rs": float(pnl),
                    "ret_pct": float(ret * 100.0) if ret is not None and not np.isnan(ret) else np.nan,
                    "holding_bars": bars,
                    "holding_minutes": mins,
                    "reason_entry": (feat[reason_entry_col].iat[entry_i] if (reason_entry_col and reason_entry_col in feat.columns) else ""),
                    "reason_exit": (feat[reason_exit_col].iat[i] if (reason_exit_col and reason_exit_col in feat.columns) else ""),
                })

            open_pos = 0
            ep = np.nan
            q = np.nan
            entry_i = None

    return pd.DataFrame(trades)


# ---------------- CORE ----------------
@dataclass
class StrategyResult:
    ticker: str
    out_path: str
    last_ts: str
    rows: int

    # realized
    n_closed_trades: int
    invested_closed: float
    realized_pnl: float

    # open
    has_open_trade: bool
    invested_open: float
    unrealized_pnl: float
    open_trade: dict | None

    # peak risk
    max_capital_used: float


def _choose_macro_list_for_ticker(ticker: str) -> list[str]:
    t = str(ticker).strip().upper()
    if (t in METAL_ETFS_GOLD) or (t in METAL_ETFS_SILVER):
        return DEFAULT_MACROS_METALS
    return DEFAULT_MACROS_EQUITY


def build_features_and_signals(ticker: str) -> StrategyResult:
    _ensure_dir(OUT_DIR)
    tkr = str(ticker).strip().upper()

    # input exists flags
    exists_15m = int((Path(DIRS["15min"]) / f"{tkr}_etf_indicators_15min.csv").exists())
    exists_1h  = int((Path(DIRS["1h"])    / f"{tkr}_etf_indicators_1h.csv").exists())
    exists_1d  = int((Path(DIRS["daily"]) / f"{tkr}_etf_indicators_daily.csv").exists())
    exists_1w  = int((Path(DIRS["weekly"])/ f"{tkr}_etf_indicators_weekly.csv").exists())

    # base load
    df15 = _read_etf_file(tkr, "15min")
    feat = df15.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in feat.columns:
            raise ValueError(f"[{tkr}] missing required column in {BASE_TF}: {c}")

    # merge higher TFs (optional)
    if exists_1h:
        df1h = _read_etf_file(tkr, "1h")
        feat = _merge_asof(feat, df1h, TF_COLS["1h"], "1h")
    if exists_1d:
        dfd = _read_etf_file(tkr, "daily")
        feat = _merge_asof(feat, dfd, TF_COLS["daily"], "1d")
    if exists_1w:
        dfw = _read_etf_file(tkr, "weekly")
        feat = _merge_asof(feat, dfw, TF_COLS["weekly"], "1w")

    feat = _fill_intraday(
        feat,
        [c for c in feat.columns if c.endswith("_1h") or c.endswith("_1d") or c.endswith("_1w")]
    )

    # ---------------- MACROS (TOGGLE) ----------------
    # Default values always present (so downstream code doesn't break)
    feat["EVENT_RISK"] = 0
    feat["HAS_MACRO"] = False
    feat["MACRO_BIAS"] = 0.0

    macro_names: list[str] = []
    if USE_MACROS:
        # only when enabled:
        macro_names = _choose_macro_list_for_ticker(tkr)

        # merge macros
        macro_series: dict[str, pd.Series] = {}
        for nm in macro_names:
            s = _read_macro_series(nm)
            if s is not None and not s.empty:
                macro_series[nm] = s

        # exists flags for macro files
        for nm in macro_names:
            feat[f"EXISTS_{nm}"] = int((Path(MACRO_DIR) / f"{nm}.csv").exists())

        for nm, s in macro_series.items():
            feat = _add_series_asof(feat, s, nm)

        # Event risk
        if "EVENT_RISK" in feat.columns:
            feat["EVENT_RISK"] = pd.to_numeric(feat["EVENT_RISK"], errors="coerce").fillna(0).clip(0, 9)
        else:
            feat["EVENT_RISK"] = 0

        # HAS_MACRO robust
        core_macro_cols = [c for c in ["USDINR", "DXY", "US_REAL_YIELD", "US10Y", "VIX", "CURVE_T10Y3M", "WTI"] if c in feat.columns]
        feat["HAS_MACRO"] = feat[core_macro_cols].notna().any(axis=1) if core_macro_cols else False

        # Macro bias
        feat["MACRO_BIAS"] = 0.0

        def _add_bias(col: str, weight: float, sign: float = 1.0):
            if col in feat.columns and feat[col].notna().any():
                z = _zscore(pd.to_numeric(feat[col], errors="coerce"), Z_WIN).fillna(0)
                feat["MACRO_BIAS"] = feat["MACRO_BIAS"] + (weight * sign * z)

        # USDINR: metals positive, equities negative
        if "USDINR" in feat.columns and feat["USDINR"].notna().any():
            z_usdinr = _zscore(pd.to_numeric(feat["USDINR"], errors="coerce"), Z_WIN).fillna(0)
            if tkr in (METAL_ETFS_GOLD | METAL_ETFS_SILVER):
                feat["MACRO_BIAS"] += 1.0 * z_usdinr
            else:
                feat["MACRO_BIAS"] += -1.0 * z_usdinr

        _add_bias("DXY", 0.7, sign=-1.0)
        _add_bias("US_REAL_YIELD", 0.8, sign=-1.0)
        _add_bias("VIX", 0.8, sign=-1.0)
        _add_bias("US10Y", 0.3, sign=-1.0)
        _add_bias("CURVE_T10Y3M", 0.2, sign=+1.0)
        _add_bias("WTI", 0.1, sign=-1.0)

        # damp bias on event risk
        feat["BIAS_DAMP"] = (1.0 - 0.08 * feat["EVENT_RISK"]).clip(0.5, 1.0)
        feat["MACRO_BIAS"] = (feat["MACRO_BIAS"] * feat["BIAS_DAMP"]).clip(-5, 5)

    # ---------------- PREMIUM / DISTORTION (unchanged) ----------------
    feat["PREMIUM_PCT"] = np.nan
    feat["FAIR_PRICE"] = np.nan

    if tkr in METAL_ETFS_GOLD and "MCX_GOLD_RS_10G" in feat.columns and feat["MCX_GOLD_RS_10G"].notna().any():
        feat["FAIR_PRICE"] = pd.to_numeric(feat["MCX_GOLD_RS_10G"], errors="coerce")
        feat["PREMIUM_PCT"] = (feat["close"] - feat["FAIR_PRICE"]) / (feat["FAIR_PRICE"] + 1e-12) * 100.0

    if tkr in METAL_ETFS_SILVER and "MCX_SILVER_RS_KG" in feat.columns and feat["MCX_SILVER_RS_KG"].notna().any():
        feat["FAIR_PRICE"] = pd.to_numeric(feat["MCX_SILVER_RS_KG"], errors="coerce")
        feat["PREMIUM_PCT"] = (feat["close"] - feat["FAIR_PRICE"]) / (feat["FAIR_PRICE"] + 1e-12) * 100.0

    feat["DISTORTION_DAY"] = 0
    if feat["PREMIUM_PCT"].notna().any():
        feat["DISTORTION_DAY"] = (feat["PREMIUM_PCT"].abs() >= PREMIUM_EXTREME).astype(int)

    # ---------------- GATES ----------------
    feat["GATE_LONG"] = True
    feat["GATE_SHORT"] = True

    # premium gate only if premium exists
    if feat["PREMIUM_PCT"].notna().any():
        feat["GATE_LONG"] = feat["PREMIUM_PCT"] <= PREMIUM_LONG_MAX
        feat["GATE_SHORT"] = feat["PREMIUM_PCT"] >= PREMIUM_SHORT_MIN

    # macro gate (toggle)
    macro_long_ok, macro_short_ok = _apply_macro_block_or_pass(feat)

    feat["GATE_LONG"]  = feat["GATE_LONG"]  & macro_long_ok  & (feat["DISTORTION_DAY"] == 0)
    feat["GATE_SHORT"] = feat["GATE_SHORT"] & macro_short_ok & (feat["DISTORTION_DAY"] == 0)

    # ---------------- CANDIDATE POSITIONAL SIGNALS (unchanged) ----------------
    feat["SIG_CAND"] = "HOLD"

    if all(c in feat.columns for c in ["EMA_50_1d", "EMA_200_1d", "EMA_50_1w", "EMA_200_1w"]):
        up_d = feat["EMA_50_1d"] > feat["EMA_200_1d"]
        up_w = feat["EMA_50_1w"] > feat["EMA_200_1w"]
        dn_d = feat["EMA_50_1d"] < feat["EMA_200_1d"]
        dn_w = feat["EMA_50_1w"] < feat["EMA_200_1w"]

        cand_long = up_d & (~dn_w) & feat["GATE_LONG"]
        cand_short = dn_d & (~up_w) & feat["GATE_SHORT"]

        feat.loc[cand_long, "SIG_CAND"] = "BUY_SWING"
        feat.loc[(feat["SIG_CAND"] == "HOLD") & cand_short, "SIG_CAND"] = "SELL_SWING"
    else:
        ema50 = _pick_col(feat, ["EMA_50"])
        ema200 = _pick_col(feat, ["EMA_200"])
        if ema50 and ema200:
            up = pd.to_numeric(feat[ema50], errors="coerce") > pd.to_numeric(feat[ema200], errors="coerce")
            dn = pd.to_numeric(feat[ema50], errors="coerce") < pd.to_numeric(feat[ema200], errors="coerce")

            cand_long = up & feat["GATE_LONG"]
            cand_short = dn & feat["GATE_SHORT"]

            feat.loc[cand_long, "SIG_CAND"] = "BUY_SWING"
            feat.loc[(feat["SIG_CAND"] == "HOLD") & cand_short, "SIG_CAND"] = "SELL_SWING"

    # ---------------- ENTRIES/EXITS + CONSUMED SIGNALS ----------------
    feat, open_trade = _build_positional_entries_exits_and_signals(
        feat=feat,
        candidate_col="SIG_CAND",
        max_hold_bars=MAX_HOLD_BARS,
    )

    feat["POS_EXIT_WORD"] = np.where(
        pd.to_numeric(feat["POS_EXIT"], errors="coerce").fillna(0).astype(int) == 1,
        "EXIT",
        ""
    )

    # ---------------- REALIZED TRADES ----------------
    trades_df = _calc_realized_trades_from_events(
        feat=feat,
        entry_col="POS_ENTRY",
        exit_col="POS_EXIT",
        dir_col="POS_DIR",
        reason_entry_col="POS_REASON_ENTRY",
        reason_exit_col="POS_REASON_EXIT",
        investment_rs=POS_INVESTMENT_RS,
        price_col="close",
        bar_minutes=BAR_MINUTES,
    )

    # ---------------- OUTPUT FILES (minimal changes) ----------------
    feat["EXISTS_15m"] = exists_15m
    feat["EXISTS_1h"] = exists_1h
    feat["EXISTS_1d"] = exists_1d
    feat["EXISTS_1w"] = exists_1w

    summary_cols = [
        "close",
        "SIG_POS",
        "SIG_CAND",
        "POS_ENTRY", "POS_EXIT", "POS_EXIT_WORD", "POS_DIR",
        "POS_REASON_ENTRY", "POS_REASON_EXIT",
        "MACRO_BIAS", "PREMIUM_PCT", "FAIR_PRICE",
        "EVENT_RISK", "DISTORTION_DAY",
        "GATE_LONG", "GATE_SHORT",
        "EXISTS_15m", "EXISTS_1h", "EXISTS_1d", "EXISTS_1w",
    ]

    # only include macro columns if macros enabled
    if USE_MACROS:
        for nm in macro_names:
            if nm in feat.columns and nm not in summary_cols:
                summary_cols.append(nm)
            ex = f"EXISTS_{nm}"
            if ex in feat.columns and ex not in summary_cols:
                summary_cols.append(ex)

    summary = feat.reset_index().rename(columns={"ts": "date"})
    keep = ["date"] + [c for c in summary_cols if c in summary.columns]
    summary = summary.loc[:, keep].copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    for c, nd in {"close": 2, "MACRO_BIAS": 2, "PREMIUM_PCT": 2, "FAIR_PRICE": 2}.items():
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").round(nd)

    out_path = Path(OUT_DIR) / f"{tkr}_signals_summary_15m.csv"
    _ensure_dir(str(out_path.parent))
    summary.to_csv(out_path, index=False)

    trades_pos_path = Path(OUT_DIR) / f"{tkr}_trades_positional.csv"
    if not trades_df.empty:
        trades_df.to_csv(trades_pos_path, index=False)
    else:
        pd.DataFrame(columns=[
            "trade_id","dir","entry_ts","exit_ts","entry_price","exit_price","qty","investment_rs",
            "pnl_rs","ret_pct","holding_bars","holding_minutes","reason_entry","reason_exit"
        ]).to_csv(trades_pos_path, index=False)

    open_path = Path(OUT_DIR) / f"{tkr}_open_trade.csv"
    if open_trade:
        pd.DataFrame([open_trade]).to_csv(open_path, index=False)
    else:
        pd.DataFrame(columns=[
            "dir","entry_ts","entry_price","last_ts","last_price","qty","investment_rs",
            "unrealized_pnl_rs","holding_bars","holding_minutes"
        ]).to_csv(open_path, index=False)

    invested_closed = float(pd.to_numeric(trades_df.get("investment_rs", 0), errors="coerce").fillna(0).sum()) if not trades_df.empty else 0.0
    realized_pnl = float(pd.to_numeric(trades_df.get("pnl_rs", 0), errors="coerce").fillna(0).sum()) if not trades_df.empty else 0.0
    n_closed = int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0

    has_open = bool(open_trade is not None)
    invested_open = float(open_trade["investment_rs"]) if has_open else 0.0
    unrealized = float(open_trade["unrealized_pnl_rs"]) if has_open else 0.0

    pos_dir = pd.to_numeric(feat.get("POS_DIR", 0), errors="coerce").fillna(0).astype(int)
    max_cap_used = float((pos_dir.ne(0).astype(int) * POS_INVESTMENT_RS).max()) if len(pos_dir) else 0.0

    last_ts = str(feat.index.max()) if len(feat.index) else ""

    return StrategyResult(
        ticker=tkr,
        out_path=str(out_path),
        last_ts=last_ts,
        rows=len(summary),

        n_closed_trades=n_closed,
        invested_closed=invested_closed,
        realized_pnl=realized_pnl,

        has_open_trade=has_open,
        invested_open=invested_open,
        unrealized_pnl=unrealized,
        open_trade=open_trade,

        max_capital_used=max_cap_used,
    )



def _load_etf_universe() -> list[str]:
    try:
        from etf_filtered_etfs_all import selected_etfs  # type: ignore
        etfs = sorted({str(x).strip().upper() for x in selected_etfs})
        if etfs:
            return etfs
    except Exception as e:
        print(f"[WARN] Could not import selected_etfs from etf_filtered_etfs_all.py: {e}")
    return ["GOLDBEES"]


def _print_ticker_summary(res: StrategyResult) -> None:
    print("\n================ RUN SUMMARY (POSITIONAL ONLY, 15m) ================")
    print(f"TICKER: {res.ticker}")
    print(f"  Closed trades      : {res.n_closed_trades}")
    print(f"  Invested (closed)  : {_fmt_rs(res.invested_closed)}  (₹{POS_INVESTMENT_RS:,.0f} per trade)")
    print(f"  Profit/Loss (real) : {_fmt_rs(res.realized_pnl)}  | ROI: {_fmt_pct(res.realized_pnl, res.invested_closed)}")
    print(f"  Max capital used   : {_fmt_rs(res.max_capital_used)}")

    if res.has_open_trade and res.open_trade:
        ot = res.open_trade
        print("  ---- OPEN / UNEXITED TRADE ----")
        print(f"  Dir               : {ot.get('dir','')}")
        print(f"  Entry TS          : {ot.get('entry_ts','')}")
        print(f"  Entry Price       : {ot.get('entry_price',''):.2f}")
        print(f"  Last TS           : {ot.get('last_ts','')}")
        lp = ot.get("last_price", np.nan)
        try:
            lp_s = f"{float(lp):.2f}"
        except Exception:
            lp_s = "NA"
        print(f"  Last Price        : {lp_s}")
        print(f"  Invested (open)   : {_fmt_rs(res.invested_open)}")
        print(f"  Unrealized P&L    : {_fmt_rs(res.unrealized_pnl)}")
        denom = res.invested_open if res.invested_open != 0 else 0.0
        print(f"  Unrealized ROI    : {_fmt_pct(res.unrealized_pnl, denom)}")
        print(f"  Holding bars      : {ot.get('holding_bars',0)}  (~{ot.get('holding_minutes',0)/60.0:.1f} hours)")
    else:
        print("  Open trade         : None")


def main():
    _ensure_dir(OUT_DIR)
    tickers = _load_etf_universe()

    print(f"[INFO] Universe size: {len(tickers)} ETFs")
    print(f"[INFO] Base timeframe: {BASE_TF} ({BAR_MINUTES:.0f} minutes/bar), MAX_HOLD_BARS={MAX_HOLD_BARS}")

    ok = 0
    failed = 0

    # overall totals
    total_closed_trades = 0
    total_invested_closed = 0.0
    total_realized_pnl = 0.0

    total_open_positions = 0
    total_invested_open = 0.0
    total_unrealized_pnl = 0.0

    for t in tickers:
        try:
            res = build_features_and_signals(t)
            ok += 1
            print(f"[OK] {res.ticker} -> {res.out_path} | rows={res.rows} | closed_trades={res.n_closed_trades} | last_ts={res.last_ts}")
            _print_ticker_summary(res)

            total_closed_trades += res.n_closed_trades
            total_invested_closed += res.invested_closed
            total_realized_pnl += res.realized_pnl

            if res.has_open_trade:
                total_open_positions += 1
                total_invested_open += res.invested_open
                total_unrealized_pnl += res.unrealized_pnl

        except Exception as e:
            failed += 1
            print(f"[FAIL] {t}: {e}")

    # net/ending value style totals
    total_profit = total_realized_pnl  # realized only
    net_investment_value = (total_invested_closed + total_invested_open) + (total_realized_pnl + total_unrealized_pnl)

    print("\n================ OVERALL SUMMARY ================")
    print(f"OK                         : {ok}")
    print(f"FAILED                     : {failed}")
    print(f"Closed trades (total)      : {total_closed_trades}")
    print(f"Total invested (closed)    : {_fmt_rs(total_invested_closed)}")
    print(f"Total profit (realized)    : {_fmt_rs(total_profit)}  | ROI: {_fmt_pct(total_profit, total_invested_closed)}")
    print(f"Open positions             : {total_open_positions}")
    print(f"Total invested (open)      : {_fmt_rs(total_invested_open)}")
    print(f"Total P&L (unrealized)     : {_fmt_rs(total_unrealized_pnl)}")
    print(f"NET INVESTMENT / END VALUE : {_fmt_rs(net_investment_value)}")


if __name__ == "__main__":
    main()
