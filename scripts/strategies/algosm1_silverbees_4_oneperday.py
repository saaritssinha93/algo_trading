# -*- coding: utf-8 -*-
"""
SILVERBEES strategy engine (intraday + positional) using saved indicator CSVs

WHAT THIS VERSION DOES (as requested)
- Intraday:
  * At most ONE entry per day (BUY or SELL). After exit, no re-entry that day.
  * Target = +1% (long) / -1% (short). SL disabled (0% by default).
  * ALWAYS EOD exit on the last 5m bar of the day if still in position.
  * Output prints "EXIT" as a word in INTRA_EXIT_WORD when exit occurs.

- Positional:
  * No new entry while a positional position is open.
  * When flat, at most ONE entry per day (even if swing signal repeats).
  * Target = +10% (long) / -10% (short). SL disabled (0% by default).

- Signals are ENTRY EVENTS (not continuous state):
  * ORB/TREND/SWING use rising-edge (first bar that turns true within the day).
  * MR still has its own per-day limiter.

- Keeps EXISTS_* flags for ETF TF files + macro files.

- NEW: P&L calculators (as requested)
  * Intraday investment per trade: ₹10,000
  * Positional investment per trade: ₹100,000
  * Produces:
      - Bar-level realized P&L columns (P&L booked on exit bar)
      - Cumulative P&L columns
      - Per-trade CSVs:
          etf_signals/{TICKER}_trades_intraday.csv
          etf_signals/{TICKER}_trades_positional.csv

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

# --- Investment amounts (used for P&L + capital-at-risk summary) ---
INTRA_INVESTMENT_RS = 10000
POS_INVESTMENT_RS   = 100000


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

# --- MR spam control ---
MR_MAX_SIGNALS_PER_DAY = 3
MR_MIN_GAP_BARS = 6

# --- Intraday TREND_PULLBACK signal controls ---
ENABLE_TREND_PULLBACK = True
PULLBACK_ATR_MULT = 0.35
TREND_EMA_FAST_CANDIDATES = ["EMA_20", "EMA_20_15m", "EMA_50", "EMA_50_15m"]
TREND_EMA_SLOW_CANDIDATES = ["EMA_50", "EMA_50_15m", "EMA_200", "EMA_200_15m"]

# ---------------- TARGET CONFIG (PERCENT TARGETS) ----------------

# Intraday exits
INTRA_TP_PCT = 0.5    # +1% profit target (long); -1% target (short)
INTRA_SL_PCT = 0   # 0 disables SL (kept so you can change later)

# Positional exits
POS_TP_PCT = 10.0     # +10% target (long); -10% target (short)
POS_SL_PCT = 0.0      # 0 disables SL (kept so you can change later)

# ---------------- P&L CONFIG (NEW) ----------------
INTRADAY_INVESTMENT_RS = 10_000.0
POSITIONAL_INVESTMENT_RS = 100_000.0

# ---- Relax intraday trade frequency ----
INTRA_MAX_TRADES_PER_DAY = 2        # set 1 to revert to old behavior
INTRA_REENTRY_COOLDOWN_BARS = 3     # wait 3 bars (15 min) after exit before next entry

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

def _edge_per_day(mask: pd.Series) -> pd.Series:
    """Rising edge within each day: True only on first bar of a new True-run."""
    if mask.empty:
        return mask
    mask = mask.fillna(False).astype(bool)
    d = mask.index.date
    prev = mask.groupby(d).shift(1).fillna(False)
    return mask & (~prev)

def _is_buy_sig(x: str) -> bool:
    return isinstance(x, str) and x.startswith("BUY")

def _is_sell_sig(x: str) -> bool:
    return isinstance(x, str) and x.startswith("SELL")

# ---------------- ENTRY/EXIT ENGINES (TARGETS) ----------------

def _build_intraday_entries_exits_one_trade_per_day_pct(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday (RELAXED):
    - Up to INTRA_MAX_TRADES_PER_DAY per day.
    - Exit on % target, or EOD.
    - SL disabled if INTRA_SL_PCT <= 0.
    - Re-entry allowed after an exit, but only after cooldown bars.
    """
    idx = feat.index
    day = idx.date

    close = pd.to_numeric(feat["close"], errors="coerce").values
    sig = feat["SIG_INTRA"].astype(str).values

    intra_dir = np.zeros(len(feat), dtype=int)
    entry = np.zeros(len(feat), dtype=int)
    exit_ = np.zeros(len(feat), dtype=int)
    reason_entry = np.array([""] * len(feat), dtype=object)
    reason_exit = np.array([""] * len(feat), dtype=object)

    unique_days = pd.Series(day).unique()
    for d in unique_days:
        mask = (day == d)
        day_ix = np.where(mask)[0]
        if len(day_ix) == 0:
            continue

        last_bar_i = day_ix[-1]

        pos = 0
        trades_done = 0
        ep = np.nan
        tp = np.nan
        sl = np.nan

        # cooldown tracking: you can re-enter only after this bar index
        next_entry_allowed_i = -10**9

        for i in day_ix:
            intra_dir[i] = pos

            # forced EOD exit
            if i == last_bar_i and pos != 0:
                exit_[i] = 1
                reason_exit[i] = "EOD_EXIT"
                pos = 0
                intra_dir[i] = 0
                ep = np.nan
                tp = np.nan
                sl = np.nan
                # no need to set cooldown since day ends anyway
                continue

            # ---------- ENTRY (only if flat, trades remaining, cooldown passed) ----------
            if pos == 0 and trades_done < INTRA_MAX_TRADES_PER_DAY and i >= next_entry_allowed_i:
                if _is_buy_sig(sig[i]):
                    pos = 1
                    trades_done += 1
                    ep = close[i]
                    tp = ep * (1.0 + INTRA_TP_PCT / 100.0)
                    sl = ep * (1.0 - INTRA_SL_PCT / 100.0) if INTRA_SL_PCT > 0 else np.nan
                    entry[i] = 1
                    reason_entry[i] = sig[i]
                    intra_dir[i] = 1
                    continue

                if _is_sell_sig(sig[i]):
                    pos = -1
                    trades_done += 1
                    ep = close[i]
                    tp = ep * (1.0 - INTRA_TP_PCT / 100.0)
                    sl = ep * (1.0 + INTRA_SL_PCT / 100.0) if INTRA_SL_PCT > 0 else np.nan
                    entry[i] = 1
                    reason_entry[i] = sig[i]
                    intra_dir[i] = -1
                    continue

            # ---------- EXIT ----------
            exited_here = False

            if pos == 1:
                if (INTRA_SL_PCT > 0) and (not np.isnan(sl)) and (close[i] <= sl):
                    exit_[i] = 1
                    reason_exit[i] = "SL_PCT"
                    pos = 0
                    exited_here = True
                elif (not np.isnan(tp)) and (close[i] >= tp):
                    exit_[i] = 1
                    reason_exit[i] = f"TP_{INTRA_TP_PCT:.2f}PCT"
                    pos = 0
                    exited_here = True

            elif pos == -1:
                if (INTRA_SL_PCT > 0) and (not np.isnan(sl)) and (close[i] >= sl):
                    exit_[i] = 1
                    reason_exit[i] = "SL_PCT"
                    pos = 0
                    exited_here = True
                elif (not np.isnan(tp)) and (close[i] <= tp):
                    exit_[i] = 1
                    reason_exit[i] = f"TP_{INTRA_TP_PCT:.2f}PCT"
                    pos = 0
                    exited_here = True

            intra_dir[i] = pos

            # if exited, clear levels + enforce cooldown
            if exited_here:
                ep = np.nan
                tp = np.nan
                sl = np.nan
                next_entry_allowed_i = i + int(INTRA_REENTRY_COOLDOWN_BARS)

    feat["INTRA_DIR"] = intra_dir
    feat["INTRA_ENTRY"] = entry
    feat["INTRA_EXIT"] = exit_
    feat["INTRA_REASON_ENTRY"] = reason_entry
    feat["INTRA_REASON_EXIT"] = reason_exit
    return feat


def _build_positional_entries_exits_one_entry_per_day_when_flat_pct(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Positional:
    - No new entry while position open.
    - When flat, at most 1 entry per day.
    - Exit on % target. SL is disabled if POS_SL_PCT <= 0.
    """
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
                reason_exit[i] = "TP_10PCT"
                pos = 0

        elif pos == -1:
            if (POS_SL_PCT > 0) and (not np.isnan(sl)) and (close[i] >= sl):
                exit_[i] = 1
                reason_exit[i] = "SL_PCT"
                pos = 0
            elif (not np.isnan(tp)) and (close[i] <= tp):
                exit_[i] = 1
                reason_exit[i] = "TP_10PCT"
                pos = 0

        if exit_[i] == 1 and pos == 0:
            ep = np.nan
            tp = np.nan
            sl = np.nan

        # entries (only if flat and not entered today)
        if pos == 0 and (not entered_today):
            if want_long:
                pos = 1
                entered_today = True
                ep = close[i]
                tp = ep * (1.0 + POS_TP_PCT / 100.0)
                sl = ep * (1.0 - POS_SL_PCT / 100.0) if POS_SL_PCT > 0 else np.nan
                entry[i] = 1
                reason_entry[i] = "BUY_SWING"
            elif want_short:
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

# ---------------- P&L ENGINE (NEW) ----------------

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
        # carry cum pnl
        cum_pnl[i] = realized

        # open position on entry
        if entry_flag[i] == 1 and open_pos == 0:
            d = int(direction[i])
            if d == 0:
                # fallback based on reason text, if present
                if reason_entry_col and reason_entry_col in feat.columns:
                    r = str(feat[reason_entry_col].iat[i])
                    d = 1 if _is_buy_sig(r) else (-1 if _is_sell_sig(r) else 0)
            if d != 0 and (not np.isnan(price[i])) and price[i] > 0:
                open_pos = d
                ep = float(price[i])
                q = float(investment_rs) / ep
                tid += 1
                entry_i = i

                entry_price[i] = ep
                qty[i] = q
                trade_id[i] = tid

        # mark trade id while in trade (helps debugging; optional)
        if open_pos != 0:
            trade_id[i] = tid

        # close on exit
        if exit_flag[i] == 1 and open_pos != 0:
            xp = float(price[i]) if (not np.isnan(price[i])) else np.nan
            exit_price[i] = xp

            if (not np.isnan(xp)) and (not np.isnan(ep)) and (not np.isnan(q)):
                pnl = q * (xp - ep) if open_pos == 1 else q * (ep - xp)
                trade_pnl[i] = pnl
                realized += pnl

                # update cum pnl immediately on exit bar
                cum_pnl[i] = realized

                # trade record
                entry_ts = feat.index[entry_i] if entry_i is not None else pd.NaT
                exit_ts = feat.index[i]
                bars = int(i - entry_i) if entry_i is not None else np.nan
                mins = float(bars) * 5.0 if entry_i is not None else np.nan
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
                    "holding_bars_5m": bars,
                    "holding_minutes": mins,
                    "reason_entry": (feat[reason_entry_col].iat[entry_i] if (reason_entry_col and entry_i is not None and reason_entry_col in feat.columns) else ""),
                    "reason_exit": (feat[reason_exit_col].iat[i] if (reason_exit_col and reason_exit_col in feat.columns) else ""),
                })

            # reset
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

    trades_df = pd.DataFrame(trades)
    return feat, trades_df

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

    # opening range
    feat = _opening_range(feat, OR_BARS)

    # TREND_DAY
    adx_col = _pick_col(feat, ["ADX_15m"]) or _pick_col(feat, ["ADX", "ADX_1h"])
    feat["TREND_DAY"] = np.nan if adx_col is None else (pd.to_numeric(feat[adx_col], errors="coerce") >= ADX_TREND_MIN_15M).astype(int)

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

    # VWAP (still used in entries)
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

    # ORB
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

    # VOL_UNIT (used by MR)
    atr_col = _pick_col(feat, ["ATR", "ATR_15m"]) or _pick_like(feat, "atr")
    if atr_col is not None and atr_col in feat.columns and feat[atr_col].notna().any():
        atr = pd.to_numeric(feat[atr_col], errors="coerce")
        feat["VOL_UNIT"] = (atr / (feat["close"] + 1e-12)) * 100.0
    else:
        feat["VOL_UNIT"] = feat["VWAP_DIST_PCT"].rolling(50).std()

    feat["VOL_UNIT"] = pd.to_numeric(feat["VOL_UNIT"], errors="coerce").fillna(VOL_UNIT_FLOOR).clip(lower=VOL_UNIT_FLOOR)

    # MR
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

    # TREND_PULLBACK
    pb_long = pd.Series(False, index=feat.index)
    pb_short = pd.Series(False, index=feat.index)
    if ENABLE_TREND_PULLBACK and has_vwap:
        ema_fast = _pick_col(feat, TREND_EMA_FAST_CANDIDATES) or _pick_like(feat, "ema_20")
        ema_slow = _pick_col(feat, TREND_EMA_SLOW_CANDIDATES) or _pick_like(feat, "ema_50") or _pick_like(feat, "ema_200")
        if ema_fast is not None and ema_slow is not None and ema_fast in feat.columns and ema_slow in feat.columns:
            ef = pd.to_numeric(feat[ema_fast], errors="coerce")
            es = pd.to_numeric(feat[ema_slow], errors="coerce")

            # pullback band: scaled by ATR (price units) if available else close*VOL_UNIT%
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

    # ENTRY EVENT intraday signals (edges)
    orb_long_event = _edge_per_day(feat["GATE_LONG"] & feat["BREAK_LONG"])
    orb_short_event = _edge_per_day(feat["GATE_SHORT"] & feat["BREAK_SHORT"])
    trend_long_event = _edge_per_day(pb_long)
    trend_short_event = _edge_per_day(pb_short)

    feat["SIG_INTRA"] = "HOLD"
    feat.loc[orb_long_event, "SIG_INTRA"] = "BUY_ORB"
    feat.loc[orb_short_event, "SIG_INTRA"] = "SELL_ORB"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & trend_long_event, "SIG_INTRA"] = "BUY_TREND"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & trend_short_event, "SIG_INTRA"] = "SELL_TREND"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & mr_long_entry, "SIG_INTRA"] = "BUY_MR"
    feat.loc[(feat["SIG_INTRA"] == "HOLD") & mr_short_entry, "SIG_INTRA"] = "SELL_MR"

    # ENTRY EVENT positional signals (edges)
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

    # Build entries/exits
    feat = _build_intraday_entries_exits_one_trade_per_day_pct(feat)
    feat = _build_positional_entries_exits_one_entry_per_day_when_flat_pct(feat)

    # Exit words
    feat["INTRA_EXIT_WORD"] = np.where(pd.to_numeric(feat["INTRA_EXIT"], errors="coerce").fillna(0).astype(int) == 1, "EXIT", "")
    feat["POS_EXIT_WORD"] = np.where(pd.to_numeric(feat["POS_EXIT"], errors="coerce").fillna(0).astype(int) == 1, "EXIT", "")

    # P&L calculations
    feat, intraday_trades = _calc_pnl_from_events(
        feat=feat,
        entry_col="INTRA_ENTRY",
        exit_col="INTRA_EXIT",
        dir_col="INTRA_DIR",
        reason_entry_col="INTRA_REASON_ENTRY",
        reason_exit_col="INTRA_REASON_EXIT",
        investment_rs=INTRADAY_INVESTMENT_RS,
        prefix="INTRA",
        price_col="close",
    )

    feat, positional_trades = _calc_pnl_from_events(
        feat=feat,
        entry_col="POS_ENTRY",
        exit_col="POS_EXIT",
        dir_col="POS_DIR",
        reason_entry_col="POS_REASON_ENTRY",
        reason_exit_col="POS_REASON_EXIT",
        investment_rs=POSITIONAL_INVESTMENT_RS,
        prefix="POS",
        price_col="close",
    )

    # add ETF input exists flags (constant)
    feat["EXISTS_5m"] = exists_5m
    feat["EXISTS_15m"] = exists_15m
    feat["EXISTS_1h"] = exists_1h
    feat["EXISTS_1d"] = exists_1d
    feat["EXISTS_1w"] = exists_1w

    # ---------------- OUTPUT ----------------

    # bar-level summary
    summary_cols = [
        "close",
        "SIG_INTRA", "INTRA_ENTRY", "INTRA_EXIT", "INTRA_EXIT_WORD", "INTRA_DIR",
        "INTRA_REASON_ENTRY", "INTRA_REASON_EXIT",
        "INTRA_ENTRY_PRICE", "INTRA_EXIT_PRICE", "INTRA_QTY", "INTRA_TRADE_PNL_RS", "INTRA_CUM_PNL_RS",
        "SIG_POS", "POS_ENTRY", "POS_EXIT", "POS_EXIT_WORD", "POS_DIR",
        "POS_REASON_ENTRY", "POS_REASON_EXIT",
        "POS_ENTRY_PRICE", "POS_EXIT_PRICE", "POS_QTY", "POS_TRADE_PNL_RS", "POS_CUM_PNL_RS",
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

    # rounding
    round_map = {
        "close": 2,
        "MACRO_BIAS": 2,
        "PREMIUM_PCT": 2,
        "FAIR_SILVERBEES": 2,
        "MCX_SILVER_RS_KG": 0,
        "GSR": 2,
        "INTRA_ENTRY_PRICE": 2,
        "INTRA_EXIT_PRICE": 2,
        "INTRA_QTY": 6,
        "INTRA_TRADE_PNL_RS": 2,
        "INTRA_CUM_PNL_RS": 2,
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

    # per-trade CSVs
    trades_intra_path = Path(OUT_DIR) / f"{ticker}_trades_intraday.csv"
    trades_pos_path = Path(OUT_DIR) / f"{ticker}_trades_positional.csv"
    if not intraday_trades.empty:
        intraday_trades.to_csv(trades_intra_path, index=False)
    else:
        # create empty header if none
        pd.DataFrame(columns=["trade_id","dir","entry_ts","exit_ts","entry_price","exit_price","qty","investment_rs","pnl_rs","ret_pct","holding_bars_5m","holding_minutes","reason_entry","reason_exit"]).to_csv(trades_intra_path, index=False)

    if not positional_trades.empty:
        positional_trades.to_csv(trades_pos_path, index=False)
    else:
        pd.DataFrame(columns=["trade_id","dir","entry_ts","exit_ts","entry_price","exit_price","qty","investment_rs","pnl_rs","ret_pct","holding_bars_5m","holding_minutes","reason_entry","reason_exit"]).to_csv(trades_pos_path, index=False)

    last_ts = str(feat.index.max())
    return StrategyResult(ticker=ticker, out_path=str(out_path), last_ts=last_ts, rows=len(summary))


def _print_run_summary(ticker: str) -> None:
    """Pretty-print trade counts, invested capital, P&L, and max capital at risk."""

    intra_fp = Path(OUT_DIR) / f"{ticker}_trades_intraday.csv"
    pos_fp = Path(OUT_DIR) / f"{ticker}_trades_positional.csv"
    bars_fp = Path(OUT_DIR) / f"{ticker}_signals_summary_5m.csv"

    # Trades tables
    intra = pd.read_csv(intra_fp) if intra_fp.exists() else pd.DataFrame()
    pos = pd.read_csv(pos_fp) if pos_fp.exists() else pd.DataFrame()

    def _safe_sum(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    n_intra = int(len(intra))
    n_pos = int(len(pos))

    invested_intra = _safe_sum(intra, "investment_rs")
    invested_pos = _safe_sum(pos, "investment_rs")
    pnl_intra = _safe_sum(intra, "pnl_rs")
    pnl_pos = _safe_sum(pos, "pnl_rs")

    invested_total = invested_intra + invested_pos
    pnl_total = pnl_intra + pnl_pos
    ending_total = invested_total + pnl_total

    # Max capital at risk (based on bar-level positions)
    max_at_risk = np.nan
    if bars_fp.exists():
        bars = pd.read_csv(bars_fp)
        intra_dir = pd.to_numeric(bars.get("INTRA_DIR", 0), errors="coerce").fillna(0).astype(int)
        pos_dir = pd.to_numeric(bars.get("POS_DIR", 0), errors="coerce").fillna(0).astype(int)
        at_risk = (intra_dir.ne(0).astype(int) * 10000) + (pos_dir.ne(0).astype(int) * 100000)
        max_at_risk = float(at_risk.max()) if len(at_risk) else 0.0
    else:
        max_at_risk = float(INTRA_INVESTMENT_RS + POS_INVESTMENT_RS)

    def _fmt_rs(x: float) -> str:
        return f"₹{x:,.0f}"

    def _fmt_pct(pnl: float, invested: float) -> str:
        if invested == 0:
            return "0.00%"
        return f"{(pnl / invested) * 100.0:.2f}%"

    print("\n================ RUN SUMMARY ================")
    print("INTRADAY")
    print(f"  Trades           : {n_intra}")
    print(f"  Invested (sum)   : {_fmt_rs(invested_intra)}  (₹{INTRA_INVESTMENT_RS:,.0f} per trade)")
    print(f"  Profit/Loss      : {_fmt_rs(pnl_intra)}  | ROI: {_fmt_pct(pnl_intra, invested_intra)}")
    print(f"  Ending Value     : {_fmt_rs(invested_intra + pnl_intra)}")

    print("\nPOSITIONAL")
    print(f"  Trades           : {n_pos}")
    print(f"  Invested (sum)   : {_fmt_rs(invested_pos)}  (₹{POS_INVESTMENT_RS:,.0f} per trade)")
    print(f"  Profit/Loss      : {_fmt_rs(pnl_pos)}  | ROI: {_fmt_pct(pnl_pos, invested_pos)}")
    print(f"  Ending Value     : {_fmt_rs(invested_pos + pnl_pos)}")

    print("\nCOMBINED")
    print(f"  Total trades     : {n_intra + n_pos}")
    print(f"  Total invested   : {_fmt_rs(invested_total)}")
    print(f"  Total P&L        : {_fmt_rs(pnl_total)}  | ROI: {_fmt_pct(pnl_total, invested_total)}")
    print(f"  Net investment   : {_fmt_rs(ending_total)}")
    print(f"  Max capital used : {_fmt_rs(max_at_risk)}")


def main():
    ticker = "SILVERBEES"
    res = build_features_and_signals(ticker)
    print(f"[OK] {res.ticker} -> {res.out_path} | rows={res.rows} | last_ts={res.last_ts}")
    print(f"[OK] Trades written: {Path(OUT_DIR) / (ticker + '_trades_intraday.csv')}")
    print(f"[OK] Trades written: {Path(OUT_DIR) / (ticker + '_trades_positional.csv')}")

    # Print run-level summary (trades, invested, P&L)
    _print_run_summary(ticker)


if __name__ == "__main__":
    main()
