# -*- coding: utf-8 -*-
"""
avwap_common.py — Shared infrastructure for AVWAP v11 Long + Short strategies
=============================================================================

Contains:
- Unified Trade dataclass
- StrategyConfig (all tuneable parameters in one place)
- Indicator computations (ATR, RSI, Stochastic, ADX, EMA, AVWAP)
- IO helpers (parquet reader, ticker listing)
- Session / time-window helpers
- Quality score computation
- Backtest metrics (Sharpe, drawdown, profit factor, etc.)
- Slippage + commission model
"""

from __future__ import annotations

import os
import glob
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)


# ===========================================================================
# UNIFIED CONFIG — replaces all module-level globals
# ===========================================================================
@dataclass
class StrategyConfig:
    """
    All tuneable parameters for one side (LONG or SHORT).
    Instantiate separate configs for each side, or share common defaults.
    """

    # --- Direction ---
    side: str = "SHORT"  # "SHORT" or "LONG"

    # --- Data paths ---
    dir_15m: str = "stocks_indicators_15min_eq"
    end_15m: str = "_stocks_indicators_15min.parquet"
    parquet_engine: str = "pyarrow"

    # --- Risk ---
    stop_pct: float = 0.0100
    target_pct: float = 0.0100

    # --- Slippage & commission (NEW) ---
    slippage_pct: float = 0.0005   # 5 bps one-way
    commission_pct: float = 0.0003  # 3 bps round-trip (STT + brokerage approx)

    # --- Impulse thresholds ---
    mod_impulse_min_atr: float = 0.45
    mod_impulse_max_atr: float = 1.00
    huge_impulse_min_atr: float = 1.60
    huge_impulse_min_range_atr: float = 2.00
    close_near_extreme_max: float = 0.25  # close-near-low (short) / close-near-high (long)

    # Pullback candle
    small_counter_max_atr: float = 0.20

    # Entry buffer
    buffer_abs: float = 0.05
    buffer_pct: float = 0.0002

    # --- Session ---
    session_start: dtime = field(default_factory=lambda: dtime(9, 15, 0))
    session_end: dtime = field(default_factory=lambda: dtime(14, 30, 0))

    # --- Time windows (Option E) ---
    use_time_windows: bool = True
    signal_windows: List[Tuple[dtime, dtime]] = field(
        default_factory=lambda: [
            (dtime(9, 15, 0), dtime(11, 30, 0)),
            (dtime(13, 0, 0), dtime(14, 30, 0)),
        ]
    )

    # --- Trend filter (Option A) ---
    adx_min: float = 25.0
    adx_slope_min: float = 1.25
    # SHORT uses rsi_max; LONG uses rsi_min — store both, strategy picks
    rsi_max_short: float = 55.0
    rsi_min_long: float = 45.0
    stochk_max: float = 75.0
    stochk_min: float = 25.0

    # ATR% volatility filter
    use_atr_pct_filter: bool = True
    atr_pct_min: float = 0.0020

    # --- AVWAP rules (Option B) ---
    require_avwap_rule: bool = True
    avwap_touch: bool = True
    avwap_min_consec_closes: int = 2
    avwap_mode: str = "any"         # "any" or "both"
    avwap_dist_atr_mult: float = 0.25

    # --- Quality upgrades ---
    max_trades_per_ticker_per_day: int = 1
    require_entry_close_confirm: bool = True
    min_bars_left_after_entry: int = 4

    # Breakeven
    enable_breakeven: bool = True
    be_trigger_pct: float = 0.0040
    be_pad_pct: float = 0.0001

    # Top-N per day
    enable_topn_per_day: bool = True
    topn_per_day: int = 30


    # Long setup toggles
    # Disable the moderate pullback-break setup by default (was a net drag in research)
    enable_setup_a_pullback_c2_break: bool = False
    # --- Output ---
    reports_dir: Path = field(default_factory=lambda: Path(".") / "reports")


def default_short_config(**overrides) -> StrategyConfig:
    """Factory for the SHORT side with typical v11 defaults."""
    base = dict(
        side="SHORT",
        stop_pct=0.0100,
        target_pct=0.0100,
        mod_impulse_min_atr=0.45,
        rsi_max_short=55.0,
        stochk_max=75.0,
        topn_per_day=8,
        signal_windows=[(dtime(9, 15, 0), dtime(11, 30, 0))],
    )
    base.update(overrides)
    return StrategyConfig(**base)


def default_long_config(**overrides) -> StrategyConfig:
    """Factory for the LONG side with typical v11 defaults."""
    base = dict(
        side="LONG",
        stop_pct=0.0100,
        target_pct=0.0200,
        mod_impulse_min_atr=0.30,
        rsi_min_long=45.0,
        stochk_min=25.0,
        stochk_max=95.0,
        topn_per_day=8,
    )
    base.update(overrides)
    return StrategyConfig(**base)


# ===========================================================================
# UNIFIED TRADE DATACLASS
# ===========================================================================
@dataclass
class Trade:
    trade_date: str
    ticker: str
    side: str              # "SHORT" or "LONG"
    setup: str
    impulse_type: str
    signal_time_ist: pd.Timestamp
    entry_time_ist: pd.Timestamp
    entry_price: float
    sl_price: float
    target_price: float
    exit_time_ist: pd.Timestamp
    exit_price: float
    outcome: str           # TARGET / SL / BE / EOD
    pnl_pct: float         # after slippage + commission
    pnl_pct_gross: float   # before costs

    # Diagnostics (present for both sides now)
    adx_signal: float = 0.0
    rsi_signal: float = 0.0
    stochk_signal: float = 0.0
    avwap_dist_atr_signal: float = 0.0
    ema20_gap_atr_signal: float = 0.0
    atr_pct_signal: float = 0.0
    quality_score: float = 0.0


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])


# ===========================================================================
# IO HELPERS
# ===========================================================================
def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Parquet support requires 'pyarrow' (pip install pyarrow)."
        ) from e


def read_15m_parquet(path: str, engine: str = "pyarrow") -> pd.DataFrame:
    _require_pyarrow()
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path, engine=engine)
    if "date" not in df.columns:
        return pd.DataFrame()

    dt = pd.to_datetime(df["date"], errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    dt = dt.dt.tz_convert(IST)

    df = df.copy()
    df["date"] = dt
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def list_tickers_15m(dir_15m: str, end_15m: str) -> List[str]:
    pattern = os.path.join(dir_15m, f"*{end_15m}")
    files = glob.glob(pattern)
    out = []
    for f in files:
        base = os.path.basename(f)
        if base.endswith(end_15m):
            out.append(base[: -len(end_15m)].upper())
    return sorted(set(out))


# ===========================================================================
# SESSION / TIME-WINDOW HELPERS
# ===========================================================================
def in_session(ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    t = ts.tz_convert(IST).time()
    return cfg.session_start <= t <= cfg.session_end


def in_signal_window(ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    if not cfg.use_time_windows:
        return True
    t = ts.tz_convert(IST).time()
    for a, b in cfg.signal_windows:
        if a <= t <= b:
            return True
    return False


def entry_buffer(price: float, cfg: StrategyConfig) -> float:
    return max(float(cfg.buffer_abs), float(price) * float(cfg.buffer_pct))


# ===========================================================================
# INDICATORS  (computed once per ticker on the full multi-day series)
# ===========================================================================
def ensure_ema(df: pd.DataFrame, span: int, col_close: str = "close") -> pd.Series:
    close = pd.to_numeric(df[col_close], errors="coerce")
    return close.ewm(span=span, adjust=False).mean()


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(14).mean()


def compute_rsi14(df: pd.DataFrame, col_close: str = "close") -> pd.Series:
    close = pd.to_numeric(df[col_close], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_stoch_14_3(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    ll = low.rolling(14).min()
    hh = high.rolling(14).max()
    denom = (hh - ll).replace(0, np.nan)

    k = 100.0 * (close - ll) / denom
    d = k.rolling(3).mean()
    return k, d


def compute_adx14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)

    plus_di = 100.0 * (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
        / atr
    )
    minus_di = 100.0 * (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
        / atr
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / 14, adjust=False).mean()


def compute_day_avwap(df_day: pd.DataFrame) -> pd.Series:
    """Anchored VWAP for a single intraday session (anchored at first bar)."""
    high = pd.to_numeric(df_day["high"], errors="coerce")
    low = pd.to_numeric(df_day["low"], errors="coerce")
    close = pd.to_numeric(df_day["close"], errors="coerce")
    vol = pd.to_numeric(df_day.get("volume", 0.0), errors="coerce").fillna(0.0)

    tp = (high + low + close) / 3.0
    pv = tp * vol

    cum_pv = pv.cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def prepare_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Add all indicator columns to the full multi-day series for one ticker.
    Re-uses pre-computed columns from parquet when available.
    """
    out = df.copy()

    # ATR
    if "ATR" in out.columns:
        out["ATR15"] = pd.to_numeric(out["ATR"], errors="coerce")
    else:
        out["ATR15"] = compute_atr14(out)

    # EMAs
    if "EMA_20" in out.columns:
        out["EMA20"] = pd.to_numeric(out["EMA_20"], errors="coerce")
    else:
        out["EMA20"] = ensure_ema(out, 20)

    if "EMA_50" in out.columns:
        out["EMA50"] = pd.to_numeric(out["EMA_50"], errors="coerce")
    else:
        out["EMA50"] = ensure_ema(out, 50)

    # RSI
    if "RSI" in out.columns:
        out["RSI15"] = pd.to_numeric(out["RSI"], errors="coerce")
    else:
        out["RSI15"] = compute_rsi14(out)

    # Stochastic
    if "Stoch_%K" in out.columns:
        out["STOCHK15"] = pd.to_numeric(out["Stoch_%K"], errors="coerce")
        out["STOCHD15"] = pd.to_numeric(
            out.get("Stoch_%D", np.nan), errors="coerce"
        )
    else:
        k, d = compute_stoch_14_3(out)
        out["STOCHK15"] = k
        out["STOCHD15"] = d

    # ADX
    if "ADX" in out.columns:
        out["ADX15"] = pd.to_numeric(out["ADX"], errors="coerce")
    else:
        out["ADX15"] = compute_adx14(out)

    out["day"] = out["date"].dt.tz_convert(IST).dt.date
    return out


# ===========================================================================
# INDICATOR MICRO-CHECKS (used in signal validation)
# ===========================================================================
def twice_increasing(df_day: pd.DataFrame, idx: int, col: str) -> bool:
    if idx < 2:
        return False
    a = float(df_day.at[idx, col])
    b = float(df_day.at[idx - 1, col])
    c = float(df_day.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a > b > c)


def twice_reducing(df_day: pd.DataFrame, idx: int, col: str) -> bool:
    if idx < 2:
        return False
    a = float(df_day.at[idx, col])
    b = float(df_day.at[idx - 1, col])
    c = float(df_day.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a < b < c)


def adx_slope_ok(df_day: pd.DataFrame, idx: int, col: str, min_slope: float) -> bool:
    if idx < 2 or col not in df_day.columns:
        return False
    a = float(df_day.at[idx, col])
    c = float(df_day.at[idx - 2, col])
    return np.isfinite(a) and np.isfinite(c) and ((a - c) >= float(min_slope))


def max_consecutive_true(flags: np.ndarray) -> int:
    best = 0
    run = 0
    for x in flags:
        if bool(x):
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


# ===========================================================================
# SLIPPAGE + COMMISSION MODEL (NEW)
# ===========================================================================
def apply_slippage_and_commission(
    entry_price: float,
    exit_price: float,
    side: str,
    cfg: StrategyConfig,
) -> Tuple[float, float]:
    """
    Returns (adjusted_entry, adjusted_exit) accounting for slippage + commission.

    Slippage: entry is worse by slippage_pct; exit is worse by slippage_pct.
    Commission: deducted as round-trip from exit.

    For SHORT: entry goes DOWN (you sell lower), exit goes UP (you buy higher).
    For LONG:  entry goes UP (you buy higher), exit goes DOWN (you sell lower).
    """
    slip = cfg.slippage_pct
    comm = cfg.commission_pct

    if side == "SHORT":
        adj_entry = entry_price * (1.0 - slip)
        adj_exit = exit_price * (1.0 + slip + comm)
    else:
        adj_entry = entry_price * (1.0 + slip)
        adj_exit = exit_price * (1.0 - slip - comm)

    return adj_entry, adj_exit


def compute_pnl_pct(
    entry_price: float,
    exit_price: float,
    side: str,
    cfg: StrategyConfig,
) -> Tuple[float, float]:
    """
    Returns (net_pnl_pct, gross_pnl_pct).
    Gross = before slippage/commission. Net = after.
    """
    if side == "SHORT":
        gross = (entry_price - exit_price) / entry_price * 100.0
    else:
        gross = (exit_price - entry_price) / entry_price * 100.0

    adj_entry, adj_exit = apply_slippage_and_commission(
        entry_price, exit_price, side, cfg
    )
    if side == "SHORT":
        net = (adj_entry - adj_exit) / adj_entry * 100.0
    else:
        net = (adj_exit - adj_entry) / adj_entry * 100.0

    return float(net), float(gross)


# ===========================================================================
# QUALITY SCORE
# ===========================================================================
def compute_quality_score_short(
    adx: float,
    avwap_dist_atr: float,
    ema_gap_atr: float,
    impulse: str,
) -> float:
    adx_n = np.clip((adx - 20.0) / 30.0, 0.0, 1.0)
    av_n = np.clip(avwap_dist_atr / 2.0, 0.0, 1.0)
    ema_n = np.clip(ema_gap_atr / 2.0, 0.0, 1.0)
    imp = 1.0 if impulse == "HUGE" else 0.6
    return float((0.45 * adx_n) + (0.35 * av_n) + (0.10 * ema_n) + (0.10 * imp))


def compute_quality_score_long(
    adx: float,
    adx_slope2: float,
    avwap_dist_atr: float,
    ema_gap_atr: float,
    impulse: str,
) -> float:
    imp_bonus = 0.25 if impulse == "HUGE" else 0.0
    return float(
        0.04 * adx
        + 0.20 * adx_slope2
        + 1.20 * avwap_dist_atr
        + 0.80 * ema_gap_atr
        + imp_bonus
    )


# ===========================================================================
# BACKTEST METRICS (NEW) — Sharpe, drawdown, profit factor, etc.
# ===========================================================================
@dataclass
class BacktestMetrics:
    total_trades: int = 0
    unique_days: int = 0
    target_count: int = 0
    sl_count: int = 0
    be_count: int = 0
    eod_count: int = 0
    hit_rate_pct: float = 0.0
    sl_rate_pct: float = 0.0
    be_rate_pct: float = 0.0
    eod_rate_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    sum_pnl_pct: float = 0.0
    avg_pnl_pct_gross: float = 0.0
    sum_pnl_pct_gross: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


def compute_backtest_metrics(df: pd.DataFrame) -> BacktestMetrics:
    """Compute comprehensive backtest metrics from a trades DataFrame."""
    m = BacktestMetrics()
    if df.empty:
        return m

    d = df.copy()
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["pnl_pct_gross"] = pd.to_numeric(
        d.get("pnl_pct_gross", d["pnl_pct"]), errors="coerce"
    ).fillna(0.0)

    n = len(d)
    m.total_trades = n
    m.unique_days = int(d["trade_date"].nunique()) if "trade_date" in d.columns else 0

    if "outcome" in d.columns:
        m.target_count = int((d["outcome"] == "TARGET").sum())
        m.sl_count = int((d["outcome"] == "SL").sum())
        m.be_count = int((d["outcome"] == "BE").sum())
        m.eod_count = int((d["outcome"] == "EOD").sum())

    m.hit_rate_pct = (m.target_count / n * 100.0) if n else 0.0
    m.sl_rate_pct = (m.sl_count / n * 100.0) if n else 0.0
    m.be_rate_pct = (m.be_count / n * 100.0) if n else 0.0
    m.eod_rate_pct = (m.eod_count / n * 100.0) if n else 0.0

    pnl = d["pnl_pct"].values
    m.avg_pnl_pct = float(np.nanmean(pnl))
    m.sum_pnl_pct = float(np.nansum(pnl))
    m.avg_pnl_pct_gross = float(np.nanmean(d["pnl_pct_gross"].values))
    m.sum_pnl_pct_gross = float(np.nansum(d["pnl_pct_gross"].values))

    # Profit factor
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) else 0.0
    m.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    m.avg_win_pct = float(np.mean(wins)) if len(wins) else 0.0
    m.avg_loss_pct = float(np.mean(losses)) if len(losses) else 0.0

    # Max drawdown on cumulative equity curve
    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    m.max_drawdown_pct = float(np.max(dd)) if len(dd) else 0.0

    # Sharpe ratio (annualized, assuming ~250 trading days)
    if n > 1:
        daily_mean = float(np.mean(pnl))
        daily_std = float(np.std(pnl, ddof=1))
        m.sharpe_ratio = (
            (daily_mean / daily_std) * math.sqrt(250) if daily_std > 0 else 0.0
        )
    else:
        m.sharpe_ratio = 0.0

    # Sortino ratio (downside deviation only)
    if n > 1:
        downside = pnl[pnl < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        daily_mean = float(np.mean(pnl))
        m.sortino_ratio = (
            (daily_mean / downside_std) * math.sqrt(250)
            if downside_std > 0
            else 0.0
        )
    else:
        m.sortino_ratio = 0.0

    # Calmar ratio
    if m.max_drawdown_pct > 0:
        m.calmar_ratio = float(m.sum_pnl_pct / m.max_drawdown_pct)
    else:
        m.calmar_ratio = float("inf") if m.sum_pnl_pct > 0 else 0.0

    # Consecutive wins/losses
    m.max_consecutive_wins = _max_consecutive(pnl > 0)
    m.max_consecutive_losses = _max_consecutive(pnl < 0)

    return m


def _max_consecutive(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def print_metrics(title: str, m: BacktestMetrics) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(f"Total trades                  : {m.total_trades}")
    print(f"Unique trade days             : {m.unique_days}")
    print(f"TARGET hits                   : {m.target_count}  | hit-rate  = {m.hit_rate_pct:.2f}%")
    print(f"SL hits                       : {m.sl_count}  | sl-rate   = {m.sl_rate_pct:.2f}%")
    print(f"BE exits                      : {m.be_count}  | be-rate   = {m.be_rate_pct:.2f}%")
    print(f"EOD exits                     : {m.eod_count}  | eod-rate  = {m.eod_rate_pct:.2f}%")
    print(f"Avg PnL % (net, per trade)    : {m.avg_pnl_pct:.4f}%")
    print(f"Sum PnL % (net, all trades)   : {m.sum_pnl_pct:.4f}%")
    print(f"Avg PnL % (gross, per trade)  : {m.avg_pnl_pct_gross:.4f}%")
    print(f"Sum PnL % (gross, all trades) : {m.sum_pnl_pct_gross:.4f}%")
    print(f"Profit factor                 : {m.profit_factor:.3f}")
    print(f"Avg winning trade             : {m.avg_win_pct:.4f}%")
    print(f"Avg losing trade              : {m.avg_loss_pct:.4f}%")
    print(f"Max drawdown (cumul PnL %)    : {m.max_drawdown_pct:.4f}%")
    print(f"Sharpe ratio (annualized)     : {m.sharpe_ratio:.3f}")
    print(f"Sortino ratio (annualized)    : {m.sortino_ratio:.3f}")
    print(f"Calmar ratio                  : {m.calmar_ratio:.3f}")
    print(f"Max consecutive wins          : {m.max_consecutive_wins}")
    print(f"Max consecutive losses        : {m.max_consecutive_losses}")
    print("=" * (42 + len(title)))


# ===========================================================================
# TOP-N PER DAY FILTER
# ===========================================================================
def apply_topn_per_day(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    if df.empty or not cfg.enable_topn_per_day or cfg.topn_per_day <= 0:
        return df

    req = {"trade_date", "quality_score", "ticker", "entry_time_ist"}
    if not req.issubset(df.columns):
        return df

    d = df.copy()
    d["quality_score"] = pd.to_numeric(d["quality_score"], errors="coerce").fillna(0.0)
    for c in ["entry_time_ist", "exit_time_ist", "signal_time_ist"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    d = d.sort_values(
        ["trade_date", "quality_score", "ticker", "entry_time_ist"],
        ascending=[True, False, True, True],
    )
    d = (
        d.groupby("trade_date", sort=False, as_index=False)
        .head(int(cfg.topn_per_day))
        .reset_index(drop=True)
    )
    return d
