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

    # --- Slippage & commission ---
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

    # --- Volume filter ---
    use_volume_filter: bool = True
    volume_sma_period: int = 20
    volume_min_ratio: float = 1.2   # impulse bar vol >= 1.2x avg volume

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

    # Trailing stop (activates after BE trigger, trails by trail_pct from best price)
    enable_trailing_stop: bool = True
    trail_pct: float = 0.0030  # trail 0.30% from best favorable price after BE trigger

    # Top-N per day
    enable_topn_per_day: bool = True
    topn_per_day: int = 30

    # Long setup toggles
    # Disable the moderate pullback-break setup by default (was a net drag in research)
    enable_setup_a_pullback_c2_break: bool = False
    # --- Output ---
    reports_dir: Path = field(default_factory=lambda: Path(".") / "reports")


def default_short_config(**overrides) -> StrategyConfig:
    """Factory for the SHORT side with optimized v11 defaults.

    Rationale:
    - SL=0.75%: tighter SL cuts losers faster (was 1.0%, most losses happen quickly)
    - TGT=1.2%: better R:R (1.6:1) captures more from momentum moves
    - BE trigger 0.50%: gives trades more room to develop, reduces excessive BE exits
    - Trail 0.30%: locks in partial gains instead of giving back to flat BE
    - Afternoon window added: captures late-day momentum setups (13:00-14:30)
    - topn_per_day=10: allows more quality trades per day
    """
    base = dict(
        side="SHORT",
        stop_pct=0.0075,
        target_pct=0.0120,
        be_trigger_pct=0.0050,
        trail_pct=0.0030,
        mod_impulse_min_atr=0.45,
        rsi_max_short=55.0,
        stochk_max=75.0,
        topn_per_day=10,
        signal_windows=[
            (dtime(9, 15, 0), dtime(11, 30, 0)),
            (dtime(13, 0, 0), dtime(14, 30, 0)),
        ],
    )
    base.update(overrides)
    return StrategyConfig(**base)


def default_long_config(**overrides) -> StrategyConfig:
    """Factory for the LONG side with optimized v11 defaults.

    Rationale:
    - SL=0.75%: tighter SL consistent with SHORT side
    - TGT=1.5%: slightly reduced from 2.0% to increase hit-rate and trade count
    - BE trigger 0.60%: wider trigger so fewer trades exit at flat BE
    - ADX slope relaxed to 0.80: allows trending stocks that are steady (not just surging)
    - topn_per_day=10: allows more quality trades per day
    """
    base = dict(
        side="LONG",
        stop_pct=0.0075,
        target_pct=0.0150,
        be_trigger_pct=0.0060,
        trail_pct=0.0030,
        adx_slope_min=0.80,
        mod_impulse_min_atr=0.30,
        rsi_min_long=45.0,
        stochk_min=25.0,
        stochk_max=95.0,
        topn_per_day=10,
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

    # Volume SMA (for volume filter)
    if "volume" in out.columns:
        vol = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
        out["VOL_SMA20"] = vol.rolling(20, min_periods=1).mean()
    else:
        out["VOL_SMA20"] = np.nan

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


# ===========================================================================
# VOLUME FILTER HELPER
# ===========================================================================
def volume_filter_pass(row: pd.Series, cfg: StrategyConfig) -> bool:
    """Check that the impulse bar has above-average volume (confirms institutional participation)."""
    if not cfg.use_volume_filter:
        return True
    vol = float(row.get("volume", 0.0)) if np.isfinite(row.get("volume", np.nan)) else 0.0
    vol_sma = float(row.get("VOL_SMA20", 0.0)) if np.isfinite(row.get("VOL_SMA20", np.nan)) else 0.0
    if vol_sma <= 0:
        return True  # no volume data available — pass through
    return vol >= (cfg.volume_min_ratio * vol_sma)


# ===========================================================================
# GRAPHICAL OUTPUT — Charts for backtest analysis
# ===========================================================================
def generate_backtest_charts(
    df: pd.DataFrame,
    short_df: pd.DataFrame,
    long_df: pd.DataFrame,
    save_dir: Path,
    ts_label: str = "",
) -> List[str]:
    """Generate and save backtest analysis charts. Returns list of saved file paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[WARN] matplotlib not installed — skipping chart generation.")
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    if df.empty:
        return saved

    d = df.copy()
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d = d.sort_values("trade_date").reset_index(drop=True)

    # --- 1. Cumulative Equity Curve (by side + combined) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    # Combined
    cum_all = d["pnl_pct"].cumsum()
    ax.plot(range(len(cum_all)), cum_all.values, label="Combined", color="black", linewidth=2)
    # SHORT
    if not short_df.empty:
        s = short_df.copy()
        s["pnl_pct"] = pd.to_numeric(s["pnl_pct"], errors="coerce").fillna(0.0)
        cum_s = s["pnl_pct"].cumsum()
        ax.plot(range(len(cum_s)), cum_s.values, label="SHORT", color="red", alpha=0.7)
    # LONG
    if not long_df.empty:
        l = long_df.copy()
        l["pnl_pct"] = pd.to_numeric(l["pnl_pct"], errors="coerce").fillna(0.0)
        cum_l = l["pnl_pct"].cumsum()
        ax.plot(range(len(cum_l)), cum_l.values, label="LONG", color="green", alpha=0.7)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (%)")
    ax.set_title("Equity Curve — Cumulative PnL % (net)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    p = save_dir / f"equity_curve_{ts_label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # --- 2. Daily PnL by Date ---
    fig, ax = plt.subplots(figsize=(14, 5))
    daily_pnl = d.groupby("trade_date")["pnl_pct"].sum()
    colors = ["green" if v >= 0 else "red" for v in daily_pnl.values]
    ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.7, width=0.8)
    ax.set_xlabel("Trading Day (index)")
    ax.set_ylabel("Daily Net PnL (%)")
    ax.set_title("Daily PnL — Sum of All Trades per Day")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    p = save_dir / f"daily_pnl_{ts_label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # --- 3. Outcome Distribution (pie chart) ---
    if "outcome" in d.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        outcome_colors = {"TARGET": "#2ecc71", "SL": "#e74c3c", "BE": "#f39c12", "EOD": "#3498db"}

        for ax, (title, sub) in zip(axes, [
            ("Combined", d),
            ("SHORT", d[d["side"] == "SHORT"]),
            ("LONG", d[d["side"] == "LONG"]),
        ]):
            if sub.empty:
                ax.set_title(f"{title} (no data)")
                continue
            counts = sub["outcome"].value_counts()
            labels = [f"{k}\n({v}, {v/len(sub)*100:.1f}%)" for k, v in counts.items()]
            cols = [outcome_colors.get(k, "gray") for k in counts.index]
            ax.pie(counts.values, labels=labels, colors=cols, startangle=90)
            ax.set_title(f"{title} Outcomes (n={len(sub)})")

        fig.suptitle("Trade Outcome Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        p = save_dir / f"outcome_distribution_{ts_label}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # --- 4. PnL Distribution Histogram ---
    fig, ax = plt.subplots(figsize=(12, 5))
    pnl_vals = d["pnl_pct"].values
    ax.hist(pnl_vals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Breakeven")
    ax.axvline(x=np.mean(pnl_vals), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean={np.mean(pnl_vals):.3f}%")
    ax.set_xlabel("PnL per Trade (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution — All Trades (net)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    p = save_dir / f"pnl_distribution_{ts_label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # --- 5. Trades per Day Histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    trades_per_day = d.groupby("trade_date").size()
    ax.hist(trades_per_day.values, bins=range(0, int(trades_per_day.max()) + 2),
            color="teal", edgecolor="white", alpha=0.8, align="left")
    ax.axvline(x=trades_per_day.mean(), color="red", linestyle="--",
               label=f"Avg={trades_per_day.mean():.1f}/day")
    ax.set_xlabel("Trades per Day")
    ax.set_ylabel("Number of Days")
    ax.set_title("Trade Frequency Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    p = save_dir / f"trades_per_day_{ts_label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # --- 6. Drawdown Chart ---
    fig, ax = plt.subplots(figsize=(14, 5))
    cum = d["pnl_pct"].cumsum().values
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    ax.fill_between(range(len(drawdown)), drawdown, color="red", alpha=0.3)
    ax.plot(range(len(drawdown)), drawdown, color="red", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown from Peak — Cumulative PnL")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    p = save_dir / f"drawdown_{ts_label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # --- 7. Win Rate by Hour of Day ---
    if "entry_time_ist" in d.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        d["entry_time_ist"] = pd.to_datetime(d["entry_time_ist"], errors="coerce")
        d["entry_hour"] = d["entry_time_ist"].dt.hour
        hourly = d.groupby("entry_hour").agg(
            count=("pnl_pct", "size"),
            wins=("pnl_pct", lambda x: (x > 0).sum()),
            avg_pnl=("pnl_pct", "mean"),
        )
        hourly["win_rate"] = hourly["wins"] / hourly["count"] * 100

        ax2 = ax.twinx()
        bars = ax.bar(hourly.index, hourly["count"], color="steelblue", alpha=0.6, label="Trade Count")
        line = ax2.plot(hourly.index, hourly["win_rate"], color="green", marker="o",
                        linewidth=2, label="Win Rate %")
        ax.set_xlabel("Hour of Day (IST)")
        ax.set_ylabel("Number of Trades")
        ax2.set_ylabel("Win Rate (%)")
        ax.set_title("Trade Activity & Win Rate by Hour")
        ax.set_xticks(hourly.index)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        p = save_dir / f"hourly_analysis_{ts_label}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # --- 8. Monthly Returns Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 4))
    d["year_month"] = d["trade_date"].dt.to_period("M")
    monthly_pnl = d.groupby("year_month")["pnl_pct"].sum()
    colors_m = ["green" if v >= 0 else "red" for v in monthly_pnl.values]
    x_labels = [str(p) for p in monthly_pnl.index]
    ax.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors_m, alpha=0.7)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Monthly Net PnL (%)")
    ax.set_title("Monthly Return Summary")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    p = save_dir / f"monthly_returns_{ts_label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    return saved
