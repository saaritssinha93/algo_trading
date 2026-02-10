# -*- coding: utf-8 -*-
"""
avwap_combined_runner.py — AVWAP v11 COMBINED LONG + SHORT runner (refactored v2)
==================================================================================

Changes from v1:
1. All outputs saved to */algo_trading/outputs instead of */algo_trading/reports
2. Entry signals still use 15-min data; exit/SL/target tracking uses 5-min data
   from */stocks_indicators_5min_eq/ for higher resolution P&L
3. Significantly expanded charting suite with more detailed & analytical charts
4. Normal Python imports (no importlib hacks)
5. Unified Trade dataclass — both sides produce identical columns
6. Parallel ticker scanning via ProcessPoolExecutor
7. Slippage + commission model baked into P&L
8. Comprehensive backtest metrics (Sharpe, Sortino, Calmar, drawdown, profit factor)
9. All config via StrategyConfig dataclass — no module-level globals
10. Cash-constrained portfolio sim uses itertuples() instead of iterrows()

Usage:
    python -m avwap_v11_refactored.avwap_combined_runner
    # or
    python avwap_v11_refactored/avwap_combined_runner.py
"""

from __future__ import annotations

import heapq
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# ===========================================================================
# CONSOLE OUTPUT TEE (stdout/stderr -> console + outputs/*.txt)
# ===========================================================================
class _Tee:
    """Write to multiple streams (e.g., console + log file)."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False

# Ensure the package is importable when running this file directly
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from avwap_v11_refactored.avwap_common import (
    IST,
    StrategyConfig,
    Trade,
    BacktestMetrics,
    default_short_config,
    default_long_config,
    now_ist,
    trades_to_df,
    apply_topn_per_day,
    compute_backtest_metrics,
    print_metrics,
    read_15m_parquet,
    list_tickers_15m,
    generate_backtest_charts,
)
from avwap_v11_refactored.avwap_short_strategy import (
    scan_all_days_for_ticker as scan_short,
)
from avwap_v11_refactored.avwap_long_strategy import (
    scan_all_days_for_ticker as scan_long,
)


# ===========================================================================
# RUNNER CONFIG (top-level orchestration settings)
# ===========================================================================
POSITION_SIZE_RS_SHORT = 50_000
POSITION_SIZE_RS_LONG = 50_000

# Intraday leverage (margin). Position sizes above are *capital/margin per trade*.
# Notional exposure = capital * leverage. Set leverage=1.0 to disable leverage effects.
INTRADAY_LEVERAGE_SHORT = 5.0
INTRADAY_LEVERAGE_LONG = 5.0

ENABLE_CASH_CONSTRAINED_PORTFOLIO_SIM = False
PORTFOLIO_START_CAPITAL_RS = 1_000_000
DISALLOW_BOTH_SIDES_SAME_TICKER_DAY = False

# Parallelism: set to 1 for serial, >1 for multi-process
MAX_WORKERS = 4

# 5-min data directory for exit resolution
DIR_5MIN = None  # Will be resolved dynamically at runtime


# ===========================================================================
# 5-MINUTE DATA READER
# ===========================================================================
def _resolve_5min_dir() -> Path:
    """
    Resolve the 5-min data directory relative to the algo_trading project root.
    Looks for a directory named 'stocks_indicators_5min_eq' in the data path.
    """
    _script_dir = Path(__file__).resolve().parent
    if _script_dir.name == "avwap_v11_refactored":
        _proj = _script_dir.parent
    else:
        _proj = _script_dir

    # Common locations to search for 5-min data
    candidates = [
        _proj / "data" / "stocks_indicators_5min_eq",
        _proj / "stocks_indicators_5min_eq",
        _proj.parent / "data" / "stocks_indicators_5min_eq",
        _proj.parent / "stocks_indicators_5min_eq",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fallback: return the first candidate (will be created/handled later)
    return candidates[0]


def read_5m_parquet(path: str, engine: str = "pyarrow") -> pd.DataFrame:
    """
    Read a 5-minute parquet file for a ticker.
    Returns empty DataFrame if file not found or read fails.
    """
    try:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p, engine=engine)
        if df.empty:
            return df
        # Ensure datetime index/column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        elif df.index.name == "datetime" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def list_tickers_5m(dir_5m: Path, suffix: str = ".parquet") -> List[str]:
    """List ticker symbols available in the 5-min data directory."""
    if not dir_5m.is_dir():
        return []
    tickers = []
    for f in dir_5m.iterdir():
        if f.name.endswith(suffix):
            tickers.append(f.stem if not suffix else f.name.replace(suffix, ""))
    return sorted(set(tickers))


# ===========================================================================
# 5-MIN EXIT RESOLUTION
# ===========================================================================
def _resolve_exits_5min(
    trades_df: pd.DataFrame,
    dir_5m: Path,
    suffix_5m: str = ".parquet",
    engine: str = "pyarrow",
) -> pd.DataFrame:
    """
    Re-evaluate exit prices, exit times, and outcomes using 5-min data
    for higher-resolution SL/target tracking.

    Entry signals and entry prices remain from 15-min scanning.
    Only the exit side is recalculated at 5-min granularity.
    """
    if trades_df.empty:
        return trades_df

    if not dir_5m.is_dir():
        print(f"[WARN] 5-min data directory not found: {dir_5m}")
        print("[WARN] Falling back to 15-min exit resolution.")
        return trades_df

    df = trades_df.copy()

    # Ensure required columns exist
    required = {"ticker", "side", "entry_price", "entry_time_ist", "stop_price", "target_price"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        print(f"[WARN] Missing columns for 5-min resolution: {missing}")
        return df

    # Convert timestamps
    for c in ["entry_time_ist", "exit_time_ist", "signal_time_ist"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Cache 5-min data per ticker to avoid re-reads
    _cache_5m: Dict[str, pd.DataFrame] = {}

    updated_rows = 0
    total_rows = len(df)

    for idx in df.index:
        ticker = str(df.at[idx, "ticker"])
        side = str(df.at[idx, "side"]).upper()
        entry_price = float(df.at[idx, "entry_price"])
        entry_time = df.at[idx, "entry_time_ist"]
        stop_price = float(df.at[idx, "stop_price"]) if pd.notna(df.at[idx, "stop_price"]) else None
        target_price = float(df.at[idx, "target_price"]) if pd.notna(df.at[idx, "target_price"]) else None

        if pd.isna(entry_time) or stop_price is None or target_price is None:
            continue

        # Load 5-min data for this ticker (cached)
        if ticker not in _cache_5m:
            # Try common naming patterns
            found = False
            for pattern in [
                f"{ticker}{suffix_5m}",
                f"{ticker}.parquet",
                f"{ticker}_5min.parquet",
            ]:
                fpath = dir_5m / pattern
                if fpath.exists():
                    _cache_5m[ticker] = read_5m_parquet(str(fpath), engine)
                    found = True
                    break
            if not found:
                _cache_5m[ticker] = pd.DataFrame()

        df_5m = _cache_5m[ticker]
        if df_5m.empty:
            continue

        # Get the trade date for EOD cutoff
        trade_date = pd.Timestamp(entry_time).normalize()

        # Filter 5-min bars after entry time and on the same day
        mask = (df_5m["datetime"] > entry_time) & (df_5m["datetime"].dt.normalize() == trade_date)
        bars = df_5m.loc[mask].sort_values("datetime")

        if bars.empty:
            continue

        # Walk through 5-min bars to find first SL or target hit
        new_exit_price = None
        new_exit_time = None
        new_outcome = None

        for _, bar in bars.iterrows():
            bar_high = float(bar.get("high", bar.get("High", np.nan)))
            bar_low = float(bar.get("low", bar.get("Low", np.nan)))
            bar_close = float(bar.get("close", bar.get("Close", np.nan)))
            bar_time = bar["datetime"]

            if np.isnan(bar_high) or np.isnan(bar_low):
                continue

            if side == "SHORT":
                # SL hit if high >= stop_price
                if bar_high >= stop_price:
                    new_exit_price = stop_price
                    new_exit_time = bar_time
                    new_outcome = "SL"
                    break
                # Target hit if low <= target_price
                if bar_low <= target_price:
                    new_exit_price = target_price
                    new_exit_time = bar_time
                    new_outcome = "TARGET"
                    break
            else:  # LONG
                # SL hit if low <= stop_price
                if bar_low <= stop_price:
                    new_exit_price = stop_price
                    new_exit_time = bar_time
                    new_outcome = "SL"
                    break
                # Target hit if high >= target_price
                if bar_high >= target_price:
                    new_exit_price = target_price
                    new_exit_time = bar_time
                    new_outcome = "TARGET"
                    break

        # If neither SL nor target hit, exit at EOD close of last bar
        if new_exit_price is None:
            last_bar = bars.iloc[-1]
            new_exit_price = float(last_bar.get("close", last_bar.get("Close", entry_price)))
            new_exit_time = last_bar["datetime"]
            new_outcome = "EOD"

        # Update the trade row
        df.at[idx, "exit_price"] = new_exit_price
        df.at[idx, "exit_time_ist"] = new_exit_time
        df.at[idx, "outcome"] = new_outcome

        # Recalculate P&L
        if side == "SHORT":
            raw_pct = (entry_price - new_exit_price) / entry_price * 100.0 if entry_price != 0 else 0.0
        else:
            raw_pct = (new_exit_price - entry_price) / entry_price * 100.0 if entry_price != 0 else 0.0

        df.at[idx, "pnl_pct_gross"] = raw_pct

        # Apply slippage + commission (read from existing columns or use defaults)
        slippage_pct = float(df.at[idx, "slippage_pct"]) if "slippage_pct" in df.columns and pd.notna(
            df.at[idx, "slippage_pct"]) else 0.0005
        commission_pct = float(df.at[idx, "commission_pct"]) if "commission_pct" in df.columns and pd.notna(
            df.at[idx, "commission_pct"]) else 0.0003
        cost_pct = (slippage_pct + commission_pct) * 100.0 * 2  # round-trip
        df.at[idx, "pnl_pct"] = raw_pct - cost_pct

        updated_rows += 1

    print(f"[5MIN] Re-resolved exits for {updated_rows}/{total_rows} trades using 5-min data.")
    return df


# ===========================================================================
# WORKER FUNCTIONS (for parallel scanning — still uses 15-min for entry signals)
# ===========================================================================
def _scan_one_ticker_short(args: Tuple[str, str, StrategyConfig]) -> List[dict]:
    """Scan one ticker on the SHORT side. Returns list of Trade dicts."""
    ticker, path, cfg = args
    df = read_15m_parquet(path, cfg.parquet_engine)
    if df.empty:
        return []
    trades = scan_short(ticker, df, cfg)
    return [asdict(t) for t in trades]


def _scan_one_ticker_long(args: Tuple[str, str, StrategyConfig]) -> List[dict]:
    """Scan one ticker on the LONG side. Returns list of Trade dicts."""
    ticker, path, cfg = args
    df = read_15m_parquet(path, cfg.parquet_engine)
    if df.empty:
        return []
    trades = scan_long(ticker, df, cfg)
    return [asdict(t) for t in trades]


# ===========================================================================
# PARALLEL SCAN RUNNER
# ===========================================================================
def _run_side_parallel(
    side: str,
    cfg: StrategyConfig,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """
    Scan all tickers for one side using ProcessPoolExecutor.
    Falls back to serial if max_workers <= 1.
    """
    tickers = list_tickers_15m(cfg.dir_15m, cfg.end_15m)
    print(f"[{side}] Tickers found: {len(tickers)}")

    worker_fn = _scan_one_ticker_short if side == "SHORT" else _scan_one_ticker_long
    task_args = [
        (t, os.path.join(cfg.dir_15m, f"{t}{cfg.end_15m}"), cfg)
        for t in tickers
    ]

    all_dicts: List[dict] = []

    if max_workers <= 1:
        # Serial fallback
        for k, args in enumerate(task_args, 1):
            result = worker_fn(args)
            all_dicts.extend(result)
            if k % 50 == 0:
                print(f"  [{side}] scanned {k}/{len(tickers)} | trades={len(all_dicts)}")
    else:
        # Parallel
        done_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_fn, a): a[0] for a in task_args}
            for future in as_completed(futures):
                done_count += 1
                try:
                    result = future.result()
                    all_dicts.extend(result)
                except Exception as e:
                    ticker = futures[future]
                    print(f"  [{side}] ERROR on {ticker}: {e}")

                if done_count % 100 == 0:
                    print(
                        f"  [{side}] scanned {done_count}/{len(tickers)} | trades={len(all_dicts)}"
                    )

    if not all_dicts:
        return pd.DataFrame()

    out = pd.DataFrame(all_dicts)

    # Apply Top-N per day
    out = apply_topn_per_day(out, cfg)

    # Ensure datetime columns
    for c in ["signal_time_ist", "entry_time_ist", "exit_time_ist"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    sort_cols = [c for c in ["trade_date", "ticker", "entry_time_ist"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


# ===========================================================================
# NOTIONAL P&L
# ===========================================================================
def _add_notional_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add capital/notional P&L columns and apply intraday leverage correctly.

    Strategy logic computes pnl_pct as *price-return %* (unlevered).
    In intraday (e.g., 5x), your *capital/margin* stays the same (POSITION_SIZE_RS_*),
    but your *notional exposure* is capital * leverage.

    We therefore:
      - preserve unlevered price-return % in pnl_pct_price / pnl_pct_gross_price
      - compute ROI% on capital (levered) in pnl_pct / pnl_pct_gross
      - compute rupee P&L on notional exposure in pnl_rs / pnl_rs_gross
    """
    if df.empty:
        return df

    d = df.copy()

    # ---- Ensure price-return % columns exist (unlevered) ----
    if "pnl_pct_price" not in d.columns:
        d["pnl_pct_price"] = pd.to_numeric(d.get("pnl_pct", 0.0), errors="coerce").fillna(0.0)

    if "pnl_pct_gross_price" not in d.columns:
        if "pnl_pct_gross" in d.columns:
            d["pnl_pct_gross_price"] = pd.to_numeric(d["pnl_pct_gross"], errors="coerce").fillna(0.0)
        elif {"entry_price", "exit_price", "side"}.issubset(d.columns):
            ep = pd.to_numeric(d["entry_price"], errors="coerce")
            xp = pd.to_numeric(d["exit_price"], errors="coerce")
            s = d["side"].astype(str).str.upper()
            denom = ep.replace(0, np.nan)
            gross = np.where(s.eq("SHORT"), (ep - xp) / denom * 100.0, (xp - ep) / denom * 100.0)
            d["pnl_pct_gross_price"] = pd.to_numeric(gross, errors="coerce").fillna(0.0)
        else:
            d["pnl_pct_gross_price"] = 0.0

    # Normalize side
    side_u = d["side"].astype(str).str.upper() if "side" in d.columns else pd.Series([""] * len(d))

    # Capital/margin per trade (Rs.)
    if "position_size_rs" not in d.columns:
        d["position_size_rs"] = np.nan
    d.loc[side_u.eq("SHORT") & d["position_size_rs"].isna(), "position_size_rs"] = float(POSITION_SIZE_RS_SHORT)
    d.loc[~side_u.eq("SHORT") & d["position_size_rs"].isna(), "position_size_rs"] = float(POSITION_SIZE_RS_LONG)
    d["position_size_rs"] = pd.to_numeric(d["position_size_rs"], errors="coerce").fillna(0.0)

    # Leverage per trade
    if "leverage" not in d.columns:
        d["leverage"] = np.nan
    d.loc[side_u.eq("SHORT") & d["leverage"].isna(), "leverage"] = float(INTRADAY_LEVERAGE_SHORT)
    d.loc[~side_u.eq("SHORT") & d["leverage"].isna(), "leverage"] = float(INTRADAY_LEVERAGE_LONG)
    d["leverage"] = pd.to_numeric(d["leverage"], errors="coerce").fillna(1.0)

    # Notional exposure (Rs.)
    d["notional_exposure_rs"] = d["position_size_rs"] * d["leverage"]

    # ROI% on capital (levered)
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct_price"], errors="coerce").fillna(0.0) * d["leverage"]
    d["pnl_pct_gross"] = pd.to_numeric(d["pnl_pct_gross_price"], errors="coerce").fillna(0.0) * d["leverage"]

    # Rupee P&L on notional exposure
    d["pnl_rs"] = (pd.to_numeric(d["pnl_pct_price"], errors="coerce").fillna(0.0) / 100.0) * d["notional_exposure_rs"]
    d["pnl_rs_gross"] = (pd.to_numeric(d["pnl_pct_gross_price"], errors="coerce").fillna(0.0) / 100.0) * d["notional_exposure_rs"]

    return d



# ===========================================================================
# CASH-CONSTRAINED PORTFOLIO SIM (optimized with itertuples)
# ===========================================================================
def _simulate_cash_constrained(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return df, {
            "start_capital": PORTFOLIO_START_CAPITAL_RS,
            "taken": 0,
            "skipped": 0,
            "net_pnl_rs": 0.0,
            "final_equity": float(PORTFOLIO_START_CAPITAL_RS),
            "roi_pct": 0.0,
            "max_concurrent": 0,
            "min_cash": float(PORTFOLIO_START_CAPITAL_RS),
        }

    # Ensure datetime
    d = df.copy()
    for c in ["entry_time_ist", "exit_time_ist"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    d = d.sort_values(["entry_time_ist", "exit_time_ist", "ticker", "side"]).reset_index(
        drop=True
    )

    cash = float(PORTFOLIO_START_CAPITAL_RS)
    open_heap: list = []  # (exit_time, size, pnl_rs)
    seen_ticker_day: set = set()

    taken_flags = np.zeros(len(d), dtype=bool)
    cash_before_arr = np.zeros(len(d))
    cash_after_arr = np.zeros(len(d))
    pos_sizes_arr = np.zeros(len(d))
    pnl_rs_sim_arr = np.zeros(len(d))

    taken = 0
    skipped = 0
    max_conc = 0
    min_cash = cash

    # Use itertuples for ~5-10x speedup over iterrows
    for row in d.itertuples():
        idx = row.Index
        entry_ts = row.entry_time_ist
        exit_ts = row.exit_time_ist

        # Release closed positions
        while open_heap and open_heap[0][0] <= entry_ts:
            _, size, pnl_rs = heapq.heappop(open_heap)
            cash += size + pnl_rs

        cb = cash
        side = str(row.side).upper()
        ticker = str(row.ticker)
        day = str(row.trade_date)

        pos = float(POSITION_SIZE_RS_SHORT if side == "SHORT" else POSITION_SIZE_RS_LONG)
        pnl = float(getattr(row, "pnl_rs", 0.0))

        take = True
        if DISALLOW_BOTH_SIDES_SAME_TICKER_DAY:
            key = (ticker, day)
            if key in seen_ticker_day:
                take = False

        if cash < pos:
            take = False

        if take:
            cash -= pos
            heapq.heappush(open_heap, (exit_ts, pos, pnl))
            taken += 1
            seen_ticker_day.add((ticker, day))
        else:
            skipped += 1
            pos = 0.0
            pnl = 0.0

        taken_flags[idx] = take
        cash_before_arr[idx] = cb
        cash_after_arr[idx] = cash
        pos_sizes_arr[idx] = pos
        pnl_rs_sim_arr[idx] = pnl

        max_conc = max(max_conc, len(open_heap))
        min_cash = min(min_cash, cash)

    # Drain remaining positions
    while open_heap:
        _, size, pnl_rs = heapq.heappop(open_heap)
        cash += size + pnl_rs

    final_equity = cash
    net_pnl = final_equity - float(PORTFOLIO_START_CAPITAL_RS)
    roi = (net_pnl / float(PORTFOLIO_START_CAPITAL_RS) * 100.0) if PORTFOLIO_START_CAPITAL_RS > 0 else 0.0

    d["taken"] = taken_flags
    d["cash_before"] = cash_before_arr
    d["cash_after"] = cash_after_arr
    d["position_size_rs_sim"] = pos_sizes_arr
    d["pnl_rs_sim"] = pnl_rs_sim_arr

    stats = {
        "start_capital": float(PORTFOLIO_START_CAPITAL_RS),
        "taken": int(taken),
        "skipped": int(skipped),
        "net_pnl_rs": float(net_pnl),
        "final_equity": float(final_equity),
        "roi_pct": float(roi),
        "max_concurrent": int(max_conc),
        "min_cash": float(min_cash),
    }
    return d, stats


def _print_portfolio(stats: Dict[str, Any]) -> None:
    print("\n================ PORTFOLIO SUMMARY (cash-constrained) ================")
    print(f"Start capital                 : Rs.{stats['start_capital']:,.2f}")
    print(f"Taken trades                  : {stats['taken']}")
    print(f"Skipped trades                : {stats['skipped']}")
    print(f"Net P&L                       : Rs.{stats['net_pnl_rs']:,.2f}")
    print(f"Final equity                  : Rs.{stats['final_equity']:,.2f}")
    print(f"ROI on start capital          : {stats['roi_pct']:.2f}%")
    print(f"Max concurrent positions      : {stats['max_concurrent']}")
    print(f"Minimum cash during run       : Rs.{stats['min_cash']:,.2f}")
    print("=" * 69)


# ===========================================================================
# NOTIONAL P&L SUMMARY
# ===========================================================================
def _print_notional_pnl(combined: pd.DataFrame) -> None:
    if "pnl_rs" not in combined.columns:
        return
    pnl_short = float(combined.loc[combined["side"].eq("SHORT"), "pnl_rs"].sum())
    pnl_long = float(combined.loc[combined["side"].eq("LONG"), "pnl_rs"].sum())
    pnl_all = float(combined["pnl_rs"].sum())

    print(f"\n{'=' * 20} NOTIONAL P&L SUMMARY (Rs.) {'=' * 20}")
    print(f"SHORT notional P&L            : Rs.{pnl_short:,.2f}")
    print(f"LONG  notional P&L            : Rs.{pnl_long:,.2f}")
    print(f"TOTAL notional P&L            : Rs.{pnl_all:,.2f}")
    print("=" * 61)


# ===========================================================================
# ENHANCED CHARTING SUITE
# ===========================================================================
def generate_enhanced_charts(
    combined: pd.DataFrame,
    short_df: pd.DataFrame,
    long_df: pd.DataFrame,
    save_dir: Path,
    ts_label: str = "",
) -> List[str]:
    """
    Generate a comprehensive set of backtest analysis charts.
    Returns list of saved file paths.

    Charts generated:
      1.  Cumulative P&L (combined, short, long) — line
      2.  Daily P&L bar chart (combined)
      3.  Drawdown curve (combined equity)
      4.  Win rate by side — bar chart
      5.  P&L distribution histogram (combined)
      6.  P&L distribution by side (overlay histograms)
      7.  Outcome breakdown — pie chart (TARGET / SL / EOD)
      8.  Outcome breakdown by side — grouped bar chart
      9.  Monthly P&L heatmap
      10. Weekday P&L analysis
      11. Hourly entry time distribution
      12. Rolling Sharpe ratio (20-trade rolling)
      13. Trade duration distribution
      14. Top 10 winners & losers
      15. Cumulative trade count over time
      16. P&L by setup/impulse type (if available)
      17. Quality score vs P&L scatter (if available)
      18. Win rate by month
      19. Average P&L by hour of day
      20. Risk-reward realized scatter
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARN] matplotlib not available — skipping chart generation.")
        return []

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []

    # ------------ Utility helpers ----------------
    def _safe_col(df_: pd.DataFrame, col: str) -> pd.Series:
        if col in df_.columns:
            return pd.to_numeric(df_[col], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(df_)), index=df_.index)

    def _save(fig, name: str):
        p = save_dir / f"{name}_{ts_label}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(str(p))

    # Prepare common series
    combined_sorted = combined.copy()
    if "trade_date" in combined_sorted.columns:
        combined_sorted["trade_date"] = pd.to_datetime(combined_sorted["trade_date"], errors="coerce")
        combined_sorted = combined_sorted.sort_values("trade_date").reset_index(drop=True)

    pnl_pct = _safe_col(combined_sorted, "pnl_pct")
    pnl_rs = _safe_col(combined_sorted, "pnl_rs")
    cum_pnl = pnl_rs.cumsum()

    # ========== CHART 1: Cumulative P&L (Combined + Short + Long) ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cum_pnl.values, label="Combined", linewidth=2, color="#2563EB")
    if not short_df.empty:
        s_pnl = _safe_col(short_df.sort_values("trade_date") if "trade_date" in short_df.columns else short_df, "pnl_rs")
        ax.plot(s_pnl.cumsum().values, label="Short", linewidth=1.5, color="#DC2626", alpha=0.8)
    if not long_df.empty:
        l_pnl = _safe_col(long_df.sort_values("trade_date") if "trade_date" in long_df.columns else long_df, "pnl_rs")
        ax.plot(l_pnl.cumsum().values, label="Long", linewidth=1.5, color="#16A34A", alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0,
                    where=cum_pnl.values >= 0, alpha=0.15, color="#2563EB")
    ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0,
                    where=cum_pnl.values < 0, alpha=0.15, color="#DC2626")
    ax.set_title("Cumulative P&L (Rs.) — Combined / Short / Long", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L (Rs.)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "01_cumulative_pnl")

    # ========== CHART 2: Daily P&L Bar Chart ==========
    if "trade_date" in combined_sorted.columns:
        daily = combined_sorted.groupby("trade_date")["pnl_rs"].sum()
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["#16A34A" if v >= 0 else "#DC2626" for v in daily.values]
        ax.bar(range(len(daily)), daily.values, color=colors, alpha=0.85, width=0.8)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("Daily Net P&L (Rs.)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("P&L (Rs.)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
        # Show date labels for first, middle, last
        n = len(daily)
        tick_positions = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n > 5 else list(range(n))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(daily.index[i])[:10] for i in tick_positions], rotation=30, fontsize=8)
        _save(fig, "02_daily_pnl")

    # ========== CHART 3: Drawdown Curve ==========
    if len(cum_pnl) > 0:
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)
        ax1.plot(cum_pnl.values, linewidth=2, color="#2563EB", label="Equity Curve")
        ax1.plot(running_max.values, linewidth=1, color="#9CA3AF", linestyle="--", label="High Watermark")
        ax1.fill_between(range(len(cum_pnl)), cum_pnl.values, running_max.values, alpha=0.2, color="#DC2626")
        ax1.set_title("Equity Curve & Drawdown", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative P&L (Rs.)")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

        ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color="#DC2626", alpha=0.4)
        ax2.plot(drawdown.values, color="#DC2626", linewidth=1)
        ax2.set_ylabel("Drawdown (Rs.)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
        plt.tight_layout()
        _save(fig, "03_drawdown_curve")

    # ========== CHART 4: Win Rate by Side ==========
    if "side" in combined_sorted.columns:
        sides_data = []
        for s_name, s_df in [("SHORT", short_df), ("LONG", long_df), ("COMBINED", combined_sorted)]:
            if s_df.empty:
                continue
            s_pnl = _safe_col(s_df, "pnl_pct")
            wins = (s_pnl > 0).sum()
            losses = (s_pnl < 0).sum()
            be = (s_pnl == 0).sum()
            total = len(s_pnl)
            wr = wins / total * 100 if total > 0 else 0
            sides_data.append({"side": s_name, "win_rate": wr, "wins": wins, "losses": losses, "breakeven": be, "total": total})

        if sides_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = range(len(sides_data))
            colors_wr = ["#DC2626", "#16A34A", "#2563EB"][:len(sides_data)]
            bars = ax.bar(x_pos, [d["win_rate"] for d in sides_data], color=colors_wr, alpha=0.85, width=0.5)
            for bar, d in zip(bars, sides_data):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{d["win_rate"]:.1f}%\n({d["wins"]}W/{d["losses"]}L/{d["breakeven"]}BE)',
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([d["side"] for d in sides_data], fontsize=12)
            ax.set_ylim(0, max(d["win_rate"] for d in sides_data) + 15)
            ax.set_title("Win Rate by Side", fontsize=14, fontweight="bold")
            ax.set_ylabel("Win Rate (%)")
            ax.grid(True, alpha=0.3, axis="y")
            _save(fig, "04_win_rate_by_side")

    # ========== CHART 5: P&L Distribution Histogram (Combined) ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    pnl_vals = pnl_pct.dropna()
    if len(pnl_vals) > 0:
        n_bins = min(80, max(20, len(pnl_vals) // 10))
        ax.hist(pnl_vals, bins=n_bins, color="#2563EB", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(pnl_vals.mean(), color="#DC2626", linestyle="--", linewidth=2, label=f"Mean: {pnl_vals.mean():.2f}%")
        ax.axvline(pnl_vals.median(), color="#F59E0B", linestyle="--", linewidth=2, label=f"Median: {pnl_vals.median():.2f}%")
        ax.axvline(0, color="grey", linewidth=1, linestyle="-")
        ax.set_title("P&L Distribution (%) — All Trades", fontsize=14, fontweight="bold")
        ax.set_xlabel("P&L (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "05_pnl_distribution_combined")

    # ========== CHART 6: P&L Distribution by Side (Overlay) ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    if not short_df.empty:
        s_pnl = _safe_col(short_df, "pnl_pct").dropna()
        if len(s_pnl) > 0:
            ax.hist(s_pnl, bins=50, color="#DC2626", alpha=0.5, edgecolor="white", linewidth=0.3, label=f"Short (μ={s_pnl.mean():.2f}%)")
    if not long_df.empty:
        l_pnl = _safe_col(long_df, "pnl_pct").dropna()
        if len(l_pnl) > 0:
            ax.hist(l_pnl, bins=50, color="#16A34A", alpha=0.5, edgecolor="white", linewidth=0.3, label=f"Long (μ={l_pnl.mean():.2f}%)")
    ax.axvline(0, color="grey", linewidth=1, linestyle="-")
    ax.set_title("P&L Distribution by Side (Overlay)", fontsize=14, fontweight="bold")
    ax.set_xlabel("P&L (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "06_pnl_distribution_by_side")

    # ========== CHART 7: Outcome Breakdown — Pie Chart ==========
    if "outcome" in combined_sorted.columns:
        outcome_counts = combined_sorted["outcome"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        colors_pie = {"TARGET": "#16A34A", "SL": "#DC2626", "EOD": "#F59E0B", "TRAIL": "#6366F1"}
        pie_colors = [colors_pie.get(o, "#9CA3AF") for o in outcome_counts.index]
        wedges, texts, autotexts = ax.pie(
            outcome_counts.values, labels=outcome_counts.index, autopct="%1.1f%%",
            colors=pie_colors, startangle=90, textprops={"fontsize": 12},
        )
        for t in autotexts:
            t.set_fontweight("bold")
        ax.set_title("Trade Outcome Breakdown", fontsize=14, fontweight="bold")
        _save(fig, "07_outcome_pie")

    # ========== CHART 8: Outcome Breakdown by Side — Grouped Bar ==========
    if "outcome" in combined_sorted.columns and "side" in combined_sorted.columns:
        cross = pd.crosstab(combined_sorted["outcome"], combined_sorted["side"])
        fig, ax = plt.subplots(figsize=(10, 6))
        cross.plot(kind="bar", ax=ax, color=["#16A34A", "#DC2626"], alpha=0.85, edgecolor="white")
        ax.set_title("Outcome Breakdown by Side", fontsize=14, fontweight="bold")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Count")
        ax.legend(title="Side", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=0)
        _save(fig, "08_outcome_by_side")

    # ========== CHART 9: Monthly P&L Heatmap ==========
    if "trade_date" in combined_sorted.columns:
        combined_sorted["_month"] = combined_sorted["trade_date"].dt.to_period("M")
        monthly = combined_sorted.groupby("_month")["pnl_rs"].sum()
        if len(monthly) > 1:
            fig, ax = plt.subplots(figsize=(14, 5))
            colors_monthly = ["#16A34A" if v >= 0 else "#DC2626" for v in monthly.values]
            bars = ax.bar(range(len(monthly)), monthly.values, color=colors_monthly, alpha=0.85, width=0.7)
            for bar, v in zip(bars, monthly.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(monthly.values).max() * 0.02,
                        f"₹{v:,.0f}", ha="center", va="bottom", fontsize=8, rotation=45)
            ax.set_xticks(range(len(monthly)))
            ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=9)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_title("Monthly P&L (Rs.)", fontsize=14, fontweight="bold")
            ax.set_ylabel("P&L (Rs.)")
            ax.grid(True, alpha=0.3, axis="y")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
            _save(fig, "09_monthly_pnl")
        combined_sorted.drop(columns=["_month"], inplace=True, errors="ignore")

    # ========== CHART 10: Weekday P&L Analysis ==========
    if "trade_date" in combined_sorted.columns:
        combined_sorted["_weekday"] = combined_sorted["trade_date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        weekday_pnl = combined_sorted.groupby("_weekday")["pnl_rs"].agg(["sum", "mean", "count"])
        weekday_pnl = weekday_pnl.reindex(day_order).dropna()
        if len(weekday_pnl) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            colors_wd = ["#16A34A" if v >= 0 else "#DC2626" for v in weekday_pnl["sum"].values]
            ax1.bar(weekday_pnl.index, weekday_pnl["sum"].values, color=colors_wd, alpha=0.85)
            ax1.set_title("Total P&L by Weekday", fontsize=12, fontweight="bold")
            ax1.set_ylabel("P&L (Rs.)")
            ax1.grid(True, alpha=0.3, axis="y")
            ax1.tick_params(axis="x", rotation=30)
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

            colors_wd2 = ["#16A34A" if v >= 0 else "#DC2626" for v in weekday_pnl["mean"].values]
            ax2.bar(weekday_pnl.index, weekday_pnl["mean"].values, color=colors_wd2, alpha=0.85)
            ax2.set_title("Avg P&L per Trade by Weekday", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Avg P&L (Rs.)")
            ax2.grid(True, alpha=0.3, axis="y")
            ax2.tick_params(axis="x", rotation=30)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
            plt.tight_layout()
            _save(fig, "10_weekday_pnl")
        combined_sorted.drop(columns=["_weekday"], inplace=True, errors="ignore")

    # ========== CHART 11: Hourly Entry Time Distribution ==========
    if "entry_time_ist" in combined_sorted.columns:
        entry_times = pd.to_datetime(combined_sorted["entry_time_ist"], errors="coerce")
        hours = entry_times.dt.hour.dropna()
        if len(hours) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            hour_counts = hours.value_counts().sort_index()
            ax.bar(hour_counts.index, hour_counts.values, color="#6366F1", alpha=0.85, width=0.7)
            ax.set_title("Trade Entry Distribution by Hour (IST)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Hour of Day (IST)")
            ax.set_ylabel("Number of Trades")
            ax.set_xticks(range(9, 16))
            ax.grid(True, alpha=0.3, axis="y")
            _save(fig, "11_hourly_entry_distribution")

    # ========== CHART 12: Rolling Sharpe Ratio (20-trade window) ==========
    if len(pnl_pct) >= 20:
        rolling_window = 20
        rolling_mean = pnl_pct.rolling(rolling_window).mean()
        rolling_std = pnl_pct.rolling(rolling_window).std()
        rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling_sharpe.values, linewidth=1.5, color="#6366F1", alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axhline(rolling_sharpe.mean(), color="#F59E0B", linewidth=1, linestyle="--",
                   label=f"Avg: {rolling_sharpe.mean():.2f}")
        ax.fill_between(range(len(rolling_sharpe)), rolling_sharpe.values, 0,
                        where=rolling_sharpe.values >= 0, alpha=0.15, color="#16A34A")
        ax.fill_between(range(len(rolling_sharpe)), rolling_sharpe.values, 0,
                        where=rolling_sharpe.values < 0, alpha=0.15, color="#DC2626")
        ax.set_title(f"Rolling Sharpe Ratio ({rolling_window}-trade window)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Sharpe Ratio (annualized)")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        _save(fig, "12_rolling_sharpe")

    # ========== CHART 13: Trade Duration Distribution ==========
    if {"entry_time_ist", "exit_time_ist"}.issubset(combined_sorted.columns):
        entry_t = pd.to_datetime(combined_sorted["entry_time_ist"], errors="coerce")
        exit_t = pd.to_datetime(combined_sorted["exit_time_ist"], errors="coerce")
        durations_min = (exit_t - entry_t).dt.total_seconds() / 60.0
        durations_min = durations_min.dropna()
        durations_min = durations_min[durations_min > 0]
        if len(durations_min) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(durations_min, bins=min(50, max(10, len(durations_min) // 5)),
                    color="#0EA5E9", alpha=0.7, edgecolor="white")
            ax.axvline(durations_min.mean(), color="#DC2626", linestyle="--", linewidth=2,
                       label=f"Mean: {durations_min.mean():.1f} min")
            ax.axvline(durations_min.median(), color="#F59E0B", linestyle="--", linewidth=2,
                       label=f"Median: {durations_min.median():.1f} min")
            ax.set_title("Trade Duration Distribution (minutes)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Duration (minutes)")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")
            _save(fig, "13_trade_duration_dist")

    # ========== CHART 14: Top 10 Winners & Losers ==========
    if "pnl_rs" in combined_sorted.columns and "ticker" in combined_sorted.columns:
        top_win = combined_sorted.nlargest(10, "pnl_rs")[["ticker", "trade_date", "side", "pnl_rs"]]
        top_loss = combined_sorted.nsmallest(10, "pnl_rs")[["ticker", "trade_date", "side", "pnl_rs"]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        if len(top_win) > 0:
            labels_w = [f"{r.ticker}\n{str(r.trade_date)[:10]}" for r in top_win.itertuples()]
            ax1.barh(range(len(top_win)), top_win["pnl_rs"].values, color="#16A34A", alpha=0.85)
            ax1.set_yticks(range(len(top_win)))
            ax1.set_yticklabels(labels_w, fontsize=9)
            ax1.set_title("Top 10 Winners (Rs.)", fontsize=12, fontweight="bold")
            ax1.set_xlabel("P&L (Rs.)")
            ax1.grid(True, alpha=0.3, axis="x")
            ax1.invert_yaxis()

        if len(top_loss) > 0:
            labels_l = [f"{r.ticker}\n{str(r.trade_date)[:10]}" for r in top_loss.itertuples()]
            ax2.barh(range(len(top_loss)), top_loss["pnl_rs"].values, color="#DC2626", alpha=0.85)
            ax2.set_yticks(range(len(top_loss)))
            ax2.set_yticklabels(labels_l, fontsize=9)
            ax2.set_title("Top 10 Losers (Rs.)", fontsize=12, fontweight="bold")
            ax2.set_xlabel("P&L (Rs.)")
            ax2.grid(True, alpha=0.3, axis="x")
            ax2.invert_yaxis()

        plt.tight_layout()
        _save(fig, "14_top_winners_losers")

    # ========== CHART 15: Cumulative Trade Count Over Time ==========
    if "trade_date" in combined_sorted.columns:
        daily_count = combined_sorted.groupby("trade_date").size().cumsum()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(range(len(daily_count)), daily_count.values, linewidth=2, color="#6366F1")
        ax.fill_between(range(len(daily_count)), daily_count.values, alpha=0.15, color="#6366F1")
        ax.set_title("Cumulative Trade Count Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Total Trades")
        ax.grid(True, alpha=0.3)
        n_dc = len(daily_count)
        if n_dc > 5:
            tick_positions = [0, n_dc // 4, n_dc // 2, 3 * n_dc // 4, n_dc - 1]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(daily_count.index[i])[:10] for i in tick_positions], rotation=30, fontsize=8)
        _save(fig, "15_cumulative_trade_count")

    # ========== CHART 16: P&L by Setup / Impulse Type (if available) ==========
    for col_name, chart_num, title in [
        ("setup", "16a", "P&L by Setup Type"),
        ("impulse_type", "16b", "P&L by Impulse Type"),
    ]:
        if col_name in combined_sorted.columns:
            grp = combined_sorted.groupby(col_name)["pnl_rs"].agg(["sum", "mean", "count"])
            grp = grp.sort_values("sum", ascending=True)
            if len(grp) > 0 and len(grp) <= 20:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, len(grp) * 0.5)))
                colors_setup = ["#16A34A" if v >= 0 else "#DC2626" for v in grp["sum"].values]
                ax1.barh(range(len(grp)), grp["sum"].values, color=colors_setup, alpha=0.85)
                ax1.set_yticks(range(len(grp)))
                ax1.set_yticklabels(grp.index, fontsize=9)
                ax1.set_title(f"Total {title} (Rs.)", fontsize=12, fontweight="bold")
                ax1.set_xlabel("Total P&L (Rs.)")
                ax1.grid(True, alpha=0.3, axis="x")

                colors_setup2 = ["#16A34A" if v >= 0 else "#DC2626" for v in grp["mean"].values]
                ax2.barh(range(len(grp)), grp["mean"].values, color=colors_setup2, alpha=0.85)
                ax2.set_yticks(range(len(grp)))
                ax2.set_yticklabels(grp.index, fontsize=9)
                ax2.set_title(f"Avg {title} per Trade (Rs.)", fontsize=12, fontweight="bold")
                ax2.set_xlabel("Avg P&L (Rs.)")
                ax2.grid(True, alpha=0.3, axis="x")
                plt.tight_layout()
                _save(fig, f"{chart_num}_pnl_by_{col_name}")

    # ========== CHART 17: Quality Score vs P&L Scatter (if available) ==========
    if "quality_score" in combined_sorted.columns:
        qs = pd.to_numeric(combined_sorted["quality_score"], errors="coerce")
        pnl_scatter = _safe_col(combined_sorted, "pnl_pct")
        mask = qs.notna()
        if mask.sum() > 5:
            fig, ax = plt.subplots(figsize=(10, 8))
            sides_col = combined_sorted.loc[mask, "side"].astype(str).str.upper() if "side" in combined_sorted.columns else pd.Series(["COMBINED"] * mask.sum())
            for s_name, color in [("SHORT", "#DC2626"), ("LONG", "#16A34A")]:
                s_mask = sides_col == s_name
                if s_mask.any():
                    ax.scatter(qs[mask][s_mask], pnl_scatter[mask][s_mask],
                               alpha=0.5, s=30, color=color, label=s_name, edgecolors="white", linewidth=0.3)
            # Trendline
            from numpy.polynomial.polynomial import polyfit
            valid = mask & qs.notna() & pnl_scatter.notna()
            if valid.sum() > 2:
                coeffs = np.polyfit(qs[valid], pnl_scatter[valid], 1)
                x_line = np.linspace(qs[valid].min(), qs[valid].max(), 100)
                ax.plot(x_line, np.polyval(coeffs, x_line), "--", color="#F59E0B", linewidth=2,
                        label=f"Trend (slope={coeffs[0]:.3f})")
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_title("Quality Score vs P&L (%)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("P&L (%)")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            _save(fig, "17_quality_score_vs_pnl")

    # ========== CHART 18: Win Rate by Month ==========
    if "trade_date" in combined_sorted.columns:
        combined_sorted["_month_str"] = combined_sorted["trade_date"].dt.to_period("M").astype(str)
        months = combined_sorted["_month_str"].unique()
        wr_monthly = []
        for m in sorted(months):
            m_df = combined_sorted[combined_sorted["_month_str"] == m]
            m_pnl = _safe_col(m_df, "pnl_pct")
            wins = (m_pnl > 0).sum()
            total = len(m_pnl)
            wr_monthly.append({"month": m, "win_rate": wins / total * 100 if total > 0 else 0, "total": total})
        if len(wr_monthly) > 1:
            fig, ax = plt.subplots(figsize=(14, 5))
            wr_df = pd.DataFrame(wr_monthly)
            ax.bar(range(len(wr_df)), wr_df["win_rate"].values, color="#0EA5E9", alpha=0.85, width=0.7)
            ax.axhline(50, color="grey", linewidth=0.8, linestyle="--", label="50% line")
            for i, row in wr_df.iterrows():
                ax.text(i, row["win_rate"] + 1, f'{row["win_rate"]:.0f}%\n(n={row["total"]})',
                        ha="center", fontsize=8)
            ax.set_xticks(range(len(wr_df)))
            ax.set_xticklabels(wr_df["month"].values, rotation=45, fontsize=9)
            ax.set_title("Win Rate by Month", fontsize=14, fontweight="bold")
            ax.set_ylabel("Win Rate (%)")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            _save(fig, "18_win_rate_by_month")
        combined_sorted.drop(columns=["_month_str"], inplace=True, errors="ignore")

    # ========== CHART 19: Average P&L by Hour of Day ==========
    if "entry_time_ist" in combined_sorted.columns:
        entry_t = pd.to_datetime(combined_sorted["entry_time_ist"], errors="coerce")
        combined_sorted["_entry_hour"] = entry_t.dt.hour
        hourly_pnl = combined_sorted.groupby("_entry_hour")["pnl_rs"].agg(["mean", "sum", "count"])
        hourly_pnl = hourly_pnl[(hourly_pnl.index >= 9) & (hourly_pnl.index <= 15)]
        if len(hourly_pnl) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            colors_h1 = ["#16A34A" if v >= 0 else "#DC2626" for v in hourly_pnl["mean"].values]
            ax1.bar(hourly_pnl.index, hourly_pnl["mean"].values, color=colors_h1, alpha=0.85, width=0.6)
            ax1.set_title("Avg P&L per Trade by Entry Hour", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Hour (IST)")
            ax1.set_ylabel("Avg P&L (Rs.)")
            ax1.grid(True, alpha=0.3, axis="y")

            colors_h2 = ["#16A34A" if v >= 0 else "#DC2626" for v in hourly_pnl["sum"].values]
            ax2.bar(hourly_pnl.index, hourly_pnl["sum"].values, color=colors_h2, alpha=0.85, width=0.6)
            ax2.set_title("Total P&L by Entry Hour", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Hour (IST)")
            ax2.set_ylabel("Total P&L (Rs.)")
            ax2.grid(True, alpha=0.3, axis="y")
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
            plt.tight_layout()
            _save(fig, "19_avg_pnl_by_hour")
        combined_sorted.drop(columns=["_entry_hour"], inplace=True, errors="ignore")

    # ========== CHART 20: Realized Risk-Reward Scatter ==========
    if {"entry_price", "exit_price", "stop_price", "side"}.issubset(combined_sorted.columns):
        ep = pd.to_numeric(combined_sorted["entry_price"], errors="coerce")
        xp = pd.to_numeric(combined_sorted["exit_price"], errors="coerce")
        sp = pd.to_numeric(combined_sorted["stop_price"], errors="coerce")
        side_col = combined_sorted["side"].astype(str).str.upper()

        # Risk = distance from entry to stop, Reward = distance from entry to exit
        risk = np.where(side_col == "SHORT", sp - ep, ep - sp)
        reward = np.where(side_col == "SHORT", ep - xp, xp - ep)
        risk = pd.to_numeric(risk, errors="coerce")
        reward = pd.to_numeric(reward, errors="coerce")

        valid_rr = (risk > 0) & risk.notna() & reward.notna()
        if valid_rr.sum() > 5:
            rr_ratio = reward[valid_rr] / risk[valid_rr]
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_rr = np.where(reward[valid_rr] > 0, "#16A34A", "#DC2626")
            ax.scatter(risk[valid_rr], reward[valid_rr], c=colors_rr, alpha=0.5, s=25, edgecolors="white", linewidth=0.3)
            # 1:1 line
            max_val = max(risk[valid_rr].max(), abs(reward[valid_rr]).max()) * 1.1
            ax.plot([0, max_val], [0, max_val], "--", color="#9CA3AF", linewidth=1, label="1:1 R:R")
            ax.plot([0, max_val], [0, max_val * 2], "--", color="#F59E0B", linewidth=1, alpha=0.5, label="1:2 R:R")
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="-")
            ax.set_title("Realized Risk vs Reward (per trade)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Risk (entry→stop distance)")
            ax.set_ylabel("Reward (entry→exit distance)")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            _save(fig, "20_risk_reward_scatter")

    print(f"[CHARTS] Generated {len(saved)} charts in {save_dir}/")
    return saved


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    # Outputs dir: always under the algo_trading project root
    _script_dir = Path(__file__).resolve().parent
    if _script_dir.name == "avwap_v11_refactored":
        _project_root = _script_dir.parent
    else:
        _project_root = _script_dir
    _outputs_dir = _project_root / "outputs"
    _outputs_dir.mkdir(parents=True, exist_ok=True)

    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    log_path = _outputs_dir / f"avwap_combined_runner_{ts}.txt"

    # Tee all console output to outputs/*.txt
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8") as _log_fh:
        sys.stdout = _Tee(_orig_stdout, _log_fh)
        sys.stderr = _Tee(_orig_stderr, _log_fh)

        try:
            print("=" * 70)
            print("AVWAP v11 COMBINED runner — LONG + SHORT (refactored v2)")
            print("  - Entry signals: 15-min data")
            print("  - Exit resolution: 5-min data (stocks_indicators_5min_eq)")
            print("  - Outputs: */algo_trading/outputs")
            print("  - Intraday leverage: "
                  f"SHORT={INTRADAY_LEVERAGE_SHORT}x | LONG={INTRADAY_LEVERAGE_LONG}x")
            print("  - P&L% reported = ROI% on *capital/margin* (levered)")
            print("    (unlevered price-return% is saved as pnl_pct_price)")
            print("=" * 70)

            # Resolve 5-min data directory
            dir_5m = _resolve_5min_dir()
            print(f"[INFO] 5-min data directory: {dir_5m}")
            if dir_5m.is_dir():
                n_files = len(list(dir_5m.glob("*.parquet")))
                print(f"[INFO] 5-min parquet files found: {n_files}")
            else:
                print("[WARN] 5-min data directory not found — will fall back to 15-min exits.")

            short_cfg = default_short_config(
                reports_dir=_outputs_dir,
            )
            long_cfg = default_long_config(
                reports_dir=_outputs_dir,
            )

            print(
                f"[INFO] SHORT config: SL={short_cfg.stop_pct*100:.1f}%, TGT={short_cfg.target_pct*100:.1f}%, "
                f"slippage={short_cfg.slippage_pct*10000:.0f}bps, comm={short_cfg.commission_pct*10000:.0f}bps"
            )
            print(
                f"[INFO] LONG  config: SL={long_cfg.stop_pct*100:.1f}%, TGT={long_cfg.target_pct*100:.1f}%, "
                f"slippage={long_cfg.slippage_pct*10000:.0f}bps, comm={long_cfg.commission_pct*10000:.0f}bps"
            )

            short_notional = POSITION_SIZE_RS_SHORT * INTRADAY_LEVERAGE_SHORT
            long_notional = POSITION_SIZE_RS_LONG * INTRADAY_LEVERAGE_LONG
            print(
                f"[INFO] Capital/margin per trade: SHORT=Rs.{POSITION_SIZE_RS_SHORT:,.0f} | LONG=Rs.{POSITION_SIZE_RS_LONG:,.0f}"
            )
            print(
                f"[INFO] Notional exposure per trade: SHORT=Rs.{short_notional:,.0f} | LONG=Rs.{long_notional:,.0f}"
            )
            print(f"[INFO] Parallelism: max_workers={MAX_WORKERS}")
            print(f"[INFO] Output directory: {_outputs_dir}")
            print(f"[INFO] Console log: {log_path}")
            print("-" * 70)

            # ---- PHASE 1: Scan for entry signals using 15-min data ----
            print("\n[PHASE 1] Scanning for entry signals using 15-min data...")
            short_df = _run_side_parallel("SHORT", short_cfg, MAX_WORKERS)
            long_df = _run_side_parallel("LONG", long_cfg, MAX_WORKERS)

            if short_df.empty and long_df.empty:
                print("[DONE] No trades found.")
                return

            # ---- PHASE 2: Re-resolve exits using 5-min data ----
            print("\n[PHASE 2] Re-resolving exits using 5-min data for higher precision...")

            # Determine 5-min file suffix by inspecting the directory
            suffix_5m = ".parquet"
            if dir_5m.is_dir():
                sample_files = list(dir_5m.glob("*"))[:5]
                for sf in sample_files:
                    if sf.suffix:
                        suffix_5m = sf.suffix
                        break

            if not short_df.empty:
                print(f"  [SHORT] {len(short_df)} trades to re-resolve...")
                short_df = _resolve_exits_5min(short_df, dir_5m, suffix_5m, short_cfg.parquet_engine)

            if not long_df.empty:
                print(f"  [LONG] {len(long_df)} trades to re-resolve...")
                long_df = _resolve_exits_5min(long_df, dir_5m, suffix_5m, long_cfg.parquet_engine)

            # ---- Apply leverage-aware P&L (capital ROI + notional rupees) ----
            if not short_df.empty:
                short_df = _add_notional_pnl(short_df)
            if not long_df.empty:
                long_df = _add_notional_pnl(long_df)

            combined = pd.concat([short_df, long_df], ignore_index=True)
            combined = _add_notional_pnl(combined)

            # --- Comprehensive metrics ---
            print_metrics("SHORT (net of slippage+comm, 5-min exits)", compute_backtest_metrics(short_df))
            print_metrics("LONG (net of slippage+comm, 5-min exits)", compute_backtest_metrics(long_df))
            print_metrics("COMBINED (net of slippage+comm, 5-min exits)", compute_backtest_metrics(combined))

            _print_notional_pnl(combined)

            # --- Optional portfolio sim ---
            if ENABLE_CASH_CONSTRAINED_PORTFOLIO_SIM:
                sim_df, pstats = _simulate_cash_constrained(combined)
                _print_portfolio(pstats)
                combined = sim_df

            # --- Save CSV ---
            out_csv = _outputs_dir / f"avwap_longshort_trades_ALL_DAYS_{ts}.csv"
            combined.to_csv(out_csv, index=False)

            # --- Generate Legacy Charts (from avwap_common) ---
            print("\n[INFO] Generating legacy backtest charts...")
            chart_dir_legacy = _outputs_dir / "charts" / "legacy"
            chart_files_legacy = generate_backtest_charts(
                combined, short_df, long_df, save_dir=chart_dir_legacy, ts_label=ts,
            )
            if chart_files_legacy:
                print(f"[INFO] {len(chart_files_legacy)} legacy charts saved to {chart_dir_legacy}/")

            # --- Generate Enhanced Charts ---
            print("\n[INFO] Generating enhanced analysis charts...")
            chart_dir_enhanced = _outputs_dir / "charts" / "enhanced"
            chart_files_enhanced = generate_enhanced_charts(
                combined, short_df, long_df, save_dir=chart_dir_enhanced, ts_label=ts,
            )
            if chart_files_enhanced:
                print(f"[INFO] {len(chart_files_enhanced)} enhanced charts saved to {chart_dir_enhanced}/")
                for cf in chart_files_enhanced:
                    print(f"  -> {Path(cf).name}")
            else:
                print("[WARN] No enhanced charts generated (matplotlib may not be installed).")

            total_charts = len(chart_files_legacy or []) + len(chart_files_enhanced or [])
            print(f"\n[INFO] Total charts generated: {total_charts}")

            # --- Sample output ---
            cols = [
                c
                for c in [
                    "trade_date", "ticker", "side", "setup", "impulse_type",
                    "quality_score", "entry_price", "exit_price", "outcome",
                    # ROI% on capital (levered) + price-return% (unlevered)
                    "pnl_pct", "pnl_pct_price",
                    "leverage", "position_size_rs", "notional_exposure_rs",
                    "pnl_rs",
                ]
                if c in combined.columns
            ]
            print("\n=============== SAMPLE (first 30 rows) ===============")
            print(combined.head(30)[cols].to_string(index=False))
            print(f"\n[FILE SAVED] {out_csv}")
            print(f"[OUTPUTS DIR] {_outputs_dir}")
            print(f"[CONSOLE LOG] {log_path}")
            print("[DONE]")

        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr

    # Also print after restoring, for convenience when running interactively
    print(f"[LOG SAVED] {log_path}")


if __name__ == "__main__":
    main()