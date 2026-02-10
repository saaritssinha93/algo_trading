# -*- coding: utf-8 -*-
"""
avwap_combined_runner.py — AVWAP v11 COMBINED LONG + SHORT runner (refactored)
==============================================================================

Improvements over the original v11 combined runner:
1. Normal Python imports (no importlib hacks)
2. Unified Trade dataclass — both sides produce identical columns
3. Parallel ticker scanning via ProcessPoolExecutor
4. Slippage + commission model baked into P&L
5. Comprehensive backtest metrics (Sharpe, Sortino, Calmar, drawdown, profit factor)
6. All config via StrategyConfig dataclass — no module-level globals
7. Optional YAML config override (future-ready)
8. Cash-constrained portfolio sim uses itertuples() instead of iterrows()

Usage:
    python -m avwap_v11_refactored.avwap_combined_runner
    # or
    python avwap_v11_refactored/avwap_combined_runner.py
"""

from __future__ import annotations

import heapq
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure the package is importable when running this file directly
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from avwap_v11_refactored.avwap_common import (
    IST,
    StrategyConfig,
    Trade,
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

ENABLE_CASH_CONSTRAINED_PORTFOLIO_SIM = False
PORTFOLIO_START_CAPITAL_RS = 1_000_000
DISALLOW_BOTH_SIDES_SAME_TICKER_DAY = False

# Parallelism: set to 1 for serial, >1 for multi-process
MAX_WORKERS = 4


# ==========================================================================
# OUTPUT PATHS
# ==========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR_5M = "stocks_indicators_5min_eq"
DATA_SUFFIX_5M = "_stocks_indicators_5min.parquet"


# ===========================================================================
# WORKER FUNCTIONS (for parallel scanning)
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
    if df.empty:
        return df
    d = df.copy()
    # Ensure we always carry gross P&L too (before slippage/commission).
    # Older report files may not have this column.
    if (
        "pnl_pct_gross" not in d.columns
        and {"entry_price", "exit_price", "side"}.issubset(d.columns)
    ):
        ep = pd.to_numeric(d["entry_price"], errors="coerce")
        xp = pd.to_numeric(d["exit_price"], errors="coerce")
        s = d["side"].astype(str).str.upper()
        denom = ep.replace(0, np.nan)
        gross = np.where(s.eq("SHORT"), (ep - xp) / denom * 100.0, (xp - ep) / denom * 100.0)
        d["pnl_pct_gross"] = pd.to_numeric(gross, errors="coerce").fillna(0.0)
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["position_size_rs"] = d["side"].map(
        lambda x: POSITION_SIZE_RS_SHORT if str(x).upper() == "SHORT" else POSITION_SIZE_RS_LONG
    )
    d["pnl_rs"] = (d["pnl_pct"] / 100.0) * d["position_size_rs"]
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


def _generate_detailed_analysis_charts(
    combined: pd.DataFrame,
    save_dir: Path,
    ts_label: str,
) -> List[str]:
    """Generate additional detailed charts for deeper diagnostics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping detailed chart generation.")
        return []

    if combined.empty:
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    d = combined.copy()
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["entry_time_ist"] = pd.to_datetime(d.get("entry_time_ist"), errors="coerce")
    d["trade_date"] = pd.to_datetime(d.get("trade_date"), errors="coerce")
    d = d.sort_values("entry_time_ist").reset_index(drop=True)

    # 1) Rolling performance profile
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    rolling_n = 50 if len(d) >= 50 else max(10, len(d) // 2)
    d["win_flag"] = (d["pnl_pct"] > 0).astype(float)
    d["roll_win_rate"] = d["win_flag"].rolling(rolling_n, min_periods=5).mean() * 100.0
    d["roll_avg_pnl"] = d["pnl_pct"].rolling(rolling_n, min_periods=5).mean()

    axes[0].plot(d.index, d["roll_win_rate"], color="darkgreen", linewidth=2)
    axes[0].axhline(50, linestyle="--", color="gray", alpha=0.6)
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_title(f"Rolling Win Rate ({rolling_n}-trade window)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(d.index, d["roll_avg_pnl"], color="navy", linewidth=2)
    axes[1].axhline(0, linestyle="--", color="gray", alpha=0.6)
    axes[1].set_xlabel("Trade #")
    axes[1].set_ylabel("Avg PnL (%)")
    axes[1].set_title(f"Rolling Average PnL ({rolling_n}-trade window)")
    axes[1].grid(True, alpha=0.3)

    p = save_dir / f"rolling_performance_{ts_label}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 2) Day-of-week and hour performance heatmap
    if d["entry_time_ist"].notna().any():
        h = d[d["entry_time_ist"].notna()].copy()
        h["weekday"] = h["entry_time_ist"].dt.day_name().str[:3]
        h["hour"] = h["entry_time_ist"].dt.hour
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heat = h.pivot_table(
            index="weekday",
            columns="hour",
            values="pnl_pct",
            aggfunc="mean",
        ).reindex(weekdays)

        fig, ax = plt.subplots(figsize=(13, 6))
        im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=45)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_title("Average PnL (%) Heatmap by Weekday x Entry Hour")
        ax.set_xlabel("Entry Hour (IST)")
        ax.set_ylabel("Weekday")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean PnL (%)")

        p = save_dir / f"weekday_hour_heatmap_{ts_label}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 3) Side-wise cumulative equity on same time axis
    if "side" in d.columns and d["entry_time_ist"].notna().any():
        fig, ax = plt.subplots(figsize=(15, 6))
        for side, color in [("SHORT", "#d62728"), ("LONG", "#2ca02c")]:
            sub = d[d["side"].astype(str).str.upper() == side].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("entry_time_ist")
            sub["cum"] = sub["pnl_pct"].cumsum()
            ax.plot(sub["entry_time_ist"], sub["cum"], label=f"{side} Cumulative", color=color, linewidth=2)

        d2 = d.sort_values("entry_time_ist").copy()
        d2["cum"] = d2["pnl_pct"].cumsum()
        ax.plot(d2["entry_time_ist"], d2["cum"], label="COMBINED", color="black", linewidth=2.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
        ax.set_title("Cumulative Equity by Side (Time-Synced)")
        ax.set_xlabel("Entry Time")
        ax.set_ylabel("Cumulative PnL (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        p = save_dir / f"equity_by_side_time_{ts_label}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved



# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    print("=" * 70)
    print("AVWAP v11 COMBINED runner — LONG + SHORT (refactored)")
    print("=" * 70)

    # Outputs dir: always under the algo_trading project root, regardless of
    # whether this script is run from avwap_v11_refactored/ or the project root.
    _script_dir = Path(__file__).resolve().parent
    if _script_dir.name == "avwap_v11_refactored":
        _project_root = _script_dir.parent
    else:
        _project_root = _script_dir
    _reports_dir = _project_root / "outputs"

    short_cfg = default_short_config(
        reports_dir=_reports_dir,
        dir_15m=DATA_DIR_5M,
        end_15m=DATA_SUFFIX_5M,
    )
    long_cfg = default_long_config(
        reports_dir=_reports_dir,
        dir_15m=DATA_DIR_5M,
        end_15m=DATA_SUFFIX_5M,
    )

    print(f"[INFO] SHORT config: SL={short_cfg.stop_pct*100:.1f}%, TGT={short_cfg.target_pct*100:.1f}%, "
          f"slippage={short_cfg.slippage_pct*10000:.0f}bps, comm={short_cfg.commission_pct*10000:.0f}bps")
    print(f"[INFO] LONG  config: SL={long_cfg.stop_pct*100:.1f}%, TGT={long_cfg.target_pct*100:.1f}%, "
          f"slippage={long_cfg.slippage_pct*10000:.0f}bps, comm={long_cfg.commission_pct*10000:.0f}bps")
    print(f"[INFO] Notional: SHORT=Rs.{POSITION_SIZE_RS_SHORT:,.0f} | LONG=Rs.{POSITION_SIZE_RS_LONG:,.0f}")
    print(f"[INFO] Parallelism: max_workers={MAX_WORKERS}")
    print("-" * 70)

    # Run both sides (could be parallelized further with threads wrapping processes)
    short_df = _run_side_parallel("SHORT", short_cfg, MAX_WORKERS)
    long_df = _run_side_parallel("LONG", long_cfg, MAX_WORKERS)

    if short_df.empty and long_df.empty:
        print("[DONE] No trades found.")
        return

    combined = pd.concat([short_df, long_df], ignore_index=True)
    combined = _add_notional_pnl(combined)

    # --- Comprehensive metrics ---
    short_metrics = compute_backtest_metrics(short_df)
    long_metrics = compute_backtest_metrics(long_df)
    combined_metrics = compute_backtest_metrics(combined)

    print_metrics("SHORT (net of slippage+comm)", short_metrics)
    print_metrics("LONG (net of slippage+comm)", long_metrics)
    print_metrics("COMBINED (net of slippage+comm)", combined_metrics)

    _print_notional_pnl(combined)

    # --- Optional portfolio sim ---
    if ENABLE_CASH_CONSTRAINED_PORTFOLIO_SIM:
        sim_df, pstats = _simulate_cash_constrained(combined)
        _print_portfolio(pstats)
        combined = sim_df

    # --- Save CSV ---
    reports_dir = short_cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = now_ist().strftime("%Y%m%d_%H%M%S")
    out_csv = reports_dir / f"avwap_longshort_trades_ALL_DAYS_{ts}.csv"
    combined.to_csv(out_csv, index=False)

    # --- Generate Charts ---
    print("\n[INFO] Generating backtest charts...")
    chart_dir = reports_dir / "charts"
    chart_files = generate_backtest_charts(
        combined, short_df, long_df, save_dir=chart_dir, ts_label=ts,
    )
    if chart_files:
        print(f"[INFO] {len(chart_files)} charts saved to {chart_dir}/")
        for cf in chart_files:
            print(f"  -> {cf}")
    else:
        print("[WARN] No charts generated (matplotlib may not be installed).")

    detailed_chart_files = _generate_detailed_analysis_charts(
        combined=combined,
        save_dir=chart_dir,
        ts_label=ts,
    )
    if detailed_chart_files:
        print(f"[INFO] {len(detailed_chart_files)} detailed charts saved to {chart_dir}/")
        for cf in detailed_chart_files:
            print(f"  -> {cf}")

    # --- Sample output ---
    cols = [
        c
        for c in [
            "trade_date", "ticker", "side", "setup", "impulse_type",
            "quality_score", "entry_price", "exit_price", "outcome",
            "pnl_pct", "pnl_pct_gross", "position_size_rs", "pnl_rs",
        ]
        if c in combined.columns
    ]
    print("\n=============== SAMPLE (first 30 rows) ===============")
    print(combined.head(30)[cols].to_string(index=False))
    print(f"\n[FILE SAVED] {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
