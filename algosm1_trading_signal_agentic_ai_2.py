# -*- coding: utf-8 -*-
"""
Offline agentic strategy search (template + optimizer) for your precomputed indicator CSVs.

UPDATED (as requested):
  - Uses ONLY:
      * Daily as context TF  -> main_indicators_daily/{TICKER}_main_indicators_daily.csv
      * 15min as exec TF     -> main_indicators_15min/{TICKER}_main_indicators_15min.csv
  - Intraday long-only simulator
  - Fixed exits:
      * Target = +2%
      * Stop   = -2%
    (No ATR-based stop/target; ATR is NOT required anymore.)

Run:
  pip install -U pandas numpy optuna
  python algosm1_trading_signal_agentic_ai.py --data-root . --n-trials 200 --top-k 10

Output:
  - results_strategies.csv
  - best_strategies.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:
    raise SystemExit("Please install optuna:  pip install optuna")


# ----------------------------- CONFIG -----------------------------

IST = "Asia/Kolkata"

# Fixed risk (requested)
FIXED_TP_PCT = 2.0
FIXED_SL_PCT = 2.0

# If you want ONLY TP/SL exits (plus EOD force-flat), keep this False.
USE_BB_MID_EXIT = False

DEFAULT_COST_BPS = 7.5  # ~0.075% round-trip as a starting point (brokerage+slippage); tune!
DAY_END_HHMM = (15, 15)          # force-flat after this time
DAY_START_HHMM = (9, 20)         # optional entry window start
DAY_LAST_ENTRY_HHMM = (14, 45)   # optional last entry time

EXEC_TF = "15min"  # locked by requirement


# ----------------------------- DATA HELPERS -----------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    if getattr(df["date"].dt, "tz", None) is None:
        df["date"] = df["date"].dt.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df["date"] = df["date"].dt.tz_convert(IST)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _add_exec_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a few extra features on the execution timeframe without changing your stored CSVs."""
    out = df.copy()
    out["EMA_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["BB_Mid"] = (out.get("Upper_Band") + out.get("Lower_Band")) / 2.0
    out["date_only"] = out["date"].dt.date
    out["hh"] = out["date"].dt.hour
    out["mm"] = out["date"].dt.minute
    return out


def _prep_daily_context(df_d: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare daily context ensuring NO look-ahead:
      - shift daily indicators by 1 day so today's intraday uses yesterday's daily state
    """
    d = df_d.copy()
    d["date_only"] = d["date"].dt.date
    context_cols = [c for c in d.columns if c not in ("date", "open", "high", "low", "close", "volume", "date_only")]
    d = d.sort_values("date_only").reset_index(drop=True)
    for c in context_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].shift(1)
    keep = ["date_only"] + context_cols
    return d[keep]


def load_merged_ticker(data_root: Path, ticker: str, exec_tf: str = EXEC_TF) -> Optional[pd.DataFrame]:
    """
    Load exec TF (15min) and merge daily context (shifted) onto it.
    Filenames must follow: {TICKER}_main_indicators_{mode}.csv
    """
    if exec_tf != "15min":
        raise ValueError("This version is locked to exec_tf='15min' only.")

    exec_path = data_root / "main_indicators_15min" / f"{ticker}_main_indicators_15min.csv"
    daily_path = data_root / "main_indicators_daily" / f"{ticker}_main_indicators_daily.csv"

    if not exec_path.exists() or not daily_path.exists():
        return None

    df_e = _add_exec_features(_read_csv(exec_path))
    df_d = _prep_daily_context(_read_csv(daily_path))

    merged = df_e.merge(df_d, on="date_only", how="left", suffixes=("", "_D"))
    merged = merged.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return merged


# ----------------------------- BACKTEST CORE -----------------------------

@dataclass
class StrategySpec:
    name: str
    family: str
    params: Dict[str, float]


def _in_time_window(row, start_hhmm=DAY_START_HHMM, last_entry_hhmm=DAY_LAST_ENTRY_HHMM) -> bool:
    hh, mm = int(row["hh"]), int(row["mm"])
    after_start = (hh > start_hhmm[0]) or (hh == start_hhmm[0] and mm >= start_hhmm[1])
    before_last = (hh < last_entry_hhmm[0]) or (hh == last_entry_hhmm[0] and mm <= last_entry_hhmm[1])
    return after_start and before_last


def _force_flat(row, day_end_hhmm=DAY_END_HHMM) -> bool:
    hh, mm = int(row["hh"]), int(row["mm"])
    return (hh > day_end_hhmm[0]) or (hh == day_end_hhmm[0] and mm >= day_end_hhmm[1])


def simulate_one_ticker(df: pd.DataFrame, spec: StrategySpec, cost_bps: float) -> pd.DataFrame:
    """
    Bar-based long-only simulator with:
      - 1 trade per day per ticker (earliest)
      - Fixed stop/target: SL=-2%, TP=+2%
      - Conservative exits: if both hit in same bar, assume STOP first.
      - Force-flat near EOD
    """
    p = spec.params
    fam = spec.family

    trades = []
    in_pos = False
    entry_price = np.nan
    entry_time = None
    stop = np.nan
    target = np.nan
    day_traded = None

    # Pre-calc numeric columns (safe)
    num_cols = ["RSI", "ADX", "EMA_20", "EMA_50", "EMA_200", "Upper_Band", "Lower_Band", "BB_Mid", "Intra_Change", "Recent_High"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def dcol(name: str) -> str:
        return f"{name}_D" if f"{name}_D" in df.columns else name

    for i in range(1, len(df)):
        row = df.iloc[i]
        date_only = row["date_only"]

        # Force-flat EOD
        if in_pos and _force_flat(row):
            exit_price = float(row["open"])
            exit_time = row["date"]
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
            pnl_pct -= 2.0 * cost_bps / 100.0
            trades.append((entry_time, exit_time, entry_price, exit_price, pnl_pct, fam))
            in_pos = False
            continue

        # One trade per day
        if (day_traded == date_only) and not in_pos:
            continue

        # If in position: check SL/TP
        if in_pos:
            lo = float(row["low"])
            hi = float(row["high"])

            hit_stop = (lo <= stop) if not math.isnan(stop) else False
            hit_tgt  = (hi >= target) if not math.isnan(target) else False

            if hit_stop or hit_tgt:
                exit_price = stop if hit_stop else target  # stop-first if both
                exit_time = row["date"]
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                pnl_pct -= 2.0 * cost_bps / 100.0
                trades.append((entry_time, exit_time, entry_price, exit_price, pnl_pct, fam))
                in_pos = False
            elif USE_BB_MID_EXIT and fam == "meanrev_bb" and pd.notna(row.get("BB_Mid")):
                if float(row["close"]) >= float(row["BB_Mid"]):
                    exit_price = float(row["open"])
                    exit_time = row["date"]
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                    pnl_pct -= 2.0 * cost_bps / 100.0
                    trades.append((entry_time, exit_time, entry_price, exit_price, pnl_pct, fam))
                    in_pos = False

            continue

        # Entry time window
        if not _in_time_window(row):
            continue

        # Daily trend filter (applies to trend/breakout only)
        daily_ok = True
        if fam in ("trend_pullback", "breakout"):
            ema_fast = row.get(dcol("EMA_50"))
            ema_slow = row.get(dcol("EMA_200"))
            adx_d = row.get(dcol("ADX"))

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                daily_ok = False
            else:
                daily_ok = float(ema_fast) > float(ema_slow)

            if daily_ok and pd.notna(adx_d):
                daily_ok = daily_ok and (float(adx_d) >= float(p["daily_adx_min"]))

        if not daily_ok:
            continue

        # Intraday triggers (15m)
        entry_ok = False

        if fam == "trend_pullback":
            rsi = row.get("RSI")
            ema20 = row.get("EMA_20")
            adx = row.get("ADX")
            if pd.notna(rsi) and pd.notna(ema20):
                entry_ok = (float(row["close"]) > float(ema20)) and (float(rsi) >= float(p["rsi_min"]))
            if entry_ok and pd.notna(adx):
                entry_ok = entry_ok and (float(adx) >= float(p["adx_min"]))

        elif fam == "breakout":
            rh = row.get("Recent_High")
            rsi = row.get("RSI")
            if pd.notna(rh) and pd.notna(rsi):
                entry_ok = (float(row["close"]) >= float(rh) * (1.0 + float(p["breakout_buffer_pct"]) / 100.0)) and (float(rsi) >= float(p["rsi_min"]))

        elif fam == "meanrev_bb":
            lb = row.get("Lower_Band")
            rsi = row.get("RSI")
            if pd.notna(lb) and pd.notna(rsi):
                entry_ok = (float(row["close"]) <= float(lb) * (1.0 - float(p["band_buffer_pct"]) / 100.0)) and (float(rsi) <= float(p["rsi_max"]))

        if not entry_ok:
            continue

        # Enter at current bar open approximation
        entry_price = float(row["open"])
        entry_time = row["date"]
        day_traded = date_only
        in_pos = True

        # Fixed exits
        stop = entry_price * (1.0 - FIXED_SL_PCT / 100.0)
        target = entry_price * (1.0 + FIXED_TP_PCT / 100.0)

    trades_df = pd.DataFrame(trades, columns=["entry_time", "exit_time", "entry_px", "exit_px", "pnl_pct", "family"])
    return trades_df


def _equity_curve(trades: pd.DataFrame) -> Tuple[float, float, float, int]:
    if trades.empty:
        return 0.0, 0.0, 0.0, 0
    rets = trades["pnl_pct"].astype(float) / 100.0
    eq = (1.0 + rets).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    total = (eq.iloc[-1] - 1.0) * 100.0
    maxdd = dd.min() * 100.0
    mu = rets.mean()
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = (mu / (sd + 1e-12)) * math.sqrt(len(rets)) if sd > 0 else (mu * math.sqrt(len(rets)))
    return float(total), float(maxdd), float(sharpe), int(len(trades))


def evaluate_universe(ticker_dfs: Dict[str, pd.DataFrame], spec: StrategySpec, cost_bps: float) -> Dict[str, float]:
    all_trades = []
    for t, df in ticker_dfs.items():
        tr = simulate_one_ticker(df, spec, cost_bps=cost_bps)
        if not tr.empty:
            tr["ticker"] = t
            all_trades.append(tr)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    total, maxdd, sharpe, ntr = _equity_curve(trades)
    win = float((trades["pnl_pct"] > 0).mean() * 100.0) if not trades.empty else 0.0
    avg = float(trades["pnl_pct"].mean()) if not trades.empty else 0.0
    return {
        "total_return_pct": total,
        "max_drawdown_pct": maxdd,
        "sharpe_like": sharpe,
        "n_trades": float(ntr),
        "win_rate_pct": win,
        "avg_trade_pct": avg,
    }


# ----------------------------- "AGENTIC" SEARCH -----------------------------

FAMILIES = ("trend_pullback", "breakout", "meanrev_bb")


def make_spec_from_trial(trial: "optuna.Trial") -> StrategySpec:
    fam = trial.suggest_categorical("family", FAMILIES)
    params: Dict[str, float] = {}

    # Daily filter params (still useful)
    if fam in ("trend_pullback", "breakout"):
        params["daily_adx_min"] = trial.suggest_float("daily_adx_min", 10.0, 30.0)

    # Intraday params
    if fam == "trend_pullback":
        params["rsi_min"] = trial.suggest_float("rsi_min", 50.0, 70.0)
        params["adx_min"] = trial.suggest_float("adx_min", 10.0, 35.0)

    elif fam == "breakout":
        params["rsi_min"] = trial.suggest_float("rsi_min", 50.0, 75.0)
        params["breakout_buffer_pct"] = trial.suggest_float("breakout_buffer_pct", 0.0, 0.6)

    elif fam == "meanrev_bb":
        params["rsi_max"] = trial.suggest_float("rsi_max", 20.0, 45.0)
        params["band_buffer_pct"] = trial.suggest_float("band_buffer_pct", 0.0, 0.6)

    name = f"{fam}_tp{FIXED_TP_PCT:.1f}_sl{FIXED_SL_PCT:.1f}"
    return StrategySpec(name=name, family=fam, params=params)


def walkforward_split_by_date(df: pd.DataFrame, split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    dates = pd.Series(sorted(df["date_only"].unique()))
    cut = int(len(dates) * split_ratio)
    train_days = set(dates.iloc[:max(cut, 1)].tolist())
    test_days = set(dates.iloc[max(cut, 1):].tolist())
    return df[df["date_only"].isin(train_days)].copy(), df[df["date_only"].isin(test_days)].copy()


def objective(trial: "optuna.Trial", ticker_dfs_all: Dict[str, pd.DataFrame], cost_bps: float) -> float:
    spec = make_spec_from_trial(trial)

    # Evaluate ONLY on test portion to reduce overfit
    ticker_dfs_test = {}
    for t, df in ticker_dfs_all.items():
        _, df_test = walkforward_split_by_date(df, split_ratio=0.7)
        ticker_dfs_test[t] = df_test

    m = evaluate_universe(ticker_dfs_test, spec, cost_bps=cost_bps)

    total = m["total_return_pct"]
    maxdd = abs(min(m["max_drawdown_pct"], 0.0))
    ntr = m["n_trades"]

    trade_penalty = 0.0 if ntr >= 30 else (30 - ntr) * 0.5
    score = total - 0.6 * maxdd - trade_penalty

    trial.set_user_attr("metrics", m)
    trial.set_user_attr("spec", asdict(spec))
    return float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".", help="Folder that contains main_indicators_* directories")
    ap.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (blank = auto-detect from 15min folder)")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    exec_dir = data_root / "main_indicators_15min"
    if not exec_dir.exists():
        raise SystemExit(f"Missing directory: {exec_dir}")

    if args.tickers.strip():
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted({p.name.split("_main_indicators_")[0] for p in exec_dir.glob("*_main_indicators_*.csv")})

    ticker_dfs = {}
    for t in tickers:
        m = load_merged_ticker(data_root, t, exec_tf="15min")
        if m is None or m.empty:
            continue
        ticker_dfs[t] = m

    if not ticker_dfs:
        raise SystemExit("No tickers loaded. Check main_indicators_daily and main_indicators_15min folders/files.")

    print(f"Loaded {len(ticker_dfs)} tickers for exec_tf=15min. Beginning search...")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda tr: objective(tr, ticker_dfs, cost_bps=args.cost_bps), n_trials=args.n_trials, show_progress_bar=True)

    rows = []
    best_specs = []
    for tr in study.best_trials[:args.top_k]:
        spec = tr.user_attrs.get("spec", {})
        met = tr.user_attrs.get("metrics", {})
        row = {
            "score": tr.value,
            **met,
            "name": spec.get("name"),
            "family": spec.get("family"),
            "params": json.dumps(spec.get("params", {}), sort_keys=True),
        }
        rows.append(row)
        best_specs.append(spec)

    out_csv = Path("results_strategies.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(out_csv, index=False)
    out_json = Path("best_strategies.json")
    out_json.write_text(json.dumps(best_specs, indent=2))

    print(f"\nSaved: {out_csv.resolve()}")
    print(f"Saved: {out_json.resolve()}")
    if rows:
        print("\nTop strategy:")
        print(pd.DataFrame(rows).head(1).to_string(index=False))


if __name__ == "__main__":
    main()
