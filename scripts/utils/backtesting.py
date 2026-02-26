import json
import time
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# IMPORTANT: use the _2 engine
import algosm1_trading_signal_agentic_ai_2 as m

DATA_ROOT = Path(r"C:\Users\Saarit\OneDrive\Desktop\Trading\algosm1\algo_trading")

EXEC_TF = "15min"
SPEC_INDEX = 0
MODE = "test"          # full / test / train
SPLIT_RATIO = 0.7
COST_BPS = float(getattr(m, "DEFAULT_COST_BPS", 7.5))

# Net P&L ₹ calculation (assumption: fixed capital deployed per trade)
CAPITAL_PER_TRADE_RS = 50_000

# >>> NEW: TP target set to 4%
TARGET_TP_PCT = 4.0

LOG_EVERY = 25
PRINT_PER_TICKER = True
IST = "Asia/Kolkata"


def setup_logger(log_path: Path):
    logger = logging.getLogger("apply_best")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _to_ist(dt_series: pd.Series) -> pd.Series:
    """Parse datetimes and ensure Asia/Kolkata tz."""
    s = pd.to_datetime(dt_series, errors="coerce", utc=False)
    try:
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
        else:
            s = s.dt.tz_convert(IST)
    except Exception:
        pass
    return s


def _max_streak(flags: np.ndarray, want_true: bool) -> int:
    best = cur = 0
    for v in flags:
        ok = bool(v) is want_true
        if ok:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def compute_metrics(trades: pd.DataFrame, capital_per_trade_rs: float):
    if trades is None or trades.empty:
        metrics = {
            "n_trades": 0,
            "net_pnl_rs": 0.0,
            "total_return_compounded_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_trade_pct": 0.0,
            "profit_factor": 0.0,
        }
        return metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    t = trades.copy()

    for col in ["pnl_pct", "entry_time", "exit_time"]:
        if col not in t.columns:
            raise ValueError(f"Missing required column in trades CSV: {col}")

    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce").fillna(0.0)
    t["entry_time"] = _to_ist(t["entry_time"])
    t["exit_time"] = _to_ist(t["exit_time"])

    t = t.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)

    # Holding time
    t["hold_min"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 60.0

    # Net P&L (₹)
    t["pnl_rs"] = (capital_per_trade_rs * t["pnl_pct"] / 100.0)

    n = int(len(t))
    wins = t["pnl_pct"] > 0
    losses = t["pnl_pct"] < 0

    win_rate = float(wins.mean() * 100.0) if n else 0.0
    avg_trade = float(t["pnl_pct"].mean()) if n else 0.0
    med_trade = float(t["pnl_pct"].median()) if n else 0.0
    best_trade = float(t["pnl_pct"].max()) if n else 0.0
    worst_trade = float(t["pnl_pct"].min()) if n else 0.0

    sum_win = float(t.loc[wins, "pnl_pct"].sum())
    sum_loss = float(t.loc[losses, "pnl_pct"].sum())  # negative
    profit_factor = float(sum_win / abs(sum_loss)) if sum_loss < 0 else (float("inf") if sum_win > 0 else 0.0)

    avg_win = float(t.loc[wins, "pnl_pct"].mean()) if wins.any() else 0.0
    avg_loss = float(t.loc[losses, "pnl_pct"].mean()) if losses.any() else 0.0  # negative
    expectancy = float((wins.mean() * avg_win) + ((1.0 - wins.mean()) * avg_loss)) if n else 0.0

    # Compounded equity curve over sequence of trades (ignores overlap)
    rets = (t["pnl_pct"].astype(float) / 100.0).to_numpy()
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    total_comp = float((eq[-1] - 1.0) * 100.0)
    maxdd = float(dd.min() * 100.0)

    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe_like = float((mu / (sd + 1e-12)) * math.sqrt(len(rets))) if sd > 0 else float(mu * math.sqrt(len(rets)))

    win_flags = wins.to_numpy(dtype=bool)
    max_win_streak = _max_streak(win_flags, True)
    max_loss_streak = _max_streak(win_flags, False)

    start = t["entry_time"].min()
    end = t["exit_time"].max()
    net_pnl_rs = float(t["pnl_rs"].sum())

    # Daily aggregation
    t["entry_date"] = t["entry_time"].dt.date
    daily = t.groupby("entry_date", as_index=False).agg(
        trades=("pnl_pct", "size"),
        day_pnl_pct=("pnl_pct", "sum"),
        day_pnl_rs=("pnl_rs", "sum"),
    )

    # Per ticker
    if "ticker" in t.columns:
        by_ticker = (t.groupby("ticker", as_index=False)
                       .agg(trades=("pnl_pct", "size"),
                            pnl_pct=("pnl_pct", "sum"),
                            pnl_rs=("pnl_rs", "sum"))
                       .sort_values("pnl_rs", ascending=False))
    else:
        by_ticker = pd.DataFrame()

    eq_df = pd.DataFrame({
        "trade_idx": np.arange(1, len(eq) + 1),
        "entry_time": t["entry_time"],
        "pnl_pct": t["pnl_pct"],
        "equity": eq,
        "drawdown": dd * 100.0,
    })

    metrics = {
        "n_trades": n,
        "start": str(start),
        "end": str(end),
        "tp_pct": TARGET_TP_PCT,
        "win_rate_pct": win_rate,
        "avg_trade_pct": avg_trade,
        "median_trade_pct": med_trade,
        "best_trade_pct": best_trade,
        "worst_trade_pct": worst_trade,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "expectancy_pct": expectancy,
        "profit_factor": profit_factor,
        "total_return_compounded_pct": total_comp,
        "max_drawdown_pct": maxdd,
        "sharpe_like": sharpe_like,
        "net_pnl_rs": net_pnl_rs,
        "capital_per_trade_rs": float(capital_per_trade_rs),
        "avg_hold_min": float(np.nanmean(t["hold_min"])) if n else 0.0,
        "median_hold_min": float(np.nanmedian(t["hold_min"])) if n else 0.0,
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
    }

    return metrics, t, daily, by_ticker, eq_df


def main():
    t0 = time.time()
    log_path = Path("apply_best_tp4_positional.log")
    logger = setup_logger(log_path)

    # ---------------- REMOVE INTRADAY LOGIC ----------------
    # Make the engine "not intraday":
    # - Allow entry anytime (effectively)
    # - Never force-flat end-of-day (set to 23:59 so it won't trigger on market data)
    if hasattr(m, "DAY_START_HHMM"):
        m.DAY_START_HHMM = (0, 0)
    if hasattr(m, "DAY_LAST_ENTRY_HHMM"):
        m.DAY_LAST_ENTRY_HHMM = (23, 59)
    if hasattr(m, "DAY_END_HHMM"):
        m.DAY_END_HHMM = (23, 59)

    # ---------------- TARGET = 4% ----------------
    # Patch engine fixed TP (and keep its SL as-is unless you want to change that too)
    if hasattr(m, "FIXED_TP_PCT"):
        m.FIXED_TP_PCT = float(TARGET_TP_PCT)

    tp = getattr(m, "FIXED_TP_PCT", None)
    sl = getattr(m, "FIXED_SL_PCT", None)

    logger.info("DATA_ROOT=%s | EXEC_TF=%s | SPEC_INDEX=%s | MODE=%s | SPLIT_RATIO=%s | COST_BPS=%s | CAPITAL_PER_TRADE_RS=%s",
                DATA_ROOT, EXEC_TF, SPEC_INDEX, MODE, SPLIT_RATIO, COST_BPS, CAPITAL_PER_TRADE_RS)
    logger.info("TP/SL (engine) = TP=%s%% | SL=%s%%", tp, sl)
    logger.info("Intraday logic disabled via DAY_* overrides (if engine uses them).")

    # load best strategies
    spec_path = Path("best_strategies.json")
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing file: {spec_path.resolve()}")

    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    if SPEC_INDEX < 0 or SPEC_INDEX >= len(specs):
        raise ValueError(f"SPEC_INDEX out of range. Found {len(specs)} specs in best_strategies.json")

    chosen = specs[SPEC_INDEX]
    spec = m.StrategySpec(name=chosen["name"], family=chosen["family"], params=chosen["params"])

    # Also force TP into params (in case engine reads spec.params instead of module constants)
    spec.params = dict(spec.params) if spec.params else {}
    spec.params["tp_pct"] = float(TARGET_TP_PCT)

    logger.info("Chosen strategy: name=%s | family=%s | params=%s", spec.name, spec.family, spec.params)

    exec_dir = DATA_ROOT / "main_indicators_15min"
    tickers = sorted({p.name.split("_main_indicators_")[0] for p in exec_dir.glob("*_main_indicators_*.csv")})
    logger.info("Tickers detected: %d (from %s)", len(tickers), exec_dir)

    all_trades = []
    loaded_ok = missing = empty = errors = tickers_with_trades = 0

    for idx, tkr in enumerate(tickers, start=1):
        try:
            df = m.load_merged_ticker(DATA_ROOT, tkr, exec_tf="15min")

            if df is None:
                missing += 1
                if PRINT_PER_TICKER:
                    logger.info("[%d/%d] %s SKIP missing daily/15min files", idx, len(tickers), tkr)
                continue
            if df.empty:
                empty += 1
                if PRINT_PER_TICKER:
                    logger.info("[%d/%d] %s SKIP empty", idx, len(tickers), tkr)
                continue

            if MODE == "train":
                df, _ = m.walkforward_split_by_date(df, split_ratio=SPLIT_RATIO)
            elif MODE == "test":
                _, df = m.walkforward_split_by_date(df, split_ratio=SPLIT_RATIO)

            if df.empty:
                empty += 1
                if PRINT_PER_TICKER:
                    logger.info("[%d/%d] %s SKIP empty after split", idx, len(tickers), tkr)
                continue

            tr = m.simulate_one_ticker(df, spec, cost_bps=COST_BPS)
            loaded_ok += 1

            if not tr.empty:
                tr["ticker"] = tkr
                all_trades.append(tr)
                tickers_with_trades += 1
                if PRINT_PER_TICKER:
                    logger.info("[%d/%d] %s trades=%d", idx, len(tickers), tkr, len(tr))
            else:
                if PRINT_PER_TICKER:
                    logger.info("[%d/%d] %s trades=0", idx, len(tickers), tkr)

            if idx % LOG_EVERY == 0:
                logger.info("Progress %d/%d | loaded_ok=%d | tickers_with_trades=%d | errors=%d",
                            idx, len(tickers), loaded_ok, tickers_with_trades, errors)

        except Exception as e:
            errors += 1
            logger.exception("[%d/%d] %s ERROR %s", idx, len(tickers), tkr, e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    out_tag = f"tp{int(TARGET_TP_PCT)}_positional"
    out_trades = Path(f"trades_{EXEC_TF}_spec{SPEC_INDEX}_{MODE}_{out_tag}.csv")
    trades.to_csv(out_trades, index=False)
    logger.info("Saved trades: %s", out_trades.resolve())

    metrics, trades2, daily, by_ticker, eq_df = compute_metrics(trades, CAPITAL_PER_TRADE_RS)

    out_metrics = Path(f"metrics_{EXEC_TF}_spec{SPEC_INDEX}_{MODE}_{out_tag}.json")
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    out_daily = Path(f"daily_pnl_{EXEC_TF}_spec{SPEC_INDEX}_{MODE}_{out_tag}.csv")
    daily.to_csv(out_daily, index=False)

    out_eq = Path(f"equity_curve_{EXEC_TF}_spec{SPEC_INDEX}_{MODE}_{out_tag}.csv")
    eq_df.to_csv(out_eq, index=False)

    if not by_ticker.empty:
        out_ticker = Path(f"ticker_pnl_{EXEC_TF}_spec{SPEC_INDEX}_{MODE}_{out_tag}.csv")
        by_ticker.to_csv(out_ticker, index=False)
        logger.info("Saved ticker P&L: %s", out_ticker.resolve())

    logger.info("Saved metrics: %s", out_metrics.resolve())
    logger.info("Saved daily P&L: %s", out_daily.resolve())
    logger.info("Saved equity curve: %s", out_eq.resolve())

    logger.info("---- SUMMARY (TP=%.2f%%, intraday OFF) ----", TARGET_TP_PCT)
    logger.info("Trades=%s | WinRate=%.2f%% | AvgTrade=%.4f%% | ProfitFactor=%s",
                metrics.get("n_trades", 0),
                metrics.get("win_rate_pct", 0.0),
                metrics.get("avg_trade_pct", 0.0),
                metrics.get("profit_factor", 0.0))
    logger.info("TotalCompReturn=%.2f%% | MaxDD=%.2f%% | SharpeLike=%.3f",
                metrics.get("total_return_compounded_pct", 0.0),
                metrics.get("max_drawdown_pct", 0.0),
                metrics.get("sharpe_like", 0.0))
    logger.info("NET P&L (₹) = %.2f (capital/trade ₹%.0f)",
                metrics.get("net_pnl_rs", 0.0),
                metrics.get("capital_per_trade_rs", CAPITAL_PER_TRADE_RS))

    logger.info("Tickers=%d | loaded_ok=%d | missing=%d | empty=%d | tickers_with_trades=%d | errors=%d",
                len(tickers), loaded_ok, missing, empty, tickers_with_trades, errors)
    logger.info("Runtime=%.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
