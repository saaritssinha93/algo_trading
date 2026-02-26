# -*- coding: utf-8 -*-
"""
TRADE-DATE positional LONG signal generator (DAILY execution, WEEKLY context).

✅ Asks user for intended trade date (IST) if not passed via CLI
✅ If intended trade date has no DAILY bar in your saved CSVs, advances to NEXT date that has a bar
✅ AS-OF rules relative to RESOLVED_TRADE_DATE:
    - Daily used: strictly BEFORE trade_date 00:00 IST  (last available daily candle before trade day)
    - Weekly context: previous fully completed week only (no same-week lookahead)
        weekly_date_visible = weekly_date + 1 day
        merge_asof(direction="backward") onto Daily timeline
✅ Outputs ONLY for RESOLVED_TRADE_DATE (one signal per ticker)
✅ Uses full history for rolling calculations, but signal evaluation uses AS-OF cut

Run:
  python positional_trade_date_input.py
  python positional_trade_date_input.py --trade-date 2025-12-29 custom01 custom02
  python positional_trade_date_input.py --trade-date 2025-12-29 all
"""

import os
import glob
import argparse
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import pytz

# ----------------------- CONFIG -----------------------
DIR_D   = "main_indicators_daily"
DIR_W   = "main_indicators_weekly"
OUT_DIR = "signals_today"   # keeping your folder name
IST = pytz.timezone("Asia/Kolkata")

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------- STRATEGY SELECTION -----------------------
DEFAULT_STRATEGY_KEYS: list[str] = [
    "custom10", "custom09", "custom08", "custom07", "custom06",
    "custom05", "custom04", "custom03", "custom02", "custom01",
]

# ----------------------- HELPERS -----------------------
def _now_ist() -> datetime:
    return datetime.now(IST)

def _ist_start_of_day(d: date) -> datetime:
    return IST.localize(datetime(d.year, d.month, d.day, 0, 0, 0))

def _safe_read_tf(path: str, tz=IST) -> pd.DataFrame:
    """
    Read CSV, parse 'date' as tz-aware IST, sort ascending.
    """
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column")

        d = pd.to_datetime(df["date"], errors="coerce")

        # If tz-aware, convert; else localize
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_convert(tz)
        else:
            d = d.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

        df["date"] = d
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"! Failed reading {path}: {e}")
        return pd.DataFrame()

def _prepare_tf_with_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Append suffix to all columns except 'date'."""
    if df.empty:
        return df
    rename_map = {c: f"{c}{suffix}" for c in df.columns if c != "date"}
    return df.rename(columns=rename_map)

def _list_tickers_from_dir(directory: str) -> set[str]:
    files = glob.glob(os.path.join(directory, "*_main_indicators*.csv"))
    tickers = set()
    for f in files:
        base = os.path.basename(f)
        if "_main_indicators" in base:
            t = base.split("_main_indicators")[0].strip()
            if t:
                tickers.add(t)
    return tickers

def _find_file(directory: str, ticker: str) -> str | None:
    hits = glob.glob(os.path.join(directory, f"{ticker}_main_indicators*.csv"))
    return hits[0] if hits else None

def _apply_asof_cut(df: pd.DataFrame, cutoff_dt_ist: datetime) -> pd.DataFrame:
    """Keep rows strictly before cutoff_dt_ist (IST)."""
    if df.empty or "date" not in df.columns:
        return df
    return df[df["date"] < cutoff_dt_ist].copy()

def _get_reference_ticker_for_calendar() -> str | None:
    """
    Pick any ticker from daily dir so we can infer which dates exist.
    """
    files = sorted(glob.glob(os.path.join(DIR_D, "*_main_indicators*.csv")))
    if not files:
        return None
    base = os.path.basename(files[0])
    if "_main_indicators" not in base:
        return None
    t = base.split("_main_indicators")[0].strip()
    return t or None

def _daily_has_bar_for_date(ticker: str, d: date) -> bool:
    """
    True if the daily CSV for ticker contains a bar whose IST date == d.
    """
    p = _find_file(DIR_D, ticker)
    if not p:
        return False
    df = _safe_read_tf(p)
    if df.empty:
        return False
    return (df["date"].dt.date == d).any()

def _resolve_trade_date(intended: date, max_lookahead_days: int = 14) -> tuple[date, str]:
    """
    Resolve to the next date that has a daily bar in your SAVED daily CSVs.
    (This does NOT fetch from broker; it uses what's in main_indicators_daily.)
    """
    ref_tkr = _get_reference_ticker_for_calendar()
    if not ref_tkr:
        raise RuntimeError(f"No daily files found in {DIR_D}. Cannot resolve trade date.")

    d = intended
    for step in range(max_lookahead_days + 1):
        if _daily_has_bar_for_date(ref_tkr, d):
            if step == 0:
                return d, "intended date has daily bar"
            return d, f"advanced by {step} day(s) until a daily bar existed (holiday/weekend/market-closed or CSV not updated)"
        d = d + timedelta(days=1)

    raise RuntimeError(
        f"Could not find any daily bar from {intended} up to {d} (lookahead {max_lookahead_days} days). "
        f"Daily CSVs likely stale for recent dates."
    )

def _prompt_trade_date(today: date) -> date:
    """
    Ask user for intended trade date. Accepts:
      - YYYY-MM-DD
      - blank -> today
    """
    while True:
        s = input(f"Enter intended TRADE DATE (IST) in YYYY-MM-DD [default {today}]: ").strip()
        if not s:
            return today
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (example: 2025-12-29).")

# ======================================================================
#                 MTF LOADER (DAILY + WEEKLY) AS-OF
# ======================================================================
def _load_mtf_context_daily_weekly_asof(ticker: str, asof_cutoff_ist: datetime) -> pd.DataFrame:
    """
    Returns a DAILY-indexed dataframe with WEEKLY columns merged in (as-of),
    using AS-OF CUT:
      - Daily: < asof_cutoff_ist
      - Weekly: < asof_cutoff_ist

    Weekly NO same-week lookahead:
      - weekly 'date' shifted by +1 day
      - merge_asof(backward)
    """
    path_d = _find_file(DIR_D, ticker)
    path_w = _find_file(DIR_W, ticker)
    if not path_d or not path_w:
        return pd.DataFrame()

    df_d = _safe_read_tf(path_d)
    df_w = _safe_read_tf(path_w)
    if df_d.empty or df_w.empty:
        return pd.DataFrame()

    # AS-OF CUTS (strictly before trade date start)
    df_d = _apply_asof_cut(df_d, asof_cutoff_ist)
    df_w = _apply_asof_cut(df_w, asof_cutoff_ist)
    if df_d.empty or df_w.empty:
        return pd.DataFrame()

    # Daily prev-day change (D-1)
    if "Daily_Change" in df_d.columns:
        df_d["Daily_Change_prev"] = pd.to_numeric(df_d["Daily_Change"], errors="coerce").shift(1)
    else:
        df_d["Daily_Change_prev"] = np.nan

    df_d = _prepare_tf_with_suffix(df_d, "_D")
    df_w = _prepare_tf_with_suffix(df_w, "_W")

    # Weekly NO same-week lookahead
    df_w_sorted = df_w.sort_values("date").copy()
    df_w_sorted["date"] = df_w_sorted["date"] + pd.Timedelta(days=1)

    df_dw = pd.merge_asof(
        df_d.sort_values("date"),
        df_w_sorted.sort_values("date"),
        on="date",
        direction="backward",
    )
    return df_dw.reset_index(drop=True)

# ======================================================================
#                 STRATEGY TEMPLATE (PARAMETRIC)
# ======================================================================
def _build_trade_date_signal_for_ticker(
    ticker: str,
    P: dict,
    trade_date_start_ist: datetime,
    asof_cutoff_ist: datetime
) -> dict | None:
    """
    Build a TRADE_DATE-ONLY signal (single row) for a ticker.
    Evaluate on latest available DAILY bar BEFORE asof_cutoff_ist.
    """
    df = _load_mtf_context_daily_weekly_asof(ticker, asof_cutoff_ist)
    if df.empty:
        return None

    cols = set(df.columns)
    need = {"date", "open_D", "high_D", "low_D", "close_D"}
    if not need.issubset(cols):
        return None

    # This is the LAST daily bar available strictly before trade date start
    r = df.iloc[-1].copy()

    def _get(name: str, default=np.nan):
        return r[name] if name in cols else default

    close = float(_get("close_D", np.nan))
    open_ = float(_get("open_D", np.nan))
    high  = float(_get("high_D", np.nan))
    if (not np.isfinite(close)) or close <= 0:
        return None

    # ------------------- TRADABILITY -------------------
    if P.get("min_price") is not None and close < float(P["min_price"]):
        return None

    # ------------------- WEEKLY FILTER -------------------
    if P.get("wk_close_above_ema200", True) and {"close_W", "EMA_200_W"}.issubset(cols):
        if not (_get("close_W") > _get("EMA_200_W")):
            return None

    if P.get("wk_rsi_min") is not None and ("RSI_W" in cols):
        if not (_get("RSI_W") >= float(P["wk_rsi_min"])):
            return None

    if P.get("wk_adx_min") is not None and ("ADX_W" in cols):
        if not (_get("ADX_W") >= float(P["wk_adx_min"])):
            return None

    # ------------------- DAILY PREV FILTER (D-1) -------------------
    dprev = _get("Daily_Change_prev_D", np.nan)
    if P.get("dprev_min") is not None and (not np.isfinite(dprev) or dprev < float(P["dprev_min"])):
        return None
    if P.get("dprev_max") is not None and (np.isfinite(dprev) and dprev > float(P["dprev_max"])):
        return None

    # ------------------- BREAKOUT LEVELS (NO LOOKAHEAD) -------------------
    high_s = pd.to_numeric(df["high_D"], errors="coerce")
    breakout_windows = P.get("breakout_windows", [20, 50, 100])
    prior_levels = [high_s.rolling(int(w), min_periods=int(w)).max().shift(1) for w in breakout_windows]

    breakout_level = prior_levels[0].iloc[-1]
    for i in range(1, len(prior_levels)):
        if pd.isna(breakout_level):
            breakout_level = prior_levels[i].iloc[-1]

    if not np.isfinite(breakout_level) or breakout_level <= 0:
        return None

    close_mult = float(P.get("breakout_close_mult", 1.0010))
    if not (close >= breakout_level * close_mult or high >= breakout_level * 1.0005):
        return None

    # ------------------- CANDLE QUALITY -------------------
    if not (close > open_):
        return None

    body_min_pct = float(P.get("body_min_pct", 0.0030))
    if not ((close - open_) >= (open_ * body_min_pct)):
        return None

    # ------------------- MOMENTUM BOOSTERS -------------------
    boosters_required = int(P.get("boosters_required", 2))
    boosters = 0

    if "RSI_D" in cols:
        rsi = float(pd.to_numeric(df["RSI_D"], errors="coerce").iloc[-1])
        if np.isfinite(rsi) and rsi >= float(P.get("rsi_min", 54.0)):
            boosters += 1

    if "MACD_Hist_D" in cols:
        macd = float(pd.to_numeric(df["MACD_Hist_D"], errors="coerce").iloc[-1])
        if np.isfinite(macd) and macd >= float(P.get("macd_hist_min", 0.0)):
            boosters += 1

    if "Daily_Change_D" in cols:
        chg = float(pd.to_numeric(df["Daily_Change_D"], errors="coerce").iloc[-1])
        if np.isfinite(chg) and chg >= float(P.get("day_change_min", 0.8)):
            boosters += 1

    if boosters < boosters_required:
        return None

    # ------------------- TREND ALIGNMENT -------------------
    if P.get("require_close_above_ema20", True) and ("EMA_20_D" in cols):
        ema20 = float(pd.to_numeric(df["EMA_20_D"], errors="coerce").iloc[-1])
        if np.isfinite(ema20) and close < ema20:
            return None

    return {
        "signal_time_ist": trade_date_start_ist,  # signal stamped at trade date start (IST)
        "asof_daily_bar": r["date"],              # last daily bar used (< trade date start)
        "ticker": ticker,
        "signal_side": "LONG",
        "entry_price": float(close),
        "strategy": P.get("name", "CUSTOMXX"),
        "note": "Trade-date signal: Daily strictly before trade date + Weekly previous completed week (shift+asof, no lookahead).",
    }

# ======================================================================
#                     10 POSITIONAL STRATEGIES
# ======================================================================
STRATEGY_PARAMS: dict[str, dict] = {
    "custom01": dict(
        name="CUSTOM01",
        wk_close_above_ema200=True,
        wk_rsi_min=45, wk_adx_min=None,
        dprev_min=0.0, dprev_max=6.0,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0010,
        body_min_pct=0.0030,
        require_close_above_ema20=True,
        boosters_required=2,
        rsi_min=54,
        macd_hist_min=0.0,
        day_change_min=0.8,
        min_price=20.0,
    ),
    "custom02": dict(
        name="CUSTOM02",
        wk_close_above_ema200=True,
        wk_rsi_min=48, wk_adx_min=16,
        dprev_min=0.0, dprev_max=5.5,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0012,
        body_min_pct=0.0035,
        require_close_above_ema20=True,
        boosters_required=2,
        rsi_min=56,
        macd_hist_min=0.0,
        day_change_min=1.0,
        min_price=25.0,
    ),
    "custom03": dict(
        name="CUSTOM03",
        wk_close_above_ema200=True,
        wk_rsi_min=50, wk_adx_min=18,
        dprev_min=0.2, dprev_max=5.0,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0015,
        body_min_pct=0.0040,
        require_close_above_ema20=True,
        boosters_required=2,
        rsi_min=57,
        macd_hist_min=0.0,
        day_change_min=1.2,
        min_price=30.0,
    ),
    "custom04": dict(
        name="CUSTOM04",
        wk_close_above_ema200=True,
        wk_rsi_min=52, wk_adx_min=20,
        dprev_min=0.3, dprev_max=4.8,
        breakout_windows=[20, 50, 100],
        breakout_close_mult=1.0018,
        body_min_pct=0.0045,
        require_close_above_ema20=True,
        boosters_required=3,
        rsi_min=58.5,
        macd_hist_min=0.0,
        day_change_min=1.4,
        min_price=40.0,
    ),
    "custom05": dict(
        name="CUSTOM05",
        wk_close_above_ema200=True,
        wk_rsi_min=54, wk_adx_min=22,
        dprev_min=0.4, dprev_max=4.5,
        breakout_windows=[20, 50, 120],
        breakout_close_mult=1.0020,
        body_min_pct=0.0050,
        require_close_above_ema20=True,
        boosters_required=3,
        rsi_min=60.0,
        macd_hist_min=0.0,
        day_change_min=1.6,
        min_price=50.0,
    ),
    "custom06": dict(name="CUSTOM06", wk_close_above_ema200=True, wk_rsi_min=55, wk_adx_min=23,
                     dprev_min=0.5, dprev_max=4.2, breakout_windows=[20, 60, 120],
                     breakout_close_mult=1.0022, body_min_pct=0.0052, require_close_above_ema20=True,
                     boosters_required=3, rsi_min=60.5, macd_hist_min=0.0, day_change_min=1.7, min_price=70.0),
    "custom07": dict(name="CUSTOM07", wk_close_above_ema200=True, wk_rsi_min=57, wk_adx_min=25,
                     dprev_min=0.6, dprev_max=4.0, breakout_windows=[20, 50, 150],
                     breakout_close_mult=1.0025, body_min_pct=0.0055, require_close_above_ema20=True,
                     boosters_required=3, rsi_min=61.5, macd_hist_min=0.0, day_change_min=1.8, min_price=80.0),
    "custom08": dict(name="CUSTOM08", wk_close_above_ema200=True, wk_rsi_min=58, wk_adx_min=26,
                     dprev_min=0.7, dprev_max=3.8, breakout_windows=[20, 50, 100],
                     breakout_close_mult=1.0028, body_min_pct=0.0060, require_close_above_ema20=True,
                     boosters_required=4, rsi_min=62.0, macd_hist_min=0.0, day_change_min=2.0, min_price=100.0),
    "custom09": dict(name="CUSTOM09", wk_close_above_ema200=True, wk_rsi_min=59, wk_adx_min=27,
                     dprev_min=0.8, dprev_max=3.6, breakout_windows=[20, 60, 120],
                     breakout_close_mult=1.0030, body_min_pct=0.0062, require_close_above_ema20=True,
                     boosters_required=4, rsi_min=63.0, macd_hist_min=0.0, day_change_min=2.2, min_price=120.0),
    "custom10": dict(name="CUSTOM10", wk_close_above_ema200=True, wk_rsi_min=60, wk_adx_min=28,
                     dprev_min=0.8, dprev_max=3.5, breakout_windows=[20, 50, 100],
                     breakout_close_mult=1.0032, body_min_pct=0.0065, require_close_above_ema20=True,
                     boosters_required=4, rsi_min=64.0, macd_hist_min=0.0, day_change_min=2.3, min_price=150.0),
}

def _parse_strategy_keys(argv: list[str]) -> list[str]:
    if not argv:
        return DEFAULT_STRATEGY_KEYS
    if len(argv) == 1 and argv[0].lower().strip() == "all":
        return sorted(STRATEGY_PARAMS.keys())
    selected = [a.lower().strip() for a in argv if a.lower().strip() in STRATEGY_PARAMS]
    if not selected:
        print("! No valid strategy keys provided; using defaults.")
        return DEFAULT_STRATEGY_KEYS
    return selected

def _latest_trade_output_if_exists(strategy_label: str, trade_ymd: str) -> str | None:
    patt = os.path.join(OUT_DIR, f"signals_trade_{strategy_label}_{trade_ymd}_*.csv")
    hits = sorted(glob.glob(patt), key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0] if hits else None

# ----------------------- MAIN -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trade-date",
        required=False,
        help="Intended trade date in YYYY-MM-DD (IST). If missing, script will ASK you."
    )
    parser.add_argument(
        "--max-lookahead",
        type=int,
        default=14,
        help="Max days to look forward to find the next date that has a daily bar (based on your saved CSVs)."
    )
    args, extra = parser.parse_known_args()

    now = _now_ist()
    today = now.date()

    # If not provided, ASK user for input
    if args.trade_date and args.trade_date.strip():
        intended = datetime.strptime(args.trade_date.strip(), "%Y-%m-%d").date()
    else:
        intended = _prompt_trade_date(today)

    strategy_keys = _parse_strategy_keys(extra)
    strategy_label = "+".join(strategy_keys)

    # Resolve trade date to next available daily bar date (in YOUR CSVs)
    trade_date, why = _resolve_trade_date(intended, max_lookahead_days=int(args.max_lookahead))
    trade_start_ist = _ist_start_of_day(trade_date)
    asof_cutoff_ist = trade_start_ist  # strict before trade day start
    trade_ymd = trade_start_ist.strftime("%Y%m%d")

    print(f"\nRUN TIME (IST):         {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"INTENDED TRADE DATE:    {intended} (IST)")
    print(f"RESOLVED TRADE DATE:    {trade_date} (IST)  -> {why}")
    print(f"SIGNALS TIMESTAMPED AT: {trade_start_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"AS-OF CUT:              using ONLY data before {asof_cutoff_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("                       (daily: last bar before trade date; weekly: prev completed week)\n")

    existing = _latest_trade_output_if_exists(strategy_label, trade_ymd)
    if existing:
        print(f"Found existing output for this trade date. Reusing: {existing}")
        try:
            df_existing = pd.read_csv(existing)
            print(f"Rows: {len(df_existing)}")
        except Exception as e:
            print(f"! Could not read existing file, will regenerate. Error: {e}")
        else:
            return

    tickers_d = _list_tickers_from_dir(DIR_D)
    tickers_w = _list_tickers_from_dir(DIR_W)
    tickers = sorted(tickers_d & tickers_w)
    if not tickers:
        print("No common tickers found across Daily and Weekly directories.")
        return

    print(f"Found {len(tickers)} tickers with both Daily+Weekly.")
    print(f"Running strategies: {strategy_label}\n")

    rows = []
    for i, ticker in enumerate(tickers, start=1):
        best_row = None
        for key in strategy_keys:
            P = STRATEGY_PARAMS.get(key.lower())
            if not P:
                continue
            sig = _build_trade_date_signal_for_ticker(
                ticker=ticker,
                P=P,
                trade_date_start_ist=trade_start_ist,
                asof_cutoff_ist=asof_cutoff_ist
            )
            if sig is not None:
                best_row = sig
                break

        if best_row is not None:
            rows.append(best_row)

        if i % 200 == 0:
            print(f"[{i}/{len(tickers)}] processed... trade_signals={len(rows)}")

    if not rows:
        print("No signals generated for RESOLVED TRADE DATE.")
        return

    out_df = pd.DataFrame(rows)
    out_df["signal_time_ist"] = pd.to_datetime(out_df["signal_time_ist"])
    out_df["signal_date"] = out_df["signal_time_ist"].dt.date

    # Hard guarantee: only resolved trade date
    out_df = out_df[out_df["signal_date"] == trade_date].copy()

    # One row per ticker for trade_date
    out_df = out_df.sort_values(["ticker", "strategy"]).drop_duplicates(
        subset=["ticker", "signal_date"], keep="first"
    )

    ts = now.strftime("%H%M%S")
    out_path = os.path.join(OUT_DIR, f"signals_trade_{strategy_label}_{trade_ymd}_{ts}_IST.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved trade-date signals: {out_path}")
    print(f"Rows: {len(out_df)}")
    print("\nSample:")
    print(out_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
