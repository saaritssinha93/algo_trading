import os
from datetime import datetime
import pandas as pd
import numpy as np
import pytz

# ---- CONFIG ----
IST = pytz.timezone("Asia/Kolkata")

SIGNALS_DIR = "signals_trade_date"
INTRADAY_DIR_5M = "main_indicators_5min"

# Toggle these to enable/disable SL & Target logic
USE_SL = True
USE_TARGET = True

# Percent values (for LONG)
STOP_PCT   = 2.0   # e.g. 1% SL
TARGET_PCT = 3.0   # e.g. 1% Target


def load_5min_data(ticker: str) -> pd.DataFrame:
    """
    Load 5-minute data for a ticker from:
        main_indicators_5min/{ticker}_main_indicators_5min.csv

    Ensures 'date' is tz-aware IST and sorted.
    """
    path = os.path.join(INTRADAY_DIR_5M, f"{ticker}_main_indicators_5min.csv")
    if not os.path.exists(path):
        print(f"[WARN] 5-min file missing for {ticker}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "date" not in df.columns:
        print(f"[WARN] No 'date' column in 5-min file for {ticker}: {path}")
        return pd.DataFrame()

    # Parse timestamp and convert/localize to IST
    dt = pd.to_datetime(df["date"], errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(IST)
    else:
        dt = dt.dt.tz_convert(IST)

    df["date"] = dt
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def evaluate_signal_flexible(row: pd.Series, df_5m: pd.DataFrame) -> dict:
    """
    Flexible TP/SL evaluator (LONG only) controlled by USE_SL / USE_TARGET:

      - Entry: row["entry_price"] at row["signal_time_ist"]
      - Same calendar day (IST) only
      - Walk forward bar-by-bar (date > signal_time_ist):

        Logic per config:
          * If USE_SL and low <= entry*(1-STOP_PCT/100)  → exit at SL on that bar
          * If USE_TARGET and high >= entry*(1+TARGET_PCT/100) → exit at TARGET on that bar
          * If both enabled and both hit in same bar -> assume SL first (conservative)
          * If neither condition enabled -> skip scanning, just time exit

      - If no enabled condition hits till day's last bar
            -> exit at last bar close ("TIME")

    Returns dict with:
        exit_time, exit_price, pnl_abs, pnl_pct, exit_reason,
        sl_hit (bool), target_hit (bool)
    """
    # Parse signal time as IST
    sig_time = pd.to_datetime(row["signal_time_ist"], errors="coerce")
    if sig_time.tzinfo is None:
        sig_time = sig_time.tz_localize(IST)
    else:
        sig_time = sig_time.tz_convert(IST)

    entry_price = float(row["entry_price"])
    if (not np.isfinite(entry_price)) or entry_price <= 0 or df_5m.empty:
        return {
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "pnl_abs": 0.0,
            "pnl_pct": 0.0,
            "exit_reason": "INVALID",
            "sl_hit": False,
            "target_hit": False,
        }

    sig_day = sig_time.date()

    # Same calendar day in 5-min data
    day_df = df_5m[df_5m["date"].dt.date == sig_day].copy()
    if day_df.empty:
        # No data for that day → treat as no move
        return {
            "exit_time": pd.NaT,
            "exit_price": entry_price,
            "pnl_abs": 0.0,
            "pnl_pct": 0.0,
            "exit_reason": "NO_DATA",
            "sl_hit": False,
            "target_hit": False,
        }

    # Only bars strictly AFTER signal time
    future = day_df[day_df["date"] > sig_time].copy()
    if future.empty:
        # No bar after signal -> exit at entry (no move)
        return {
            "exit_time": sig_time,
            "exit_price": entry_price,
            "pnl_abs": 0.0,
            "pnl_pct": 0.0,
            "exit_reason": "NO_FUTURE_BARS",
            "sl_hit": False,
            "target_hit": False,
        }

    # If neither SL nor Target is enabled → straight time exit
    if not USE_SL and not USE_TARGET:
        last_bar = future.iloc[-1]
        exit_price = float(last_bar.get("close", entry_price))
        pnl_abs = exit_price - entry_price
        pnl_pct = (pnl_abs / entry_price) * 100.0 if entry_price > 0 else 0.0
        return {
            "exit_time": last_bar["date"],
            "exit_price": exit_price,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "exit_reason": "TIME",
            "sl_hit": False,
            "target_hit": False,
        }

    # Ensure needed columns exist when any TP/SL logic is enabled
    if not {"high", "low", "close"}.issubset(future.columns):
        # Can't run TP/SL without OHLC -> treat as time-based exit
        last_bar = future.iloc[-1]
        exit_price = float(last_bar.get("close", entry_price))
        pnl_abs = exit_price - entry_price
        pnl_pct = (pnl_abs / entry_price) * 100.0 if entry_price > 0 else 0.0
        return {
            "exit_time": last_bar["date"],
            "exit_price": exit_price,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "exit_reason": "NO_OHLC",
            "sl_hit": False,
            "target_hit": False,
        }

    # Pre-compute levels (only used if corresponding flag is True)
    if USE_TARGET:
        target_price = entry_price * (1.0 + TARGET_PCT / 100.0)
    else:
        target_price = None

    if USE_SL:
        sl_price = entry_price * (1.0 - STOP_PCT / 100.0)
    else:
        sl_price = None

    # Default → time-based exit at last bar (if no TP/SL hit)
    last_bar = future.iloc[-1]
    exit_time = last_bar["date"]
    exit_price = float(last_bar["close"])
    exit_reason = "TIME"
    sl_hit_flag = False
    target_hit_flag = False

    # Walk forward to find first enabled TP/SL
    for _, bar in future.iterrows():
        bar_high = float(bar["high"])
        bar_low  = float(bar["low"])

        sl_hit_cond = USE_SL and (bar_low <= sl_price)
        target_hit_cond = USE_TARGET and (bar_high >= target_price)

        # Both enabled & both hit in same bar → assume SL first
        if sl_hit_cond and target_hit_cond:
            exit_time = bar["date"]
            exit_price = sl_price
            exit_reason = "SL"
            sl_hit_flag = True
            target_hit_flag = False
            break
        elif sl_hit_cond:
            exit_time = bar["date"]
            exit_price = sl_price
            exit_reason = "SL"
            sl_hit_flag = True
            target_hit_flag = False
            break
        elif target_hit_cond:
            exit_time = bar["date"]
            exit_price = target_price
            exit_reason = "TARGET"
            sl_hit_flag = False
            target_hit_flag = True
            break

    pnl_abs = exit_price - entry_price
    pnl_pct = (pnl_abs / entry_price) * 100.0 if entry_price > 0 else 0.0

    return {
        "exit_time": exit_time,
        "exit_price": exit_price,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "sl_hit": sl_hit_flag,
        "target_hit": target_hit_flag,
    }


def main():
    # --- Today ymd string (based on IST) ---
    today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    ymd = today_start.strftime("%Y%m%d")

    signals_path = os.path.join(SIGNALS_DIR, f"intraday_entries_{ymd}.csv")
    if not os.path.exists(signals_path):
        print(f"[ERROR] Signals file not found: {signals_path}")
        return

    print(f"Reading signals from: {signals_path}")
    sig_df = pd.read_csv(signals_path)

    required_cols = [
        "ticker",
        "timeframe",
        "signal_time_ist",
        "entry_price",
        "reason",
        "rsi",
        "adx",
        "macd_hist",
        "intra_change",
        "daily_change",
        "signal_date",
    ]
    missing = [c for c in required_cols if c not in sig_df.columns]
    if missing:
        print(f"[ERROR] Missing columns in signals CSV: {missing}")
        return

    # Filter only 5min timeframe signals
    sig_5m = sig_df[sig_df["timeframe"].astype(str).str.lower() == "5min"].copy()
    if sig_5m.empty:
        print("No 5min signals found in today's file.")
        return

    print(f"Found {len(sig_5m)} 5min signals.")

    # Pre-load 5min data per ticker
    tickers = sorted(sig_5m["ticker"].dropna().unique())
    data_5m = {}
    for t in tickers:
        df_5m = load_5min_data(t)
        if df_5m.empty:
            print(f"[WARN] Skipping ticker with no usable 5-min data: {t}")
        data_5m[t] = df_5m

    # Evaluate each signal with flexible TP/SL logic
    exit_times = []
    exit_prices = []
    pnl_abs_list = []
    pnl_pct_list = []
    exit_reasons = []
    sl_hits = []
    target_hits = []

    for _, row in sig_5m.iterrows():
        t = row["ticker"]
        df_5m = data_5m.get(t, pd.DataFrame())
        result = evaluate_signal_flexible(row, df_5m)

        exit_times.append(result["exit_time"])
        exit_prices.append(result["exit_price"])
        pnl_abs_list.append(result["pnl_abs"])
        pnl_pct_list.append(result["pnl_pct"])
        exit_reasons.append(result["exit_reason"])
        sl_hits.append(result["sl_hit"])
        target_hits.append(result["target_hit"])

    # Attach results
    sig_5m["exit_time"] = exit_times
    sig_5m["exit_price"] = exit_prices
    sig_5m["pnl_abs"] = pnl_abs_list
    sig_5m["pnl_pct"] = pnl_pct_list
    sig_5m["exit_reason"] = exit_reasons
    sig_5m["sl_hit"] = sl_hits
    sig_5m["target_hit"] = target_hits

    out_df = sig_5m.copy()

    # ---- Net P&L calculations ----
    valid = out_df[np.isfinite(out_df["pnl_abs"]) & np.isfinite(out_df["pnl_pct"])]

    net_pnl_abs = valid["pnl_abs"].sum()
    # Equal-weighted average % P&L across all signals
    net_pnl_pct = valid["pnl_pct"].mean() if not valid.empty else 0.0

    n_target = (valid["exit_reason"] == "TARGET").sum()
    n_sl     = (valid["exit_reason"] == "SL").sum()
    n_time   = (valid["exit_reason"] == "TIME").sum()

    print("\n=== SUMMARY (5min signals, flexible TP/SL) ===")
    print(f"USE_SL={USE_SL}, USE_TARGET={USE_TARGET}")
    print(f"STOP_PCT={STOP_PCT:.2f}%, TARGET_PCT={TARGET_PCT:.2f}%")
    print(f"Total signals evaluated: {len(valid)}")
    print(f"  TARGET hits : {n_target}")
    print(f"  SL hits     : {n_sl}")
    print(f"  TIME exits  : {n_time}")
    print(f"\nNet P&L (absolute price units): {net_pnl_abs:.4f}")
    print(f"Net P&L (% - equal-weighted avg): {net_pnl_pct:.2f}%")

    # Save evaluated file
    out_path = os.path.join(
        SIGNALS_DIR,
        f"intraday_entries_{ymd}_evaluated_flexible_tp_sl.csv"
    )
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved evaluated P&L to: {out_path}")

    # Small preview
    print("\nSample rows:")
    print(
        out_df[
            [
                "ticker",
                "signal_time_ist",
                "entry_price",
                "exit_time",
                "exit_price",
                "exit_reason",
                "sl_hit",
                "target_hit",
                "pnl_abs",
                "pnl_pct",
            ]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
