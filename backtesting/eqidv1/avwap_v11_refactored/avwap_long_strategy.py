# -*- coding: utf-8 -*-
"""
avwap_long_strategy.py — LONG side logic for AVWAP v11
======================================================

Only contains direction-specific code:
- Green impulse classification
- AVWAP support checks (price above AVWAP)
- Long exit simulation
- Long-specific signal validation
- scan_one_day / scan_all_days entry points

All shared infrastructure comes from avwap_common.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from avwap_v11_refactored.avwap_common import (
    IST,
    StrategyConfig,
    Trade,
    default_long_config,
    read_15m_parquet,
    list_tickers_15m,
    in_session,
    in_signal_window,
    entry_buffer,
    prepare_indicators,
    compute_day_avwap,
    twice_increasing,
    twice_reducing,
    adx_slope_ok,
    max_consecutive_true,
    compute_quality_score_long,
    compute_pnl_pct,
    trades_to_df,
    apply_topn_per_day,
    volume_filter_pass,
)


# ===========================================================================
# IMPULSE CLASSIFICATION (GREEN CANDLES)
# ===========================================================================
def classify_green_impulse(row: pd.Series, cfg: StrategyConfig) -> str:
    o = float(row["open"])
    c = float(row["close"])
    h = float(row["high"])
    low = float(row["low"])
    atr = float(row["ATR15"])

    if not np.isfinite(atr) or atr <= 0:
        return ""
    if c <= o:  # must be green
        return ""

    body = abs(c - o)
    rng = h - low
    if not np.isfinite(rng) or rng <= 0:
        return ""

    close_near_high = ((h - c) / rng) <= cfg.close_near_extreme_max

    if (body >= cfg.huge_impulse_min_atr * atr) or (
        rng >= cfg.huge_impulse_min_range_atr * atr
    ):
        return "HUGE"

    if (
        body >= cfg.mod_impulse_min_atr * atr
        and body <= cfg.mod_impulse_max_atr * atr
        and close_near_high
    ):
        return "MODERATE"

    return ""


# ===========================================================================
# AVWAP SUPPORT (LONG: price must be ABOVE AVWAP, with support evidence)
# ===========================================================================
def avwap_support_pass(
    df_day: pd.DataFrame,
    impulse_idx: int,
    entry_idx: int,
    atr_entry: float,
    cfg: StrategyConfig,
) -> tuple:
    """
    Returns (ok: bool, avwap_dist_atr: float).
    Evidence: touch-support (low <= AVWAP and close > AVWAP) + consecutive closes above.
    """
    if not cfg.require_avwap_rule:
        return True, 0.0
    if entry_idx <= impulse_idx:
        return False, 0.0

    seg = df_day.iloc[impulse_idx + 1 : entry_idx + 1].copy()
    if seg.empty:
        return False, 0.0

    av = pd.to_numeric(seg["AVWAP"], errors="coerce")
    lo = pd.to_numeric(seg["low"], errors="coerce")
    cl = pd.to_numeric(seg["close"], errors="coerce")

    # Touch support: low <= AVWAP and close > AVWAP
    touch_ok = False
    if cfg.avwap_touch:
        touch_ok = bool(((lo <= av) & (cl > av)).fillna(False).any())

    # Consecutive closes above AVWAP
    consec_ok = False
    n = cfg.avwap_min_consec_closes
    if n > 0:
        consec_ok = _count_max_consec_above(cl, av) >= n

    mode = cfg.avwap_mode.strip().lower()
    if mode == "both":
        evidence_ok = touch_ok and consec_ok
    else:
        evidence_ok = touch_ok or consec_ok

    # Distance from AVWAP at entry
    entry_close = float(df_day.at[entry_idx, "close"])
    entry_avwap = float(df_day.at[entry_idx, "AVWAP"])
    if not (
        np.isfinite(entry_close)
        and np.isfinite(entry_avwap)
        and np.isfinite(atr_entry)
        and atr_entry > 0
    ):
        return False, 0.0

    avwap_dist = entry_close - entry_avwap
    avwap_dist_atr = avwap_dist / atr_entry
    dist_ok = avwap_dist >= (cfg.avwap_dist_atr_mult * atr_entry)

    return bool(evidence_ok and dist_ok), float(avwap_dist_atr)


def _count_max_consec_above(close_s: pd.Series, avwap_s: pd.Series) -> int:
    cond = (close_s > avwap_s).fillna(False).astype(bool).tolist()
    best = 0
    cur = 0
    for v in cond:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ===========================================================================
# EXIT SIMULATION (LONG)
# ===========================================================================
def simulate_exit_long(
    df_day: pd.DataFrame, entry_idx: int, entry_price: float, cfg: StrategyConfig
) -> tuple:
    """Walk forward within day until TARGET / SL / BE / TRAIL / EOD.

    Exit priority: SL wins ties over TARGET (conservative).
    Trailing stop: after BE trigger, trail from best (highest) price seen.
    """
    sl = entry_price * (1.0 - cfg.stop_pct)
    tgt = entry_price * (1.0 + cfg.target_pct)

    sl_curr = float(sl)
    be_armed = False
    be_level = entry_price * (1.0 + cfg.be_pad_pct)
    best_price = entry_price  # best (highest) price seen for long

    for k in range(entry_idx + 1, len(df_day)):
        hi = float(df_day.at[k, "high"])
        lo = float(df_day.at[k, "low"])
        ts = df_day.at[k, "date"]

        # Track best favorable price (highest for LONG)
        if np.isfinite(hi) and hi > best_price:
            best_price = hi

        if (
            cfg.enable_breakeven
            and not be_armed
            and np.isfinite(hi)
            and hi >= entry_price * (1.0 + cfg.be_trigger_pct)
        ):
            be_armed = True
            sl_curr = max(sl_curr, be_level)

        # Trailing stop: after BE armed, trail from best price
        if be_armed and cfg.enable_trailing_stop:
            trail_sl = best_price * (1.0 - cfg.trail_pct)
            sl_curr = max(sl_curr, trail_sl)  # trailing stop can only move up (tighter)

        hit_sl = np.isfinite(lo) and lo <= sl_curr
        hit_tg = np.isfinite(hi) and hi >= tgt

        if hit_sl and hit_tg:
            if be_armed:
                return k, ts, float(sl_curr), "BE"
            return k, ts, float(sl_curr), "SL"
        if hit_sl:
            if be_armed:
                return k, ts, float(sl_curr), "BE"
            return k, ts, float(sl_curr), "SL"
        if hit_tg:
            return k, ts, float(tgt), "TARGET"

    last = len(df_day) - 1
    return last, df_day.at[last, "date"], float(df_day.at[last, "close"]), "EOD"


# ===========================================================================
# TREND FILTER VALIDATION (LONG)
# ===========================================================================
def _trend_filter_long(
    df_day: pd.DataFrame, i: int, c1: pd.Series, cfg: StrategyConfig
) -> bool:
    """Returns True if all Option-A trend conditions pass for LONG."""
    adx1 = float(df_day.at[i, "ADX15"]) if "ADX15" in df_day.columns else np.nan
    rsi1 = float(df_day.at[i, "RSI15"]) if "RSI15" in df_day.columns else np.nan
    k1 = float(df_day.at[i, "STOCHK15"]) if "STOCHK15" in df_day.columns else np.nan
    d1 = float(df_day.at[i, "STOCHD15"]) if "STOCHD15" in df_day.columns else np.nan

    adx_ok = (
        np.isfinite(adx1)
        and adx1 >= cfg.adx_min
        and twice_increasing(df_day, i, "ADX15")
        and adx_slope_ok(df_day, i, "ADX15", cfg.adx_slope_min)
    )
    rsi_ok = (
        np.isfinite(rsi1)
        and rsi1 >= cfg.rsi_min_long
        and twice_increasing(df_day, i, "RSI15")
    )
    stoch_ok = (
        np.isfinite(k1)
        and np.isfinite(d1)
        and k1 >= cfg.stochk_min
        and k1 <= cfg.stochk_max
        and k1 > d1
        and twice_increasing(df_day, i, "STOCHK15")
    )

    if not (adx_ok and rsi_ok and stoch_ok):
        return False

    # Strict EMA + AVWAP for LONG
    close1 = float(c1["close"])
    ema20 = float(c1["EMA20"]) if np.isfinite(c1.get("EMA20", np.nan)) else np.nan
    ema50 = float(c1["EMA50"]) if np.isfinite(c1.get("EMA50", np.nan)) else np.nan
    avwap1 = float(c1["AVWAP"]) if np.isfinite(c1.get("AVWAP", np.nan)) else np.nan

    if not (np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(avwap1)):
        return False

    return (ema20 > ema50) and (close1 > ema20) and (close1 > avwap1)


# ===========================================================================
# SCAN ONE DAY (LONG)
# ===========================================================================
def scan_one_day(
    ticker: str, df_day: pd.DataFrame, day_str: str, cfg: StrategyConfig
) -> List[Trade]:
    if len(df_day) < 7:
        return []

    trades: List[Trade] = []
    i = 2

    while i < len(df_day) - 3:
        if len(trades) >= cfg.max_trades_per_ticker_per_day:
            break

        c1 = df_day.iloc[i]

        if not in_signal_window(c1["date"], cfg):
            i += 1
            continue

        impulse = classify_green_impulse(c1, cfg)
        if impulse == "":
            i += 1
            continue

        # ATR sanity
        atr1 = float(c1["ATR15"]) if np.isfinite(c1.get("ATR15", np.nan)) else np.nan
        close1 = float(c1["close"])
        if not (np.isfinite(atr1) and atr1 > 0 and np.isfinite(close1) and close1 > 0):
            i += 1
            continue

        if cfg.use_atr_pct_filter and (atr1 / close1) < cfg.atr_pct_min:
            i += 1
            continue

        # Volume confirmation: impulse bar should have above-average volume
        if not volume_filter_pass(c1, cfg):
            i += 1
            continue

        if not _trend_filter_long(df_day, i, c1, cfg):
            i += 1
            continue

        # Diagnostic values
        adx1 = float(df_day.at[i, "ADX15"]) if "ADX15" in df_day.columns else 0.0
        rsi1 = float(df_day.at[i, "RSI15"]) if "RSI15" in df_day.columns else 0.0
        k1 = float(df_day.at[i, "STOCHK15"]) if "STOCHK15" in df_day.columns else 0.0
        ema20 = float(c1["EMA20"]) if np.isfinite(c1.get("EMA20", np.nan)) else 0.0
        atr_pct = atr1 / close1

        signal_time = c1["date"]
        high1 = float(c1["high"])
        open1 = float(c1["open"])

        def _bars_left_ok(eidx: int) -> bool:
            return (len(df_day) - 1 - eidx) >= cfg.min_bars_left_after_entry

        def _close_confirm_ok(eidx: int, trigger: float) -> bool:
            if not cfg.require_entry_close_confirm:
                return True
            cl = float(df_day.at[eidx, "close"])
            return np.isfinite(cl) and cl > trigger

        def _make_trade(
            entry_idx: int, entry_price: float, setup: str, avwap_dist_atr: float
        ) -> tuple:
            exit_idx, exit_time, exit_price, outcome = simulate_exit_long(
                df_day, entry_idx, entry_price, cfg
            )
            net_pnl, gross_pnl = compute_pnl_pct(entry_price, exit_price, "LONG", cfg)
            adx_slope2 = (
                float(df_day.at[i, "ADX15"] - df_day.at[i - 2, "ADX15"]) if i >= 2 else 0.0
            )
            ema_gap_atr = (close1 - ema20) / atr1 if atr1 > 0 else 0.0
            qscore = compute_quality_score_long(
                adx1, adx_slope2, avwap_dist_atr, ema_gap_atr, impulse
            )
            return (
                Trade(
                    trade_date=day_str,
                    ticker=ticker,
                    side="LONG",
                    setup=setup,
                    impulse_type=impulse,
                    signal_time_ist=signal_time,
                    entry_time_ist=df_day.at[entry_idx, "date"],
                    entry_price=entry_price,
                    sl_price=entry_price * (1.0 - cfg.stop_pct),
                    target_price=entry_price * (1.0 + cfg.target_pct),
                    exit_time_ist=exit_time,
                    exit_price=exit_price,
                    outcome=outcome,
                    pnl_pct=net_pnl,
                    pnl_pct_gross=gross_pnl,
                    adx_signal=adx1,
                    rsi_signal=rsi1,
                    stochk_signal=k1,
                    avwap_dist_atr_signal=avwap_dist_atr,
                    ema20_gap_atr_signal=ema_gap_atr,
                    atr_pct_signal=atr_pct,
                    quality_score=qscore,
                ),
                exit_idx,
            )

        # ---------------------------------------------------------------
        # SETUP A (MODERATE): break C1 high, or pullback + break C2 high
        # ---------------------------------------------------------------
        if impulse == "MODERATE":
            c2 = df_day.iloc[i + 1]
            buf1 = entry_buffer(high1, cfg)
            trigger = high1 + buf1

            # Option 1: break C1 high on C2
            if float(c2["high"]) > trigger:
                entry_idx = i + 1
                entry_ts = df_day.at[entry_idx, "date"]

                if not in_signal_window(entry_ts, cfg):
                    i += 1
                    continue
                if not _close_confirm_ok(entry_idx, trigger):
                    i += 1
                    continue
                if not _bars_left_ok(entry_idx):
                    i += 1
                    continue

                atr_entry = float(df_day.at[entry_idx, "ATR15"])
                ok_support, avwap_dist_atr = avwap_support_pass(
                    df_day, i, entry_idx, atr_entry, cfg
                )
                if not ok_support:
                    i += 1
                    continue

                trade, exit_idx = _make_trade(
                    entry_idx, trigger, "A_MOD_BREAK_C1_HIGH", avwap_dist_atr
                )
                trades.append(trade)
                i = exit_idx + 1
                continue

            # Option 2 (optional): small red pullback C2 (above AVWAP), then break C2 high on C3
            # Controlled by cfg.enable_setup_a_pullback_c2_break (default False)
            c2o, c2c = float(c2["open"]), float(c2["close"])
            c2_body = abs(c2c - c2o)
            c2_atr = float(c2["ATR15"]) if np.isfinite(c2["ATR15"]) else atr1
            c2_avwap = float(c2["AVWAP"]) if np.isfinite(c2["AVWAP"]) else np.nan

            c2_small_red = (
                (c2c < c2o)
                and np.isfinite(c2_atr)
                and (c2_body <= cfg.small_counter_max_atr * c2_atr)
            )
            c2_above_avwap = np.isfinite(c2_avwap) and (c2c > c2_avwap)

            if cfg.enable_setup_a_pullback_c2_break and c2_small_red and c2_above_avwap and (i + 2 < len(df_day)):
                c3 = df_day.iloc[i + 2]
                high2 = float(c2["high"])
                buf2 = entry_buffer(high2, cfg)
                trigger2 = high2 + buf2

                if float(c3["high"]) > trigger2:
                    entry_idx = i + 2
                    entry_ts = df_day.at[entry_idx, "date"]

                    if not in_signal_window(entry_ts, cfg):
                        i += 1
                        continue
                    if not _close_confirm_ok(entry_idx, trigger2):
                        i += 1
                        continue
                    if not _bars_left_ok(entry_idx):
                        i += 1
                        continue

                    atr_entry = float(df_day.at[entry_idx, "ATR15"])
                    ok_support, avwap_dist_atr = avwap_support_pass(
                        df_day, i, entry_idx, atr_entry, cfg
                    )
                    if not ok_support:
                        i += 1
                        continue

                    trade, exit_idx = _make_trade(
                        entry_idx, trigger2, "A_PULLBACK_C2_THEN_BREAK_C2_HIGH", avwap_dist_atr
                    )
                    trades.append(trade)
                    i = exit_idx + 1
                    continue

            i += 1
            continue

        # ---------------------------------------------------------------
        # SETUP B (HUGE): pullback holds, then break pullback high
        # ---------------------------------------------------------------
        if impulse == "HUGE":
            pull_end = min(i + 3, len(df_day) - 1)
            pull = df_day.iloc[i + 1 : pull_end + 1].copy()
            if pull.empty:
                i += 1
                continue

            mid_body = (open1 + close1) / 2.0

            pull_atr = pd.to_numeric(pull["ATR15"], errors="coerce").fillna(atr1)
            pull_body = (
                pd.to_numeric(pull["close"], errors="coerce")
                - pd.to_numeric(pull["open"], errors="coerce")
            ).abs()
            pull_red = pd.to_numeric(pull["close"], errors="coerce") < pd.to_numeric(
                pull["open"], errors="coerce"
            )
            pull_small = pull_body <= (cfg.small_counter_max_atr * pull_atr)

            if not bool((pull_red & pull_small).any()):
                i += 1
                continue

            lows = pd.to_numeric(pull["low"], errors="coerce")
            closes = pd.to_numeric(pull["close"], errors="coerce")
            avwaps = pd.to_numeric(pull["AVWAP"], errors="coerce")

            hold_mid = bool((lows > mid_body).fillna(False).all())
            hold_avwap = bool((closes > avwaps).fillna(False).all())
            if not (hold_mid or hold_avwap):
                i += 1
                continue

            pull_high = float(pd.to_numeric(pull["high"], errors="coerce").max())
            if not np.isfinite(pull_high):
                i += 1
                continue

            trigger = pull_high + entry_buffer(pull_high, cfg)
            entered = False

            for j in range(pull_end + 1, len(df_day)):
                tsj = df_day.at[j, "date"]
                if not in_signal_window(tsj, cfg):
                    continue

                closej = float(df_day.at[j, "close"])
                avwapj = (
                    float(df_day.at[j, "AVWAP"])
                    if np.isfinite(df_day.at[j, "AVWAP"])
                    else np.nan
                )
                if np.isfinite(avwapj) and closej <= avwapj:
                    break

                if float(df_day.at[j, "high"]) > trigger:
                    if not _close_confirm_ok(j, trigger):
                        continue
                    if not _bars_left_ok(j):
                        continue

                    atr_entry = float(df_day.at[j, "ATR15"])
                    ok_support, avwap_dist_atr = avwap_support_pass(
                        df_day, i, j, atr_entry, cfg
                    )
                    if not ok_support:
                        continue

                    trade, exit_idx = _make_trade(
                        j, trigger, "B_HUGE_GREEN_PULLBACK_HOLD_THEN_BREAK", avwap_dist_atr
                    )
                    trades.append(trade)
                    i = exit_idx + 1
                    entered = True
                    break

            if entered:
                continue
            i += 1
            continue

        i += 1

    return trades


# ===========================================================================
# SCAN ALL DAYS FOR ONE TICKER
# ===========================================================================
def scan_all_days_for_ticker(
    ticker: str, df_full: pd.DataFrame, cfg: StrategyConfig
) -> List[Trade]:
    if df_full.empty:
        return []

    for c in ["open", "high", "low", "close"]:
        if c not in df_full.columns:
            return []

    df = df_full[df_full["date"].apply(lambda ts: in_session(ts, cfg))].copy()
    if df.empty:
        return []

    df = df.sort_values("date").reset_index(drop=True)
    df = prepare_indicators(df, cfg)

    all_trades: List[Trade] = []
    for day_val, df_day in df.groupby("day", sort=True):
        df_day = df_day.copy().reset_index(drop=True)
        if len(df_day) < 7:
            continue
        df_day["AVWAP"] = compute_day_avwap(df_day)
        trades = scan_one_day(ticker, df_day, str(day_val), cfg)
        if trades:
            all_trades.extend(trades)

    return all_trades


# ===========================================================================
# RUN ALL TICKERS (called by combined runner)
# ===========================================================================
def run_long_scan(cfg: Optional[StrategyConfig] = None) -> List[Trade]:
    if cfg is None:
        cfg = default_long_config()

    tickers = list_tickers_15m(cfg.dir_15m, cfg.end_15m)
    print(f"[LONG] Tickers found: {len(tickers)}")

    all_trades: List[Trade] = []
    for k, t in enumerate(tickers, start=1):
        path = os.path.join(cfg.dir_15m, f"{t}{cfg.end_15m}")
        df = read_15m_parquet(path, cfg.parquet_engine)
        if df.empty:
            continue

        trades = scan_all_days_for_ticker(t, df, cfg)
        if trades:
            all_trades.extend(trades)

        if k % 50 == 0:
            print(f"  [LONG] scanned {k}/{len(tickers)} | trades_so_far={len(all_trades)}")

    return all_trades
