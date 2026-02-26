# -*- coding: utf-8 -*-
"""
Read signals_{ymd}_*.csv from signals_trade_date/ and, for those tickers only,
scan 5min and 15min data (main_indicators_5min/, main_indicators_15min/)
to generate intraday entry points for TODAY (IST).

Supports TWO entry styles:
    • aggressive_breakout
    • pullback

Outputs:
  signals_trade_date/intraday_entries_{ymd}.csv
"""

import os
import glob
from datetime import datetime, time
import pandas as pd
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")

SIGNALS_DIR = "signals_trade_date"
DIR_5M = "main_indicators_5min"
DIR_15M = "main_indicators_15min"

# Trading session window
SESSION_START = time(9, 20)
SESSION_END   = time(15, 20)


def _today_start_ist() -> pd.Timestamp:
    now_ist = pd.Timestamp.now(tz=IST)
    return now_ist.normalize()


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_signals_today(ymd: str) -> pd.DataFrame:
    pattern = os.path.join(SIGNALS_DIR, f"signals_{ymd}_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
            d["__source_file"] = os.path.basename(p)
            dfs.append(d)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")

    if not dfs:
        raise RuntimeError("Signals files found but none could be read.")

    out = pd.concat(dfs, ignore_index=True)

    # Normalize ticker col
    if "ticker" not in out.columns:
        raise KeyError("signals_trade_date files must include a 'ticker' column")

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"] != ""]

    return out


def _find_ticker_file(directory: str, ticker: str) -> str | None:
    """
    Looks for files like:
      {directory}/{ticker}_main_indicators_5min.csv
      {directory}/{ticker}_main_indicators_*.csv
    """
    candidates = glob.glob(os.path.join(directory, f"{ticker}_main_indicators_*.csv"))
    if not candidates:
        return None
    # Prefer the shortest name (often the canonical one)
    candidates = sorted(
        candidates, key=lambda x: (len(os.path.basename(x)), os.path.basename(x))
    )
    return candidates[0]


def _safe_read_main_indicators(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    # If date is naive, treat as IST
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        if getattr(df["date"].dt, "tz", None) is None:
            df["date"] = df["date"].dt.tz_localize(
                IST, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            df["date"] = df["date"].dt.tz_convert(IST)

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure date_only exists
    if "date_only" not in df.columns:
        df["date_only"] = df["date"].dt.date

    # Numeric coercions (only those we use)
    df = _coerce_numeric(
        df,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "RSI",
            "ADX",
            "MACD_Hist",
            "VWAP",
            "Recent_High",
            "Recent_Low",
            "Intra_Change",
            "Prev_Day_Close",
            "Daily_Change",
            "EMA_50",
            "EMA_200",
            "Upper_Band",
            "Lower_Band",
        ],
    )

    return df


def _filter_today_session(df: pd.DataFrame, today_start_ist: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df

    today_end_ist = today_start_ist + pd.Timedelta(days=1)
    d = df[(df["date"] >= today_start_ist) & (df["date"] < today_end_ist)].copy()
    if d.empty:
        return d

    # Session filter
    t = d["date"].dt.time
    d = d[(t >= SESSION_START) & (t <= SESSION_END)].copy()
    return d

def _entry_scan_pullback_breakout(
    df_today: pd.DataFrame, ticker: str, tf_label: str
) -> dict | None:
    """
    Main entry logic: finds either
        • aggressive_breakout
        • pullback
    in a moderately strong uptrend.

    RELAXED + EXACT VERSION

    Common trend filter (must pass):
        - EMA_50, EMA_200, close present & non-NaN
        - EMA_50 > EMA_200
        - close >= EMA_50 * 0.995       (allow tiny dip around EMA_50)
        - If RSI exists: RSI >= 45
        - If ADX exists: ADX >= 12

    Aggressive breakout (relaxed, but directional):
        - requires Recent_High
        - close >= Recent_High * 1.0005
        - if VWAP exists: close >= VWAP * 1.000
        - if Daily_Change exists: Daily_Change >= +0.7%
        - if Intra_Change exists: Intra_Change >= +0.25%
        - if RSI exists: RSI >= 52
        - if ADX exists: ADX >= 18
        - if MACD_Hist exists: MACD_Hist >= 0
        - if volume + vol_ma20 available: volume >= 1.10 * vol_ma20

    Pullback in uptrend (much more relaxed, still bullish bias):
        - Uptrend: trend_ok
        - if VWAP exists: close >= VWAP * 0.99
        - Mean touch:
            * low <= EMA_50 * 1.01
              OR (if VWAP exists) low <= VWAP * 1.01
        - Structure:
            * if Recent_Low exists: low > Recent_Low * 0.985
        - Candle:
            * if Intra_Change exists: Intra_Change >= -0.7
            * if Daily_Change exists: -2.0% <= Daily_Change <= +1.5%
        - Momentum:
            * if RSI exists: 35 <= RSI <= 65
            * if MACD_Hist exists: MACD_Hist >= -0.1
            * if ADX exists: ADX >= 10
        - Volume:
            * if volume + vol_ma20 available: volume >= 0.90 * vol_ma20

    Extra analysis fields are also computed and returned.
    """
    if df_today.empty:
        return None

    d = df_today.copy()

    # --- Trend confirmation (relaxed) ---
    needed = {"EMA_50", "EMA_200", "close"}
    if not needed.issubset(d.columns):
        return None

    ema_valid = d["EMA_50"].notna() & d["EMA_200"].notna()
    trend_base = ema_valid & (d["EMA_50"] > d["EMA_200"]) & (d["close"] >= d["EMA_50"] * 0.995)

    # Optional RSI / ADX constraints
    if "RSI" in d.columns:
        rsi_base = d["RSI"] >= 45
    else:
        rsi_base = pd.Series(True, index=d.index)

    if "ADX" in d.columns:
        adx_base = d["ADX"] >= 12
    else:
        adx_base = pd.Series(True, index=d.index)

    trend_ok = trend_base & rsi_base & adx_base
    if not trend_ok.any():
        return None

    # --- Flags for optional columns ---
    has_vwap = "VWAP" in d.columns and d["VWAP"].notna().any()
    has_rh   = "Recent_High" in d.columns and d["Recent_High"].notna().any()
    has_rl   = "Recent_Low" in d.columns and d["Recent_Low"].notna().any()
    has_ic   = "Intra_Change" in d.columns and d["Intra_Change"].notna().any()
    has_dc   = "Daily_Change" in d.columns and d["Daily_Change"].notna().any()
    has_macd = "MACD_Hist" in d.columns and d["MACD_Hist"].notna().any()
    has_rsi  = "RSI" in d.columns and d["RSI"].notna().any()
    has_adx  = "ADX" in d.columns and d["ADX"].notna().any()

    # --- Volume baseline ---
    if "volume" in d.columns and d["volume"].notna().any():
        d["vol_ma20"] = d["volume"].rolling(20, min_periods=10).mean()
    else:
        d["vol_ma20"] = np.nan

    vol_ma_valid = d["vol_ma20"].notna()

    # ========= AGGRESSIVE BREAKOUT (relaxed, exact) =========
    if has_rh:
        breakout_price = d["close"] >= d["Recent_High"] * 1.0005
    else:
        breakout_price = pd.Series(False, index=d.index)

    if has_vwap:
        breakout_price = breakout_price & (d["close"] >= d["VWAP"] * 1.000)

    if has_dc:
        breakout_price = breakout_price & (d["Daily_Change"] >= 0.7)

    if has_ic:
        breakout_price = breakout_price & (d["Intra_Change"] >= 0.25)

    if has_rsi:
        breakout_price = breakout_price & (d["RSI"] >= 52)

    if has_adx:
        breakout_price = breakout_price & (d["ADX"] >= 18)

    if has_macd:
        breakout_price = breakout_price & (d["MACD_Hist"] >= 0)

    if "volume" in d.columns:
        vol_ok_breakout = vol_ma_valid & (d["volume"] >= d["vol_ma20"] * 1.10)
    else:
        vol_ok_breakout = pd.Series(True, index=d.index)

    aggressive_mask = trend_ok & breakout_price & vol_ok_breakout

    # ========= PULLBACK (more relaxed but bullish) =========
    pb_trend = trend_ok.copy()
    if has_vwap:
        pb_trend = pb_trend & (d["close"] >= d["VWAP"] * 0.99)

    touch_ema = d["low"] <= d["EMA_50"] * 1.01
    if has_vwap:
        touch_vwap = d["low"] <= d["VWAP"] * 1.01
        touch_mean = touch_ema | touch_vwap
    else:
        touch_mean = touch_ema

    if has_rl:
        structure_ok = d["low"] > d["Recent_Low"] * 0.985
    else:
        structure_ok = pd.Series(True, index=d.index)

    if has_ic:
        ic_ok = d["Intra_Change"] >= -0.7
    else:
        ic_ok = pd.Series(True, index=d.index)

    if has_dc:
        dc_ok = (d["Daily_Change"] >= -2.0) & (d["Daily_Change"] <= 1.5)
    else:
        dc_ok = pd.Series(True, index=d.index)

    if has_rsi:
        rsi_mid = (d["RSI"] >= 35) & (d["RSI"] <= 65)
    else:
        rsi_mid = pd.Series(True, index=d.index)

    if has_macd:
        macd_ok_pb = d["MACD_Hist"] >= -0.1
    else:
        macd_ok_pb = pd.Series(True, index=d.index)

    if has_adx:
        adx_ok_pb = d["ADX"] >= 10
    else:
        adx_ok_pb = pd.Series(True, index=d.index)

    if "volume" in d.columns:
        vol_ok_pullback = vol_ma_valid & (d["volume"] >= d["vol_ma20"] * 0.90)
    else:
        vol_ok_pullback = pd.Series(True, index=d.index)

    pullback_mask = (
        pb_trend
        & touch_mean
        & structure_ok
        & ic_ok
        & dc_ok
        & rsi_mid
        & macd_ok_pb
        & adx_ok_pb
        & vol_ok_pullback
    )

    # ========= COMBINE =========
    combined_mask = aggressive_mask | pullback_mask
    hit_idx = d.index[combined_mask.fillna(False)]
    if len(hit_idx) == 0:
        return None

    i = hit_idx[0]
    row = d.loc[i]

    if aggressive_mask.loc[i]:
        reason = "aggressive_breakout"
    elif pullback_mask.loc[i]:
        reason = "pullback"
    else:
        reason = "unknown"

    # ========= EXTRA ANALYSIS FIELDS (same style as previous version) =========
    close = float(row.get("close", np.nan))
    open_ = float(row.get("open", np.nan))
    high  = float(row.get("high", np.nan))
    low   = float(row.get("low", np.nan))

    ema50  = float(row.get("EMA_50", np.nan))
    ema200 = float(row.get("EMA_200", np.nan))
    vwap   = float(row.get("VWAP", np.nan)) if has_vwap else np.nan
    rh     = float(row.get("Recent_High", np.nan)) if has_rh else np.nan
    rl     = float(row.get("Recent_Low", np.nan)) if has_rl else np.nan
    prev_close = float(row.get("Prev_Day_Close", np.nan))

    upper_band = float(row.get("Upper_Band", np.nan))
    lower_band = float(row.get("Lower_Band", np.nan))

    # Trend strength in % (EMA_50 vs EMA_200)
    if np.isfinite(ema50) and np.isfinite(ema200) and ema200 != 0:
        trend_strength_pct = (ema50 / ema200 - 1.0) * 100.0
    else:
        trend_strength_pct = np.nan

    # Distance from recent high (%)
    if np.isfinite(rh) and rh != 0:
        distance_from_recent_high_pct = (rh - close) / rh * 100.0
    else:
        distance_from_recent_high_pct = np.nan

    # Distance from VWAP (%)
    if np.isfinite(vwap) and vwap != 0:
        distance_from_vwap_pct = (close - vwap) / vwap * 100.0
    else:
        distance_from_vwap_pct = np.nan

    # Bollinger band position (0=lower, 100=upper)
    if np.isfinite(upper_band) and np.isfinite(lower_band) and upper_band != lower_band:
        band_pos_pct = (close - lower_band) / (upper_band - lower_band) * 100.0
    else:
        band_pos_pct = np.nan

    # RSI zone label
    rsi_val = float(row.get("RSI", np.nan))
    if not np.isfinite(rsi_val):
        rsi_zone = "unknown"
    elif rsi_val < 30:
        rsi_zone = "oversold"
    elif rsi_val < 45:
        rsi_zone = "weak"
    elif rsi_val < 60:
        rsi_zone = "neutral"
    elif rsi_val < 70:
        rsi_zone = "bullish"
    else:
        rsi_zone = "overbought"

    # Candle shape
    if np.isfinite(high) and np.isfinite(low) and high > low:
        body = abs(close - open_)
        rng = high - low
        candle_body_pct = body / rng * 100.0

        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        upper_wick_pct = upper_wick / rng * 100.0
        lower_wick_pct = lower_wick / rng * 100.0
    else:
        candle_body_pct = np.nan
        upper_wick_pct = np.nan
        lower_wick_pct = np.nan

    # Gap vs previous close
    if np.isfinite(prev_close) and prev_close != 0:
        gap_from_prev_close_pct = (open_ - prev_close) / prev_close * 100.0
    else:
        gap_from_prev_close_pct = np.nan

    return {
        "ticker": ticker,
        "timeframe": tf_label,
        "signal_time_ist": row["date"],
        "entry_price": close if np.isfinite(close) else np.nan,
        "reason": reason,
        "rsi": float(row["RSI"]) if "RSI" in d.columns and pd.notna(row.get("RSI")) else np.nan,
        "adx": float(row["ADX"]) if "ADX" in d.columns and pd.notna(row.get("ADX")) else np.nan,
        "macd_hist": float(row["MACD_Hist"])
        if "MACD_Hist" in d.columns and pd.notna(row.get("MACD_Hist"))
        else np.nan,
        "intra_change": float(row["Intra_Change"])
        if "Intra_Change" in d.columns and pd.notna(row.get("Intra_Change"))
        else np.nan,
        "daily_change": float(row["Daily_Change"])
        if "Daily_Change" in d.columns and pd.notna(row.get("Daily_Change"))
        else np.nan,
        "trend_strength_pct": float(trend_strength_pct),
        "distance_from_recent_high_pct": float(distance_from_recent_high_pct),
        "distance_from_vwap_pct": float(distance_from_vwap_pct),
        "band_pos_pct": float(band_pos_pct),
        "rsi_zone": rsi_zone,
        "candle_body_pct": float(candle_body_pct),
        "upper_wick_pct": float(upper_wick_pct),
        "lower_wick_pct": float(lower_wick_pct),
        "gap_from_prev_close_pct": float(gap_from_prev_close_pct),
    }



def main():
    today_start = _today_start_ist()
    ymd = today_start.strftime("%Y%m%d")

    sig = _read_signals_today(ymd)
    tickers = sorted(sig["ticker"].unique().tolist())
    print(
        f"[INFO] {len(sig)} rows from signals_{ymd}_*.csv; "
        f"unique tickers: {len(tickers)}"
    )

    out_rows = []

    for tkr in tickers:
        # --- 5m timeframe ---
        p5 = _find_ticker_file(DIR_5M, tkr)
        if p5:
            df5 = _safe_read_main_indicators(p5)
            df5_today = _filter_today_session(df5, today_start)
            e5 = _entry_scan_pullback_breakout(df5_today, tkr, "5min")
            if e5:
                out_rows.append(e5)
        else:
            print(f"[WARN] Missing 5min file for {tkr}")

        # --- 15m timeframe ---
        p15 = _find_ticker_file(DIR_15M, tkr)
        if p15:
            df15 = _safe_read_main_indicators(p15)
            df15_today = _filter_today_session(df15, today_start)
            e15 = _entry_scan_pullback_breakout(df15_today, tkr, "15min")
            if e15:
                out_rows.append(e15)
        else:
            print(f"[WARN] Missing 15min file for {tkr}")

    if not out_rows:
        print("[INFO] No intraday entries found for today (pullback / aggressive breakout).")
        return

    out = pd.DataFrame(out_rows)

    # Keep earliest per ticker across both TFs
    out["signal_time_ist"] = pd.to_datetime(
        out["signal_time_ist"], errors="coerce"
    )
    out = (
        out.sort_values(["ticker", "signal_time_ist", "timeframe"])
        .reset_index(drop=True)
    )
    out = out.groupby("ticker", as_index=False).first()

    out["signal_date"] = today_start.date()

    save_path = os.path.join(SIGNALS_DIR, f"intraday_entries_{ymd}.csv")
    out.to_csv(save_path, index=False)

    print(f"[DONE] Saved {len(out)} entries => {save_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
