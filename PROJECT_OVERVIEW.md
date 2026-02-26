# algo_trading — Project Overview
> Automated trading system using Zerodha KiteConnect API for NSE/BSE equities and ETFs.
> Last catalogued: 2026-02-26

---

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Infrastructure & Configuration](#2-infrastructure--configuration)
3. [Strategy A — AVWAP v11 (Stocks, Intraday 15-min)](#3-strategy-a--avwap-v11-stocks-intraday-15-min)
4. [Strategy B — AVWAP v7 (ETFs, Intraday 15-min)](#4-strategy-b--avwap-v7-etfs-intraday-15-min)
5. [Strategy C — Multi-TF Positional (Stocks & ETFs, Weekly/Daily)](#5-strategy-c--multi-tf-positional-stocks--etfs-weekdaily)
6. [Strategy D — Single-Ticker (SILVERBEES / GOLDBEES)](#6-strategy-d--single-ticker-silverbees--goldbees)
7. [Macro Overlay](#7-macro-overlay)
8. [Data Pipeline](#8-data-pipeline)
9. [Backtesting Results & Outputs](#9-backtesting-results--outputs)
10. [Data Directories](#10-data-directories)
11. [Logs, Reports & Account Exports](#11-logs-reports--account-exports)
12. [Windows Launch Scripts (bat/)](#12-windows-launch-scripts-bat)
13. [Archived & Stub Code](#13-archived--stub-code)

---

## 1. Project Summary

This is an **algorithmic trading system** built on top of the [Zerodha KiteConnect](https://kite.trade/) Python SDK. It covers the full lifecycle:

- **Authentication** — automated Zerodha login via Selenium + TOTP
- **Data fetching** — continuous OHLCV data across 6 timeframes (5min, 15min, 1h, 3h, daily, weekly) stored as Parquet
- **Indicator computation** — ATR, RSI, Stochastic, ADX, EMA, AVWAP and more, one file per ticker per timeframe
- **Signal generation** — multi-timeframe scans producing long/short entry signals
- **Backtesting** — P&L analysis with Sharpe, Sortino, Calmar, drawdown, and profit-factor metrics
- **Live trading** — scheduled runners that fire every 15 minutes during NSE market hours (09:15–15:30 IST)
- **Macro overlay** — global regime filters (VIX, DXY, USDINR, Gold, Silver, US Yields)

Two instrument universes are supported: **ETFs** and **Equities (stocks)**, across both **positional** (daily/weekly) and **intraday** (15-min) strategies.

Total tracked files (excluding `.git` / `__pycache__`): **~33,000+** — the majority are per-ticker Parquet data files.

---

## 2. Infrastructure & Configuration

### Credentials & Token Files
| File | Purpose |
|------|---------|
| `api_key.txt` | Zerodha API key + secret (one per line) |
| `access_token.txt` | Session access token written after login |
| `request_token.txt` | OAuth request token from Zerodha redirect |
| `tokens_cache.json` | Mixed instrument token cache (symbol → token) |
| `etf_tokens_cache.json` | ETF-specific token cache |
| `stocks_tokens_cache.json` | Equity-specific token cache |
| `Merged_NSE_BSE_Instruments.csv` | Master instrument list (NSE + BSE merged) |
| `ticker.txt` | Ad-hoc ticker list for manual runs |

### Authentication Script
| Script | Purpose |
|--------|---------|
| `algosm1_authentication.py` | Automated Zerodha login via Selenium + TOTP (Chrome WebDriver) |
| `backtesting/eqidv1/algosm1_authentication.py` | Same, wired to eqidv1 paths |

### Zerodha Account Export Scripts
| Script | Purpose |
|--------|---------|
| `zerodha_kite_export.py` | Exports holdings, margins, ETF master list to `kite_exports/` |
| `algosm1_kit_export.py` | Similar KiteConnect export utility |

### Instrument Universe Filters
| Script | Universe |
|--------|---------|
| `algosm1_filtered_etfs.py` | ETF universe for strategy runs |
| `etf_filtered_etfs.py` | ETF universe for live scanner |
| `etf_filtered_etfs_all.py` | Full ETF universe (all instruments) |
| `et4_filtered_stocks.py` | Filtered equity universe (CNC/delivery) |
| `et4_filtered_stocks_MIS.py` | Filtered equity universe (MIS/intraday) |
| `algosm1_gettickerlist_today.py` | Fetches today's active ticker list from Zerodha |

---

## 3. Strategy A — AVWAP v11 (Stocks, Intraday 15-min)

The most mature strategy in the codebase. Uses Anchored VWAP (AVWAP) with ATR, RSI, Stochastic, ADX, and volume/volatility filters to generate long and short entries on 15-min candles, with 5-min data for exit resolution.

### Core Module: `avwap_v11_refactored/`
| File | Role |
|------|------|
| `avwap_common.py` | **Central shared library.** `Trade` dataclass, `StrategyConfig` (all tuneable params), all indicator computations, IO helpers, session/time-window helpers, quality-score, full backtest metrics, slippage + commission model |
| `avwap_long_strategy.py` | Long-only strategy logic |
| `avwap_short_strategy.py` | Short-only strategy logic |
| `avwap_combined_runner.py` | **Main backtest entry point.** Runs long + short together, parallel scanning via `ProcessPoolExecutor`, outputs to `outputs/` |
| `__init__.py` | Package marker |
| `avwap_combined_runner.py` (root) | Thin wrapper delegating to the package above |

> A **refined copy** of this module lives in `backtesting/eqidv1/avwap_v11_refactored/` with tighter parameters: SL 0.75% (vs 1.0%), better R:R (SHORT TGT 1.2%, LONG TGT 1.5%), stricter ADX/RSI/Stoch thresholds, volume filter (impulse bar ≥ 1.2× SMA-20), and ATR% volatility filter (ATR/close ≥ 0.20%).

### Live Signal Scanners (Strategy A)
| Script | Scope | Notes |
|--------|-------|-------|
| `stocks_live_trading_signal_15m_v11_combined_parquet.py` | All NSE stocks | Fires every 15 min in market hours, writes to `out_live_checks/` + `out_live_signals/` |
| `eqidv1_live_trading_signal_15m_v11_combined_parquet.py` | eqidv1 equity universe | Uses refined backtesting/eqidv1 params, runs via `run_eqidv1_live_signals.bat` |

### Profit Analysis / Backtest Scripts (Equities, iterative series)
| Script | Notes |
|--------|-------|
| `algosm1_trading_signal_profit_analysis.py` | v1 baseline |
| `algosm1_trading_signal_profit_analysisv2.py` | v2 |
| `algosm1_trading_signal_profit_analysisv3.py` | v3 |
| `algosm1_trading_signal_profit_analysisv4.py` | v4 |
| `algosm1_trading_signal_profit_analysis_v5_strategyselectv1.py` | v5 with strategy selection |
| `algosm1_trading_signal_profit_analysis_v5_intraday_strategyv2.py` | Intraday variant |
| `algosm1_trading_signal_profit_analysis_v5_intraday_strategyselectv1.py` | Intraday + strategy select |
| `algosm1_trading_signal_profit_analysis_v5_intraday_strategyselectv2.py` | Intraday + strategy select v2 |
| `algosm1_trading_signal_profit_analysis_v5_positional_strategyv2.py` | Positional variant |
| `algosm1_trading_signal_profit_analysis_id_v4.py` | Intraday v4 |
| `algosm1_trading_signal_profit_analysis_eq_15m_v8_intraday_parquet.py` | EQ 15-min intraday v8 (parquet) |
| `algosm1_trading_signal_profit_analysis_eq_15m_v9_intraday_parquet.py` | v9 |
| `algosm1_trading_signal_profit_analysis_eq_15m_v10_intraday_parquet.py` | v10 |
| `algosm1_trading_signal_profit_analysis_eq_15m_v11_intraday_parquet.py` | v11 intraday |
| `algosm1_trading_signal_profit_analysis_eq_15m_v11_long_intraday_parquet.py` | v11 long-only |
| `algosm1_trading_signal_profit_analysis_eq_15m_v11_combined_parquet.py` | **Latest:** combined long+short |

### Signal Generation Helpers (Strategy A)
| Script | Purpose |
|--------|---------|
| `algosm1_trading_signal.py` | Base multi-TF signal generator |
| `algosm1_trading_signal_long.py` | Long-only signals |
| `algosm1_trading_signal_agentic_ai.py` | AI-assisted signal layer (v1) |
| `algosm1_trading_signal_agentic_ai_2.py` | AI-assisted signal layer (v2) |

### Scheduled Data Updaters (Strategy A — Stocks)
| Script | Trigger | Universe |
|--------|---------|---------|
| `stocks_eod_daily_weekly_scheduler_for_15mins_data.py` | Every 15 min during market hours | All stocks |
| `stocks_eod_daily_weekly_scheduler_for_daily_1540_update.py` | 15:40 EOD | All stocks |
| `eqidv1_eod_scheduler_for_15mins_data.py` | Every 15 min | eqidv1 stocks |
| `eqidv1_eod_scheduler_for_1540_update.py` | 15:40 EOD | eqidv1 stocks |

---

## 4. Strategy B — AVWAP v7 (ETFs, Intraday 15-min)

An earlier version of the AVWAP strategy applied to the ETF universe. Preceded v11 and operates on ETF indicator parquets.

### Live Signal Scanner (Strategy B)
| Script | Notes |
|--------|-------|
| `etf_live_trading_signal_15m_v7_parquet.py` | **Production scanner.** Runs two checks per 15-min slot (A + B passes), writes to `out_live_checks/` + `out_live_signals/`, maintains booking ledger in `etf_signals/` |
| `etf_live_trading_signal_15m_v7_parquet_fulltest.py` | Full historical test mode |
| `etf_live_trading_signal_15m_v7_parquet_fulltest_today.py` | Test against today's data only |

### Profit Analysis / Backtest Scripts (ETFs, iterative series)
| Script | Notes |
|--------|-------|
| `algosm1_trading_signal_profit_analysis_etf_v2.py` | ETF v2 baseline |
| `algosm1_trading_signal_profit_analysis_etf_v3.py` | v3 |
| `algosm1_trading_signal_profit_analysis_etf_v4.py` | v4 |
| `algosm1_trading_signal_profit_analysis_etf_v5.py` | v5 |
| `algosm1_trading_signal_profit_analysis_etf_v5_strategyselect.py` | v5 with strategy selection |
| `algosm1_trading_signal_profit_analysis_etf_v5_strategyselectv1.py` | v5 strategy select v1 |
| `algosm1_trading_signal_profit_analysis_etf_v6.py` | v6 |
| `algosm1_trading_signal_profit_analysis_etf_v7.py` | **Latest ETF hourly version** |
| `algosm1_trading_signal_profit_analysis_etf_1h_v6.py` | 1-hour timeframe v6 |
| `algosm1_trading_signal_profit_analysis_etf_1h_v6_parquet.py` | 1-hour v6 (parquet) |
| `algosm1_trading_signal_profit_analysis_etf_1h_1tickeratatime_v6.py` | 1-hour v6, serial mode |
| `algosm1_trading_signal_profit_analysis_etf_1h_1tickeratatime_v6_parquet.py` | 1-hour v6 serial (parquet) |
| `algosm1_trading_signal_profit_analysis_etf_15m_v7.py` | 15-min v7 |
| `algosm1_trading_signal_profit_analysis_etf_15m_v7_parquet.py` | 15-min v7 (parquet) |
| `algosm1_trading_signal_profit_analysis_etf_15m_v7_parquet_old.py` | Old parquet variant (kept for reference) |
| `algosm1_trading_signal_profit_analysis_etf_15m_1tickeratatime_v7.py` | 15-min v7, serial mode |
| `algosm1_trading_signal_profit_analysis_etf_15m_1tickeratatime_v7_parquet.py` | 15-min v7 serial (parquet) |
| `algosm1_trading_signal_profit_analysis_etf_15m_v8_intraday_parquet.py` | 15-min intraday v8 (parquet) |

### Signal & Trend Scan Helpers (ETFs)
| Script | Purpose |
|--------|---------|
| `algosm1_trading_signal_etf_trendscan.py` | Multi-TF ETF trend scanner |
| `etf_get_signals_to_csv.py` | Exports current ETF signals to CSV |

### Scheduled Data Updaters (Strategy B — ETFs)
| Script | Trigger |
|--------|---------|
| `etf_eod_daily_weekly_scheduler_for_15mins_data.py` | Every 15 min during market hours |
| `etf_eod_daily_weekly_scheduler_for_daily_1540_update.py` | 15:40 EOD |

---

## 5. Strategy C — Multi-TF Positional (Stocks & ETFs, Weekly/Daily)

Positional strategy using weekly and daily timeframe alignment to identify swing trades. Entries are typically taken the following market day based on EOD signals.

### Core Scripts
| Script | Notes |
|--------|-------|
| `algosm1_positional_weekly_daily.py` | Base positional: combines weekly + daily signals |
| `algosm1_positional_weekly_daily_select_v1.py` | With ticker pre-selection filter |
| `algosm1_positional_weekly_daily_select_v1_livetoday.py` | Runs live signal check for today |
| `algosm1_positional_weekly_daily_select_v1_livetoday_entrypoint.py` | **Main entrypoint for live positional** |
| `algosm1_positional_weekly_daily_select_v1_livetoday_entrypoint_evaluation.py` | Evaluation / paper-trade mode |
| `algosm1_positional_weekly_daily_select_v1_livetoday_v2.py` | v2 live variant |
| `algosm1_positional_weekly_daily_select_v1_closedtoday_fornextmarketday.py` | Generates next-day signals from today's close |
| `algosm1_positional_weekly_daily_select_v1_liveonedateinput.py` | Accepts a single date as input |
| `algosm1_positional_weekly_daily_select_v1_liveonedateinput_v2.py` | v2 |
| `algosm1_positional_weekly_daily_select_v1_liveonedateinput_v3.py` | v3 |
| `algosm1_positional_weekly_daily_select_v1_liveonedateinput_v4.py` | v4 (latest date-input version) |

### Hourly Variants
| Script | Notes |
|--------|-------|
| `algosm1_positional_weekly_daily_select_hourly_v1.py` | Positional with hourly entry timing v1 |
| `algosm1_positional_weekly_daily_select_hourly_v2.py` | v2 |

### All-ETF One-Per-Day Positional
| Script | Notes |
|--------|-------|
| `algosm1_all_4_oneperday_onlypositional.py` | All instruments, one trade/day, positional |
| `algosm1_alletfs_4_oneperday_onlypositional.py` | ETF-only version |

---

## 6. Strategy D — Single-Ticker (SILVERBEES / GOLDBEES)

Dedicated strategies for two high-liquidity commodity ETFs: SILVERBEES (silver) and GOLDBEES (gold).

### SILVERBEES
| Script | Notes |
|--------|-------|
| `algosm1_silverbees.py` | v1 baseline |
| `algosm1_silverbees_2.py` | v2 |
| `algosm1_silverbees_3.py` | v3 |
| `algosm1_silverbees_4_oneperday.py` | One trade per day |
| `algosm1_silverbees_4_oneperday_onlypositional.py` | Positional only, one trade/day |
| `algosm1_silverbees_macro.py` | Silverbees + macro regime filter |

### GOLDBEES
| Script | Notes |
|--------|-------|
| `algosm1_goldbees_4_oneperday_onlypositional.py` | One trade/day, positional only |

### Combined
| Script | Notes |
|--------|-------|
| `algosm1_silverbees_goldbees_macro.py` | Runs Silver + Gold strategy with unified macro filter |

---

## 7. Macro Overlay

Global regime filter applied on top of all strategies. Data is built and stored as daily CSV files in `macro_inputs/`.

### Builder Script
| Script | Purpose |
|--------|---------|
| `algosm1_all_macro.py` | Pulls all macro data (USDINR, DXY via FRED, US yields, VIX, WTI, Gold/Silver spot and MCX prices) and writes to `macro_inputs/` |

### Macro Input Files (`macro_inputs/`)
| File | Indicator |
|------|-----------|
| `USDINR.csv` | USD/INR exchange rate |
| `DXY.csv` | US Dollar Index (FRED DTWEXBGS proxy) |
| `US10Y.csv` | US 10-Year Treasury yield |
| `US_REAL_YIELD.csv` | US 10-Year real yield (FRED DFII10) |
| `CURVE_T10Y3M.csv` | Yield curve spread (10Y minus 3M) |
| `VIX.csv` | CBOE Volatility Index |
| `WTI.csv` | WTI Crude Oil price |
| `XAUUSD.csv` | Gold spot (USD) |
| `XAGUSD.csv` | Silver spot (USD) |
| `GOLD_SPOT_INR_10G.csv` | Gold spot (₹ per 10g) |
| `SILVER_SPOT_INR_KG.csv` | Silver spot (₹ per kg) |
| `MCX_GOLD_RS_10G.csv` | MCX Gold futures (₹ per 10g) |
| `MCX_SILVER_RS_KG.csv` | MCX Silver futures (₹ per kg) |
| `EVENT_RISK.csv` | Binary event-risk flag (RBI/Fed meeting days, etc.) |

---

## 8. Data Pipeline

### Historical Data Fetchers (all timeframes)
These scripts download OHLCV data from KiteConnect and build the local data store. They evolved from CSV to Parquet storage.

| Script | Universe | Format | Notes |
|--------|---------|--------|-------|
| `algosm1_trading_data_continous_run_historical_1d.py` | Mixed | CSV | Single TF: daily |
| `algosm1_trading_data_continous_run_historical_1hr.py` | Mixed | CSV | Single TF: 1h |
| `algosm1_trading_data_continous_run_historical_1w.py` | Mixed | CSV | Single TF: weekly |
| `algosm1_trading_data_continous_run_historical_3hr.py` | Mixed | CSV | Single TF: 3h |
| `algosm1_trading_data_continous_run_historical_5min.py` | Mixed | CSV | Single TF: 5min |
| `algosm1_trading_data_continous_run_historical_alltf.py` | Mixed | CSV | All TFs, v1 |
| `algosm1_trading_data_continous_run_historical_alltf_v2.py` | Mixed | CSV | All TFs, v2 |
| `algosm1_trading_data_continous_run_historical_alltf_v3.py` | Mixed | CSV | All TFs, v3 |
| `algosm1_trading_data_continous_run_historical_alltf_v3_etfsonly.py` | ETFs | CSV | v3, ETFs |
| `algosm1_trading_data_continous_run_historical_etfs_alltf.py` | ETFs | CSV | ETF-specific |
| `algosm1_trading_data_continous_run_historical_alltf_v3_stocksonly.py` | Stocks | CSV | v3, stocks |
| `algosm1_trading_data_continous_run_historical_alltf_v3_parquet_etfsonly.py` | ETFs | **Parquet** | v3, ETFs, parquet |
| `algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py` | Stocks | **Parquet** | v3, stocks, parquet (**current standard**) |
| `backtesting/eqidv1/algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py` | eqidv1 stocks | Parquet | eqidv1 copy |

### Backtesting Utilities
| Script | Purpose |
|--------|---------|
| `backtesting.py` | General backtesting utility functions |

---

## 9. Backtesting Results & Outputs

### Root-level Result Files (spec0 backtest suite)
| File | Contents |
|------|---------|
| `best_strategies.json` | Best parameter sets from strategy optimisation |
| `results_strategies.csv` | Strategy variant comparison table |
| `trades_15min_spec0_test.csv` | Trade log — base spec |
| `trades_15min_spec0_test_0915_1130.csv` | Trade log — morning session only |
| `trades_15min_spec0_test_tp4_positional.csv` | Trade log — TP4 positional |
| `trades_5min_spec0_full.csv` | 5-min full trade log |
| `daily_pnl_15min_spec0_test.csv` | Daily P&L |
| `equity_curve_15min_spec0_test.csv` | Equity curve |
| `ticker_pnl_15min_spec0_test.csv` | Per-ticker P&L |
| `metrics_15min_spec0_test.json` | Summary metrics (Sharpe, drawdown, etc.) |
| *(…`_0915_1130` and `_tp4_positional` variants for daily P&L, equity curve, ticker P&L, metrics)* | |
| `apply_best.log` | Log from applying best-strategy params |
| `apply_best_tp4_positional.log` | TP4 positional variant log |

### AVWAP Combined Runner Outputs (`outputs/`)
| Sub-path | Contents |
|---------|---------|
| `avwap_longshort_trades_ALL_DAYS_*.csv` | Full long+short trade logs, timestamped |
| `avwap_combined_runner_*.txt` | Console output / metrics summary per run |
| `charts/daily_pnl_*.png` | Daily P&L bar charts |
| `charts/drawdown_*.png` | Drawdown charts |
| `charts/enhanced/` | Enhanced charting suite outputs |
| `charts/legacy/` | Older chart formats |
| `metrics/` | JSON metrics files |

---

## 10. Data Directories

### ETF Data (root level)
Parallel CSV and Parquet stores for each timeframe (~19 ETFs each):

| Directory | Format | Timeframes |
|-----------|--------|-----------|
| `etf_cache_{tf}/` | CSV | 5min, 15min, 1h, 3h, daily, weekly |
| `etf_cache_{tf}_pq/` | Parquet | Same |
| `etf_cache_15min_pq1/` | Parquet | Alternate 15-min store |
| `etf_indicators_{tf}/` | CSV | Same timeframes |
| `etf_indicators_{tf}_pq/` | Parquet | Same |
| `etf_indicators_15min_pq1/` | Parquet | Alternate 15-min indicators |

### Equity/Stocks Data (root level)
| Directory | Status | Contents |
|-----------|--------|---------|
| `stocks_cache_{tf}_eq/` (6 dirs) | Empty | Intended raw OHLCV store — data lives in `backtesting/eqidv1/` |
| `stocks_indicators_{tf}_eq/` (6 dirs) | **Active** | ~1,029 parquet files per TF (one per NSE ticker) |

### eqidv1 Backtesting Data (`backtesting/eqidv1/`)
| Directory | Contents |
|-----------|---------|
| `stocks_cache_{tf}_eq/` | Raw OHLCV parquets per ticker, per TF |
| `stocks_indicators_{tf}_eq/` | Indicator parquets per ticker, per TF |

### Signal Outputs
| Directory | Contents |
|-----------|---------|
| `signals/` | Point-in-time multi-TF signal snapshots (parquet, timestamped) |
| `etf_signals/` | ETF booking ledger + signal snapshots from live scanner runs (Jan–Feb 2026) |
| `etf_signals_live/` | Lightweight live signal files, date-subdir organised |
| `out_live_checks/{date}/` | Full-universe ETF scan results per 15-min slot (two passes A/B) |
| `out_live_checks_15m/{date}/` | ETF + stocks 15-min scan results |
| `out_live_signals/{date}/` | Filtered: only tickers where a signal fired (ETF, hourly) |
| `out_live_signals_15m/{date}/` | Filtered: signal-only tickers (15-min) |

> Live output dates covered: 2026-01-29 through 2026-02-20.

---

## 11. Logs, Reports & Account Exports

### Runtime Logs (`logs/`)
| File | Contents |
|------|---------|
| `auth.log` | Authentication events |
| `algosm1_history.log` | Historical data fetch history |
| `live_15m_updater.log` | ETF 15-min updater |
| `live_15m_updater_stocks.log` | Stocks 15-min updater |
| `live_signals_15m.log` | ETF live signal scanner |
| `live_signals_15m_stocks.log` | Stocks live signal scanner |
| `eod_daily_weekly_1540.log` | ETF EOD (15:40) scheduler |
| `eod_daily_weekly_1540_stocks.log` | Stocks EOD scheduler |
| `eqidv3_eod_1540.log` | eqidv3 EOD scheduler |
| `eqidv3_live_15m_updater.log` | eqidv3 15-min updater |
| `eqidv3_live_signals_15m.log` | eqidv3 live signals |
| `kite_20260102.log` | KiteConnect session (2026-01-02) |
| `watchlist_20260102.log` | Watchlist creation |
| `avwap_live_state_v11.json` | Persisted live state (open positions, last candle) |
| `create_watchlist_fail.png` | Screenshot of failed watchlist creation |
| `etf_fetcher_run.log` (root) | ETF fetch run log |
| `fetcher_run.log` (root) | General fetch log |
| `stocks_fetcher_run.log` (root) | Stocks fetch log |

### Missing-Data Reports
Three parallel trees tracking data quality, each with identical structure:

| Location | Universe |
|----------|---------|
| `etf_missing_reports/` | ETFs |
| `stocks_missing_reports/` | Equities |
| `reports/stocks_missing_reports/` | Equities (reports copy) |
| `backtesting/eqidv1/reports/stocks_missing_reports/` | eqidv1 equities |

Each tree contains:
- `missing_files/` — text files listing tickers with **no data at all** per timeframe (5min, 15min, 1h, 3h, daily, weekly)
- `missing_rows/{timeframe}/` — per-ticker parquets listing **specific missing candle timestamps**

### Zerodha Account Exports (`kite_exports/`)
Point-in-time account snapshots dated 2026-01-28:

| File | Contents |
|------|---------|
| `holdings_20260128.csv` | Full holdings dump |
| `holdings_equity_stocks_20260128.csv` | Equity stock holdings only |
| `holdings_etfs_20260128.csv` | ETF holdings only |
| `holdings_unknown_20260128.csv` | Unclassified holdings |
| `margins_20260128.csv` | Margin availability |
| `nse_etf_master_20260128.csv` | NSE ETF master list |

---

## 12. Windows Launch Scripts (`bat/`)

One-click batch files for key processes on Windows:

| Script | Starts |
|--------|--------|
| `run_auth.bat` | Authentication (Selenium auto-login) |
| `run_15m_updater.bat` | ETF 15-min data updater |
| `run_15m_updater_stocks.bat` | Stocks 15-min data updater |
| `run_eod_1540.bat` | ETF EOD (15:40) updater |
| `run_eod_1540_stocks.bat` | Stocks EOD updater |
| `run_live_signals.bat` | ETF live signal scanner |
| `run_live_signals_stocks.bat` | Stocks live signal scanner |
| `run_eqidv1_15m_updater.bat` | eqidv1 equity 15-min updater |
| `run_eqidv1_eod_1540.bat` | eqidv1 equity EOD updater |
| `run_eqidv1_live_signals.bat` | eqidv1 equity live signal scanner |

---

## 13. Archived & Stub Code

### `archived/`
Scripts retired when superseded by newer versions:

| File | Replaced by |
|------|------------|
| `algosm1_trading_data_continous_run_historical_alltf_v3_parquet_stocksonly.py` | `backtesting/eqidv1/` version |
| `avwap_combined_runner.py` | `avwap_v11_refactored/avwap_combined_runner.py` |

### `scripts/` (empty stub)
A planned modular reorganisation with subdirectories (`auth/`, `avwap/`, `data_fetchers/`, `live_trading/`, `schedulers/`, `signal_analysis/`, `strategies/`, `utils/`) — all currently empty. All active code remains at the root level.

---

*Document auto-generated by Claude — 2026-02-26*
