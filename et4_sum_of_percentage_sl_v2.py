import time
import pandas as pd
from typing import Optional, List
from datetime import datetime

MARGIN_MULTIPLIER = 5  # change if needed
RUN_EVERY_5_MIN = True  # <- set to False for one-shot run
SLEEP_SECONDS = 5 * 60  # 5 minutes


def _read_result_df(date: str) -> Optional[pd.DataFrame]:
    filename = f"result_finalsl_5min_v2_{date}.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found for date {date}.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        return None

    if 'Ticker' not in df.columns:
        print(f"Error: 'Ticker' column missing in {filename}")
        return None
    if 'Net Percentage' not in df.columns:
        print(f"Error: 'Net Percentage' column missing in {filename}")
        return None

    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    df['Net Percentage'] = pd.to_numeric(df['Net Percentage'], errors='coerce')
    df = df[df['Ticker'].str.lower() != 'overall'].copy()
    df = df.drop_duplicates(subset=['Ticker'], keep='first')
    return df


def _read_papertrade_df(date: str) -> Optional[pd.DataFrame]:
    filename = f"papertrade_5min_v2_{date}.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found for date {date}.")
        return None
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None

    needed = ['Ticker', 'Trend Type', 'Total value']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing} in {filename}")
        return None

    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    df['Trend Type'] = df['Trend Type'].astype(str).str.strip().str.title()
    df['Total value'] = pd.to_numeric(df['Total value'], errors='coerce').fillna(0.0)
    df = df.drop_duplicates(subset=['Ticker'], keep='first')
    return df


def _join_for_date(date: str) -> Optional[pd.DataFrame]:
    pt = _read_papertrade_df(date)
    rs = _read_result_df(date)
    if pt is None or rs is None:
        return None

    merged = pd.merge(
        pt[['Ticker', 'Trend Type', 'Total value']],
        rs[['Ticker', 'Net Percentage']],
        on='Ticker',
        how='left'
    )
    merged['Net Percentage'] = merged['Net Percentage'].fillna(0.0)
    merged['PL'] = (merged['Net Percentage'] / 100.0) * merged['Total value']
    return merged


def sum_net_percentage_for_date(date: str) -> Optional[float]:
    merged = _join_for_date(date)
    if merged is None:
        return None
    return float(merged['Net Percentage'].sum())


def sum_total_value_for_date(date: str) -> float:
    pt = _read_papertrade_df(date)
    if pt is None:
        return 0.0
    return float(pt['Total value'].sum())


def count_tickers_for_date(date: str) -> int:
    pt = _read_papertrade_df(date)
    if pt is None:
        return 0
    return int(pt['Ticker'].nunique())


def _compute_pl_breakout(merged: pd.DataFrame, margin_mult: float):
    pl_total = float(merged['PL'].sum())
    pl_margins_total = pl_total * margin_mult

    bull = merged[merged['Trend Type'].str.lower() == 'bullish']
    bear = merged[merged['Trend Type'].str.lower() == 'bearish']

    pl_bull = float(bull['PL'].sum()) if not bull.empty else 0.0
    pl_bear = float(bear['PL'].sum()) if not bear.empty else 0.0

    return {
        'PL_total': pl_total,
        'PL_margins_total': pl_margins_total,
        'PL_bull': pl_bull,
        'PL_bear': pl_bear,
        'PL_margins_bull': pl_bull * margin_mult,
        'PL_margins_bear': pl_bear * margin_mult,
    }


def calculate_avg_percentage_for_date(date: str) -> float:
    total_net_percentage = sum_net_percentage_for_date(date)
    count = count_tickers_for_date(date)
    if total_net_percentage is None or count == 0:
        return 0.0
    return float(total_net_percentage) / float(count)


def process_multiple_dates(
    dates: List[str],
    margin_mult: float = MARGIN_MULTIPLIER,
    output_csv_path: str = "summary_5min_results.csv",
    pretty_csv_path: Optional[str] = None,
    print_console: bool = True
) -> pd.DataFrame:
    """
    Build a per-date summary and write it to CSV (numeric, unformatted).
    Optionally also write a pretty-formatted CSV and print the same pretty table to console.
    Returns the numeric dataframe.

    Also prints counts of Positive/Negative/Flat days based on 'Profit/Loss (Margins)',
    plus the Positive:Negative ratio and % total_profit/loss.
    """
    rows = []

    for date in dates:
        merged = _join_for_date(date)
        if merged is None or merged.empty:
            rows.append({
                "Date": date,
                "Total Value": 0.0,
                "Total Net %": 0.0,
                "Avg %": 0.0,
                "Ticker Count": 0,
                "Profit/Loss": 0.0,
                "Profit/Loss (Margins)": 0.0,
                "Profit/Loss (Bull)": 0.0,
                "Profit/Loss (Margins) (Bull)": 0.0,
                "Profit/Loss (Bear)": 0.0,
                "Profit/Loss (Margins) (Bear)": 0.0,
            })
            continue

        total_value = float(merged['Total value'].sum())
        total_net_percentage = float(merged['Net Percentage'].sum())
        ticker_count = int(merged['Ticker'].nunique())
        avg_percentage = (total_net_percentage / ticker_count) if ticker_count else 0.0

        pl_parts = _compute_pl_breakout(merged, margin_mult)

        rows.append({
            "Date": date,
            "Total Value": total_value,
            "Total Net %": total_net_percentage,
            "Avg %": avg_percentage,
            "Ticker Count": ticker_count,
            "Profit/Loss": pl_parts['PL_total'],
            "Profit/Loss (Margins)": pl_parts['PL_margins_total'],
            "Profit/Loss (Bull)": pl_parts['PL_bull'],
            "Profit/Loss (Margins) (Bull)": pl_parts['PL_margins_bull'],
            "Profit/Loss (Bear)": pl_parts['PL_bear'],
            "Profit/Loss (Margins) (Bear)": pl_parts['PL_margins_bear'],
        })

    # Numeric (raw) dataframe
    df_results = pd.DataFrame(rows)

    # Totals row (numeric)
    totals_row = {"Date": "TOTAL"}
    sum_cols = [
        "Total Value", "Total Net %", "Ticker Count",
        "Profit/Loss", "Profit/Loss (Margins)",
        "Profit/Loss (Bull)", "Profit/Loss (Margins) (Bull)",
        "Profit/Loss (Bear)", "Profit/Loss (Margins) (Bear)"
    ]
    for col in sum_cols:
        totals_row[col] = df_results[col].sum()
    totals_row["Avg %"] = (
        (totals_row["Total Net %"] / totals_row["Ticker Count"])
        if totals_row["Ticker Count"] else 0.0
    )
    df_results = pd.concat([df_results, pd.DataFrame([totals_row])], ignore_index=True)

    # Write numeric CSV
    df_results.to_csv(output_csv_path, index=False)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved numeric summary to: {output_csv_path}")

    # Build pretty copy (for console + optional pretty CSV)
    df_pretty = df_results.copy()

    def _fmt2(x):
        try: return f"{float(x):,.2f}"
        except: return x
    def _fmt_int(x):
        try: return f"{int(x):,d}"
        except: return x

    df_pretty["Total Value"] = df_pretty["Total Value"].apply(_fmt2)
    df_pretty["Total Net %"] = df_pretty["Total Net %"].apply(_fmt2)
    df_pretty["Avg %"] = df_pretty["Avg %"].apply(_fmt2)
    df_pretty["Ticker Count"] = df_pretty["Ticker Count"].apply(_fmt_int)
    df_pretty["Profit/Loss"] = df_pretty["Profit/Loss"].apply(_fmt2)
    df_pretty["Profit/Loss (Margins)"] = df_pretty["Profit/Loss (Margins)"].apply(_fmt2)
    df_pretty["Profit/Loss (Bull)"] = df_pretty["Profit/Loss (Bull)"].apply(_fmt2)
    df_pretty["Profit/Loss (Margins) (Bull)"] = df_pretty["Profit/Loss (Margins) (Bull)"].apply(_fmt2)
    df_pretty["Profit/Loss (Bear)"] = df_pretty["Profit/Loss (Bear)"].apply(_fmt2)
    df_pretty["Profit/Loss (Margins) (Bear)"] = df_pretty["Profit/Loss (Margins) (Bear)"].apply(_fmt2)

    # Print to console
    print("\n===== Summary (Pretty) =====")
    print(df_pretty.to_string(index=False))

    # ---- Day outcome counts (based on Profit/Loss (Margins)) ----
    df_days = df_results[df_results["Date"] != "TOTAL"].copy()
    pos_days = int((df_days["Profit/Loss (Margins)"] > 0).sum())
    neg_days = int((df_days["Profit/Loss (Margins)"] < 0).sum())
    flat_days = int((df_days["Profit/Loss (Margins)"] == 0).sum())
    total_days = len(df_days)

    print("\n----- Day Outcome Counts (by Profit/Loss (Margins)) -----")
    print(f"Positive days : {pos_days}")
    print(f"Negative days : {neg_days}")
    print(f"Flat days     : {flat_days}")
    if total_days:
        win_rate = (pos_days / total_days) * 100.0
        print(f"Win rate      : {win_rate:.2f}%  ({pos_days}/{total_days})")

    # ---- Positive:Negative ratio ----
    if neg_days > 0:
        ratio_str = f"{(pos_days / neg_days):.2f}"
    else:
        ratio_str = "∞" if pos_days > 0 else "0.00"
    print(f"Positive : Negative (ratio) = {pos_days}:{neg_days} ({ratio_str})")

    # ---- % total_profit/loss over all days ----
    total_pl_margins_all = float(df_days["Profit/Loss (Margins)"].sum())
    total_total_value_all = float(df_days["Total Value"].sum())
    pct_total_pl = (total_pl_margins_all / total_total_value_all * 100.0) if total_total_value_all else 0.0
    print(f"% total_profit/loss : {pct_total_pl:.2f}%  ( {total_pl_margins_all:,.2f} / {total_total_value_all:,.2f} )")

    # Optional pretty CSV
    if pretty_csv_path:
        df_pretty.to_csv(pretty_csv_path, index=False)
        print(f"Saved pretty-formatted summary to: {pretty_csv_path}")

    return df_results


def run_once():
    """Runs a single summary build for the configured date list."""
    dates = [
        # January 2025
        "2025-01-01","2025-01-02","2025-01-03",
        "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
        "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
        "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
        "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

        # February 2025
        "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
        "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
        "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
        "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

        # March 2025
        "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
        "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
        "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
        "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
        "2025-03-31",
        
        # April 2025
        "2025-04-01","2025-04-02","2025-04-03","2025-04-04",
        "2025-04-07","2025-04-08","2025-04-09","2025-04-10",
        "2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18",
        "2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25",
        "2025-04-28","2025-04-29","2025-04-30",

        # May 2025
        "2025-05-02",
        "2025-05-05","2025-05-06","2025-05-07","2025-05-08","2025-05-09",
        "2025-05-12","2025-05-13","2025-05-14","2025-05-15","2025-05-16",
        "2025-05-19","2025-05-20","2025-05-21","2025-05-22","2025-05-23",
        "2025-05-26","2025-05-27","2025-05-28","2025-05-29","2025-05-30",
        
        # June 2025
        "2025-06-02","2025-06-03","2025-06-04","2025-06-05","2025-06-06",
        "2025-06-09","2025-06-10","2025-06-11","2025-06-12","2025-06-13",
        "2025-06-16","2025-06-17","2025-06-18","2025-06-19","2025-06-20",
        "2025-06-23","2025-06-24","2025-06-25","2025-06-26","2025-06-27",
        "2025-06-30",

        # July 2025
        "2025-07-01","2025-07-02","2025-07-03","2025-07-04",
        "2025-07-07","2025-07-08","2025-07-09","2025-07-10"
    ]

    process_multiple_dates(
        dates,
        margin_mult=MARGIN_MULTIPLIER,
        output_csv_path="summary_5min_resultsv2.csv",
        pretty_csv_path="summary_5min_results_prettyv2.csv",
        print_console=True
    )


if __name__ == "__main__":
    try:
        if RUN_EVERY_5_MIN:
            print("Starting 5-minute scheduler. Press Ctrl+C to stop.")
            while True:
                print("\n" + "=" * 80)
                print(f"Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                run_once()
                print("=" * 80)
                print(f"Next run in {SLEEP_SECONDS//60} minutes...")
                time.sleep(SLEEP_SECONDS)
        else:
            run_once()
    except KeyboardInterrupt:
        print("\nStopped by user.")
