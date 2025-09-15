import pandas as pd

def sum_net_percentage_for_date(date):
    """
    Reads the CSV file 'result_final_{date}.csv' and returns the sum of the 'Net Percentage'
    values for all rows where Ticker != 'Overall'.
    """
    filename = f"result_final_{date}.csv"
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found for date {date}.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file '{filename}': {e}")
        return None

    # Ensure the "Net Percentage" column is numeric.
    df['Net Percentage'] = pd.to_numeric(df['Net Percentage'], errors='coerce')

    # Filter out rows where Ticker is "Overall" (case-insensitive).
    df_filtered = df[df['Ticker'].str.strip().str.lower() != 'overall']
    
    total_net_percentage = df_filtered['Net Percentage'].sum()
    return total_net_percentage


def sum_total_value_for_date(date: str) -> float:
    """
    Reads the CSV file 'papertrade_{date}.csv' and returns the sum of the 'Total value'
    column for all rows for that date.
    """
    filename = f"papertrade_{date}.csv"
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found for date {date}.")
        return 0.0
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return 0.0

    df['Total value'] = pd.to_numeric(df['Total value'], errors='coerce')
    total_value = df['Total value'].sum()
    return total_value


def count_tickers_for_date(date: str) -> int:
    """
    Reads the CSV file 'papertrade_{date}.csv' and returns the number of unique tickers for that date.
    """
    filename = f"papertrade_{date}.csv"
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found for date {date}.")
        return 0
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return 0

    ticker_count = df['Ticker'].nunique()
    return ticker_count


def calculate_avg_percentage_for_date(date: str) -> float:
    """
    Calculates the average percentage for the given date using the formula:
    
        Avg Percentage = (Total Net Percentage / Count of tickers)
        
    If the required data cannot be computed, returns 0.0.
    """
    total_net_percentage = sum_net_percentage_for_date(date)
    total_value = sum_total_value_for_date(date)
    count = count_tickers_for_date(date)
    
    if total_net_percentage is None:
        print(f"Net Percentage could not be computed for {date}.")
        return 0.0

    if count == 0:
        print(f"No tickers found for {date}. Cannot compute average percentage.")
        return 0.0

    avg_percentage = total_net_percentage / count
    return avg_percentage


def process_multiple_dates(dates: list) -> None:
    """
    Processes multiple dates by gathering the Total value, Total Net Percentage,
    Average Percentage, and profit/loss figures for each date, then prints the results as a table.
    Adds a TOTAL row including navg = (TOTAL of Total Net %) / (TOTAL of Ticker Count).
    """
    results = []

    for date in dates:
        total_value = sum_total_value_for_date(date)
        total_net_percentage = sum_net_percentage_for_date(date)
        avg_percentage = calculate_avg_percentage_for_date(date)
        count = count_tickers_for_date(date)

        # Calculate profit/loss and profit/loss with margins.
        p_l = (avg_percentage / 100.0) * total_value if avg_percentage is not None else 0.0
        p_l_m = p_l * 5

        results.append({
            "Date": date,
            "Total Value": total_value,
            "Total Net %": total_net_percentage,
            "Avg %": avg_percentage,
            "Ticker Count": count,
            "Profit/Loss": p_l,
            "Profit/Loss (Margins)": p_l_m,
        })

    df_results = pd.DataFrame(results)

    # Compute totals (NaNs are ignored by sum).
    totals_row = {"Date": "TOTAL"}
    numeric_columns = ["Total Value", "Total Net %", "Avg %", "Ticker Count", "Profit/Loss", "Profit/Loss (Margins)"]
    for col in numeric_columns:
        totals_row[col] = pd.to_numeric(df_results[col], errors="coerce").sum()

    # Compute navg = TOTAL Total Net % / TOTAL Ticker Count
    total_net_sum = totals_row["Total Net %"]
    total_ticker_sum = totals_row["Ticker Count"]
    navg_value = (total_net_sum / total_ticker_sum) if total_ticker_sum else 0.0

    # Add a 'navg' column; blank for per-date rows, value only on TOTAL row.
    df_results["navg"] = ""
    totals_row["navg"] = navg_value

    # Append TOTAL row
    totals_df = pd.DataFrame([totals_row])
    df_results = pd.concat([df_results, totals_df], ignore_index=True)

    # Format numbers for display.
    def fmt2(x):
        return f"{x:,.2f}" if isinstance(x, (int, float)) and pd.notna(x) else x

    def fmt0(x):
        return f"{x:,.0f}" if isinstance(x, (int, float)) and pd.notna(x) else x

    df_results["Total Value"] = df_results["Total Value"].apply(fmt2)
    df_results["Total Net %"] = df_results["Total Net %"].apply(fmt2)
    df_results["Avg %"] = df_results["Avg %"].apply(fmt2)
    df_results["Ticker Count"] = df_results["Ticker Count"].apply(fmt0)
    df_results["Profit/Loss"] = df_results["Profit/Loss"].apply(fmt2)
    df_results["Profit/Loss (Margins)"] = df_results["Profit/Loss (Margins)"].apply(fmt2)
    df_results["navg"] = df_results["navg"].apply(fmt2)

    print(df_results.to_string(index=False))



if __name__ == "__main__":
    # Define the list of dates to process.
    dates = [
        # January 2025
        "2025-01-01","2025-01-02","2025-01-03",
        "2025-01-06","2025-01-07","2025-01-08","2025-01-09","2025-01-10",
        "2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17",
        "2025-01-20","2025-01-21","2025-01-22","2025-01-23","2025-01-24",
        # 2025-01-26 is Republic Day (Sunday+holiday)
        "2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31",

        # February 2025
        "2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07",
        "2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14",
        "2025-02-17","2025-02-18","2025-02-19","2025-02-20","2025-02-21",
        "2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28",

        # March 2025
        "2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07",
        "2025-03-10","2025-03-11","2025-03-12","2025-03-13",
        # 2025-03-14 is Holi (holiday)
        "2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21",
        "2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28",
        "2025-03-31"

      
    ]
    process_multiple_dates(dates)
