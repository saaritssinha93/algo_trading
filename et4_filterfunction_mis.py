import pandas as pd

def extract_symbols_from_gsheet(
    sheet_id,
    gid,
    output_file="MIS_stocks.py"
):
    """
    1) Fetch the Google Sheet as CSV.
    2) Skip the first 2 rows (based on your snippet).
    3) Assign column names manually.
    4) Extract the 'Symbol' column.
    5) Write symbols to MIS_stocks.py as a Python list.
    6) Also return the symbols (as a Python list) to use in memory.
    """
    # Build export URL:
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # Read the CSV while skipping the top 2 rows (they are not purely headers)
    # and set our own column headers.
    df = pd.read_csv(
        csv_url,
        skiprows=2,
        names=[
            "ISIN",
            "Symbol",
            "BSE Symbol",
            "Var+ ELM+Adhoc margin",
            "MIS Margin(%)",
            "MIS Multiplier",
            "CO Margin(%)",
            "CO Upper Trigger",
            "CO Multiplier",
        ],
    )

    # Extract the 'Symbol' column, remove NaN, get unique, make a list
    symbols_list = (
        df["Symbol"]
        .dropna()
        .unique()
        .tolist()
    )

    # Write to MIS_stocks.py, so you can import or read it later
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"shares = {symbols_list}\n")

    return symbols_list


def compare_mis_with_et4(mis_shares, filtered_stocks_module="et4_filtered_stocks_market_cap"):
    """
    Import the 'filtered_stocks' list from et4_filtered_stocks_market_cap.py
    and show which stocks appear in both lists. Prints and returns the result.
    """
    # Dynamically import the other Python file
    # (make sure et4_filtered_stocks_market_cap.py is in the same folder and has 'filtered_stocks' defined)
    module = __import__(filtered_stocks_module)
    selected_stocks = module.selected_stocks

    common_stocks = set(mis_shares).intersection(set(selected_stocks))
    print("Stocks present in BOTH files:", sorted(common_stocks))
    print("Total common stocks:", len(common_stocks))
    return common_stocks


def write_common_stocks(common_stocks, output_file="et4_filtered_stocks_MIS.py"):
    """
    Write the common stocks to a file as a Python list named 'filtered_stocks_MIS'.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"selected_stocks = {common_stocks}\n")


if __name__ == "__main__":
    # Replace with your actual Sheet ID and GID
    SHEET_ID = "1fLTsNpFJPK349RTjs0GRSXJZD-5soCUkZt9eSMTJ2m4"
    GID = "288818195"

    # Step 1: Extract symbols from Google Sheet -> MIS_stocks.py
    mis_shares = extract_symbols_from_gsheet(SHEET_ID, GID)

    # Step 2: Compare with stocks in et4_filtered_stocks_market_cap.py
    common = compare_mis_with_et4(mis_shares)

    # Step 3: Write the common stocks to et4_filtered_stocks_MIS.py
    write_common_stocks(common)
