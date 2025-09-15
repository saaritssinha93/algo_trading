import pandas as pd
from et4_filtered_stocks_MIS import selected_stocks

# Read the CSV file generated earlier
df = pd.read_csv("mis_sheet.csv")

# --- NEW: Remove trailing 'x' or 'X' from the second column ---
# Assume the second column contains the margin allowed values.
df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.replace(r'[xX]$', '', regex=True)

# Determine which column contains the stock names.
# If a column named "Stocks allowed for MIS" exists, use that; otherwise, use the first column.
if "Stocks allowed for MIS" in df.columns:
    stock_column = "Stocks allowed for MIS"
else:
    stock_column = df.columns[0]

# Debug: print unique stock names from the CSV
print("Stocks in mis_sheet.csv:", df[stock_column].dropna().unique())

# Compute the intersection between the stocks from the CSV and the selected_stocks from the module.
common_stocks = set(df[stock_column].dropna().unique()) & selected_stocks
print("Common stocks:", common_stocks)

# Filter the DataFrame to only include rows with common stocks.
df_filtered = df[df[stock_column].isin(common_stocks)]

# Write the filtered data back to mis_sheet.csv (overwriting the existing file).
df_filtered.to_csv("mis_sheet.csv", index=False)
print("mis_sheet.csv has been updated with only the common stocks and updated margin values.")

