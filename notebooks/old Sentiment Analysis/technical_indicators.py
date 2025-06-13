import pandas as pd
import numpy as np

from pathlib import Path

current_dir = Path(__file__).parent

# Load your dataset
df = pd.read_csv(current_dir / "df_with_yahoo_data.csv", parse_dates=["date"])

# Drop specified columns
# df = df.drop(columns=["revty", "saley", "capxy", "Adj Close", "open", "high", "low"], errors='ignore')

# df.dropna(inplace=True)

# Rename month columns
month_map = {
    f"month_{i}": month for i, month in enumerate([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ], start=1)
}
df.rename(columns=month_map, inplace=True)

# Replace True/False with 1/0 in month columns
month_cols = list(month_map.values())
df[month_cols] = df[month_cols].astype(int)

# Sort values for group-wise operations
df = df.sort_values(["PERMNO", "date"])

# # Technical indicator functions
# def calculate_rsi(series, window=14):
#     delta = series.diff()
#     up = delta.clip(lower=0)
#     down = -delta.clip(upper=0)
#     ma_up = up.rolling(window).mean()
#     ma_down = down.rolling(window).mean()
#     rs = ma_up / ma_down
#     return 100 - (100 / (1 + rs))

# # Group-wise calculations
# def add_technical_indicators(group):
#     group = group.copy()
#     group['EMA_Close'] = group['close'].ewm(span=14, adjust=False).mean()
#     group['EMA'] = (group['close'] - group['EMA_Close'])/ group['close']

#     rolling_std = group['close'].rolling(window=14).std()
#     group['Volatility'] = rolling_std / group['close']

#     group['RSI'] = calculate_rsi(group['close'])
    
#     # MACD
#     ema12 = group['close'].ewm(span=12, adjust=False).mean()
#     ema26 = group['close'].ewm(span=26, adjust=False).mean()
#     group['MACD'] = ema12 - ema26
#     group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
#     group['MACD_diff'] = (group['MACD'] - group['MACD_Signal']) / group['close']


#     return group

# # Apply technical indicators to each stock (PERMNO)
# df = df.groupby("PERMNO", group_keys=False).apply(add_technical_indicators)

# df.drop(columns=["close", "EMA_Close", "MACD", "MACD_Signal"], inplace=True, errors='ignore')


# Get the current columns
cols = list(df.columns)

# Remove cik, Symbol, y if they exist, to avoid duplicates
for col_to_move in ['cik', 'Symbol', 'y']:
    if col_to_move in cols:
        cols.remove(col_to_move)

# Insert cik and Symbol at positions 2 and 3 (0-based indexing)
cols.insert(2, 'cik')
cols.insert(3, 'Symbol')

# Append y at the end
cols.append('y')

# Reorder the dataframe columns
df = df[cols]


# Save cleaned dataset
df.to_csv(current_dir / "donnees.csv", index=False)
