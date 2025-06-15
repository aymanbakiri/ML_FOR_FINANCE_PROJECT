import pandas as pd
import yfinance as yf

from pathlib import Path

current_dir = Path(__file__).parent

# Load main CSV containing CUSIP
df = pd.read_csv(current_dir / 'data/df_with_symbol.csv')

# Convert 'date' to datetime just in case
df['date'] = pd.to_datetime(df['date'])

# Create a function to fetch historical data for a single symbol and date
def fetch_yahoo_row(symbol, date):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=date, end=date + pd.Timedelta(days=1))  # fetch that day only
        if not hist.empty:
            row = hist.iloc[0]
            return pd.Series({
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Adj Close': row['Close'],  # or use 'row['Adj Close']'
                'Volume': row['Volume']
            })
        else:
            return pd.Series([None] * 6, index=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    except Exception as e:
        print(f"Error fetching {symbol} on {date}: {e}")
        return pd.Series([None] * 6, index=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Apply row-wise (can be slow; batching preferred for large datasets)
df_yahoo = df.apply(lambda row: fetch_yahoo_row(row['Symbol'], row['date']), axis=1)

# Combine with original DataFrame
df_combined = pd.concat([df, df_yahoo], axis=1)

# Save result
df_combined.to_csv(current_dir / 'data/df_yahoo.csv', index=False)
print("Saved merged data with Yahoo Finance columns to df_with_yahoo_data.csv")
print(df_combined)

