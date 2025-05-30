import pandas as pd

from pathlib import Path

current_dir = Path(__file__).parent

# Load main CSV containing CUSIP
df_main = pd.read_csv(current_dir / 'data/df_with_cik.csv')

# Load cik-cusip Excel mapping file
df_cik_ticker = pd.read_csv(current_dir / 'data/cik-ticker.csv')  # Symbol,Name,LastSale,MarketCap,ADR TSO,IPOyear,Sector,Industry,Summary Quote,X10,CIK

# Normalize CIK columns:
# Remove decimal places, convert to string, strip, and zero-pad
df_main['cik'] = df_main['cik'].apply(lambda x: str(int(float(x))).zfill(10) if pd.notna(x) else x)
df_cik_ticker['CIK'] = df_cik_ticker['CIK'].apply(lambda x: str(int(float(x))).zfill(10) if pd.notna(x) else x)

# Merge on normalized CIK
df_merged = pd.merge(
    df_main,
    df_cik_ticker[['CIK', 'Symbol']],
    how='left',
    left_on='cik',
    right_on='CIK'
)

# Drop duplicate CIK column
df_merged.drop(columns=['CIK'], inplace=True)

# Save final output
output_path = current_dir / 'data/df_with_symbol.csv'
df_merged.to_csv(output_path, index=False)

print(df_merged)