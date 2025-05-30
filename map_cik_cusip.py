import pandas as pd

from pathlib import Path

current_dir = Path(__file__).parent
reports = pd.read_parquet(current_dir / 'data/mda_text.parquet')

# Load Data
df_main = pd.read_csv(current_dir / 'data/output_data.csv')

# Load cik-cusip Excel mapping file
df_cik_cusip = pd.read_csv(current_dir / 'cik-cusip.csv')

# Normalize CUSIP columns
df_main['CUSIP'] = df_main['CUSIP'].str.upper().str.strip()
df_cik_cusip['cusip6'] = df_cik_cusip['cusip6'].str.upper().str.strip()
df_cik_cusip['cusip8'] = df_cik_cusip['cusip8'].str.upper().str.strip()

# Create helper columns
df_main['cusip8'] = df_main['CUSIP'].str[:8]
df_main['cusip6'] = df_main['CUSIP'].str[:6]

# Merge on 8-digit CUSIP
df_merged = pd.merge(
    df_main,
    df_cik_cusip[['cik', 'cusip8']],
    how='left',
    left_on='cusip8',
    right_on='cusip8'
)

# Identify missing cik rows
missing_mask = df_merged['cik'].isna()

if missing_mask.any():
    # For missing rows, merge on 6-digit CUSIP
    df_missing = df_merged.loc[missing_mask, ['cusip6']].copy()
    df_missing = df_missing.merge(
        df_cik_cusip[['cik', 'cusip6']],
        how='left',
        on='cusip6'
    )
    # Now assign by matching indices
    df_merged.loc[df_missing.index, 'cik'] = df_missing['cik'].values

# Drop helper columns
df_merged.drop(columns=['cusip6', 'cusip8'], inplace=True)

# Save to CSV
df_merged.to_csv("df_with_cik.csv", index=False)

print("Saved merged data with CIK to df_with_cik.csv")
print(df_merged.head())

