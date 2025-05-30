# import pandas as pd

# from pathlib import Path

# current_dir = Path(__file__).parent

# # Load the CSV with cik from previous step
# df_main = pd.read_csv(current_dir / 'df_with_cik.csv')

# # Load the reports dataframe (assuming from an Excel or CSV file)
# df_reports = pd.read_parquet(current_dir / 'mda_text.parquet')

# print(df_main)
# print('')
# print(df_reports)

# # # Normalize cik columns (strip and leading zeros if needed)
# # df_main['cik'] = df_main['cik'].astype(str).str.strip().str.zfill(10)
# # df_reports['cik'] = df_reports['cik'].astype(str).str.strip().str.zfill(10)

# # # Filter df_reports for only 10-K submissions (if needed)
# # df_reports_10k = df_reports[df_reports['submission_type'] == '10-K']

# # # Merge on cik to add the report text and metadata
# # df_merged = pd.merge(
# #     df_main,
# #     df_reports_10k[['cik', 'text']],
# #     how='left',
# #     on='cik'
# # )

# # # Save the merged dataframe with text
# # # df_merged.to_csv("df_with_cik_and_text.csv", index=False)

# # print("Merged dataframe with report text saved as df_with_cik_and_text.csv")
# # # print(df_merged.head())
# # print(df_merged)


import pandas as pd
from pathlib import Path

# Set current directory
current_dir = Path(__file__).parent

# Load main and report datasets
df_main = pd.read_csv(current_dir / 'df_with_cik.csv')
df_main = df_main.head(10)
df_reports = pd.read_parquet(current_dir / 'mda_text.parquet')

# Drop rows where CIK is missing in df_main
df_main = df_main.dropna(subset=['cik'])

# Convert CIKs in df_main to string, remove decimal, and zero-pad to 10 digits
df_main['cik'] = df_main['cik'].astype(int).astype(str).str.zfill(10)

# Normalize CIKs in df_reports
df_reports['cik'] = df_reports['cik'].astype(str).str.strip().str.zfill(10)

# Filter 10-K only
df_reports_10k = df_reports[df_reports['submission_type'] == '10-K']

# Merge on cik
df_merged = pd.merge(
    df_main,
    df_reports_10k[['cik', 'text']],
    how='left',
    on='cik'
)

# Drop rows where text is missing
df_merged = df_merged.dropna(subset=['text'])

# Save result
df_merged.to_csv(current_dir / 'df_with_cik_and_text.csv', index=False)

print(df_merged.head())
