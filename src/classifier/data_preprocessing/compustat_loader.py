import pandas as pd
import calendar

# Select a larger fundamental set (YTD flows + per‐share metrics)
fundamental_cols = [
    'revty',    # Revenue YTD
    'saley',    # Sales YTD
    'capxy',    # CapEx YTD
    'oibdpy',   # EBITDA YTD
    'rdipay',   # R&D expense YTD
    'xsgay',    # SG&A expense YTD
    'txpdy',    # Tax provision YTD
    'epsfxy',   # Diluted EPS ex‐extra YTD
    'cshfdy',   # Diluted shares YTD (millions)
    'xoptepsy'  # Option expense per share YTD
]

# Load Monthly CRSP
COMP_PATH = '../data/CompFirmCharac.csv'

df_comp = pd.read_csv(
    COMP_PATH,
    parse_dates=['datadate'], dayfirst=True,
)


# Keep only Industrial & Consolidated
df_comp = df_comp[
    (df_comp['consol'] == 'C')
].copy()

# Trim & parse keys/dates
df_comp['cusip'] = df_comp['cusip'].astype(str).str[:8]
df_comp['datadate'] = pd.to_datetime(df_comp['datadate'])
df_comp = df_comp.dropna(subset=['cusip','datadate']).copy()

# Build “effective_date” = datadate + 45 calendar days,
#      so that we only use Q data ~45 days after quarter‐end.
df_comp['effective_date'] = df_comp['datadate'] + pd.Timedelta(days=45)
df_comp = df_comp.set_index('effective_date').sort_index()

df_comp_small = df_comp[['cusip'] + fundamental_cols].copy()

# For each “cusip + quarter,” drop exact duplicates
df_comp_small = df_comp_small.reset_index().drop_duplicates(
    subset=['cusip','effective_date']
).set_index('effective_date').sort_index()

df_comp = df_comp_small.copy()


