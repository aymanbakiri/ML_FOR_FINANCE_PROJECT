import pandas as pd
from scipy.stats import zscore

# Calculate market cap (absolute price * shares outstanding)
df_quarter_end['market_cap'] = df_quarter_end['prcdy'].abs() * df_quarter_end['cshfdy']

# Drop rows with missing or zero market cap to avoid division issues
df_quarter_end = df_quarter_end[df_quarter_end['market_cap'] > 0]

# Normalize key quarterly variables by market cap
df_quarter_end['niity_norm'] = df_quarter_end['niity'] / df_quarter_end['market_cap']       # Net Income
df_quarter_end['setdy_norm'] = df_quarter_end['setdy'] / df_quarter_end['market_cap']       # Shareholders’ Equity (quarter-end)
df_quarter_end['revty_norm'] = df_quarter_end['revty'] / df_quarter_end['market_cap']       # Revenue
df_quarter_end['capxy_norm'] = df_quarter_end['capxy'] / df_quarter_end['market_cap']       # Capital Expenditures
df_quarter_end['invchy_norm'] = df_quarter_end['invchy'] / df_quarter_end['market_cap']     # Inventory Changes
df_quarter_end['recchy_norm'] = df_quarter_end['recchy'] / df_quarter_end['market_cap']     # Accounts Receivable Changes
df_quarter_end['dlcchy_norm'] = df_quarter_end['dlcchy'] / df_quarter_end['market_cap']     # Current Debt Changes
df_quarter_end['dltry_norm'] = df_quarter_end['dltry'] / df_quarter_end['market_cap']       # Long-Term Debt
df_quarter_end['dpcy_norm'] = df_quarter_end['dpcy'] / df_quarter_end['market_cap']         # Depreciation
df_quarter_end['chechy_norm'] = df_quarter_end['chechy'] / df_quarter_end['market_cap']     # Cash & Cash Equivalents Changes
df_quarter_end['wcapcy_norm'] = df_quarter_end['wcapcy'] / df_quarter_end['market_cap']     # Working Capital
df_quarter_end['scstkcy_norm'] = df_quarter_end['scstkcy'] / df_quarter_end['market_cap']   # Common Shares Outstanding

# Standardize (z-score) all normalized variables for comparability
cols_to_standardize = [
    'niity_norm', 'setdy_norm', 'revty_norm', 'capxy_norm', 'invchy_norm',
    'recchy_norm', 'dlcchy_norm', 'dltry_norm', 'dpcy_norm', 'chechy_norm',
    'wcapcy_norm', 'scstkcy_norm'
]

for col in cols_to_standardize:
    df_quarter_end[col.replace('_norm', '_z')] = zscore(df_quarter_end[col].fillna(0))
