import pandas as pd
import numpy as np
from data_preprocessing.crsp_loader import df_crsp
from data_preprocessing.compustat_loader import df_comp, fundamental_cols




# Merge (for every month, we get the most recent Compustat row ≤ that month’s date)
df_merged = pd.merge_asof(
    left = df_crsp.reset_index(),
    right = df_comp.reset_index(),
    left_on = 'date',
    right_on = 'effective_date',
    by = 'cusip',
    direction = 'backward',
    allow_exact_matches=True
).set_index('date')

# Drop rows where any of our fundamentals are NA, since we can’t compute ratios otherwise
df_merged = df_merged.dropna(subset=fundamental_cols + ['y']).copy()

df = df_merged.copy()

# Engineer simple ratios (safe‐guard divisions by zero)
df['EBIT_margin']      = df['oibdpy'] / df['saley'].replace({0: np.nan})
df['R&D_intensity']    = df['rdipay'] / df['saley'].replace({0: np.nan})
df['SGA_intensity']    = df['xsgay'] / df['saley'].replace({0: np.nan})
df['Tax_rate']         = df['txpdy']  / df['oibdpy'].replace({0: np.nan})
df['Capex_to_Revenue'] = df['capxy']  / df['revty'].replace({0: np.nan})

# Compute QoQ growth rates on YTD fundamentals
for col in ['revty','oibdpy','rdipay','xsgay']:
    df[col + '_QoQ_growth'] = df.groupby('cusip')[col].pct_change()  # (This QY / Last QY) - 1

# Optionally we can drop some raw dollar‐amount YTD columns if we want just ratios
# df = df.drop(columns=['revty','saley','capxy','oibdpy','rdipay','xsgay','txpdy','epsfxy','cshfdy','xoptepsy'])




# Drop any rows where engineered features are NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
