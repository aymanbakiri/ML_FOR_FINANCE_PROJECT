import pandas as pd
import numpy as np
import calendar

# Load Monthly CRSP
CRSP_PATH = '../data/monthly_crsp.csv'
df_crsp = pd.read_csv(
    CRSP_PATH ,
    parse_dates=['MthCalDt'],
    usecols=['PERMNO','CUSIP','MthCalDt','MthRet']
)

# Keep only rows where MthRet is available and cast to float
df_crsp = df_crsp.dropna(subset=['MthRet']).copy()
df_crsp['MthRet'] = df_crsp['MthRet'].astype(float)

# Sort by CUSIP, date so that shift is correct
df_crsp['date'] = pd.to_datetime(df_crsp['MthCalDt'].astype(str), format='mixed')
df_crsp = df_crsp.sort_values(['CUSIP','date']).reset_index(drop=True)

# Create next‐month return target (binary)
df_crsp['Ret_t1'] = df_crsp.groupby('CUSIP')['MthRet'].shift(-1)
# df_crsp['y'] = df_crsp.groupby('CUSIP')['MthRet'].shift(-1)
df_crsp['y'] = (df_crsp['Ret_t1'] > 0).astype(int)
df_crsp = df_crsp.dropna(subset=['y']).copy()

# Add technical indicators and months

def compute_close(y_series):
    close = (1 + y_series.fillna(0)).cumprod()
    close.iloc[0] = 1.0
    return close

df_crsp = df_crsp.sort_values(["PERMNO", "date"])
df_crsp["close"] = df_crsp.groupby("PERMNO")["MthRet"].apply(compute_close).reset_index(level=0, drop=True)

def calculate_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# Group-wise calculations
def add_technical_indicators(group):
    group = group.copy()
    group['EMA_Close'] = group['close'].ewm(span=14, adjust=False).mean()
    group['EMA'] = (group['close'] - group['EMA_Close'])/ group['close']

    rolling_std = group['close'].rolling(window=14).std()
    group['Volatility'] = rolling_std / group['close']

    group['RSI'] = calculate_rsi(group['close'])
    
    # MACD
    ema12 = group['close'].ewm(span=12, adjust=False).mean()
    ema26 = group['close'].ewm(span=26, adjust=False).mean()
    group['MACD_diff'] = ema12 - ema26
    group['MACD_Signal'] = group['MACD_diff'].ewm(span=9, adjust=False).mean()
    group['MACD'] = (group['MACD_diff'] - group['MACD_Signal']) / group['close']


    return group

# Apply technical indicators to each stock (PERMNO)
df_crsp = df_crsp.groupby("PERMNO", group_keys=False).apply(add_technical_indicators, include_groups=False)

df_crsp.drop(columns=["close", "EMA_Close", "MACD_diff", "MACD_Signal", 'y_forward'], inplace=True, errors='ignore')

df_crsp['month'] = df_crsp['date'].dt.month.map(lambda x: calendar.month_name[x])

month_dummies = pd.get_dummies(
    df_crsp['month'],
    drop_first=True,    # drop one level to avoid perfect multicollinearity
    dtype=int
)

df_crsp = pd.concat([df_crsp, month_dummies], axis=1)
df_crsp = df_crsp.drop(columns=['month'])

# Get the current columns
cols = list(df_crsp.columns)

# Move 'y' to the end if it exists
if 'y' in cols:
    cols.remove('y')
    cols.append('y')

# Reorder the DataFrame columns
df_crsp = df_crsp[cols]

df_crsp = df_crsp.dropna()

# Generate price‐history features (momentum + volatility)
#
#   - 3M momentum: cumulative return over past 3 months (t-3 → t-1)
#   - 6M momentum: cumulative return over past 6 months
#   - 12M momentum: cumulative return over past 12 months
#   - 12M rolling volatility: std of monthly returns over past 12 months
#
def compute_momentum_and_vol(df):
    df = df.sort_values('date')
    # Rolling log(1+return), because cumulative product of (1 + ret) = exp(sum(log(1+ret)))
    df['log1p_ret'] = np.log1p(df['MthRet'])
    df['log1p_ret_shift1'] = df.groupby('CUSIP')['log1p_ret'].shift(1)
    df['cum12_1_log'] = df.groupby('CUSIP')['log1p_ret_shift1'].rolling(window=11).sum().reset_index(0,drop=True)
    df['mom_12_1'] = np.expm1(df['cum12_1_log'])
    df['cum3m_log'] = df.groupby('CUSIP')['log1p_ret'].rolling(window=3, min_periods=3).sum().reset_index(0,drop=True)
    df['cum6m_log'] = df.groupby('CUSIP')['log1p_ret'].rolling(window=6, min_periods=6).sum().reset_index(0,drop=True)
    df['cum12m_log'] = df.groupby('CUSIP')['log1p_ret'].rolling(window=12, min_periods=12).sum().reset_index(0,drop=True)
    df['momentum_3m'] = np.expm1(df['cum3m_log'])    # exp(sum)-1 => (1+r1)*(1+r2)*(1+r3) - 1
    df['momentum_6m'] = np.expm1(df['cum6m_log'])
    df['momentum_12m'] = np.expm1(df['cum12m_log'])
    df['volatility_12m'] = df.groupby('CUSIP')['MthRet'].rolling(window=12, min_periods=12).std().reset_index(0,drop=True)
    # Drop intermediate log columns
    return df.drop(columns=['log1p_ret','cum3m_log','cum6m_log','cum12m_log'])

df_crsp = compute_momentum_and_vol(df_crsp)

# Trim CUSIP to 8 characters (for merging) and drop NA
df_crsp['cusip'] = df_crsp['CUSIP'].astype(str).str[:8]
df_crsp = df_crsp.dropna(subset=['cusip']).copy()

# Set df_crsp index to “date”
df_crsp = df_crsp.set_index('date').sort_index()


