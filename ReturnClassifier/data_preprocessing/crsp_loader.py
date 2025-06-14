import pandas as pd
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

# Add months
df_crsp['month'] = df_crsp['date'].dt.month.map(lambda x: calendar.month_name[x])

# Create dummy variables with month names as column names
month_dummies = pd.get_dummies(df_crsp['month']).astype(int)

# Concatenate dummies with original dataframe
df_crsp = pd.concat([df_crsp, month_dummies], axis=1)

# drop the intermediate 'month' column 
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

#Set df_crsp index to “date”
df_crsp = df_crsp.set_index('date').sort_index()
