import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path


analyzer = SentimentIntensityAnalyzer()

current_dir = Path(__file__).parent
reports = pd.read_parquet(current_dir / 'mda_text.parquet')

# text = str(reports.iloc[0])
text = str(reports.iloc[0]['text'])

score = analyzer.polarity_scores(text)['compound']

if score >= 0.05:
    trend = 1
elif score <= -0.05:
    trend = -1
else:
    trend = 0

print(trend)



