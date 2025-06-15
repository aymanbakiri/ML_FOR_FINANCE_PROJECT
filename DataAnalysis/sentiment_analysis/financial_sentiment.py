import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter

positive_words = {"growth", "profit", "increase", "gain", "strong", "exceed"}
negative_words = {"loss", "decline", "decrease", "weak", "impairment", "drop"}

def financial_sentiment(text, pos_words, neg_words):
    words = text.lower().split()
    word_counts = Counter(words)
    pos = sum(word_counts[w] for w in pos_words if w in word_counts)
    neg = sum(word_counts[w] for w in neg_words if w in word_counts)
    
    score = pos - neg
    return 1 if score > 0 else -1 if score < 0 else 0

current_dir = Path(__file__).parent
reports = pd.read_parquet(current_dir / 'mda_text.parquet')

# text = str(reports.iloc[0])
text = str(reports.iloc[0]['text'])

sentiment = financial_sentiment(text, positive_words, negative_words)
print(sentiment)