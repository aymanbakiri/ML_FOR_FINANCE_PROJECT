from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd

# We load the pretrained FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# We define the mapping from FinBERT's class indices to sentiment labels
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# We define a function that splits long texts and averages FinBERT predictions across chunks
def chunk_and_classify(text, max_tokens=512, stride=256):
    # We tokenize the entire text without truncation to get the full input_ids
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    n = len(tokens)

    sentiments = []

    # We iterate through the text using a sliding window
    for i in range(0, n, stride):
        chunk = tokens[i:i+max_tokens]
        if len(chunk) == 0:
            continue

        # We decode and re-tokenize to ensure compatibility with the model
        input_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        inputs = tokenizer(input_chunk, return_tensors="pt", truncation=True, max_length=max_tokens)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()
            sentiments.append(probs)

        if i + max_tokens >= n:
            break

    # We average probabilities over all chunks and return the dominant sentiment label
    if sentiments:
        avg_sentiment = np.mean(sentiments, axis=0)
        label = label_map[np.argmax(avg_sentiment)]
        return label
    else:
        return "neutral"

# We apply the function to the DataFrame column that contains the MD&A texts
df["sentiment"] = df["text"].apply(chunk_and_classify)
