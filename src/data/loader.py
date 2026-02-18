import pandas as pd
from src.config import LABELS, DATASET_PATHS


def load_smsa_dataset():
    sentiment_map = {"positive": "POSITIVE", "neutral": "NEUTRAL", "negative": "NEGATIVE"}
    df = pd.read_csv(DATASET_PATHS["smsa"], sep="\t", header=None, names=["text", "label"])
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]

    samples = []
    for _, row in df.iterrows():
        label = row['label'].strip().lower()
        if label in sentiment_map:
            samples.append({"text": row['text'], "expected": sentiment_map[label]})

    return samples


def load_tweets_dataset():
    df = pd.read_csv(DATASET_PATHS["tweets"], sep="\t", engine="python")
    df_sample = df.sample(frac=1/20, random_state=42)

    samples = []
    for _, row in df_sample.iterrows():
        try:
            sentiment_id = int(row['sentiment'])
            if sentiment_id in LABELS:
                samples.append({"text": row['Tweet'], "expected": LABELS[sentiment_id]})
        except ValueError:
            continue

    return samples
