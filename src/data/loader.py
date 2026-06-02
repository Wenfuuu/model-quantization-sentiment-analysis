import os

import pandas as pd

from src.config import (
    DATASET_PATHS,
    EVAL_DATASETS,
    LABELS,
    NUSAX_SOURCE_LABEL_REMAP,
)


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


def load_nusax_ind_dataset(split: str = "test"):
    key = f"nusax_ind_{split}"
    if key not in DATASET_PATHS:
        raise ValueError(f"unknown NusaX split: {split!r}")
    path = DATASET_PATHS[key]
    if not path.exists():
        raise FileNotFoundError(
            f"NusaX prepared file not found: {path}\n"
            "Run: python scripts/prepare_datasets.py --include-nusax"
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]

    samples = []
    for _, row in df.iterrows():
        label = str(row["label"]).strip().lower()
        if label in NUSAX_SOURCE_LABEL_REMAP:
            samples.append({
                "text": str(row["text"]),
                "expected": NUSAX_SOURCE_LABEL_REMAP[label],
            })

    return samples


def load_eval_dataset(name: str):
    if name == "smsa":
        return load_smsa_dataset()
    if name == "nusax_ind":
        return load_nusax_ind_dataset(split="test")
    raise ValueError(
        f"unknown eval dataset: {name!r} (known: {EVAL_DATASETS})"
    )


def select_eval_samples(config: dict):
    env = os.environ.get("EVAL_DATASET", "").strip().lower()
    chosen = env if env in EVAL_DATASETS else config.get("dataset", "smsa")
    if chosen == "nusax_ind":
        return load_nusax_ind_dataset(split="test")
    if chosen == "tweets":
        return load_tweets_dataset()
    return load_smsa_dataset()


def prompt_eval_dataset(default: str = "smsa") -> str:
    env = os.environ.get("EVAL_DATASET", "").strip().lower()
    if env in EVAL_DATASETS:
        print(f"  [eval dataset] using {env} (from EVAL_DATASET env)")
        return env

    print("\n  Select Evaluation Dataset:")
    for i, name in enumerate(EVAL_DATASETS, 1):
        suffix = " (default)" if name == default else ""
        print(f"  [{i}] {name}{suffix}")
    raw = input(f"\n  Enter choice (default {default}): ").strip()
    if not raw:
        return default
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(EVAL_DATASETS):
            return EVAL_DATASETS[idx]
    if raw in EVAL_DATASETS:
        return raw
    print(f"  Unrecognised choice {raw!r}; falling back to {default}.")
    return default
