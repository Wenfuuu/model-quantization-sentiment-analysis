import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    BASE_DIR,
    DEVICE,
    LABELS,
    TRAINING_SEEDS,
    MODEL_REGISTRY,
    DEFAULT_MODEL_TAG,
    _tag_suffix,
    fp32_seed_dir,
)
from src.evaluation.batch_eval import evaluate_accuracy

DATASETS = {
    "smsa":      BASE_DIR / "data" / "processed" / "smsa_test_v2.csv",
    "nusax_ind": BASE_DIR / "data" / "processed" / "nusax_ind_test.csv",
}

LABEL_TO_ID = {
    "positive": 0,
    "neutral":  1,
    "negative": 2,
}

class CSVTextDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_length: int = 128):
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{csv_path} must have columns 'text' and 'label'")
        if df["label"].dtype == object:
            mapped = df["label"].astype(str).str.strip().str.lower().map(LABEL_TO_ID)
            if mapped.isna().any():
                bad = df.loc[mapped.isna(), "label"].unique().tolist()
                raise ValueError(f"unmapped labels in {csv_path}: {bad}")
            labels = mapped.astype(int).tolist()
        else:
            labels = df["label"].astype(int).tolist()
        texts = df["text"].astype(str).tolist()
        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels":         self.labels[i],
        }

def evaluate_one(model_dir: Path, csv_path: Path, batch_size: int, device: torch.device) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=len(LABELS))
    model.to(device).eval()
    ds = CSVTextDataset(csv_path, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate_accuracy(model, dl, device)

def main():
    ap = argparse.ArgumentParser(description="Per-seed FP32 accuracy on SmSA or NusaX-Indo (per backbone).")
    ap.add_argument("--dataset",   choices=sorted(DATASETS), required=True)
    ap.add_argument("--model-tag", choices=sorted(MODEL_REGISTRY), default=DEFAULT_MODEL_TAG)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=TRAINING_SEEDS)
    args = ap.parse_args()

    csv_path = DATASETS[args.dataset]
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}; run Stage 1 (prepare_datasets) first")

    out_dir = BASE_DIR / "outputs" / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"accuracy_{args.dataset}{_tag_suffix(args.model_tag)}.json"

    print(f"dataset     = {args.dataset} ({csv_path})")
    print(f"model_tag   = {args.model_tag}")
    print(f"seeds       = {args.seeds}")
    print(f"device      = {DEVICE}")
    print(f"out         = {out_path}\n")

    per_seed = {}
    for seed in args.seeds:
        model_dir = fp32_seed_dir(seed, args.model_tag)
        if not model_dir.exists():
            print(f"  seed={seed:>4}  SKIP (missing {model_dir})")
            continue
        acc = evaluate_one(model_dir, csv_path, args.batch_size, DEVICE)
        print(f"  seed={seed:>4}  acc={acc:.4f}")
        per_seed[str(seed)] = round(acc, 4)

    accs = list(per_seed.values())
    summary = {
        "dataset":   args.dataset,
        "model_tag": args.model_tag,
        "n_seeds":   len(accs),
        "per_seed":  per_seed,
        "mean_acc":  round(statistics.mean(accs), 4) if accs else None,
        "std_acc":   round(statistics.stdev(accs), 4) if len(accs) >= 2 else 0.0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
