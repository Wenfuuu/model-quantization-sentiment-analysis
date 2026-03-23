import os
import re
import sys
import json
import time
import random
import argparse
import warnings
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_hf_cache = str(_project_root / ".hf_cache")
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)
os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DEVICE

warnings.filterwarnings("ignore")

BASE_DIR        = Path(__file__).parent.parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
_DEFAULT_SAVE_ROOT = BASE_DIR / "models"

MODEL_ID    = "indobenchmark/indobert-base-p2"
MAX_LENGTH  = 128
SEEDS       = [42, 123, 456]

LABEL2ID    = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = [ID2LABEL[i] for i in range(len(LABEL2ID))]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

class SMSADataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        df = pd.read_csv(path)
        df = df.dropna(subset=["text", "label"])
        df["text"]  = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        df = df[df["text"] != ""]
        df = df[df["label"].isin(LABEL2ID.values())]
        df = df.reset_index(drop=True)

        self.raw_texts  = df["text"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.raw_texts)

    def __getitem__(self, idx: int) -> dict:
        text = preprocess_text(self.raw_texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

def run_epoch(
    model, loader, optimizer, scheduler, device, train: bool,
    return_probs: bool = False,
) -> tuple:
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="  train" if train else "  eval ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if return_probs:
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            all_probs.extend(probs)

    avg_loss = total_loss / max(1, len(loader))
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if return_probs:
        return avg_loss, acc, macro_f1, all_preds, all_labels, all_probs
    return avg_loss, acc, macro_f1, all_preds, all_labels


def train_one_seed(seed: int, args, save_dir: Path) -> dict:
    set_all_seeds(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  SEED {seed}  |  IndoBERT -> SMSA 3-label")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"  Data   : {PROCESSED_DIR}")
    print(f"  Save   : {save_dir}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABEL2ID),
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    tokenizer.save_pretrained(save_dir)

    print("\nLoading datasets ...")
    train_set = SMSADataset(PROCESSED_DIR / "smsa_train_v2.csv", tokenizer)
    valid_set = SMSADataset(PROCESSED_DIR / "smsa_val_v2.csv",   tokenizer)
    test_set  = SMSADataset(PROCESSED_DIR / "smsa_test_v2.csv",  tokenizer)
    print(f"  Train: {len(train_set):,}  |  Val: {len(valid_set):,}  |  Test: {len(test_set):,}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = 0.0
    history     = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, train=True,
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, valid_loader, None, None, DEVICE, train=False,
        )
        elapsed = time.time() - t0
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}  macro-F1={train_f1:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  macro-F1={val_f1:.4f}  [{elapsed:.0f}s]")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss":   val_loss,   "val_acc":   val_acc,   "val_f1":   val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(save_dir)
            print(f"  -> New best val macro-F1={best_val_f1:.4f}  checkpoint saved")

    print("\nLoading best checkpoint for test evaluation ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(DEVICE)
    best_model.eval()

    _, test_acc, _, test_preds, test_labels, test_probs = run_epoch(
        best_model, test_loader, None, None, DEVICE, train=False, return_probs=True,
    )

    precision, recall, wf1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="weighted", zero_division=0,
    )
    per_class_f1_arr = f1_score(test_labels, test_preds, average=None, zero_division=0)

    print(f"\n  Test accuracy : {test_acc:.4f}")
    print(f"  Weighted F1   : {wf1:.4f}")
    print()
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, zero_division=0))

    probs_arr = np.array(test_probs)
    pred_df = pd.DataFrame({
        "sample_id":  range(len(test_labels)),
        "text":       test_set.raw_texts,
        "true_label": test_labels,
        "pred_label": test_preds,
        "prob_pos":   probs_arr[:, 0].round(6),
        "prob_neu":   probs_arr[:, 1].round(6),
        "prob_neg":   probs_arr[:, 2].round(6),
    })
    preds_path = save_dir / "test_predictions.csv"
    pred_df.to_csv(preds_path, index=False, encoding="utf-8")
    print(f"  Predictions saved  -> {preds_path}")

    metrics = {
        "seed":               seed,
        "accuracy":           float(test_acc),
        "weighted_precision": float(precision),
        "weighted_recall":    float(recall),
        "weighted_f1":        float(wf1),
        "per_class_f1": {
            LABEL_NAMES[i]: float(per_class_f1_arr[i]) for i in range(len(LABEL_NAMES))
        },
    }
    metrics_path = save_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved      -> {metrics_path}")

    full_results = {
        "model_id": MODEL_ID,
        "save_dir": str(save_dir),
        "seed":     seed,
        "hyperparameters": {
            "epochs":          args.epochs,
            "lr":              args.lr,
            "batch_size":      args.batch_size,
            "max_length":      MAX_LENGTH,
            "device":          str(DEVICE),
            "weight_decay":    0.01,
            "warmup_fraction": 0.1,
            "grad_clip_norm":  1.0,
        },
        "data_files": {
            "train": str(PROCESSED_DIR / "smsa_train_v2.csv"),
            "valid": str(PROCESSED_DIR / "smsa_val_v2.csv"),
            "test":  str(PROCESSED_DIR / "smsa_test_v2.csv"),
        },
        "train_sizes": {
            "train": len(train_set), "valid": len(valid_set), "test": len(test_set),
        },
        "training_history":  history,
        "best_val_macro_f1": float(best_val_f1),
        **metrics,
    }
    with open(save_dir / "finetune_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return metrics

def main(args):
    if args.seed is not None:
        save_dir = (
            Path(args.save_dir) if args.save_dir
            else _DEFAULT_SAVE_ROOT / f"fp32_seed{args.seed}"
        )
        train_one_seed(args.seed, args, save_dir)
        return

    save_root   = Path(args.save_root) if args.save_root else _DEFAULT_SAVE_ROOT
    seeds       = args.seeds
    all_metrics = []

    for seed in seeds:
        save_dir = save_root / f"fp32_seed{seed}"
        m = train_one_seed(seed, args, save_dir)
        all_metrics.append(m)

    accs = [m["accuracy"]    for m in all_metrics]
    wf1s = [m["weighted_f1"] for m in all_metrics]

    print("\n" + "=" * 60)
    print("  MULTI-SEED SUMMARY  (SmSA test set)")
    print("=" * 60)
    print(f"  Seeds : {seeds}")
    for m in all_metrics:
        print(f"  seed={m['seed']:3d}  acc={m['accuracy']:.4f}  weighted-F1={m['weighted_f1']:.4f}")
    print(f"  ---")
    print(f"  Accuracy    mean={np.mean(accs):.4f}  std={np.std(accs, ddof=0):.4f}")
    print(f"  Weighted-F1 mean={np.mean(wf1s):.4f}  std={np.std(wf1s, ddof=0):.4f}")
    print("=" * 60)

    agg_path = save_root / "aggregated_results.json"
    agg_data = {
        "seeds":      seeds,
        "per_seed":   all_metrics,
        "accuracy":    {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=0))},
        "weighted_f1": {"mean": float(np.mean(wf1s)), "std": float(np.std(wf1s, ddof=0))},
    }
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Aggregated results saved -> {agg_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune IndoBERT on SMSA (3-label) from data/processed/smsa_*_v2.csv. "
            "Default: loops over --seeds [42 123 456], saving each to models/fp32_seed<N>/. "
            "Single-seed backward-compat mode: pass --seed N (and optionally --save-dir PATH)."
        )
    )
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="Seeds for multi-seed run (default: 42 123 456).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Single seed (backward-compat). When set, only this seed runs.",
    )
    p.add_argument(
        "--save-root", type=str, default=None,
        help="Root dir for seed subdirs. Default: models/.",
    )
    p.add_argument(
        "--save-dir", type=str, default=None,
        help="Exact save path for single-seed mode (backward-compat).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())