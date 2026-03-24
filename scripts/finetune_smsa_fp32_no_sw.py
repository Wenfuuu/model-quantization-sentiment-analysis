import os
import sys
import json
import time
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
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DEVICE
from src.utils import set_seed

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
SAVE_BASE  = BASE_DIR / "models"
MODEL_ID   = "indobenchmark/indobert-base-p2"
MAX_LENGTH = 128
SEEDS      = [42, 123, 456]

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k.upper() for k, v in LABEL2ID.items()}


def preprocess_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SMSADataset(Dataset):
    """Load preprocessed SmSA CSV (columns: text, label)."""

    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        df = pd.read_csv(path)
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]

        self.texts      = df["text"].tolist()
        self.labels     = df["label"].astype(int).tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = preprocess_text(self.texts[idx])
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


def run_epoch(model, loader, optimizer, scheduler, device, train: bool) -> tuple:
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  train" if train else "  val  ", leave=False):
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

    avg_loss  = total_loss / max(1, len(loader))
    acc       = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, all_preds, all_labels


def evaluate_with_probs(model, loader, device):
    """Evaluate and return predictions with per-class probabilities."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  test ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_single_seed(
    seed: int,
    save_dir: Path,
    *,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
) -> dict:
    """Train IndoBERT for one seed; return results dict."""
    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  FINETUNING: IndoBERT -> SMSA 3-label (seed={seed})")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Device:  {DEVICE}")
    print(f"  Epochs:  {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Save to: {save_dir}")
    print("=" * 60)

    # Always load fresh from HuggingFace
    print("\nLoading tokenizer and model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABEL2ID),
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    tokenizer.save_pretrained(save_dir)

    print("\nLoading preprocessed datasets ...")
    train_set = SMSADataset(DATA_DIR / "smsa_train_v2.csv", tokenizer)
    valid_set = SMSADataset(DATA_DIR / "smsa_val_v2.csv",   tokenizer)
    test_set  = SMSADataset(DATA_DIR / "smsa_test_v2.csv",  tokenizer)
    print(f"  Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch}/{epochs}]")

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, train=True
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, valid_loader, None, None, DEVICE, train=False
        )
        elapsed = time.time() - epoch_start

        print(f"  Train -> loss={train_loss:.4f}  acc={train_acc:.4f}  macro-F1={train_f1:.4f}")
        print(f"  Valid -> loss={val_loss:.4f}  acc={val_acc:.4f}  macro-F1={val_f1:.4f}  [{elapsed:.0f}s]")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss":   val_loss,   "val_acc":   val_acc,   "val_f1":  val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(save_dir)
            print(f"  -> New best val macro-F1={best_f1:.4f} -- model saved")

    # ---- Test evaluation ----
    print("\nLoading best checkpoint for test evaluation ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE)

    test_preds, test_labels, test_probs = evaluate_with_probs(
        best_model, test_loader, DEVICE
    )

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    label_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]

    cls_report_dict = classification_report(
        test_labels, test_preds, target_names=label_names,
        zero_division=0, output_dict=True,
    )
    print(f"\nTest accuracy:  {test_acc:.4f}")
    print(f"Test macro-F1:  {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        test_labels, test_preds, target_names=label_names, zero_division=0
    ))

    # ---- Save predictions CSV ----
    pred_df = pd.DataFrame({
        "sample_id":  range(len(test_preds)),
        "text":       test_set.texts,
        "true_label": test_labels,
        "pred_label": test_preds,
        "prob_pos":   test_probs[:, 0],
        "prob_neu":   test_probs[:, 1],
        "prob_neg":   test_probs[:, 2],
    })
    pred_path = save_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"Predictions saved -> {pred_path}")

    # ---- Save metrics JSON ----
    metrics = {
        "accuracy":           test_acc,
        "weighted_precision": cls_report_dict["weighted avg"]["precision"],
        "weighted_recall":    cls_report_dict["weighted avg"]["recall"],
        "weighted_f1":        cls_report_dict["weighted avg"]["f1-score"],
        "per_class_f1": {
            name: cls_report_dict[name]["f1-score"] for name in label_names
        },
    }
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")

    # ---- Save full results (backward compat with aggregation utils) ----
    results = {
        "model_id": MODEL_ID,
        "save_dir": str(save_dir),
        "seed": seed,
        "hyperparameters": {
            "epochs":     epochs,
            "lr":         lr,
            "batch_size": batch_size,
            "max_length": MAX_LENGTH,
            "device":     str(DEVICE),
            "weight_decay": 0.01,
            "warmup_fraction": 0.1,
            "grad_clip_norm": 1.0,
        },
        "train_sizes": {
            "train": len(train_set),
            "valid": len(valid_set),
            "test":  len(test_set),
        },
        "training_history": history,
        "best_val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }
    with open(save_dir / "finetune_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return results


def main(args):
    # Backward compat: --seed (single) for subprocess mode
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = args.seeds

    print("\n" + "=" * 60)
    print("  MULTI-SEED FINETUNING: IndoBERT -> SMSA 3-label")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print("=" * 60)

    all_results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#' * 60}")
        print(f"# SEED {i}/{len(seeds)}: {seed}")
        print(f"{'#' * 60}\n")

        if args.save_dir and len(seeds) == 1:
            save_dir = Path(args.save_dir)
        else:
            save_dir = SAVE_BASE / f"fp32_seed{seed}"

        result = train_single_seed(
            seed, save_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        all_results.append(result)

    # ---- Aggregated summary ----
    if len(all_results) > 1:
        accs = [r["test_accuracy"] for r in all_results]
        f1s  = [r["test_macro_f1"] for r in all_results]
        print("\n" + "=" * 60)
        print("  AGGREGATED RESULTS")
        print("=" * 60)
        for r in all_results:
            print(f"  Seed {r['seed']:>4d}:  acc={r['test_accuracy']:.4f}  macro-F1={r['test_macro_f1']:.4f}")
        print(f"\n  SmSA accuracy:  {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}")
        print(f"  SmSA macro-F1:  {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}")
        print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune IndoBERT on SMSA (3-label) with multi-seed support. "
                    "Uses preprocessed CSV data from data/processed/.",
    )
    p.add_argument("--seeds",      type=int, nargs="+", default=SEEDS,
                   help="Seeds for multi-seed training (default: 42 123 456).")
    p.add_argument("--seed",       type=int, default=None,
                   help="Single seed (backward compat; takes precedence over --seeds).")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--save-dir",   type=str,   default=None,
                   help="Override save directory (single-seed mode only).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
