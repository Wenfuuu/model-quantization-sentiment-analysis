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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    warnings.warn("Sastrawi not installed — stopword removal will be skipped.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DEVICE

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "src" / "finetune_3label"
SAVE_DIR   = BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"
MODEL_ID   = "indobenchmark/indobert-base-p2"
MAX_LENGTH = 128

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k.upper() for k, v in LABEL2ID.items()}

def preprocess_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if SASTRAWI_AVAILABLE:
        text = _stopword_remover.remove(text)
    return text

class SMSADataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
        df = df.dropna(subset=["text", "label"])
        df["text"]  = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["text"] != ""]
        df = df[df["label"].isin(LABEL2ID)]

        self.texts  = df["text"].tolist()
        self.labels = [LABEL2ID[l] for l in df["label"]]
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


def main(args):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    history_path = SAVE_DIR / "training_history.json"

    print("=" * 60)
    print("  FINETUNING: IndoBERT → SMSA 3-label")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Device:  {DEVICE}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  LR:      {args.lr}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  Save to: {SAVE_DIR}")
    print("=" * 60)

    print("\nLoading tokenizer and model ...")
    _local_has_config = (SAVE_DIR / "tokenizer_config.json").exists()
    _tok_source = str(SAVE_DIR) if _local_has_config else MODEL_ID
    _has_weights = any(
        (SAVE_DIR / w).exists()
        for w in ["model.safetensors", "pytorch_model.bin"]
    )
    _model_source = str(SAVE_DIR) if _has_weights else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(_tok_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        _model_source,
        num_labels=len(LABEL2ID),
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Tokenizer saved to {SAVE_DIR}")

    print("\nLoading datasets ...")
    train_set = SMSADataset(DATA_DIR / "train.tsv", tokenizer)
    valid_set = SMSADataset(DATA_DIR / "valid.tsv", tokenizer)
    test_set  = SMSADataset(DATA_DIR / "test.tsv",  tokenizer)
    print(f"  Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(0.1 * total_steps)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1   = 0.0
    history   = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, train=True
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = run_epoch(
            model, valid_loader, None, None, DEVICE, train=False
        )
        elapsed = time.time() - epoch_start

        print(f"  Train → loss={train_loss:.4f}  acc={train_acc:.4f}  macro-F1={train_f1:.4f}")
        print(f"  Valid → loss={val_loss:.4f}  acc={val_acc:.4f}  macro-F1={val_f1:.4f}  [{elapsed:.0f}s]")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,  "train_acc": train_acc,  "train_f1": train_f1,
            "val_loss":   val_loss,    "val_acc":   val_acc,    "val_f1":  val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(SAVE_DIR)
            print(f"  ✓ New best val macro-F1={best_f1:.4f} — model saved to {SAVE_DIR}")

    print("\nLoading best checkpoint for test evaluation ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
    best_model.eval()

    _, test_acc, test_f1, test_preds, test_labels = run_epoch(
        best_model, test_loader, None, None, DEVICE, train=False
    )
    label_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
    cls_report  = classification_report(test_labels, test_preds,
                                        target_names=label_names, zero_division=0)
    print(f"\nTest accuracy:  {test_acc:.4f}")
    print(f"Test macro-F1:  {test_f1:.4f}")
    print("\nClassification report:\n" + cls_report)

    results = {
        "model_id": MODEL_ID,
        "save_dir": str(SAVE_DIR),
        "hyperparameters": {
            "epochs":     args.epochs,
            "lr":         args.lr,
            "batch_size": args.batch_size,
            "max_length": MAX_LENGTH,
            "device":     str(DEVICE),
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
    with open(SAVE_DIR / "finetune_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch-size", type=int,   default=16)
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
