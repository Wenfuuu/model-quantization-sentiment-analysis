from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_hf_cache = str(_PROJECT_ROOT / ".hf_cache")
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)
os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_ID   = "indobenchmark/indobert-base-p2"
MAX_LENGTH = 128
EPOCHS     = 3
LR         = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
WARMUP_FRACTION = 0.1
GRAD_CLIP  = 1.0
SEEDS      = [42, 123, 456]

ID2LABEL  = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
LABEL2ID  = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = 3

DATA_DIR    = _PROJECT_ROOT / "data" / "processed"
MODELS_DIR  = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"

TRAIN_CSV = DATA_DIR / "smsa_train_v2.csv"
VAL_CSV   = DATA_DIR / "smsa_val_v2.csv"
TEST_SETS = {
    "smsa":  DATA_DIR / "smsa_test_v2.csv",
    "casa":  DATA_DIR / "casa_test_v2.csv",
    "hoasa": DATA_DIR / "hoasa_test_v2.csv",
}

try:
    from src.config import DEVICE
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  [seed] All random states set to {seed}")


class SentimentCSVDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        df = pd.read_csv(path)
        required = {"text", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        df = df.dropna(subset=["text", "label"])
        df["text"]  = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        df = df[df["text"] != ""]
        df = df[df["label"].isin(ID2LABEL)]

        self.texts      = df["text"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_epoch(
    model, loader, optimizer, scheduler, device, *, train: bool
) -> tuple[float, float, float, list, list]:
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    desc = "  train" if train else "  eval "
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc      = accuracy_score(all_labels, all_preds)
    w_f1     = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, w_f1, all_preds, all_labels


def collect_predictions(
    model, loader, device
) -> tuple[list[int], list[int], np.ndarray]:
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  infer ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.append(probs)

    return all_true, all_pred, np.concatenate(all_probs, axis=0)


def compute_metrics(true_labels: list[int], pred_labels: list[int]) -> dict:
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        lbl: {
            "precision": report[lbl]["precision"],
            "recall":    report[lbl]["recall"],
            "f1":        report[lbl]["f1-score"],
            "support":   report[lbl]["support"],
        }
        for lbl in label_names
    }
    return {
        "accuracy":    accuracy_score(true_labels, pred_labels),
        "macro_f1":    f1_score(true_labels, pred_labels, average="macro",    zero_division=0),
        "weighted_f1": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "per_class":   per_class,
    }


def save_predictions(
    path: Path,
    texts: list[str],
    true_labels: list[int],
    pred_labels: list[int],
    probs: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (text, true, pred, prob) in enumerate(
        zip(texts, true_labels, pred_labels, probs)
    ):
        rows.append({
            "sample_id": idx,
            "text":       text,
            "true_label": ID2LABEL[true],
            "pred_label": ID2LABEL[pred],
            "prob_pos":   float(prob[0]),
            "prob_neu":   float(prob[1]),
            "prob_neg":   float(prob[2]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def train_one_seed(seed: int) -> dict:
    print(f"\n{'='*70}")
    print(f"#  SEED {seed}")
    print(f"{'='*70}\n")

    set_seed(seed)

    model_dir   = MODELS_DIR / f"fp32_seed{seed}"
    results_dir = RESULTS_DIR / f"seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    tokenizer.save_pretrained(model_dir)

    print("Loading datasets ...")
    train_set = SentimentCSVDataset(TRAIN_CSV, tokenizer)
    val_set   = SentimentCSVDataset(VAL_CSV,   tokenizer)
    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nHyperparameters: epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, "
          f"wd={WEIGHT_DECAY}, warmup={warmup_steps}/{total_steps} steps\n")

    best_val_wf1 = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"[Epoch {epoch}/{EPOCHS}]")

        tr_loss, tr_acc, tr_wf1, _, _ = run_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, train=True
        )
        va_loss, va_acc, va_wf1, _, _ = run_epoch(
            model, val_loader, None, None, DEVICE, train=False
        )
        elapsed = time.time() - t0

        print(f"  Train → loss={tr_loss:.4f}  acc={tr_acc:.4f}  wF1={tr_wf1:.4f}")
        print(f"  Val   → loss={va_loss:.4f}  acc={va_acc:.4f}  wF1={va_wf1:.4f}  [{elapsed:.0f}s]")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_weighted_f1": tr_wf1,
            "val_loss":   va_loss, "val_acc":   va_acc, "val_weighted_f1":   va_wf1,
        })

        if va_wf1 > best_val_wf1:
            best_val_wf1 = va_wf1
            model.save_pretrained(model_dir)
            print(f"  ✓ New best val wF1={best_val_wf1:.4f}  →  checkpoint saved to {model_dir}")

    print(f"\nReloading best checkpoint from {model_dir} ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)

    all_metrics: dict[str, dict] = {}

    for ds_name, csv_path in TEST_SETS.items():
        if not csv_path.exists():
            print(f"  [SKIP] {ds_name}: file not found → {csv_path}")
            continue

        print(f"\nEvaluating on {ds_name.upper()} ({csv_path.name}) ...")
        ds      = SentimentCSVDataset(csv_path, tokenizer)
        loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        true_labels, pred_labels, probs = collect_predictions(best_model, loader, DEVICE)

        metrics = compute_metrics(true_labels, pred_labels)
        all_metrics[ds_name] = metrics

        label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
        report_str  = classification_report(
            true_labels, pred_labels, target_names=label_names, zero_division=0
        )
        print(f"  accuracy={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}"
              f"  weighted-F1={metrics['weighted_f1']:.4f}")
        print(report_str)

        pred_csv = results_dir / f"fp32_{ds_name}_predictions.csv"
        save_predictions(pred_csv, ds.texts, true_labels, pred_labels, probs)
        print(f"  Predictions saved → {pred_csv}")

    metrics_path = results_dir / "fp32_metrics.json"
    metrics_out = {
        "seed": seed,
        "model_id": MODEL_ID,
        "checkpoint_dir": str(model_dir),
        "hyperparameters": {
            "epochs":           EPOCHS,
            "lr":               LR,
            "weight_decay":     WEIGHT_DECAY,
            "batch_size":       BATCH_SIZE,
            "max_length":       MAX_LENGTH,
            "warmup_fraction":  WARMUP_FRACTION,
            "grad_clip":        GRAD_CLIP,
            "device":           str(DEVICE),
        },
        "best_val_weighted_f1": best_val_wf1,
        "training_history":     history,
        "test_metrics":         all_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved → {metrics_path}")

    return metrics_out


def print_aggregated_summary(seed_results: list[dict]) -> None:
    seeds = [r["seed"] for r in seed_results]

    print("\n" + "=" * 70)
    print("  AGGREGATED SUMMARY ACROSS SEEDS")
    print(f"  Seeds: {seeds}")
    print("=" * 70)
    ds_names = sorted(
        set(ds for r in seed_results for ds in r.get("test_metrics", {}))
    )

    for ds_name in ds_names:
        print(f"\n  [{ds_name.upper()}]")

        for metric_key, label in [
            ("accuracy",    "accuracy   "),
            ("macro_f1",    "macro-F1   "),
            ("weighted_f1", "weighted-F1"),
        ]:
            vals = [
                r["test_metrics"][ds_name][metric_key]
                for r in seed_results
                if ds_name in r.get("test_metrics", {})
            ]
            if not vals:
                continue
            arr = np.array(vals)
            per_seed = "  ".join(
                f"seed={s}: {v:.4f}" for s, v in zip(seeds, vals)
            )
            print(f"    {label}  {arr.mean():.4f} ± {arr.std():.4f}"
                  f"  ({per_seed})")

        if ds_name == "smsa":
            print()
            label_names = list(ID2LABEL.values())
            for lbl in label_names:
                vals = [
                    r["test_metrics"]["smsa"]["per_class"].get(lbl, {}).get("f1", 0.0)
                    for r in seed_results
                    if "smsa" in r.get("test_metrics", {})
                ]
                if not vals:
                    continue
                arr = np.array(vals)
                print(f"    {lbl:<10} F1  {arr.mean():.4f} ± {arr.std():.4f}")

    print("\n" + "=" * 70)

    agg_path = RESULTS_DIR / "fp32_aggregated_metrics.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)

    agg: dict = {"seeds": seeds, "datasets": {}}
    for ds_name in ds_names:
        ds_agg: dict = {}
        for metric_key in ("accuracy", "macro_f1", "weighted_f1"):
            vals = [
                r["test_metrics"][ds_name][metric_key]
                for r in seed_results
                if ds_name in r.get("test_metrics", {})
            ]
            if vals:
                arr = np.array(vals)
                ds_agg[metric_key] = {
                    "mean": float(arr.mean()),
                    "std":  float(arr.std()),
                    "values": [float(v) for v in vals],
                }
        agg["datasets"][ds_name] = ds_agg

    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"  Aggregated metrics saved → {agg_path}\n")


def main() -> None:
    print("\n" + "=" * 70)
    print("  IndoBERT FP32 Fine-tuning  |  3-class Indonesian Sentiment")
    print(f"  Model : {MODEL_ID}")
    print(f"  Device: {DEVICE}")
    print(f"  Seeds : {SEEDS}")
    print("=" * 70)

    missing = []
    for p in [TRAIN_CSV, VAL_CSV]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("\n[ERROR] Missing required data files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun  python scripts/prepare_datasets.py  first.")
        sys.exit(1)

    for ds_name, path in TEST_SETS.items():
        if not path.exists():
            print(f"  [warn] Test set '{ds_name}' not found at {path} — will be skipped.")

    seed_results = []
    for seed in SEEDS:
        result = train_one_seed(seed)
        seed_results.append(result)

    print_aggregated_summary(seed_results)


if __name__ == "__main__":
    main()
