from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import LABELS, DEVICE
from .config import QATConfig


class QATTrainer:
    def __init__(self, config: QATConfig):
        self.config = config
        self.device = config.device or DEVICE

        self.dataset = load_from_disk(str(config.data_path))
        self.train_loader = DataLoader(
            self.dataset["train"],
            batch_size=config.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.dataset["validation"],
            batch_size=config.batch_size,
            shuffle=False,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            num_labels=len(LABELS),
        )

        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(self.model, inplace=True)

        if config.prepared_checkpoint:
            state = torch.load(config.prepared_checkpoint, map_location="cpu")
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = int(config.warmup_ratio * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _move_batch(self, batch: dict) -> dict:
        batch = dict(batch)
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="QAT Training"):
            batch = self._move_batch(batch)
            outputs = self.model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    @torch.inference_mode()
    def _evaluate(self) -> float:
        self.model.eval()
        preds, labels = [], []

        for batch in tqdm(self.val_loader, desc="QAT Eval"):
            batch = self._move_batch(batch)
            outputs = self.model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

        return float(np.mean(np.array(preds) == np.array(labels)))

    def train(self) -> Tuple[float, Path]:
        best_acc = 0.0
        best_path = None

        for epoch in range(self.config.epochs):
            print(f"\n--- QAT Epoch {epoch + 1}/{self.config.epochs} ---")
            train_loss = self._train_epoch()
            val_acc = self._evaluate()
            print(f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = self.config.save_dir / "qat_trained.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best QAT checkpoint to {best_path}")

        print(f"\nQAT complete. Best Validation Accuracy: {best_acc:.4f}")
        return best_acc, best_path

import json
import random
import time
import warnings

import pandas as pd
import torch.nn.functional as F
import torch.quantization as tq
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoTokenizer

from src.config import DEVICE as _DEVICE
from src.evaluation.calibration import expected_calibration_error

def _ece(confs, corr):
    return expected_calibration_error(confs, corr, n_bins=10)["ece"]

_ID2LABEL  = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
_NUM_LABELS = 3


class SentimentCSVDataset(Dataset):
    """Load preprocessed SmSA CSV (columns: text, label)."""

    def __init__(self, path: Path, tokenizer, max_length: int = 128):
        df = pd.read_csv(path)
        df = df.dropna(subset=["text", "label"])
        df["text"]  = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        df = df[df["text"] != ""]
        df = df[df["label"].isin(_ID2LABEL)]

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


def apply_qat_config(model) -> None:
    model.train()
    model.qconfig = tq.QConfig(
        activation=tq.default_fake_quant,
        weight=tq.default_weight_fake_quant,
    )
    if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
        model.bert.embeddings.qconfig = None
        print("  [qat] Embedding layer excluded from quantization")
    tq.prepare_qat(model, inplace=True)
    print("  [qat] Fake-quantization observers attached (INT8-level noise)")


def strip_observers(state_dict: dict) -> dict:
    observer_markers = (
        "activation_post_process",
        "fake_quant",
        "scale",
        "zero_point",
        "observer_enabled",
        "fake_quant_enabled",
    )
    return {
        k: v for k, v in state_dict.items()
        if not any(m in k for m in observer_markers)
    }


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  [seed] All random states set to {seed}")


def _run_qat_epoch(model, loader, optimizer, scheduler, device, *, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    desc = "  train" if train else "  val  "
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    w_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, w_f1


def _collect_predictions(model, loader, device):
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  test ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_probs.append(probs)

    return all_true, all_pred, np.concatenate(all_probs, axis=0)


def train_qat_seed(
    seed: int,
    *,
    fp32_ckpt: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    models_dir: Path,
    device: torch.device = _DEVICE,
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 16,
    weight_decay: float = 0.01,
    max_length: int = 128,
) -> dict:
    print(f"\n{'='*70}")
    print(f"#  QAT  SEED {seed}")
    print(f"{'='*70}\n")

    _set_all_seeds(seed)

    obs_dir   = models_dir / f"qat_seed{seed}_with_observers"
    clean_dir = models_dir / f"qat_seed{seed}_clean"
    for d in (obs_dir, clean_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not fp32_ckpt.exists():
        raise FileNotFoundError(
            f"FP32 checkpoint not found: {fp32_ckpt}\n"
            "Run scripts/finetune_smsa_fp32_no_sw.py first."
        )

    print(f"Loading FP32 checkpoint from {fp32_ckpt} ...")
    tokenizer = AutoTokenizer.from_pretrained(fp32_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        fp32_ckpt, num_labels=_NUM_LABELS, ignore_mismatched_sizes=True,
    )

    apply_qat_config(model)
    model = model.to(device)

    print("Loading datasets ...")
    train_set = SentimentCSVDataset(train_csv, tokenizer, max_length)
    val_set   = SentimentCSVDataset(val_csv,   tokenizer, max_length)
    test_set  = SentimentCSVDataset(test_csv,  tokenizer, max_length)
    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nHyperparameters: epochs={epochs}, lr={lr}, batch={batch_size}, "
          f"wd={weight_decay}, warmup={warmup_steps}/{total_steps} steps, device={device}\n")

    best_val_wf1    = 0.0
    best_state_dict = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"[Epoch {epoch}/{epochs}]")

        tr_loss, tr_acc, tr_wf1 = _run_qat_epoch(
            model, train_loader, optimizer, scheduler, device, train=True
        )
        va_loss, va_acc, va_wf1 = _run_qat_epoch(
            model, val_loader, None, None, device, train=False
        )
        elapsed = time.time() - t0

        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  wF1={tr_wf1:.4f}")
        print(f"  Val    loss={va_loss:.4f}  acc={va_acc:.4f}  wF1={va_wf1:.4f}  [{elapsed:.0f}s]")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_weighted_f1": tr_wf1,
            "val_loss":   va_loss, "val_acc":   va_acc, "val_weighted_f1":   va_wf1,
        })

        if va_wf1 > best_val_wf1:
            best_val_wf1 = va_wf1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [ckpt] New best val wF1={best_val_wf1:.4f}")

    print(f"\nSaving with-observers checkpoint -> {obs_dir}")
    torch.save(best_state_dict, obs_dir / "qat_state_dict.pt")
    tokenizer.save_pretrained(obs_dir)
    model.config.save_pretrained(obs_dir)
    with open(obs_dir / "qat_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_fp32_checkpoint": str(fp32_ckpt),
            "seed": seed,
            "best_val_weighted_f1": best_val_wf1,
            "qconfig": "default_fake_quant / default_weight_fake_quant",
            "embeddings_excluded": True,
            "training_history": history,
        }, f, indent=2)

    print(f"Saving clean checkpoint -> {clean_dir}")
    clean_state = strip_observers(best_state_dict)
    clean_model = AutoModelForSequenceClassification.from_pretrained(
        fp32_ckpt, num_labels=_NUM_LABELS, ignore_mismatched_sizes=True,
    )
    missing, unexpected = clean_model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in clean model")
    print(f"  Skipped {len(unexpected)} observer keys")
    clean_model.save_pretrained(clean_dir)
    tokenizer.save_pretrained(clean_dir)

    clean_model = clean_model.to(device)
    print(f"\nEvaluating clean model on SmSA test ...")

    true_labels, pred_labels, probs = _collect_predictions(clean_model, test_loader, device)

    label_names = [_ID2LABEL[i] for i in range(_NUM_LABELS)]
    cls_report = classification_report(
        true_labels, pred_labels, target_names=label_names,
        output_dict=True, zero_division=0,
    )

    test_acc = accuracy_score(true_labels, pred_labels)
    test_macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    print(f"  accuracy={test_acc:.4f}  macro-F1={test_macro_f1:.4f}")
    print(classification_report(
        true_labels, pred_labels, target_names=label_names, zero_division=0
    ))

    pred_df = pd.DataFrame({
        "sample_id":  range(len(pred_labels)),
        "text":       test_set.texts,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "prob_pos":   probs[:, 0],
        "prob_neu":   probs[:, 1],
        "prob_neg":   probs[:, 2],
    })
    pred_df.to_csv(clean_dir / "predictions.csv", index=False, encoding="utf-8")
    print(f"  Predictions saved -> {clean_dir / 'predictions.csv'}")

    metrics = {
        "accuracy":           test_acc,
        "weighted_precision": cls_report["weighted avg"]["precision"],
        "weighted_recall":    cls_report["weighted avg"]["recall"],
        "weighted_f1":        cls_report["weighted avg"]["f1-score"],
        "macro_f1":           test_macro_f1,
        "per_class_f1": {lbl: cls_report[lbl]["f1-score"] for lbl in label_names},
    }
    _conf = np.max(probs, axis=1).tolist()
    _corr = [int(p == t) for p, t in zip(pred_labels, true_labels)]
    metrics["ece"] = _ece(_conf, _corr)
    print(f"  ECE={metrics['ece']:.4f}")
    with open(clean_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {clean_dir / 'metrics.json'}")

    fp32_pred_path = fp32_ckpt / "predictions.csv"
    agreement_rate = None
    if fp32_pred_path.exists():
        fp32_df = pd.read_csv(fp32_pred_path)
        fp32_preds = fp32_df["pred_label"].astype(int).tolist()
        n_agree = sum(a == b for a, b in zip(fp32_preds, pred_labels))
        agreement_rate = n_agree / max(1, len(fp32_preds))
        print(f"\n  [agreement] FP32 vs QAT-FP32 on SmSA: {agreement_rate*100:.2f}%")
    else:
        print(f"\n  [agreement] FP32 predictions not found at {fp32_pred_path} -- skipping")

    results = {
        "seed": seed,
        "source_fp32_checkpoint": str(fp32_ckpt),
        "hyperparameters": {
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "batch_size": batch_size, "max_length": max_length,
            "device": str(device),
        },
        "best_val_weighted_f1": best_val_wf1,
        "training_history": history,
        "test_metrics": metrics,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "smsa_agreement_with_fp32": agreement_rate,
    }
    with open(clean_dir / "qat_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
