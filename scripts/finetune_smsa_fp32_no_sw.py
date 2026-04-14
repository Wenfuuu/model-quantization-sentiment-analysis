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
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.explanation_drift import spearman_rank_correlation

def _ece(confs, corr):
    return expected_calibration_error(confs, corr, n_bins=10)["ece"]

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
    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  FINETUNING: IndoBERT -> SMSA 3-label (seed={seed})")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Device:  {DEVICE}")
    print(f"  Epochs:  {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Save to: {save_dir}")
    print("=" * 60)

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

    metrics = {
        "accuracy":           test_acc,
        "weighted_precision": cls_report_dict["weighted avg"]["precision"],
        "weighted_recall":    cls_report_dict["weighted avg"]["recall"],
        "weighted_f1":        cls_report_dict["weighted avg"]["f1-score"],
        "per_class_f1": {
            name: cls_report_dict[name]["f1-score"] for name in label_names
        },
    }
    _conf = np.max(test_probs, axis=1).tolist()
    _corr = [int(p == t) for p, t in zip(test_preds, test_labels)]
    metrics["ece"] = _ece(_conf, _corr)
    print(f"Test ECE:       {metrics['ece']:.4f}")
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")

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

def _write_train_log(log_path: Path, seed: int, history: list) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# FP32 Control Retrain — seed={seed}\n")
        f.write("# epoch | train_loss | train_acc | train_f1 | val_loss | val_acc | val_f1\n")
        for h in history:
            f.write(
                f"{h['epoch']} | {h['train_loss']:.6f} | {h['train_acc']:.6f} | "
                f"{h['train_f1']:.6f} | {h['val_loss']:.6f} | "
                f"{h['val_acc']:.6f} | {h['val_f1']:.6f}\n"
            )
    print(f"  Training log saved -> {log_path}")

def _compute_attribution_stability(
    baseline_dir: Path,
    control_dir: Path,
    test_texts: list,
    seed: int,
    n_samples: int = 100,
) -> dict:
    from src.xai.integrated_gradients import IntegratedGradientsExplainer

    print(f"\n  [stability] Loading baseline from {baseline_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(baseline_dir))
    model_a = AutoModelForSequenceClassification.from_pretrained(
        str(baseline_dir), num_labels=len(LABEL2ID), ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model_a.eval()

    print(f"  [stability] Loading control from {control_dir} ...")
    model_b = AutoModelForSequenceClassification.from_pretrained(
        str(control_dir), num_labels=len(LABEL2ID), ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model_b.eval()

    explainer_a = IntegratedGradientsExplainer(model_a, tokenizer, DEVICE, precision="fp32")
    explainer_b = IntegratedGradientsExplainer(model_b, tokenizer, DEVICE, precision="fp32")

    texts = test_texts[:n_samples]
    rhos = []
    print(f"  [stability] Computing IG attribution stability over {len(texts)} samples ...")
    for i, text in enumerate(texts):
        try:
            attr_a = explainer_a.explain(text)
            attr_b = explainer_b.explain(text)
            scores_a = attr_a["scores"]
            scores_b = attr_b["scores"]
            if hasattr(scores_a, "tolist"):
                scores_a = scores_a.tolist()
            if hasattr(scores_b, "tolist"):
                scores_b = scores_b.tolist()
            rho, _ = spearman_rank_correlation(
                attr_a["tokens"], scores_a,
                attr_b["tokens"], scores_b,
            )
            if not np.isnan(rho):
                rhos.append(rho)
        except Exception as exc:
            warnings.warn(f"  [stability] Sample {i} skipped: {exc}")
            continue
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(texts)} samples done  (running mean rho={np.mean(rhos):.4f})")

    del model_a, model_b

    rho_mean = float(np.mean(rhos)) if rhos else float("nan")
    rho_std  = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0
    return {"seed": seed, "rho_mean": rho_mean, "rho_std": rho_std, "n_samples": len(rhos)}


def _aggregate_stability_results(seeds: list) -> dict:
    """Read per-seed stability JSONs and write stability_summary.json."""
    results_dir = BASE_DIR / "results" / "fp32_control"
    rho_means = []
    for seed in seeds:
        p = results_dir / f"stability_seed{seed}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            v = data.get("rho_mean", float("nan"))
            if not np.isnan(v):
                rho_means.append(v)

    if not rho_means:
        print("  [stability] No stability results found; skipping summary.")
        return {}

    overall_mean = float(np.mean(rho_means))
    overall_std  = float(np.std(rho_means, ddof=1)) if len(rho_means) > 1 else 0.0
    interpretation = "retraining_clean" if overall_mean > 0.97 else "retraining_confounded"

    summary = {
        "rho_mean":       overall_mean,
        "rho_std":        overall_std,
        "interpretation": interpretation,
    }
    out_path = results_dir / "stability_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Stability summary saved -> {out_path}")
    print(f"  rho_mean={overall_mean:.4f}  rho_std={overall_std:.4f}  → {interpretation}")
    return summary

def main(args):
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = args.seeds

    _ctrl = getattr(args, "control_retrain", False)
    _mode = "FP32-CONTROL (no fake-quant observers)" if _ctrl else "FP32-baseline"
    print("\n" + "=" * 60)
    print(f"  MULTI-SEED FINETUNING [{_mode}]: IndoBERT -> SMSA 3-label")
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
        elif getattr(args, "control_retrain", False):
            save_dir = SAVE_BASE / f"fp32_control_seed{seed}"
        else:
            save_dir = SAVE_BASE / f"fp32_seed{seed}"

        result = train_single_seed(
            seed, save_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        all_results.append(result)

        _ckpt_root = BASE_DIR / "checkpoints"
        _ckpt_root.mkdir(parents=True, exist_ok=True)
        if getattr(args, "control_retrain", False):
            ckpt_path = SAVE_BASE / f"fp32_control_seed{seed}.pt"
            _canonical  = _ckpt_root / f"fp32_control_seed{seed}.pt"
        else:
            ckpt_path  = SAVE_BASE / f"fp32_seed{seed}.pt"
            _canonical = _ckpt_root / f"fp32_seed{seed}.pt"
        _saved_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        torch.save(_saved_model.state_dict(), ckpt_path)
        torch.save(_saved_model.state_dict(), _canonical)
        del _saved_model
        print(f"  Checkpoint saved -> {_canonical}")

        if getattr(args, "control_retrain", False):
            log_path = BASE_DIR / "logs" / f"fp32_control_seed{seed}_train.log"
            _write_train_log(log_path, seed, result["training_history"])

            baseline_dir = SAVE_BASE / f"fp32_seed{seed}"
            if not baseline_dir.exists():
                warnings.warn(
                    f"  [stability] Baseline checkpoint not found at {baseline_dir}. "
                    "Run normal FP32 fine-tuning first.  Skipping stability.",
                    UserWarning,
                )
            else:
                n_stab = getattr(args, "n_stability_samples", 100)
                test_texts = SMSADataset(DATA_DIR / "smsa_test_v2.csv",
                                         AutoTokenizer.from_pretrained(str(baseline_dir))).texts
                stability = _compute_attribution_stability(
                    baseline_dir=baseline_dir,
                    control_dir=save_dir,
                    test_texts=test_texts,
                    seed=seed,
                    n_samples=n_stab,
                )
                stab_dir = BASE_DIR / "results" / "fp32_control"
                stab_dir.mkdir(parents=True, exist_ok=True)
                stab_path = stab_dir / f"stability_seed{seed}.json"
                with open(stab_path, "w", encoding="utf-8") as _f:
                    json.dump(stability, _f, indent=2)
                print(f"  Stability result saved -> {stab_path}")
                print(f"  rho_mean={stability['rho_mean']:.4f}  "
                      f"rho_std={stability['rho_std']:.4f}  "
                      f"n_samples={stability['n_samples']}")

    if getattr(args, "control_retrain", False) and len(all_results) > 1:
        _aggregate_stability_results([r["seed"] for r in all_results])

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
    p.add_argument(
        "--control-retrain",
        action="store_true",
        default=False,
        help=(
            "Retrain FP32 from the pretrained HuggingFace checkpoint "
            "(indobenchmark/indobert-base-p2) with identical hyperparameters "
            "but NO quantization observers. Provides the ablation baseline that "
            "isolates retraining gradient steps (a) from fake-quant noise (b). "
            "Saves to checkpoints/fp32_control_seed{seed}.pt and "
            "logs/fp32_control_seed{seed}_train.log, then automatically runs "
            "IG attribution stability against the original FP32 baseline."
        ),
    )
    p.add_argument(
        "--n-stability-samples",
        type=int,
        default=100,
        help="Number of test samples used for attribution stability evaluation "
             "(only active with --control-retrain). Default: 100.",
    )
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
