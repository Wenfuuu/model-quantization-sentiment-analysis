"""
PTQ (Post-Training Quantization) from FP32 checkpoints — multi-seed.

For each seed in [42, 123, 456]:
  1. Load models/fp32_seed{SEED}/
  2. Apply FP16 / INT8 / INT4 quantization (via src.quantization.ptq)
  3. Evaluate each on SmSA test set
  4. Save quantized models + predictions + metrics

Latency measured on seed 42 only (20 runs, 5 warmup, first 50 samples).
Model sizes measured once (same architecture → identical across seeds).
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
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
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from src.quantization.ptq import PTQQuantizer

warnings.filterwarnings("ignore")

MAX_LENGTH  = 128
SEEDS       = [42, 123, 456]
VARIANTS    = ["fp16", "int8", "int4"]

LATENCY_SEED   = 42
LATENCY_RUNS   = 20
LATENCY_WARMUP = 5
LATENCY_MAX_SAMPLES = 50

ID2LABEL   = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = 3

DATA_DIR    = _PROJECT_ROOT / "data"  / "processed"
MODELS_DIR  = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"

TEST_CSV = DATA_DIR / "smsa_test_v2.csv"

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

class SentimentCSVDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        df = pd.read_csv(path)
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

def collect_predictions(
    model, loader, device,
) -> tuple[list[int], list[int], np.ndarray]:
    """Run inference, returning (true_labels, pred_labels, probs[N,3])."""
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  infer ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits.float()
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.append(probs)

    return all_true, all_pred, np.concatenate(all_probs, axis=0)

def compute_metrics(true_labels: list[int], pred_labels: list[int]) -> dict:
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names, output_dict=True, zero_division=0,
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
    prec, rec, wf1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0,
    )
    return {
        "accuracy":           accuracy_score(true_labels, pred_labels),
        "weighted_precision": float(prec),
        "weighted_recall":    float(rec),
        "weighted_f1":        float(wf1),
        "macro_f1":           f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "per_class":          per_class,
    }

def save_predictions(
    path: Path,
    texts: list[str],
    true_labels: list[int],
    pred_labels: list[int],
    probs: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": idx,
            "text":       text,
            "true_label": ID2LABEL[true],
            "pred_label": ID2LABEL[pred],
            "prob_pos":   float(prob[0]),
            "prob_neu":   float(prob[1]),
            "prob_neg":   float(prob[2]),
        }
        for idx, (text, true, pred, prob) in enumerate(
            zip(texts, true_labels, pred_labels, probs)
        )
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

def get_model_size_mb(model) -> float:
    tmp = tempfile.mktemp(suffix=".pth")
    try:
        torch.save(model.state_dict(), tmp)
        return os.path.getsize(tmp) / (1024 * 1024)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def measure_latency(
    model, tokenizer, texts: list[str], device,
    n_runs: int = LATENCY_RUNS,
    warmup: int = LATENCY_WARMUP,
    max_samples: int = LATENCY_MAX_SAMPLES,
) -> dict:
    model.eval()
    latencies: list[float] = []

    for text in tqdm(texts[:max_samples], desc="  latency", leave=False):
        enc  = tokenizer(text, max_length=MAX_LENGTH, truncation=True,
                         padding="max_length", return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            for _ in range(warmup):
                model(input_ids=ids, attention_mask=mask)
            times: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(input_ids=ids, attention_mask=mask)
                times.append(time.perf_counter() - t0)
        latencies.append(float(np.mean(times)))

    return {
        "mean_ms":   float(np.mean(latencies) * 1000),
        "std_ms":    float(np.std(latencies)  * 1000),
        "min_ms":    float(np.min(latencies)  * 1000),
        "max_ms":    float(np.max(latencies)  * 1000),
        "median_ms": float(np.median(latencies) * 1000),
        "n_samples": len(latencies),
    }

def quantize_model(fp32_model, variant: str):
    ptq = PTQQuantizer(fp32_model)
    if variant == "fp16":
        model, _ = ptq.quantize_fp16()
    elif variant == "int8":
        model, _ = ptq.quantize_int8()
    elif variant == "int4":
        model, _ = ptq.quantize_int4()
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model.eval()
    return model

def save_quantized_checkpoint(model, variant: str, save_dir: Path, tokenizer, fp32_ckpt: Path) -> float:
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)

    if variant == "fp16":
        model.save_pretrained(save_dir)
        for name in ("model.safetensors", "pytorch_model.bin"):
            p = save_dir / name
            if p.exists():
                return p.stat().st_size / (1024 * 1024)
        return get_model_size_mb(model)
    else:
        state_path = save_dir / f"model_{variant}.pth"
        torch.save(model.state_dict(), state_path)
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(fp32_ckpt)
            cfg.save_pretrained(save_dir)
        except Exception:
            pass
        return state_path.stat().st_size / (1024 * 1024)

def load_fp32_smsa_preds(seed: int) -> list[str] | None:
    pred_path = RESULTS_DIR / f"seed{seed}" / "fp32_smsa_predictions.csv"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path)
    return df["pred_label"].tolist()

def ptq_one_seed(seed: int, model_sizes: dict, latency_stats: dict) -> dict:
    print(f"\n{'='*70}")
    print(f"#  PTQ  SEED {seed}")
    print(f"{'='*70}\n")

    set_seed(seed)

    fp32_ckpt   = MODELS_DIR / f"fp32_seed{seed}"
    results_dir = RESULTS_DIR / f"seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not fp32_ckpt.exists():
        raise FileNotFoundError(
            f"FP32 checkpoint not found: {fp32_ckpt}\n"
            "Run scripts/finetune_fp32_multiseed.py first."
        )

    print(f"Loading FP32 checkpoint from {fp32_ckpt} ...")
    tokenizer = AutoTokenizer.from_pretrained(fp32_ckpt)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(
        fp32_ckpt, num_labels=NUM_LABELS, ignore_mismatched_sizes=True,
    )
    fp32_model.eval()

    if "fp32" not in model_sizes:
        model_sizes["fp32"] = get_model_size_mb(fp32_model)
        print(f"  FP32 size: {model_sizes['fp32']:.2f} MB")

    test_ds  = SentimentCSVDataset(TEST_CSV, tokenizer)
    test_ldr = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"  Test set: {len(test_ds)} samples")

    fp32_preds = load_fp32_smsa_preds(seed)

    if seed == LATENCY_SEED and "fp32" not in latency_stats:
        fp32_cpu = fp32_model.cpu()
        fp32_cpu.eval()
        latency_stats["fp32"] = measure_latency(
            fp32_cpu, tokenizer, test_ds.texts, torch.device("cpu"),
        )
        print(f"  FP32 latency: {latency_stats['fp32']['mean_ms']:.2f} ms/sample")

    seed_metrics: dict[str, dict] = {}

    for variant in VARIANTS:
        print(f"\n  --- {variant.upper()} ---")

        q_model = quantize_model(fp32_model, variant)
        device = torch.device("cpu")

        save_dir = MODELS_DIR / f"ptq_{variant}_seed{seed}"
        size_mb  = save_quantized_checkpoint(
            q_model, variant, save_dir, tokenizer, fp32_ckpt,
        )
        if variant not in model_sizes:
            model_sizes[variant] = size_mb
        print(f"  saved: {save_dir}  ({size_mb:.2f} MB)")

        true_labels, pred_labels, probs = collect_predictions(q_model, test_ldr, device)
        metrics = compute_metrics(true_labels, pred_labels)
        seed_metrics[variant] = metrics

        label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
        print(f"  accuracy={metrics['accuracy']:.4f}  "
              f"macro-F1={metrics['macro_f1']:.4f}  "
              f"weighted-F1={metrics['weighted_f1']:.4f}")
        print(classification_report(true_labels, pred_labels,
                                    target_names=label_names, zero_division=0))

        pred_csv = results_dir / f"ptq_{variant}_smsa_predictions.csv"
        save_predictions(pred_csv, test_ds.texts, true_labels, pred_labels, probs)
        print(f"  predictions -> {pred_csv}")

        if fp32_preds is not None:
            ptq_label_strs = [ID2LABEL[p] for p in pred_labels]
            n_agree = sum(a == b for a, b in zip(fp32_preds, ptq_label_strs))
            agr = n_agree / max(1, len(fp32_preds))
            metrics["agreement_with_fp32"] = agr
            print(f"  agreement vs FP32: {agr*100:.2f}%")
        else:
            metrics["agreement_with_fp32"] = None
            print("  agreement: FP32 preds not found — skipped")

        if seed == LATENCY_SEED and variant not in latency_stats:
            latency_stats[variant] = measure_latency(
                q_model, tokenizer, test_ds.texts, device,
            )
            print(f"  latency: {latency_stats[variant]['mean_ms']:.2f} ms/sample")

        del q_model

    metrics_out = {
        "seed": seed,
        "source_fp32_checkpoint": str(fp32_ckpt),
        "variants": seed_metrics,
    }
    metrics_path = results_dir / "ptq_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrics saved -> {metrics_path}")

    return metrics_out

def print_aggregated_table(
    seed_results: list[dict],
    model_sizes: dict,
    latency_stats: dict,
) -> None:
    seeds = [r["seed"] for r in seed_results]

    print("\n" + "=" * 70)
    print("  AGGREGATED SUMMARY — PTQ from FP32 checkpoints")
    print(f"  Seeds: {seeds}")
    print("=" * 70)

    header = f"  {'Variant':<8} {'Accuracy':>18} {'Macro-F1':>18} {'Weighted-F1':>18} {'Size MB':>9}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for variant in VARIANTS:
        accs = [r["variants"][variant]["accuracy"] for r in seed_results]
        mf1s = [r["variants"][variant]["macro_f1"] for r in seed_results]
        wf1s = [r["variants"][variant]["weighted_f1"] for r in seed_results]
        size = model_sizes.get(variant, 0)
        print(f"  {variant.upper():<8}"
              f" {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
              f" {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f}"
              f" {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}"
              f" {size:>8.1f}")

    if "fp32" in model_sizes:
        print(f"  {'FP32':<8} {'(baseline)':>18} {'':>18} {'':>18} {model_sizes['fp32']:>8.1f}")

    if "fp32" in model_sizes:
        fp32_sz = model_sizes["fp32"]
        print(f"\n  Size reduction vs FP32 ({fp32_sz:.1f} MB):")
        for variant in VARIANTS:
            sz = model_sizes.get(variant, 0)
            reduction = (1 - sz / fp32_sz) * 100 if fp32_sz > 0 else 0
            print(f"    {variant.upper()}: {sz:.1f} MB  ({reduction:.1f}% smaller)")

    if latency_stats:
        print(f"\n  Latency (seed {LATENCY_SEED}, {LATENCY_RUNS} runs, {LATENCY_WARMUP} warmup):")
        for key in ["fp32"] + VARIANTS:
            if key in latency_stats:
                lat = latency_stats[key]
                print(f"    {key.upper():<8} {lat['mean_ms']:.2f} +/- {lat['std_ms']:.2f} ms/sample")

    print(f"\n  Per-seed accuracy:")
    for variant in VARIANTS:
        per = "  ".join(
            f"seed={r['seed']}: {r['variants'][variant]['accuracy']:.4f}"
            for r in seed_results
        )
        print(f"    {variant.upper():<8} {per}")

    print(f"\n  Agreement rate vs FP32 (SmSA):")
    for variant in VARIANTS:
        parts = []
        for r in seed_results:
            agr = r["variants"][variant].get("agreement_with_fp32")
            parts.append(
                f"seed={r['seed']}: {agr*100:.2f}%" if agr is not None else f"seed={r['seed']}: n/a"
            )
        print(f"    {variant.upper():<8} {'  '.join(parts)}")

    agg_path = RESULTS_DIR / "ptq_aggregated_metrics.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)

    agg: dict = {
        "seeds": seeds,
        "model_sizes_mb": model_sizes,
        "latency_seed42": latency_stats,
        "variants": {},
    }
    for variant in VARIANTS:
        accs = [r["variants"][variant]["accuracy"]    for r in seed_results]
        wf1s = [r["variants"][variant]["weighted_f1"] for r in seed_results]
        agg["variants"][variant] = {
            "accuracy":    {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "weighted_f1": {"mean": float(np.mean(wf1s)), "std": float(np.std(wf1s))},
            "per_seed_accuracy": {r["seed"]: r["variants"][variant]["accuracy"] for r in seed_results},
        }

    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"\n  Aggregated metrics saved -> {agg_path}")
    print("=" * 70)

def main() -> None:
    print("\n" + "=" * 70)
    print("  PTQ from FP32 Checkpoints  |  FP16 / INT8 / INT4")
    print(f"  Device : {DEVICE}  (PTQ models run on CPU)")
    print(f"  Seeds  : {SEEDS}")
    print(f"  Latency: seed {LATENCY_SEED} only, {LATENCY_RUNS} runs, {LATENCY_WARMUP} warmup")
    print("=" * 70)

    missing: list[str] = []
    if not TEST_CSV.exists():
        missing.append(str(TEST_CSV))
    for seed in SEEDS:
        ckpt = MODELS_DIR / f"fp32_seed{seed}"
        if not ckpt.exists():
            missing.append(str(ckpt))
    if missing:
        print("\n[ERROR] Missing required files/directories:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    model_sizes:   dict[str, float] = {}
    latency_stats: dict[str, dict]  = {}
    seed_results:  list[dict]       = []

    for seed in SEEDS:
        result = ptq_one_seed(seed, model_sizes, latency_stats)
        seed_results.append(result)

    print_aggregated_table(seed_results, model_sizes, latency_stats)

if __name__ == "__main__":
    main()