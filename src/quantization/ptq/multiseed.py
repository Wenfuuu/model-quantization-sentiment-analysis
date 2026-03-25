from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.quantization.ptq.engine import PTQQuantizer
from src.quantization.utils import save_quantized_model, get_model_size
from src.evaluation.evaluator import get_model_param_memory_mb
from src.quantization.qat.trainer import SentimentCSVDataset
from src.evaluation.calibration import expected_calibration_error

def _ece(confs, corr):
    return expected_calibration_error(confs, corr, n_bins=10)["ece"]

_LABEL_NAMES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
_NUM_LABELS = 3

def _collect_predictions(model, loader):
    """Batch inference → (true_labels, pred_labels, probs_array)."""
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", leave=False):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(logits.float().argmax(dim=-1).cpu().numpy().tolist())
            all_probs.append(probs)

    return all_true, all_pred, np.concatenate(all_probs, axis=0)


def _compute_metrics(true_labels, pred_labels):
    cls_report = classification_report(
        true_labels, pred_labels, target_names=_LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "weighted_precision": cls_report["weighted avg"]["precision"],
        "weighted_recall": cls_report["weighted avg"]["recall"],
        "weighted_f1": cls_report["weighted avg"]["f1-score"],
        "per_class_f1": {lbl: cls_report[lbl]["f1-score"] for lbl in _LABEL_NAMES},
    }


def _save_predictions_csv(pred_labels, true_labels, probs, texts, save_path):
    pd.DataFrame({
        "sample_id": range(len(pred_labels)),
        "text": texts,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "prob_pos": probs[:, 0],
        "prob_neu": probs[:, 1],
        "prob_neg": probs[:, 2],
    }).to_csv(save_path, index=False, encoding="utf-8")


def _measure_latency(model, tokenizer, test_texts, *, num_runs=20, warmup=5):
    model.eval()
    all_latencies = []

    for text in tqdm(test_texts, desc="  latency", leave=False):
        inputs = tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        )

        with torch.no_grad():
            for _ in range(warmup):
                model(**inputs)

            for _ in range(num_runs):
                start = time.perf_counter()
                model(**inputs)
                end = time.perf_counter()
                all_latencies.append(end - start)

    return {
        "mean_ms": float(np.mean(all_latencies) * 1000),
        "std_ms": float(np.std(all_latencies) * 1000),
        "median_ms": float(np.median(all_latencies) * 1000),
        "min_ms": float(np.min(all_latencies) * 1000),
        "max_ms": float(np.max(all_latencies) * 1000),
        "num_runs_per_sample": num_runs,
        "warmup_per_sample": warmup,
        "total_samples": len(test_texts),
    }

def _evaluate_variant(variant_name, model, test_loader, test_texts,
                      fp32_pred, save_dir, *, measure_latency, tokenizer,
                      num_runs, warmup):
    save_dir.mkdir(parents=True, exist_ok=True)

    true_labels, pred_labels, probs = _collect_predictions(model, test_loader)
    metrics = _compute_metrics(true_labels, pred_labels)
    _conf = np.max(probs, axis=1).tolist()
    _corr = [int(p == t) for p, t in zip(pred_labels, true_labels)]
    metrics["ece"] = _ece(_conf, _corr)
    print(f"  {variant_name.upper()} accuracy={metrics['accuracy']:.4f}  "
          f"macro-F1={metrics['macro_f1']:.4f}  ECE={metrics['ece']:.4f}")

    model_path = save_dir / f"model_{variant_name}.pth"
    save_quantized_model(model, model_path)
    size_mb = get_model_size(model_path)
    param_mb = get_model_param_memory_mb(model)
    print(f"  Model saved: {model_path} ({size_mb:.2f} MB)")

    _save_predictions_csv(pred_labels, true_labels, probs, test_texts,
                          save_dir / "predictions.csv")

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    n_agree = sum(a == b for a, b in zip(fp32_pred, pred_labels))
    agreement = n_agree / max(1, len(fp32_pred))
    print(f"  vs FP32 agreement: {agreement * 100:.2f}%")

    result = {
        "metrics": metrics,
        "agreement_with_fp32": agreement,
        "model_size_mb": size_mb,
        "param_memory_mb": param_mb,
    }

    if measure_latency:
        print(f"  Measuring latency ({num_runs} runs, {warmup} warmup) ...")
        lat = _measure_latency(model, tokenizer, test_texts,
                               num_runs=num_runs, warmup=warmup)
        result["latency"] = lat
        print(f"  mean={lat['mean_ms']:.2f}ms  median={lat['median_ms']:.2f}ms")

    return result, pred_labels

def ptq_single_seed(
    seed: int,
    *,
    fp32_ckpt: Path,
    test_csv: Path,
    models_dir: Path,
    batch_size: int = 16,
    measure_latency: bool = False,
    num_runs: int = 20,
    warmup: int = 5,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"#  PTQ  SEED {seed}")
    print(f"{'=' * 70}\n")

    if not fp32_ckpt.exists():
        raise FileNotFoundError(
            f"FP32 checkpoint not found: {fp32_ckpt}\n"
            "Run finetuning first."
        )

    print(f"Loading FP32 checkpoint from {fp32_ckpt} ...")
    tokenizer = AutoTokenizer.from_pretrained(fp32_ckpt)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(
        fp32_ckpt, num_labels=_NUM_LABELS, ignore_mismatched_sizes=True,
    ).cpu().eval()

    test_set = SentimentCSVDataset(test_csv, tokenizer, max_length=128)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    test_texts = test_set.texts
    print(f"  Test samples: {len(test_set)}")

    print("\n  [FP32] Evaluating baseline ...")
    fp32_true, fp32_pred, fp32_probs = _collect_predictions(fp32_model, test_loader)
    fp32_metrics = _compute_metrics(fp32_true, fp32_pred)
    _conf = np.max(fp32_probs, axis=1).tolist()
    _corr = [int(p == t) for p, t in zip(fp32_pred, fp32_true)]
    fp32_metrics["ece"] = _ece(_conf, _corr)
    print(f"  FP32 accuracy={fp32_metrics['accuracy']:.4f}  "
          f"macro-F1={fp32_metrics['macro_f1']:.4f}  ECE={fp32_metrics['ece']:.4f}")

    fp32_save_dir = models_dir / f"ptq_fp32_seed{seed}"
    fp32_save_dir.mkdir(parents=True, exist_ok=True)
    _save_predictions_csv(fp32_pred, fp32_true, fp32_probs, test_texts,
                          fp32_save_dir / "predictions.csv")
    with open(fp32_save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(fp32_metrics, f, indent=2)

    fp32_model_path = fp32_save_dir / "model_fp32.pth"
    save_quantized_model(fp32_model, fp32_model_path)
    fp32_size_mb = get_model_size(fp32_model_path)

    seed_results = {
        "seed": seed,
        "source_fp32_checkpoint": str(fp32_ckpt),
        "fp32": {"metrics": fp32_metrics, "model_size_mb": fp32_size_mb},
    }

    if measure_latency:
        print(f"  [FP32] Measuring latency ({num_runs} runs, {warmup} warmup) ...")
        fp32_lat = _measure_latency(fp32_model, tokenizer, test_texts,
                                    num_runs=num_runs, warmup=warmup)
        seed_results["fp32"]["latency"] = fp32_lat
        print(f"  mean={fp32_lat['mean_ms']:.2f}ms  median={fp32_lat['median_ms']:.2f}ms")

    ptq = PTQQuantizer(fp32_model)

    variant_specs = [
        ("fp16", ptq.quantize_fp16),
        ("int8", ptq.quantize_int8),
        ("int4", ptq.quantize_int4),
    ]

    for vname, quant_fn in variant_specs:
        print(f"\n  [{vname.upper()}] Quantizing ...")
        q_model, q_time = quant_fn()
        print(f"  Quantization time: {q_time:.2f}s")

        save_dir = models_dir / f"ptq_{vname}_seed{seed}"
        result, _ = _evaluate_variant(
            vname, q_model, test_loader, test_texts, fp32_pred, save_dir,
            measure_latency=measure_latency, tokenizer=tokenizer,
            num_runs=num_runs, warmup=warmup,
        )
        result["quant_time_s"] = q_time
        seed_results[vname] = result

        del q_model

    out_path = models_dir / f"ptq_seed{seed}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(seed_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved -> {out_path}")

    return seed_results
