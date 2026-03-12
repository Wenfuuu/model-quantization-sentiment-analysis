import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_hf_cache = str(_project_root / ".hf_cache")
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)
os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)
os.environ.setdefault("MPLCONFIGDIR", str(_project_root / ".cache" / "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import json
import warnings
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "datasets"))

from src.config import LABELS, DEVICE
from src.models import ModelManager
from src.models.base import BaseModel
from src.quantization.ptq import PTQQuantizer
from src.xai import LIMEExplainer, SHAPExplainer
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.data import load_smsa_dataset
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import statistical_test
from src.utils import set_seed
from src.evaluation.explanation_drift import (
    aggregate_explanation_drift,
    wilcoxon_drift_test,
    lime_to_attribution,
    shap_to_attribution,
    ig_to_attribution,
)
from src.evaluation.calibration import (
    expected_calibration_error,
    brier_score,
    extract_calibration_inputs,
    plot_reliability_diagram,
    plot_calibration_comparison,
    compare_calibration,
)
from src.evaluation.per_class_analysis import (
    per_class_report,
    per_class_delta,
    confusion_matrix_arrays,
    plot_confusion_matrix_comparison,
    plot_per_class_f1_comparison,
    mcnemar_test,
    mcnemar_per_class,
)
from linguistic_probes import (
    get_probe_samples,
    probe_accuracy_by_phenomenon,
    PROBE_SET,
    PHENOMENON_CATEGORIES,
    save_probes_as_tsv,
    save_probes_as_json,
    probe_set_summary,
)

warnings.filterwarnings("ignore")

LABEL_NAMES = [LABELS[k] for k in sorted(LABELS.keys())]
OUTPUT_ROOT = Path(__file__).parent.parent / "outputs" / "attribution_comparison"
MODEL_PATH  = Path(__file__).parent.parent / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"

XAI_SAMPLE_LIMIT = 50
SEED = 42
set_seed(SEED)

def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path}")


def _sample_for_xai(test_samples: list, n: int, seed: int = SEED) -> list:
    if n is None or n >= len(test_samples):
        return test_samples

    rng = np.random.default_rng(seed)
    by_label: dict = {}
    for s in test_samples:
        by_label.setdefault(s["expected"], []).append(s)

    per_class = n // len(by_label)
    sampled = []
    for label, items in by_label.items():
        idx = rng.choice(len(items), size=min(per_class, len(items)), replace=False)
        sampled.extend([items[i] for i in idx])

    remaining = [s for s in test_samples if s not in sampled]
    rng.shuffle(remaining)
    sampled.extend(remaining[: n - len(sampled)])
    return sampled

def run_evaluation(model: BaseModel, samples: list, tag: str, use_fp16: bool = False) -> dict:
    print(f"\n[EVAL] {tag}")
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(samples, num_runs=20, warmup=5, use_fp16=use_fp16)
    print(f"  Accuracy:     {results['accuracy']*100:.4f}%")
    print(f"  Avg conf:     {results['avg_confidence']*100:.4f}%")
    print(f"  Latency mean: {results['latency_stats']['mean']*1000:.4f} ms")
    return results

def run_calibration(eval_results: dict, tag: str, output_dir: Path) -> dict:
    print(f"\n[CALIB] {tag}")
    preds = eval_results["predictions"]
    confidences, correct, all_probs, true_indices = extract_calibration_inputs(preds, LABELS)

    cal = expected_calibration_error(confidences, correct, n_bins=10)
    bs = brier_score(all_probs, true_indices, n_classes=len(LABELS))
    cal["brier_score"] = bs

    print(f"  ECE:          {cal['ece']:.4f}")
    print(f"  Brier Score:  {bs:.4f}")
    print(f"  Overconf rate: {cal['overconfidence_rate']:.4f}")

    plot_reliability_diagram(cal, tag, output_dir / f"reliability_{tag.replace(' ', '_')}.png")
    return cal

def run_per_class(
    eval_a: dict,
    eval_b: dict,
    tag_a: str,
    tag_b: str,
    output_dir: Path,
) -> tuple:
    print(f"\n[PER-CLASS] {tag_a} vs {tag_b}")
    true_labels = [p["expected"]  for p in eval_a["predictions"]]
    pred_a      = [p["predicted"] for p in eval_a["predictions"]]
    pred_b      = [p["predicted"] for p in eval_b["predictions"]]

    report_a = per_class_report(true_labels, pred_a, LABEL_NAMES)
    report_b = per_class_report(true_labels, pred_b, LABEL_NAMES)
    delta    = per_class_delta(report_a, report_b, LABEL_NAMES)

    print(f"  Class F1 ({tag_a} → {tag_b}):")
    for cls in LABEL_NAMES:
        fa = report_a[cls]["f1"]
        fb = report_b[cls]["f1"]
        print(f"    {cls:10s}: {fa:.4f} → {fb:.4f}  (Δ{fb-fa:+.4f})")

    slug = f"{tag_a}_vs_{tag_b}".replace("-", "_").replace(" ", "_")
    cm_a = confusion_matrix_arrays(true_labels, pred_a, LABEL_NAMES)
    cm_b = confusion_matrix_arrays(true_labels, pred_b, LABEL_NAMES)
    plot_confusion_matrix_comparison(
        cm_a, cm_b, LABEL_NAMES, tag_a, tag_b,
        output_dir / f"confusion_matrix_{slug}.png",
    )
    plot_per_class_f1_comparison(
        report_a, report_b, LABEL_NAMES, tag_a, tag_b,
        output_dir / f"per_class_f1_{slug}.png",
    )

    mcnemar_overall = mcnemar_test(true_labels, pred_a, pred_b)
    print(f"\n  McNemar's test (overall): {mcnemar_overall['interpretation']}")

    mcnemar_cls = mcnemar_per_class(true_labels, pred_a, pred_b, LABEL_NAMES)
    for cls, res in mcnemar_cls.items():
        print(f"    {cls}: p={res['p_value']:.4f}, significant={res['significant']}")

    return report_a, report_b, delta, mcnemar_overall, mcnemar_cls

def _collect_lime(model: BaseModel, samples: list, use_fp16: bool = False) -> list:
    lime = LIMEExplainer(model, LABELS, use_fp16=use_fp16)
    results = []
    for s in tqdm(samples, desc="  LIME"):
        exp = lime.explain(s["text"], num_features=30, num_samples=500)
        pred_idx = int(np.argmax(exp.predict_proba))
        results.append({
            "predicted_label": LABEL_NAMES[pred_idx],
            "top_features": exp.as_list(label=pred_idx),
            "probabilities": {LABEL_NAMES[j]: float(exp.predict_proba[j])
                              for j in range(len(LABEL_NAMES))},
        })
    return results

def _collect_shap(model: BaseModel, samples: list, use_fp16: bool = False) -> list:
    shap_exp = SHAPExplainer(model, LABELS, use_fp16=use_fp16)
    results = []
    for s in tqdm(samples, desc="  SHAP"):
        shap_values = shap_exp.explain(s["text"], max_evals=300)
        pred_cls = int(np.argmax(shap_exp.predict_proba(s["text"])))
        token_imp = {}
        if hasattr(shap_values[0], "data") and hasattr(shap_values[0], "values"):
            for tok, val_vec in zip(shap_values[0].data, shap_values[0].values):
                if isinstance(tok, str) and tok.strip():
                    token_imp[tok] = float(val_vec[pred_cls])
        sorted_imp = sorted(token_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        results.append({
            "predicted_label": LABEL_NAMES[pred_cls],
            "token_importance": sorted_imp,
        })
    return results

def _collect_ig(model: BaseModel, samples: list) -> list:
    ig = IntegratedGradientsExplainer(model.model, model.tokenizer, device=model.device)
    results = []
    for s in tqdm(samples, desc="  IG"):
        try:
            res = ig.explain(s["text"], steps=50)
            results.append(res)
        except Exception as e:
            results.append({"tokens": [], "scores": [], "predicted_class": -1, "error": str(e)})
    return results

def run_attribution_drift(
    lime_fp32: list,
    lime_int8: list,
    shap_fp32: list,
    shap_int8: list,
    ig_fp32: list,
    ig_int8: list,
    eval_fp32: dict,
    eval_int8: dict,
    samples: list,
    output_dir: Path,
    tag_a: str = "FP32",
    tag_b: str = "INT8",
) -> dict:
    print(f"\n[ATTRIBUTION DRIFT] {tag_a} vs {tag_b}")

    pred_fp32 = [r["predicted_label"] for r in lime_fp32]
    pred_int8 = [r["predicted_label"] for r in lime_int8]

    attr_lime_fp32 = [lime_to_attribution(r) for r in lime_fp32]
    attr_lime_int8 = [lime_to_attribution(r) for r in lime_int8]
    attr_shap_fp32 = [shap_to_attribution(r) for r in shap_fp32]
    attr_shap_int8 = [shap_to_attribution(r) for r in shap_int8]
    attr_ig_fp32   = [ig_to_attribution(r)   for r in ig_fp32]
    attr_ig_int8   = [ig_to_attribution(r)   for r in ig_int8]

    results = {}

    for method, attr_a, attr_b in [
        ("lime", attr_lime_fp32, attr_lime_int8),
        ("shap", attr_shap_fp32, attr_shap_int8),
        ("ig",   attr_ig_fp32,   attr_ig_int8),
    ]:
        drift = aggregate_explanation_drift(
            attr_a, attr_b,
            predictions_a=pred_fp32,
            predictions_b=pred_int8,
            k_values=(3, 5, 10),
        )
        wilcoxon = wilcoxon_drift_test(attr_a, attr_b)
        results[method] = {"drift": drift, "wilcoxon": wilcoxon}

        rho = drift["spearman_rho"]["mean"]
        j5 = drift["jaccard_top5"]["mean"]
        flip = drift["sign_flip_rate"]["mean"]
        print(f"  {method.upper():5s}: ρ={rho:.4f}, Jaccard@5={j5:.4f}, "
              f"sign-flip={flip:.4f}  ({wilcoxon['interpretation']})")

    _plot_spearman_distributions(
        attr_lime_fp32, attr_lime_int8,
        attr_shap_fp32, attr_shap_int8,
        attr_ig_fp32, attr_ig_int8,
        output_dir,
        tag_a=tag_a,
        tag_b=tag_b,
    )

    return results

def _plot_spearman_distributions(
    attr_lime_a, attr_lime_b,
    attr_shap_a, attr_shap_b,
    attr_ig_a, attr_ig_b,
    output_dir: Path,
    tag_a: str = "FP32",
    tag_b: str = "INT8",
) -> None:
    from src.evaluation.explanation_drift import spearman_rank_correlation

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"LIME": "#2196F3", "SHAP": "#4CAF50", "IG": "#FF9800"}

    for ax, (name, a_list, b_list) in zip(axes, [
        ("LIME", attr_lime_a, attr_lime_b),
        ("SHAP", attr_shap_a, attr_shap_b),
        ("IG",   attr_ig_a,   attr_ig_b),
    ]):
        rhos = []
        for exp_a, exp_b in zip(a_list, b_list):
            rho, _ = spearman_rank_correlation(
                exp_a["tokens"], exp_a["scores"],
                exp_b["tokens"], exp_b["scores"],
            )
            if not np.isnan(rho):
                rhos.append(rho)

        if rhos:
            ax.hist(rhos, bins=20, color=colors[name], alpha=0.7, edgecolor="white")
            ax.axvline(np.median(rhos), color="black", linestyle="--", linewidth=1.5,
                       label=f"Median={np.median(rhos):.3f}")
            ax.set_xlabel(f"Spearman ρ ({tag_a} vs {tag_b})", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_xlim(-1, 1)
            ax.set_title(f"{name} Attribution Rank Correlation", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    slug = f"{tag_a}_vs_{tag_b}".replace("-", "_").replace(" ", "_")
    plt.suptitle(f"Distribution of Per-sample Explanation Rank Correlation ({tag_a} vs {tag_b})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_dir / f"spearman_distributions_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close()

def run_linguistic_probes(
    model_fp32: BaseModel,
    model_b: BaseModel,
    output_dir: Path,
    tag_b: str = "INT8",
    run_attribution: bool = True,
) -> dict:
    print(f"\n[LINGUISTIC PROBES] FP32 vs {tag_b}")
    probe_samples = get_probe_samples(include_minimal_pairs=False)
    print(f"  Running {len(probe_samples)} probes on FP32 and {tag_b} ...")

    save_probes_as_tsv(output_dir / "probe_set.tsv")
    save_probes_as_json(output_dir / "probe_set.json")

    def _predict_all(model: BaseModel, samples: list) -> list:
        results = []
        for s in samples:
            pred = model.predict(s["text"])
            results.append({
                "text": s["text"],
                "expected": s["expected"],
                "predicted": pred["label"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "phenomenon": s["meta"]["phenomenon"],
                "phenomenon_tokens": s["meta"]["phenomenon_tokens"],
            })
        return results

    preds_fp32 = _predict_all(model_fp32, probe_samples)
    preds_b    = _predict_all(model_b,    probe_samples)

    acc_fp32 = probe_accuracy_by_phenomenon(preds_fp32, PROBE_SET)
    acc_b    = probe_accuracy_by_phenomenon(preds_b,    PROBE_SET)

    print(f"  Per-phenomenon accuracy (FP32 → {tag_b}):")
    for phenom in acc_fp32:
        a32 = acc_fp32[phenom]["accuracy"]
        ab  = acc_b.get(phenom, {}).get("accuracy", float("nan"))
        n   = acc_fp32[phenom]["total"]
        print(f"    {phenom:20s} (n={n}): {a32:.2f} → {ab:.2f}  Δ{ab-a32:+.2f}")

    mp_samples    = get_probe_samples(include_minimal_pairs=True)
    preds_mp_fp32 = _predict_all(model_fp32, mp_samples)
    preds_mp_b    = _predict_all(model_b,    mp_samples)

    probe_attribution_results = {}
    if run_attribution:
        print("  Running LIME attribution on probe sentences ...")
        lime_fp32 = LIMEExplainer(model_fp32, LABELS)
        lime_int8 = LIMEExplainer(model_b,    LABELS)

        for i, (sample, probe) in enumerate(zip(probe_samples, PROBE_SET)):
            exp_fp32 = lime_fp32.explain(sample["text"], num_features=20, num_samples=300)
            exp_int8 = lime_int8.explain(sample["text"], num_features=20, num_samples=300)

            pred_fp32_idx = int(np.argmax(exp_fp32.predict_proba))
            pred_int8_idx = int(np.argmax(exp_int8.predict_proba))

            feat_fp32 = dict(exp_fp32.as_list(label=pred_fp32_idx))
            feat_int8 = dict(exp_int8.as_list(label=pred_int8_idx))

            token_summary = {}
            for tok in probe.phenomenon_tokens:
                token_summary[tok] = {
                    "attribution_fp32": feat_fp32.get(tok, 0.0),
                    "attribution_int8": feat_int8.get(tok, 0.0),
                }

            probe_attribution_results[i] = {
                "text": sample["text"],
                "phenomenon": probe.phenomenon,
                "phenomenon_tokens": probe.phenomenon_tokens,
                "expected": probe.expected_label,
                "predicted_fp32": LABEL_NAMES[pred_fp32_idx],
                f"predicted_{tag_b}": LABEL_NAMES[pred_int8_idx],
                "phenomenon_token_attributions": token_summary,
                "top_features_fp32": exp_fp32.as_list(label=pred_fp32_idx)[:10],
                f"top_features_{tag_b}": exp_int8.as_list(label=pred_int8_idx)[:10],
            }

        _plot_probe_attribution_shift(probe_attribution_results, PROBE_SET, output_dir, tag_b=tag_b)

    results = {
        "probe_set_summary": probe_set_summary(),
        "accuracy_fp32":       {k: v for k, v in acc_fp32.items()},
        f"accuracy_{tag_b}":   {k: v for k, v in acc_b.items()},
        "predictions_fp32":    preds_fp32,
        f"predictions_{tag_b}": preds_b,
        "minimal_pair_fp32":   preds_mp_fp32,
        f"minimal_pair_{tag_b}": preds_mp_b,
        "probe_attribution":   probe_attribution_results,
    }

    return results

def _plot_probe_attribution_shift(
    probe_results: dict,
    probes,
    output_dir: Path,
    tag_b: str = "INT8",
) -> None:
    from collections import defaultdict
    output_dir.mkdir(parents=True, exist_ok=True)

    by_phenom = defaultdict(list)
    for idx, res in probe_results.items():
        for tok, atts in res["phenomenon_token_attributions"].items():
            by_phenom[res["phenomenon"]].append({
                "text_short": res["text"][:40] + "…",
                "token": tok,
                "fp32": atts["attribution_fp32"],
                "int8": atts["attribution_int8"],
            })

    fig, axes = plt.subplots(
        len(by_phenom), 1,
        figsize=(10, 3 * len(by_phenom)),
        constrained_layout=True,
    )
    if len(by_phenom) == 1:
        axes = [axes]

    for ax, (phenom, entries) in zip(axes, by_phenom.items()):
        labels = [f"{e['token']} ({e['text_short']})" for e in entries]
        fp32_vals = [e["fp32"] for e in entries]
        int8_vals = [e["int8"] for e in entries]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, fp32_vals, w, label="FP32", color="#2196F3", alpha=0.8)
        ax.bar(x + w / 2, int8_vals, w, label=tag_b,  color="#F44336", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set_ylabel("LIME attribution")
        ax.set_title(f"Phenomenon: {phenom}", fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle(f"Attribution on Phenomenon Tokens: FP32 vs {tag_b}", fontsize=13,
                 fontweight="bold")
    plt.savefig(str(output_dir / "probe_attribution_shift.png"), dpi=150, bbox_inches="tight")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Attribution comparison: FP32 vs PTQ (INT8/FP16/INT4) vs QAT (INT8)")
    parser.add_argument("--probes-only", action="store_true",
                        help="Only run linguistic probe analysis (skip full eval + XAI)")
    parser.add_argument("--no-ig", action="store_true",
                        help="Skip Integrated Gradients (faster, but loses gradient attribution)")
    parser.add_argument("--xai-limit", type=int, default=XAI_SAMPLE_LIMIT,
                        help=f"Max samples for LIME/SHAP/IG (default: {XAI_SAMPLE_LIMIT})")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH),
                        help="Path to finetuned FP32 model directory")
    parser.add_argument("--qat-int8-path", type=str, default=None,
                        help="Path to QAT-INT8 checkpoint directory")
    parser.add_argument("--qat-fp16-path", type=str, default=None,
                        help="Path to QAT-FP16 checkpoint directory")
    return parser.parse_args()

def _check_model_weights(model_path: str) -> None:
    p = Path(model_path)
    weight_files = (
        list(p.glob("model.safetensors"))
        + list(p.glob("pytorch_model.bin"))
        + list(p.glob("model-*.safetensors"))
    )
    if weight_files:
        return
    sys.exit(1)

def main():
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    _check_model_weights(args.model_path)

    print("=" * 70)
    print("  ATTRIBUTION COMPARISON: FP32 vs PTQ vs QAT")
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {args.model_path}")
    print("=" * 70)

    print("\n[LOAD] FP32")
    base_model_fp32 = ModelManager.load_model(args.model_path)

    ptq = PTQQuantizer(base_model_fp32.model)

    print("[LOAD] PTQ-INT8 (dynamic, CPU)")
    model_int8_raw, _ = ptq.quantize_int8()
    model_ptq_int8 = BaseModel(model_int8_raw, base_model_fp32.tokenizer, device=torch.device("cpu"))

    print("[LOAD] PTQ-FP16")
    model_fp16_raw, _ = ptq.quantize_fp16()
    model_ptq_fp16 = BaseModel(model_fp16_raw, base_model_fp32.tokenizer)

    print("[LOAD] PTQ-INT4")
    model_int4_raw, _ = ptq.quantize_int4()
    model_ptq_int4 = BaseModel(model_int4_raw, base_model_fp32.tokenizer)

    model_qat_int8, model_qat_fp16 = None, None
    if args.qat_int8_path and Path(args.qat_int8_path).exists():
        print(f"[LOAD] QAT-INT8 from {args.qat_int8_path}")
        model_qat_int8 = ModelManager.load_model(args.qat_int8_path)
    else:
        print("[SKIP] QAT-INT8 (pass --qat-int8-path to enable)")

    if args.qat_fp16_path and Path(args.qat_fp16_path).exists():
        print(f"[LOAD] QAT-FP16 from {args.qat_fp16_path}")
        model_qat_fp16 = ModelManager.load_model(args.qat_fp16_path)
    else:
        print("[SKIP] QAT-FP16 (pass --qat-fp16-path to enable)")

    variants = [
        ("PTQ-INT8", model_ptq_int8),
        ("PTQ-FP16", model_ptq_fp16),
        ("PTQ-INT4", model_ptq_int4),
    ]
    if model_qat_int8:
        variants.append(("QAT-INT8", model_qat_int8))
    if model_qat_fp16:
        variants.append(("QAT-FP16", model_qat_fp16))

    print("\n[DATA] Loading SMSA test set")
    test_samples = load_smsa_dataset()
    print(f"  {len(test_samples)} test samples")

    all_results = {}

    if not args.probes_only:
        eval_fp32 = run_evaluation(base_model_fp32, test_samples, "FP32")
        evals = {"FP32": eval_fp32}
        for tag, model in variants:
            evals[tag] = run_evaluation(model, test_samples, tag)

        all_results["evaluation"] = {
            t: {"accuracy": ev["accuracy"],
                "avg_confidence": ev["avg_confidence"],
                "latency_stats": ev["latency_stats"]}
            for t, ev in evals.items()
        }

        all_results["latency_significance"] = {}
        print()
        for tag, ev in evals.items():
            if tag == "FP32":
                continue
            lat_test = statistical_test(eval_fp32["latencies"], ev["latencies"])
            all_results["latency_significance"][tag] = lat_test
            print(f"  Latency FP32 vs {tag}: p={lat_test['p_value']:.6f}, "
                  f"Cohen's d={lat_test['cohens_d']:.4f}")

        calib_dir = OUTPUT_ROOT / "calibration"
        cals = {}
        for tag, ev in evals.items():
            cals[tag] = run_calibration(ev, tag, calib_dir)
        plot_calibration_comparison(cals, calib_dir / "reliability_comparison.png")
        all_results["calibration"] = cals

        perclass_dir = OUTPUT_ROOT / "per_class"
        all_results["per_class"] = {}
        for tag, model in variants:
            r_a, r_b, delta, mn_ov, mn_cls = run_per_class(
                eval_fp32, evals[tag], "FP32", tag, perclass_dir
            )
            all_results["per_class"][tag] = {
                "report_FP32": r_a,
                f"report_{tag}": r_b,
                "delta": delta,
                "mcnemar_overall": mn_ov,
                "mcnemar_per_class": mn_cls,
            }

        xai_samples = _sample_for_xai(test_samples, args.xai_limit)
        print(f"\n[XAI] Running on {len(xai_samples)} samples (stratified)")

        _empty_ig = [{"tokens": [], "scores": [], "predicted_class": -1}] * len(xai_samples)

        lime_fp32 = _collect_lime(base_model_fp32, xai_samples)
        shap_fp32 = _collect_shap(base_model_fp32, xai_samples)
        ig_fp32   = _collect_ig(base_model_fp32, xai_samples) if not args.no_ig else _empty_ig[:]

        xai_raw = {"xai_sample_texts": [s["text"] for s in xai_samples],
                   "lime_FP32": lime_fp32, "shap_FP32": shap_fp32}

        all_results["attribution_drift"] = {}
        drift_dir = OUTPUT_ROOT / "attribution_drift"
        for tag, model in variants:
            lime_q = _collect_lime(model, xai_samples)
            shap_q = _collect_shap(model, xai_samples)
            ig_q   = _collect_ig(model, xai_samples) if not args.no_ig else _empty_ig[:]

            drift_res = run_attribution_drift(
                lime_fp32, lime_q,
                shap_fp32, shap_q,
                ig_fp32,   ig_q,
                eval_fp32, evals[tag],
                xai_samples,
                drift_dir / tag.lower().replace("-", "_"),
                tag_a="FP32", tag_b=tag,
            )
            all_results["attribution_drift"][tag] = drift_res
            xai_raw[f"lime_{tag}"] = lime_q
            xai_raw[f"shap_{tag}"] = shap_q

        _save_json(xai_raw, OUTPUT_ROOT / "xai" / "raw_attributions.json")

    probe_dir = OUTPUT_ROOT / "linguistic_probes"
    all_results["linguistic_probes"] = {}
    for tag, model in variants:
        probe_res = run_linguistic_probes(
            base_model_fp32, model,
            probe_dir / tag.lower().replace("-", "_"),
            tag_b=tag,
            run_attribution=True,
        )
        all_results["linguistic_probes"][tag] = {
            k: v for k, v in probe_res.items() if k != "probe_attribution"
        }
        _save_json(
            probe_res.get("probe_attribution", {}),
            probe_dir / tag.lower().replace("-", "_") / "probe_attribution_detail.json",
        )

    _save_json(all_results, OUTPUT_ROOT / "results.json")

    print("\n" + "=" * 70)
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("=" * 70)

if __name__ == "__main__":
    main()
