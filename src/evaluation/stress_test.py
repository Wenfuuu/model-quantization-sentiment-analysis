import random
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Sequence
from .calibration import expected_calibration_error, brier_score, plot_calibration_comparison


EDGE_CASE_SAMPLES = {
    "negation": [
        {"text": "saya tidak suka sama sekali", "expected": "NEGATIVE", "description": "single negation"},
        {"text": "tidak tidak tidak bagus", "expected": "NEGATIVE", "description": "triple negation"},
        {"text": "bukan berarti saya tidak senang", "expected": "POSITIVE", "description": "double negation (positive)"},
    ],
    "mixed_sentiment": [
        {"text": "makanannya enak tapi pelayanannya sangat buruk", "expected": "NEGATIVE", "description": "positive then negative"},
        {"text": "harganya mahal tapi kualitasnya luar biasa bagus", "expected": "POSITIVE", "description": "negative then positive"},
        {"text": "tempatnya bagus makanannya biasa saja pelayanannya lambat", "expected": "NEGATIVE", "description": "mixed three clauses"},
    ],
    "short_ambiguous": [
        {"text": "oke", "expected": "NEUTRAL", "description": "single word neutral"},
        {"text": "lumayan", "expected": "NEUTRAL", "description": "single word ambiguous"},
        {"text": "ya", "expected": "NEUTRAL", "description": "minimal response"},
    ],
    "long_repetitive": [
        {
            "text": " ".join(["sangat bagus sekali pelayanannya memuaskan dan makanannya enak"] * 8),
            "expected": "POSITIVE",
            "description": "long repetitive positive (128+ tokens)",
        },
        {
            "text": " ".join(["buruk sekali pelayanannya mengecewakan dan tidak memuaskan"] * 8),
            "expected": "NEGATIVE",
            "description": "long repetitive negative (128+ tokens)",
        },
    ],
    "slang_oov": [
        {"text": "gokil banget sih ini mah", "expected": "POSITIVE", "description": "informal slang positive"},
        {"text": "anjir jelek bgt dah males", "expected": "NEGATIVE", "description": "informal slang negative"},
        {"text": "wkwkwk receh bgt asli", "expected": "NEUTRAL", "description": "internet slang"},
    ],
    "neutral_leaning": [
        {"text": "biasa saja tidak ada yang istimewa", "expected": "NEUTRAL", "description": "explicitly neutral"},
        {"text": "saya tidak memiliki pendapat tentang ini", "expected": "NEUTRAL", "description": "no opinion stated"},
    ],
}


def run_edge_case_test(
    models: Dict[str, object],
    output_dir: Path,
    use_fp16_map: Optional[Dict[str, bool]] = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_fp16_map is None:
        use_fp16_map = {}

    results = {}
    precisions = list(models.keys())

    for precision, model in models.items():
        use_fp16 = use_fp16_map.get(precision, False)
        precision_results = {}

        for category, samples in EDGE_CASE_SAMPLES.items():
            cat_results = []
            for sample in samples:
                pred = model.predict(sample["text"], use_fp16=use_fp16)
                cat_results.append({
                    "text": sample["text"],
                    "expected": sample["expected"],
                    "predicted": pred["label"],
                    "confidence": pred["confidence"],
                    "correct": pred["label"] == sample["expected"],
                    "description": sample["description"],
                })
            precision_results[category] = cat_results

        results[precision] = precision_results

    summary = {}
    categories = list(EDGE_CASE_SAMPLES.keys())
    for precision in precisions:
        summary[precision] = {}
        total_correct = 0
        total_samples = 0
        for category in categories:
            cat_results = results[precision][category]
            cat_correct = sum(1 for r in cat_results if r["correct"])
            cat_total = len(cat_results)
            summary[precision][category] = {
                "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
                "correct": cat_correct,
                "total": cat_total,
                "avg_confidence": np.mean([r["confidence"] for r in cat_results]),
            }
            total_correct += cat_correct
            total_samples += cat_total
        summary[precision]["_overall"] = {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "correct": total_correct,
            "total": total_samples,
        }

    json_data = {"results": {}, "summary": summary}
    for p in precisions:
        json_data["results"][p] = {}
        for cat in categories:
            json_data["results"][p][cat] = results[p][cat]

    json_path = output_dir / "edge_case_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    _plot_edge_case_heatmap(summary, categories, precisions, output_dir)

    return {"results": results, "summary": summary}


def _plot_edge_case_heatmap(summary, categories, precisions, output_dir):
    data = np.array([
        [summary[p][cat]["accuracy"] * 100 for cat in categories]
        for p in precisions
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.8), max(4, len(precisions) * 1.2)))

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=10)
    ax.set_yticks(range(len(precisions)))
    ax.set_yticklabels([p.upper() for p in precisions], fontsize=11)

    for i in range(len(precisions)):
        for j in range(len(categories)):
            val = data[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)

    ax.set_title("Stress Test: Linguistic Edge Case Accuracy by Category & Precision",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = output_dir / "edge_case_heatmap.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _apply_char_dropout(text: str, rate: float, rng: random.Random) -> str:
    if rate <= 0:
        return text
    return "".join(c for c in text if rng.random() > rate)


def _apply_token_shuffle(text: str, rate: float, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    result = words[:]
    for i in range(len(result) - 1):
        if rng.random() < rate:
            result[i], result[i + 1] = result[i + 1], result[i]
    return " ".join(result)


def run_noise_robustness_test(
    models: Dict[str, object],
    test_samples: List[dict],
    output_dir: Path,
    noise_levels: Optional[List[float]] = None,
    max_samples: int = 50,
    seed: int = 42,
    use_fp16_map: Optional[Dict[str, bool]] = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_fp16_map is None:
        use_fp16_map = {}
    if noise_levels is None:
        noise_levels = [0.0, 0.10, 0.20, 0.30]

    rng = random.Random(seed)
    samples = test_samples[:max_samples]
    precisions = list(models.keys())

    noise_types = ["char_dropout", "token_shuffle"]
    results = {nt: {} for nt in noise_types}

    for noise_type in noise_types:
        for precision, model in models.items():
            use_fp16 = use_fp16_map.get(precision, False)
            level_results = {}

            for level in noise_levels:
                correct = 0
                confidences = []

                for sample in samples:
                    if noise_type == "char_dropout":
                        noisy_text = _apply_char_dropout(sample["text"], level, rng)
                    else:
                        noisy_text = _apply_token_shuffle(sample["text"], level, rng)

                    if not noisy_text.strip():
                        noisy_text = sample["text"]

                    pred = model.predict(noisy_text, use_fp16=use_fp16)
                    if pred["label"] == sample["expected"]:
                        correct += 1
                    confidences.append(pred["confidence"])

                level_results[level] = {
                    "accuracy": correct / len(samples),
                    "avg_confidence": float(np.mean(confidences)),
                    "num_samples": len(samples),
                }

            results[noise_type][precision] = level_results

    json_path = output_dir / "noise_robustness_results.json"
    serializable = {}
    for nt in noise_types:
        serializable[nt] = {}
        for p in precisions:
            serializable[nt][p] = {str(k): v for k, v in results[nt][p].items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    _plot_noise_robustness(results, noise_types, noise_levels, precisions, output_dir)

    return results


def _plot_noise_robustness(results, noise_types, noise_levels, precisions, output_dir):
    colors = {"fp32": "#2196F3", "fp16": "#FF9800", "int8": "#F44336", "int4": "#9C27B0"}
    markers = {"fp32": "o", "fp16": "s", "int8": "^", "int4": "D"}

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5))
    if len(noise_types) == 1:
        axes = [axes]

    for ax, noise_type in zip(axes, noise_types):
        for precision in precisions:
            accs = [results[noise_type][precision][level]["accuracy"] * 100
                    for level in noise_levels]
            color = colors.get(precision, "#000000")
            marker = markers.get(precision, "o")
            ax.plot(
                [int(l * 100) for l in noise_levels], accs,
                color=color, marker=marker, linewidth=2, markersize=7,
                label=precision.upper(),
            )

        ax.set_xlabel("Noise Level (%)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"Noise Robustness: {noise_type.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle("Stress Test: Accuracy Under Input Noise", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "noise_robustness_curves.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_calibration_stress_test(
    all_predictions: Dict[str, list],
    output_dir: Path,
    n_bins: int = 10,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precisions = list(all_predictions.keys())
    results = {}

    for precision in precisions:
        preds = all_predictions[precision]

        confidences = [p["confidence"] for p in preds]
        correct = [int(p["predicted"] == p["expected"]) for p in preds]

        cal = expected_calibration_error(confidences, correct, n_bins=n_bins)

        label_order = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        has_probs = all(
            "probabilities" in p and all(l in p["probabilities"] for l in label_order)
            for p in preds
        )
        if has_probs:
            probs_per_class = [
                [p["probabilities"][l] for l in label_order]
                for p in preds
            ]
            label_to_id = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
            true_labels = [label_to_id.get(p["expected"], 0) for p in preds]
            bs = brier_score(probs_per_class, true_labels, n_classes=3)
            cal["brier_score"] = bs

        results[precision] = cal

    json_data = {}
    for p in precisions:
        json_data[p] = {
            "ece": results[p]["ece"],
            "mce": results[p]["mce"],
            "avg_confidence": results[p]["avg_confidence"],
            "avg_accuracy": results[p]["avg_accuracy"],
            "overconfidence_rate": results[p]["overconfidence_rate"],
            "confidence_accuracy_gap": results[p]["confidence_accuracy_gap"],
            "brier_score": results[p].get("brier_score"),
        }

    json_path = output_dir / "calibration_stress_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    plot_labels = {p: p.upper() for p in precisions}
    results_for_plot = {plot_labels[p]: results[p] for p in precisions}
    plot_calibration_comparison(results_for_plot, output_dir / "calibration_comparison.png")
    print(f"  Saved: {output_dir / 'calibration_comparison.png'}")

    _plot_calibration_summary(json_data, precisions, output_dir)

    return results


def _plot_calibration_summary(json_data, precisions, output_dir):
    colors = {"fp32": "#2196F3", "fp16": "#FF9800", "int8": "#F44336", "int4": "#9C27B0"}

    metrics = ["ece", "mce", "overconfidence_rate"]
    metric_labels = ["ECE", "MCE", "Overconfidence\nRate"]
    if json_data[precisions[0]].get("brier_score") is not None:
        metrics.append("brier_score")
        metric_labels.append("Brier\nScore")

    n_metrics = len(metrics)
    n_prec = len(precisions)
    x = np.arange(n_metrics)
    width = 0.8 / n_prec

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 3), 5))

    for i, precision in enumerate(precisions):
        values = [json_data[precision].get(m, 0) or 0 for m in metrics]
        offset = (i - n_prec / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            color=colors.get(precision, "#999"), alpha=0.85,
            label=precision.upper(), edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Stress Test: Calibration Metrics Across Precisions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "calibration_summary.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
