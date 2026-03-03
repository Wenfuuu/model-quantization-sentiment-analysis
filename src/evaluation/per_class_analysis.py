import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Sequence
from scipy import stats
from scipy.stats import chi2

def per_class_report(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> dict:
    true = list(true_labels)
    pred = list(predicted_labels)
    assert len(true) == len(pred)

    result = {}
    for cls in class_names:
        tp = sum(t == cls and p == cls for t, p in zip(true, pred))
        fp = sum(t != cls and p == cls for t, p in zip(true, pred))
        fn = sum(t == cls and p != cls for t, p in zip(true, pred))
        support = sum(t == cls for t in true)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        result[cls] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
            "tp": tp, "fp": fp, "fn": fn,
        }

    macro_prec = np.mean([result[c]["precision"] for c in class_names])
    macro_rec  = np.mean([result[c]["recall"]    for c in class_names])
    macro_f1   = np.mean([result[c]["f1"]        for c in class_names])

    total = len(true)
    w_prec = sum(result[c]["precision"] * result[c]["support"] for c in class_names) / total
    w_rec  = sum(result[c]["recall"]    * result[c]["support"] for c in class_names) / total
    w_f1   = sum(result[c]["f1"]        * result[c]["support"] for c in class_names) / total

    result["_macro"] = {
        "precision": float(macro_prec),
        "recall": float(macro_rec),
        "f1": float(macro_f1),
    }
    result["_weighted"] = {
        "precision": float(w_prec),
        "recall": float(w_rec),
        "f1": float(w_f1),
    }
    result["_accuracy"] = float(sum(t == p for t, p in zip(true, pred)) / len(true))

    return result


def per_class_delta(
    report_a: dict,
    report_b: dict,
    class_names: Sequence[str],
) -> dict:
    delta = {}
    for cls in list(class_names) + ["_macro", "_weighted"]:
        a = report_a[cls]
        b = report_b[cls]
        delta[cls] = {
            "delta_f1":       round(b["f1"]       - a["f1"],       4),
            "delta_precision": round(b["precision"] - a["precision"], 4),
            "delta_recall":    round(b["recall"]    - a["recall"],    4),
        }
    return delta

def confusion_matrix_arrays(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> np.ndarray:
    n = len(class_names)
    idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, predicted_labels):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    normalize: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_comparison(
    cm_a: np.ndarray,
    cm_b: np.ndarray,
    class_names: Sequence[str],
    name_a: str,
    name_b: str,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, cm, name in zip(axes, [cm_a, cm_b], [name_a, name_b]):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name}")

    plt.suptitle("Per-class Performance: FP32 vs Quantized", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

def mcnemar_test(
    true_labels: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    correction: bool = True,
) -> dict:
    true = list(true_labels)
    pred_a = list(predictions_a)
    pred_b = list(predictions_b)
    assert len(true) == len(pred_a) == len(pred_b)

    n00 = sum(ta == t and tb == t for t, ta, tb in zip(true, pred_a, pred_b))
    n01 = sum(ta == t and tb != t for t, ta, tb in zip(true, pred_a, pred_b))
    n10 = sum(ta != t and tb == t for t, ta, tb in zip(true, pred_a, pred_b))
    n11 = sum(ta != t and tb != t for t, ta, tb in zip(true, pred_a, pred_b))

    n_discordant = n01 + n10

    if n_discordant == 0:
        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "n_discordant": 0,
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "odds_ratio": 1.0,
            "significant": False,
            "interpretation": "No discordant pairs — models make identical errors.",
        }

    if correction:
        chi2_stat = (abs(n01 - n10) - 1.0) ** 2 / n_discordant
    else:
        chi2_stat = (n01 - n10) ** 2 / n_discordant

    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    odds_ratio = (n01 / n10) if n10 > 0 else float("inf")

    interpretation = (
        f"Significant difference (p={p_value:.4f} < 0.05). "
        + ("Model A (FP32) is significantly better." if n01 > n10
           else "Model B (quantized) is significantly better.")
        if p_value < 0.05
        else f"No significant difference (p={p_value:.4f} ≥ 0.05)."
    )

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n_discordant": n_discordant,
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
        "significant": bool(p_value < 0.05),
        "interpretation": interpretation,
        "correction": "Yates" if correction else "none",
    }


def mcnemar_per_class(
    true_labels: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    class_names: Sequence[str],
    correction: bool = True,
) -> dict:
    results = {}
    for cls in class_names:
        mask = [t == cls for t in true_labels]
        t_cls  = [t for t, m in zip(true_labels, mask)  if m]
        pa_cls = [p for p, m in zip(predictions_a, mask) if m]
        pb_cls = [p for p, m in zip(predictions_b, mask) if m]
        results[cls] = mcnemar_test(t_cls, pa_cls, pb_cls, correction=correction)
        results[cls]["class"] = cls
        results[cls]["n_class_samples"] = len(t_cls)
    return results

def plot_per_class_f1_comparison(
    report_fp32: dict,
    report_quantized: dict,
    class_names: Sequence[str],
    name_a: str,
    name_b: str,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(class_names))
    width = 0.35
    f1_a = [report_fp32[c]["f1"] for c in class_names]
    f1_b = [report_quantized[c]["f1"] for c in class_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(x - width / 2, f1_a, width, label=name_a, color="#2196F3", alpha=0.85)
    bars_b = ax.bar(x + width / 2, f1_b, width, label=name_b, color="#F44336", alpha=0.85)

    for xa, fa, fb in zip(x, f1_a, f1_b):
        delta = fb - fa
        color = "#2e7d32" if delta >= 0 else "#c62828"
        ax.text(xa, max(fa, fb) + 0.01, f"Δ{delta:+.3f}", ha="center",
                va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in class_names], fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Per-class F1: {name_a} vs {name_b}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
