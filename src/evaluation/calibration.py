import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from scipy import stats

def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)

    assert len(conf) == len(corr), "Length mismatch"
    assert np.all((conf >= 0) & (conf <= 1)), "Confidences must be in [0, 1]"

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(conf, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            bin_accs.append(float("nan"))
            bin_confs.append(float("nan"))
            bin_sizes.append(0)
            continue
        bin_accs.append(float(corr[mask].mean()))
        bin_confs.append(float(conf[mask].mean()))
        bin_sizes.append(int(mask.sum()))

    n = len(conf)
    ece = sum(
        size / n * abs(acc - conf_val)
        for acc, conf_val, size in zip(bin_accs, bin_confs, bin_sizes)
        if not np.isnan(acc)
    )

    mce = max(
        (abs(acc - conf_val) for acc, conf_val in zip(bin_accs, bin_confs) if not np.isnan(acc)),
        default=float("nan"),
    )

    overconf_rate = float(np.mean((conf > 0.9) & (corr == 0)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "avg_confidence": float(conf.mean()),
        "avg_accuracy": float(corr.mean()),
        "overconfidence_rate": overconf_rate,
        "confidence_accuracy_gap": float(conf.mean() - corr.mean()),
        "n_bins": n_bins,
        "strategy": strategy,
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_sizes": bin_sizes,
    }


def brier_score(
    confidences_per_class: List[List[float]],
    true_labels: List[int],
    n_classes: int,
) -> float:
    probs = np.array(confidences_per_class, dtype=float)
    n = len(true_labels)
    one_hot = np.zeros((n, n_classes), dtype=float)
    for i, y in enumerate(true_labels):
        one_hot[i, y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def extract_calibration_inputs(predictions: List[dict], label_map: dict) -> Tuple[list, list, list, list]:
    inv_map = {v: k for k, v in label_map.items()}
    confidences, correct, all_probs, true_indices = [], [], [], []

    for pred in predictions:
        confidences.append(pred["confidence"] / 100.0 if pred["confidence"] > 1.0 else pred["confidence"])
        correct.append(int(pred["predicted"] == pred["expected"]))
        probs_dict = pred.get("probabilities", {})
        probs_vec = [probs_dict.get(label_map[k], 0.0) / 100.0
                     if probs_dict.get(label_map[k], 0.0) > 1.0
                     else probs_dict.get(label_map[k], 0.0)
                     for k in sorted(label_map.keys())]
        all_probs.append(probs_vec)
        true_indices.append(inv_map.get(pred["expected"], -1))

    return confidences, correct, all_probs, true_indices

def plot_reliability_diagram(
    calibration_results: dict,
    model_name: str,
    output_path: Path,
    color: str = "#4444ff",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_centers = [
        (lo + hi) / 2
        for lo, hi in zip(
            calibration_results["bin_edges"][:-1],
            calibration_results["bin_edges"][1:],
        )
    ]
    accs = calibration_results["bin_accuracies"]
    confs = calibration_results["bin_confidences"]
    sizes = calibration_results["bin_sizes"]

    valid = [(bc, a, c, s) for bc, a, c, s in zip(bin_centers, accs, confs, sizes)
             if not np.isnan(a) and s > 0]
    if not valid:
        return

    bcs, avs, cvs, szs = zip(*valid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
    ax.bar(bcs, avs, width=0.08, color=color, alpha=0.6, label="Accuracy", align="center")
    ax.bar(bcs, cvs, width=0.08, color="none", edgecolor=color, linewidth=1.5,
           label="Avg confidence", align="center")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability Diagram — {model_name}")
    ax.legend(fontsize=9)
    ece = calibration_results["ece"]
    mce = calibration_results["mce"]
    ax.text(0.05, 0.92, f"ECE = {ece:.4f}\nMCE = {mce:.4f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    ax2 = axes[1]
    ax2.bar(bcs, szs, width=0.08, color=color, alpha=0.7, align="center")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Sample count")
    ax2.set_xlim(0, 1)
    ax2.set_title(f"Confidence Distribution — {model_name}")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_comparison(
    results_dict: dict,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"FP32": "#2196F3", "FP32_ONNX": "#4CAF50", "INT8": "#F44336", "FP16": "#FF9800"}
    default_cmap = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration", zorder=0)

    for i, (name, cal_res) in enumerate(results_dict.items()):
        color = colors.get(name, default_cmap[i % len(default_cmap)])
        bin_edges = cal_res["bin_edges"]
        bin_centers = [(lo + hi) / 2 for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
        accs = cal_res["bin_accuracies"]
        szs = cal_res["bin_sizes"]

        valid_bc = [bc for bc, a, s in zip(bin_centers, accs, szs) if not np.isnan(a) and s > 0]
        valid_ac = [a for a, s in zip(accs, szs) if not np.isnan(a) and s > 0]

        ece = cal_res["ece"]
        ax.plot(valid_bc, valid_ac, "o-", color=color, linewidth=2.0,
                label=f"{name} (ECE={ece:.4f})", markersize=5)

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram: Calibration Across Precision Levels", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

def compare_calibration(
    cal_fp32: dict,
    cal_quantized: dict,
    alpha: float = 0.05,
) -> dict:
    delta_ece = cal_quantized["ece"] - cal_fp32["ece"]
    delta_conf_acc_gap = (
        cal_quantized["confidence_accuracy_gap"] - cal_fp32["confidence_accuracy_gap"]
    )
    delta_overconf = cal_quantized["overconfidence_rate"] - cal_fp32["overconfidence_rate"]

    return {
        "ece_fp32": cal_fp32["ece"],
        "ece_quantized": cal_quantized["ece"],
        "delta_ece": delta_ece,
        "direction": "worse" if delta_ece > 0 else "better",
        "avg_confidence_fp32": cal_fp32["avg_confidence"],
        "avg_confidence_quantized": cal_quantized["avg_confidence"],
        "confidence_accuracy_gap_fp32": cal_fp32["confidence_accuracy_gap"],
        "confidence_accuracy_gap_quantized": cal_quantized["confidence_accuracy_gap"],
        "delta_confidence_accuracy_gap": delta_conf_acc_gap,
        "overconfidence_rate_fp32": cal_fp32["overconfidence_rate"],
        "overconfidence_rate_quantized": cal_quantized["overconfidence_rate"],
        "delta_overconfidence_rate": delta_overconf,
        "brier_score_fp32": cal_fp32.get("brier_score"),
        "brier_score_quantized": cal_quantized.get("brier_score"),
    }

def run_ste_calibration():
    import pandas as pd
    from scipy.stats import spearmanr

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR      = _PROJECT_ROOT / "results" / "attributions"
    _RESULT_CSV   = _PROJECT_ROOT / "results" / "ste_calibration.csv"

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    sample_ids = df_sub["sample_id"].tolist()[:20]
    print(f"\n  STE Calibration: first 20 sample IDs from {_SUBSAMPLE_CSV.name}")

    COMPARE_VARIANTS = ["qat_onnx_int8", "qat_onnx_int4"]
    rows = []
    missing_ig = 0

    for vname in COMPARE_VARIANTS:
        for sid in sample_ids:
            ig_path  = _OUT_DIR / f"ig_qat_ste_{sid}.npy"
            occ_path = _OUT_DIR / f"occ_{vname}_{sid}.npy"

            if not ig_path.exists():
                missing_ig += 1
                continue
            if not occ_path.exists():
                print(f"  [SKIP] occ_{vname}_{sid}.npy not found")
                continue

            ig_scores  = np.load(ig_path).astype(np.float64)
            occ_scores = np.load(occ_path).astype(np.float64)
            L = min(len(ig_scores), len(occ_scores))
            rho, _ = spearmanr(ig_scores[:L], occ_scores[:L])
            rows.append({"sample_id": sid, "variant": vname, "spearman_rho": float(rho)})

    if missing_ig > 0:
        print(f"  [WARN] {missing_ig} ig_qat_ste_*.npy files missing - run STE-IG first (option [3])")
    if not rows:
        print("  [WARN] No data to compute. Aborting.")
        return

    df = pd.DataFrame(rows)
    _RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_RESULT_CSV, index=False, encoding="utf-8")
    print(f"  Saved {len(rows)} rows -> {_RESULT_CSV}")

    print()
    for vname in COMPARE_VARIANTS:
        sub = df[df["variant"] == vname]["spearman_rho"]
        if sub.empty:
            print(f"  {vname}: no data")
            continue
        mean_rho = sub.mean()
        if mean_rho >= 0.6:
            verdict = "VALID"
        elif mean_rho >= 0.4:
            verdict = "MARGINAL"
        else:
            verdict = "UNRELIABLE"
        print(f"  {vname:20s}: mean_rho={mean_rho:.3f}  n={len(sub)}  -> {verdict}")