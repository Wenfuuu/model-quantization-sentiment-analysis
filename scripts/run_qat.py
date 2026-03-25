import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

if "--generate-ece" in sys.argv:
    import pandas as _pd
    _BASE = Path(__file__).parent.parent

    def _ece_arrays(c, r, n_bins=10):
        c, r = np.asarray(c, float), np.asarray(r, float)
        edges, ece, n = np.linspace(0.0, 1.0, n_bins + 1), 0.0, len(c)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (c > lo) & (c <= hi)
            if m.sum():
                ece += m.sum() / n * abs(r[m].mean() - c[m].mean())
        return float(ece)

    def _ece_csv(p):
        if not p.exists(): return None
        df = _pd.read_csv(p)
        return _ece_arrays(df[["prob_pos","prob_neu","prob_neg"]].max(1).tolist(),
                           (df["pred_label"]==df["true_label"]).astype(int).tolist())

    _SEEDS = [42, 123, 456]
    _MODELS = _BASE / "models"
    _VARIANTS = [
        ("FP32",          lambda s: _MODELS / f"fp32_seed{s}"           / "predictions.csv"),
        ("PTQ-FP16",      lambda s: _MODELS / f"ptq_fp16_seed{s}"       / "predictions.csv"),
        ("PTQ-INT8",      lambda s: _MODELS / f"ptq_int8_seed{s}"       / "predictions.csv"),
        ("PTQ-INT4",      lambda s: _MODELS / f"ptq_int4_seed{s}"       / "predictions.csv"),
        ("QAT-FP32",      lambda s: _MODELS / f"qat_seed{s}_clean"      / "predictions.csv"),
        ("QAT-ONNX-FP16", lambda s: _MODELS / f"qat_onnx_fp16_seed{s}"  / "predictions.csv"),
        ("QAT-ONNX-INT8", lambda s: _MODELS / f"qat_onnx_int8_seed{s}"  / "predictions.csv"),
        ("QAT-ONNX-INT4", lambda s: _MODELS / f"qat_onnx_int4_seed{s}"  / "predictions.csv"),
    ]

    rows = []
    for variant, path_fn in _VARIANTS:
        for seed in _SEEDS:
            ece = _ece_csv(path_fn(seed))
            if ece is not None:
                rows.append({"variant": variant, "seed": seed, "ece": ece})
                print(f"  {variant:<18}  seed={seed}  ECE={ece:.6f}")

    if rows:
        _out = _BASE / "results"
        _out.mkdir(parents=True, exist_ok=True)

        _df = _pd.DataFrame(rows)
        _df.to_csv(_out / "ece_all.csv", index=False)
        print(f"\n  Saved -> {_out / 'ece_all.csv'}")

        _summ = (_df.groupby("variant")["ece"]
                    .agg(mean_ece="mean", std_ece="std")
                    .reset_index()
                    .sort_values("mean_ece"))
        _summ.to_csv(_out / "ece_summary.csv", index=False)
        print(f"  Saved -> {_out / 'ece_summary.csv'}")

        print("\n" + "=" * 58)
        print(f"  {'Variant':<18}  {'Mean ECE':>10}  {'Std ECE':>10}")
        print("  " + "-" * 54)
        for _, r in _summ.iterrows():
            s = f"{r['std_ece']:.6f}" if not np.isnan(r["std_ece"]) else "    N/A"
            print(f"  {r['variant']:<18}  {r['mean_ece']:>10.6f}  {s:>10}")
        print("=" * 58)
    sys.exit(0)

from src.config import BASE_DIR, DATASET_PATHS, TRAINING_SEEDS
from src.quantization.qat.config import FinetuneQATConfig
from src.quantization.qat.eager import EagerQATTrainer
from src.quantization.qat.trainer import train_qat_seed
from src.quantization.qat.eager import qat_onnx_single_seed
from src.visualization import QuantizationPlotter
from scripts.run_xai import generate_qat_divergences

_DATA_DIR   = BASE_DIR / "data" / "processed"
_MODELS_DIR = BASE_DIR / "models"


def get_default_config(method, quant_type, sample_frac=1.0):
    output_dir = BASE_DIR / "outputs"

    save_name = f"indobert-qat-{quant_type}-smsa"
    results_name = f"indobert-qat-{quant_type}-smsa"

    return FinetuneQATConfig(
        model_id="indobenchmark/indobert-base-p2",
        train_file=DATASET_PATHS["smsa_train"],
        valid_file=DATASET_PATHS["smsa_valid"],
        test_file=DATASET_PATHS["smsa"],
        save_dir=output_dir / save_name,
        results_dir=output_dir / results_name,
        sample_frac=sample_frac,
    )

def run_eager_qat(quant_type, dataset_path=None, sample_frac=1.0, evaluate_only=False, num_runs=20):
    config = get_default_config("eager", quant_type, sample_frac=sample_frac)
    trainer = EagerQATTrainer(config, quantization_type=quant_type)

    if evaluate_only:
        print(f"\n[Evaluate Only] Evaluating {quant_type.upper()} ONNX model...")
        results = trainer.evaluate_onnx(dataset_path=dataset_path, num_runs=num_runs)
        return results

    if quant_type == "fp32":
        print(f"\n[Step 1/3] Training base model (FP32 eager)...")
        trainer.train()

        print(f"\n[Step 2/3] Exporting to ONNX...")
        trainer.export_to_onnx()

        print(f"\n[Step 3/3] Evaluating FP32 ONNX model...")
        results = trainer.evaluate_onnx(dataset_path=dataset_path, num_runs=num_runs)
        return results

    print(f"\n[Step 1/4] Training with QAT ({quant_type.upper()} eager)...")
    trainer.train()

    print(f"\n[Step 2/4] Exporting to ONNX...")
    trainer.export_to_onnx()

    print(f"\n[Step 3/4] Quantizing ONNX to {quant_type.upper()}...")
    onnx_path = trainer.quantize_onnx()

    print(f"\n[Step 4/4] Evaluating {quant_type.upper()} ONNX model...")
    results = trainer.evaluate_onnx(onnx_path, dataset_path=dataset_path, num_runs=num_runs)
    return results

def run_multiseed_qat(
    seeds: list = None,
    *,
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 16,
) -> None:
    if seeds is None:
        seeds = list(TRAINING_SEEDS)

    train_csv = _DATA_DIR / "smsa_train_v2.csv"
    val_csv   = _DATA_DIR / "smsa_val_v2.csv"
    test_csv  = _DATA_DIR / "smsa_test_v2.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found: {p}\n"
                "Run scripts/prepare_datasets.py first."
            )

    if len(seeds) < 3:
        warnings.warn(
            f"Only {len(seeds)} seed(s) requested. Publication-grade variance "
            "estimation requires at least 3 independent training runs.",
            UserWarning, stacklevel=2,
        )

    print("\n" + "=" * 70)
    print("  MULTI-SEED QAT: FP32 → QAT-FP32 (fake quantisation)")
    print(f"  Seeds: {seeds}  |  epochs={epochs}  lr={lr}  batch={batch_size}")
    print("=" * 70 + "\n")

    all_results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*70}")
        print(f"#  SEED {i}/{len(seeds)}: {seed}")
        print(f"{'#'*70}")

        fp32_ckpt = _MODELS_DIR / f"fp32_seed{seed}"
        result = train_qat_seed(
            seed,
            fp32_ckpt=fp32_ckpt,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            models_dir=_MODELS_DIR,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        all_results.append(result)

    print("\n" + "=" * 70)
    print("  MULTI-SEED QAT — AGGREGATED RESULTS")
    print("=" * 70)

    accs  = [r["test_accuracy"]  for r in all_results]
    mf1s  = [r["test_macro_f1"]  for r in all_results]
    agrees = [r["smsa_agreement_with_fp32"] for r in all_results
              if r["smsa_agreement_with_fp32"] is not None]

    fp32_accs = {}
    for r in all_results:
        fp32_metrics = Path(r["source_fp32_checkpoint"]) / "metrics.json"
        if fp32_metrics.exists():
            with open(fp32_metrics) as f:
                fp32_accs[r["seed"]] = json.load(f).get("accuracy")

    print(f"\n  {'Seed':>6}  {'FP32 Acc':>10}  {'QAT Acc':>10}  {'Δ':>8}  {'Agreement':>10}")
    print("  " + "-" * 56)

    for r in all_results:
        s = r["seed"]
        qat_acc = r["test_accuracy"]
        fp32_acc = fp32_accs.get(s)
        agr = r["smsa_agreement_with_fp32"]

        fp32_str  = f"{fp32_acc:.4f}" if fp32_acc is not None else "  n/a"
        delta_str = f"{(qat_acc - fp32_acc):+.4f}" if fp32_acc is not None else "  n/a"
        agr_str   = f"{agr*100:.2f}%" if agr is not None else "  n/a"

        print(f"  {s:>6d}  {fp32_str:>10}  {qat_acc:>10.4f}  {delta_str:>8}  {agr_str:>10}")

    print(f"\n  QAT accuracy:   {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
    print(f"  QAT macro-F1:   {np.mean(mf1s):.4f} ± {np.std(mf1s, ddof=1):.4f}")
    if agrees:
        print(f"  Agreement rate: {np.mean(agrees)*100:.2f}% ± {np.std(agrees, ddof=1)*100:.2f}%")

    print()
    for r in all_results:
        s = r["seed"]
        qat_acc = r["test_accuracy"]
        fp32_acc = fp32_accs.get(s)
        agr = r["smsa_agreement_with_fp32"]

        if fp32_acc is not None and (qat_acc - fp32_acc) > 0.01:
            print(f"  [!] Seed {s}: QAT accuracy >1pp HIGHER than FP32 — lr may be too high")
        if fp32_acc is not None and (fp32_acc - qat_acc) > 0.01:
            print(f"  [!] Seed {s}: QAT accuracy >1pp LOWER than FP32 — quantization noise may be too large")
        if agr is not None and agr < 0.97:
            print(f"  [!] Seed {s}: Agreement {agr*100:.2f}% < 97% — check training stability")

    print("\n" + "=" * 70)
    print("  Multi-seed QAT complete.")
    print("=" * 70 + "\n")

    agg_dir = BASE_DIR / "outputs" / "multi-seed"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_path = agg_dir / "aggregated_qat_results.json"
    agg = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "qat_accuracy_mean": float(np.mean(accs)),
        "qat_accuracy_std":  float(np.std(accs, ddof=1)),
        "qat_macro_f1_mean": float(np.mean(mf1s)),
        "qat_macro_f1_std":  float(np.std(mf1s, ddof=1)),
        "per_seed": all_results,
    }
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"  Aggregated results saved -> {agg_path}")

def run_multiseed_qat_onnx(seeds: list = None) -> None:
    """Export clean QAT checkpoints to ONNX, quantise (INT8/FP16/INT4),
    evaluate with ORT, and generate the combined summary CSV."""

    if seeds is None:
        seeds = list(TRAINING_SEEDS)

    test_csv = _DATA_DIR / "smsa_test_v2.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed test set not found: {test_csv}\n"
            "Run scripts/prepare_datasets.py first."
        )

    print("\n" + "=" * 70)
    print("  MULTI-SEED QAT → ONNX EXPORT + QUANTIZATION")
    print(f"  Seeds: {seeds}")
    print("=" * 70 + "\n")

    all_results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#' * 70}")
        print(f"#  SEED {i}/{len(seeds)}: {seed}")
        print(f"{'#' * 70}")

        qat_clean = _MODELS_DIR / f"qat_seed{seed}_clean"
        fp32_ckpt = _MODELS_DIR / f"fp32_seed{seed}"
        result = qat_onnx_single_seed(
            seed,
            qat_clean_dir=qat_clean,
            fp32_ckpt=fp32_ckpt,
            test_csv=test_csv,
            models_dir=_MODELS_DIR,
        )
        all_results.append(result)

    onnx_variants = ["int8", "fp16", "int4"]

    print("\n" + "=" * 70)
    print("  QAT-ONNX — AGGREGATED RESULTS")
    print("=" * 70)

    print(f"\n  {'Variant':<12} {'Acc (mean±std)':>22} "
          f"{'Macro-F1 (mean±std)':>22} {'Agreement':>12}")
    print("  " + "-" * 72)

    for v in onnx_variants:
        accs = [r[v]["metrics"]["accuracy"] for r in all_results if v in r]
        mf1s = [r[v]["metrics"]["macro_f1"] for r in all_results if v in r]
        agrs = [r[v]["agreement_with_fp32"] for r in all_results
                if v in r and r[v]["agreement_with_fp32"] is not None]

        acc_str = f"{np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}"
        mf1_str = f"{np.mean(mf1s):.4f} ± {np.std(mf1s, ddof=1):.4f}"
        agr_str = f"{np.mean(agrs) * 100:.2f}%" if agrs else "n/a"

        print(f"  QAT-{v.upper():<7} {acc_str:>22} {mf1_str:>22} {agr_str:>12}")

    print("\n" + "=" * 70)
    print("  QAT-ONNX export complete.")
    print("=" * 70 + "\n")

    _generate_combined_csv(seeds)


def _ece_from_arrays(confidences, correct, n_bins=10):
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)


def _ece_from_csv(pred_csv: Path) -> "float | None":
    if not pred_csv.exists():
        return None
    df = pd.read_csv(pred_csv)
    confs = df[["prob_pos", "prob_neu", "prob_neg"]].max(axis=1).tolist()
    corr  = (df["pred_label"] == df["true_label"]).astype(int).tolist()
    return _ece_from_arrays(confs, corr)


def _generate_combined_csv(seeds: list) -> None:
    import pandas as pd

    rows = []

    for seed in seeds:
        fp32_pred_path = _MODELS_DIR / f"fp32_seed{seed}" / "predictions.csv"
        fp32_pred = None
        if fp32_pred_path.exists():
            fp32_pred = pd.read_csv(fp32_pred_path)["pred_label"].astype(int).tolist()

        fp32_metrics = _MODELS_DIR / f"fp32_seed{seed}" / "metrics.json"
        if fp32_metrics.exists():
            m = json.loads(fp32_metrics.read_text())
            rows.append({
                "variant": "FP32",
                "seed": seed,
                "smsa_acc": m.get("accuracy"),
                "smsa_f1": m.get("macro_f1") or m.get("weighted_f1"),
                "agreement_rate": 1.0,
                "ece": _ece_from_csv(_MODELS_DIR / f"fp32_seed{seed}" / "predictions.csv"),
            })

        for ptq_v in ["fp16", "int8", "int4"]:
            d = _MODELS_DIR / f"ptq_{ptq_v}_seed{seed}"
            mp = d / "metrics.json"
            pp = d / "predictions.csv"
            if mp.exists():
                m = json.loads(mp.read_text())
                agr = _agreement(fp32_pred, pp)
                rows.append({
                    "variant": f"PTQ-{ptq_v.upper()}",
                    "seed": seed,
                    "smsa_acc": m.get("accuracy"),
                    "smsa_f1": m.get("macro_f1") or m.get("weighted_f1"),
                    "agreement_rate": agr,
                    "ece": _ece_from_csv(pp),
                })

        qat_clean = _MODELS_DIR / f"qat_seed{seed}_clean"
        mp = qat_clean / "metrics.json"
        pp = qat_clean / "predictions.csv"
        if mp.exists():
            m = json.loads(mp.read_text())
            agr = _agreement(fp32_pred, pp)
            rows.append({
                "variant": "QAT-FP32",
                "seed": seed,
                "smsa_acc": m.get("accuracy"),
                "smsa_f1": m.get("macro_f1") or m.get("weighted_f1"),
                "agreement_rate": agr,
                "ece": _ece_from_csv(pp),
            })

        for ov in ["fp16", "int8", "int4"]:
            d = _MODELS_DIR / f"qat_onnx_{ov}_seed{seed}"
            mp = d / "metrics.json"
            pp = d / "predictions.csv"
            if mp.exists():
                m = json.loads(mp.read_text())
                agr = _agreement(fp32_pred, pp)
                rows.append({
                    "variant": f"QAT-ONNX-{ov.upper()}",
                    "seed": seed,
                    "smsa_acc": m.get("accuracy"),
                    "smsa_f1": m.get("macro_f1") or m.get("weighted_f1"),
                    "agreement_rate": agr,
                    "ece": _ece_from_csv(pp),
                })

    if not rows:
        print("  [warn] No results found for combined CSV.")
        return

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("  COMBINED 8-VARIANT SUMMARY (mean ± std across seeds)")
    print("=" * 70)

    print(f"\n  {'Variant':<16} {'Accuracy':>22} {'Macro-F1':>22} {'Agreement':>12}")
    print("  " + "-" * 76)

    for variant in df["variant"].unique():
        vdf = df[df["variant"] == variant]
        acc_m, acc_s = vdf["smsa_acc"].mean(), vdf["smsa_acc"].std(ddof=1)
        f1_m, f1_s = vdf["smsa_f1"].mean(), vdf["smsa_f1"].std(ddof=1)
        agr_vals = vdf["agreement_rate"].dropna()
        agr_str = (f"{agr_vals.mean() * 100:.2f}%" if len(agr_vals) > 0
                   else "n/a")

        print(f"  {variant:<16} {acc_m:.4f} ± {acc_s:.4f}    "
              f"{f1_m:.4f} ± {f1_s:.4f}    {agr_str:>12}")

    out_dir = BASE_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "classification_summary_multiseed.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Combined CSV saved -> {csv_path}")

    ece_rows = df[["variant", "seed", "ece"]].dropna(subset=["ece"]).copy()
    if not ece_rows.empty:
        ece_all_path = out_dir / "ece_all.csv"
        ece_rows.to_csv(ece_all_path, index=False, encoding="utf-8")
        print(f"  ECE (all seeds)   -> {ece_all_path}")

        ece_summary = (
            ece_rows.groupby("variant")["ece"]
            .agg(mean_ece="mean", std_ece="std")
            .reset_index()
            .sort_values("mean_ece")
            .reset_index(drop=True)
        )
        ece_summary_path = out_dir / "ece_summary.csv"
        ece_summary.to_csv(ece_summary_path, index=False, encoding="utf-8")
        print(f"  ECE (summary)     -> {ece_summary_path}")

        print("\n" + "=" * 58)
        print(f"  {'Variant':<18}  {'Mean ECE':>10}  {'Std ECE':>10}")
        print("  " + "-" * 54)
        for _, r in ece_summary.iterrows():
            std_str = f"{r['std_ece']:.6f}" if not np.isnan(r["std_ece"]) else "    N/A"
            print(f"  {r['variant']:<18}  {r['mean_ece']:>10.6f}  {std_str:>10}")
        print("=" * 58)


def _agreement(fp32_pred: list | None, pred_csv: Path) -> float | None:
    if fp32_pred is None or not pred_csv.exists():
        return None
    import pandas as pd
    other = pd.read_csv(pred_csv)["pred_label"].astype(int).tolist()
    n = min(len(fp32_pred), len(other))
    return sum(a == b for a, b in zip(fp32_pred[:n], other[:n])) / max(1, n)


def interactive_menu():
    print("\n" + "=" * 60)
    print("  QAT (QUANTIZATION-AWARE TRAINING) EXPERIMENT RUNNER")
    print("=" * 60)

    print("\n  Select Pipeline:")
    print("  [1] ONNX QAT (Eager — original pipeline)")
    print("  [2] Multi-Seed QAT (FP32 → QAT-FP32, seeds 42/123/456)")
    print("  [3] Multi-Seed QAT-ONNX (QAT → ONNX export + INT8/FP16/INT4)")
    print("  [4] Generate ECE summary from existing prediction CSVs")

    pipeline_choice = input("\n  Enter choice (1/2/3/4): ").strip()

    if pipeline_choice == "2":
        return "multiseed", None, None, None, None, None
    if pipeline_choice == "3":
        return "multiseed_onnx", None, None, None, None, None
    if pipeline_choice == "4":
        return "generate_ece", None, None, None, None, None

    methods = ["eager"]

    print("\n  Select Mode:")
    print("  [1] Train + Evaluate")
    print("  [2] Evaluate Only (skip training, use existing model)")

    mode_choice = input("\n  Enter choice (1/2): ").strip()
    evaluate_only = mode_choice == "2"

    print("\n  Select Quantization Type:")
    print("  [1] FP16")
    print("  [2] INT8")
    print("  [3] INT4")
    print("  [4] All (FP32 + FP16 + INT8 + INT4)")

    quant_choice = input("\n  Enter choice (1/2/3/4): ").strip()

    if quant_choice == "1":
        quant_types = ["fp16"]
    elif quant_choice == "2":
        quant_types = ["int8"]
    elif quant_choice == "3":
        quant_types = ["int4"]
    else:
        quant_types = ["fp32", "fp16", "int8", "int4"]

    dataset_path = str(DATASET_PATHS["smsa"])
    sample_frac = 1.0

    num_runs_input = input("\n  Inference runs per sample (0 = skip latency, default=20): ").strip()
    if num_runs_input:
        num_runs = max(0, int(num_runs_input))
    else:
        num_runs = 20

    return methods, quant_types, dataset_path, sample_frac, evaluate_only, num_runs

def _load_evaluation_json(method, quant_type):
    output_dir = BASE_DIR / "outputs"
    json_path = output_dir / f"indobert-qat-{quant_type}-smsa" / f"evaluation_results_{quant_type}_eager.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

def _generate_qat_comparison(method, quant_types):
    all_results = {}
    for qt in quant_types:
        data = _load_evaluation_json(method, qt)
        if data is not None:
            all_results[qt] = data

    if not all_results:
        print(f"\nSkipping {method.upper()} comparison graph (no results found)")
        return

    first_data = next(iter(all_results.values()))
    if 'fp32_metrics' in first_data and 'fp32_latencies' in first_data:
        all_results['fp32'] = {
            'overall_metrics': first_data['fp32_metrics'],
            'latencies': first_data['fp32_latencies'],
            'latency_stats': first_data.get('fp32_latency_stats', {}),
            'classification_report': first_data.get('fp32_classification_report', {}),
            'memory_usage_mb': first_data.get('fp32_memory_usage_mb', 0),
        }

    if 'fp32' not in all_results:
        import os
        output_dir = BASE_DIR / "outputs"
        for qt in quant_types:
            fp32_json = output_dir / f"indobert-qat-{qt}-smsa" / "evaluation_results_fp32_eager.json"
            if fp32_json.exists():
                with open(fp32_json, 'r') as f:
                    all_results['fp32'] = json.load(f)
                break

    if len(all_results) < 2:
        print(f"\nSkipping {method.upper()} comparison graph (need at least 2 results, found {len(all_results)})")
        return

    first_data = next(iter(all_results.values()))
    if 'fp32_metrics' in first_data and 'fp32_latencies' in first_data:
        all_results['fp32'] = {
            'overall_metrics': first_data['fp32_metrics'],
            'latencies': first_data['fp32_latencies'],
            'latency_stats': first_data.get('fp32_latency_stats', {}),
            'classification_report': first_data.get('fp32_classification_report', {}),
            'memory_usage_mb': first_data.get('fp32_memory_usage_mb', 0),
        }

    import os
    output_dir = BASE_DIR / "outputs"
    model_sizes = {}
    for qt in quant_types:
        if qt not in all_results:
            continue
        model_dir = output_dir / f"indobert-qat-{qt}-smsa"
        onnx_file = model_dir / f"model_qat_{qt}.onnx"
        if onnx_file.exists():
            model_sizes[qt] = os.path.getsize(onnx_file) / (1024 * 1024)
        if 'fp32' not in model_sizes:
            fp32_onnx = model_dir / "model_qat.onnx"
            if fp32_onnx.exists():
                model_sizes['fp32'] = os.path.getsize(fp32_onnx) / (1024 * 1024)

    plotter = QuantizationPlotter(output_dir)
    memory_usages = {}
    for qt in quant_types:
        if qt in all_results:
            memory_usages[qt] = all_results[qt].get('memory_usage_mb', 0)
    if 'fp32' in all_results:
        memory_usages['fp32'] = all_results['fp32'].get('memory_usage_mb', 0)
    chart_path = plotter.create_qat_comparison_plot(all_results, method, model_sizes=model_sizes, memory_usages=memory_usages)
    print(f"\nQAT {method.upper()} comparison chart saved to: {chart_path}")

def run_qat_from_menu(methods, quant_types, dataset_path=None, sample_frac=1.0, evaluate_only=False, num_runs=20):
    total = len(methods) * len(quant_types)
    current = 0

    for method in methods:
        for quant_type in quant_types:
            current += 1
            mode_label = "Evaluating" if evaluate_only else "Running"
            print("\n" + "=" * 70)
            print(f"[{current}/{total}] {mode_label} {method.upper()} QAT with {quant_type.upper()}")
            print("=" * 70)

            run_eager_qat(quant_type, dataset_path=dataset_path, sample_frac=sample_frac, evaluate_only=evaluate_only, num_runs=num_runs)

    for method in methods:
        _generate_qat_comparison(method, quant_types)

    for method in methods:
        experiment_key = f"qat_{method}_smsa"
        print(f"\n  Generating prediction divergences for {method.upper()}...")
        generate_qat_divergences(experiment_key)

    print("\n" + "=" * 70)
    print("All QAT experiments completed!")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Run QAT experiments for 3-label SMSA sentiment analysis"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["eager"],
        default="eager",
        help="QAT method: eager (ONNX pipeline)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        choices=["int8", "int4", "all"],
        default="all",
        help="Quantization type: int8, int4, or all (FP16 QAT is retired)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["smsa"],
        default="smsa",
        help="Evaluation dataset: smsa (test.tsv)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to use for Tweets dataset (0.0-1.0, default: 1.0 = all data)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        default=False,
        help="Skip training and only evaluate existing models",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=20,
        help="Inference runs per sample for latency measurement (0 = skip latency)",
    )
    parser.add_argument(
        "--multiseed-qat",
        action="store_true",
        default=False,
        help="Run multi-seed QAT from FP32 checkpoints (seeds 42/123/456)",
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=3,
        help="Number of QAT training epochs (multi-seed mode only)",
    )
    parser.add_argument(
        "--qat-lr",
        type=float,
        default=1e-5,
        help="QAT learning rate (multi-seed mode only)",
    )
    parser.add_argument(
        "--multiseed-qat-onnx",
        action="store_true",
        default=False,
        help="Export clean QAT checkpoints to ONNX, quantise, and evaluate",
    )
    parser.add_argument(
        "--generate-ece",
        action="store_true",
        default=False,
        help="Generate results/ece_all.csv and results/ece_summary.csv from existing prediction CSVs",
    )
    args = parser.parse_args()

    if args.generate_ece:
        _generate_combined_csv(list(TRAINING_SEEDS))
        return

    if args.multiseed_qat:
        run_multiseed_qat(epochs=args.qat_epochs, lr=args.qat_lr)
        return

    if args.multiseed_qat_onnx:
        run_multiseed_qat_onnx()
        return

    methods = ["eager"]
    quant_types = ["int8", "int4"] if args.quant_type == "all" else [args.quant_type]

    dataset_path = str(DATASET_PATHS["smsa"])

    run_qat_from_menu(methods, quant_types, dataset_path=dataset_path, sample_frac=args.sample_frac, evaluate_only=args.evaluate_only, num_runs=args.num_runs)

if __name__ == "__main__":
    main()
