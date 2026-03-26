import sys
import os
import tempfile
from pathlib import Path

try:
    hf_cache_dir = Path.home() / ".cache" / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
except (PermissionError, FileNotFoundError):
    hf_cache_dir = Path(tempfile.gettempdir()) / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(hf_cache_dir)

sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_ptq import interactive_menu as ptq_menu, run_ptq_experiment, run_multiseed_ptq
from scripts.run_xai import interactive_menu as xai_menu, run_xai_experiment, run_xai_diagnostics, run_ste_ig_analysis, run_lime_attribution, run_shap_attribution, run_occlusion_attribution
from scripts.run_qat import interactive_menu as qat_menu, run_qat_from_menu, run_multiseed_qat, run_multiseed_qat_onnx, _generate_combined_csv
from scripts.run_stress_test import interactive_menu as stress_menu, run_stress_test_experiment
from scripts.finetune_multi_seed import (
    main as finetune_main,
    DEFAULT_SEEDS,
    _AGG_OUTPUT_FILE,
    _FINETUNE_SCRIPT,
)
from src.config import EXPERIMENT_CONFIGS
from src.utils import print_section


def run_ptq():
    selected, num_runs_override = ptq_menu()

    if selected == "multiseed":
        run_multiseed_ptq()
        print_section("MULTI-SEED PTQ COMPLETED")
        return

    print("\n" + "=" * 80)
    print(f"STARTING PTQ EXPERIMENTS - {len(selected)} VERSION(S) TO RUN")
    print("=" * 80)
    for i, key in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {key}")
    print("=" * 80 + "\n")

    all_results = {}
    for idx, version_key in enumerate(selected, 1):
        print("\n" + "#" * 80)
        print(f"# RUNNING EXPERIMENT [{idx}/{len(selected)}]: {version_key.upper()}")
        print("#" * 80 + "\n")
        result = run_ptq_experiment(version_key, num_runs_override=num_runs_override)
        all_results[version_key] = result
        print(f"\n  Completed [{idx}/{len(selected)}]: {version_key}")

    print_section("ALL EXPERIMENTS COMPLETED")
    for key, res in all_results.items():
        print(f"\n{key}:")
        print(f"  Accuracy: FP32={res['fp32_results']['accuracy']*100:.4f}% | FP16={res['fp16_results']['accuracy']*100:.4f}% | INT8={res['int8_results']['accuracy']*100:.4f}% | INT4={res['int4_results']['accuracy']*100:.4f}%")
        print(f"  Size: FP32={res['fp32_size_mb']:.4f}MB | FP16={res['fp16_size_mb']:.4f}MB | INT8={res['int8_size_mb']:.4f}MB | INT4={res['int4_size_mb']:.4f}MB")


def run_qat():
    methods, quant_types, dataset_path, sample_frac, evaluate_only, num_runs = qat_menu()

    if methods == "multiseed":
        run_multiseed_qat()
        print_section("MULTI-SEED QAT COMPLETED")
        return

    if methods == "multiseed_onnx":
        run_multiseed_qat_onnx()
        print_section("MULTI-SEED QAT-ONNX COMPLETED")
        return

    if methods == "generate_ece":
        from src.config import TRAINING_SEEDS
        _generate_combined_csv(list(TRAINING_SEEDS))
        print_section("ECE SUMMARY GENERATED")
        return

    total = len(methods) * len(quant_types)
    combos = [f"{m.upper()} {q.upper()}" for m in methods for q in quant_types]

    mode_label = "EVALUATE ONLY" if evaluate_only else "TRAIN + EVALUATE"
    print("\n" + "=" * 80)
    print(f"STARTING QAT EXPERIMENTS - {total} COMBINATION(S) [{mode_label}]")
    if dataset_path:
        print(f"Evaluation Dataset: {dataset_path}")
    print(f"Inference runs per sample: {num_runs}")
    print("=" * 80)
    for i, combo in enumerate(combos, 1):
        print(f"  [{i}/{total}] {combo}")
    print("=" * 80 + "\n")

    run_qat_from_menu(methods, quant_types, dataset_path=dataset_path, sample_frac=sample_frac, evaluate_only=evaluate_only, num_runs=num_runs)

    print_section("ALL QAT EXPERIMENTS COMPLETED")


def run_xai():
    experiment_key, precisions, num_samples, divergence_samples = xai_menu()

    if experiment_key == "ste_ig":
        run_ste_ig_analysis()
        print_section("STE-IG ANALYSIS COMPLETED")
        return

    if experiment_key == "lime_attr":
        run_lime_attribution()
        print_section("LIME ATTRIBUTION COMPLETED")
        return

    if experiment_key == "shap_attr":
        run_shap_attribution()
        print_section("SHAP ATTRIBUTION COMPLETED")
        return

    if experiment_key == "occ_attr":
        run_occlusion_attribution()
        print_section("OCCLUSION ATTRIBUTION COMPLETED")
        return

    print("\n" + "=" * 80)
    print(f"STARTING XAI ANALYSIS: {experiment_key}")
    print(f"Precisions: {', '.join(precisions)}")
    if divergence_samples:
        print(f"Mode: Divergence analysis ({len(divergence_samples)} samples)")
    else:
        print(f"Mode: Auto-select ({num_samples} samples)")
    print("=" * 80)

    run_xai_experiment(experiment_key, precisions, num_samples, divergence_samples)

    print_section("XAI ANALYSIS COMPLETED")

def run_xai_diagnostics_menu():
    run_xai_diagnostics()

def run_stress():
    selected_experiments, selected_tests = stress_menu()

    total = len(selected_experiments)
    print("\n" + "=" * 80)
    print(f"STARTING STRESS TESTS - {total} EXPERIMENT(S)")
    print("=" * 80)
    for i, key in enumerate(selected_experiments, 1):
        print(f"  [{i}/{total}] {key}")
    print("=" * 80 + "\n")

    for idx, version_key in enumerate(selected_experiments, 1):
        print("\n" + "#" * 80)
        print(f"# STRESS TEST [{idx}/{total}]: {version_key.upper()}")
        print("#" * 80 + "\n")
        run_stress_test_experiment(version_key, tests=selected_tests)
        print(f"\n  Completed [{idx}/{total}]: {version_key}")

    print_section("ALL STRESS TESTS COMPLETED")


def run_finetune():
    import argparse

    print("\n" + "=" * 60)
    print("  FINETUNING: IndoBERT on SMSA (3-label)")
    print("  Preprocessing: lowercase + whitespace normalisation (no stopword removal)")
    print("=" * 60)

    seeds      = DEFAULT_SEEDS
    epochs     = 3
    lr         = 2e-5
    batch_size = 16

    args = argparse.Namespace(
        seeds=seeds,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        no_skip=False,
        agg_output=str(_AGG_OUTPUT_FILE),
        finetune_script=str(_FINETUNE_SCRIPT),
        ckpt_suffix="",
    )

    print("\n" + "=" * 80)
    print(f"STARTING FINETUNING - {len(seeds)} SEED(S)")
    print(f"  Seeds: {seeds} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print("=" * 80 + "\n")

    finetune_main(args)

    print_section("FINETUNING COMPLETED")


def main():
    print("\n" + "=" * 60)
    print("  MODEL QUANTIZATION & SENTIMENT ANALYSIS")
    print("=" * 60)

    print("\n  What would you like to run?")
    print("  [1] PTQ (Post-Training Quantization)")
    print("  [2] QAT (Quantization-Aware Training)")
    print("  [3] XAI (Explainability Analysis)")
    print("  [4] Stress Test (Robustness Analysis)")
    print("  [5] Finetune (IndoBERT on SMSA)")
    print("  [6] XAI Diagnostics (Alignment/Attention/IG Metrics)")

    choice = input("\n  Enter choice (1-6): ").strip()

    if choice == "1":
        run_ptq()
    elif choice == "2":
        run_qat()
    elif choice == "3":
        run_xai()
    elif choice == "4":
        run_stress()
    elif choice == "5":
        run_finetune()
    elif choice == "6":
        run_xai_diagnostics_menu()
    else:
        print("\n  Invalid choice.")


if __name__ == "__main__":
    main()

