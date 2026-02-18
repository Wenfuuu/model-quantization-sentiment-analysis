import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_ptq import interactive_menu as ptq_menu, run_ptq_experiment
from scripts.run_xai import interactive_menu as xai_menu, run_xai_experiment
from src.config import EXPERIMENT_CONFIGS
from src.utils import print_section


def run_ptq():
    selected, num_runs_override = ptq_menu()

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
        print(f"  Accuracy: FP32={res['fp32_results']['accuracy']*100:.2f}% | FP16={res['fp16_results']['accuracy']*100:.2f}% | INT8={res['int8_results']['accuracy']*100:.2f}% | INT4={res['int4_results']['accuracy']*100:.2f}%")
        print(f"  Size: FP32={res['fp32_size_mb']:.1f}MB | FP16={res['fp16_size_mb']:.1f}MB | INT8={res['int8_size_mb']:.1f}MB | INT4={res['int4_size_mb']:.1f}MB")


def run_xai():
    experiment_key, precisions, num_samples, divergence_samples = xai_menu()

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
    print(f"Results saved to: outputs/{experiment_key}/xai/comparison/")
    print("  - lime_comparison_sample_N.png : LIME side-by-side across precisions")
    print("  - shap_comparison_sample_N.png : SHAP side-by-side across precisions")
    print("  - prediction_summary.png       : Prediction correctness table")


def main():
    print("\n" + "=" * 60)
    print("  MODEL QUANTIZATION & SENTIMENT ANALYSIS")
    print("=" * 60)

    print("\n  What would you like to run?")
    print("  [1] PTQ (Post-Training Quantization)")
    print("  [2] QAT (Quantization-Aware Training)")
    print("  [3] XAI (Explainability Analysis)")

    choice = input("\n  Enter choice (1/2/3): ").strip()

    if choice == "1":
        run_ptq()
    elif choice == "2":
        print("\n  QAT is not implemented yet.")
    elif choice == "3":
        run_xai()
    else:
        print("\n  Invalid choice.")


if __name__ == "__main__":
    main()
