import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_ptq import interactive_menu as ptq_menu, run_ptq_experiment
from scripts.run_fake_ptq import interactive_menu as fake_ptq_menu, run_fake_ptq_experiment
from scripts.run_xai import interactive_menu as xai_menu, run_xai_experiment
from scripts.run_qat import interactive_menu as qat_menu, run_qat_from_menu
from scripts.run_stress_test import interactive_menu as stress_menu, run_stress_test_experiment
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


def run_fake_ptq():
    selected, num_runs_override = fake_ptq_menu()

    print("\n" + "=" * 80)
    print(f"STARTING FAKE PTQ EXPERIMENTS - {len(selected)} VERSION(S) TO RUN")
    print("=" * 80)
    for i, key in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {key}")
    print("=" * 80 + "\n")

    all_results = {}
    for idx, version_key in enumerate(selected, 1):
        print("\n" + "#" * 80)
        print(f"# RUNNING FAKE PTQ EXPERIMENT [{idx}/{len(selected)}]: {version_key.upper()}")
        print("#" * 80 + "\n")
        result = run_fake_ptq_experiment(version_key, num_runs_override=num_runs_override)
        all_results[version_key] = result
        print(f"\n  Completed [{idx}/{len(selected)}]: {version_key}")

    print_section("ALL FAKE PTQ EXPERIMENTS COMPLETED")
    for key, res in all_results.items():
        print(f"\n{key}:")
        print(f"  Accuracy: FP32={res['fp32_results']['accuracy']*100:.2f}% | FakeFP16={res['fp16_results']['accuracy']*100:.2f}% | FakeINT8={res['int8_results']['accuracy']*100:.2f}% | FakeINT4={res['int4_results']['accuracy']*100:.2f}%")
        print(f"  Size: FP32={res['fp32_size_mb']:.1f}MB | FakeFP16={res['fp16_size_mb']:.1f}MB | FakeINT8={res['int8_size_mb']:.1f}MB | FakeINT4={res['int4_size_mb']:.1f}MB")


def run_qat():
    methods, quant_types, dataset_path = qat_menu()

    total = len(methods) * len(quant_types)
    combos = [f"{m.upper()} {q.upper()}" for m in methods for q in quant_types]

    print("\n" + "=" * 80)
    print(f"STARTING QAT EXPERIMENTS - {total} COMBINATION(S) TO RUN")
    if dataset_path:
        print(f"Evaluation Dataset: {dataset_path}")
    print("=" * 80)
    for i, combo in enumerate(combos, 1):
        print(f"  [{i}/{total}] {combo}")
    print("=" * 80 + "\n")

    run_qat_from_menu(methods, quant_types, dataset_path=dataset_path)

    print_section("ALL QAT EXPERIMENTS COMPLETED")


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


def main():
    print("\n" + "=" * 60)
    print("  MODEL QUANTIZATION & SENTIMENT ANALYSIS")
    print("=" * 60)

    print("\n  What would you like to run?")
    print("  [1] PTQ (Post-Training Quantization)")
    print("  [2] QAT (Quantization-Aware Training)")
    print("  [3] XAI (Explainability Analysis)")
    print("  [4] Stress Test (Robustness Analysis)")

    choice = input("\n  Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        print("\n  Select PTQ Method:")
        print("  [1] PTQ Original (Real Quantization)")
        print("  [2] PTQ Fake Quantization (Simulated)")

        ptq_choice = input("\n  Enter choice (1/2): ").strip()

        if ptq_choice == "2":
            run_fake_ptq()
        else:
            run_ptq()
    elif choice == "2":
        run_qat()
    elif choice == "3":
        run_xai()
    elif choice == "4":
        run_stress()
    else:
        print("\n  Invalid choice.")


if __name__ == "__main__":
    main()

