import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BASE_DIR, DATASET_PATHS
from src.quantization.qat.config import FinetuneQATConfig
from src.quantization.qat.eager import EagerQATTrainer
from src.quantization.qat.fake import FakeQATTrainer
from src.visualization import QuantizationPlotter


def get_default_config(method, quant_type, sample_frac=1.0):
    output_dir = BASE_DIR / "outputs"

    if method == "eager":
        save_name = f"indobert-qat-{quant_type}-smsa"
        results_name = f"indobert-qat-{quant_type}-smsa"
    else:
        save_name = f"indobert-smsa-qat-{quant_type}-fake"
        results_name = f"indobert-smsa-qat-{quant_type}-fake"

    return FinetuneQATConfig(
        model_id="indobenchmark/indobert-base-p2",
        train_file=DATASET_PATHS["smsa_train"],
        valid_file=DATASET_PATHS["smsa_valid"],
        test_file=DATASET_PATHS["smsa"],
        save_dir=output_dir / save_name,
        results_dir=output_dir / results_name,
        sample_frac=sample_frac,
    )


def run_eager_qat(quant_type, dataset_path=None, sample_frac=1.0):
    config = get_default_config("eager", quant_type, sample_frac=sample_frac)
    trainer = EagerQATTrainer(config, quantization_type=quant_type)

    print(f"\n[Step 1/4] Training with QAT ({quant_type.upper()} eager)...")
    trainer.train()

    print(f"\n[Step 2/4] Exporting to ONNX...")
    trainer.export_to_onnx()

    print(f"\n[Step 3/4] Quantizing ONNX to {quant_type.upper()}...")
    onnx_path = trainer.quantize_onnx()

    print(f"\n[Step 4/4] Evaluating {quant_type.upper()} ONNX model...")
    results = trainer.evaluate_onnx(onnx_path, dataset_path=dataset_path)
    return results


def run_fake_qat(quant_type, dataset_path=None, sample_frac=1.0):
    config = get_default_config("fake", quant_type, sample_frac=sample_frac)
    trainer = FakeQATTrainer(config, quantization_type=quant_type)

    print(f"\n[Step 1/2] Training with fake QAT ({quant_type.upper()})...")
    trainer.train()

    print(f"\n[Step 2/2] Evaluating {quant_type.upper()} fake QAT model...")
    results = trainer.evaluate(dataset_path=dataset_path)
    return results


def interactive_menu():
    print("\n" + "=" * 60)
    print("  QAT (QUANTIZATION-AWARE TRAINING) EXPERIMENT RUNNER")
    print("=" * 60)

    print("\n  Select QAT Method:")
    print("  [1] Eager (Train -> ONNX Export -> ONNX Quantize -> Eval)")
    print("  [2] Fake  (Train with fake quant -> HuggingFace save -> Eval)")
    print("  [3] Both")

    method_choice = input("\n  Enter choice (1/2/3): ").strip()

    print("\n  Select Quantization Type:")
    print("  [1] INT8")
    print("  [2] INT4")
    print("  [3] All (INT8 + INT4)")
    print("  [NOTE] FP16 QAT is retired — use PTQ-FP16 instead.")

    quant_choice = input("\n  Enter choice (1/2/3): ").strip()

    if method_choice == "1":
        methods = ["eager"]
    elif method_choice == "2":
        methods = ["fake"]
    else:
        methods = ["eager", "fake"]

    if quant_choice == "1":
        quant_types = ["int8"]
    elif quant_choice == "2":
        quant_types = ["int4"]
    else:
        quant_types = ["int8", "int4"]

    print("\n  Select Evaluation Dataset:")
    print(f"  [1] SMSA (test.tsv) - {DATASET_PATHS['smsa']}")
    print(f"  [2] Tweets (INA_TweetsPPKM) - {DATASET_PATHS['tweets']}")
    print("  [3] Default (datasets/test.tsv)")

    dataset_choice = input("\n  Enter choice (1/2/3): ").strip()

    if dataset_choice == "1":
        dataset_path = str(DATASET_PATHS["smsa"])
    elif dataset_choice == "2":
        dataset_path = str(DATASET_PATHS["tweets"])
    else:
        dataset_path = None

    sample_frac = 1.0
    if dataset_choice == "2":
        pct_input = input("\n  Berapa persen data Tweets yang digunakan? (1-100, default=100): ").strip()
        if pct_input:
            sample_frac = max(1, min(100, int(pct_input))) / 100.0

    return methods, quant_types, dataset_path, sample_frac, sample_frac


def _load_evaluation_json(method, quant_type):
    output_dir = BASE_DIR / "outputs"
    if method == "eager":
        json_path = output_dir / f"indobert-qat-{quant_type}-smsa" / f"evaluation_results_{quant_type}_eager.json"
    else:
        json_path = output_dir / f"indobert-smsa-qat-{quant_type}-fake" / f"evaluation_results_{quant_type}_fake.json"
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

    if len(all_results) < 2:
        print(f"\nSkipping {method.upper()} comparison graph (need at least 2 results, found {len(all_results)})")
        return

    import os
    output_dir = BASE_DIR / "outputs"
    model_sizes = {}
    for qt in quant_types:
        if qt not in all_results:
            continue
        if method == "eager":
            model_dir = output_dir / f"indobert-qat-{qt}-smsa"
            onnx_file = model_dir / f"model_qat_{qt}.onnx"
            if onnx_file.exists():
                model_sizes[qt] = os.path.getsize(onnx_file) / (1024 * 1024)
        else:
            model_dir = output_dir / f"indobert-smsa-qat-{qt}-fake"
            pth_file = model_dir / f"model_{qt}.pth"
            if pth_file.exists():
                model_sizes[qt] = os.path.getsize(pth_file) / (1024 * 1024)
            elif (model_dir / "model.safetensors").exists():
                model_sizes[qt] = os.path.getsize(model_dir / "model.safetensors") / (1024 * 1024)

    plotter = QuantizationPlotter(output_dir)
    chart_path = plotter.create_qat_comparison_plot(all_results, method, model_sizes=model_sizes)
    print(f"\nQAT {method.upper()} comparison chart saved to: {chart_path}")


def run_qat_from_menu(methods, quant_types, dataset_path=None, sample_frac=1.0):
    total = len(methods) * len(quant_types)
    current = 0

    for method in methods:
        for quant_type in quant_types:
            current += 1
            print("\n" + "=" * 70)
            print(f"[{current}/{total}] Running {method.upper()} QAT with {quant_type.upper()}")
            print("=" * 70)

            if method == "eager":
                run_eager_qat(quant_type, dataset_path=dataset_path, sample_frac=sample_frac)
            else:
                run_fake_qat(quant_type, dataset_path=dataset_path, sample_frac=sample_frac)

    for method in methods:
        _generate_qat_comparison(method, quant_types)

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
        choices=["eager", "fake", "all"],
        default="all",
        help="QAT method: eager (ONNX pipeline), fake (HuggingFace save), or all",
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
        choices=["smsa", "tweets", "default"],
        default="default",
        help="Evaluation dataset: smsa (test.tsv), tweets (INA_TweetsPPKM), or default (datasets/test.tsv)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to use for Tweets dataset (0.0-1.0, default: 1.0 = all data)",
    )
    args = parser.parse_args()

    methods = ["eager", "fake"] if args.method == "all" else [args.method]
    quant_types = ["int8", "int4"] if args.quant_type == "all" else [args.quant_type]

    if args.dataset in DATASET_PATHS:
        dataset_path = str(DATASET_PATHS[args.dataset])
    else:
        dataset_path = None

    run_qat_from_menu(methods, quant_types, dataset_path=dataset_path, sample_frac=args.sample_frac)


if __name__ == "__main__":
    main()
