import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BASE_DIR
from src.quantization.qat.config import FinetuneQATConfig
from src.quantization.qat.eager import EagerQATTrainer
from src.quantization.qat.fake import FakeQATTrainer


def get_default_config(method, quant_type):
    data_dir = BASE_DIR / "src" / "finetune_3label"

    if method == "eager":
        save_name = f"indobert-qat-{quant_type}-smsa"
        results_name = f"indobert-qat-{quant_type}-smsa"
    else:
        save_name = f"indobert-smsa-qat-{quant_type}-fake"
        results_name = f"indobert-smsa-qat-{quant_type}-fake"

    return FinetuneQATConfig(
        model_id="indobenchmark/indobert-base-p2",
        train_file=data_dir / "train.tsv",
        valid_file=data_dir / "valid.tsv",
        test_file=data_dir / "test.tsv",
        save_dir=data_dir / "results" / save_name,
        results_dir=data_dir / "results" / results_name,
    )


def run_eager_qat(quant_type):
    config = get_default_config("eager", quant_type)
    trainer = EagerQATTrainer(config, quantization_type=quant_type)

    print(f"\n[Step 1/4] Training with QAT ({quant_type.upper()} eager)...")
    trainer.train()

    print(f"\n[Step 2/4] Exporting to ONNX...")
    trainer.export_to_onnx()

    print(f"\n[Step 3/4] Quantizing ONNX to {quant_type.upper()}...")
    onnx_path = trainer.quantize_onnx()

    print(f"\n[Step 4/4] Evaluating {quant_type.upper()} ONNX model...")
    trainer.evaluate_onnx(onnx_path)


def run_fake_qat(quant_type):
    config = get_default_config("fake", quant_type)
    trainer = FakeQATTrainer(config, quantization_type=quant_type)

    print(f"\n[Step 1/2] Training with fake QAT ({quant_type.upper()})...")
    trainer.train()

    print(f"\n[Step 2/2] Evaluating {quant_type.upper()} fake QAT model...")
    trainer.evaluate()


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
    print("  [2] FP16")
    print("  [3] Both")

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
        quant_types = ["fp16"]
    else:
        quant_types = ["int8", "fp16"]

    return methods, quant_types


def run_qat_from_menu(methods, quant_types):
    total = len(methods) * len(quant_types)
    current = 0

    for method in methods:
        for quant_type in quant_types:
            current += 1
            print("\n" + "=" * 70)
            print(f"[{current}/{total}] Running {method.upper()} QAT with {quant_type.upper()}")
            print("=" * 70)

            if method == "eager":
                run_eager_qat(quant_type)
            else:
                run_fake_qat(quant_type)

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
        choices=["int8", "fp16", "all"],
        default="all",
        help="Quantization type: int8, fp16, or all",
    )
    args = parser.parse_args()

    methods = ["eager", "fake"] if args.method == "all" else [args.method]
    quant_types = ["int8", "fp16"] if args.quant_type == "all" else [args.quant_type]

    run_qat_from_menu(methods, quant_types)


if __name__ == "__main__":
    main()
