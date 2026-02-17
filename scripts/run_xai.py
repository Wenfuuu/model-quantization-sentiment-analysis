import sys
import torch
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENT_CONFIGS, LABELS, DEVICE
from src.data import load_smsa_dataset, load_tweets_dataset
from src.models import ModelManager
from src.quantization.ptq import PTQQuantizer
from src.models.base import BaseModel
from src.xai import LIMEExplainer, SHAPExplainer
from src.utils import print_section

warnings.filterwarnings('ignore')


def select_samples(dataset_samples, num_samples=3):
    by_label = {}
    for sample in dataset_samples:
        label = sample["expected"]
        if label not in by_label:
            by_label[label] = sample

    selected = list(by_label.values())

    if len(selected) < num_samples:
        for sample in dataset_samples:
            if sample not in selected:
                selected.append(sample)
                if len(selected) >= num_samples:
                    break

    return selected[:num_samples]


def run_xai_for_model(base_model, precision_name, samples, output_dir, use_fp16=False):
    xai_dir = output_dir / "xai" / precision_name
    lime_dir = xai_dir / "lime"
    shap_dir = xai_dir / "shap"

    print_section(f"LIME EXPLANATIONS - {precision_name.upper()}")

    lime_explainer = LIMEExplainer(base_model, LABELS, use_fp16=use_fp16)

    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}/{len(samples)}: \"{sample['text'][:80]}...\"")
        print(f"  Expected: {sample['expected']}")

        lime_result = lime_explainer.explain_and_save(
            sample["text"],
            lime_dir / f"sample_{i+1}.html",
            num_features=10,
            num_samples=300
        )

        print(f"  Predicted: {lime_result['predicted_label']}")
        print(f"  Probabilities: ", end="")
        for label, prob in lime_result["prediction_probabilities"].items():
            print(f"{label}={prob*100:.1f}% ", end="")
        print()
        print(f"  Top Features (LIME):")
        for feat, weight in lime_result["top_features"][:5]:
            direction = "+" if weight > 0 else "-"
            print(f"    {direction} \"{feat}\": {weight:.4f}")
        print(f"  Saved: {lime_result['output_path']}")

    print_section(f"SHAP EXPLANATIONS - {precision_name.upper()}")

    shap_explainer = SHAPExplainer(base_model, LABELS, use_fp16=use_fp16)

    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}/{len(samples)}: \"{sample['text'][:80]}...\"")
        print(f"  Expected: {sample['expected']}")

        shap_result = shap_explainer.explain_and_save(
            sample["text"],
            shap_dir / f"sample_{i+1}.png",
            max_evals=200
        )

        print(f"  Predicted: {shap_result['predicted_label']}")
        print(f"  Top Token Importance (SHAP values):")
        for token, value in shap_result["token_importance"][:5]:
            direction = "+" if value > 0 else "-"
            print(f"    {direction} \"{token}\": {value:.4f}")
        print(f"  Saved: {shap_result['output_path']}")


def interactive_menu():
    print("\n" + "=" * 60)
    print("  XAI ANALYSIS RUNNER")
    print("=" * 60)

    print("\n  Select Quantization Method:")
    print("  [1] PTQ (Post-Training Quantization)")
    print("  [2] QAT (Quantization-Aware Training) [Not Implemented]")

    method_choice = input("\n  Enter choice (1/2): ").strip()

    if method_choice == "2":
        print("\n  QAT is not implemented yet.")
        sys.exit(0)

    print("\n  Select Model:")
    print("  [1] Original IndoBERT (indobenchmark/indobert-base-p2)")
    print("  [2] Finetuned IndoBERT (indobert-fp32-smsa-3label)")
    print("  [3] Both")

    model_choice = input("\n  Enter choice (1/2/3): ").strip()

    print("\n  Select Dataset:")
    print("  [1] SMSA (test.tsv)")
    print("  [2] Tweets (INA_TweetsPPKM)")

    dataset_choice = input("\n  Enter choice (1/2): ").strip()

    print("\n  Select Precision:")
    print("  [1] FP32 (Original)")
    print("  [2] FP16 (Half Precision)")
    print("  [3] INT8 (Dynamic Quantization)")
    print("  [4] INT4 (4-bit Quantization)")
    print("  [5] All Precisions")

    precision_choice = input("\n  Enter choice (1/2/3/4/5): ").strip()

    num_samples_str = input("\n  Number of samples to explain (default 3): ").strip()
    num_samples = int(num_samples_str) if num_samples_str else 3

    models = []
    if model_choice == "1":
        models = ["original"]
    elif model_choice == "2":
        models = ["finetuned"]
    else:
        models = ["original", "finetuned"]

    dataset = "smsa" if dataset_choice == "1" else "tweets"

    precisions = []
    if precision_choice == "1":
        precisions = ["fp32"]
    elif precision_choice == "2":
        precisions = ["fp16"]
    elif precision_choice == "3":
        precisions = ["int8"]
    elif precision_choice == "4":
        precisions = ["int4"]
    else:
        precisions = ["fp32", "fp16", "int8", "int4"]

    experiment_keys = [f"{m}_{dataset}" for m in models]

    return experiment_keys, precisions, num_samples


def run_xai_experiment(version_key, precisions, num_samples):
    config = EXPERIMENT_CONFIGS[version_key]
    output_dir = Path(config["output_dir"])

    print_section(f"XAI EXPERIMENT: {version_key}")
    print(f"Model: {config['model_id']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"Num samples: {num_samples}")

    if config["dataset"] == "smsa":
        all_samples = load_smsa_dataset()
    else:
        all_samples = load_tweets_dataset()

    samples = select_samples(all_samples, num_samples)
    print(f"\nSelected {len(samples)} representative samples:")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. [{s['expected']}] \"{s['text'][:60]}...\"")

    print(f"\nLoading model: {config['model_id']}")
    base_model = ModelManager.load_model(config['model_id'])

    for precision in precisions:
        if precision == "fp32":
            run_xai_for_model(base_model, "fp32", samples, output_dir)

        elif precision == "fp16":
            ptq = PTQQuantizer(base_model.model)
            model_fp16, fp16_time = ptq.quantize_fp16()
            print(f"\n  FP16 quantization: {fp16_time:.2f}s")
            fp16_model = BaseModel(model_fp16, base_model.tokenizer)
            run_xai_for_model(fp16_model, "fp16", samples, output_dir, use_fp16=True)

        elif precision == "int8":
            ptq = PTQQuantizer(base_model.model)
            model_int8, int8_time = ptq.quantize_int8()
            print(f"\n  INT8 quantization: {int8_time:.2f}s")
            int8_model = BaseModel(model_int8, base_model.tokenizer, device=torch.device("cpu"))
            run_xai_for_model(int8_model, "int8", samples, output_dir)

        elif precision == "int4":
            ptq = PTQQuantizer(base_model.model)
            model_int4, int4_time = ptq.quantize_int4()
            print(f"\n  INT4 quantization: {int4_time:.2f}s")
            int4_model = BaseModel(model_int4, base_model.tokenizer)
            run_xai_for_model(int4_model, "int4", samples, output_dir)


if __name__ == "__main__":
    experiment_keys, precisions, num_samples = interactive_menu()

    print("\n" + "=" * 80)
    print(f"STARTING XAI ANALYSIS - {len(experiment_keys)} EXPERIMENT(S)")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"Samples per precision: {num_samples}")
    print("=" * 80)

    for idx, key in enumerate(experiment_keys, 1):
        print(f"\n[{idx}/{len(experiment_keys)}] {key}")
        run_xai_experiment(key, precisions, num_samples)

    print_section("XAI ANALYSIS COMPLETED")
    print("LIME results: HTML files (open in browser to see word-level importance)")
    print("SHAP results: PNG bar charts (SHAP value per token)")
    print("\nWhat the results tell you:")
    print("  - LIME highlights which words pushed the prediction toward each sentiment")
    print("    Positive weights = word supports the predicted class")
    print("    Negative weights = word opposes the predicted class")
    print("  - SHAP shows each token's contribution (Shapley value) to the prediction")
    print("    Higher absolute value = more influential token")
    print("  - Comparing across precisions shows if quantization changes model reasoning")
