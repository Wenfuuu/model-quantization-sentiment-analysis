import sys
import torch
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def collect_xai_results(base_model, precision_name, samples, use_fp16=False):
    lime_explainer = LIMEExplainer(base_model, LABELS, use_fp16=use_fp16)
    shap_explainer = SHAPExplainer(base_model, LABELS, use_fp16=use_fp16)

    lime_results = []
    shap_results = []

    print(f"\n  Running LIME for {precision_name.upper()}...")
    for i, sample in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        explanation = lime_explainer.explain(sample["text"], num_features=10, num_samples=300)
        predicted_idx = int(np.argmax(explanation.predict_proba))
        label_names = [LABELS[j] for j in sorted(LABELS.keys())]
        features = explanation.as_list(label=predicted_idx)
        lime_results.append({
            "predicted_label": label_names[predicted_idx],
            "top_features": features,
            "probabilities": {label_names[j]: float(explanation.predict_proba[j]) for j in range(len(label_names))}
        })

    print(f"  Running SHAP for {precision_name.upper()}...")
    for i, sample in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        shap_values = shap_explainer.explain(sample["text"], max_evals=200)
        predicted_class = int(np.argmax(shap_explainer.predict_proba(sample["text"])))
        token_importance = {}
        if hasattr(shap_values[0], 'data') and hasattr(shap_values[0], 'values'):
            data = shap_values[0].data
            values = shap_values[0].values
            for j, token in enumerate(data):
                if isinstance(token, str) and token.strip():
                    token_importance[token] = float(values[j][predicted_class])
        sorted_imp = sorted(token_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        label_names = [LABELS[j] for j in sorted(LABELS.keys())]
        shap_results.append({
            "predicted_label": label_names[predicted_class],
            "token_importance": sorted_imp
        })

    return lime_results, shap_results


def generate_lime_comparison(all_lime, samples, precisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(len(samples)):
        fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 8))
        if len(precisions) == 1:
            axes = [axes]

        fig.suptitle(f"LIME Comparison - Sample {sample_idx + 1}\n\"{samples[sample_idx]['text']}\"\nExpected: {samples[sample_idx]['expected']}", fontsize=10, fontweight='bold', wrap=True)

        for col, precision in enumerate(precisions):
            ax = axes[col]
            lime_data = all_lime[precision][sample_idx]
            features = lime_data["top_features"][:10]

            if features:
                words = [f[0] for f in features]
                weights = [f[1] for f in features]
                colors = ['#ff4444' if w < 0 else '#4444ff' for w in weights]

                y_pos = range(len(words))
                ax.barh(y_pos, weights, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words, fontsize=9)
                ax.invert_yaxis()
                ax.axvline(x=0, color='black', linewidth=0.5)

            ax.set_title(f"{precision.upper()}\nPred: {lime_data['predicted_label']}", fontweight='bold')
            ax.set_xlabel("LIME weight")

        plt.tight_layout()
        path = output_dir / f"lime_comparison_sample_{sample_idx + 1}.png"
        plt.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def generate_shap_comparison(all_shap, samples, precisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(len(samples)):
        fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 8))
        if len(precisions) == 1:
            axes = [axes]

        fig.suptitle(f"SHAP Comparison - Sample {sample_idx + 1}\n\"{samples[sample_idx]['text']}\"\nExpected: {samples[sample_idx]['expected']}", fontsize=10, fontweight='bold', wrap=True)

        for col, precision in enumerate(precisions):
            ax = axes[col]
            shap_data = all_shap[precision][sample_idx]
            token_imp = shap_data["token_importance"][:10]

            if token_imp:
                tokens = [t[0] for t in token_imp]
                values = [t[1] for t in token_imp]
                colors = ['#ff4444' if v < 0 else '#4444ff' for v in values]

                y_pos = range(len(tokens))
                ax.barh(y_pos, values, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(tokens, fontsize=9)
                ax.invert_yaxis()
                ax.axvline(x=0, color='black', linewidth=0.5)

            ax.set_title(f"{precision.upper()}\nPred: {shap_data['predicted_label']}", fontweight='bold')
            ax.set_xlabel("SHAP value")

        plt.tight_layout()
        path = output_dir / f"shap_comparison_sample_{sample_idx + 1}.png"
        plt.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def generate_prediction_summary(all_lime, samples, precisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(precisions) * 3), max(4, len(samples) * 1.5)))

    cell_text = []
    row_labels = []
    for i, sample in enumerate(samples):
        row = []
        for precision in precisions:
            lime_data = all_lime[precision][i]
            pred = lime_data["predicted_label"]
            match = "Y" if pred == sample["expected"] else "N"
            row.append(f"{pred} ({match})")
        cell_text.append(row)
        row_labels.append(f"S{i+1}: {sample['expected']}")

    col_labels = [p.upper() for p in precisions]
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i, sample in enumerate(samples):
        for j, precision in enumerate(precisions):
            pred = all_lime[precision][i]["predicted_label"]
            cell = table[i + 1, j]
            if pred == sample["expected"]:
                cell.set_facecolor('#d4edda')
            else:
                cell.set_facecolor('#f8d7da')

    ax.set_title("Prediction Comparison Across Precisions\n(Green = Correct, Red = Wrong)", fontweight='bold', fontsize=12, pad=20)
    plt.tight_layout()
    path = output_dir / "prediction_summary.png"
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


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
    comparison_dir = output_dir / "xai" / "comparison"

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
        print(f"  {i}. [{s['expected']}] \"{s['text']}\"")

    print(f"\nLoading model: {config['model_id']}")
    base_model = ModelManager.load_model(config['model_id'])

    all_lime = {}
    all_shap = {}

    for precision in precisions:
        print_section(f"COLLECTING XAI DATA - {precision.upper()}")

        if precision == "fp32":
            lime_res, shap_res = collect_xai_results(base_model, "fp32", samples)

        elif precision == "fp16":
            ptq = PTQQuantizer(base_model.model)
            model_fp16, fp16_time = ptq.quantize_fp16()
            print(f"  FP16 quantization: {fp16_time:.2f}s")
            fp16_model = BaseModel(model_fp16, base_model.tokenizer)
            lime_res, shap_res = collect_xai_results(fp16_model, "fp16", samples, use_fp16=True)

        elif precision == "int8":
            ptq = PTQQuantizer(base_model.model)
            model_int8, int8_time = ptq.quantize_int8()
            print(f"  INT8 quantization: {int8_time:.2f}s")
            int8_model = BaseModel(model_int8, base_model.tokenizer, device=torch.device("cpu"))
            lime_res, shap_res = collect_xai_results(int8_model, "int8", samples)

        elif precision == "int4":
            ptq = PTQQuantizer(base_model.model)
            model_int4, int4_time = ptq.quantize_int4()
            print(f"  INT4 quantization: {int4_time:.2f}s")
            int4_model = BaseModel(model_int4, base_model.tokenizer)
            lime_res, shap_res = collect_xai_results(int4_model, "int4", samples)

        all_lime[precision] = lime_res
        all_shap[precision] = shap_res

        for i, (lr, sr) in enumerate(zip(lime_res, shap_res)):
            print(f"\n  Sample {i+1}: Pred={lr['predicted_label']} | Expected={samples[i]['expected']}")
            print(f"    LIME top 3: {', '.join(f'{f[0]}({f[1]:+.3f})' for f in lr['top_features'][:3])}")
            print(f"    SHAP top 3: {', '.join(f'{t[0]}({t[1]:+.3f})' for t in sr['token_importance'][:3])}")

    print_section("GENERATING COMPARISON CHARTS")

    print("\n  LIME Comparison:")
    generate_lime_comparison(all_lime, samples, precisions, comparison_dir)

    print("\n  SHAP Comparison:")
    generate_shap_comparison(all_shap, samples, precisions, comparison_dir)

    print("\n  Prediction Summary:")
    generate_prediction_summary(all_lime, samples, precisions, comparison_dir)


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
    print("Results saved to: outputs/{experiment}/xai/comparison/")
    print("  - lime_comparison_sample_N.png : LIME feature importance side by side")
    print("  - shap_comparison_sample_N.png : SHAP token importance side by side")
    print("  - prediction_summary.png       : Prediction correctness table")
