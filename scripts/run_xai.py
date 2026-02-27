import sys
import torch
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENT_CONFIGS, QAT_EXPERIMENT_CONFIGS, LABELS, DEVICE
from src.data import load_smsa_dataset, load_tweets_dataset
from src.models import ModelManager
from src.quantization.ptq import PTQQuantizer
from src.models.base import BaseModel
from src.xai import LIMEExplainer, SHAPExplainer, IntegratedGradientsExplainer, OcclusionExplainer
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


def load_divergences(experiment_key):
    config = EXPERIMENT_CONFIGS[experiment_key]
    output_dir = Path(config["output_dir"])
    divergence_path = output_dir / "prediction_divergences.json"
    
    if not divergence_path.exists():
        print(f"\n  Divergence file not found: {divergence_path}")
        print("  Please run PTQ experiment first to generate divergence data.")
        return None
    
    with open(divergence_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def collect_xai_results(base_model, precision_name, samples, use_fp16=False):
    lime_explainer = LIMEExplainer(base_model, LABELS, use_fp16=use_fp16)
    shap_explainer = SHAPExplainer(base_model, LABELS, use_fp16=use_fp16)
    occlusion_explainer = OcclusionExplainer(base_model, LABELS, use_fp16=use_fp16)

    lime_results = []
    shap_results = []
    ig_results = []
    occlusion_results = []

    print(f"\n  Running LIME for {precision_name.upper()}...")
    for i, sample in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        explanation = lime_explainer.explain(sample["text"], num_features=30, num_samples=300)
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
        sorted_imp = sorted(token_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        label_names = [LABELS[j] for j in sorted(LABELS.keys())]
        shap_results.append({
            "predicted_label": label_names[predicted_class],
            "token_importance": sorted_imp
        })

    print(f"  Running Integrated Gradients for {precision_name.upper()}...")
    ig_explainer = IntegratedGradientsExplainer(base_model.model, base_model.tokenizer, device=base_model.device)
    label_names = [LABELS[j] for j in sorted(LABELS.keys())]
    for i, sample in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        ig_result = ig_explainer.explain(sample["text"], steps=30)
        ig_results.append({
            "predicted_label": label_names[ig_result["predicted_class"]],
            "tokens": ig_result["tokens"],
            "scores": ig_result["scores"].tolist() if hasattr(ig_result["scores"], "tolist") else list(ig_result["scores"]),
        })

    print(f"  Running Occlusion for {precision_name.upper()}...")
    for i, sample in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        occ_result = occlusion_explainer.explain(sample["text"], window_size=1)
        occlusion_results.append(occ_result)

    return lime_results, shap_results, ig_results, occlusion_results


def generate_lime_comparison(all_lime, samples, precisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(len(samples)):
        num_words = len(samples[sample_idx]["text"].split())
        fig_height = max(8, num_words * 0.4)
        fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), fig_height))
        if len(precisions) == 1:
            axes = [axes]

        fig.suptitle(f"LIME Comparison - Sample {sample_idx + 1}\n\"{samples[sample_idx]['text']}\"\nExpected: {samples[sample_idx]['expected']}", fontsize=10, fontweight='bold', wrap=True)

        for col, precision in enumerate(precisions):
            ax = axes[col]
            lime_data = all_lime[precision][sample_idx]
            features = lime_data["top_features"]

            if features:
                feature_dict = {f[0]: f[1] for f in features}
                sentence_words = samples[sample_idx]["text"].split()
                ordered_words = []
                ordered_weights = []
                for word in sentence_words:
                    ordered_words.append(word)
                    ordered_weights.append(feature_dict.get(word, 0.0))
                for f_word, f_weight in features:
                    if f_word not in ordered_words:
                        ordered_words.append(f_word)
                        ordered_weights.append(f_weight)

                colors = ['#ff4444' if w < 0 else '#4444ff' for w in ordered_weights]

                y_pos = range(len(ordered_words))
                ax.barh(y_pos, ordered_weights, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(ordered_words, fontsize=9)
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
        num_words = len(samples[sample_idx]["text"].split())
        fig_height = max(8, num_words * 0.4)
        fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), fig_height))
        if len(precisions) == 1:
            axes = [axes]

        fig.suptitle(f"SHAP Comparison - Sample {sample_idx + 1}\n\"{samples[sample_idx]['text']}\"\nExpected: {samples[sample_idx]['expected']}", fontsize=10, fontweight='bold', wrap=True)

        for col, precision in enumerate(precisions):
            ax = axes[col]
            shap_data = all_shap[precision][sample_idx]
            token_imp = shap_data["token_importance"]

            if token_imp:
                token_dict = {t[0]: t[1] for t in token_imp}
                sentence_words = samples[sample_idx]["text"].split()
                ordered_tokens = []
                ordered_values = []
                for word in sentence_words:
                    ordered_tokens.append(word)
                    ordered_values.append(token_dict.get(word, 0.0))
                for t_word, t_val in token_imp:
                    if t_word not in ordered_tokens:
                        ordered_tokens.append(t_word)
                        ordered_values.append(t_val)

                colors = ['#ff4444' if v < 0 else '#4444ff' for v in ordered_values]

                y_pos = range(len(ordered_tokens))
                ax.barh(y_pos, ordered_values, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(ordered_tokens, fontsize=9)
                ax.invert_yaxis()
                ax.axvline(x=0, color='black', linewidth=0.5)

            ax.set_title(f"{precision.upper()}\nPred: {shap_data['predicted_label']}", fontweight='bold')
            ax.set_xlabel("SHAP value")

        plt.tight_layout()
        path = output_dir / f"shap_comparison_sample_{sample_idx + 1}.png"
        plt.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def generate_ig_comparison(all_ig, samples, precisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(len(samples)):
        fig_height = max(8, 20 * 0.4)
        fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), fig_height))
        if len(precisions) == 1:
            axes = [axes]

        fig.suptitle(f"Integrated Gradients Comparison - Sample {sample_idx + 1}\n\"{samples[sample_idx]['text']}\"\nExpected: {samples[sample_idx]['expected']}", fontsize=10, fontweight='bold', wrap=True)

        for col, precision in enumerate(precisions):
            ax = axes[col]
            ig_data = all_ig[precision][sample_idx]
            tokens = ig_data["tokens"]
            scores = ig_data["scores"]

            display_tokens = []
            display_scores = []
            for t, s in zip(tokens, scores):
                if t not in ["[CLS]", "[SEP]", "[PAD]"]:
                    display_tokens.append(t)
                    display_scores.append(s)

            if display_tokens:
                colors = ['#ff4444' if s < 0 else '#4444ff' for s in display_scores]
                y_pos = range(len(display_tokens))
                ax.barh(y_pos, display_scores, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(display_tokens, fontsize=9)
                ax.invert_yaxis()
                ax.axvline(x=0, color='black', linewidth=0.5)

            ax.set_title(f"{precision.upper()}\nPred: {ig_data['predicted_label']}", fontweight='bold')
            ax.set_xlabel("IG attribution")

        plt.tight_layout()
        path = output_dir / f"ig_comparison_sample_{sample_idx + 1}.png"
        plt.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def generate_occlusion_comparison(all_occ, samples, precisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(len(samples)):
        num_words = len(samples[sample_idx]["text"].split())
        fig_height = max(8, num_words * 0.4)
        fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), fig_height))
        if len(precisions) == 1:
            axes = [axes]

        fig.suptitle(f"Occlusion Comparison - Sample {sample_idx + 1}\n\"{samples[sample_idx]['text']}\"\nExpected: {samples[sample_idx]['expected']}", fontsize=10, fontweight='bold', wrap=True)

        for col, precision in enumerate(precisions):
            ax = axes[col]
            occ_data = all_occ[precision][sample_idx]
            token_imp = occ_data["all_tokens_ordered"]

            if token_imp:
                tokens_list = [t[0] for t in token_imp]
                values_list = [t[1] for t in token_imp]
                colors = ['#ff4444' if v < 0 else '#4444ff' for v in values_list]

                y_pos = range(len(tokens_list))
                ax.barh(y_pos, values_list, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(tokens_list, fontsize=9)
                ax.invert_yaxis()
                ax.axvline(x=0, color='black', linewidth=0.5)

            ax.set_title(f"{precision.upper()}\nPred: {occ_data['predicted_label']}", fontweight='bold')
            ax.set_xlabel("Confidence drop")

        plt.tight_layout()
        path = output_dir / f"occlusion_comparison_sample_{sample_idx + 1}.png"
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
    print("  [2] QAT (Quantization-Aware Training)")

    method_choice = input("\n  Enter choice (1/2): ").strip()

    if method_choice == "2":
        return _qat_menu()

    print("\n  Select Model:")
    print("  [1] Original IndoBERT (indobenchmark/indobert-base-p2)")
    print("  [2] Finetuned IndoBERT (indobert-fp32-smsa-3label)")

    model_choice = input("\n  Enter choice (1/2): ").strip()

    print("\n  Select Dataset:")
    print("  [1] SMSA (test.tsv)")
    print("  [2] Tweets (INA_TweetsPPKM)")

    dataset_choice = input("\n  Enter choice (1/2): ").strip()

    model = "original" if model_choice == "1" else "finetuned"
    dataset = "smsa" if dataset_choice == "1" else "tweets"
    experiment_key = f"{model}_{dataset}"

    print("\n  Sample Selection Mode:")
    print("  [1] Auto-select by label diversity")
    print("  [2] From prediction divergences (requires PTQ run)")

    sample_mode = input("\n  Enter choice (1/2): ").strip()

    divergence_samples = None
    precisions = None

    if sample_mode == "2":
        div_data = load_divergences(experiment_key)
        if div_data is None:
            sys.exit(1)

        divergences = div_data["divergences"]
        if not divergences:
            print("\n  No divergences found - all models agree!")
            print("  Switching to auto-select mode.")
            sample_mode = "1"
        else:
            print(f"\n  Found {len(divergences)} divergent samples (out of {div_data['total_samples']} total)")

            print("\n  Compare models:")
            print("  [1] All precisions (all divergences)")
            print("  [2] FP32 vs FP16")
            print("  [3] FP32 vs INT8")
            print("  [4] FP32 vs INT4")
            print("  [5] INT8 vs INT4")
            print("  [6] Specific model vs all others")
            print("  [7] Custom precisions")

            compare_choice = input("\n  Enter choice (1-7): ").strip()

            pair_filter = None
            if compare_choice == "2":
                pair_filter = ["fp32", "fp16"]
                precisions = ["fp32", "fp16"]
            elif compare_choice == "3":
                pair_filter = ["fp32", "int8"]
                precisions = ["fp32", "int8"]
            elif compare_choice == "4":
                pair_filter = ["fp32", "int4"]
                precisions = ["fp32", "int4"]
            elif compare_choice == "5":
                pair_filter = ["int8", "int4"]
                precisions = ["int8", "int4"]
            elif compare_choice == "6":
                print("\n  Which model to compare against all others?")
                print("  [1] FP32  [2] FP16  [3] INT8  [4] INT4")
                m_choice = input("  Enter choice: ").strip()
                model_map = {"1": "fp32", "2": "fp16", "3": "int8", "4": "int4"}
                target_model = model_map.get(m_choice, "int8")
                pair_filter = [target_model, "all"]
                precisions = ["fp32", "fp16", "int8", "int4"]
            elif compare_choice == "7":
                custom = input("\n  Enter precisions comma-separated (e.g., fp32,int8,int4): ").strip()
                precisions = [p.strip().lower() for p in custom.split(",")]
                if len(precisions) == 2:
                    pair_filter = precisions[:]
            else:
                pair_filter = None
                precisions = ["fp32", "fp16", "int8", "int4"]

            if pair_filter:
                if pair_filter[1] == "all":
                    target = pair_filter[0]
                    filtered = []
                    for d in divergences:
                        preds = d["predictions"]
                        target_label = preds[target]["label"]
                        if any(preds[p]["label"] != target_label for p in preds if p != target):
                            filtered.append(d)
                else:
                    filtered = []
                    for d in divergences:
                        preds = d["predictions"]
                        if preds[pair_filter[0]]["label"] != preds[pair_filter[1]]["label"]:
                            filtered.append(d)
            else:
                filtered = divergences

            if not filtered:
                print(f"\n  No divergences found for the selected comparison.")
                print("  Switching to auto-select mode.")
                sample_mode = "1"
            else:
                all_precs = ["fp32", "fp16", "int8", "int4"]
                print(f"\n  {len(filtered)} divergent samples found:")
                for idx, d in enumerate(filtered, 1):
                    preds_str = "  ".join(f"{p.upper()}={d['predictions'][p]['label']}({d['predictions'][p]['confidence']*100:.1f}%)" for p in all_precs)
                    print(f"\n  [{idx}] Sample #{d['sample_idx']+1}: Expected={d['expected']}")
                    print(f"      {preds_str}")
                    print(f"      \"{d['text']}\"")

                select_str = input(f"\n  Select samples: [A]ll or enter numbers (e.g., 1,3,5): ").strip()

                if select_str.upper() == "A" or select_str == "":
                    selected_divs = filtered
                else:
                    indices = [int(x.strip()) - 1 for x in select_str.split(",")]
                    selected_divs = [filtered[i] for i in indices if 0 <= i < len(filtered)]

                divergence_samples = [{"text": d["text"], "expected": d["expected"]} for d in selected_divs]
                num_samples = len(divergence_samples)

                print(f"\n  Selected {num_samples} divergent samples for XAI analysis")
                print(f"  Precisions: {', '.join(p.upper() for p in precisions)}")

                return experiment_key, precisions, num_samples, divergence_samples

    if sample_mode != "2":
        print("\n  Select Precision:")
        print("  [1] FP32 (Original)")
        print("  [2] FP16 (Half Precision)")
        print("  [3] INT8 (Dynamic Quantization)")
        print("  [4] INT4 (4-bit Quantization)")
        print("  [5] All Precisions")

        precision_choice = input("\n  Enter choice (1/2/3/4/5): ").strip()

        num_samples_str = input("\n  Number of samples to explain (default 3): ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 3

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

    return experiment_key, precisions, num_samples, None


def _qat_menu():
    print("\n  Select QAT Method:")
    print("  [1] Eager (ONNX pipeline models)")
    print("  [2] Fake  (HuggingFace saved models)")

    qat_method_choice = input("\n  Enter choice (1/2): ").strip()

    if qat_method_choice == "1":
        experiment_key = "qat_eager_smsa"
    else:
        experiment_key = "qat_fake_smsa"

    print("\n  Select Quantization Type:")
    print("  [1] INT8")
    print("  [2] FP16")
    print("  [3] INT4")
    print("  [4] All")

    quant_choice = input("\n  Enter choice (1/2/3/4): ").strip()

    if quant_choice == "1":
        precisions = ["int8"]
    elif quant_choice == "2":
        precisions = ["fp16"]
    elif quant_choice == "3":
        precisions = ["int4"]
    else:
        precisions = ["int8", "fp16", "int4"]

    num_samples_str = input("\n  Number of samples to explain (default 3): ").strip()
    num_samples = int(num_samples_str) if num_samples_str else 3

    return experiment_key, precisions, num_samples, None


def _is_qat_experiment(version_key):
    return version_key in QAT_EXPERIMENT_CONFIGS


def run_xai_experiment(version_key, precisions, num_samples, divergence_samples=None):
    is_qat = _is_qat_experiment(version_key)

    if is_qat:
        config = QAT_EXPERIMENT_CONFIGS[version_key]
    else:
        config = EXPERIMENT_CONFIGS[version_key]

    output_dir = Path(config["output_dir"])
    comparison_dir = output_dir / "xai" / "comparison"

    print_section(f"XAI EXPERIMENT: {version_key}")
    if not is_qat:
        print(f"Model: {config['model_id']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Precisions: {', '.join(precisions)}")

    if divergence_samples:
        samples = divergence_samples
        print(f"Mode: Divergence analysis ({len(samples)} divergent samples)")
    else:
        if config["dataset"] == "smsa":
            all_samples = load_smsa_dataset()
        else:
            all_samples = load_tweets_dataset()
        samples = select_samples(all_samples, num_samples)
        print(f"Mode: Auto-select ({num_samples} samples)")

    print(f"\nSelected {len(samples)} samples for XAI analysis:")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. [{s['expected']}] \"{s['text']}\"")

    if not is_qat:
        print(f"\nLoading model: {config['model_id']}")
        base_model = ModelManager.load_model(config['model_id'])

    all_lime = {}
    all_shap = {}
    all_ig = {}
    all_occ = {}

    for precision in precisions:
        print_section(f"COLLECTING XAI DATA - {precision.upper()}")

        if is_qat:
            model_path = config["model_paths"][precision]
            if not Path(model_path).exists():
                print(f"  Model not found: {model_path}")
                print(f"  Please run QAT training first.")
                continue
            print(f"  Loading QAT model: {model_path}")
            qat_model = ModelManager.load_model(model_path)
            use_fp16 = precision == "fp16"
            lime_res, shap_res, ig_res, occ_res = collect_xai_results(qat_model, precision, samples, use_fp16=use_fp16)

        elif precision == "fp32":
            lime_res, shap_res, ig_res, occ_res = collect_xai_results(base_model, "fp32", samples)

        elif precision == "fp16":
            ptq = PTQQuantizer(base_model.model)
            model_fp16, fp16_time = ptq.quantize_fp16()
            print(f"  FP16 quantization: {fp16_time:.2f}s")
            fp16_model = BaseModel(model_fp16, base_model.tokenizer)
            lime_res, shap_res, ig_res, occ_res = collect_xai_results(fp16_model, "fp16", samples, use_fp16=True)

        elif precision == "int8":
            ptq = PTQQuantizer(base_model.model)
            model_int8, int8_time = ptq.quantize_int8()
            print(f"  INT8 quantization: {int8_time:.2f}s")
            int8_model = BaseModel(model_int8, base_model.tokenizer, device=torch.device("cpu"))
            lime_res, shap_res, ig_res, occ_res = collect_xai_results(int8_model, "int8", samples)

        elif precision == "int4":
            ptq = PTQQuantizer(base_model.model)
            model_int4, int4_time = ptq.quantize_int4()
            print(f"  INT4 quantization: {int4_time:.2f}s")
            int4_model = BaseModel(model_int4, base_model.tokenizer)
            lime_res, shap_res, ig_res, occ_res = collect_xai_results(int4_model, "int4", samples)

        all_lime[precision] = lime_res
        all_shap[precision] = shap_res
        all_ig[precision] = ig_res
        all_occ[precision] = occ_res

        for i in range(len(samples)):
            lr = lime_res[i]
            sr = shap_res[i]
            ir = ig_res[i]
            oc = occ_res[i]
            print(f"\n  Sample {i+1}: Pred={lr['predicted_label']} | Expected={samples[i]['expected']}")
            print(f"    LIME top 3: {', '.join(f'{f[0]}({f[1]:+.3f})' for f in lr['top_features'][:3])}")
            print(f"    SHAP top 3: {', '.join(f'{t[0]}({t[1]:+.3f})' for t in sr['token_importance'][:3])}")
            ig_tokens_scores = [(ir['tokens'][j], ir['scores'][j]) for j in range(len(ir['tokens'])) if ir['tokens'][j] not in ['[CLS]', '[SEP]', '[PAD]']]
            ig_sorted = sorted(ig_tokens_scores, key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"    IG top 3: {', '.join(f'{t}({s:+.3f})' for t, s in ig_sorted)}")
            print(f"    Occlusion top 3: {', '.join(f'{t[0]}({t[1]:+.3f})' for t in oc['token_importance'][:3])}")

    print_section("GENERATING COMPARISON CHARTS")

    print("\n  LIME Comparison:")
    generate_lime_comparison(all_lime, samples, precisions, comparison_dir)

    print("\n  SHAP Comparison:")
    generate_shap_comparison(all_shap, samples, precisions, comparison_dir)

    print("\n  Integrated Gradients Comparison:")
    generate_ig_comparison(all_ig, samples, precisions, comparison_dir)

    print("\n  Occlusion Comparison:")
    generate_occlusion_comparison(all_occ, samples, precisions, comparison_dir)

    print("\n  Prediction Summary:")
    generate_prediction_summary(all_lime, samples, precisions, comparison_dir)


if __name__ == "__main__":
    experiment_key, precisions, num_samples, divergence_samples = interactive_menu()

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
    print("  - lime_comparison_sample_N.png      : LIME feature importance side by side")
    print("  - shap_comparison_sample_N.png      : SHAP token importance side by side")
    print("  - ig_comparison_sample_N.png        : Integrated Gradients attribution side by side")
    print("  - occlusion_comparison_sample_N.png : Occlusion confidence drop side by side")
    print("  - prediction_summary.png            : Prediction correctness table")
