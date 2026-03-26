import sys
import time
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
from src.xai import (
    LIMEExplainer,
    SHAPExplainer,
    IntegratedGradientsExplainer,
    OcclusionExplainer,
    build_alignment,
    build_alignment_batch,
    project_subword_to_word,
    fragmentation_report,
    analyze_attention_batch,
    compare_attention_batch,
    aggregate_attention_comparisons,
    save_attention_results,
    save_attention_comparisons,
    attribution_similarity,
    integrated_gradients_tokens,
    gradient_times_input_tokens,
    InsertionDeletionEvaluator,
    layer_cls_similarity,
)
from src.utils import print_section, set_seed

warnings.filterwarnings('ignore')

set_seed(42)

_INT2LABEL = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
_SUBSAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "explainability_subsample_v2.csv"

def select_samples(dataset_samples, num_samples=50):
    import pandas as pd
    import random as _random

    if _SUBSAMPLE_CSV.exists():
        df = pd.read_csv(_SUBSAMPLE_CSV)
        samples = [
            {"text": row["text"], "expected": _INT2LABEL[int(row["true_label"])]}
            for _, row in df.iterrows()
        ]
        print(f"  [select_samples] Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")
        return samples

    print(f"  [select_samples] {_SUBSAMPLE_CSV.name} not found — falling back to stratified random selection")
    by_label = {}
    for sample in dataset_samples:
        label = sample["expected"]
        by_label.setdefault(label, []).append(sample)

    labels = sorted(by_label.keys())
    n_labels = len(labels)
    per_label = max(1, num_samples // n_labels)

    rng = _random.Random(42)
    selected = []
    for label in labels:
        pool = by_label[label]
        rng.shuffle(pool)
        selected.extend(pool[:per_label])

    selected_ids = set(map(id, selected))
    remaining = [s for s in dataset_samples if id(s) not in selected_ids]
    rng.shuffle(remaining)
    for s in remaining:
        if len(selected) >= num_samples:
            break
        selected.append(s)

    return selected[:num_samples]

def _resolve_config(version_key):
    is_qat = _is_qat_experiment(version_key)
    config = QAT_EXPERIMENT_CONFIGS[version_key] if is_qat else EXPERIMENT_CONFIGS[version_key]
    return is_qat, config

def _resolve_samples(config, num_samples, divergence_samples):
    if divergence_samples:
        return divergence_samples
    if config["dataset"] == "smsa":
        all_samples = load_smsa_dataset()
    else:
        all_samples = load_tweets_dataset()
    return select_samples(all_samples, num_samples)

def _save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_models_for_precisions(version_key, precisions):
    is_qat, config = _resolve_config(version_key)
    models = {}

    if is_qat:
        for precision in precisions:
            model_path = config["model_paths"].get(precision)
            if not model_path or not Path(model_path).exists():
                print(f"  Model not found for {precision.upper()}: {model_path}")
                continue
            print(f"  Loading QAT model: {model_path}")
            models[precision] = ModelManager.load_model(model_path)
        return models

    print(f"\nLoading model: {config['model_id']}")
    base_model = ModelManager.load_model(config["model_id"])

    for precision in precisions:
        if precision == "fp32":
            models[precision] = base_model
            continue

        ptq = PTQQuantizer(base_model.model)
        if precision == "fp16":
            model_fp16, fp16_time = ptq.quantize_fp16()
            print(f"  FP16 quantization: {fp16_time:.2f}s")
            models[precision] = BaseModel(model_fp16, base_model.tokenizer)
        elif precision == "int8":
            model_int8, int8_time = ptq.quantize_int8()
            print(f"  INT8 quantization: {int8_time:.2f}s")
            models[precision] = BaseModel(model_int8, base_model.tokenizer, device=torch.device("cpu"))
        elif precision == "int4":
            model_int4, int4_time = ptq.quantize_int4()
            print(f"  INT4 quantization: {int4_time:.2f}s")
            models[precision] = BaseModel(model_int4, base_model.tokenizer)

    return models

def load_divergences(experiment_key):
    if experiment_key in QAT_EXPERIMENT_CONFIGS:
        config = QAT_EXPERIMENT_CONFIGS[experiment_key]
    else:
        config = EXPERIMENT_CONFIGS[experiment_key]
    output_dir = Path(config["output_dir"])
    divergence_path = output_dir / "prediction_divergences.json"

    if not divergence_path.exists():
        return None

    with open(divergence_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def generate_qat_divergences(experiment_key):
    config = QAT_EXPERIMENT_CONFIGS[experiment_key]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["dataset"] == "smsa":
        test_samples = load_smsa_dataset()
    else:
        test_samples = load_tweets_dataset()

    onnx_paths = {}
    for precision, model_path in config["model_paths"].items():
        candidate = Path(model_path).parent / f"model_qat_{precision}.onnx"
        if candidate.exists():
            onnx_paths[precision] = candidate

    if onnx_paths:
        import onnxruntime as ort
        from transformers import AutoTokenizer as _HFTokenizer

        available_precisions = list(onnx_paths.keys())
        print(f"\n  QAT eager ONNX models found: {', '.join(p.upper() for p in available_precisions)}")

        tokenizer_source = next(
            v for v in config["model_paths"].values() if Path(v).exists()
        )
        tokenizer = _HFTokenizer.from_pretrained(tokenizer_source)

        sessions = {}
        for precision, onnx_path in onnx_paths.items():
            print(f"\n  Loading ONNX {precision.upper()}: {onnx_path}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            opts.log_severity_level = 3
            sessions[precision] = ort.InferenceSession(
                str(onnx_path), opts, providers=["CPUExecutionProvider"]
            )

        print(f"\n  Running ONNX inference on {len(test_samples)} samples...")
        all_preds = {p: [] for p in available_precisions}

        for sample in test_samples:
            text = sample["text"].lower().strip()
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="np",
            )
            input_ids = enc["input_ids"].astype(np.int64)
            attention_mask = enc["attention_mask"].astype(np.int64)

            for precision, session in sessions.items():
                logits = session.run(
                    None,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )[0]
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                pred_idx = int(np.argmax(logits, axis=1)[0])
                all_preds[precision].append({
                    "label": LABELS[pred_idx],
                    "confidence": float(probs[0, pred_idx]),
                })

    else:
        available_precisions = []
        models = {}

        for precision, model_path in config["model_paths"].items():
            if Path(model_path).exists():
                print(f"\n  Loading QAT {precision.upper()} model: {model_path}")
                models[precision] = ModelManager.load_model(model_path)
                available_precisions.append(precision)
            else:
                print(f"\n  Skipping {precision.upper()}: model not found at {model_path}")

        if len(available_precisions) < 2:
            print("\n  Need at least 2 QAT models to compare divergences.")
            return None

        print(f"\n  Running inference on {len(test_samples)} samples with {len(available_precisions)} models...")
        all_preds = {p: [] for p in available_precisions}

        for sample in test_samples:
            for precision in available_precisions:
                pred = models[precision].predict(sample["text"], use_fp16=(precision == "fp16"))
                all_preds[precision].append({
                    "label": pred["label"],
                    "confidence": float(pred["confidence"]),
                })

    divergences = []
    for i, sample in enumerate(test_samples):
        preds_by_precision = {p: all_preds[p][i] for p in available_precisions}
        labels_set = set(preds_by_precision[p]["label"] for p in available_precisions)
        if len(labels_set) > 1:
            divergences.append({
                "sample_idx": i,
                "text": sample["text"],
                "expected": sample["expected"],
                "predictions": preds_by_precision,
            })

    divergence_data = {
        "experiment": experiment_key,
        "total_samples": len(test_samples),
        "num_divergences": len(divergences),
        "divergences": divergences,
    }

    divergence_path = output_dir / "prediction_divergences.json"
    with open(divergence_path, "w", encoding="utf-8") as f:
        json.dump(divergence_data, f, ensure_ascii=False, indent=2)

    print(f"\n  Found {len(divergences)} divergent samples out of {len(test_samples)} total")
    print(f"  Saved to: {divergence_path}")

    return divergence_data

class OnnxBaseModel:
    def __init__(self, session, tokenizer, hf_model, device):
        self.session = session
        self.tokenizer = tokenizer
        self.model = hf_model
        self.device = device

    def predict(self, text, use_fp16=False):
        text = text.lower().strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        start_time = time.perf_counter()
        logits = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        end_time = time.perf_counter()

        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        pred_idx = int(np.argmax(logits, axis=1)[0])

        return {
            "label": LABELS[pred_idx],
            "class_id": pred_idx,
            "confidence": float(probs[0, pred_idx]),
            "probabilities": {LABELS[i]: float(probs[0, i]) for i in range(len(LABELS))},
            "inference_time": end_time - start_time,
        }

class _OnnxTorchAdapter:
    """Duck-types OnnxBaseModel as a minimal HF PyTorch model for FaithfulnessEvaluator."""
    def __init__(self, onnx_model: OnnxBaseModel):
        self._session = onnx_model.session
        self.training = False

    def parameters(self):
        yield torch.zeros(1, device="cpu")

    def eval(self):  self.training = False; return self
    def train(self): self.training = True;  return self

    def __call__(self, **kwargs):
        ids  = kwargs["input_ids"].cpu().numpy().astype(np.int64)
        mask = kwargs["attention_mask"].cpu().numpy().astype(np.int64)
        logits_np = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        class _R: logits = None
        r = _R()
        r.logits = torch.tensor(logits_np, dtype=torch.float32)
        return r


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
    from src.xai.integrated_gradients import IG_SUPPORTED_PRECISIONS
    label_names = [LABELS[j] for j in sorted(LABELS.keys())]
    if precision_name not in IG_SUPPORTED_PRECISIONS:
        print(f"    [SKIP] IG not supported for {precision_name.upper()} "
              f"(dynamic quantization breaks autograd). "
              f"Supported precisions: {sorted(IG_SUPPORTED_PRECISIONS)}")
        ig_results = [None] * len(samples)
    else:
        ig_explainer = IntegratedGradientsExplainer(
            base_model.model, base_model.tokenizer,
            device=base_model.device, precision=precision_name,
        )
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

            if ig_data is None:
                ax.text(0.5, 0.5, "IG not available\n(dynamic quantization\nbreaks autograd)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray", style="italic")
                ax.set_title(f"{precision.upper()}\n[not supported]", fontweight='bold')
                ax.set_xlabel("IG attribution")
                continue

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


def run_ste_ig_analysis():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_DIR = _PROJECT_ROOT / "models" / "qat_seed42_with_observers"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        print("  Run: python scripts/prepare_datasets.py  (requires models/fp32_seed42/predictions.csv)")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"],
         "expected": _INT2LABEL[int(row["true_label"])]}
        for _, row in df_sub.iterrows()
    ]
    print(f"\n  Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")

    print(f"\n  Loading FP32 model: {_FP32_DIR}")
    fp32_model = ModelManager.load_model(str(_FP32_DIR))
    fp32_model.model.eval()

    if not _QAT_DIR.exists():
        print(f"  [ERROR] QAT model directory not found: {_QAT_DIR}")
        print("  Run QAT training first (multiseed_qat saves qat_seed42_with_observers/).")
        return
    print(f"\n  Loading QAT model: {_QAT_DIR}")
    qat_model = ModelManager.load_model(str(_QAT_DIR))
    qat_model.model.eval()

    print("\n  Verifying fake quantizers in QAT model:")
    observer_names = [
        name for name, mod in qat_model.model.named_modules()
        if "FakeQuantize" in type(mod).__name__ or "Observer" in type(mod).__name__
    ]
    if observer_names:
        print(f"  Found {len(observer_names)} observer/fake-quantizer module(s):")
        for n in observer_names[:10]:
            print(f"    {n}")
        if len(observer_names) > 10:
            print(f"    ... and {len(observer_names) - 10} more")
    else:
        print("  [WARN] No FakeQuantize/Observer modules found. Gradients may not flow via STE.")

    tokenizer = fp32_model.tokenizer

    ig_fp32 = IntegratedGradientsExplainer(
        fp32_model.model, tokenizer, device=fp32_model.device, precision="fp32"
    )
    ig_qat = IntegratedGradientsExplainer(
        qat_model.model, tokenizer, device=qat_model.device, precision="qat_ste"
    )

    metadata_rows = []
    example_count = 0

    print(f"\n  Running IG + GxI on {len(samples)} samples (n_steps=30, MEAN aggregation)...\n")

    for idx, sample in enumerate(samples):
        sid = sample["sample_id"]
        text = sample["text"]
        expected = sample["expected"]

        alignment = build_alignment(text, tokenizer)

        res_fp32 = ig_fp32.explain(text, steps=30)
        words_fp32, ig_word_scores_fp32 = project_subword_to_word(
            res_fp32["tokens"], res_fp32["scores"], alignment, strategy="mean"
        )

        res_qat = ig_qat.explain(text, steps=30)
        _, ig_word_scores_qat = project_subword_to_word(
            res_qat["tokens"], res_qat["scores"], alignment, strategy="mean"
        )

        np.save(_OUT_DIR / f"ig_fp32_{sid}.npy", ig_word_scores_fp32)
        np.save(_OUT_DIR / f"ig_qat_ste_{sid}.npy", ig_word_scores_qat)

        gxi_tokens_fp32, gxi_sw_fp32 = gradient_times_input_tokens(
            fp32_model.model, tokenizer, text, target=res_fp32["predicted_class"]
        )
        _, gxi_word_scores_fp32 = project_subword_to_word(
            gxi_tokens_fp32, gxi_sw_fp32, alignment, strategy="mean"
        )

        gxi_tokens_qat, gxi_sw_qat = gradient_times_input_tokens(
            qat_model.model, tokenizer, text, target=res_qat["predicted_class"]
        )
        _, gxi_word_scores_qat = project_subword_to_word(
            gxi_tokens_qat, gxi_sw_qat, alignment, strategy="mean"
        )

        np.save(_OUT_DIR / f"gxi_fp32_{sid}.npy", gxi_word_scores_fp32)
        np.save(_OUT_DIR / f"gxi_qat_ste_{sid}.npy", gxi_word_scores_qat)

        metadata_rows.append({
            "sample_id": sid,
            "text": text,
            "expected": expected,
            "predicted_fp32": _INT2LABEL.get(res_fp32["predicted_class"], str(res_fp32["predicted_class"])),
            "predicted_qat_ste": _INT2LABEL.get(res_qat["predicted_class"], str(res_qat["predicted_class"])),
            "n_words": len(words_fp32),
            "ig_npy_fp32": f"ig_fp32_{sid}.npy",
            "ig_npy_qat_ste": f"ig_qat_ste_{sid}.npy",
            "gxi_npy_fp32": f"gxi_fp32_{sid}.npy",
            "gxi_npy_qat_ste": f"gxi_qat_ste_{sid}.npy",
        })

        if example_count < 3:
            example_count += 1
            print(f"  ── Example {example_count}: [{expected}] \"{text[:80]}\"")
            top5 = lambda ws, ss: sorted(zip(ws, ss), key=lambda x: -abs(x[1]))[:5]
            fmt = lambda pairs: "  ".join(f"{w}({s:+.3f})" for w, s in pairs)
            print(f"    IG  FP32  top-5: {fmt(top5(words_fp32, ig_word_scores_fp32))}")
            print(f"    IG  QAT   top-5: {fmt(top5(words_fp32, ig_word_scores_qat))}")
            print(f"    GxI FP32  top-5: {fmt(top5(words_fp32, gxi_word_scores_fp32))}")
            print(f"    GxI QAT   top-5: {fmt(top5(words_fp32, gxi_word_scores_qat))}")
            print()

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(samples)}] done")

    meta_path = _OUT_DIR / "ig_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False, encoding="utf-8")
    print(f"\n  Saved metadata → {meta_path}")
    print(f"  Saved {len(metadata_rows)} × 4 .npy files in {_OUT_DIR}")
    print("  File pattern: ig_fp32_N.npy  ig_qat_ste_N.npy  gxi_fp32_N.npy  gxi_qat_ste_N.npy")


def run_lime_attribution():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _LOG_PATH = _OUT_DIR / "lime_errors.log"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        print("  Run: python scripts/prepare_datasets.py")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"],
         "expected": _INT2LABEL[int(row["true_label"])]}
        for _, row in df_sub.iterrows()
    ]
    print(f"\n  Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")

    if not _FP32_DIR.exists():
        print(f"  [ERROR] FP32 model not found: {_FP32_DIR}")
        return
    print(f"\n  Loading FP32 base: {_FP32_DIR}")
    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()

    def _load_onnx(variant):
        import onnxruntime as ort
        onnx_dir = _MODELS_DIR / f"qat_onnx_{variant}_seed42"
        onnx_file = onnx_dir / f"model_qat_{variant}.onnx"
        if not onnx_file.exists():
            print(f"  [SKIP] ONNX file not found: {onnx_file}")
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxBaseModel(session, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        return BaseModel(m, fp32_base.tokenizer, device=fp32_base.device)

    VARIANTS = [
        ("fp32",          lambda: fp32_base,           False),
        ("ptq_fp16",      lambda: _build_ptq("fp16"),  True),
        ("ptq_int8",      lambda: _build_ptq("int8"),  False),
        ("ptq_int4",      lambda: _build_ptq("int4"),  False),
        ("qat_fp32",      lambda: (ModelManager.load_model(str(_QAT_CLEAN_DIR))
                                   if _QAT_CLEAN_DIR.exists() else None), False),
        ("qat_onnx_fp16", lambda: _load_onnx("fp16"), True),
        ("qat_onnx_int8", lambda: _load_onnx("int8"), False),
        ("qat_onnx_int4", lambda: _load_onnx("int4"), False),
    ]

    n_variants = len(VARIANTS)
    n_samples  = len(samples)
    total_expected = n_variants * n_samples
    global_run = 0
    global_done = 0

    print(f"\n  Running LIME: {n_variants} variants × {n_samples} samples "
          f"× 1000 perturbations ≈ {total_expected} total calls")
    print(f"  Output: {_OUT_DIR}")
    print(f"  Error log: {_LOG_PATH}")

    for vname, load_fn, use_fp16 in VARIANTS:
        print(f"\n{'─'*60}")
        print(f"  [{vname.upper()}]  Loading model...")

        model = load_fn()
        if model is None:
            print(f"  [SKIP] {vname} — model or ONNX file not found.")
            global_run += n_samples
            continue

        lime_explainer = LIMEExplainer(model, LABELS, use_fp16=use_fp16, random_state=42)

        for sample in samples:
            global_run += 1
            sid  = sample["sample_id"]
            text = sample["text"]
            out_path = _OUT_DIR / f"lime_{vname}_{sid}.npy"

            if out_path.exists():
                global_done += 1
                if global_run % 50 == 0:
                    print(f"  [progress] {global_run}/{total_expected} "
                          f"({global_done} saved, {global_run - global_done} skipped/error)")
                continue

            try:
                n_words = len(text.split())
                exp = lime_explainer.explain(
                    text,
                    num_features=max(n_words, 10),
                    num_samples=1000,
                )
                pred_idx = int(np.argmax(exp.predict_proba))
                scores_dict = dict(exp.as_list(label=pred_idx))
                word_scores = np.array(
                    [scores_dict.get(w, 0.0) for w in text.split()],
                    dtype=np.float32,
                )
                np.save(out_path, word_scores)
                global_done += 1
            except Exception as exc:
                with open(_LOG_PATH, "a", encoding="utf-8") as _log:
                    _log.write(f"{vname}\t{sid}\t{type(exc).__name__}: {exc}\n")

            if global_run % 50 == 0:
                pct = 100 * global_run / total_expected
                print(f"  [progress] {global_run}/{total_expected} ({pct:.1f}%)  "
                      f"{global_done} saved  variant={vname}  sample_id={sid}")

    print(f"\n  LIME attribution complete.")
    print(f"  {global_done}/{total_expected} files saved to {_OUT_DIR}")
    print(f"  File pattern: lime_{{variant}}_{{sample_id}}.npy")


def run_shap_attribution():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _LOG_PATH = _OUT_DIR / "shap_errors.log"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        print("  Run: python scripts/prepare_datasets.py")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"],
         "expected": _INT2LABEL[int(row["true_label"])]}
        for _, row in df_sub.iterrows()
    ]
    print(f"\n  Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")

    if not _FP32_DIR.exists():
        print(f"  [ERROR] FP32 model not found: {_FP32_DIR}")
        return
    print(f"\n  Loading FP32 base: {_FP32_DIR}")
    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()

    def _load_onnx(variant):
        import onnxruntime as ort
        onnx_dir = _MODELS_DIR / f"qat_onnx_{variant}_seed42"
        onnx_file = onnx_dir / f"model_qat_{variant}.onnx"
        if not onnx_file.exists():
            print(f"  [SKIP] ONNX file not found: {onnx_file}")
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxBaseModel(session, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        return BaseModel(m, fp32_base.tokenizer, device=fp32_base.device)

    VARIANTS = [
        ("fp32",          lambda: fp32_base,           False),
        ("ptq_fp16",      lambda: _build_ptq("fp16"),  True),
        ("ptq_int8",      lambda: _build_ptq("int8"),  False),
        ("ptq_int4",      lambda: _build_ptq("int4"),  False),
        ("qat_fp32",      lambda: (ModelManager.load_model(str(_QAT_CLEAN_DIR))
                                   if _QAT_CLEAN_DIR.exists() else None), False),
        ("qat_onnx_fp16", lambda: _load_onnx("fp16"), True),
        ("qat_onnx_int8", lambda: _load_onnx("int8"), False),
        ("qat_onnx_int4", lambda: _load_onnx("int4"), False),
    ]

    n_variants = len(VARIANTS)
    n_samples  = len(samples)
    total_expected = n_variants * n_samples
    global_run = 0
    global_done = 0

    print(f"\n  Running SHAP: {n_variants} variants x {n_samples} samples "
          f"x 500 max_evals ~ {total_expected} total calls")
    print(f"  Output: {_OUT_DIR}")
    print(f"  Error log: {_LOG_PATH}")

    for vname, load_fn, use_fp16 in VARIANTS:
        print(f"\n{'-'*60}")
        print(f"  [{vname.upper()}]  Loading model...")

        model = load_fn()
        if model is None:
            print(f"  [SKIP] {vname} - model or ONNX file not found.")
            global_run += n_samples
            continue

        shap_explainer = SHAPExplainer(model, LABELS, use_fp16=use_fp16)

        for sample in samples:
            global_run += 1
            sid  = sample["sample_id"]
            text = sample["text"]
            out_path = _OUT_DIR / f"shap_{vname}_{sid}.npy"

            if out_path.exists():
                global_done += 1
                if global_run % 50 == 0:
                    print(f"  [progress] {global_run}/{total_expected} "
                          f"({global_done} saved, {global_run - global_done} skipped/error)")
                continue

            try:
                shap_values = shap_explainer.explain(text, max_evals=500)
                predicted_class = int(np.argmax(shap_explainer.predict_proba(text)))
                scores_dict = {}
                if hasattr(shap_values[0], 'data') and hasattr(shap_values[0], 'values'):
                    data = shap_values[0].data
                    values = shap_values[0].values
                    for j, token in enumerate(data):
                        if isinstance(token, str) and token.strip():
                            scores_dict[token.strip()] = float(values[j][predicted_class])
                word_scores = np.array(
                    [scores_dict.get(w, 0.0) for w in text.split()],
                    dtype=np.float32,
                )
                np.save(out_path, word_scores)
                global_done += 1
            except Exception as exc:
                with open(_LOG_PATH, "a", encoding="utf-8") as _log:
                    _log.write(f"{vname}\t{sid}\t{type(exc).__name__}: {exc}\n")

            if global_run % 50 == 0:
                pct = 100 * global_run / total_expected
                print(f"  [progress] {global_run}/{total_expected} ({pct:.1f}%)  "
                      f"{global_done} saved  variant={vname}  sample_id={sid}")

    print(f"\n  SHAP attribution complete.")
    print(f"  {global_done}/{total_expected} files saved to {_OUT_DIR}")
    print(f"  File pattern: shap_{{variant}}_{{sample_id}}.npy")


def run_occlusion_attribution():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _LOG_PATH = _OUT_DIR / "occ_errors.log"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        print("  Run: python scripts/prepare_datasets.py")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"],
         "expected": _INT2LABEL[int(row["true_label"])]}
        for _, row in df_sub.iterrows()
    ]
    print(f"\n  Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")

    if not _FP32_DIR.exists():
        print(f"  [ERROR] FP32 model not found: {_FP32_DIR}")
        return
    print(f"\n  Loading FP32 base: {_FP32_DIR}")
    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()

    def _load_onnx(variant):
        import onnxruntime as ort
        onnx_dir = _MODELS_DIR / f"qat_onnx_{variant}_seed42"
        onnx_file = onnx_dir / f"model_qat_{variant}.onnx"
        if not onnx_file.exists():
            print(f"  [SKIP] ONNX file not found: {onnx_file}")
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxBaseModel(session, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        return BaseModel(m, fp32_base.tokenizer, device=fp32_base.device)

    VARIANTS = [
        ("fp32",          lambda: fp32_base,           False),
        ("ptq_fp16",      lambda: _build_ptq("fp16"),  True),
        ("ptq_int8",      lambda: _build_ptq("int8"),  False),
        ("ptq_int4",      lambda: _build_ptq("int4"),  False),
        ("qat_fp32",      lambda: (ModelManager.load_model(str(_QAT_CLEAN_DIR))
                                   if _QAT_CLEAN_DIR.exists() else None), False),
        ("qat_onnx_fp16", lambda: _load_onnx("fp16"), True),
        ("qat_onnx_int8", lambda: _load_onnx("int8"), False),
        ("qat_onnx_int4", lambda: _load_onnx("int4"), False),
    ]

    n_variants = len(VARIANTS)
    n_samples  = len(samples)
    total_expected = n_variants * n_samples
    global_run = 0
    global_done = 0

    print(f"\n  Running Occlusion: {n_variants} variants x {n_samples} samples "
          f"(window_size=1, [MASK] substitution)")
    print(f"  Output: {_OUT_DIR}")
    print(f"  Error log: {_LOG_PATH}")

    for vname, load_fn, use_fp16 in VARIANTS:
        print(f"\n{'-'*60}")
        print(f"  [{vname.upper()}]  Loading model...")

        model = load_fn()
        if model is None:
            print(f"  [SKIP] {vname} - model or ONNX file not found.")
            global_run += n_samples
            continue

        occlusion_explainer = OcclusionExplainer(model, LABELS, use_fp16=use_fp16)

        for sample in samples:
            global_run += 1
            sid  = sample["sample_id"]
            text = sample["text"]
            out_path = _OUT_DIR / f"occ_{vname}_{sid}.npy"

            if out_path.exists():
                global_done += 1
                if global_run % 50 == 0:
                    print(f"  [progress] {global_run}/{total_expected} "
                          f"({global_done} saved, {global_run - global_done} skipped/error)")
                continue

            try:
                occ_result = occlusion_explainer.explain(text, window_size=1)
                word_scores = np.array(
                    [score for _, score in occ_result["all_tokens_ordered"]],
                    dtype=np.float32,
                )
                np.save(out_path, word_scores)
                global_done += 1
            except Exception as exc:
                with open(_LOG_PATH, "a", encoding="utf-8") as _log:
                    _log.write(f"{vname}\t{sid}\t{type(exc).__name__}: {exc}\n")

            if global_run % 50 == 0:
                pct = 100 * global_run / total_expected
                print(f"  [progress] {global_run}/{total_expected} ({pct:.1f}%)  "
                      f"{global_done} saved  variant={vname}  sample_id={sid}")

    print(f"\n  Occlusion attribution complete.")
    print(f"  {global_done}/{total_expected} files saved to {_OUT_DIR}")
    print(f"  File pattern: occ_{{variant}}_{{sample_id}}.npy")

def run_ste_calibration():
    import pandas as pd
    from scipy.stats import spearmanr

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        print(f"  [WARN] {missing_ig} ig_qat_ste_*.npy files missing — run STE-IG first (option [3])")
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


def run_random_baselines():
    import pandas as pd
    from src.xai.random_baseline import RandomAttributionBaseline

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [(int(row["sample_id"]), row["text"]) for _, row in df_sub.iterrows()]

    baseline  = RandomAttributionBaseline(baseline_seed=42, n_draws=30)
    n_draws   = baseline.n_draws
    n_total   = len(samples) * n_draws
    n_done    = 0
    n_skipped = 0

    print(f"\n  Random Baselines: {len(samples)} samples x {n_draws} draws = {n_total} files")
    print(f"  sigma source priority: ig_fp32 > occ_fp32 > shap_fp32")
    print(f"  Output: {_OUT_DIR}")

    for sid, text in samples:
        sigma_source = None
        for prefix in ("ig_fp32", "occ_fp32", "shap_fp32"):
            p = _OUT_DIR / f"{prefix}_{sid}.npy"
            if p.exists():
                sigma_source = p
                break

        if sigma_source is None:
            print(f"  [SKIP] sid={sid}: no sigma source (ig_fp32/occ_fp32/shap_fp32)")
            n_skipped += n_draws
            continue

        real_scores = np.load(sigma_source).astype(np.float64)
        words = text.split()
        L = min(len(words), len(real_scores))
        real_scores = real_scores[:L]
        words_trimmed = words[:L]

        for i in range(n_draws):
            out_path = _OUT_DIR / f"random_{sid}_{i}.npy"
            if out_path.exists():
                n_done += 1
                continue
            result = baseline.draw(text, words_trimmed, real_scores, i)
            np.save(out_path, result.scores.astype(np.float32))
            n_done += 1

        if n_done % 500 == 0 or n_done + n_skipped == n_total:
            print(f"  [progress] {n_done}/{n_total}  skipped={n_skipped}")

    print(f"\n  Random baselines complete: {n_done}/{n_total} files saved  skipped={n_skipped}")
    print(f"  File pattern: random_{{sample_id}}_{{run}}.npy")

def run_cross_seed_verification():
    import pandas as pd
    import onnxruntime as ort
    from scipy.stats import spearmanr

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _MODELS_DIR    = _PROJECT_ROOT / "models"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"]}
        for _, row in df_sub.iterrows()
    ][:20]
    print(f"\n  Cross-seed verification: {len(samples)} samples, seeds [42, 123, 456]")

    all_rows = []

    for seed in [42, 123, 456]:
        print(f"\n  [{seed}]  loading models...")

        fp32_model = ModelManager.load_model(str(_MODELS_DIR / f"fp32_seed{seed}"))
        fp32_model.model.eval()

        ptq_q = PTQQuantizer(fp32_model.model)
        ptq_m, _ = ptq_q.quantize_int4()
        ptq_model = BaseModel(ptq_m, fp32_model.tokenizer, device=fp32_model.device)

        onnx_file = _MODELS_DIR / f"qat_onnx_int4_seed{seed}" / "model_qat_int4.onnx"
        if not onnx_file.exists():
            print(f"  [SKIP] {onnx_file} not found")
            qat_model = None
        else:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            opts.log_severity_level = 3
            session = ort.InferenceSession(
                str(onnx_file), opts, providers=["CPUExecutionProvider"]
            )
            qat_model = OnnxBaseModel(session, fp32_model.tokenizer, None, torch.device("cpu"))

        fp32_occ = OcclusionExplainer(fp32_model, LABELS, use_fp16=False)
        ptq_occ  = OcclusionExplainer(ptq_model,  LABELS, use_fp16=False)
        qat_occ  = OcclusionExplainer(qat_model,  LABELS, use_fp16=False) if qat_model else None

        for sample in samples:
            sid  = sample["sample_id"]
            text = sample["text"]

            sfx = "" if seed == 42 else f"_s{seed}"
            fp32_path = _OUT_DIR / f"occ_fp32{sfx}_{sid}.npy"
            ptq_path  = _OUT_DIR / f"occ_ptq_int4{sfx}_{sid}.npy"
            qat_path  = _OUT_DIR / f"occ_qat_onnx_int4{sfx}_{sid}.npy"

            if not fp32_path.exists():
                r = fp32_occ.explain(text, window_size=1)
                np.save(fp32_path, np.array([s for _, s in r["all_tokens_ordered"]], dtype=np.float32))
            if not ptq_path.exists():
                r = ptq_occ.explain(text, window_size=1)
                np.save(ptq_path,  np.array([s for _, s in r["all_tokens_ordered"]], dtype=np.float32))
            if qat_occ and not qat_path.exists():
                r = qat_occ.explain(text, window_size=1)
                np.save(qat_path,  np.array([s for _, s in r["all_tokens_ordered"]], dtype=np.float32))

            if not fp32_path.exists():
                print(f"  [SKIP] sid={sid} seed={seed}: fp32 missing")
                continue

            fp32_s = np.load(fp32_path).astype(np.float64)

            if ptq_path.exists():
                ptq_s = np.load(ptq_path).astype(np.float64)
                L = min(len(fp32_s), len(ptq_s))
                rho, _ = spearmanr(fp32_s[:L], ptq_s[:L])
                all_rows.append({"seed": seed, "sample_id": sid,
                                  "comparison": "fp32_vs_ptq_int4", "spearman_rho": float(rho)})

            if qat_path.exists():
                qat_s = np.load(qat_path).astype(np.float64)
                L = min(len(fp32_s), len(qat_s))
                rho, _ = spearmanr(fp32_s[:L], qat_s[:L])
                all_rows.append({"seed": seed, "sample_id": sid,
                                  "comparison": "fp32_vs_qat_onnx_int4", "spearman_rho": float(rho)})

        n_seed = len([r for r in all_rows if r["seed"] == seed])
        print(f"  seed={seed}: {n_seed} correlation rows computed")

    if not all_rows:
        print("  [WARN] No rows computed.")
        return

    df = pd.DataFrame(all_rows)
    per_path = _RES_DIR / "cross_seed_verification.csv"
    df.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df)} rows -> {per_path}")

    seed42_means = {
        cmp: df[(df["seed"] == 42) & (df["comparison"] == cmp)]["spearman_rho"].mean()
        for cmp in df["comparison"].unique()
    }
    summary_rows = []
    for (cmp, seed), grp in df.groupby(["comparison", "seed"]):
        mean_rho = grp["spearman_rho"].mean()
        std_rho  = grp["spearman_rho"].std()
        ref      = seed42_means.get(cmp, float("nan"))
        delta    = abs(mean_rho - ref)
        verdict  = "CONSISTENT" if delta <= 0.10 else "VARIABLE"
        summary_rows.append({"comparison": cmp, "seed": seed,
                              "mean_rho": round(mean_rho, 4), "std_rho": round(std_rho, 4),
                              "delta_vs_seed42": round(delta, 4), "verdict": verdict})

    df_sum = pd.DataFrame(summary_rows)
    sum_path = _RES_DIR / "cross_seed_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary -> {sum_path}\n")
    for _, row in df_sum.iterrows():
        print(f"  {row['comparison']:30s}  seed={row['seed']}  "
              f"mean_rho={row['mean_rho']:.3f}  delta={row['delta_vs_seed42']:.3f}  -> {row['verdict']}")

def run_stability_analysis():
    import pandas as pd
    from scipy.stats import wilcoxon as scipy_wilcoxon
    from scipy import stats as scipy_stats
    from src.evaluation.explanation_drift import (
        spearman_rank_correlation, top_k_jaccard, bootstrap_mean_ci,
    )

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [(int(row["sample_id"]), row["text"]) for _, row in df_sub.iterrows()]

    VARIANTS_7 = ["ptq_fp16", "ptq_int8", "ptq_int4", "qat_fp32",
                  "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4"]
    METHODS    = ["lime", "occ", "shap"]

    print(f"\n  Stability analysis: {len(samples)} samples, "
          f"{len(METHODS)} methods x {len(VARIANTS_7)} variants")
    print(f"  Loading from: {_OUT_DIR}")

    per_rows = []

    def _load_pair(method, vname, sid, text):
        fp32_path = _OUT_DIR / f"{method}_fp32_{sid}.npy"
        var_path  = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
        if not fp32_path.exists() or not var_path.exists():
            return None
        words   = text.split()
        fp32_s  = np.load(fp32_path).astype(np.float64)
        var_s   = np.load(var_path).astype(np.float64)
        L = min(len(words), len(fp32_s), len(var_s))
        return words[:L], fp32_s[:L].tolist(), var_s[:L].tolist()

    for method in METHODS:
        for vname in VARIANTS_7:
            n_found = 0
            for sid, text in samples:
                pair = _load_pair(method, vname, sid, text)
                if pair is None:
                    continue
                words_l, fp32_l, var_l = pair
                rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l, var_l)
                j3  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=3)
                j5  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=5)
                j10 = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=10)
                per_rows.append({"method": method, "variant": vname, "sample_id": sid,
                                  "spearman_rho": rho,
                                  "jaccard_k3": j3, "jaccard_k5": j5, "jaccard_k10": j10})
                n_found += 1
            if n_found:
                print(f"  {method:5s} x {vname:20s}: {n_found} samples")

    ig_found = sum(1 for sid, _ in samples if (_OUT_DIR / f"ig_fp32_{sid}.npy").exists())
    if ig_found > 0:
        for sid, text in samples:
            pair = _load_pair("ig", "qat_ste", sid, text)
            if pair is None:
                continue
            words_l, fp32_l, var_l = pair
            rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l, var_l)
            j3  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=3)
            j5  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=5)
            j10 = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=10)
            per_rows.append({"method": "ig", "variant": "qat_ste", "sample_id": sid,
                              "spearman_rho": rho,
                              "jaccard_k3": j3, "jaccard_k5": j5, "jaccard_k10": j10})
        print(f"  ig    x qat_ste             : {ig_found} samples")

    if not per_rows:
        print("  [WARN] No .npy pairs found. Run LIME/OCC/SHAP steps first.")
        return

    df_per = pd.DataFrame(per_rows)
    per_path = _RES_DIR / "stability_perSample.csv"
    df_per.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df_per)} rows -> {per_path}")

    n_comparisons    = df_per.groupby(["method", "variant"]).ngroups
    alpha_bonferroni = 0.05 / n_comparisons

    sum_rows = []
    for (method, vname), grp in df_per.groupby(["method", "variant"]):
        rhos = grp["spearman_rho"].dropna().tolist()
        j5s  = grp["jaccard_k5"].dropna().tolist()
        if len(rhos) < 2:
            continue

        rho_lo, rho_hi = bootstrap_mean_ci(rhos)
        j5_lo,  j5_hi  = bootstrap_mean_ci(j5s)

        diffs = [r - 1.0 for r in rhos]
        try:
            w_stat, w_p = scipy_wilcoxon(diffs, alternative="less")
            z         = scipy_stats.norm.ppf(w_p)
            effect_r  = abs(z) / np.sqrt(len(rhos))
        except ValueError:
            w_stat = w_p = effect_r = float("nan")

        sum_rows.append({
            "method": method, "variant": vname, "n": len(rhos),
            "mean_rho":     round(float(np.mean(rhos)), 4),
            "std_rho":      round(float(np.std(rhos)),  4),
            "ci95_lo_rho":  round(rho_lo, 4),
            "ci95_hi_rho":  round(rho_hi, 4),
            "mean_j5":      round(float(np.mean(j5s)), 4),
            "ci95_lo_j5":   round(j5_lo,  4),
            "ci95_hi_j5":   round(j5_hi,  4),
            "wilcoxon_stat":          round(w_stat,    4),
            "wilcoxon_p":             round(w_p,       6),
            "bonferroni_alpha":        round(alpha_bonferroni, 6),
            "significant_bonferroni":  bool(w_p < alpha_bonferroni),
            "effect_size_r":           round(effect_r,  4),
        })

    df_sum = pd.DataFrame(sum_rows)
    sum_path = _RES_DIR / "stability_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary ({n_comparisons} comparisons, "
          f"Bonferroni alpha={alpha_bonferroni:.5f}) -> {sum_path}")

    floor_rows = []
    for method in METHODS:
        for sid, text in samples:
            fp32_path  = _OUT_DIR / f"{method}_fp32_{sid}.npy"
            rand_files = sorted(_OUT_DIR.glob(f"random_{sid}_*.npy"))
            if not fp32_path.exists() or not rand_files:
                continue
            words  = text.split()
            fp32_s = np.load(fp32_path).astype(np.float64)
            L      = min(len(words), len(fp32_s))
            fp32_l = fp32_s[:L].tolist()
            words_l = words[:L]
            rho_vals, j5_vals = [], []
            for rf in rand_files:
                rand_s = np.load(rf).astype(np.float64)
                Lr = min(L, len(rand_s))
                rand_l = rand_s[:Lr].tolist()
                rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l[:Lr], rand_l)
                j5     = top_k_jaccard(words_l, fp32_l, words_l[:Lr], rand_l, k=5)
                rho_vals.append(rho)
                j5_vals.append(j5)
            floor_rows.append({"method": method, "sample_id": sid,
                                "floor_mean_rho": round(float(np.nanmean(rho_vals)), 4),
                                "floor_mean_j5":  round(float(np.nanmean(j5_vals)),  4)})

    df_floor = pd.DataFrame(floor_rows)
    floor_path = _RES_DIR / "stability_random_floor.csv"
    df_floor.to_csv(floor_path, index=False, encoding="utf-8")
    print(f"  Saved random floor -> {floor_path}")

    # Print verdict
    print(f"\n  {'method':5s}  {'variant':20s}  {'rho':>6s} [95% CI]        "
          f"{'J@5':>5s}  {'r':>5s}  sig?")
    for _, row in df_sum.sort_values(["method", "mean_rho"]).iterrows():
        sig = "*" if row["significant_bonferroni"] else " "
        print(f"  {row['method']:5s}  {row['variant']:20s}  "
              f"{row['mean_rho']:6.3f} [{row['ci95_lo_rho']:.3f},{row['ci95_hi_rho']:.3f}]  "
              f"{row['mean_j5']:5.3f}  {row['effect_size_r']:5.3f}  {sig}")

    compute_power_analysis()

def run_faithfulness_evaluation():
    import pandas as pd
    from src.evaluation.faithfulness import FaithfulnessEvaluator

    _PROJECT_ROOT  = Path(__file__).resolve().parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"
    _FP32_DIR      = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR    = _PROJECT_ROOT / "models"

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = list(df_sub.itertuples(index=False))

    print(f"\n  Loading FP32 base: {_FP32_DIR}")
    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()

    def _load_onnx(variant):
        import onnxruntime as ort
        onnx_dir  = _MODELS_DIR / f"qat_onnx_{variant}_seed42"
        onnx_file = onnx_dir / f"model_qat_{variant}.onnx"
        if not onnx_file.exists():
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        return OnnxBaseModel(session, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        return BaseModel(m, fp32_base.tokenizer, device=fp32_base.device)

    VARIANTS = [
        ("fp32",          lambda: fp32_base,           False),
        ("ptq_fp16",      lambda: _build_ptq("fp16"),  True),
        ("ptq_int8",      lambda: _build_ptq("int8"),  False),
        ("ptq_int4",      lambda: _build_ptq("int4"),  False),
        ("qat_fp32",      lambda: (ModelManager.load_model(str(_QAT_CLEAN_DIR))
                                   if _QAT_CLEAN_DIR.exists() else None), False),
        ("qat_onnx_fp16", lambda: _load_onnx("fp16"), True),
        ("qat_onnx_int8", lambda: _load_onnx("int8"), False),
        ("qat_onnx_int4", lambda: _load_onnx("int4"), False),
    ]

    METHODS = ["lime", "occ", "shap"]
    per_rows = []

    for method in METHODS:
        for vname, load_fn, _ in VARIANTS:
            model = load_fn()
            if model is None:
                print(f"  [SKIP] {method} x {vname}")
                continue

            if isinstance(model, OnnxBaseModel):
                infer_model = _OnnxTorchAdapter(model)
                device = torch.device("cpu")
            else:
                infer_model = model.model
                device = model.device

            evaluator = FaithfulnessEvaluator(
                infer_model, fp32_base.tokenizer, device=device, k_values=(5,)
            )

            n_found = 0
            for sample in samples:
                sid  = int(sample.sample_id)
                text = sample.text
                npy_path = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
                if not npy_path.exists():
                    continue
                words  = text.split()
                scores = np.load(npy_path).astype(np.float64)
                L = min(len(words), len(scores))
                if L < 1:
                    continue
                try:
                    result = evaluator.evaluate(
                        text, words[:L], scores[:L].tolist(),
                        method=method, precision=vname, token_level="word",
                    )
                    fk = result.per_k[5]
                    per_rows.append({
                        "method": method, "variant": vname, "sample_id": sid,
                        "orig_conf": round(fk.original_confidence, 4),
                        "suff":      fk.sufficiency       if not np.isnan(fk.sufficiency)       else float("nan"),
                        "comp":      fk.comprehensiveness if not np.isnan(fk.comprehensiveness) else float("nan"),
                        "n_content": fk.n_content_tokens,
                        "k":         fk.k,
                    })
                    n_found += 1
                except Exception as exc:
                    print(f"  [ERR] {method} {vname} sid={sid}: {exc}")

            print(f"  {method:5s} x {vname:20s}: {n_found} samples")

    if not per_rows:
        print("  [WARN] No .npy files found. Run LIME/OCC/SHAP steps first.")
        return

    df_per = pd.DataFrame(per_rows)
    per_path = _RES_DIR / "faithfulness_perSample.csv"
    df_per.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df_per)} rows -> {per_path}")

    sum_rows = []
    for (method, vname), grp in df_per.groupby(["method", "variant"]):
        sum_rows.append({
            "method": method, "variant": vname, "n": len(grp),
            "mean_suff": round(float(grp["suff"].dropna().mean()), 4),
            "std_suff":  round(float(grp["suff"].dropna().std()),  4),
            "mean_comp": round(float(grp["comp"].dropna().mean()), 4),
            "std_comp":  round(float(grp["comp"].dropna().std()),  4),
        })
    df_sum = pd.DataFrame(sum_rows)
    sum_path = _RES_DIR / "faithfulness_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary -> {sum_path}")

    print(f"\n  {'method':5s}  {'variant':20s}  {'mean_suff':>9s}  {'mean_comp':>9s}")
    for _, row in df_sum.sort_values(["method", "variant"]).iterrows():
        print(f"  {row['method']:5s}  {row['variant']:20s}  "
              f"{row['mean_suff']:9.4f}  {row['mean_comp']:9.4f}")

def compute_power_analysis():
    import math
    per_path = _RES_DIR / "stability_perSample.csv"
    if not per_path.exists():
        print("  [WARN] stability_perSample.csv not found. Run Stability Analysis first.")
        return

    df = pd.read_csv(per_path)
    # Z_alpha/2 (0.05 two-tailed) + Z_beta (0.80 power) = 1.960 + 0.842 = 2.802
    # one-sided Wilcoxon-equivalent: Z_0.05 + Z_0.20 = 1.645 + 0.842 = 2.487
    Z = 2.487
    n = 50
    power_rows = []
    print(f"\n  Power analysis (alpha=0.05, power=0.80, n={n}):")
    print(f"  {'method':5s}  {'observed_sd':>11s}  {'min_detectable_delta_rho':>24s}")
    for method, grp in df.groupby("method"):
        sd = float(grp["spearman_rho"].dropna().std())
        delta = Z * sd / math.sqrt(n)
        power_rows.append({"method": method, "observed_sd": round(sd, 4),
                            "min_detectable_delta_rho": round(delta, 4)})
        print(f"  {method:5s}  {sd:11.4f}  {delta:24.4f}")

    out_path = _RES_DIR / "power_analysis.csv"
    pd.DataFrame(power_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved -> {out_path}")

def interactive_menu():
    print("\n" + "=" * 60)
    print("  XAI ANALYSIS RUNNER")
    print("=" * 60)

    print("\n  Select Quantization Method:")
    print("  [1] PTQ (Post-Training Quantization)")
    print("  [2] QAT (Quantization-Aware Training)")
    print("  [3] STE-IG (LayerIntegratedGradients: FP32 vs QAT-with-observers)")
    print("  [4] LIME Attribution (all 8 seed-42 variants, 1000 samples, resumable)")
    print("  [5] SHAP Attribution (all 8 seed-42 variants, max_evals=500, resumable)")
    print("  [6] Occlusion Attribution (all 8 seed-42 variants, window=1, resumable)")
    print("  [7] STE Calibration (Spearman: ig_qat_ste vs occ_qat_onnx_int8/int4, first 20 samples)")
    print("  [8] Random Baselines (30 draws x 50 samples, sigma from fp32 attributions)")
    print("  [9] Cross-seed Verification (Occlusion seeds 123/456 vs 42, first 20 samples)")
    print("  [10] Stability Analysis (FP32 vs 7 variants, Spearman+Jaccard+Bootstrap+Bonferroni)")
    print("  [11] Faithfulness Evaluation (Suff+Comp k=5, all 8 variants x LIME/OCC/SHAP)")

    method_choice = input("\n  Enter choice (1-11): ").strip()

    if method_choice == "2":
        return _qat_menu()

    if method_choice == "3":
        return "ste_ig", [], 50, None

    if method_choice == "4":
        return "lime_attr", [], 50, None

    if method_choice == "5":
        return "shap_attr", [], 50, None

    if method_choice == "6":
        return "occ_attr", [], 50, None

    if method_choice == "7":
        return "ste_cal", [], 20, None

    if method_choice == "8":
        return "rand_base", [], 50, None

    if method_choice == "9":
        return "cross_seed", [], 20, None

    if method_choice == "10":
        return "stability", [], 50, None

    if method_choice == "11":
        return "faithfulness", [], 50, None

    print("\n  Select Model:")
    print("  [1] Original IndoBERT (indobenchmark/indobert-base-p2)")
    print("  [2] Finetuned IndoBERT (indobert-fp32-smsa-3label)")

    model_choice = input("\n  Enter choice (1/2): ").strip()

    model = "original" if model_choice == "1" else "finetuned"
    experiment_key = f"{model}_smsa"

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
                    preds_str = "  ".join(f"{p.upper()}={d['predictions'][p]['label']}({d['predictions'][p]['confidence']*100:.4f}%)" for p in all_precs)
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

        num_samples_str = input("\n  Number of samples to explain (default 50): ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 50

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
    experiment_key = "qat_eager_smsa"

    print("\n  Sample Selection Mode:")
    print("  [1] Auto-select by label diversity")
    print("  [2] From prediction divergences (find where quantizations disagree)")

    sample_mode = input("\n  Enter choice (1/2): ").strip()

    if sample_mode == "2":
        div_data = load_divergences(experiment_key)

        if div_data is None:
            print("\n  No divergence file found. Generating by comparing QAT models...")
            div_data = generate_qat_divergences(experiment_key)

        if div_data is None:
            print("\n  Cannot generate divergences (models not found). Switching to auto-select.")
            sample_mode = "1"
        else:
            divergences = div_data["divergences"]

            if not divergences:
                print("\n  No divergences found - all QAT models agree on every sample!")
                print("  Switching to auto-select mode.")
                sample_mode = "1"
            else:
                available_precisions = list(divergences[0]["predictions"].keys())

                print(f"\n  Found {len(divergences)} divergent samples (out of {div_data['total_samples']} total)")
                print(f"  Models compared: {', '.join(p.upper() for p in available_precisions)}")
                print(f"\n  Divergent samples:")

                for idx, d in enumerate(divergences, 1):
                    preds_str = "  ".join(
                        f"{p.upper()}={d['predictions'][p]['label']}({d['predictions'][p]['confidence']*100:.4f}%)"
                        for p in available_precisions
                    )
                    print(f"\n  [{idx}] Sample #{d['sample_idx']+1}: Expected={d['expected']}")
                    print(f"      {preds_str}")
                    print(f"      \"{d['text']}\"")

                select_str = input(f"\n  Select samples: [A]ll or enter numbers (e.g., 1,3,5): ").strip()

                if select_str.upper() == "A" or select_str == "":
                    selected_divs = divergences
                else:
                    indices = [int(x.strip()) - 1 for x in select_str.split(",")]
                    selected_divs = [divergences[i] for i in indices if 0 <= i < len(divergences)]

                divergence_samples = [{"text": d["text"], "expected": d["expected"]} for d in selected_divs]
                precisions = available_precisions

                print(f"\n  Selected {len(divergence_samples)} divergent samples for XAI analysis")
                print(f"  Precisions: {', '.join(p.upper() for p in precisions)}")

                return experiment_key, precisions, len(divergence_samples), divergence_samples

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
    qat_hf_model = None

    for precision in precisions:
        print_section(f"COLLECTING XAI DATA - {precision.upper()}")

        if is_qat:
            model_path = config["model_paths"][precision]
            if not Path(model_path).exists():
                print(f"  Model not found: {model_path}")
                print(f"  Please run QAT training first.")
                continue

            onnx_path = Path(model_path).parent / f"model_qat_{precision}.onnx"
            if onnx_path.exists():
                import onnxruntime as ort
                print(f"  Loading QAT ONNX {precision.upper()}: {onnx_path}")
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                opts.log_severity_level = 3
                session = ort.InferenceSession(
                    str(onnx_path), opts, providers=["CPUExecutionProvider"]
                )
                if qat_hf_model is None:
                    print(f"  Loading HF model for IG: {model_path}")
                    qat_hf_model = ModelManager.load_model(model_path, device=torch.device("cpu"))
                onnx_model = OnnxBaseModel(
                    session, qat_hf_model.tokenizer, qat_hf_model.model, torch.device("cpu")
                )
                lime_res, shap_res, ig_res, occ_res = collect_xai_results(onnx_model, precision, samples)
            else:
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
            ir = ig_res[i]  # may be None for INT8/INT4
            oc = occ_res[i]
            print(f"\n  Sample {i+1}: Pred={lr['predicted_label']} | Expected={samples[i]['expected']}")
            print(f"    LIME top 3: {', '.join(f'{f[0]}({f[1]:+.3f})' for f in lr['top_features'][:3])}")
            print(f"    SHAP top 3: {', '.join(f'{t[0]}({t[1]:+.3f})' for t in sr['token_importance'][:3])}")
            if ir is not None:
                ig_tokens_scores = [
                    (ir['tokens'][j], ir['scores'][j])
                    for j in range(len(ir['tokens']))
                    if ir['tokens'][j] not in ['[CLS]', '[SEP]', '[PAD]']
                ]
                ig_sorted = sorted(ig_tokens_scores, key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"    IG top 3: {', '.join(f'{t}({s:+.3f})' for t, s in ig_sorted)}")
            else:
                print(f"    IG: [not available for {precision} — gradient-based method requires FP32/FP16]")
            print(f"    Occlusion top 3: {', '.join(f'{t[0]}({t[1]:+.3f})' for t in oc['token_importance'][:3])}")

    # --- Subword aggregation comparison (sum vs mean) for IG --------------------
    # Reviewer note: summing subword attributions inflates scores for
    # morphologically complex tokens (e.g. reduplication, affixation).
    # Mean-pooling is the recommended primary strategy; both are reported
    # so the paper can discuss the difference.
    _ig_precisions_with_data = [p for p in precisions if all_ig.get(p) and all_ig[p][0] is not None]
    if len(_ig_precisions_with_data) >= 2:
        from src.xai.alignment import build_alignment_batch, project_subword_to_word
        _texts = [s["text"] for s in samples]
        # Use tokenizer from the base model (FP32 — same vocab for all variants)
        if not is_qat and "fp32" in precisions:
            _tokenizer = base_model.tokenizer
        else:
            _tokenizer = ModelManager.load_model(config["model_id"]).tokenizer
        _alignments = build_alignment_batch(_texts, _tokenizer, verbose=False)

        _aggregation_report = {}
        for _prec in _ig_precisions_with_data:
            _sum_scores, _mean_scores = [], []
            for _ig_r, _al in zip(all_ig[_prec], _alignments):
                _sw_tokens = _ig_r["tokens"]
                _sw_scores = _ig_r["scores"]
                _, _ws = project_subword_to_word(_sw_tokens, _sw_scores, _al, strategy="sum")
                _, _wm = project_subword_to_word(_sw_tokens, _sw_scores, _al, strategy="mean")
                _sum_scores.append(_ws.tolist())
                _mean_scores.append(_wm.tolist())
            _aggregation_report[_prec] = {"sum": _sum_scores, "mean": _mean_scores}

        _agg_path = comparison_dir / "subword_aggregation_comparison.json"
        _agg_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(_agg_path, "w", encoding="utf-8") as _f:
            _json.dump({
                "note": "sum inflates morphologically complex tokens; mean is preferred for reduplication/affixation",
                "precisions": _ig_precisions_with_data,
                "aggregations": _aggregation_report,
            }, _f, ensure_ascii=False, indent=2)
        print(f"\n  Subword aggregation comparison (sum vs mean) saved: {_agg_path}")
    # ---------------------------------------------------------------------------

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


def run_alignment_diagnostics(version_key, num_samples, divergence_samples=None):
    is_qat, config = _resolve_config(version_key)
    samples = _resolve_samples(config, num_samples, divergence_samples)
    texts = [s["text"] for s in samples]

    if is_qat:
        tokenizer_source = None
        for model_path in config["model_paths"].values():
            if Path(model_path).exists():
                tokenizer_source = model_path
                break
        if tokenizer_source is None:
            print("  No QAT model paths found for tokenizer.")
            return
        tokenizer = ModelManager.load_model(tokenizer_source).tokenizer
    else:
        tokenizer = ModelManager.load_model(config["model_id"]).tokenizer

    print(f"\n  Running word-subword alignment for {len(texts)} samples...")
    alignments = build_alignment_batch(texts, tokenizer, verbose=True)

    report = fragmentation_report(alignments)
    per_sample = []
    for sample, alignment in zip(samples, alignments):
        multi = [
            {"index": idx, "word": word, "subwords": sw}
            for idx, word, sw in alignment.words_with_multiple_subwords()
        ]
        per_sample.append({
            "text": sample["text"],
            "expected": sample["expected"],
            "n_words": alignment.n_words,
            "n_subwords": alignment.n_subwords,
            "avg_fragments": alignment.average_fragmentation(),
            "max_fragments": max(alignment.fragmentation_per_word) if alignment.fragmentation_per_word else 0,
            "words_multi": multi,
        })

    output_dir = Path(config["output_dir"]) / "xai" / "diagnostics"
    output_path = output_dir / "alignment_report.json"
    payload = {
        "experiment": version_key,
        "n_samples": len(samples),
        "summary": report,
        "samples": per_sample,
    }
    _save_json(payload, output_path)
    print(f"  Saved alignment report: {output_path}")


def run_attention_diagnostics(version_key, precisions, num_samples, divergence_samples=None):
    is_qat, config = _resolve_config(version_key)
    samples = _resolve_samples(config, num_samples, divergence_samples)
    texts = [s["text"] for s in samples]

    print("\n  Loading models for attention diagnostics...")
    models = _load_models_for_precisions(version_key, precisions)
    if not models:
        print("  No models available for attention diagnostics.")
        return

    output_dir = Path(config["output_dir"]) / "xai" / "diagnostics" / "attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": version_key,
        "precisions": list(models.keys()),
        "n_samples": len(samples),
        "mode": "divergence" if divergence_samples else "auto",
    }

    available = []
    for precision, model in models.items():
        print(f"\n  Analyzing attention for {precision.upper()}...")
        try:
            results = analyze_attention_batch(
                texts,
                model.model,
                model.tokenizer,
                precision=precision,
                device=model.device,
                keep_layer_rollouts=False,
                verbose=True,
            )
        except RuntimeError as exc:
            print(f"  Skipping {precision.upper()} attention: {exc}")
            continue

        save_attention_results(
            results,
            output_dir / f"attention_{precision}.json",
            metadata=metadata,
        )
        available.append(precision)

    if len(available) < 2:
        print("  Need at least two attention results to compare.")
        return

    base_precision = available[0]
    base_model = models[base_precision]
    for variant_precision in available[1:]:
        variant_model = models[variant_precision]
        print(f"\n  Comparing attention: {base_precision.upper()} vs {variant_precision.upper()}...")
        try:
            comparisons = compare_attention_batch(
                texts,
                base_model.model,
                variant_model.model,
                base_model.tokenizer,
                base_precision=base_precision,
                variant_precision=variant_precision,
                device=base_model.device,
                verbose=True,
            )
        except RuntimeError as exc:
            print(f"  Skipping comparison {variant_precision.upper()}: {exc}")
            continue

        save_attention_comparisons(
            comparisons,
            output_dir / f"attention_compare_{base_precision}_vs_{variant_precision}.json",
            metadata=metadata,
        )
        summary = aggregate_attention_comparisons(comparisons)
        _save_json(
            summary,
            output_dir / f"attention_summary_{base_precision}_vs_{variant_precision}.json",
        )


def run_ig_metrics_diagnostics(version_key, base_precision, variant_precision, num_samples, divergence_samples=None):
    is_qat, config = _resolve_config(version_key)
    samples = _resolve_samples(config, num_samples, divergence_samples)
    texts = [s["text"] for s in samples]

    print("\n  Loading models for IG metrics...")
    models = _load_models_for_precisions(version_key, [base_precision, variant_precision])
    base_model = models.get(base_precision)
    variant_model = models.get(variant_precision)
    if base_model is None or variant_model is None:
        print("  Missing models for the requested precisions.")
        return

    tokenizer = base_model.tokenizer
    base_eval = InsertionDeletionEvaluator(tokenizer, base_model.device)
    variant_eval = InsertionDeletionEvaluator(tokenizer, variant_model.device)

    per_sample = []
    sim_cos = []
    sim_spear = []
    sim_topk = []
    cls_sims = []
    del_auc_base = []
    ins_auc_base = []
    del_auc_var = []
    ins_auc_var = []

    for sample, text in zip(samples, texts):
        sim = attribution_similarity(base_model.model, variant_model.model, tokenizer, text)
        cls_sim = layer_cls_similarity(base_model.model, variant_model.model, tokenizer, text)

        tokens_b, attrs_b = integrated_gradients_tokens(base_model.model, tokenizer, text)
        tokens_v, attrs_v = integrated_gradients_tokens(variant_model.model, tokenizer, text)

        id_base = base_eval.evaluate(base_model.model, tokens_b, attrs_b)
        id_var = variant_eval.evaluate(variant_model.model, tokens_v, attrs_v)

        per_sample.append({
            "text": text,
            "expected": sample["expected"],
            "similarity": sim,
            "cls_similarity": cls_sim,
            "base": {
                "precision": base_precision,
                "tokens": tokens_b,
                "attributions": attrs_b.tolist(),
                "deletion_auc": id_base.deletion_auc,
                "insertion_auc": id_base.insertion_auc,
            },
            "variant": {
                "precision": variant_precision,
                "tokens": tokens_v,
                "attributions": attrs_v.tolist(),
                "deletion_auc": id_var.deletion_auc,
                "insertion_auc": id_var.insertion_auc,
            },
        })

        sim_cos.append(sim["cosine"])
        sim_spear.append(sim["spearman"])
        sim_topk.append(sim["topk_overlap"])
        cls_sims.append(cls_sim)
        del_auc_base.append(id_base.deletion_auc)
        ins_auc_base.append(id_base.insertion_auc)
        del_auc_var.append(id_var.deletion_auc)
        ins_auc_var.append(id_var.insertion_auc)

    cls_sims_arr = np.array(cls_sims, dtype=np.float64) if cls_sims else np.array([])
    cls_mean = cls_sims_arr.mean(axis=0).tolist() if cls_sims_arr.size else []

    summary = {
        "experiment": version_key,
        "base_precision": base_precision,
        "variant_precision": variant_precision,
        "n_samples": len(samples),
        "similarity_mean": {
            "cosine": float(np.mean(sim_cos)) if sim_cos else 0.0,
            "spearman": float(np.mean(sim_spear)) if sim_spear else 0.0,
            "topk_overlap": float(np.mean(sim_topk)) if sim_topk else 0.0,
        },
        "cls_similarity_mean": cls_mean,
        "deletion_auc": {
            "base_mean": float(np.mean(del_auc_base)) if del_auc_base else 0.0,
            "variant_mean": float(np.mean(del_auc_var)) if del_auc_var else 0.0,
        },
        "insertion_auc": {
            "base_mean": float(np.mean(ins_auc_base)) if ins_auc_base else 0.0,
            "variant_mean": float(np.mean(ins_auc_var)) if ins_auc_var else 0.0,
        },
    }

    output_dir = Path(config["output_dir"]) / "xai" / "diagnostics"
    output_path = output_dir / f"ig_metrics_{base_precision}_vs_{variant_precision}.json"
    payload = {"summary": summary, "samples": per_sample}
    _save_json(payload, output_path)
    print(f"  Saved IG metrics: {output_path}")


def run_xai_diagnostics():
    experiment_key, precisions, num_samples, divergence_samples = interactive_menu()

    print("\n" + "=" * 80)
    print(f"STARTING XAI DIAGNOSTICS: {experiment_key}")
    print(f"Precisions: {', '.join(precisions)}")
    if divergence_samples:
        print(f"Mode: Divergence analysis ({len(divergence_samples)} samples)")
    else:
        print(f"Mode: Auto-select ({num_samples} samples)")
    print("=" * 80)

    print("\n  Select Diagnostics:")
    print("  [1] Word-Subword Alignment")
    print("  [2] Attention Diagnostics")
    print("  [3] IG Metrics")
    print("  [4] All")

    diag_choice = input("\n  Enter choice (1/2/3/4): ").strip()

    if diag_choice in ("1", "4"):
        run_alignment_diagnostics(experiment_key, num_samples, divergence_samples)

    if diag_choice in ("2", "4"):
        run_attention_diagnostics(experiment_key, precisions, num_samples, divergence_samples)

    if diag_choice in ("3", "4"):
        if len(precisions) < 2:
            print("\n  IG metrics require at least two precisions.")
        else:
            base_precision = precisions[0]
            variant_precision = precisions[1]
            print(f"\n  IG metrics pair: {base_precision.upper()} vs {variant_precision.upper()}")
            run_ig_metrics_diagnostics(
                experiment_key,
                base_precision,
                variant_precision,
                num_samples,
                divergence_samples,
            )

    print_section("XAI DIAGNOSTICS COMPLETED")


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
