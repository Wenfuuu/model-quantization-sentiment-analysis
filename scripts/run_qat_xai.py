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

from src.config import BASE_DIR, LABELS, DEVICE
from src.models.base import BaseModel
from src.quantization.qat import load_qat_model_for_xai
from src.xai import LIMEExplainer, SHAPExplainer, IntegratedGradientsExplainer, OcclusionExplainer
from src.utils import print_section
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')

QAT_OUTPUT_DIR = BASE_DIR / "src" / "finetune_3label" / "results"

SAMPLE_TEXTS = [
    "Film ini sangat bagus dan menghibur, saya suka sekali",
    "Biasa saja, tidak terlalu bagus tapi juga tidak buruk",
    "Sangat mengecewakan, buang waktu dan uang saya",
]


def get_qat_model_paths():
    eager_paths = {
        "int8": QAT_OUTPUT_DIR / "indobert-qat-int8-smsa" / "hf_model",
        "fp16": QAT_OUTPUT_DIR / "indobert-qat-fp16-smsa" / "hf_model",
        "int4": QAT_OUTPUT_DIR / "indobert-qat-int4-smsa" / "hf_model",
    }
    
    fake_paths = {
        "int8": QAT_OUTPUT_DIR / "indobert-smsa-qat-int8-fake",
        "fp16": QAT_OUTPUT_DIR / "indobert-smsa-qat-fp16-fake",
        "int4": QAT_OUTPUT_DIR / "indobert-smsa-qat-int4-fake",
    }
    
    return eager_paths, fake_paths


def load_model_for_xai(model_path, use_fp16=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(LABELS),
    )
    
    if use_fp16:
        model = model.half()
    
    return BaseModel(model, tokenizer, device=DEVICE)


def run_lime_analysis(base_model, samples, output_dir, precision_name, use_fp16=False):
    print(f"\n  Running LIME for {precision_name.upper()}...")
    lime_explainer = LIMEExplainer(base_model, LABELS, use_fp16=use_fp16)
    
    results = []
    for i, text in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        explanation = lime_explainer.explain(text, num_features=30, num_samples=300)
        predicted_idx = int(np.argmax(explanation.predict_proba))
        label_names = [LABELS[j] for j in sorted(LABELS.keys())]
        features = explanation.as_list(label=predicted_idx)
        results.append({
            "text": text,
            "predicted_label": label_names[predicted_idx],
            "top_features": features[:10],
            "probabilities": {label_names[j]: float(explanation.predict_proba[j]) for j in range(len(label_names))}
        })
    
    output_path = output_dir / f"lime_results_{precision_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"    Results saved to: {output_path}")
    return results


def run_ig_analysis(base_model, samples, output_dir, precision_name):
    print(f"\n  Running Integrated Gradients for {precision_name.upper()}...")
    ig_explainer = IntegratedGradientsExplainer(
        base_model.model, 
        base_model.tokenizer, 
        device=base_model.device
    )
    
    label_names = [LABELS[j] for j in sorted(LABELS.keys())]
    results = []
    
    for i, text in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        ig_result = ig_explainer.explain(text, steps=30)
        results.append({
            "text": text,
            "predicted_label": label_names[ig_result["predicted_class"]],
            "tokens": ig_result["tokens"],
            "scores": ig_result["scores"].tolist() if hasattr(ig_result["scores"], "tolist") else list(ig_result["scores"]),
        })
    
    output_path = output_dir / f"ig_results_{precision_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"    Results saved to: {output_path}")
    return results


def run_shap_analysis(base_model, samples, output_dir, precision_name, use_fp16=False):
    print(f"\n  Running SHAP for {precision_name.upper()}...")
    shap_explainer = SHAPExplainer(base_model, LABELS, use_fp16=use_fp16)
    
    label_names = [LABELS[j] for j in sorted(LABELS.keys())]
    results = []
    
    for i, text in enumerate(samples):
        print(f"    Sample {i+1}/{len(samples)}")
        shap_values = shap_explainer.explain(text, max_evals=200)
        predicted_class = int(np.argmax(shap_explainer.predict_proba(text)))
        
        token_importance = {}
        if hasattr(shap_values[0], 'data') and hasattr(shap_values[0], 'values'):
            data = shap_values[0].data
            values = shap_values[0].values
            for j, token in enumerate(data):
                if isinstance(token, str) and token.strip():
                    token_importance[token] = float(values[j][predicted_class])
        
        sorted_imp = sorted(token_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        results.append({
            "text": text,
            "predicted_label": label_names[predicted_class],
            "token_importance": sorted_imp[:10]
        })
    
    output_path = output_dir / f"shap_results_{precision_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"    Results saved to: {output_path}")
    return results


def run_xai_for_qat_model(model_path, output_dir, precision_name, use_fp16=False):
    print_section(f"XAI Analysis for QAT {precision_name.upper()} Model")
    
    if not Path(model_path).exists():
        print(f"  Model not found: {model_path}")
        print(f"  Please run QAT training first.")
        return None
    
    print(f"  Loading model from: {model_path}")
    base_model = load_model_for_xai(model_path, use_fp16=use_fp16)
    print(f"  Model loaded on device: {base_model.device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lime_results = run_lime_analysis(base_model, SAMPLE_TEXTS, output_dir, precision_name, use_fp16)
    ig_results = run_ig_analysis(base_model, SAMPLE_TEXTS, output_dir, precision_name)
    shap_results = run_shap_analysis(base_model, SAMPLE_TEXTS, output_dir, precision_name, use_fp16)
    
    return {
        "lime": lime_results,
        "ig": ig_results,
        "shap": shap_results,
    }


def interactive_menu():
    print("\n" + "=" * 60)
    print("  QAT XAI (EXPLAINABILITY) RUNNER")
    print("=" * 60)
    
    print("\n  Select QAT Method:")
    print("  [1] Eager (ONNX pipeline models)")
    print("  [2] Fake  (HuggingFace saved models)")
    print("  [3] Both")
    
    method_choice = input("\n  Enter choice (1/2/3): ").strip()
    
    print("\n  Select Quantization Type:")
    print("  [1] INT8")
    print("  [2] FP16")
    print("  [3] INT4")
    print("  [4] All")
    
    quant_choice = input("\n  Enter choice (1/2/3/4): ").strip()
    
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
    elif quant_choice == "3":
        quant_types = ["int4"]
    else:
        quant_types = ["int8", "fp16", "int4"]
    
    return methods, quant_types


def main():
    print_section("QAT XAI (EXPLAINABILITY) ANALYSIS")
    print(f"Device: {DEVICE}")
    
    methods, quant_types = interactive_menu()
    
    eager_paths, fake_paths = get_qat_model_paths()
    
    for method in methods:
        for quant_type in quant_types:
            print("\n" + "=" * 70)
            print(f"Running XAI for {method.upper()} QAT - {quant_type.upper()}")
            print("=" * 70)
            
            if method == "eager":
                model_path = eager_paths.get(quant_type)
                output_dir = QAT_OUTPUT_DIR / f"indobert-qat-{quant_type}-smsa" / "xai_results"
            else:
                model_path = fake_paths.get(quant_type)
                output_dir = QAT_OUTPUT_DIR / f"indobert-smsa-qat-{quant_type}-fake" / "xai_results"
            
            if model_path and Path(model_path).exists():
                use_fp16 = quant_type == "fp16"
                run_xai_for_qat_model(model_path, output_dir, quant_type, use_fp16)
            else:
                print(f"  Model not found: {model_path}")
                print(f"  Please run QAT training first for {method} {quant_type}.")
    
    print("\n" + "=" * 70)
    print("QAT XAI Analysis completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
