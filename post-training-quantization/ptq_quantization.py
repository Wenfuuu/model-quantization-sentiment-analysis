import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys
import time
import copy
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


VERSIONS = {
    "original_smsa": {
        "model_id": "indobenchmark/indobert-base-p2",
        "dataset": "smsa",
        "output_dir": "./outputs/original-smsa",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "finetuned_smsa": {
        "model_id": "./finetuned-model/indobert-fp32-smsa-3label-finetuned",
        "dataset": "smsa",
        "output_dir": "./outputs/finetuned-smsa",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "original_tweets": {
        "model_id": "indobenchmark/indobert-base-p2",
        "dataset": "tweets",
        "output_dir": "./outputs/original-tweets",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "finetuned_tweets": {
        "model_id": "./finetuned-model/indobert-fp32-smsa-3label-finetuned",
        "dataset": "tweets",
        "output_dir": "./outputs/finetuned-tweets",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
}

LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
QUANTIZATION_DTYPE = torch.qint8
TARGET_LAYERS = {torch.nn.Linear}


def load_smsa_dataset():
    sentiment_map = {"positive": "POSITIVE", "neutral": "NEUTRAL", "negative": "NEGATIVE"}
    df = pd.read_csv("./datasets/test.tsv", sep="\t", header=None, names=["text", "label"])
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]

    samples = []
    for _, row in df.iterrows():
        label = row['label'].strip().lower()
        if label in sentiment_map:
            samples.append({"text": row['text'], "expected": sentiment_map[label]})

    return samples


def load_tweets_dataset():
    file_path = "./datasets/INA_TweetsPPKM_Labeled_Pure.csv"
    df = pd.read_csv(file_path, sep="\t", engine="python")
    df_sample = df.sample(frac=1/20, random_state=42)

    samples = []
    for _, row in df_sample.iterrows():
        try:
            sentiment_id = int(row['sentiment'])
            if sentiment_id in LABELS:
                samples.append({"text": row['Tweet'], "expected": LABELS[sentiment_id]})
        except ValueError:
            continue

    return samples


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def predict(model, text, tokenizer, use_fp16=False):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    if use_fp16:
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.perf_counter()

    logits = outputs.logits.float()
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_class_id = logits.argmax().item()
    confidence = probabilities[predicted_class_id].item()

    return {
        "label": LABELS[predicted_class_id],
        "class_id": predicted_class_id,
        "confidence": confidence,
        "probabilities": {LABELS[i]: prob.item() for i, prob in enumerate(probabilities)},
        "inference_time": end_time - start_time
    }


def evaluate_model(model, samples, tokenizer, num_runs=20, warmup=5, use_fp16=False):
    results = {"predictions": [], "latencies": [], "accuracy": 0, "avg_confidence": 0}
    correct = 0
    total_confidence = 0

    total_inferences = len(samples) * (warmup + num_runs + 1)
    print(f"Total samples: {len(samples)}")
    print(f"Total inference operations: {total_inferences:,}\n")

    pbar = tqdm(samples, desc="Evaluating samples", unit="sample")

    for sample in pbar:
        for _ in range(warmup):
            _ = predict(model, sample["text"], tokenizer, use_fp16=use_fp16)

        sample_latencies = []
        for _ in range(num_runs):
            pred = predict(model, sample["text"], tokenizer, use_fp16=use_fp16)
            sample_latencies.append(pred["inference_time"])

        final_pred = predict(model, sample["text"], tokenizer, use_fp16=use_fp16)

        results["predictions"].append({
            "text": sample["text"],
            "expected": sample["expected"],
            "predicted": final_pred["label"],
            "confidence": final_pred["confidence"],
            "probabilities": final_pred["probabilities"],
            "avg_latency": np.mean(sample_latencies),
            "std_latency": np.std(sample_latencies)
        })

        results["latencies"].extend(sample_latencies)

        if final_pred["label"] == sample["expected"]:
            correct += 1
        total_confidence += final_pred["confidence"]

        current_accuracy = correct / len(results["predictions"])
        pbar.set_postfix({"Accuracy": f"{current_accuracy:.2%}", "Correct": f"{correct}/{len(results['predictions'])}"})

    results["accuracy"] = correct / len(samples)
    results["avg_confidence"] = total_confidence / len(samples)
    results["latency_stats"] = {
        "mean": np.mean(results["latencies"]),
        "std": np.std(results["latencies"]),
        "min": np.min(results["latencies"]),
        "max": np.max(results["latencies"]),
        "median": np.median(results["latencies"])
    }

    return results


class INT4Quantizer:
    def __init__(self, bits=4):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_tensor(self, tensor):
        tensor_float = tensor.float()
        abs_max = tensor_float.abs().max()
        if abs_max == 0:
            return torch.zeros_like(tensor_float, dtype=torch.int8), torch.tensor(1.0)
        scale = abs_max / self.qmax
        quantized = torch.clamp(torch.round(tensor_float / scale), self.qmin, self.qmax)
        quantized = quantized.to(torch.int8)
        return quantized, scale

    def dequantize_tensor(self, quantized, scale):
        return quantized.float() * scale


class INT4Linear(nn.Module):
    def __init__(self, original_linear, quantizer):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        weight_quantized, weight_scale = quantizer.quantize_tensor(original_linear.weight.data)
        self.register_buffer('weight_quantized', weight_quantized)
        self.register_buffer('weight_scale', weight_scale)
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        self.quantizer = quantizer

    def forward(self, x):
        weight_dequant = self.quantizer.dequantize_tensor(self.weight_quantized, self.weight_scale)
        return nn.functional.linear(x, weight_dequant, self.bias)


def quantize_model_int4(model):
    quantizer = INT4Quantizer(bits=4)
    model_int4 = copy.deepcopy(model)

    def replace_linear_with_int4(module, name=''):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                int4_linear = INT4Linear(child, quantizer)
                setattr(module, child_name, int4_linear)
            else:
                replace_linear_with_int4(child, f"{name}.{child_name}" if name else child_name)

    replace_linear_with_int4(model_int4)
    return model_int4


def run_experiment(version_key):
    config = VERSIONS[version_key]
    output_dir = config["output_dir"]
    num_runs = config["num_inference_runs"]
    warmup = config["warmup_runs"]
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"# EXPERIMENT: {version_key}")
    print(f"# Model: {config['model_id']}")
    print(f"# Dataset: {config['dataset']}")
    print(f"# Output: {output_dir}")
    print("#" * 80)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"\nLoading model: {config['model_id']}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"],
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True
    )
    model_fp32.eval()

    print(f"\nModel loaded successfully")
    print(f"  - Architecture: {model_fp32.config.model_type}")
    print(f"  - Hidden Size: {model_fp32.config.hidden_size}")
    print(f"  - Num Labels: {model_fp32.config.num_labels}")
    print(f"  - Vocab Size: {model_fp32.config.vocab_size}")

    total_params, trainable_params = count_parameters(model_fp32)
    print(f"\nTotal Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    if config["dataset"] == "smsa":
        test_samples = load_smsa_dataset()
    else:
        test_samples = load_tweets_dataset()

    print(f"\nPrepared {len(test_samples)} test samples:")
    print(f"  - Positive: {sum(1 for s in test_samples if s['expected'] == 'POSITIVE')}")
    print(f"  - Negative: {sum(1 for s in test_samples if s['expected'] == 'NEGATIVE')}")
    print(f"  - Neutral: {sum(1 for s in test_samples if s['expected'] == 'NEUTRAL')}")

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (FP32 Model)")
    print("=" * 60)

    fp32_results = evaluate_model(model_fp32, test_samples, tokenizer, num_runs=num_runs, warmup=warmup)

    print(f"\nAccuracy:         {fp32_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence:   {fp32_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency:     {fp32_results['latency_stats']['mean']*1000:.2f} ms")
    print(f"Median Latency:   {fp32_results['latency_stats']['median']*1000:.2f} ms")

    fp32_path = os.path.join(output_dir, "model_fp32.pth")
    torch.save(model_fp32.state_dict(), fp32_path)
    fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"\nFP32 model saved: {fp32_path} ({fp32_size_mb:.2f} MB)")

    print("\n" + "=" * 60)
    print("HALF-PRECISION CONVERSION (FP16)")
    print("=" * 60)

    model_fp16 = copy.deepcopy(model_fp32)
    model_fp16 = model_fp16.half()
    model_fp16.eval()

    fp16_path = os.path.join(output_dir, "model_fp16.pth")
    torch.save(model_fp16.state_dict(), fp16_path)
    fp16_size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"FP16 model saved: {fp16_path} ({fp16_size_mb:.2f} MB)")
    print(f"Size Reduction: {(1 - fp16_size_mb/fp32_size_mb)*100:.2f}%")

    fp16_results = evaluate_model(model_fp16, test_samples, tokenizer, num_runs=num_runs, warmup=warmup, use_fp16=True)

    print(f"\nAccuracy:         {fp16_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence:   {fp16_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency:     {fp16_results['latency_stats']['mean']*1000:.2f} ms")

    print("\n" + "=" * 60)
    print("DYNAMIC QUANTIZATION (INT8)")
    print("=" * 60)

    quantization_start = time.perf_counter()
    model_int8 = torch.quantization.quantize_dynamic(model_fp32, TARGET_LAYERS, dtype=QUANTIZATION_DTYPE)
    quantization_time = time.perf_counter() - quantization_start
    print(f"Quantization completed in {quantization_time:.2f} seconds")

    int8_path = os.path.join(output_dir, "model_int8.pth")
    torch.save(model_int8.state_dict(), int8_path)
    int8_size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"INT8 model saved: {int8_path} ({int8_size_mb:.2f} MB)")
    print(f"Size Reduction: {(1 - int8_size_mb/fp32_size_mb)*100:.2f}%")

    int8_results = evaluate_model(model_int8, test_samples, tokenizer, num_runs=num_runs, warmup=warmup)

    print(f"\nAccuracy:         {int8_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence:   {int8_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency:     {int8_results['latency_stats']['mean']*1000:.2f} ms")

    print("\n" + "=" * 60)
    print("INT4 QUANTIZATION")
    print("=" * 60)

    int4_start = time.perf_counter()
    model_int4 = quantize_model_int4(model_fp32)
    model_int4.eval()
    int4_time = time.perf_counter() - int4_start
    print(f"INT4 Quantization completed in {int4_time:.2f} seconds")

    int4_path = os.path.join(output_dir, "model_int4.pth")
    torch.save(model_int4.state_dict(), int4_path)
    int4_size_mb = os.path.getsize(int4_path) / (1024 * 1024)
    print(f"INT4 model saved: {int4_path} ({int4_size_mb:.2f} MB)")
    print(f"Size Reduction: {(1 - int4_size_mb/fp32_size_mb)*100:.2f}%")

    int4_results = evaluate_model(model_int4, test_samples, tokenizer, num_runs=num_runs, warmup=warmup)

    print(f"\nAccuracy:         {int4_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence:   {int4_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency:     {int4_results['latency_stats']['mean']*1000:.2f} ms")

    print("\n" + "=" * 120)
    print("QUANTIZATION COMPARISON SUMMARY (FP32 vs FP16 vs INT8 vs INT4)")
    print("=" * 120)

    comparison_data = {
        "Metric": [
            "Model Size (MB)", "Accuracy (%)", "Avg Confidence (%)",
            "Mean Latency (ms)", "Median Latency (ms)", "Std Latency (ms)"
        ],
        "FP32 (Baseline)": [
            f"{fp32_size_mb:.2f}", f"{fp32_results['accuracy']*100:.2f}",
            f"{fp32_results['avg_confidence']*100:.2f}",
            f"{fp32_results['latency_stats']['mean']*1000:.2f}",
            f"{fp32_results['latency_stats']['median']*1000:.2f}",
            f"{fp32_results['latency_stats']['std']*1000:.2f}"
        ],
        "FP16 (Half)": [
            f"{fp16_size_mb:.2f}", f"{fp16_results['accuracy']*100:.2f}",
            f"{fp16_results['avg_confidence']*100:.2f}",
            f"{fp16_results['latency_stats']['mean']*1000:.2f}",
            f"{fp16_results['latency_stats']['median']*1000:.2f}",
            f"{fp16_results['latency_stats']['std']*1000:.2f}"
        ],
        "INT8 (Quantized)": [
            f"{int8_size_mb:.2f}", f"{int8_results['accuracy']*100:.2f}",
            f"{int8_results['avg_confidence']*100:.2f}",
            f"{int8_results['latency_stats']['mean']*1000:.2f}",
            f"{int8_results['latency_stats']['median']*1000:.2f}",
            f"{int8_results['latency_stats']['std']*1000:.2f}"
        ],
        "INT4 (4-bit)": [
            f"{int4_size_mb:.2f}", f"{int4_results['accuracy']*100:.2f}",
            f"{int4_results['avg_confidence']*100:.2f}",
            f"{int4_results['latency_stats']['mean']*1000:.2f}",
            f"{int4_results['latency_stats']['median']*1000:.2f}",
            f"{int4_results['latency_stats']['std']*1000:.2f}"
        ],
        "FP16 vs FP32": [
            f"{(1 - fp16_size_mb/fp32_size_mb)*100:+.2f}%",
            f"{(fp16_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%",
            f"{(fp16_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.2f}%",
            f"{((fp16_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.2f}%",
            f"{((fp16_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.2f}%",
            "N/A"
        ],
        "INT8 vs FP32": [
            f"{(1 - int8_size_mb/fp32_size_mb)*100:+.2f}%",
            f"{(int8_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%",
            f"{(int8_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.2f}%",
            f"{((int8_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.2f}%",
            f"{((int8_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.2f}%",
            "N/A"
        ],
        "INT4 vs FP32": [
            f"{(1 - int4_size_mb/fp32_size_mb)*100:+.2f}%",
            f"{(int4_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%",
            f"{(int4_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.2f}%",
            f"{((int4_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.2f}%",
            f"{((int4_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.2f}%",
            "N/A"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("=" * 120)

    print("\n" + "=" * 140)
    print("DETAILED PREDICTION COMPARISON (FP32 vs FP16 vs INT8 vs INT4)")
    print("=" * 140)

    prediction_comparison = []
    for i, (fp32_pred, fp16_pred, int8_pred, int4_pred) in enumerate(
        zip(fp32_results["predictions"], fp16_results["predictions"],
            int8_results["predictions"], int4_results["predictions"])
    ):
        fp16_match = "Y" if fp32_pred["predicted"] == fp16_pred["predicted"] else "N"
        int8_match = "Y" if fp32_pred["predicted"] == int8_pred["predicted"] else "N"
        int4_match = "Y" if fp32_pred["predicted"] == int4_pred["predicted"] else "N"
        prediction_comparison.append({
            "#": i+1,
            "Text": fp32_pred["text"][:30] + "...",
            "Expected": fp32_pred["expected"],
            "FP32": fp32_pred["predicted"],
            "FP32%": f"{fp32_pred['confidence']*100:.1f}%",
            "FP16": fp16_pred["predicted"],
            "FP16%": f"{fp16_pred['confidence']*100:.1f}%",
            "FP16=FP32": fp16_match,
            "INT8": int8_pred["predicted"],
            "INT8%": f"{int8_pred['confidence']*100:.1f}%",
            "INT8=FP32": int8_match,
            "INT4": int4_pred["predicted"],
            "INT4%": f"{int4_pred['confidence']*100:.1f}%",
            "INT4=FP32": int4_match
        })

    df_predictions = pd.DataFrame(prediction_comparison)
    print(df_predictions.to_string(index=False))

    fp16_mismatches = sum(1 for p in prediction_comparison if p["FP16=FP32"] == "N")
    int8_mismatches = sum(1 for p in prediction_comparison if p["INT8=FP32"] == "N")
    int4_mismatches = sum(1 for p in prediction_comparison if p["INT4=FP32"] == "N")
    print(f"\nFP16 vs FP32 Consistency: {len(prediction_comparison) - fp16_mismatches}/{len(prediction_comparison)} ({(1-fp16_mismatches/len(prediction_comparison))*100:.1f}%)")
    print(f"INT8 vs FP32 Consistency: {len(prediction_comparison) - int8_mismatches}/{len(prediction_comparison)} ({(1-int8_mismatches/len(prediction_comparison))*100:.1f}%)")
    print(f"INT4 vs FP32 Consistency: {len(prediction_comparison) - int4_mismatches}/{len(prediction_comparison)} ({(1-int4_mismatches/len(prediction_comparison))*100:.1f}%)")

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    t_stat_fp16, p_value_fp16 = stats.ttest_ind(fp32_results["latencies"], fp16_results["latencies"])
    print("\n1. FP16 vs FP32 Latency (Independent t-test)")
    print(f"   FP32 Mean: {np.mean(fp32_results['latencies'])*1000:.4f} ms")
    print(f"   FP16 Mean: {np.mean(fp16_results['latencies'])*1000:.4f} ms")
    print(f"   t-statistic: {t_stat_fp16:.4f}")
    print(f"   p-value: {p_value_fp16:.6f}")
    print(f"   Significant (p < 0.05): {'Yes' if p_value_fp16 < 0.05 else 'No'}")

    t_stat_int8, p_value_int8 = stats.ttest_ind(fp32_results["latencies"], int8_results["latencies"])
    print("\n2. INT8 vs FP32 Latency (Independent t-test)")
    print(f"   FP32 Mean: {np.mean(fp32_results['latencies'])*1000:.4f} ms")
    print(f"   INT8 Mean: {np.mean(int8_results['latencies'])*1000:.4f} ms")
    print(f"   t-statistic: {t_stat_int8:.4f}")
    print(f"   p-value: {p_value_int8:.6f}")
    print(f"   Significant (p < 0.05): {'Yes' if p_value_int8 < 0.05 else 'No'}")

    t_stat_int4, p_value_int4 = stats.ttest_ind(fp32_results["latencies"], int4_results["latencies"])
    print("\n3. INT4 vs FP32 Latency (Independent t-test)")
    print(f"   FP32 Mean: {np.mean(fp32_results['latencies'])*1000:.4f} ms")
    print(f"   INT4 Mean: {np.mean(int4_results['latencies'])*1000:.4f} ms")
    print(f"   t-statistic: {t_stat_int4:.4f}")
    print(f"   p-value: {p_value_int4:.6f}")
    print(f"   Significant (p < 0.05): {'Yes' if p_value_int4 < 0.05 else 'No'}")

    pooled_std_int8 = np.sqrt(((len(fp32_results['latencies'])-1)*np.std(fp32_results['latencies'])**2 +
                          (len(int8_results['latencies'])-1)*np.std(int8_results['latencies'])**2) /
                         (len(fp32_results['latencies']) + len(int8_results['latencies']) - 2))
    cohens_d_int8 = (np.mean(fp32_results['latencies']) - np.mean(int8_results['latencies'])) / pooled_std_int8

    pooled_std_int4 = np.sqrt(((len(fp32_results['latencies'])-1)*np.std(fp32_results['latencies'])**2 +
                          (len(int4_results['latencies'])-1)*np.std(int4_results['latencies'])**2) /
                         (len(fp32_results['latencies']) + len(int4_results['latencies']) - 2))
    cohens_d_int4 = (np.mean(fp32_results['latencies']) - np.mean(int4_results['latencies'])) / pooled_std_int4

    print(f"\n4. Effect Size (Cohen's d)")
    effect_int8 = "negligible" if abs(cohens_d_int8) < 0.2 else "small" if abs(cohens_d_int8) < 0.5 else "medium" if abs(cohens_d_int8) < 0.8 else "large"
    effect_int4 = "negligible" if abs(cohens_d_int4) < 0.2 else "small" if abs(cohens_d_int4) < 0.5 else "medium" if abs(cohens_d_int4) < 0.8 else "large"
    print(f"   INT8 Cohen's d: {cohens_d_int8:.4f} ({effect_int8} effect)")
    print(f"   INT4 Cohen's d: {cohens_d_int4:.4f} ({effect_int4} effect)")

    fp32_confidences = [p["confidence"] for p in fp32_results["predictions"]]
    fp16_confidences = [p["confidence"] for p in fp16_results["predictions"]]
    int8_confidences = [p["confidence"] for p in int8_results["predictions"]]
    int4_confidences = [p["confidence"] for p in int4_results["predictions"]]

    conf_t_stat_fp16, conf_p_value_fp16 = stats.ttest_rel(fp32_confidences, fp16_confidences)
    conf_t_stat_int8, conf_p_value_int8 = stats.ttest_rel(fp32_confidences, int8_confidences)
    conf_t_stat_int4, conf_p_value_int4 = stats.ttest_rel(fp32_confidences, int4_confidences)

    print(f"\n5. Confidence Score Comparison (Paired t-test)")
    print(f"   FP32 Mean Confidence: {np.mean(fp32_confidences)*100:.2f}%")
    print(f"   FP16 Mean Confidence: {np.mean(fp16_confidences)*100:.2f}%")
    print(f"   INT8 Mean Confidence: {np.mean(int8_confidences)*100:.2f}%")
    print(f"   INT4 Mean Confidence: {np.mean(int4_confidences)*100:.2f}%")
    print(f"   FP16 vs FP32 p-value: {conf_p_value_fp16:.6f}")
    print(f"   INT8 vs FP32 p-value: {conf_p_value_int8:.6f}")
    print(f"   INT4 vs FP32 p-value: {conf_p_value_int4:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'PTQ Analysis: {version_key} (FP32 vs FP16 vs INT8 vs INT4)', fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    sizes = [fp32_size_mb, fp16_size_mb, int8_size_mb, int4_size_mb]
    labels_plot = ['FP32\n(Baseline)', 'FP16\n(Half)', 'INT8\n(8-bit)', 'INT4\n(4-bit)']
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
    bars = ax1.bar(labels_plot, sizes, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Model Size (MB)', fontweight='bold')
    ax1.set_title('Model Size Comparison', fontweight='bold')
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{size:.1f} MB', ha='center', va='bottom', fontweight='bold')
    fp16_reduction = (1 - fp16_size_mb/fp32_size_mb) * 100
    int8_reduction = (1 - int8_size_mb/fp32_size_mb) * 100
    int4_reduction = (1 - int4_size_mb/fp32_size_mb) * 100
    ax1.annotate(f'FP16: {fp16_reduction:.1f}% reduction\nINT8: {int8_reduction:.1f}% reduction\nINT4: {int4_reduction:.1f}% reduction',
                 xy=(0.5, 0.25), xycoords='axes fraction',
                 fontsize=9, ha='center', color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax2 = axes[0, 1]
    latency_data = [np.array(fp32_results['latencies'])*1000,
                    np.array(fp16_results['latencies'])*1000,
                    np.array(int8_results['latencies'])*1000,
                    np.array(int4_results['latencies'])*1000]
    bp = ax2.boxplot(latency_data, labels=['FP32', 'FP16', 'INT8', 'INT4'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#9b59b6')
    bp['boxes'][2].set_facecolor('#2ecc71')
    bp['boxes'][3].set_facecolor('#e74c3c')
    ax2.set_ylabel('Inference Latency (ms)', fontweight='bold')
    ax2.set_title('Latency Distribution Comparison', fontweight='bold')

    ax3 = axes[1, 0]
    x = np.arange(len(test_samples))
    width = 0.2
    ax3.bar(x - 1.5*width, [p['confidence']*100 for p in fp32_results['predictions']],
            width, label='FP32', color='#3498db', alpha=0.8)
    ax3.bar(x - 0.5*width, [p['confidence']*100 for p in fp16_results['predictions']],
            width, label='FP16', color='#9b59b6', alpha=0.8)
    ax3.bar(x + 0.5*width, [p['confidence']*100 for p in int8_results['predictions']],
            width, label='INT8', color='#2ecc71', alpha=0.8)
    ax3.bar(x + 1.5*width, [p['confidence']*100 for p in int4_results['predictions']],
            width, label='INT4', color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Sample Index', fontweight='bold')
    ax3.set_ylabel('Confidence (%)', fontweight='bold')
    ax3.set_title('Per-Sample Confidence Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(i+1) for i in x])
    ax3.legend()
    ax3.set_ylim(0, 110)

    ax4 = axes[1, 1]
    metrics = ['Accuracy', 'Avg Confidence', 'Speed Gain']
    fp32_vals = [fp32_results['accuracy']*100, fp32_results['avg_confidence']*100, 100]
    speedup_fp16 = (fp32_results['latency_stats']['mean'] / fp16_results['latency_stats']['mean']) * 100
    speedup_int8 = (fp32_results['latency_stats']['mean'] / int8_results['latency_stats']['mean']) * 100
    speedup_int4 = (fp32_results['latency_stats']['mean'] / int4_results['latency_stats']['mean']) * 100
    fp16_vals = [fp16_results['accuracy']*100, fp16_results['avg_confidence']*100, speedup_fp16]
    int8_vals = [int8_results['accuracy']*100, int8_results['avg_confidence']*100, speedup_int8]
    int4_vals = [int4_results['accuracy']*100, int4_results['avg_confidence']*100, speedup_int4]

    x = np.arange(len(metrics))
    ax4.bar(x - 1.5*width, fp32_vals, width, label='FP32', color='#3498db', alpha=0.8)
    ax4.bar(x - 0.5*width, fp16_vals, width, label='FP16', color='#9b59b6', alpha=0.8)
    ax4.bar(x + 0.5*width, int8_vals, width, label='INT8', color='#2ecc71', alpha=0.8)
    ax4.bar(x + 1.5*width, int4_vals, width, label='INT4', color='#e74c3c', alpha=0.8)
    ax4.set_ylabel('Percentage (%)', fontweight='bold')
    ax4.set_title('Overall Performance Metrics', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'quantization_analysis.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nVisualization saved to: {chart_path}")

    print("\n" + "=" * 80)
    print("RESEARCH CONCLUSIONS")
    print("=" * 80)

    print(f"""
EXPERIMENT SUMMARY
{'='*60}
Model:          {config['model_id']}
Dataset:        {config['dataset']}
Methods:        FP32 (Baseline), FP16 (Half-Precision), INT8 (Dynamic), INT4 (4-bit)
Target Layers:  Linear Layers
Test Samples:   {len(test_samples)}
Inference Runs: {num_runs} per sample

KEY FINDINGS
{'='*60}
1. Model Size Reduction:
   - Original (FP32): {fp32_size_mb:.2f} MB
   - Half-Precision (FP16): {fp16_size_mb:.2f} MB ({(1 - fp16_size_mb/fp32_size_mb)*100:.2f}% reduction)
   - Quantized (INT8): {int8_size_mb:.2f} MB ({(1 - int8_size_mb/fp32_size_mb)*100:.2f}% reduction)
   - Quantized (INT4): {int4_size_mb:.2f} MB ({(1 - int4_size_mb/fp32_size_mb)*100:.2f}% reduction)

2. Accuracy Impact:
   - FP32 Accuracy: {fp32_results['accuracy']*100:.2f}%
   - FP16 Accuracy: {fp16_results['accuracy']*100:.2f}% ({(fp16_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%)
   - INT8 Accuracy: {int8_results['accuracy']*100:.2f}% ({(int8_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%)
   - INT4 Accuracy: {int4_results['accuracy']*100:.2f}% ({(int4_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%)

3. Inference Latency:
   - FP32 Mean: {fp32_results['latency_stats']['mean']*1000:.2f} ms
   - FP16 Mean: {fp16_results['latency_stats']['mean']*1000:.2f} ms ({(fp32_results['latency_stats']['mean']/fp16_results['latency_stats']['mean']):.2f}x speedup)
   - INT8 Mean: {int8_results['latency_stats']['mean']*1000:.2f} ms ({(fp32_results['latency_stats']['mean']/int8_results['latency_stats']['mean']):.2f}x speedup)
   - INT4 Mean: {int4_results['latency_stats']['mean']*1000:.2f} ms ({(fp32_results['latency_stats']['mean']/int4_results['latency_stats']['mean']):.2f}x speedup)

4. Prediction Consistency:
   - FP16 vs FP32: {len(test_samples) - fp16_mismatches}/{len(test_samples)} matches ({(1-fp16_mismatches/len(test_samples))*100:.1f}%)
   - INT8 vs FP32: {len(test_samples) - int8_mismatches}/{len(test_samples)} matches ({(1-int8_mismatches/len(test_samples))*100:.1f}%)
   - INT4 vs FP32: {len(test_samples) - int4_mismatches}/{len(test_samples)} matches ({(1-int4_mismatches/len(test_samples))*100:.1f}%)
""")

    return {
        "version": version_key,
        "fp32_results": fp32_results,
        "fp16_results": fp16_results,
        "int8_results": int8_results,
        "int4_results": int4_results,
        "fp32_size_mb": fp32_size_mb,
        "fp16_size_mb": fp16_size_mb,
        "int8_size_mb": int8_size_mb,
        "int4_size_mb": int4_size_mb,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        for key in selected:
            if key not in VERSIONS:
                print(f"Unknown version: {key}")
                print(f"Available: {list(VERSIONS.keys())}")
                sys.exit(1)
    else:
        selected = list(VERSIONS.keys())

    all_results = {}
    for version_key in selected:
        result = run_experiment(version_key)
        all_results[version_key] = result

    print("\n\n" + "#" * 80)
    print("# ALL EXPERIMENTS COMPLETED")
    print("#" * 80)
    for key, res in all_results.items():
        print(f"\n  {key}:")
        print(f"    Accuracy: FP32={res['fp32_results']['accuracy']*100:.2f}% | FP16={res['fp16_results']['accuracy']*100:.2f}% | INT8={res['int8_results']['accuracy']*100:.2f}% | INT4={res['int4_results']['accuracy']*100:.2f}%")
        print(f"    Size:     FP32={res['fp32_size_mb']:.1f}MB | FP16={res['fp16_size_mb']:.1f}MB | INT8={res['int8_size_mb']:.1f}MB | INT4={res['int4_size_mb']:.1f}MB")
