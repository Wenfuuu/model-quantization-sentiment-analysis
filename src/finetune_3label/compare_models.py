from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
import json
import time
from pathlib import Path

INFERENCE_RUNS = 20
WARMUP_RUNS = 5
SAMPLE_TOTAL = 500
OUTPUT_DIR = "./comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("MODEL COMPARISON: FP32 vs FP16 vs INT8 QAT")
print("="*70)

FP32_MODEL = "./models/indobert-fp32-smsa-3label-finetuned"
FP16_MODEL = "./models/indobert-smsa-qat-fp16-fake"
INT8_MODEL = "./models/indobert-smsa-qat-int8-fake"

label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {0: "positive", 1: "neutral", 2: "negative"}

print("\n[1/8] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FP32_MODEL)
print(f"✓ Tokenizer loaded")

print("\n[2/8] Loading test dataset...")
dataset = load_dataset(
    "csv",
    data_files={"test": "test.tsv"},
    delimiter="\t",
    column_names=["text", "label"]
)

dataset["test"] = dataset["test"].select(range(min(SAMPLE_TOTAL, len(dataset["test"]))))
print(f"✓ Loaded {len(dataset['test'])} test samples")

print("\n[3/8] Preprocessing dataset...")
stopword_factory = StopWordRemoverFactory()
indonesian_stopwords = stopword_factory.get_stop_words()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in indonesian_stopwords]
    return " ".join(words)

def map_labels(df):
    df["label"] = [label2id[label] for label in df["label"]]
    return df

dataset = dataset.map(map_labels, batched=True)
dataset = dataset.map(
    lambda examples: {"text": [preprocess_text(text) for text in examples["text"]]},
    batched=True
)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"✓ Preprocessing complete")

print("\n[4/8] Loading models...")
print("  Loading FP32 (Baseline)...")
model_fp32 = AutoModelForSequenceClassification.from_pretrained(
    FP32_MODEL,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)
model_fp32.eval()
print("  ✓ FP32 loaded")

print("  Loading FP16...")
model_fp16 = AutoModelForSequenceClassification.from_pretrained(FP16_MODEL)
model_fp16.eval()
print("  ✓ FP16 loaded")

print("  Loading INT8 QAT...")
model_int8 = AutoModelForSequenceClassification.from_pretrained(INT8_MODEL)
model_int8.eval()
print("  ✓ INT8 QAT loaded")

# Get model sizes
print("\n[5/8] Calculating model sizes...")
def get_model_size_mb(model, model_path=None):
    """Calculate model size in MB"""
    if model_path and os.path.exists(model_path):
        # Get size from saved files
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                    total_size += os.path.getsize(os.path.join(root, file))
        return total_size / (1024 * 1024)  # Convert to MB
    else:
        # Calculate from model parameters
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

size_fp32 = get_model_size_mb(model_fp32)
size_fp16 = get_model_size_mb(model_fp16, FP16_MODEL)
size_int8 = get_model_size_mb(model_int8, INT8_MODEL)

print(f"  FP32: {size_fp32:.2f} MB")
print(f"  FP16: {size_fp16:.2f} MB")
print(f"  INT8: {size_int8:.2f} MB")

# Evaluation function
def evaluate_model(model, dataset, model_name):
    """Evaluate model and return metrics"""
    print(f"\n  Evaluating {model_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    predictions = []
    confidences = []
    labels = []
    latencies = []
    
    # Warmup runs
    print(f"    Warmup ({WARMUP_RUNS} runs)...")
    for i in range(min(WARMUP_RUNS, len(dataset))):
        inputs = tokenizer(
            dataset[i]["text"] if isinstance(dataset[i]["text"], str) else "",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
    
    # Inference runs
    print(f"    Inference ({len(dataset)} samples, {INFERENCE_RUNS} runs each)...")
    for idx in range(len(dataset)):
        inputs = tokenizer(
            dataset[idx]["text"] if isinstance(dataset[idx]["text"], str) else "",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Multiple inference runs for latency measurement
        sample_latencies = []
        for _ in range(INFERENCE_RUNS):
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            end_time = time.perf_counter()
            sample_latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies.append(np.mean(sample_latencies))
        
        # Get prediction and confidence
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs).item() * 100
        pred = torch.argmax(logits, dim=-1).item()
        
        predictions.append(pred)
        confidences.append(confidence)
        labels.append(dataset[idx]["label"])
        
        if (idx + 1) % 100 == 0:
            print(f"      Processed {idx + 1}/{len(dataset)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    avg_confidence = np.mean(confidences)
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    
    print(f"    ✓ {model_name} evaluation complete")
    
    return {
        "accuracy": accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "avg_confidence": avg_confidence,
        "mean_latency": mean_latency,
        "median_latency": median_latency,
        "std_latency": std_latency,
        "predictions": predictions,
        "confidences": confidences,
        "labels": labels,
        "latencies": latencies
    }

# Evaluate all models
print("\n[6/8] Evaluating models...")
results_fp32 = evaluate_model(model_fp32, dataset["test"], "FP32")
results_fp16 = evaluate_model(model_fp16, dataset["test"], "FP16")
results_int8 = evaluate_model(model_int8, dataset["test"], "INT8 QAT")

# Calculate comparison percentages
print("\n[7/8] Calculating comparisons...")
def calc_diff(baseline, value):
    """Calculate percentage difference"""
    if baseline == 0:
        return 0.0
    return ((value - baseline) / baseline) * 100

size_fp16_diff = calc_diff(size_fp32, size_fp16)
size_int8_diff = calc_diff(size_fp32, size_int8)

acc_fp16_diff = results_fp16["accuracy"] - results_fp32["accuracy"]
acc_int8_diff = results_int8["accuracy"] - results_fp32["accuracy"]

conf_fp16_diff = results_fp16["avg_confidence"] - results_fp32["avg_confidence"]
conf_int8_diff = results_int8["avg_confidence"] - results_fp32["avg_confidence"]

lat_mean_fp16_diff = calc_diff(results_fp32["mean_latency"], results_fp16["mean_latency"])
lat_mean_int8_diff = calc_diff(results_fp32["mean_latency"], results_int8["mean_latency"])

lat_median_fp16_diff = calc_diff(results_fp32["median_latency"], results_fp16["median_latency"])
lat_median_int8_diff = calc_diff(results_fp32["median_latency"], results_int8["median_latency"])

# Create comparison table
print("\n[8/8] Generating comparison table...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Table data
table_data = [
    ["Metric", "FP32 (Baseline)", "FP16 (Half)", "INT8 (Quantized)", "FP16 vs FP32", "INT8 vs FP32"],
    [
        "Model Size (MB)",
        f"{size_fp32:.2f}",
        f"{size_fp16:.2f}",
        f"{size_int8:.2f}",
        f"{size_fp16_diff:+.2f}%",
        f"{size_int8_diff:+.2f}%"
    ],
    [
        "Accuracy (%)",
        f"{results_fp32['accuracy']:.2f}",
        f"{results_fp16['accuracy']:.2f}",
        f"{results_int8['accuracy']:.2f}",
        f"{acc_fp16_diff:+.2f}%",
        f"{acc_int8_diff:+.2f}%"
    ],
    [
        "Avg Confidence (%)",
        f"{results_fp32['avg_confidence']:.2f}",
        f"{results_fp16['avg_confidence']:.2f}",
        f"{results_int8['avg_confidence']:.2f}",
        f"{conf_fp16_diff:+.2f}%",
        f"{conf_int8_diff:+.2f}%"
    ],
    [
        "Mean Latency (ms)",
        f"{results_fp32['mean_latency']:.2f}",
        f"{results_fp16['mean_latency']:.2f}",
        f"{results_int8['mean_latency']:.2f}",
        f"{lat_mean_fp16_diff:+.2f}%",
        f"{lat_mean_int8_diff:+.2f}%"
    ],
    [
        "Median Latency (ms)",
        f"{results_fp32['median_latency']:.2f}",
        f"{results_fp16['median_latency']:.2f}",
        f"{results_int8['median_latency']:.2f}",
        f"{lat_median_fp16_diff:+.2f}%",
        f"{lat_median_int8_diff:+.2f}%"
    ],
    [
        "Std Latency (ms)",
        f"{results_fp32['std_latency']:.2f}",
        f"{results_fp16['std_latency']:.2f}",
        f"{results_int8['std_latency']:.2f}",
        "N/A",
        "N/A"
    ]
]

# Create table
table = ax.table(
    cellText=table_data,
    loc='center',
    cellLoc='center',
    colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(weight='bold', color='white')

# Style data rows - alternate colors
for i in range(1, 7):
    for j in range(6):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        else:
            cell.set_facecolor('#FFFFFF')
        
        # Color code the comparison columns
        if j >= 4 and i < 6:  # Comparison columns (not std row)
            text = cell.get_text().get_text()
            if '+' in text and j == 4:  # FP16 vs FP32
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#E8F8F5')  # Light green for reduction
                else:
                    cell.set_facecolor('#FEF5E7')  # Light yellow for increase
            elif '-' in text and j == 4:
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#FEF5E7')  # Light yellow
                else:
                    cell.set_facecolor('#FADBD8')  # Light red for decrease
            elif '+' in text and j == 5:  # INT8 vs FP32
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#E8F8F5')  # Light green for reduction
                else:
                    cell.set_facecolor('#FEF5E7')  # Light yellow for increase
            elif '-' in text and j == 5:
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#FEF5E7')  # Light yellow
                else:
                    cell.set_facecolor('#FADBD8')  # Light red for decrease

# Add title
fig.suptitle(
    'QUANTIZATION COMPARISON SUMMARY (FP32 vs FP16 vs INT8 QAT)',
    fontsize=14,
    fontweight='bold',
    y=0.98
)

# Add subtitle with test configuration
fig.text(
    0.5, 0.92,
    f'Test Configuration: {SAMPLE_TOTAL} samples | {WARMUP_RUNS} warmup runs | {INFERENCE_RUNS} inference runs per sample',
    ha='center',
    fontsize=9,
    style='italic',
    color='gray'
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_table.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Comparison table saved to {OUTPUT_DIR}/comparison_table.png")

# Save detailed results to JSON
print("\nSaving detailed results...")
detailed_results = {
    "test_configuration": {
        "sample_total": SAMPLE_TOTAL,
        "warmup_runs": WARMUP_RUNS,
        "inference_runs": INFERENCE_RUNS
    },
    "model_sizes_mb": {
        "fp32": round(size_fp32, 2),
        "fp16": round(size_fp16, 2),
        "int8": round(size_int8, 2)
    },
    "fp32_baseline": {
        "accuracy": round(results_fp32["accuracy"], 2),
        "precision": round(results_fp32["precision"], 2),
        "recall": round(results_fp32["recall"], 2),
        "f1": round(results_fp32["f1"], 2),
        "avg_confidence": round(results_fp32["avg_confidence"], 2),
        "mean_latency_ms": round(results_fp32["mean_latency"], 2),
        "median_latency_ms": round(results_fp32["median_latency"], 2),
        "std_latency_ms": round(results_fp32["std_latency"], 2)
    },
    "fp16_half": {
        "accuracy": round(results_fp16["accuracy"], 2),
        "precision": round(results_fp16["precision"], 2),
        "recall": round(results_fp16["recall"], 2),
        "f1": round(results_fp16["f1"], 2),
        "avg_confidence": round(results_fp16["avg_confidence"], 2),
        "mean_latency_ms": round(results_fp16["mean_latency"], 2),
        "median_latency_ms": round(results_fp16["median_latency"], 2),
        "std_latency_ms": round(results_fp16["std_latency"], 2)
    },
    "int8_qat": {
        "accuracy": round(results_int8["accuracy"], 2),
        "precision": round(results_int8["precision"], 2),
        "recall": round(results_int8["recall"], 2),
        "f1": round(results_int8["f1"], 2),
        "avg_confidence": round(results_int8["avg_confidence"], 2),
        "mean_latency_ms": round(results_int8["mean_latency"], 2),
        "median_latency_ms": round(results_int8["median_latency"], 2),
        "std_latency_ms": round(results_int8["std_latency"], 2)
    },
    "comparisons": {
        "fp16_vs_fp32": {
            "size_diff_pct": round(size_fp16_diff, 2),
            "accuracy_diff": round(acc_fp16_diff, 2),
            "confidence_diff": round(conf_fp16_diff, 2),
            "mean_latency_diff_pct": round(lat_mean_fp16_diff, 2),
            "median_latency_diff_pct": round(lat_median_fp16_diff, 2)
        },
        "int8_vs_fp32": {
            "size_diff_pct": round(size_int8_diff, 2),
            "accuracy_diff": round(acc_int8_diff, 2),
            "confidence_diff": round(conf_int8_diff, 2),
            "mean_latency_diff_pct": round(lat_mean_int8_diff, 2),
            "median_latency_diff_pct": round(lat_median_int8_diff, 2)
        }
    }
}

with open(os.path.join(OUTPUT_DIR, "comparison_results.json"), "w") as f:
    json.dump(detailed_results, f, indent=4)

print(f"✓ Detailed results saved to {OUTPUT_DIR}/comparison_results.json")

# Print summary to console
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\nModel Sizes:")
print(f"  FP32: {size_fp32:.2f} MB")
print(f"  FP16: {size_fp16:.2f} MB ({size_fp16_diff:+.2f}%)")
print(f"  INT8: {size_int8:.2f} MB ({size_int8_diff:+.2f}%)")
print(f"\nAccuracy:")
print(f"  FP32: {results_fp32['accuracy']:.2f}%")
print(f"  FP16: {results_fp16['accuracy']:.2f}% ({acc_fp16_diff:+.2f}%)")
print(f"  INT8: {results_int8['accuracy']:.2f}% ({acc_int8_diff:+.2f}%)")
print(f"\nAverage Confidence:")
print(f"  FP32: {results_fp32['avg_confidence']:.2f}%")
print(f"  FP16: {results_fp16['avg_confidence']:.2f}% ({conf_fp16_diff:+.2f}%)")
print(f"  INT8: {results_int8['avg_confidence']:.2f}% ({conf_int8_diff:+.2f}%)")
print(f"\nMean Latency:")
print(f"  FP32: {results_fp32['mean_latency']:.2f} ms")
print(f"  FP16: {results_fp16['mean_latency']:.2f} ms ({lat_mean_fp16_diff:+.2f}%)")
print(f"  INT8: {results_int8['mean_latency']:.2f} ms ({lat_mean_int8_diff:+.2f}%)")
print("="*70)
print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
print("  - comparison_table.png")
print("  - comparison_results.json")
print("\nComparison complete!")
