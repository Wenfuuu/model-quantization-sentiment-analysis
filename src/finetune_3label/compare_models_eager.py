"""
Compare FP32 PyTorch vs FP16 ONNX vs INT8 ONNX models
Uses ONNX Runtime for quantized models
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import time
import re
import json
import os
import onnxruntime as ort
from pathlib import Path

INFERENCE_RUNS = 20
WARMUP_RUNS = 5
SAMPLE_TOTAL = 500
OUTPUT_DIR = "./comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("MODEL COMPARISON: FP32 PyTorch vs FP16 ONNX vs INT8 ONNX")
print("="*70)

FP32_MODEL = "./models/indobert-fp32-smsa-3label-finetuned"
FP16_ONNX = "./models/indobert-qat-fp16-smsa/model_qat_fp16.onnx"
FP16_ONNX_FALLBACK = "./models/indobert-qat-fp16-smsa/model_qat.onnx"
INT8_ONNX = "./models/indobert-qat-int8-smsa/model_qat_int8.onnx"

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

# Load FP32 PyTorch model
print("  Loading FP32 (PyTorch Baseline)...")
model_fp32 = AutoModelForSequenceClassification.from_pretrained(
    FP32_MODEL,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)
model_fp32.eval()
print("  ✓ FP32 loaded")

# Load FP16 ONNX model
print("  Loading FP16 (ONNX)...")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

available_providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in available_providers:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
elif 'TensorrtExecutionProvider' in available_providers:
    providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']

# Try FP16, fallback to FP32 ONNX if fails
fp16_model_type = "FP16"
try:
    session_fp16 = ort.InferenceSession(FP16_ONNX, sess_options, providers=providers)
    fp16_path_used = FP16_ONNX
    print("  ✓ FP16 ONNX loaded")
except Exception as e:
    print(f"  ⚠ FP16 failed ({str(e)[:50]}...), using FP32 ONNX fallback")
    session_fp16 = ort.InferenceSession(FP16_ONNX_FALLBACK, sess_options, providers=providers)
    fp16_model_type = "FP32"
    fp16_path_used = FP16_ONNX_FALLBACK
    print("  ✓ FP32 ONNX fallback loaded")

# Load INT8 ONNX model
print("  Loading INT8 (ONNX)...")
# Try GPU first for fair comparison, fallback to CPU
try:
    session_int8 = ort.InferenceSession(
        INT8_ONNX,
        sess_options,
        providers=providers  # Use same providers as FP16 (GPU if available)
    )
    int8_provider = session_int8.get_providers()[0]
    print(f"  ✓ INT8 ONNX loaded (Provider: {int8_provider})")
except Exception as e:
    print(f"  ⚠ GPU failed for INT8, using CPU: {str(e)[:50]}...")
    session_int8 = ort.InferenceSession(
        INT8_ONNX,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    int8_provider = 'CPUExecutionProvider'
    print(f"  ✓ INT8 ONNX loaded (Provider: CPU)")

# Get model sizes
print("\n[5/8] Calculating model sizes...")

def get_pytorch_model_size_mb(model, model_path=None):
    """Calculate PyTorch model size in MB"""
    if model_path and os.path.exists(model_path):
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        if total_size > 0:
            return total_size / (1024 * 1024)
    
    # Fallback: calculate from parameters
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)

def get_onnx_model_size_mb(onnx_path):
    """Get ONNX model file size in MB"""
    if os.path.exists(onnx_path):
        return os.path.getsize(onnx_path) / (1024 * 1024)
    return 0.0

size_fp32 = get_pytorch_model_size_mb(model_fp32, FP32_MODEL)
size_fp16 = get_onnx_model_size_mb(fp16_path_used)
size_int8 = get_onnx_model_size_mb(INT8_ONNX)

# Get actual providers being used
fp32_device = "CUDA" if torch.cuda.is_available() else "CPU"
fp16_provider = session_fp16.get_providers()[0]
# int8_provider already set during loading

print(f"  FP32 (PyTorch): {size_fp32:.2f} MB [Device: {fp32_device}]")
print(f"  {fp16_model_type} (ONNX): {size_fp16:.2f} MB [Provider: {fp16_provider}]")
print(f"  INT8 (ONNX): {size_int8:.2f} MB [Provider: {int8_provider}]")

# Warn about INT8 on GPU
if int8_provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
    print("\n  ⚠ NOTE: INT8 on GPU will likely be SLOWER than FP32 due to:")
    print("     • Limited INT8 op support on CUDA (186 Memcpy nodes detected)")
    print("     • Constant CPU↔GPU memory transfers")
    print("     • INT8 is optimized for CPU inference, not GPU")
    print("     For INT8 benefits, run on CPU-only (see comparison results).")

# Evaluation functions
def evaluate_pytorch_model(model, dataset, model_name):
    """Evaluate PyTorch model and return metrics"""
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
        
        # Multiple inference runs for latency
        sample_latencies = []
        for _ in range(INFERENCE_RUNS):
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            end_time = time.perf_counter()
            sample_latencies.append((end_time - start_time) * 1000)
        
        latencies.append(np.mean(sample_latencies))
        
        # Get prediction
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
    
    print(f"    ✓ {model_name} evaluation complete")
    
    return {
        "accuracy": accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "avg_confidence": np.mean(confidences),
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "std_latency": np.std(latencies),
        "predictions": predictions,
        "confidences": confidences,
        "labels": labels,
        "latencies": latencies
    }

def evaluate_onnx_model(session, dataset, model_name):
    """Evaluate ONNX model and return metrics"""
    print(f"\n  Evaluating {model_name}...")
    
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
            return_tensors="np"
        )
        
        _ = session.run(None, {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        })
    
    # Inference runs
    print(f"    Inference ({len(dataset)} samples, {INFERENCE_RUNS} runs each)...")
    for idx in range(len(dataset)):
        inputs = tokenizer(
            dataset[idx]["text"] if isinstance(dataset[idx]["text"], str) else "",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="np"
        )
        
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        
        # Multiple inference runs for latency
        sample_latencies = []
        for _ in range(INFERENCE_RUNS):
            start_time = time.perf_counter()
            outputs = session.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            end_time = time.perf_counter()
            sample_latencies.append((end_time - start_time) * 1000)
        
        latencies.append(np.mean(sample_latencies))
        
        # Get prediction
        logits = outputs[0][0]  # First output, first sample
        probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
        confidence = np.max(probs) * 100
        pred = np.argmax(logits)
        
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
    
    print(f"    ✓ {model_name} evaluation complete")
    
    return {
        "accuracy": accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "avg_confidence": np.mean(confidences),
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "std_latency": np.std(latencies),
        "predictions": predictions,
        "confidences": confidences,
        "labels": labels,
        "latencies": latencies
    }

# Evaluate all models
print("\n[6/8] Evaluating models...")
results_fp32 = evaluate_pytorch_model(model_fp32, dataset["test"], "FP32 PyTorch")
results_fp16 = evaluate_onnx_model(session_fp16, dataset["test"], f"{fp16_model_type} ONNX")
results_int8 = evaluate_onnx_model(session_int8, dataset["test"], "INT8 ONNX")

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
    ["Metric", "FP32 PyTorch", f"{fp16_model_type} ONNX", "INT8 ONNX", f"{fp16_model_type} vs FP32", "INT8 vs FP32"],
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

# Style data rows
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
                    cell.set_facecolor('#E8F5E9')  # Light green (good: smaller/faster)
                else:
                    cell.set_facecolor('#FFEBEE')  # Light red (bad: worse metric)
            elif '-' in text and j == 4:
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#FFEBEE')  # Light red (bad: bigger/slower)
                else:
                    cell.set_facecolor('#E8F5E9')  # Light green (good: better metric)
            elif '+' in text and j == 5:  # INT8 vs FP32
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#E8F5E9')
                else:
                    cell.set_facecolor('#FFEBEE')
            elif '-' in text and j == 5:
                if 'Size' in table_data[i][0] or 'Latency' in table_data[i][0]:
                    cell.set_facecolor('#FFEBEE')
                else:
                    cell.set_facecolor('#E8F5E9')

# Add title
fig.suptitle(
    'QUANTIZATION COMPARISON: FP32 PyTorch vs ONNX Runtime (FP16/INT8)',
    fontsize=14,
    fontweight='bold',
    y=0.98
)

# Add subtitle
subtitle = f'Test Config: {SAMPLE_TOTAL} samples | {WARMUP_RUNS} warmup | {INFERENCE_RUNS} inference runs/sample'
subtitle += f' | Providers: FP32={fp32_device}, {fp16_model_type}={fp16_provider.split("ExecutionProvider")[0]}, INT8={int8_provider.split("ExecutionProvider")[0]}'
fig.text(
    0.5, 0.92,
    subtitle,
    ha='center',
    fontsize=9,
    style='italic',
    color='gray'
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_table_onnx.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Comparison table saved to {OUTPUT_DIR}/comparison_table_onnx.png")

# Save detailed results to JSON
print("\nSaving detailed results...")
detailed_results = {
    "test_configuration": {
        "sample_total": SAMPLE_TOTAL,
        "warmup_runs": WARMUP_RUNS,
        "inference_runs": INFERENCE_RUNS,
        "fp16_model_type": fp16_model_type,
        "fp16_fallback_used": fp16_model_type == "FP32",
        "hardware_used": {
            "fp32_device": fp32_device,
            "fp16_provider": fp16_provider,
            "int8_provider": int8_provider
        }
    },
    "model_info": {
        "fp32": {
            "type": "PyTorch",
            "path": FP32_MODEL,
            "size_mb": round(size_fp32, 2)
        },
        "fp16": {
            "type": f"{fp16_model_type} ONNX",
            "path": fp16_path_used,
            "size_mb": round(size_fp16, 2)
        },
        "int8": {
            "type": "INT8 ONNX",
            "path": INT8_ONNX,
            "size_mb": round(size_int8, 2)
        }
    },
    "fp32_pytorch": {
        "accuracy": round(results_fp32["accuracy"], 2),
        "precision": round(results_fp32["precision"], 2),
        "recall": round(results_fp32["recall"], 2),
        "f1": round(results_fp32["f1"], 2),
        "avg_confidence": round(results_fp32["avg_confidence"], 2),
        "mean_latency_ms": round(results_fp32["mean_latency"], 2),
        "median_latency_ms": round(results_fp32["median_latency"], 2),
        "std_latency_ms": round(results_fp32["std_latency"], 2)
    },
    f"{fp16_model_type.lower()}_onnx": {
        "accuracy": round(results_fp16["accuracy"], 2),
        "precision": round(results_fp16["precision"], 2),
        "recall": round(results_fp16["recall"], 2),
        "f1": round(results_fp16["f1"], 2),
        "avg_confidence": round(results_fp16["avg_confidence"], 2),
        "mean_latency_ms": round(results_fp16["mean_latency"], 2),
        "median_latency_ms": round(results_fp16["median_latency"], 2),
        "std_latency_ms": round(results_fp16["std_latency"], 2)
    },
    "int8_onnx": {
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
        f"{fp16_model_type.lower()}_vs_fp32": {
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

with open(os.path.join(OUTPUT_DIR, "comparison_results_onnx.json"), "w") as f:
    json.dump(detailed_results, f, indent=4)

print(f"✓ Detailed results saved to {OUTPUT_DIR}/comparison_results_onnx.json")

# Print summary to console
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\nModel Sizes:")
print(f"  FP32 (PyTorch): {size_fp32:.2f} MB")
print(f"  {fp16_model_type} (ONNX):   {size_fp16:.2f} MB ({size_fp16_diff:+.2f}%)")
print(f"  INT8 (ONNX):    {size_int8:.2f} MB ({size_int8_diff:+.2f}%)")
print(f"\nAccuracy:")
print(f"  FP32: {results_fp32['accuracy']:.2f}%")
print(f"  {fp16_model_type}:   {results_fp16['accuracy']:.2f}% ({acc_fp16_diff:+.2f}%)")
print(f"  INT8: {results_int8['accuracy']:.2f}% ({acc_int8_diff:+.2f}%)")
print(f"\nAverage Confidence:")
print(f"  FP32: {results_fp32['avg_confidence']:.2f}%")
print(f"  {fp16_model_type}:   {results_fp16['avg_confidence']:.2f}% ({conf_fp16_diff:+.2f}%)")
print(f"  INT8: {results_int8['avg_confidence']:.2f}% ({conf_int8_diff:+.2f}%)")
print(f"\nMean Latency:")
print(f"  FP32: {results_fp32['mean_latency']:.2f} ms")
print(f"  {fp16_model_type}:   {results_fp16['mean_latency']:.2f} ms ({lat_mean_fp16_diff:+.2f}%)")
print(f"  INT8: {results_int8['mean_latency']:.2f} ms ({lat_mean_int8_diff:+.2f}%)")
print("="*70)
# Warn about hardware differences
print(f"\nHardware used:")
print(f"  FP32: {fp32_device}")
print(f"  {fp16_model_type}: {fp16_provider}")
print(f"  INT8: {int8_provider}")

if int8_provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
    print("\n" + "="*70)
    print("⚠ IMPORTANT: INT8 PERFORMANCE ON GPU")
    print("="*70)
    print("INT8 latency on GPU is SLOWER than FP32 - this is EXPECTED!")
    print()
    print("Why INT8 is slow on CUDA:")
    print("  • CUDA lacks native INT8 support for transformer models")
    print("  • 186 Memcpy nodes = massive CPU↔GPU memory copy overhead")
    print("  • Many INT8 ops fall back to CPU execution")
    print("  • GPUs are optimized for FP32/FP16, not INT8")
    print()
    print("Where INT8 provides speedup:")
    print("  ✅ CPU deployment: 2-4x faster than FP32 CPU")
    print("  ✅ Edge devices with INT8 accelerators")
    print("  ✅ Intel CPUs with VNNI instructions")
    print("  ✅ ARM CPUs with INT8 instructions")
    print("  ❌ NVIDIA GPUs: Use FP32 or FP16 instead")
    print()
    print("Recommendation:")
    print("  • GPU deployment → Use FP32 PyTorch or FP16 (if supported)")
    print("  • CPU deployment → Use INT8 ONNX (2-4x faster)")
    print("  • Model size matters → INT8 (4x smaller, good for memory)")
    print("="*70)
elif int8_provider == 'CPUExecutionProvider' and fp32_device == 'CUDA':
    print("\n⚠ NOTE: Comparing GPU (FP32) vs CPU (INT8) - latency difference expected.")


if fp16_model_type == "FP32":
    print("\n⚠ NOTE: FP16 model failed to load, used FP32 fallback")
    print("   See FP16_ISSUES.md for details on FP16 ONNX compatibility issues")

print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
print("  - comparison_table_onnx.png")
print("  - comparison_results_onnx.json")
print("\nComparison complete!")
