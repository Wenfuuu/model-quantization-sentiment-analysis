from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import re
import json

# Check if bitsandbytes is available
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
except ImportError:
    print("ERROR: bitsandbytes not installed. Please install it:")
    print("pip install bitsandbytes")
    exit(1)

MODEL_PATH = "./models/indobert-smsa-qat-int4-fake"
OUTPUT_DIR = "./results/indobert-smsa-qat-int4-fake"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure 4-bit quantization for loading
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config
)

print(f"INT4 Model loaded from: {MODEL_PATH}")
print(f"Number of parameters: {model.num_parameters():,}")
print(f"Model is quantized to 4-bit")

model.eval()

dataset = load_dataset(
    "csv",
    data_files={"test": "test.tsv"},
    delimiter="\t",
    column_names=["text", "label"]
)

print(f"Test samples: {len(dataset['test']):,}")

label2id = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}

id2label = {
    0: "positive",
    1: "neutral",
    2: "negative"
}

def map_labels(df):
    df["label"] = [label2id[label] for label in df["label"]]
    return df

dataset = dataset.map(map_labels, batched=True)

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

def preprocess_dataset(examples):
    examples["text"] = [preprocess_text(text) for text in examples["text"]]
    return examples

dataset = dataset.map(preprocess_dataset, batched=True)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

eval_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=8,  # Smaller batch size for 4-bit
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("Running evaluation on test set (INT4)...")
results = trainer.evaluate()

print("\nTest Set Results (INT4)")
print("=" * 70)
for k, v in results.items():
    print(f"{k}: {v:.4f}")
print("=" * 70)

predictions_output = trainer.predict(tokenized_dataset["test"])
y_pred = predictions_output.predictions.argmax(-1)
y_true = predictions_output.label_ids

label_names = ["Positive", "Neutral", "Negative"]
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - INT4")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_int4.png"), dpi=300)
plt.close()

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=label_names,
    output_dict=True,
    zero_division=0
)

with open(os.path.join(OUTPUT_DIR, "evaluation_results_int4.json"), "w") as f:
    json.dump(
        {
            "overall_metrics": results,
            "classification_report": report_dict
        },
        f,
        indent=4
    )

print("\nDetailed Classification Report (INT4)")
print("=" * 70)
print(classification_report(y_true, y_pred, target_names=label_names))
print("=" * 70)

print(f"\nEvaluation artifacts saved to: {OUTPUT_DIR}")
