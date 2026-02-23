"""
Evaluate INT8 ONNX model with ONNX Runtime
"""
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import re
import json
import os
import onnxruntime as ort

print("="*70)
print("Evaluating INT8 ONNX Model - SMSA Sentiment")
print("="*70)

save_path = "./models/indobert-qat-int8-smsa"
tokenizer = AutoTokenizer.from_pretrained(save_path)
print("✓ Tokenizer loaded")

# Load stopwords
stopword_factory = StopWordRemoverFactory()
indonesian_stopwords = stopword_factory.get_stop_words()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in indonesian_stopwords]
    return ' '.join(words)

# Load INT8 ONNX model
onnx_path = f"{save_path}/model_qat_int8.onnx"

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    onnx_path,
    sess_options,
    providers=['CPUExecutionProvider']
)

print(f"✓ INT8 ONNX model loaded")
print(f"  Provider: {session.get_providers()}")

# Load test dataset
def tokenize_fn(batch):
    preprocessed = [preprocess_text(text) for text in batch["text"]]
    return tokenizer(
        preprocessed,
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = load_dataset(
    'csv',
    data_files={'test': 'test.tsv'},
    delimiter='\t',
    column_names=['text', 'label']
)

label2id = {'positive': 0, 'neutral': 1, 'negative': 2}
id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}

def map_labels(df):
    df['label'] = [label2id[label] for label in df['label']]
    return df

dataset = dataset.map(map_labels, batched=True)
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

print(f"✓ Test samples: {len(tokenized_dataset['test']):,}")

# Run inference
print("\n" + "="*70)
print("Running INT8 inference on test set...")
print("="*70)

predictions = []
true_labels = []
inference_times = []
batch_size = 16
num_samples = len(tokenized_dataset['test'])

for i in range(0, num_samples, batch_size):
    batch_end = min(i + batch_size, num_samples)
    batch = tokenized_dataset['test'][i:batch_end]
    
    input_ids = np.array(batch['input_ids'], dtype=np.int64)
    attention_mask = np.array(batch['attention_mask'], dtype=np.int64)
    
    start_time = time.time()
    outputs = session.run(None, {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })
    inference_time = time.time() - start_time
    inference_times.append(inference_time)
    
    logits = outputs[0]
    batch_predictions = np.argmax(logits, axis=1)
    
    predictions.extend(batch_predictions)
    true_labels.extend(batch['label'])
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {batch_end}/{num_samples} samples...")

predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='weighted'
)

# Performance stats
total_time = sum(inference_times)
avg_time_per_batch = np.mean(inference_times) * 1000
samples_per_second = num_samples / total_time

print("\n" + "="*70)
print("INT8 ONNX Model Results")
print("="*70)
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print("="*70)
print("\nPerformance:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Avg per batch ({batch_size} samples): {avg_time_per_batch:.2f}ms")
print(f"  Samples/second: {samples_per_second:.2f}")
print("="*70)

print("\nClassification Report:")
print(classification_report(true_labels, predictions, 
                          target_names=['positive', 'neutral', 'negative']))

# Create confusion matrix
label_names = ['Positive', 'Neutral', 'Negative']
cm = confusion_matrix(true_labels, predictions)

output_dir = "./results/indobert-qat-int8-smsa"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - INT8 ONNX')
plt.tight_layout()
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_int8.png')
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()
print(f"\n✓ Confusion matrix saved to: {confusion_matrix_path}")

# Save detailed results to JSON
report_dict = classification_report(
    true_labels,
    predictions,
    target_names=label_names,
    output_dict=True,
    zero_division=0
)

results_data = {
    'model_type': 'INT8',
    'provider': session.get_providers()[0],
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'total_samples': int(num_samples),
        'total_time_seconds': float(total_time),
        'avg_time_per_batch_ms': float(avg_time_per_batch),
        'samples_per_second': float(samples_per_second)
    },
    'classification_report': report_dict
}

results_path = os.path.join(output_dir, 'evaluation_results_int8.json')
with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=4)
print(f"✓ Evaluation results saved to: {results_path}")

print("\n✓ Evaluation complete!")
print("\nThis INT8 model benefits from:")
print("  • QAT training (better accuracy than PTQ)")
print("  • True INT8 weights (4x smaller)")
print("  • Optimized INT8 operations (2-4x faster)")
