"""
Evaluate QAT model using ONNX Runtime for optimized inference
Provides better performance than PyTorch eager mode
"""
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed")
    print("Install it with: pip install onnxruntime")
    exit(1)

print("="*70)
print("Evaluating QAT Model with ONNX Runtime")
print("="*70)

# Load tokenizer
save_path_qat = "./models/indobert-qat-int8-combined"
tokenizer = AutoTokenizer.from_pretrained(save_path_qat)
print("✓ Tokenizer loaded")

# Load ONNX model
onnx_path = f"{save_path_qat}/model_qat.onnx"

# Configure ONNX Runtime session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

# Create session
session = ort.InferenceSession(
    onnx_path,
    sess_options,
    providers=['CPUExecutionProvider']  # Use CPU for INT8 inference
)

print(f"✓ ONNX model loaded from: {onnx_path}")
print(f"  Execution Provider: {session.get_providers()}")

# Load test dataset
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = load_dataset(
    'csv',
    data_files={'test': './combined/test.csv'}
)

print(f"✓ Test samples: {len(dataset['test']):,}")

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=['text', 'source']
)

print("✓ Tokenization complete")

# Run inference
print("\nRunning inference on test set...")
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
    
    # Run inference
    start_time = time.time()
    outputs = session.run(
        None,
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    )
    inference_time = time.time() - start_time
    inference_times.append(inference_time)
    
    # Get predictions
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
    true_labels, predictions, average='binary'
)

# Calculate inference statistics
total_time = sum(inference_times)
avg_time_per_batch = np.mean(inference_times) * 1000  # Convert to ms
samples_per_second = num_samples / total_time

print("\n" + "="*70)
print("ONNX Runtime QAT Results")
print("="*70)
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print("="*70)
print("\nPerformance Metrics:")
print(f"  Total inference time: {total_time:.2f}s")
print(f"  Avg time per batch ({batch_size} samples): {avg_time_per_batch:.2f}ms")
print(f"  Samples per second: {samples_per_second:.2f}")
print("="*70)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(true_labels), 
            yticklabels=np.unique(true_labels))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("ONNX QAT Model - Confusion Matrix")

output_dir = "./results/indobert-qat-combined"
plt.savefig(f"{output_dir}/confusion_matrix_onnx.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Confusion matrix saved to: {output_dir}/confusion_matrix_onnx.png")

print("\nEvaluation complete!")
