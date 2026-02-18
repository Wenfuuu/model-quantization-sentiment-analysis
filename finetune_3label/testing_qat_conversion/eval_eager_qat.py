from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.quantization as quantization
import os

print("Loading saved QAT INT8 model...")

# Load tokenizer
save_path_qat = "./models/indobert-qat-int8-combined"
tokenizer = AutoTokenizer.from_pretrained(save_path_qat)
print(f"Tokenizer loaded from: {save_path_qat}")

# Load base model structure
model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2",
    num_labels=2
)
print(f"Base model loaded: {model.__class__.__name__}")

# Prepare model for quantization (same as training)
model.train()
model.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.fake_quantize.default_fake_quant,
    weight=torch.quantization.default_weight_fake_quant
)

if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
    model.bert.embeddings.qconfig = None

# Prepare for QAT (we'll keep it in fake quantization mode instead of converting)
model_qat = quantization.prepare_qat(model, inplace=False)

# Load trained weights
model_qat.load_state_dict(torch.load(f"{save_path_qat}/model_qat.pth", map_location='cpu'))
model_qat.eval()  # Set to eval mode (keeps fake quantization active)
model_qat = model_qat.cpu()
print("QAT model weights loaded successfully (CPU mode - using fake quantization)")

# Load and tokenize test dataset
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = load_dataset(
    'csv',
    data_files={
        'test': './combined/test.csv'
    }
)

print(f"Test samples: {len(dataset['test']):,}")

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=['text', 'source']
)

print("Tokenization complete!")

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Create trainer for evaluation
output_dir = "./results/indobert-qat-combined"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=16,
    report_to="none",
    fp16=False,
    no_cuda=True  # Force CPU usage for quantized INT8 model
)

trainer_qat = Trainer(
    model=model_qat,
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("Evaluating QAT model on test set (with fake quantization)...")
print("="*70)

test_results = trainer_qat.evaluate(tokenized_dataset['test'])

print("\nQAT INT8 Test Set Results:")
print("="*70)
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print("="*70)

# Generate predictions and confusion matrix
print("\nGenerating confusion matrix...")
predictions_output = trainer_qat.predict(tokenized_dataset['test'])
y_pred = predictions_output.predictions.argmax(-1)
y_true = predictions_output.label_ids
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("QAT INT8 Model - Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")
plt.show()

print("\nEvaluation complete!")
