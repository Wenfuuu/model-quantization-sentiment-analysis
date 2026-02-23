from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.quantization as quantization
import os

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

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
        'train': './combined/train.csv',
        'validation': './combined/val.csv',
        'test': './combined/test.csv'
    }
)

print(f"Train samples: {len(dataset['train']):,}")
print(f"Validation samples: {len(dataset['validation']):,}")
print(f"Test samples: {len(dataset['test']):,}")
print(f"\nColumns: {dataset['train'].column_names}")
print(f"\nSample: {dataset['train'][0]}")

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=['text', 'source']
)

print("Tokenization complete!")
print(f"Columns after tokenization: {tokenized_dataset['train'].column_names}")
print(f"\nSample tokenized data:")
print(f"  input_ids length: {len(tokenized_dataset['train'][0]['input_ids'])}")
print(f"  attention_mask length: {len(tokenized_dataset['train'][0]['attention_mask'])}")

model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2",
    num_labels=2
)

print(f"FP32 Model loaded: {model.__class__.__name__}")
print(f"Number of parameters: {model.num_parameters():,}")

model.train()

model.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.fake_quantize.default_fake_quant,
    weight=torch.quantization.default_weight_fake_quant
)

if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
    model.bert.embeddings.qconfig = None
    print("Embedding Layer excluded from Quantization (Safe Mode)")

# Prepare model for QAT (inserts fake quantization modules)
model_qat = quantization.prepare_qat(model, inplace=False)

print(f"\nModel prepared for QAT")
print(f"Fake quantization modules inserted")
print(f"Model will simulate INT8 quantization during training")

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

output_dir = "./results/indobert-qat-combined"

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    report_to="none",
    fp16=False,
    push_to_hub=False
)

print("QAT Training arguments configured:")
print(f"  Output directory: {output_dir}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Quantization: INT8 (fake quant during training)")

trainer = Trainer(
    model=model_qat,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("QAT Trainer initialized successfully!")

print("Starting QAT training...")
print("Note: Training with fake quantization may be 10-20% slower than FP32")
print("="*70)

train_result = trainer.train()

print("\n" + "="*70)
print("QAT Training completed!")
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

model_qat.eval()

print("\nModel training complete (with fake quantization)")
print(f"Model will be saved in QAT format for deployment")

save_path_qat = "./models/indobert-qat-int8-combined"
os.makedirs(save_path_qat, exist_ok=True)

torch.save(model_qat.state_dict(), f"{save_path_qat}/model_qat.pth")
tokenizer.save_pretrained(save_path_qat)

print(f"\nQAT model saved to: {save_path_qat}")
print("\nNext steps:")
print("  1. Export to ONNX: python export_to_onnx.py")
print("  2. Quantize ONNX: python quantize_onnx_dynamic.py")
print("  3. Evaluate INT8: python eval_onnx_int8_dynamic.py")

qat_size = os.path.getsize(f"{save_path_qat}/model_qat.pth") / (1024 * 1024)

print("\nModel Size:")
print("="*70)
print(f"QAT model (FP32 with fake quant): {qat_size:.2f} MB")
print(f"After ONNX INT8 quantization: ~{qat_size / 4:.2f} MB (estimated)")
print("="*70)
print("\nNote: This is still FP32 size. True INT8 requires ONNX export.")