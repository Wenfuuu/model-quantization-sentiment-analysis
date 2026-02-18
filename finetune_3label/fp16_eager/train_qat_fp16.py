from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter

import numpy as np
import torch
import torch.quantization as quantization
import os
import re

print("="*70)
print("FP16 QAT Training - SMSA Sentiment Analysis")
print("="*70)

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")

# Load stopwords for preprocessing
stopword_factory = StopWordRemoverFactory()
indonesian_stopwords = stopword_factory.get_stop_words()
print(f"✓ Loaded {len(indonesian_stopwords)} Indonesian stop words")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in indonesian_stopwords]
    return ' '.join(words)

def tokenize_fn(batch):
    # Preprocess texts
    preprocessed = [preprocess_text(text) for text in batch["text"]]
    return tokenizer(
        preprocessed,
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Load dataset
dataset = load_dataset(
    'csv',
    data_files={
        'train': 'train.tsv',
        'validation': 'valid.tsv',
        'test': 'test.tsv'
    },
    delimiter='\t',
    column_names=['text', 'label']
)

print(f"✓ Train samples: {len(dataset['train']):,}")
print(f"✓ Validation samples: {len(dataset['validation']):,}")
print(f"✓ Test samples: {len(dataset['test']):,}")

# Label mapping
label2id = {'positive': 0, 'neutral': 1, 'negative': 2}
id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}

print("\nLabel mapping:")
for label, idx in label2id.items():
    print(f"  {label} -> {idx}")

def map_labels(df):
    df['label'] = [label2id[label] for label in df['label']]
    return df

dataset = dataset.map(map_labels, batched=True)

# Tokenize
tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=['text']
)

print("✓ Tokenization and preprocessing complete")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

print(f"✓ FP16 Model loaded: {model.num_parameters():,} parameters")

# Configure QAT for FP16
model.train()
model.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.fake_quantize.default_fake_quant,
    weight=torch.quantization.default_weight_fake_quant
)

if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
    model.bert.embeddings.qconfig = None
    print("✓ Embedding layer excluded from quantization")

# Prepare model for QAT
model_qat = quantization.prepare_qat(model, inplace=False)
print("✓ Model prepared for QAT (fake FP16 quantization)")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

output_dir = "./results/indobert-qat-fp16-smsa"

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
    logging_steps=50,
    report_to="none",
    fp16=False,
    push_to_hub=False
)

print("\nTraining configuration:")
print(f"  Output directory: {output_dir}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Quantization: FP16 fake quantization (QAT)")

trainer = Trainer(
    model=model_qat,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("\n" + "="*70)
print("Starting QAT training...")
print("="*70)

train_result = trainer.train()

print("\n" + "="*70)
print("Training completed!")
print("="*70)
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds")

# Save model
model_qat.eval()
save_path = "./models/indobert-qat-fp16-smsa"
os.makedirs(save_path, exist_ok=True)

torch.save(model_qat.state_dict(), f"{save_path}/model_qat.pth")
tokenizer.save_pretrained(save_path)

print(f"\n✓ QAT model saved to: {save_path}")

qat_size = os.path.getsize(f"{save_path}/model_qat.pth") / (1024 * 1024)
print(f"\nModel size: {qat_size:.2f} MB (FP32 with QAT-trained weights)")
print("\nNext steps:")
print("  1. Export to ONNX: python export_to_onnx.py")
print("  2. Quantize to FP16: python quantize_onnx_fp16.py")
print("  3. Evaluate FP16: python eval_onnx_fp16.py")
print("="*70)
