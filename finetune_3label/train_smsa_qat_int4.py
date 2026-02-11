from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
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
        'train': 'train.tsv',
        'validation': 'valid.tsv',
        'test': 'test.tsv'
    },
    delimiter='\t',
    column_names=['text', 'label']
)

print(f"Train samples: {len(dataset['train']):,}")
print(f"Validation samples: {len(dataset['validation']):,}")
print(f"Test samples: {len(dataset['test']):,}")
print(f"\nColumns: {dataset['train'].column_names}")
print(f"\nSample: {dataset['train'][0]}")
print(f"\nLabel distribution in training set:")

label_counts = Counter(dataset['train']['label'])
for label, count in label_counts.items():
    print(f"  {label}: {count}")

# label mapping
label2id = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}
id2label = {
    0: 'positive',
    1: 'neutral',
    2: 'negative'
}

print("Label mapping:")
for label, idx in label2id.items():
    print(f"  {label} -> {idx}")

def map_labels(df):
    df['label'] = [label2id[label] for label in df['label']]
    return df

dataset = dataset.map(
    map_labels,
    batched=True,
    desc="Mapping labels to numeric values"
)

print("\nAfter mapping, sample label:", dataset['train'][0]['label'])
print(f"Label type: {type(dataset['train'][0]['label'])}")

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # normalized float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # nested quantization for better memory efficiency
)

print("4-bit quantization configuration:")
print(f"  Quantization type: NF4 (Normalized Float 4-bit)")
print(f"  Compute dtype: FP16")
print(f"  Double quantization: Enabled")

model_int4 = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    quantization_config=quantization_config,
)

print(f"\nINT4 Model loaded: {model_int4.__class__.__name__}")
print(f"Number of parameters: {model_int4.num_parameters():,}")
print(f"Number of labels: {model_int4.config.num_labels}")
print(f"Label mapping: {model_int4.config.label2id}")
print(f"Model quantized to 4-bit using bitsandbytes")

# Enable gradient checkpointing to save memory
model_int4.gradient_checkpointing_enable()
model_int4.config.use_cache = False

# Prepare model for k-bit training
from peft import prepare_model_for_kbit_training
model_int4 = prepare_model_for_kbit_training(model_int4)

print(f"Model prepared for 4-bit training")

stopword_factory = StopWordRemoverFactory()
indonesian_stopwords = stopword_factory.get_stop_words()

print(f"Loaded {len(indonesian_stopwords)} Indonesian stop words")
print(f"Sample stop words: {list(indonesian_stopwords)[:10]}")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # convert to lowercase
    text = text.lower()
    
    # remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # remove stop words
    words = text.split()
    words = [word for word in words if word not in indonesian_stopwords]
    text = ' '.join(words)
    
    return text

sample_text = "Ini adalah contoh kalimat! Apakah preprocessing berfungsi? 123 #test"
print(f"Original: {sample_text}")
print(f"Preprocessed: {preprocess_text(sample_text)}")

print("Applying preprocessing to dataset...")
print("Before preprocessing:")
print(f"  Sample text: {dataset['train'][0]['text'][:100]}...")

def preprocess_dataset(examples):
    """Apply preprocessing to a batch of examples"""
    examples['text'] = [preprocess_text(text) for text in examples['text']]
    return examples

dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    desc="Preprocessing text"
)

print("\nAfter preprocessing:")
print(f"  Sample text: {dataset['train'][0]['text'][:100]}...")
print("\nPreprocessing complete!")

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=['text']
)

print("Tokenization complete!")
print(f"Columns after tokenization: {tokenized_dataset['train'].column_names}")
print(f"\nSample tokenized data:")
print(f"  input_ids length: {len(tokenized_dataset['train'][0]['input_ids'])}")
print(f"  attention_mask length: {len(tokenized_dataset['train'][0]['attention_mask'])}")


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

output_dir = "./results/indobert-smsa-qat-int4-fake"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-4,  # Slightly higher LR for quantized training
    per_device_train_batch_size=8,  # Smaller batch size due to memory constraints
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    report_to="none",
    fp16=True,  # Use FP16 for compute
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
    push_to_hub=False
)

print("INT4 Training arguments configured:")
print(f"  Output directory: {output_dir}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Quantization: INT4 (4-bit NF4)")
print(f"  Optimizer: 8-bit AdamW")

trainer = Trainer(
    model=model_int4,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("INT4 Trainer initialized successfully!")

print("Starting INT4 training...")
print("Note: 4-bit training uses significantly less memory but may be slower")
print("="*70)

train_result = trainer.train()

print("\n" + "="*70)
print("INT4 Training completed!")
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

# Save the final model
model_save_path = "./models/indobert-smsa-qat-int4-fake"
os.makedirs(model_save_path, exist_ok=True)

print(f"\nSaving INT4 model to {model_save_path}...")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Model and tokenizer saved successfully!")
print("\nNote: This model is quantized and requires bitsandbytes for inference")
