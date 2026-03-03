import os
import re
import json
import time

import numpy as np
import torch
import torch.quantization as quantization
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict, disable_caching
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FinetuneQATConfig
from src.utils import set_seed

class FakeQATTrainer:
    def __init__(self, config: FinetuneQATConfig, quantization_type: str = "int8"):
        self.config = config
        self.quantization_type = quantization_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_and_preprocess(self, splits=None):
        import pandas as pd
        disable_caching()
        if splits is None:
            splits = {
                'train': str(self.config.train_file),
                'validation': str(self.config.valid_file),
                'test': str(self.config.test_file),
            }

        label2id = self.config.label2id
        preprocess = self._preprocess_text
        tokenizer = self.tokenizer
        max_length = self.config.max_length

        dataset_dict = {}
        for split_name, split_path in splits.items():
            df_preview = pd.read_csv(split_path, sep='\t', nrows=1)
            if 'Tweet' in df_preview.columns and 'sentiment' in df_preview.columns:
                df = pd.read_csv(split_path, sep='\t', engine='python')
                if self.config.sample_frac < 1.0:
                    df = df.sample(frac=self.config.sample_frac, random_state=42).reset_index(drop=True)
                id2label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
                df['text'] = df['Tweet']
                df['label'] = df['sentiment'].map(id2label_map)
            else:
                df = pd.read_csv(split_path, sep='\t', header=None, names=['text', 'label'])
            df = df.dropna(subset=['text', 'label'])
            df['text'] = df['text'].apply(preprocess)
            df['label'] = df['label'].map(label2id)
            dataset_dict[split_name] = Dataset.from_pandas(df[['text', 'label']], preserve_index=False)

        dataset = DatasetDict(dataset_dict)

        def tokenize_fn(batch):
            return tokenizer(
                batch['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        return tokenized

    def _compute_metrics(self, eval_pred):
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
            'f1': f1,
        }

    def train(self):
        if self.quantization_type == "fp16":
            return self._train_fp16()
        if self.quantization_type == "int8":
            return self._train_int8()
        elif self.quantization_type == "int4":
            return self._train_int4()
        raise ValueError(f"Unknown quantization_type: {self.quantization_type!r}")

    def _train_int8(self):
        set_seed(42)
        print("=" * 70)
        print("INT8 Fake QAT Training - SMSA Sentiment Analysis")
        print("=" * 70)

        tokenized_dataset = self._load_and_preprocess()

        print(f"Train samples: {len(tokenized_dataset['train']):,}")
        print(f"Validation samples: {len(tokenized_dataset['validation']):,}")
        print(f"Test samples: {len(tokenized_dataset['test']):,}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
        )

        print(f"Model loaded: {model.num_parameters():,} parameters")

        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            model.bert.embeddings.qconfig = None
            print("Embedding layer excluded from quantization")

        quantization.prepare_qat(model, inplace=True)
        print("Model prepared for QAT (fake INT8 quantization)")

        output_dir = str(self.config.results_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="none",
            fp16=False,
            no_cuda=True,
            push_to_hub=False,
        )

        print(f"Output directory: {output_dir}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print("Quantization: INT8 (fake quant during training)")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=self._compute_metrics,
            processing_class=self.tokenizer,
        )

        print("\nStarting INT8 fake QAT training...")
        train_result = trainer.train()

        print("\n" + "=" * 70)
        print("INT8 Fake QAT Training completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(
            f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds"
        )
        print(
            f"Training samples/second:"
            f" {train_result.metrics['train_samples_per_second']:.2f}"
        )

        save_path = str(self.config.save_dir)
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        ptq_model_path = os.path.join(save_path, "model_int8.pth")
        torch.save(model.state_dict(), ptq_model_path)
        ptq_size = os.path.getsize(ptq_model_path) / (1024 * 1024)
        print(f"Model saved to: {save_path}")
        print(f"PTQ-compatible model saved: {ptq_model_path} ({ptq_size:.2f} MB)")

        return train_result

    def _train_fp16(self):
        set_seed(42)
        print("=" * 70)
        print("FP16 Fake QAT Training - SMSA Sentiment Analysis")
        print("=" * 70)

        tokenized_dataset = self._load_and_preprocess()

        print(f"Train samples: {len(tokenized_dataset['train']):,}")
        print(f"Validation samples: {len(tokenized_dataset['validation']):,}")
        print(f"Test samples: {len(tokenized_dataset['test']):,}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
        )

        print(f"Model loaded: {model.num_parameters():,} parameters")
        print("Model will use mixed precision training (FP16)")

        output_dir = str(self.config.results_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="none",
            fp16=False,
            no_cuda=True,
            push_to_hub=False,
        )

        print(f"Output directory: {output_dir}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print("Precision: FP16 (Mixed Precision)")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=self._compute_metrics,
            processing_class=self.tokenizer,
        )

        print("\nStarting FP16 mixed precision training...")
        train_result = trainer.train()

        print("\n" + "=" * 70)
        print("FP16 Training completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(
            f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds"
        )
        print(
            f"Training samples/second:"
            f" {train_result.metrics['train_samples_per_second']:.2f}"
        )

        save_path = str(self.config.save_dir)
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        ptq_model_path = os.path.join(save_path, "model_fp16.pth")
        torch.save(model.state_dict(), ptq_model_path)
        ptq_size = os.path.getsize(ptq_model_path) / (1024 * 1024)
        print(f"Model saved to: {save_path}")
        print(f"PTQ-compatible model saved: {ptq_model_path} ({ptq_size:.2f} MB)")

        return train_result

    def _train_int4(self):
        from torch.quantization.fake_quantize import FakeQuantize
        from torch.quantization.observer import MovingAverageMinMaxObserver

        set_seed(42)
        print("=" * 70)
        print("INT4 Fake QAT Training - SMSA Sentiment Analysis")
        print("=" * 70)

        tokenized_dataset = self._load_and_preprocess()

        print(f"Train samples: {len(tokenized_dataset['train']):,}")
        print(f"Validation samples: {len(tokenized_dataset['validation']):,}")
        print(f"Test samples: {len(tokenized_dataset['test']):,}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
        )

        print(f"Model loaded: {model.num_parameters():,} parameters")

        model.train()

        int4_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_fake_quant,
            weight=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=-8,
                quant_max=7,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
            ),
        )

        model.qconfig = int4_qconfig

        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            model.bert.embeddings.qconfig = None
            print("Embedding layer excluded from quantization")

        quantization.prepare_qat(model, inplace=True)
        print("Model prepared for QAT (4-bit symmetric weight fake quantization)")

        output_dir = str(self.config.results_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="none",
            fp16=False,
            no_cuda=True,
            push_to_hub=False,
        )

        print(f"Output directory: {output_dir}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print("Quantization: INT4 (4-bit symmetric weight fake quant)")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=self._compute_metrics,
            processing_class=self.tokenizer,
        )

        print("\nStarting INT4 fake QAT training...")
        train_result = trainer.train()

        print("\n" + "=" * 70)
        print("INT4 Fake QAT Training completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(
            f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds"
        )
        print(
            f"Training samples/second:"
            f" {train_result.metrics['train_samples_per_second']:.2f}"
        )

        save_path = str(self.config.save_dir)
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        ptq_model_path = os.path.join(save_path, "model_int4.pth")
        torch.save(model.state_dict(), ptq_model_path)
        ptq_size = os.path.getsize(ptq_model_path) / (1024 * 1024)
        print(f"Model saved to: {save_path}")
        print(f"PTQ-compatible model saved: {ptq_model_path} ({ptq_size:.2f} MB)")

        return train_result

    def evaluate(self, model_path=None, dataset_path=None):
        if model_path is None:
            model_path = str(self.config.save_dir)

        test_file = dataset_path if dataset_path else str(self.config.test_file)

        print("=" * 70)
        print(f"Evaluating {self.quantization_type.upper()} Fake QAT Model")
        print(f"Dataset: {test_file}")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.config.num_labels,
        )

        if self.quantization_type == "fp16":
            model = model.half()
            print("Applied FP16 precision reduction to model weights")
        elif self.quantization_type == "int4":
            from src.quantization.ptq.int4 import INT4Quantizer
            quantizer = INT4Quantizer(bits=4)
            model = quantizer.quantize_model(model)
            print("Applied INT4 symmetric weight-only quantization")

        model.eval()

        print(f"Model loaded from: {model_path}")
        print(f"Number of parameters: {model.num_parameters():,}")

        tokenized_dataset = self._load_and_preprocess(
            splits={'test': test_file}
        )

        print(f"Test samples: {len(tokenized_dataset['test']):,}")

        results_dir = str(self.config.results_dir)

        eval_args = TrainingArguments(
            output_dir=results_dir,
            overwrite_output_dir=True,
            per_device_eval_batch_size=self.config.batch_size,
            fp16=False,
            no_cuda=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=self._compute_metrics,
            processing_class=tokenizer,
        )

        print(
            f"Running evaluation on test set"
            f" ({self.quantization_type.upper()} fake QAT)..."
        )
        results = trainer.evaluate()

        print(f"\nTest Set Results ({self.quantization_type.upper()} Fake QAT)")
        print("=" * 70)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        print("=" * 70)

        predictions_output = trainer.predict(tokenized_dataset["test"])
        y_pred = predictions_output.predictions.argmax(-1)
        y_true = predictions_output.label_ids

        print("\nMeasuring per-sample inference latency...")
        device = next(model.parameters()).device
        per_sample_latencies = []
        batch_size = self.config.batch_size
        num_samples = len(tokenized_dataset["test"])

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch = tokenized_dataset["test"][i:batch_end]
                input_ids = torch.tensor(batch["input_ids"]).to(device)
                attention_mask = torch.tensor(batch["attention_mask"]).to(device)

                start_time = time.time()
                model(input_ids=input_ids, attention_mask=attention_mask)
                elapsed = time.time() - start_time

                batch_actual_size = batch_end - i
                per_sample_time = elapsed / batch_actual_size
                per_sample_latencies.extend([per_sample_time] * batch_actual_size)

        latency_stats = {
            'mean': float(np.mean(per_sample_latencies)),
            'std': float(np.std(per_sample_latencies)),
            'min': float(np.min(per_sample_latencies)),
            'max': float(np.max(per_sample_latencies)),
            'median': float(np.median(per_sample_latencies)),
        }
        print(f"Mean Latency: {latency_stats['mean']*1000:.2f} ms/sample")

        label_names = [
            self.config.id2label[i].capitalize()
            for i in range(self.config.num_labels)
        ]
        cm = confusion_matrix(y_true, y_pred)

        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(
            f"Confusion Matrix - {self.quantization_type.upper()} Fake QAT"
        )
        plt.tight_layout()
        cm_path = os.path.join(
            results_dir,
            f"confusion_matrix_{self.quantization_type}_fake.png",
        )
        plt.savefig(cm_path, dpi=300)
        plt.close()

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        print("\nDetailed Classification Report")
        print("=" * 70)
        print(classification_report(y_true, y_pred, target_names=label_names))
        print("=" * 70)

        results_path = os.path.join(
            results_dir,
            f"evaluation_results_{self.quantization_type}_fake.json",
        )
        with open(results_path, "w") as f:
            json.dump(
                {
                    "model_type": self.quantization_type.upper(),
                    "method": "fake",
                    "overall_metrics": results,
                    "latencies": [float(x) for x in per_sample_latencies],
                    "latency_stats": latency_stats,
                    "classification_report": report_dict,
                },
                f,
                indent=4,
            )

        print(f"\nConfusion matrix saved to: {cm_path}")
        print(f"Evaluation results saved to: {results_path}")

        return results
