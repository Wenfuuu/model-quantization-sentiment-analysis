import os
import re
import json

import numpy as np
import torch
import torch.quantization as quantization
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
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
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_and_preprocess(self, splits=None):
        if splits is None:
            splits = {
                'train': str(self.config.train_file),
                'validation': str(self.config.valid_file),
                'test': str(self.config.test_file),
            }

        dataset = load_dataset(
            'csv',
            data_files=splits,
            delimiter='\t',
            column_names=['text', 'label'],
        )

        label2id = self.config.label2id

        def map_labels(df):
            df['label'] = [label2id[label] for label in df['label']]
            return df

        dataset = dataset.map(map_labels, batched=True)

        preprocess = self._preprocess_text

        def preprocess_dataset(examples):
            examples['text'] = [preprocess(text) for text in examples['text']]
            return examples

        dataset = dataset.map(preprocess_dataset, batched=True)

        tokenizer = self.tokenizer
        max_length = self.config.max_length

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        tokenized = dataset.map(
            tokenize_fn, batched=True, remove_columns=['text']
        )
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
            raise NotImplementedError(
                "FP16 QAT has been retired. PyTorch's fbgemm qconfig applies "
                "INT8 quantization regardless of the 'fp16' label — FP16 QAT "
                "is not a meaningful operation in this backend. "
                "Use PTQ-FP16 (model.half()) for FP16 inference instead."
            )
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
            tokenizer=self.tokenizer,
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

        print(f"Model saved to: {save_path}")

        return train_result

    def _train_fp16(self):
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
            fp16=True,
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
            tokenizer=self.tokenizer,
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

        print(f"Model saved to: {save_path}")

        return train_result

    def _train_int4(self):
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

        set_seed(42)
        print("=" * 70)
        print("INT4 Fake QAT Training - SMSA Sentiment Analysis")
        print("=" * 70)

        tokenized_dataset = self._load_and_preprocess()

        print(f"Train samples: {len(tokenized_dataset['train']):,}")
        print(f"Validation samples: {len(tokenized_dataset['validation']):,}")
        print(f"Test samples: {len(tokenized_dataset['test']):,}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        print(f"INT4 Model loaded: {model.num_parameters():,} parameters")

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        output_dir = str(self.config.results_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="none",
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            push_to_hub=False,
        )

        print(f"Output directory: {output_dir}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print("Quantization: INT4 (4-bit NF4 with LoRA)")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer,
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

        print(f"Model saved to: {save_path}")

        return train_result

    def evaluate(self, model_path=None):
        if model_path is None:
            model_path = str(self.config.save_dir)

        print("=" * 70)
        print(f"Evaluating {self.quantization_type.upper()} Fake QAT Model")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.quantization_type == "int4":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                quantization_config=quantization_config,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

        model.eval()

        print(f"Model loaded from: {model_path}")
        print(f"Number of parameters: {model.num_parameters():,}")

        tokenized_dataset = self._load_and_preprocess(
            splits={'test': str(self.config.test_file)}
        )

        print(f"Test samples: {len(tokenized_dataset['test']):,}")

        use_fp16 = self.quantization_type in ("fp16", "int4")

        results_dir = str(self.config.results_dir)

        eval_args = TrainingArguments(
            output_dir=results_dir,
            per_device_eval_batch_size=self.config.batch_size,
            fp16=use_fp16,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=self._compute_metrics,
            tokenizer=tokenizer,
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
                    "classification_report": report_dict,
                },
                f,
                indent=4,
            )

        print(f"\nConfusion matrix saved to: {cm_path}")
        print(f"Evaluation results saved to: {results_path}")

        return results
