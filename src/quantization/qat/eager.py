import os
import re
import time
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
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FinetuneQATConfig


class EagerQATTrainer:
    def __init__(self, config: FinetuneQATConfig, quantization_type: str = "int8"):
        self.config = config
        self.quantization_type = quantization_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        stopword_factory = StopWordRemoverFactory()
        self.stopwords = stopword_factory.get_stop_words()

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)

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
        tokenizer = self.tokenizer
        max_length = self.config.max_length

        def preprocess_and_tokenize(batch):
            preprocessed = [preprocess(text) for text in batch["text"]]
            return tokenizer(
                preprocessed,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        tokenized = dataset.map(
            preprocess_and_tokenize, batched=True, remove_columns=['text']
        )
        return tokenized

    def train(self):
        print("=" * 70)
        print(
            f"{self.quantization_type.upper()} Eager QAT Training"
            " - SMSA Sentiment Analysis"
        )
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
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.fake_quantize.default_fake_quant,
            weight=torch.quantization.default_weight_fake_quant,
        )

        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            model.bert.embeddings.qconfig = None
            print("Embedding layer excluded from quantization")

        model_qat = quantization.prepare_qat(model, inplace=False)
        print("Model prepared for QAT (fake quantization)")

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
                'f1': f1,
            }

        output_dir = str(self.config.results_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to="none",
            fp16=False,
            push_to_hub=False,
        )

        print(f"Output directory: {output_dir}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Quantization: {self.quantization_type.upper()} eager (QAT)")

        trainer = Trainer(
            model=model_qat,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
        )

        print("\nStarting QAT training...")
        train_result = trainer.train()

        print("=" * 70)
        print("Training completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(
            f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds"
        )

        model_qat.eval()
        save_path = str(self.config.save_dir)
        os.makedirs(save_path, exist_ok=True)

        torch.save(model_qat.state_dict(), os.path.join(save_path, "model_qat.pth"))
        self.tokenizer.save_pretrained(save_path)

        qat_size = (
            os.path.getsize(os.path.join(save_path, "model_qat.pth")) / (1024 * 1024)
        )
        print(f"QAT model saved to: {save_path} ({qat_size:.2f} MB)")

        return train_result

    def export_to_onnx(self):
        import warnings

        warnings.filterwarnings('ignore')

        print("=" * 70)
        print(
            f"Exporting {self.quantization_type.upper()} QAT Model to ONNX Format"
        )
        print("=" * 70)

        save_path = str(self.config.save_dir)
        qat_state_dict = torch.load(
            os.path.join(save_path, "model_qat.pth"), map_location='cpu'
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=self.config.num_labels,
        )

        missing_keys, unexpected_keys = model.load_state_dict(
            qat_state_dict, strict=False
        )
        print("QAT-trained weights loaded")
        print(f"  Loaded: {len(qat_state_dict) - len(unexpected_keys)} parameters")
        print(f"  Skipped: {len(unexpected_keys)} fake quantization observers")

        model.eval()
        model = model.cpu()

        dummy_input = self.tokenizer(
            "Contoh teks untuk eksport",
            padding="max_length",
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt",
        )

        onnx_path = os.path.join(save_path, "model_qat.onnx")

        print(f"Exporting to ONNX (opset 14)...")
        print(f"Output: {onnx_path}")

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'},
                },
                opset_version=14,
                do_constant_folding=True,
                verbose=False,
            )

        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Export complete! Model size: {onnx_size:.2f} MB")

        return onnx_path

    def quantize_onnx(self):
        save_path = str(self.config.save_dir)
        model_fp32_onnx = os.path.join(save_path, "model_qat.onnx")

        if self.quantization_type == "int8":
            return self._quantize_onnx_int8(model_fp32_onnx)
        return self._quantize_onnx_fp16(model_fp32_onnx)

    def _quantize_onnx_int8(self, model_fp32_onnx):
        from onnxruntime.quantization import quantize_dynamic, QuantType

        model_int8_onnx = model_fp32_onnx.replace(".onnx", "_int8.onnx")

        print("=" * 70)
        print("Quantizing ONNX Model to Dynamic INT8")
        print("=" * 70)

        quantize_dynamic(
            model_input=model_fp32_onnx,
            model_output=model_int8_onnx,
            weight_type=QuantType.QInt8,
        )

        fp32_size = os.path.getsize(model_fp32_onnx) / (1024 * 1024)
        int8_size = os.path.getsize(model_int8_onnx) / (1024 * 1024)
        reduction = (1 - int8_size / fp32_size) * 100

        print(f"FP32 model: {fp32_size:.2f} MB")
        print(f"INT8 model: {int8_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")

        return model_int8_onnx

    def _quantize_onnx_fp16(self, model_fp32_onnx):
        import onnx
        from onnxconverter_common import float16

        model_fp16_onnx = model_fp32_onnx.replace(".onnx", "_fp16.onnx")

        print("=" * 70)
        print("Converting ONNX Model to FP16")
        print("=" * 70)

        onnx_model = onnx.load(model_fp32_onnx)
        onnx_model_fp16 = float16.convert_float_to_float16(
            onnx_model, keep_io_types=True
        )
        onnx.save(onnx_model_fp16, model_fp16_onnx)

        fp32_size = os.path.getsize(model_fp32_onnx) / (1024 * 1024)
        fp16_size = os.path.getsize(model_fp16_onnx) / (1024 * 1024)
        reduction = (1 - fp16_size / fp32_size) * 100

        print(f"FP32 model: {fp32_size:.2f} MB")
        print(f"FP16 model: {fp16_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")

        return model_fp16_onnx

    def evaluate_onnx(self, onnx_model_path=None):
        import onnxruntime as ort

        if onnx_model_path is None:
            save_path = str(self.config.save_dir)
            if self.quantization_type == "int8":
                onnx_model_path = os.path.join(save_path, "model_qat_int8.onnx")
            else:
                onnx_model_path = os.path.join(save_path, "model_qat_fp16.onnx")

        print("=" * 70)
        print(f"Evaluating {self.quantization_type.upper()} ONNX Model")
        print("=" * 70)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            onnx_model_path,
            sess_options,
            providers=['CPUExecutionProvider'],
        )
        print(f"ONNX model loaded: {onnx_model_path}")
        print(f"Provider: {session.get_providers()}")

        tokenized_dataset = self._load_and_preprocess(
            splits={'test': str(self.config.test_file)}
        )

        num_samples = len(tokenized_dataset['test'])
        print(f"Test samples: {num_samples:,}")

        predictions = []
        true_labels = []
        inference_times = []
        batch_size = self.config.batch_size

        print("\nRunning inference on test set...")

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch = tokenized_dataset['test'][i:batch_end]

            input_ids = np.array(batch['input_ids'], dtype=np.int64)
            attention_mask = np.array(batch['attention_mask'], dtype=np.int64)

            start_time = time.time()
            outputs = session.run(
                None,
                {'input_ids': input_ids, 'attention_mask': attention_mask},
            )
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

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )

        total_time = sum(inference_times)
        avg_time_per_batch = np.mean(inference_times) * 1000
        samples_per_second = num_samples / total_time

        print("\n" + "=" * 70)
        print(f"{self.quantization_type.upper()} ONNX Model Results")
        print("=" * 70)
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("=" * 70)
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Avg per batch ({batch_size} samples): {avg_time_per_batch:.2f}ms")
        print(f"Samples/second: {samples_per_second:.2f}")

        label_names = [
            self.config.id2label[i].capitalize()
            for i in range(self.config.num_labels)
        ]

        print("\nClassification Report:")
        print(
            classification_report(
                true_labels, predictions, target_names=label_names
            )
        )

        cm = confusion_matrix(true_labels, predictions)

        results_dir = str(self.config.results_dir)
        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - {self.quantization_type.upper()} ONNX')
        plt.tight_layout()
        cm_path = os.path.join(
            results_dir,
            f'confusion_matrix_{self.quantization_type}_eager.png',
        )
        plt.savefig(cm_path, dpi=300)
        plt.close()

        report_dict = classification_report(
            true_labels,
            predictions,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        results_data = {
            'model_type': self.quantization_type.upper(),
            'method': 'eager',
            'provider': session.get_providers()[0],
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'total_samples': int(num_samples),
                'total_time_seconds': float(total_time),
                'avg_time_per_batch_ms': float(avg_time_per_batch),
                'samples_per_second': float(samples_per_second),
            },
            'classification_report': report_dict,
        }

        results_path = os.path.join(
            results_dir,
            f'evaluation_results_{self.quantization_type}_eager.json',
        )
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"\nConfusion matrix saved to: {cm_path}")
        print(f"Evaluation results saved to: {results_path}")

        return results_data
