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


class EagerQATTrainer:
    def __init__(self, config: FinetuneQATConfig, quantization_type: str = "int8"):
        self.config = config
        self.quantization_type = quantization_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
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

    def train(self):
        set_seed(42)
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
            activation=torch.quantization.default_fake_quant,
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
            overwrite_output_dir=True,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to="none",
            fp16=False,
            no_cuda=True,
            save_safetensors=False,
            push_to_hub=False,
            save_total_limit=1,
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
            processing_class=self.tokenizer,
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

        clean_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
        )
        qat_state = model_qat.state_dict()
        missing_keys, unexpected_keys = clean_model.load_state_dict(qat_state, strict=False)
        if missing_keys:
            print(f"WARNING: {len(missing_keys)} missing keys not loaded: {missing_keys[:5]}")
        print(f"Loaded {len(qat_state) - len(unexpected_keys)}/{len(qat_state)} keys, skipped {len(unexpected_keys)} QAT observer keys")
        clean_model.eval()

        ptq_model_path = os.path.join(save_path, f"model_{self.quantization_type}.pth")
        torch.save(clean_model.state_dict(), ptq_model_path)
        clean_model.save_pretrained(os.path.join(save_path, "hf_model"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "hf_model"))

        ptq_size = os.path.getsize(ptq_model_path) / (1024 * 1024)
        print(f"PTQ-compatible model saved: {ptq_model_path} ({ptq_size:.2f} MB)")
        print(f"HuggingFace model saved: {os.path.join(save_path, 'hf_model')}")

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
        elif self.quantization_type == "int4":
            return self._quantize_onnx_int4(model_fp32_onnx)
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
        from onnx import numpy_helper, TensorProto
        from onnxconverter_common import float16

        model_fp16_onnx = model_fp32_onnx.replace(".onnx", "_fp16.onnx")

        print("=" * 70)
        print("Converting ONNX Model to FP16")
        print("=" * 70)

        onnx_model = onnx.load(model_fp32_onnx)
        onnx_model_fp16 = float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=False,
        )

        for initializer in onnx_model_fp16.graph.initializer:
            if initializer.data_type == TensorProto.FLOAT:
                data = numpy_helper.to_array(initializer).astype(np.float16)
                new_init = numpy_helper.from_array(data, initializer.name)
                initializer.CopyFrom(new_init)

        for node in onnx_model_fp16.graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                    data = numpy_helper.to_array(attr.t).astype(np.float16)
                    new_tensor = numpy_helper.from_array(data)
                    attr.t.CopyFrom(new_tensor)
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == TensorProto.FLOAT:
                        attr.i = TensorProto.FLOAT16

        for graph_input in onnx_model_fp16.graph.input:
            if graph_input.type.tensor_type.elem_type == TensorProto.FLOAT:
                graph_input.type.tensor_type.elem_type = TensorProto.FLOAT16

        for graph_output in onnx_model_fp16.graph.output:
            if graph_output.type.tensor_type.elem_type == TensorProto.FLOAT:
                graph_output.type.tensor_type.elem_type = TensorProto.FLOAT16

        while len(onnx_model_fp16.graph.value_info) > 0:
            onnx_model_fp16.graph.value_info.pop()

        onnx.save(onnx_model_fp16, model_fp16_onnx)

        fp32_size = os.path.getsize(model_fp32_onnx) / (1024 * 1024)
        fp16_size = os.path.getsize(model_fp16_onnx) / (1024 * 1024)
        reduction = (1 - fp16_size / fp32_size) * 100

        print(f"FP32 model: {fp32_size:.2f} MB")
        print(f"FP16 model: {fp16_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")

        return model_fp16_onnx

    def _quantize_onnx_int4(self, model_fp32_onnx):
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        model_int4_onnx = model_fp32_onnx.replace(".onnx", "_int4.onnx")

        print("=" * 70)
        print("Quantizing ONNX Model to INT4 (4-bit MatMul)")
        print("=" * 70)

        onnx_model = onnx.load(model_fp32_onnx)
        quant = MatMulNBitsQuantizer(onnx_model, block_size=128, is_symmetric=True, bits=4)
        quant.process()
        onnx.save(quant.model.model, model_int4_onnx)

        fp32_size = os.path.getsize(model_fp32_onnx) / (1024 * 1024)
        int4_size = os.path.getsize(model_int4_onnx) / (1024 * 1024)
        reduction = (1 - int4_size / fp32_size) * 100

        print(f"FP32 model: {fp32_size:.2f} MB")
        print(f"INT4 model: {int4_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")

        return model_int4_onnx

    def _run_onnx_inference(self, session, tokenized_dataset, num_runs=20, warmup_runs=5):
        num_samples = len(tokenized_dataset['test'])
        predictions = []
        true_labels = []
        confidences = []
        per_sample_latencies = []

        for i in range(num_samples):
            sample = tokenized_dataset['test'][i]
            input_ids = np.array([sample['input_ids']], dtype=np.int64)
            attention_mask = np.array([sample['attention_mask']], dtype=np.int64)

            for _ in range(warmup_runs):
                session.run(
                    None,
                    {'input_ids': input_ids, 'attention_mask': attention_mask},
                )

            sample_latencies = []
            for _ in range(num_runs):
                start_time = time.time()
                outputs = session.run(
                    None,
                    {'input_ids': input_ids, 'attention_mask': attention_mask},
                )
                elapsed = time.time() - start_time
                sample_latencies.append(elapsed)

            per_sample_latencies.append(float(np.mean(sample_latencies)))

            logits = outputs[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            confidences.append(float(np.max(probs)))
            predictions.append(int(np.argmax(logits, axis=1)[0]))
            true_labels.append(sample['label'])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples...")

        return np.array(predictions), np.array(true_labels), confidences, per_sample_latencies

    def _compute_onnx_metrics(self, predictions, true_labels, confidences, per_sample_latencies):
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        total_time = sum(per_sample_latencies)
        num_samples = len(true_labels)
        samples_per_second = num_samples / total_time if total_time > 0 else 0
        latency_stats = {
            'mean': float(np.mean(per_sample_latencies)),
            'std': float(np.std(per_sample_latencies)),
            'min': float(np.min(per_sample_latencies)),
            'max': float(np.max(per_sample_latencies)),
            'median': float(np.median(per_sample_latencies)),
        }
        label_names = [
            self.config.id2label[i].capitalize()
            for i in range(self.config.num_labels)
        ]
        report_dict = classification_report(
            true_labels, predictions, target_names=label_names,
            output_dict=True, zero_division=0,
        )
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'avg_confidence': float(np.mean(confidences)),
            'total_samples': int(num_samples),
            'total_time_seconds': float(total_time),
            'samples_per_second': float(samples_per_second),
        }
        return metrics, latency_stats, report_dict

    def _save_confusion_matrix(self, true_labels, predictions, label_names, title, save_path):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
        )
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return cm

    def evaluate_onnx(self, onnx_model_path=None, dataset_path=None):
        import onnxruntime as ort
        import psutil

        if onnx_model_path is None:
            save_path = str(self.config.save_dir)
            if self.quantization_type == "fp32":
                onnx_model_path = os.path.join(save_path, "model_qat.onnx")
            elif self.quantization_type == "int8":
                onnx_model_path = os.path.join(save_path, "model_qat_int8.onnx")
            elif self.quantization_type == "int4":
                onnx_model_path = os.path.join(save_path, "model_qat_int4.onnx")
            else:
                onnx_model_path = os.path.join(save_path, "model_qat_fp16.onnx")

        test_file = dataset_path if dataset_path else str(self.config.test_file)
        save_path = str(self.config.save_dir)
        fp32_onnx_path = os.path.join(save_path, "model_qat.onnx")
        results_dir = str(self.config.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        num_runs = 20
        warmup_runs = 5
        label_names = [
            self.config.id2label[i].capitalize()
            for i in range(self.config.num_labels)
        ]

        print("=" * 70)
        print(f"Evaluating {self.quantization_type.upper()} ONNX Model (with FP32 Baseline)")
        print(f"Dataset: {test_file}")
        print("=" * 70)

        process = psutil.Process(os.getpid())

        tokenized_dataset = self._load_and_preprocess(
            splits={'test': test_file}
        )
        num_samples = len(tokenized_dataset['test'])
        print(f"Test samples: {num_samples:,}")

        print(f"\n{'=' * 70}")
        print("FP32 ONNX Baseline Evaluation")
        print(f"{'=' * 70}")

        mem_before_fp32 = process.memory_info().rss / (1024 * 1024)
        fp32_sess_options = ort.SessionOptions()
        fp32_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        fp32_session = ort.InferenceSession(
            fp32_onnx_path, fp32_sess_options,
            providers=['CPUExecutionProvider'],
        )
        mem_after_fp32 = process.memory_info().rss / (1024 * 1024)
        fp32_memory_mb = mem_after_fp32 - mem_before_fp32
        print(f"FP32 ONNX model loaded: {fp32_onnx_path}")

        print(f"\nRunning FP32 inference ({warmup_runs} warm-up + {num_runs} timed runs per sample)...")
        fp32_preds, fp32_labels, fp32_confidences, fp32_latencies = self._run_onnx_inference(
            fp32_session, tokenized_dataset, num_runs, warmup_runs
        )
        fp32_metrics, fp32_latency_stats, fp32_report = self._compute_onnx_metrics(
            fp32_preds, fp32_labels, fp32_confidences, fp32_latencies
        )

        print(f"\nFP32 ONNX Baseline Results:")
        print(f"  Accuracy:  {fp32_metrics['accuracy']:.4f}")
        print(f"  Precision: {fp32_metrics['precision']:.4f}")
        print(f"  Recall:    {fp32_metrics['recall']:.4f}")
        print(f"  F1 Score:  {fp32_metrics['f1']:.4f}")
        print(f"  Mean Latency: {fp32_latency_stats['mean']*1000:.2f} ms/sample")

        print("\nFP32 Classification Report:")
        print(classification_report(fp32_labels, fp32_preds, target_names=label_names))

        fp32_cm_path = os.path.join(results_dir, 'confusion_matrix_fp32_eager.png')
        self._save_confusion_matrix(
            fp32_labels, fp32_preds, label_names,
            'Confusion Matrix - FP32 ONNX Baseline (Eager)', fp32_cm_path,
        )
        print(f"FP32 confusion matrix saved to: {fp32_cm_path}")

        del fp32_session

        if self.quantization_type == "fp32":
            fp32_results_data = {
                'model_type': 'FP32',
                'method': 'eager',
                'provider': 'CPUExecutionProvider',
                'memory_usage_mb': fp32_memory_mb,
                'overall_metrics': fp32_metrics,
                'latencies': [float(x) for x in fp32_latencies],
                'latency_stats': fp32_latency_stats,
                'classification_report': fp32_report,
            }
            fp32_results_path = os.path.join(results_dir, 'evaluation_results_fp32_eager.json')
            with open(fp32_results_path, 'w') as f:
                json.dump(fp32_results_data, f, indent=4)
            print(f"\nFP32 results saved to: {fp32_results_path}")
            return fp32_results_data

        print(f"\n{'=' * 70}")
        print(f"{self.quantization_type.upper()} ONNX Quantized Evaluation")
        print(f"{'=' * 70}")

        sess_options = ort.SessionOptions()
        if self.quantization_type == "fp16":
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        else:
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        mem_before_quant = process.memory_info().rss / (1024 * 1024)
        session = ort.InferenceSession(
            onnx_model_path, sess_options,
            providers=['CPUExecutionProvider'],
        )
        mem_after_quant = process.memory_info().rss / (1024 * 1024)
        quant_memory_mb = mem_after_quant - mem_before_quant
        print(f"Quantized ONNX model loaded: {onnx_model_path}")
        print(f"Provider: {session.get_providers()}")

        print(f"\nRunning {self.quantization_type.upper()} inference ({warmup_runs} warm-up + {num_runs} timed runs per sample)...")
        q_preds, q_labels, q_confidences, q_latencies = self._run_onnx_inference(
            session, tokenized_dataset, num_runs, warmup_runs
        )
        q_metrics, q_latency_stats, q_report = self._compute_onnx_metrics(
            q_preds, q_labels, q_confidences, q_latencies
        )

        print(f"\n{self.quantization_type.upper()} ONNX Results:")
        print(f"  Accuracy:  {q_metrics['accuracy']:.4f}")
        print(f"  Precision: {q_metrics['precision']:.4f}")
        print(f"  Recall:    {q_metrics['recall']:.4f}")
        print(f"  F1 Score:  {q_metrics['f1']:.4f}")
        print(f"  Mean Latency: {q_latency_stats['mean']*1000:.2f} ms/sample")

        print(f"\n{self.quantization_type.upper()} Classification Report:")
        print(classification_report(q_labels, q_preds, target_names=label_names))

        q_cm_path = os.path.join(
            results_dir,
            f'confusion_matrix_{self.quantization_type}_eager.png',
        )
        self._save_confusion_matrix(
            q_labels, q_preds, label_names,
            f'Confusion Matrix - {self.quantization_type.upper()} ONNX (Eager)', q_cm_path,
        )

        print(f"\n{'=' * 70}")
        print("FP32 vs Quantized Comparison")
        print(f"{'=' * 70}")
        print(f"  FP32 Accuracy:     {fp32_metrics['accuracy']:.4f}")
        print(f"  {self.quantization_type.upper()} Accuracy:  {q_metrics['accuracy']:.4f}")
        print(f"  Accuracy Delta:    {q_metrics['accuracy'] - fp32_metrics['accuracy']:+.4f}")
        print(f"  FP32 F1:           {fp32_metrics['f1']:.4f}")
        print(f"  {self.quantization_type.upper()} F1:        {q_metrics['f1']:.4f}")
        print(f"  F1 Delta:          {q_metrics['f1'] - fp32_metrics['f1']:+.4f}")
        print(f"  FP32 Latency:      {fp32_latency_stats['mean']*1000:.2f} ms")
        print(f"  {self.quantization_type.upper()} Latency:   {q_latency_stats['mean']*1000:.2f} ms")

        fp32_results_data = {
            'model_type': 'FP32',
            'method': 'eager',
            'provider': 'CPUExecutionProvider',
            'memory_usage_mb': fp32_memory_mb,
            'overall_metrics': fp32_metrics,
            'latencies': [float(x) for x in fp32_latencies],
            'latency_stats': fp32_latency_stats,
            'classification_report': fp32_report,
        }

        fp32_results_path = os.path.join(results_dir, 'evaluation_results_fp32_eager.json')
        with open(fp32_results_path, 'w') as f:
            json.dump(fp32_results_data, f, indent=4)

        results_data = {
            'model_type': self.quantization_type.upper(),
            'method': 'eager',
            'provider': session.get_providers()[0],
            'memory_usage_mb': quant_memory_mb,
            'fp32_memory_usage_mb': fp32_memory_mb,
            'fp32_metrics': fp32_metrics,
            'fp32_latency_stats': fp32_latency_stats,
            'fp32_latencies': [float(x) for x in fp32_latencies],
            'fp32_classification_report': fp32_report,
            'overall_metrics': q_metrics,
            'latencies': [float(x) for x in q_latencies],
            'latency_stats': q_latency_stats,
            'classification_report': q_report,
        }

        results_path = os.path.join(
            results_dir,
            f'evaluation_results_{self.quantization_type}_eager.json',
        )
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"\nConfusion matrix saved to: {q_cm_path}")
        print(f"Evaluation results saved to: {results_path}")
        print(f"FP32 results saved to: {fp32_results_path}")

        return results_data
