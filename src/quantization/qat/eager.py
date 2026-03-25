import gc
import os
import re
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.quantization as quantization
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict, disable_caching
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

_LABEL_NAMES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
_NUM_LABELS = 3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FinetuneQATConfig
from src.utils import set_seed
from src.evaluation.calibration import expected_calibration_error

def _ece(confs, corr):
    return expected_calibration_error(confs, corr, n_bins=10)["ece"]


class EagerQATTrainer:
    def __init__(self, config: FinetuneQATConfig, quantization_type: str = "int8"):
        self.config = config
        self.quantization_type = quantization_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    @staticmethod
    def _get_rss_mb():
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    @staticmethod
    def _release_memory():
        gc.collect()

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

        model_int4_onnx = model_fp32_onnx.replace(".onnx", "_int4.onnx")

        print("=" * 70)
        print("Quantizing ONNX Model to INT4 (4-bit MatMul)")
        print("=" * 70)

        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

            onnx_model = onnx.load(model_fp32_onnx)
            quant = MatMulNBitsQuantizer(
                onnx_model,
                block_size=128,
                is_symmetric=True,
                bits=4,
            )
            quant.process()
            onnx.save(quant.model.model, model_int4_onnx)
        except ModuleNotFoundError:
            try:
                from onnxruntime.quantization import matmul_4bits_quantizer, quant_utils
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime with INT4 quantization support is required."
                ) from exc

            quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=128,
                is_symmetric=True,
                accuracy_level=4,
            )
            quant_model = quant_utils.load_model_with_shape_infer(Path(model_fp32_onnx))
            quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
                quant_model,
                nodes_to_exclude=None,
                nodes_to_include=None,
                algo_config=quant_config,
            )
            quant.process()
            quant.model.save_model_to_file(model_int4_onnx, True)

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

            outputs = session.run(
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

            per_sample_latencies.append(float(np.mean(sample_latencies)) if sample_latencies else 0.0)

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

    def evaluate_onnx(self, onnx_model_path=None, dataset_path=None, num_runs=20):
        import onnxruntime as ort
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
        warmup_runs = 5 if num_runs > 0 else 0
        label_names = [
            self.config.id2label[i].capitalize()
            for i in range(self.config.num_labels)
        ]

        print("=" * 70)
        print(f"Evaluating {self.quantization_type.upper()} ONNX Model (with FP32 Baseline)")
        print(f"Dataset: {test_file}")
        print("=" * 70)

        tokenized_dataset = self._load_and_preprocess(
            splits={'test': test_file}
        )
        num_samples = len(tokenized_dataset['test'])
        print(f"Test samples: {num_samples:,}")

        print(f"\n{'=' * 70}")
        print("FP32 ONNX Baseline Evaluation")
        print(f"{'=' * 70}")

        fp32_sess_options = ort.SessionOptions()
        fp32_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._release_memory()
        mem_before = self._get_rss_mb()
        fp32_session = ort.InferenceSession(
            fp32_onnx_path, fp32_sess_options,
            providers=['CPUExecutionProvider'],
        )
        _sample = tokenized_dataset['test'][0]
        fp32_session.run(
            None,
            {
                'input_ids': np.array([_sample['input_ids']], dtype=np.int64),
                'attention_mask': np.array([_sample['attention_mask']], dtype=np.int64),
            },
        )
        fp32_memory_mb = self._get_rss_mb() - mem_before
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
        self._release_memory()

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

        self._release_memory()
        mem_before = self._get_rss_mb()
        session = ort.InferenceSession(
            onnx_model_path, sess_options,
            providers=['CPUExecutionProvider'],
        )
        _sample = tokenized_dataset['test'][0]
        session.run(
            None,
            {
                'input_ids': np.array([_sample['input_ids']], dtype=np.int64),
                'attention_mask': np.array([_sample['attention_mask']], dtype=np.int64),
            },
        )
        quant_memory_mb = self._get_rss_mb() - mem_before
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


# ---------------------------------------------------------------------------
# Module-level helpers for multi-seed QAT-ONNX pipeline
# ---------------------------------------------------------------------------

def export_model_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    *,
    max_length: int = 128,
) -> Path:
    """Export a HuggingFace checkpoint directory to ONNX (opset 14)."""
    warnings.filterwarnings("ignore")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=_NUM_LABELS, ignore_mismatched_sizes=True,
    )
    model.eval().cpu()

    dummy = tokenizer(
        "Contoh teks untuk eksport",
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits":         {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False,
        )

    size_mb = os.path.getsize(str(onnx_path)) / (1024 * 1024)
    print(f"  ONNX exported: {onnx_path} ({size_mb:.2f} MB)")
    return onnx_path


def _print_size_reduction(label: str, fp32_path: Path, quant_path: Path) -> None:
    fp32_mb = os.path.getsize(str(fp32_path)) / (1024 * 1024)
    q_mb = os.path.getsize(str(quant_path)) / (1024 * 1024)
    red = (1 - q_mb / fp32_mb) * 100
    print(f"  {label}: {q_mb:.2f} MB  (reduction {red:.1f}%)")


def _create_onnx_session(onnx_path: Path, *, is_fp16: bool = False):
    """Create an ORT InferenceSession on CPU."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if is_fp16
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        str(onnx_path), opts, providers=["CPUExecutionProvider"],
    )


def quantize_onnx_int8(fp32_onnx: Path) -> Path:
    """Dynamic INT8 quantization via onnxruntime."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    out = fp32_onnx.with_name(fp32_onnx.stem + "_int8.onnx")
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    _print_size_reduction("INT8", fp32_onnx, out)
    return out


def quantize_onnx_fp16(fp32_onnx: Path) -> Path:
    """Convert ONNX model weights and graph to FP16."""
    import onnx
    from onnx import numpy_helper, TensorProto
    from onnxconverter_common import float16

    out = fp32_onnx.with_name(fp32_onnx.stem + "_fp16.onnx")

    onnx_model = onnx.load(str(fp32_onnx))
    model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=False)

    for init in model_fp16.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            data = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(data, init.name))

    for node in model_fp16.graph.node:
        for attr in node.attribute:
            if (attr.type == onnx.AttributeProto.TENSOR
                    and attr.t.data_type == TensorProto.FLOAT):
                data = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(data))
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16

    for gi in model_fp16.graph.input:
        if gi.type.tensor_type.elem_type == TensorProto.FLOAT:
            gi.type.tensor_type.elem_type = TensorProto.FLOAT16
    for go in model_fp16.graph.output:
        if go.type.tensor_type.elem_type == TensorProto.FLOAT:
            go.type.tensor_type.elem_type = TensorProto.FLOAT16

    while len(model_fp16.graph.value_info) > 0:
        model_fp16.graph.value_info.pop()

    onnx.save(model_fp16, str(out))
    _print_size_reduction("FP16", fp32_onnx, out)
    return out


def quantize_onnx_int4(fp32_onnx: Path) -> Path:
    """INT4 weight-only quantization via MatMulNBitsQuantizer."""
    import onnx

    out = fp32_onnx.with_name(fp32_onnx.stem + "_int4.onnx")

    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        model = onnx.load(str(fp32_onnx))
        quant = MatMulNBitsQuantizer(model, block_size=128, is_symmetric=True, bits=4)
        quant.process()
        onnx.save(quant.model.model, str(out))
    except (ModuleNotFoundError, ImportError):
        from onnxruntime.quantization import matmul_4bits_quantizer, quant_utils

        cfg = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=128, is_symmetric=True, accuracy_level=4,
        )
        qm = quant_utils.load_model_with_shape_infer(fp32_onnx)
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            qm, nodes_to_exclude=None, nodes_to_include=None, algo_config=cfg,
        )
        quant.process()
        quant.model.save_model_to_file(str(out), True)

    _print_size_reduction("INT4", fp32_onnx, out)
    return out


def evaluate_onnx_on_csv(
    onnx_path: Path,
    tokenizer,
    test_csv: Path,
    *,
    max_length: int = 128,
    is_fp16: bool = False,
) -> tuple:
    """Run ONNX inference over a preprocessed CSV and return metrics."""
    session = _create_onnx_session(onnx_path, is_fp16=is_fp16)

    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    true_labels = df["label"].astype(int).tolist()

    pred_labels = []
    all_probs = []

    for text in tqdm(texts, desc="  onnx eval", leave=False):
        enc = tokenizer(
            text, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="np",
        )
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = session.run(None, feed)[0]

        logits_f = logits.astype(np.float64)
        shifted = logits_f - logits_f.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        pred_labels.append(int(np.argmax(logits, axis=1)[0]))
        all_probs.append(probs[0].astype(np.float64))

    probs_arr = np.array(all_probs)

    cls_report = classification_report(
        true_labels, pred_labels, target_names=_LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "weighted_precision": cls_report["weighted avg"]["precision"],
        "weighted_recall": cls_report["weighted avg"]["recall"],
        "weighted_f1": cls_report["weighted avg"]["f1-score"],
        "per_class_f1": {lbl: cls_report[lbl]["f1-score"] for lbl in _LABEL_NAMES},
    }

    del session
    return true_labels, pred_labels, probs_arr, metrics


def qat_onnx_single_seed(
    seed: int,
    *,
    qat_clean_dir: Path,
    fp32_ckpt: Path,
    test_csv: Path,
    models_dir: Path,
    max_length: int = 128,
) -> dict:
    """Full QAT-ONNX pipeline for one seed: export → quantize → evaluate."""
    print(f"\n{'=' * 70}")
    print(f"#  QAT-ONNX  SEED {seed}")
    print(f"{'=' * 70}\n")

    if not qat_clean_dir.exists():
        raise FileNotFoundError(
            f"Clean QAT checkpoint not found: {qat_clean_dir}\n"
            "Run multi-seed QAT first."
        )

    tokenizer = AutoTokenizer.from_pretrained(qat_clean_dir)

    base_dir = models_dir / f"qat_onnx_seed{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = export_model_to_onnx(
        qat_clean_dir, base_dir / "model_qat.onnx", max_length=max_length,
    )

    fp32_pred = None
    fp32_pred_path = fp32_ckpt / "predictions.csv"
    if fp32_pred_path.exists():
        fp32_pred = pd.read_csv(fp32_pred_path)["pred_label"].astype(int).tolist()
    else:
        print(f"  [warn] FP32 predictions not found at {fp32_pred_path}")

    quant_fns = {
        "int8": (quantize_onnx_int8, False),
        "fp16": (quantize_onnx_fp16, True),
        "int4": (quantize_onnx_int4, False),
    }

    seed_results = {"seed": seed, "source_qat_checkpoint": str(qat_clean_dir)}

    for vname, (quant_fn, is_fp16) in quant_fns.items():
        print(f"\n  [{vname.upper()}] Quantizing ONNX ...")
        q_onnx = quant_fn(fp32_onnx)

        save_dir = models_dir / f"qat_onnx_{vname}_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dest_onnx = save_dir / q_onnx.name
        if dest_onnx != q_onnx:
            shutil.copy2(str(q_onnx), str(dest_onnx))

        print(f"  [{vname.upper()}] Evaluating with ONNX Runtime ...")
        true_labels, pred_labels, probs, metrics = evaluate_onnx_on_csv(
            dest_onnx, tokenizer, test_csv, max_length=max_length, is_fp16=is_fp16,
        )
        _conf = np.max(probs, axis=1).tolist()
        _corr = [int(p == t) for p, t in zip(pred_labels, true_labels)]
        metrics["ece"] = _ece(_conf, _corr)
        print(f"  {vname.upper()} accuracy={metrics['accuracy']:.4f}  "
              f"macro-F1={metrics['macro_f1']:.4f}  ECE={metrics['ece']:.4f}")

        agreement = None
        if fp32_pred is not None:
            n_agree = sum(a == b for a, b in zip(fp32_pred, pred_labels))
            agreement = n_agree / max(1, len(fp32_pred))
            print(f"  vs FP32 agreement: {agreement * 100:.2f}%")

        texts = pd.read_csv(test_csv)["text"].astype(str).tolist()
        pd.DataFrame({
            "sample_id": range(len(pred_labels)),
            "text": texts[:len(pred_labels)],
            "true_label": true_labels,
            "pred_label": pred_labels,
            "prob_pos": probs[:, 0],
            "prob_neu": probs[:, 1],
            "prob_neg": probs[:, 2],
        }).to_csv(save_dir / "predictions.csv", index=False, encoding="utf-8")

        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        model_size_mb = os.path.getsize(str(dest_onnx)) / (1024 * 1024)
        seed_results[vname] = {
            "metrics": metrics,
            "agreement_with_fp32": agreement,
            "model_size_mb": model_size_mb,
            "onnx_path": str(dest_onnx),
        }

    out_path = models_dir / f"qat_onnx_seed{seed}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(seed_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved -> {out_path}")

    return seed_results
