import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from src.config import (
    HF_DATASET_PATH,
    LABELS,
    DEVICE,
    FP32_MODEL_DIR,
    PTQ_MODEL_PATH,
    QAT_MODEL_PATH,
)

from src.evaluation.batch_eval import (
    build_dataloader,
    evaluate_accuracy,
    evaluate_latency,
    load_dataset,
)

def load_fp32(model_dir: Path, device: torch.device) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=len(LABELS))
    model.to(device).eval()
    return model

def load_ptq(base_model_dir: Path, state_path: Path, device: torch.device) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=len(LABELS))
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.to(device)
    return model

def load_qat(base_model_dir: Path, state_path: Path, device: torch.device) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=len(LABELS))
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate FP32/PTQ/QAT models for accuracy and latency")
    parser.add_argument("--data", type=Path, default=HF_DATASET_PATH, help="Path to Hugging Face dataset (processed)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for accuracy evaluation")
    parser.add_argument("--runs", type=int, default=100, help="Latency measurement runs")
    parser.add_argument("--warmup", type=int, default=10, help="Latency warmup iterations")
    parser.add_argument("--fp32-dir", type=Path, default=FP32_MODEL_DIR, help="Directory of FP32 checkpoint (save_pretrained)")
    parser.add_argument("--ptq-path", type=Path, default=PTQ_MODEL_PATH, help="Path to PTQ state dict")
    parser.add_argument("--qat-path", type=Path, default=QAT_MODEL_PATH, help="Path to QAT state dict")
    args = parser.parse_args()

    device = DEVICE

    dataset = load_dataset(args.data, split="test")
    dataloader = build_dataloader(dataset, batch_size=args.batch_size)

    print("\nLoading models...")
    fp32 = load_fp32(args.fp32_dir, device)
    ptq = load_ptq(args.fp32_dir, args.ptq_path, device)
    qat = load_qat(args.fp32_dir, args.qat_path, device)

    print("\nEvaluating accuracy...")
    acc_fp32 = evaluate_accuracy(fp32, dataloader, device)
    acc_ptq = evaluate_accuracy(ptq, dataloader, device)
    acc_qat = evaluate_accuracy(qat, dataloader, device)

    print("\nMeasuring latency (ms)...")
    sample = dataset[0]
    lat_fp32 = evaluate_latency(fp32, sample, runs=args.runs, warmup=args.warmup, device=device)
    lat_ptq = evaluate_latency(ptq, sample, runs=args.runs, warmup=args.warmup, device=device)
    lat_qat = evaluate_latency(qat, sample, runs=args.runs, warmup=args.warmup, device=device)

    size_fp32 = (args.fp32_dir / "model.safetensors").stat().st_size / 1024 / 1024 if (args.fp32_dir / "model.safetensors").exists() else 0.0
    size_ptq = args.ptq_path.stat().st_size / 1024 / 1024 if args.ptq_path.exists() else 0.0
    size_qat = args.qat_path.stat().st_size / 1024 / 1024 if args.qat_path.exists() else 0.0

    print("\n==== RESULTS ====")
    print(f"FP32 Accuracy : {acc_fp32:.4f}")
    print(f"PTQ  Accuracy : {acc_ptq:.4f}")
    print(f"QAT  Accuracy : {acc_qat:.4f}\n")

    print(f"FP32 Latency (ms): {lat_fp32:.2f}")
    print(f"PTQ  Latency (ms): {lat_ptq:.2f}")
    print(f"QAT  Latency (ms): {lat_qat:.2f}\n")

    print(f"FP32 Size (MB): {size_fp32:.1f}")
    print(f"PTQ  Size (MB): {size_ptq:.1f}")
    print(f"QAT  Size (MB): {size_qat:.1f}")

if __name__ == "__main__":
    main()
