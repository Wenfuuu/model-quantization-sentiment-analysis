from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from src.config import DEVICE

def load_dataset(split_dir: Path, split: str = "test"):
    dataset = load_from_disk(str(split_dir))
    ds_split = dataset[split]
    for cols in (["input_ids", "attention_mask", "label"], ["input_ids", "attention_mask", "labels"]):
        try:
            ds_split = ds_split.with_format("torch", columns=cols)
            break
        except Exception:
            continue
    return ds_split

def build_dataloader(dataset, batch_size: int = 16, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: torch.device = DEVICE) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "labels" in batch:
                labels = batch["labels"].to(device)
            elif "label" in batch:
                labels = batch["label"].to(device)
            else:
                raise KeyError("Batch must contain labels or label column")

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(preds)

    return correct / max(1, total)

def evaluate_latency(
    model: torch.nn.Module,
    sample: dict,
    runs: int = 100,
    warmup: int = 10,
    device: torch.device = DEVICE,
) -> float:
    model.eval()

    ids = sample["input_ids"].unsqueeze(0).to(device)
    mask = sample["attention_mask"].unsqueeze(0).to(device)

    for _ in range(max(0, warmup)):
        _ = model(input_ids=ids, attention_mask=mask)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

            if start and end:
                start.record()
                _ = model(input_ids=ids, attention_mask=mask)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0
            else:
                import time

                t0 = time.perf_counter()
                _ = model(input_ids=ids, attention_mask=mask)
                elapsed = time.perf_counter() - t0

            times.append(elapsed)

    return float(np.mean(times) * 1000)  # milliseconds

def load_quantized_model(base_model_id: str, state_path: Path, device: torch.device = DEVICE) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(base_model_id)
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model