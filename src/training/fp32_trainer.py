from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.config import DEVICE, LABELS

@dataclass
class FP32TrainConfig:
    model_id: str
    data_path: Path
    save_dir: Path
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    device: torch.device = DEVICE

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)


class FP32Trainer:
    def __init__(self, config: FP32TrainConfig):
        self.config = config
        self.device = config.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            num_labels=len(LABELS),
        ).to(self.device)

        self.dataset = load_from_disk(str(config.data_path))
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.train_loader = DataLoader(
            self.dataset["train"],
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        self.val_loader = DataLoader(
            self.dataset["validation"],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = int(config.warmup_ratio * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _move_batch(self, batch: dict) -> dict:
        batch = dict(batch)
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            batch = self._move_batch(batch)
            outputs = self.model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    @torch.inference_mode()
    def _evaluate(self) -> float:
        self.model.eval()
        preds, labels = [], []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = self._move_batch(batch)
            outputs = self.model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

        return float(np.mean(np.array(preds) == np.array(labels)))

    def train(self) -> Tuple[float, Path]:
        best_acc = 0.0
        best_path = None

        for epoch in range(self.config.epochs):
            print(f"\n--- FP32 Epoch {epoch + 1}/{self.config.epochs} ---")
            train_loss = self._train_epoch()
            val_acc = self._evaluate()
            print(f"Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = self.config.save_dir / "fp32_trained"
                self.model.save_pretrained(best_path)
                self.tokenizer.save_pretrained(best_path)
                print(f"Saved best checkpoint to {best_path}")

        print(f"\nTraining complete. Best Accuracy: {best_acc:.4f}")
        return best_acc, best_path
