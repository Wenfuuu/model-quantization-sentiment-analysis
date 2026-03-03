from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import LABELS, DEVICE
from .config import QATConfig


class QATTrainer:
    def __init__(self, config: QATConfig):
        self.config = config
        self.device = config.device or DEVICE

        self.dataset = load_from_disk(str(config.data_path))
        self.train_loader = DataLoader(
            self.dataset["train"],
            batch_size=config.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.dataset["validation"],
            batch_size=config.batch_size,
            shuffle=False,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            num_labels=len(LABELS),
        )

        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(self.model, inplace=True)

        if config.prepared_checkpoint:
            state = torch.load(config.prepared_checkpoint, map_location="cpu")
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)

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

        for batch in tqdm(self.train_loader, desc="QAT Training"):
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

        for batch in tqdm(self.val_loader, desc="QAT Eval"):
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
            print(f"\n--- QAT Epoch {epoch + 1}/{self.config.epochs} ---")
            train_loss = self._train_epoch()
            val_acc = self._evaluate()
            print(f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = self.config.save_dir / "qat_trained.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best QAT checkpoint to {best_path}")

        print(f"\nQAT complete. Best Validation Accuracy: {best_acc:.4f}")
        return best_acc, best_path
