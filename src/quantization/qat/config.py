from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import torch

from src.config import DEVICE


@dataclass
class QATConfig:
    model_id: str
    data_path: Path
    save_dir: Path
    prepared_checkpoint: Optional[Path] = None
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    device: torch.device = DEVICE

    def __post_init__(self) -> None:
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.prepared_checkpoint:
            self.prepared_checkpoint = Path(self.prepared_checkpoint)


@dataclass
class FinetuneQATConfig:
    model_id: str
    train_file: Path
    valid_file: Path
    test_file: Path
    save_dir: Path
    results_dir: Path
    num_labels: int = 3
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 128
    sample_frac: float = 1.0
    label2id: Dict[str, int] = field(default_factory=lambda: {
        'positive': 0, 'neutral': 1, 'negative': 2
    })
    device: torch.device = DEVICE

    def __post_init__(self) -> None:
        for attr in ('train_file', 'valid_file', 'test_file', 'save_dir', 'results_dir'):
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, Path(val))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.id2label = {v: k for k, v in self.label2id.items()}
