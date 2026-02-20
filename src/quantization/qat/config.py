from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
