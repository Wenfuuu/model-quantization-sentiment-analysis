from .trainer import QATTrainer
from .config import QATConfig, FinetuneQATConfig


def __getattr__(name: str):
    if name == "EagerQATTrainer":
        from .eager import EagerQATTrainer
        return EagerQATTrainer
    if name == "FakeQATTrainer":
        from .fake import FakeQATTrainer
        return FakeQATTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_qat_model_for_xai(model_path, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src.models.base import BaseModel
    from src.config import DEVICE
    
    if device is None:
        device = DEVICE
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    return BaseModel(model, tokenizer, device=device)


__all__ = [
    "QATTrainer",
    "QATConfig",
    "FinetuneQATConfig",
    "EagerQATTrainer",
    "FakeQATTrainer",
    "load_qat_model_for_xai",
]
