from .trainer import QATTrainer, train_qat_seed, apply_qat_config, strip_observers, SentimentCSVDataset
from .config import QATConfig, FinetuneQATConfig


def __getattr__(name: str):
    if name == "EagerQATTrainer":
        from .eager import EagerQATTrainer
        return EagerQATTrainer
    if name == "FakeQATTrainer":
        from .fake import FakeQATTrainer
        return FakeQATTrainer
    _onnx_exports = {
        "qat_onnx_single_seed", "export_model_to_onnx",
        "quantize_onnx_int8", "quantize_onnx_fp16", "quantize_onnx_int4",
        "evaluate_onnx_on_csv",
    }
    if name in _onnx_exports:
        import importlib
        mod = importlib.import_module(".eager", package=__name__)
        return getattr(mod, name)
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
    "train_qat_seed",
    "apply_qat_config",
    "strip_observers",
    "SentimentCSVDataset",
    "qat_onnx_single_seed",
    "export_model_to_onnx",
    "quantize_onnx_int8",
    "quantize_onnx_fp16",
    "quantize_onnx_int4",
    "evaluate_onnx_on_csv",
]
