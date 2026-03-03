from .ptq import PTQQuantizer
from .utils import save_quantized_model, load_quantized_model

def __getattr__(name: str):
    if name == "QATTrainer":
        from .qat import QATTrainer
        return QATTrainer
    if name == "EagerQATTrainer":
        from .qat import EagerQATTrainer
        return EagerQATTrainer
    if name == "FakeQATTrainer":
        from .qat import FakeQATTrainer
        return FakeQATTrainer
    if name == "FinetuneQATConfig":
        from .qat import FinetuneQATConfig
        return FinetuneQATConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PTQQuantizer",
    "QATTrainer",
    "EagerQATTrainer",
    "FakeQATTrainer",
    "FinetuneQATConfig",
    "save_quantized_model",
    "load_quantized_model",
]
