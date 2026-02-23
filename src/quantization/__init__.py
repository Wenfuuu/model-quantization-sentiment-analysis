from .ptq import PTQQuantizer
from .utils import save_quantized_model, load_quantized_model

def __getattr__(name: str):
    if name == "QATTrainer":
        from .qat import QATTrainer
        return QATTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["PTQQuantizer", "QATTrainer", "save_quantized_model", "load_quantized_model"]
