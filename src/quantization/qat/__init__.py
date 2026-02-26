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


__all__ = [
    "QATTrainer",
    "QATConfig",
    "FinetuneQATConfig",
    "EagerQATTrainer",
    "FakeQATTrainer",
]
