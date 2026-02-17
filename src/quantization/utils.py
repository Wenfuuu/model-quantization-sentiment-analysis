import torch
import os
from pathlib import Path


def save_quantized_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_quantized_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(str(path), weights_only=True))
    return model


def get_model_size(model_path):
    return os.path.getsize(str(model_path)) / (1024 * 1024)
