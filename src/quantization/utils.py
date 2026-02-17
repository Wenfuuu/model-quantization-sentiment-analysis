import torch
import os


def save_quantized_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_quantized_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model


def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)
