import torch
from src.config import QUANTIZATION_DTYPE, TARGET_LAYERS


class INT8Quantizer:
    @staticmethod
    def quantize(model):
        model_cpu = model.cpu()
        return torch.quantization.quantize_dynamic(model_cpu, TARGET_LAYERS, dtype=QUANTIZATION_DTYPE)
