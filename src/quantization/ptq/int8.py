import torch
from src.config import QUANTIZATION_DTYPE, TARGET_LAYERS


class INT8Quantizer:
    @staticmethod
    def quantize(model):
        return torch.quantization.quantize_dynamic(model, TARGET_LAYERS, dtype=QUANTIZATION_DTYPE)
