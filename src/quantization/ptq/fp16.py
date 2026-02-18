import copy
from src.config import DEVICE


class FP16Quantizer:
    @staticmethod
    def quantize(model):
        model_fp16 = copy.deepcopy(model)
        model_fp16 = model_fp16.half()
        model_fp16 = model_fp16.to(DEVICE)
        model_fp16.eval()
        return model_fp16
