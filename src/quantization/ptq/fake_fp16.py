import copy
import torch
from src.config import DEVICE


class FakeFP16Quantizer:
    @staticmethod
    def quantize(model):
        model_fake = copy.deepcopy(model)
        with torch.no_grad():
            for param in model_fake.parameters():
                param.data = param.data.half().float()
        model_fake = model_fake.to(DEVICE)
        model_fake.eval()
        return model_fake
