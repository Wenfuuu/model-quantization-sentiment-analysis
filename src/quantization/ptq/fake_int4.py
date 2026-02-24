import copy
import torch
import torch.nn as nn
from src.config import DEVICE


class FakeINT4Quantizer:
    @staticmethod
    def quantize(model):
        model_fake = copy.deepcopy(model)
        with torch.no_grad():
            for name, module in model_fake.named_modules():
                if isinstance(module, nn.Linear):
                    w = module.weight.data.float()
                    abs_max = w.abs().max()
                    if abs_max == 0:
                        continue
                    scale = abs_max / 7.0
                    module.weight.data = torch.fake_quantize_per_tensor_affine(
                        w, float(scale), 0, -8, 7
                    )
        model_fake = model_fake.to(DEVICE)
        model_fake.eval()
        return model_fake
