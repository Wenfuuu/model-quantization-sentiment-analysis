import torch
import torch.nn as nn
import copy


class INT4Quantizer:
    def __init__(self, bits=4):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_tensor(self, tensor):
        tensor_float = tensor.float()
        abs_max = tensor_float.abs().max()
        if abs_max == 0:
            return torch.zeros_like(tensor_float, dtype=torch.int8), torch.tensor(1.0)
        scale = abs_max / self.qmax
        quantized = torch.clamp(torch.round(tensor_float / scale), self.qmin, self.qmax)
        quantized = quantized.to(torch.int8)
        return quantized, scale

    def dequantize_tensor(self, quantized, scale):
        return quantized.float() * scale

    def quantize_model(self, model):
        model_int4 = copy.deepcopy(model)

        def replace_linear_with_int4(module, name=''):
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    int4_linear = INT4Linear(child, self)
                    setattr(module, child_name, int4_linear)
                else:
                    replace_linear_with_int4(child, f"{name}.{child_name}" if name else child_name)

        replace_linear_with_int4(model_int4)
        return model_int4


class INT4Linear(nn.Module):
    def __init__(self, original_linear, quantizer):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        weight_quantized, weight_scale = quantizer.quantize_tensor(original_linear.weight.data)
        self.register_buffer('weight_quantized', weight_quantized)
        self.register_buffer('weight_scale', weight_scale)
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.register_buffer('bias', None)
        self.quantizer = quantizer

    def forward(self, x):
        weight_dequant = self.quantizer.dequantize_tensor(self.weight_quantized, self.weight_scale)
        return nn.functional.linear(x, weight_dequant, self.bias)
