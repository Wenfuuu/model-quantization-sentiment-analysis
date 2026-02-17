import time
from .dynamic import DynamicQuantizer
from .fp16 import FP16Quantizer
from .int4 import INT4Quantizer


class PTQQuantizer:
    def __init__(self, model):
        self.model = model

    def quantize_fp16(self):
        start_time = time.perf_counter()
        quantized_model = FP16Quantizer.quantize(self.model)
        end_time = time.perf_counter()
        return quantized_model, end_time - start_time

    def quantize_int8(self):
        start_time = time.perf_counter()
        quantized_model = DynamicQuantizer.quantize(self.model)
        end_time = time.perf_counter()
        return quantized_model, end_time - start_time

    def quantize_int4(self):
        start_time = time.perf_counter()
        quantizer = INT4Quantizer(bits=4)
        quantized_model = quantizer.quantize_model(self.model)
        quantized_model.eval()
        end_time = time.perf_counter()
        return quantized_model, end_time - start_time
