"""
Apply dynamic INT8 quantization to ONNX model
"""
import os

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("ERROR: onnxruntime not installed")
    print("Install: pip install onnxruntime")
    exit(1)

print("="*70)
print("Quantizing ONNX Model to Dynamic INT8")
print("="*70)

base_path = "./models/indobert-qat-int8-smsa"
model_fp32 = f"{base_path}/model_qat.onnx"
model_int8 = f"{base_path}/model_qat_int8.onnx"

if not os.path.exists(model_fp32):
    print(f"ERROR: ONNX model not found at {model_fp32}")
    print("Run export_to_onnx.py first!")
    exit(1)

print(f"Input:  {model_fp32}")
print(f"Output: {model_int8}")
print("\nApplying dynamic INT8 quantization...")

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8
)

# Compare sizes
fp32_size = os.path.getsize(model_fp32) / (1024 * 1024)
int8_size = os.path.getsize(model_int8) / (1024 * 1024)
reduction = (1 - int8_size / fp32_size) * 100

print("\n" + "="*70)
print("Quantization Complete!")
print("="*70)
print(f"FP32 model:     {fp32_size:.2f} MB")
print(f"INT8 model:     {int8_size:.2f} MB")
print(f"Size reduction: {reduction:.1f}%")
print("="*70)
print("\nNext step: python eval_onnx_int8.py")
