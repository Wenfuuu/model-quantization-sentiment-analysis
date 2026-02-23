"""
Convert ONNX model to FP16 for reduced size and faster inference
Uses more robust conversion with keep_io_types to avoid type mismatches
"""
import onnx
from onnxconverter_common import float16
import os

print("="*70)
print("Converting ONNX Model to FP16")
print("="*70)

base_path = "./models/indobert-qat-fp16-smsa"
model_fp32 = f"{base_path}/model_qat.onnx"
model_fp16 = f"{base_path}/model_qat_fp16.onnx"

if not os.path.exists(model_fp32):
    print(f"ERROR: ONNX model not found at {model_fp32}")
    print("Run export_to_onnx.py first!")
    exit(1)

print(f"Input:  {model_fp32}")
print(f"Output: {model_fp16}")
print("\nConverting to FP16 (keeping I/O as FP32 for compatibility)...")

# Load FP32 model
onnx_model = onnx.load(model_fp32)

# Convert to FP16 with keep_io_types=True to avoid type mismatches
# This keeps inputs/outputs as FP32 while converting internal ops to FP16
onnx_model_fp16 = float16.convert_float_to_float16(
    onnx_model,
    keep_io_types=True  # Critical: keeps I/O as FP32 to avoid Cast errors
)

# Save FP16 model
onnx.save(onnx_model_fp16, model_fp16)

# Compare file sizes
fp32_size = os.path.getsize(model_fp32) / (1024 * 1024)
fp16_size = os.path.getsize(model_fp16) / (1024 * 1024)
reduction = (1 - fp16_size / fp32_size) * 100

print("\n" + "="*70)
print("Conversion Complete!")
print("="*70)
print(f"FP32 model:    {fp32_size:.2f} MB")
print(f"FP16 model:    {fp16_size:.2f} MB")
print(f"Size reduction: {reduction:.1f}%")
print("="*70)
print("\n✓ FP16 model ready for GPU inference")
print("  I/O kept as FP32, internal operations converted to FP16")
print("\nNext step: python eval_onnx_fp16.py")
