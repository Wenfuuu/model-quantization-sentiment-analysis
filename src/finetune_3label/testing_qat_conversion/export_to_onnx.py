"""
Export QAT-trained model to ONNX format for optimized INT8 inference
This properly loads QAT-trained weights for better post-quantization accuracy
ONNX Runtime provides better INT8 support for transformer models
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.quantization
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Exporting QAT Model to ONNX Format")
print("="*70)

# Load tokenizer
save_path_qat = "./models/indobert-qat-int8-combined"
tokenizer = AutoTokenizer.from_pretrained(save_path_qat)

# Load the QAT state dict
print("Loading QAT-trained weights...")
qat_state_dict = torch.load(f"{save_path_qat}/model_qat.pth", map_location='cpu')

# Create a base model (FP32, no quantization wrappers)
model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2",
    num_labels=2
)

# Load QAT weights into base model
# strict=False allows loading weights while skipping fake quantization observers
missing_keys, unexpected_keys = model.load_state_dict(qat_state_dict, strict=False)

print(f"✓ QAT-trained weights loaded into FP32 model")
print(f"  Loaded: {len(qat_state_dict) - len(unexpected_keys)} parameters")
print(f"  Skipped: {len(unexpected_keys)} fake quantization observers")

if missing_keys:
    print(f"  Warning: {len(missing_keys)} keys not found in QAT checkpoint")

model.eval()
model = model.cpu()

print("✓ Model ready for ONNX export")

# Create dummy input for export
dummy_input = tokenizer(
    "This is a sample text for export",
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt"
)

# Export to ONNX
onnx_path = "./models/indobert-qat-int8-combined/model_qat.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

print("\nExporting to ONNX (opset 14)...")
print(f"Output: {onnx_path}")

# Export with torch.no_grad() for clean export
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )

print("✓ Model exported to ONNX successfully")

# Get file size
onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"\nONNX model size: {onnx_size:.2f} MB (FP32 with QAT-trained weights)")

print("\n" + "="*70)
print("Export Complete!")
print("="*70)
print("\n✓ QAT-trained weights successfully exported to ONNX format")
print("✓ Model ready for INT8 quantization\n")
print("Next steps:")
print("  1. Quantize to INT8: python quantize_onnx_dynamic.py")
print("  2. Evaluate INT8:   python eval_onnx_int8_dynamic.py")
