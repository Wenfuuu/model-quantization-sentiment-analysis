"""
Export QAT-trained INT8 model to ONNX format
Preserves QAT-trained weights for better INT8 accuracy
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Exporting INT8 QAT Model to ONNX Format")
print("="*70)

save_path_qat = "./models/indobert-qat-int8-smsa"
tokenizer = AutoTokenizer.from_pretrained(save_path_qat)

# Load QAT state dict
print("Loading QAT-trained weights...")
qat_state_dict = torch.load(f"{save_path_qat}/model_qat.pth", map_location='cpu')

# Create base model
model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2",
    num_labels=3
)

# Load QAT weights (skip fake quantization modules)
missing_keys, unexpected_keys = model.load_state_dict(qat_state_dict, strict=False)

print(f"✓ QAT-trained weights loaded")
print(f"  Loaded: {len(qat_state_dict) - len(unexpected_keys)} parameters")
print(f"  Skipped: {len(unexpected_keys)} fake quantization observers")

model.eval()
model = model.cpu()

# Create dummy input
dummy_input = tokenizer(
    "Contoh teks untuk eksport",
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt"
)

onnx_path = f"{save_path_qat}/model_qat.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

print(f"\nExporting to ONNX (opset 14)...")
print(f"Output: {onnx_path}")

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

onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"✓ Export complete! Model size: {onnx_size:.2f} MB")

print("\n" + "="*70)
print("Next step: python quantize_onnx_int8.py")
print("="*70)
