# FP16 QAT Pipeline - SMSA Sentiment Analysis

⚠️ **IMPORTANT NOTICE:** FP16 ONNX conversion has **compatibility issues** with BERT/transformer models due to type casting problems in ONNX Runtime. 

**Recommendation: Use [int8_eager](../int8_eager/) pipeline instead**, which provides:
- ✅ Better compression (4x vs 2x)
- ✅ Better compatibility (works on CPU & GPU)
- ✅ Better performance on CPU (2-4x speedup)
- ✅ No ONNX conversion issues

---

**This pipeline is provided for reference only.** The QAT training works, but FP16 ONNX export has known issues with transformer architectures.

For production use, **choose INT8** for CPU deployment or **use FP32 on GPU** for maximum compatibility.

---

## 📋 Known Issues

### FP16 ONNX Conversion Problems
- ❌ Type casting errors with BERT models
- ❌ `tensor(float16)` vs `tensor(float)` mismatches
- ❌ ONNX Runtime CUDAExecutionProvider incompatibility
- ❌ Limited benefit over FP32 on modern GPUs

### Why INT8 is Better
- ✅ 4x compression (vs 2x for FP16)
- ✅ Stable ONNX conversion
- ✅ Excellent CPU performance
- ✅ Production-ready
- ✅ Better accuracy retention with QAT

## 🎯 Alternative: Use INT8 Pipeline

```bash
# Navigate to int8_eager instead
cd ../int8_eager

# Follow the same workflow
python train_qat_int8.py
python export_to_onnx.py
python quantize_onnx_int8.py
python eval_onnx_int8.py
```

**Result:** 4x smaller model, 2-4x faster, no conversion issues! ✅

---

## Original Pipeline (Reference Only)

## 🎯 Overview

This pipeline trains an IndoBERT model with QAT simulation and deploys it as a **true FP16 ONNX model** for:
- ✅ **2x smaller** model size (~210 MB vs ~418 MB)
- ✅ **Faster inference** on GPU/ARM hardware
- ✅ **Minimal accuracy loss** thanks to QAT training
- ✅ **Production-ready** ONNX format

⚠️ **Important:** FP16 inference **requires GPU or ARM processors**. For CPU deployment, use the [int8_eager](../int8_eager/) pipeline instead.

## 📋 Pipeline Steps

```
Step 1: QAT Training
├── train_qat_fp16.py
└── Trains with fake FP16 quantization
    Output: models/indobert-qat-fp16-smsa/model_qat.pth

Step 2: Export to ONNX
├── export_to_onnx.py
└── Exports QAT weights to ONNX (FP32)
    Output: model_qat.onnx

Step 3: Convert to FP16
├── quantize_onnx_fp16.py
└── Converts ONNX model from FP32 to FP16
    Output: model_qat_fp16.onnx (2x smaller!)

Step 4: Evaluate FP16 Model
└── eval_onnx_fp16.py
    Tests accuracy and performance
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch transformers datasets onnx onnxruntime onnxconverter-common
pip install scikit-learn Sastrawi
```

### Data Preparation

Place your TSV files in the **parent directory** (`bt-test/`):
- `train.tsv` - Training data
- `valid.tsv` - Validation data
- `test.tsv` - Test data

**Format:** Tab-separated with columns `[text, label]`
**Labels:** `positive`, `neutral`, `negative`

### Run Complete Pipeline

```bash
# Navigate to fp16_eager folder
cd fp16_eager

# Step 1: Train with QAT (~15-20 minutes)
python train_qat_fp16.py

# Step 2: Export to ONNX (~30 seconds)
python export_to_onnx.py

# Step 3: Convert to FP16 (~10 seconds)
python quantize_onnx_fp16.py

# Step 4: Evaluate FP16 model (~2-3 minutes)
python eval_onnx_fp16.py
```

## 📊 Expected Results

### Model Size
| Model Type | Size | Reduction |
|-----------|------|-----------|
| FP32 Original | ~418 MB | Baseline |
| FP16 Quantized | ~210 MB | **50% smaller** ✅ |

### Performance
- **Accuracy**: ~0.5-1% loss vs FP32 (minimal!)
- **Speed**: 1.3-1.8x faster on modern CPUs with FP16 support
- **Memory**: 2x less RAM usage

## 📁 Output Structure

```
fp16_eager/
├── train_qat_fp16.py           # Training script
├── export_to_onnx.py            # ONNX export
├── quantize_onnx_fp16.py        # FP16 conversion
├── eval_onnx_fp16.py            # Evaluation
├── README.md                    # This file
│
└── models/indobert-qat-fp16-smsa/
    ├── model_qat.pth            # QAT trained weights (PyTorch)
    ├── model_qat.onnx           # FP32 ONNX
    ├── model_qat_fp16.onnx      # FP16 ONNX (FINAL MODEL)
    └── tokenizer files...
```

## 🔧 Script Details

### 1. train_qat_fp16.py
**What it does:**
- Loads SMSA dataset (train, valid, test)
- Preprocesses text (lowercasing, stopword removal)
- Trains IndoBERT with QAT simulation
- Saves QAT-trained weights

**Key configurations:**
- Model: `indobenchmark/indobert-base-p2`
- Labels: 3 classes (positive, neutral, negative)
- Batch size: 16
- Epochs: 3
- Max sequence length: 128

### 2. export_to_onnx.py
**What it does:**
- Loads QAT-trained weights
- Strips fake quantization observers
- Exports clean FP32 ONNX model

**Output:** Compatible ONNX model with QAT benefits

### 3. quantize_onnx_fp16.py
**What it does:**
- Converts all FP32 weights → FP16
- Reduces model size by ~50%
- Uses `onnxconverter-common` library

**Output:** True FP16 ONNX model

### 4. eval_onnx_fp16.py
**What it does:**
- Loads FP16 ONNX model
- Runs inference on test set
- Reports accuracy, precision, recall, F1
- Measures inference speed

**Output:** Performance metrics and classification report

## 💡 Why QAT for FP16?

| Approach | Accuracy Loss | Training Time | Deployment |
|----------|--------------|---------------|------------|
| **PTQ (Post-Training)** | 2-4% | None | Quick |
| **QAT (This Pipeline)** | 0.5-1% ✅ | ~15 min | Better quality |

**QAT advantage:** Training with fake quantization helps model adapt to lower precision, resulting in better accuracy when actually deployed in FP16.

## 🐛 Troubleshooting

### Issue: "File not found: train.tsv"
**Solution:** Place TSV files in parent directory (`bt-test/`), not in `fp16_eager/`

### Issue: "onnxconverter-common not found"
**Solution:** `pip install onnxconverter-common`

### Issue: "Sastrawi import error"
**Solution:** `pip install Sastrawi`

### Issue: "ONNX export fails"
**Solution:** Ensure PyTorch >= 1.13 and transformers >= 4.30

## 🚀 Production Deployment

### Using Python/ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load FP16 model
session = ort.InferenceSession(
    "models/indobert-qat-fp16-smsa/model_qat_fp16.onnx",
    providers=['CPUExecutionProvider']
)

# Run inference
outputs = session.run(
    None,
    {
        'input_ids': input_ids_array,
        'attention_mask': attention_mask_array
    }
)
logits = outputs[0]
predictions = np.argmax(logits, axis=1)
```

### Benefits in Production
- ✅ 2x smaller deployment package
- ✅ Faster inference (especially on ARM/mobile)
- ✅ Lower memory footprint
- ✅ Cross-platform compatibility (ONNX)

## 📚 Additional Notes

### Text Preprocessing
This pipeline includes Indonesian text preprocessing:
- Lowercasing
- Special character removal
- Stopword removal (using Sastrawi)

### Label Mapping
- `positive` → 0
- `neutral` → 1
- `negative` → 2

### Hardware Recommendations
- **CPU (x86/x64)**: ⚠️ FP16 not well supported - use INT8 instead!
- **GPU (NVIDIA)**: ✅ Best for FP16 (CUDA, Tensor Cores)
- **ARM/Mobile**: ✅ Good FP16 support
- **RAM**: 8GB minimum

### Providers for ONNX Runtime

**For FP16 inference:**
```python
# GPU (best for FP16)
providers=['CUDAExecutionProvider']

# CPU (falls back to FP32)
providers=['CPUExecutionProvider']  # ⚠️ FP16 not supported!
```

**Recommendation:** If deploying on CPU, use the [int8_eager](../int8_eager/) pipeline instead for better performance.

## ✅ Summary

This FP16 QAT pipeline provides:
1. ✅ Quantization-Aware Training for better FP16 accuracy
2. ✅ Clean ONNX export preserving QAT benefits
3. ✅ True FP16 deployment (2x size reduction)
4. ✅ Minimal accuracy loss (<1% typically)
5. ✅ Production-ready inference pipeline

**Result:** High-quality FP16 model optimized through QAT training! 🎉
