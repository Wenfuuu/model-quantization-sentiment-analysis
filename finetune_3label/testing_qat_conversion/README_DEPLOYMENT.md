# QAT Model Training & Deployment Guide

Complete pipeline for **Quantization-Aware Training (QAT)** and **true INT8 deployment** using ONNX Runtime.

## 🎯 What is QAT?

**Quantization-Aware Training (QAT)** trains your model while simulating INT8 quantization. This produces weights that maintain accuracy when actually quantized to INT8, unlike Post-Training Quantization (PTQ) which can lose significant accuracy.

## 📋 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Train with QAT                                     │
│  → Trains model with fake INT8 quantization                 │
│  → Output: model_qat.pth (FP32 with QAT-trained weights)   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Export to ONNX                                     │
│  → Converts QAT model to ONNX format                        │
│  → Output: model_qat.onnx (FP32 ONNX with QAT weights)     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Apply Dynamic INT8 Quantization                    │
│  → Converts ONNX weights from FP32 to INT8                  │
│  → Output: model_qat_int8_dynamic.onnx (True INT8!)        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Evaluate INT8 Model                                │
│  → Tests accuracy and performance of INT8 model             │
│  → Measures speed improvements and model size               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers datasets onnx onnxruntime scikit-learn
```

### Complete Workflow

#### 1️⃣ Train with QAT (takes ~10-15 minutes)
```bash
python train_eager_qat.py
```
**What it does:**
- Loads IndoBERT model
- Trains with fake INT8 quantization (QAT)
- Saves QAT-trained weights to `models/indobert-qat-int8-combined/model_qat.pth`

**Output:**
- QAT model: ~418 MB (FP32 format, but QAT-trained)
- Training metrics printed to console

---

#### 2️⃣ Export to ONNX (takes ~30 seconds)
```bash
python export_to_onnx.py
```
**What it does:**
- Loads QAT-trained weights into clean FP32 model
- Exports to ONNX format (opset 14)
- Preserves QAT-optimized weight values

**Output:**
- ONNX model: `models/indobert-qat-int8-combined/model_qat.onnx` (~418 MB)

---

#### 3️⃣ Quantize to INT8 (takes ~10 seconds)
```bash
python quantize_onnx_dynamic.py
```
**What it does:**
- Converts FP32 weights → INT8 weights
- Applies dynamic quantization
- Creates optimized INT8 ONNX model

**Output:**
- INT8 ONNX: `model_qat_int8_dynamic.onnx` (~105 MB, **4x smaller!**)

---

#### 4️⃣ Evaluate INT8 Model (takes ~2-3 minutes)
```bash
python eval_onnx_int8_dynamic.py
```
**What it does:**
- Runs inference on test set
- Measures accuracy, precision, recall, F1
- Reports inference speed (samples/second)
- Generates confusion matrix

**Output:**
- Performance metrics
- Inference speed comparison
- Confusion matrix saved to `results/`

---

## 📊 Expected Results

### Model Size
- **FP32 Original**: ~418 MB
- **INT8 Quantized**: ~105 MB
- **Reduction**: 75% smaller ✅

### Performance
- **Accuracy**: Minimal loss (<1-2% vs FP32) thanks to QAT
- **Speed**: 2-4x faster inference on CPU
- **Memory**: 4x less RAM usage

### Why QAT is Better
- **PTQ (Post-Training)**: ~3-5% accuracy loss
- **QAT (This pipeline)**: ~0.5-2% accuracy loss ✅

---

## 📁 File Structure

```
testing_qat_conversion/
├── train_eager_qat.py              # Step 1: QAT training
├── export_to_onnx.py                # Step 2: ONNX export
├── quantize_onnx_dynamic.py         # Step 3: INT8 quantization
├── eval_onnx_int8_dynamic.py        # Step 4: Evaluation
├── eval_onnx_qat.py                 # Optional: Eval FP32 ONNX
├── eval_eager_qat.py                # Optional: Eval PyTorch QAT
└── README_DEPLOYMENT.md             # This file

models/indobert-qat-int8-combined/
├── model_qat.pth                    # QAT-trained weights (PyTorch)
├── model_qat.onnx                   # FP32 ONNX with QAT weights
├── model_qat_int8_dynamic.onnx      # INT8 quantized ONNX (FINAL)
└── tokenizer files...

combined/
├── train.csv                        # Training data
├── val.csv                          # Validation data
└── test.csv                         # Test data
```

---

## 🔧 Troubleshooting

### Issue: "model_qat.pth not found"
**Solution:** Run `train_eager_qat.py` first, or check if file is named `model_int8.pth` (rename it).

### Issue: "ONNX export fails"
**Solution:** Make sure you have `torch>=1.13` and compatible `transformers` version.

### Issue: "Quantization takes too long"
**Solution:** This is normal for first run. Subsequent runs are faster (~10 sec).

### Issue: "Accuracy drops significantly"
**Solution:** If >5% drop, retrain with QAT. QAT should minimize accuracy loss.

---

## 🎓 Understanding QAT vs PTQ

| Aspect | PTQ (Post-Training) | QAT (This Pipeline) |
|--------|---------------------|---------------------|
| Training | Not required | Trains with fake quantization |
| Accuracy | 3-5% loss typical | 0.5-2% loss typical ✅ |
| Time | Fast (minutes) | Slower (needs training) |
| Best for | Quick deployment | Production quality ✅ |

---

## 🚀 Production Deployment

### Using Python
```python
import onnxruntime as ort
import numpy as np

# Load INT8 model
session = ort.InferenceSession(
    "models/indobert-qat-int8-combined/model_qat_int8_dynamic.onnx",
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

### Using ONNX Runtime Server
Deploy as REST API for production serving.

---

## 📚 Additional Resources

- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Hugging Face Optimization](https://huggingface.co/docs/optimum/index)

---

## ✅ Summary

This pipeline provides:
- ✅ True QAT training for better INT8 accuracy
- ✅ ONNX export with QAT-trained weights preserved
- ✅ Dynamic INT8 quantization (4x smaller, 2-4x faster)
- ✅ Easy evaluation and deployment

**Result:** Production-ready INT8 model with minimal accuracy loss!
