# Quantization-Aware Training (QAT) Pipelines

Complete QAT training and deployment pipelines for model quantization with ONNX Runtime.

## 📁 Available Pipelines

### 1. **testing_qat_conversion/** - INT8 QAT (Combined Dataset)
Full INT8 QAT pipeline on combined sentiment analysis dataset (2 labels: positive/negative).

- **Dataset:** Combined CSV (train.csv, val.csv, test.csv)
- **Labels:** 2 classes (binary sentiment)
- **Target:** INT8 quantization
- **Size reduction:** 4x (418 MB → 105 MB)
- **Speed improvement:** 2-4x

📖 **[Read Guide](testing_qat_conversion/README_DEPLOYMENT.md)**

```bash
cd testing_qat_conversion
python train_eager_qat.py       # Train
python export_to_onnx.py         # Export
python quantize_onnx_dynamic.py  # Quantize
python eval_onnx_int8_dynamic.py # Evaluate
```

---

### 2. **fp16_eager/** - FP16 QAT (SMSA Dataset) ⚠️ **Not Recommended**

⚠️ **Known Issues:** FP16 ONNX conversion has compatibility problems with transformer models. Use **int8_eager** instead.

- **Dataset:** TSV format (train.tsv, valid.tsv, test.tsv)
- **Labels:** 3 classes (positive, neutral, negative)
- **Target:** FP16 quantization
- **Status:** ❌ ONNX conversion issues
- **Recommendation:** Use int8_eager for production

**This pipeline has known FP16 ONNX compatibility issues. Skip to int8_eager.**

📖 **[Read Guide](fp16_eager/README.md)**

---

### 3. **int8_eager/** - INT8 QAT (SMSA Dataset)
INT8 QAT pipeline on SMSA sentiment analysis dataset (3 labels).

- **Dataset:** TSV format (train.tsv, valid.tsv, test.tsv)
- **Labels:** 3 classes (positive, neutral, negative)
- **Target:** INT8 quantization
- **Size reduction:** 4x (418 MB → 105 MB)
- **Speed improvement:** 2-4x
- **Preprocessing:** Indonesian stopword removal

📖 **[Read Guide](int8_eager/README.md)**

```bash
cd int8_eager
python train_qat_int8.py      # Train
python export_to_onnx.py      # Export
python quantize_onnx_int8.py  # Quantize
python eval_onnx_int8.py      # Evaluate
```

---

## 🆚 Pipeline Comparison

| Pipeline | Dataset | Labels | Quantization | Size | Speed | Status |
|----------|---------|--------|--------------|------|-------|--------|
| **testing_qat_conversion** | Combined CSV | 2 | INT8 | ÷4 | 2-4x | ✅ Production |
| **fp16_eager** | SMSA TSV | 3 | FP16 | ÷2 | - | ⚠️ ONNX issues |
| **int8_eager** | SMSA TSV | 3 | INT8 | ÷4 | 2-4x | ✅ **Recommended** |

---

## 🎯 Which Pipeline Should I Use?

### Choose **testing_qat_conversion** if:
- ✅ You have binary classification (2 labels)
- ✅ Dataset is in CSV format
- ✅ You want maximum compression (INT8)

### ~~Choose **fp16_eager** if:~~ ⚠️ **Not Recommended**
- ❌ FP16 ONNX has known compatibility issues
- ❌ Type casting errors with transformer models
- **Use int8_eager instead**

### Choose **int8_eager** if: ⭐ **Recommended**
- ✅ You have multi-class classification (3+ labels)
- ✅ Dataset is in TSV format with Indonesian text
- ✅ You want maximum compression and compatibility
- ✅ You're deploying on CPU or GPU
- ✅ You want production-ready solution

---

## 📊 What is QAT?

**Quantization-Aware Training (QAT)** trains your model while simulating lower precision (FP16/INT8). This allows the model to adapt to quantization, resulting in:

✅ **Better accuracy** than Post-Training Quantization (PTQ)  
✅ **Smaller models** (2-4x size reduction)  
✅ **Faster inference** (1.5-4x speedup)  
✅ **Production-ready** ONNX deployment

### QAT vs PTQ

| Method | Accuracy Loss | Training Required | Best For |
|--------|--------------|-------------------|----------|
| **PTQ** | 3-5% | No | Quick experiments |
| **QAT** | 0.5-2% ✅ | Yes | Production ✅ |

---

## 🚀 Quick Start (Any Pipeline)

### 1. Install Dependencies

```bash
# Core packages (all pipelines)
pip install torch transformers datasets onnx onnxruntime

# For FP16/INT8 pipelines (SMSA)
pip install Sastrawi scikit-learn

# For FP16 conversion
pip install onnxconverter-common
```

### 2. Prepare Data

**For testing_qat_conversion:**
- Place in `testing_qat_conversion/combined/`
- Format: CSV files (train.csv, val.csv, test.csv)
- Columns: `text`, `source`, `label`

**For fp16_eager / int8_eager:**
- Place in `bt-test/` (parent directory)
- Format: TSV files (train.tsv, valid.tsv, test.tsv)
- Columns: `text`, `label` (tab-separated)

### 3. Run Pipeline

Choose your pipeline folder and run the 4 scripts in order:
1. `train_*.py` - Train with QAT
2. `export_to_onnx.py` - Export to ONNX
3. `quantize_*.py` - Apply quantization
4. `eval_*.py` - Evaluate model

---

## 📁 Repository Structure

```
bt-test/
├── testing_qat_conversion/     # INT8 QAT (Combined, 2 labels)
│   ├── train_eager_qat.py
│   ├── export_to_onnx.py
│   ├── quantize_onnx_dynamic.py
│   ├── eval_onnx_int8_dynamic.py
│   └── README_DEPLOYMENT.md
│
├── fp16_eager/                 # FP16 QAT (SMSA, 3 labels)
│   ├── train_qat_fp16.py
│   ├── export_to_onnx.py
│   ├── quantize_onnx_fp16.py
│   ├── eval_onnx_fp16.py
│   └── README.md
│
├── int8_eager/                 # INT8 QAT (SMSA, 3 labels)
│   ├── train_qat_int8.py
│   ├── export_to_onnx.py
│   ├── quantize_onnx_int8.py
│   ├── eval_onnx_int8.py
│   └── README.md
│
├── fp16_fake/                  # Legacy fake quantization
└── int8_fake/                  # Legacy fake quantization
```

---

## 🎓 Key Concepts

### Quantization Types

**INT8 (8-bit integer)**
- Most aggressive compression (4x)
- Best CPU performance
- ~1-2% accuracy loss with QAT

**FP16 (16-bit float)**
- Moderate compression (2x)
- Best GPU performance
- ~0.5-1% accuracy loss with QAT

### Deployment Methods

**Fake Quantization (fp16_fake, int8_fake):**
- ❌ Simulates quantization but runs in FP32
- ❌ No real size/speed benefits
- ✅ Good for training and testing

**True Quantization (eager pipelines):**
- ✅ Real FP16/INT8 operations
- ✅ Actual size and speed improvements
- ✅ Production-ready ONNX models

---

## 🔧 Common Operations

### Check Model Size
```bash
ls -lh models/*/model_*.onnx
ls -lh models/*/model_*.pth
```

### Test ONNX Model
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
print("Model loaded successfully!")
```

### Compare Accuracy
Run evaluation scripts for both FP32 baseline and quantized model to compare metrics.

---

## 📚 Additional Resources

- [PyTorch Quantization Guide](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Model Optimization Best Practices](https://huggingface.co/docs/optimum/index)

---

## ✅ Summary

This repository provides **three complete QAT pipelines**:

1. **testing_qat_conversion** - INT8 for binary sentiment (combined dataset)
2. **fp16_eager** - FP16 for multi-class sentiment (SMSA)
3. **int8_eager** - INT8 for multi-class sentiment (SMSA)

Each pipeline includes:
- ✅ QAT training
- ✅ ONNX export
- ✅ True quantization
- ✅ Comprehensive evaluation
- ✅ Production deployment guide

Choose the pipeline that matches your dataset and deployment target! 🚀
