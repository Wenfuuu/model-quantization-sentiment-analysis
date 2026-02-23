# INT8 QAT Pipeline - SMSA Sentiment Analysis

Complete **Quantization-Aware Training (QAT)** pipeline for **INT8 deployment** on SMSA sentiment analysis dataset.

## 🎯 Overview

This pipeline trains an IndoBERT model with QAT simulation and deploys it as a **true INT8 ONNX model** for:
- ✅ **4x smaller** model size (~105 MB vs ~418 MB)
- ✅ **2-4x faster** inference on CPU
- ✅ **Minimal accuracy loss** thanks to QAT training
- ✅ **Production-ready** ONNX format

## 📋 Pipeline Steps

```
Step 1: QAT Training
├── train_qat_int8.py
└── Trains with fake INT8 quantization
    Output: models/indobert-qat-int8-smsa/model_qat.pth

Step 2: Export to ONNX
├── export_to_onnx.py
└── Exports QAT weights to ONNX (FP32)
    Output: model_qat.onnx

Step 3: Quantize to INT8
├── quantize_onnx_int8.py
└── Converts ONNX weights from FP32 to INT8
    Output: model_qat_int8.onnx (4x smaller!)

Step 4: Evaluate INT8 Model
└── eval_onnx_int8.py
    Tests accuracy and performance
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch transformers datasets onnx onnxruntime
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
# Navigate to int8_eager folder
cd int8_eager

# Step 1: Train with QAT (~15-20 minutes)
python train_qat_int8.py

# Step 2: Export to ONNX (~30 seconds)
python export_to_onnx.py

# Step 3: Quantize to INT8 (~10 seconds)
python quantize_onnx_int8.py

# Step 4: Evaluate INT8 model (~2-3 minutes)
python eval_onnx_int8.py
```

## 📊 Expected Results

### Model Size
| Model Type | Size | Reduction |
|-----------|------|-----------|
| FP32 Original | ~418 MB | Baseline |
| INT8 Quantized | ~105 MB | **75% smaller** ✅ |

### Performance
- **Accuracy**: ~1-2% loss vs FP32 (QAT minimizes this!)
- **Speed**: 2-4x faster on CPU
- **Memory**: 4x less RAM usage

### QAT vs PTQ Comparison
| Method | Accuracy Loss | Best For |
|--------|--------------|----------|
| PTQ (Post-Training) | 3-5% | Quick deployment |
| **QAT (This Pipeline)** | **1-2%** ✅ | **Production quality** |

## 📁 Output Structure

```
int8_eager/
├── train_qat_int8.py            # Training script
├── export_to_onnx.py             # ONNX export
├── quantize_onnx_int8.py         # INT8 conversion
├── eval_onnx_int8.py             # Evaluation
├── README.md                     # This file
│
└── models/indobert-qat-int8-smsa/
    ├── model_qat.pth             # QAT trained weights (PyTorch)
    ├── model_qat.onnx            # FP32 ONNX
    ├── model_qat_int8.onnx       # INT8 ONNX (FINAL MODEL)
    └── tokenizer files...
```

## 🔧 Script Details

### 1. train_qat_int8.py
**What it does:**
- Loads SMSA dataset (train, valid, test)
- Preprocesses text (lowercasing, stopword removal)
- Trains IndoBERT with INT8 QAT simulation
- Uses per-tensor quantization for compatibility
- Saves QAT-trained weights

**Key configurations:**
- Model: `indobenchmark/indobert-base-p2`
- Labels: 3 classes (positive, neutral, negative)
- Batch size: 16
- Epochs: 3
- Max sequence length: 128
- **Quantization:** Per-tensor INT8 (compatible with BERT)

**QAT Config:**
```python
model.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.fake_quantize.default_fake_quant,
    weight=torch.quantization.default_weight_fake_quant
)
# Embedding layer excluded (not quantized)
```

### 2. export_to_onnx.py
**What it does:**
- Loads QAT-trained weights
- Strips fake quantization observers
- Exports clean FP32 ONNX model
- Preserves QAT-optimized weight values

**Why FP32 export?**
QAT weights are still FP32 but trained to be quantization-aware. True INT8 conversion happens in the next step.

### 3. quantize_onnx_int8.py
**What it does:**
- Applies dynamic INT8 quantization to ONNX
- Converts all weights: FP32 → INT8
- Uses ONNX Runtime's quantization tools
- Achieves 4x size reduction

**Quantization type:** Dynamic INT8
- Weights: INT8 (quantized offline)
- Activations: INT8 (quantized at runtime)

### 4. eval_onnx_int8.py
**What it does:**
- Loads INT8 ONNX model
- Runs inference on test set
- Reports accuracy, precision, recall, F1
- Measures inference speed (samples/second)
- Generates classification report

**Output:** Complete performance evaluation with per-class metrics

## 💡 Why QAT for INT8?

### The Problem with PTQ
Post-Training Quantization (PTQ) directly converts FP32 → INT8 without retraining:
- ❌ Can lose 3-5% accuracy
- ❌ Sensitive to outliers
- ❌ No adaptation to quantization errors

### QAT Solution
Quantization-Aware Training simulates INT8 during training:
- ✅ Model learns to compensate for quantization
- ✅ Only 1-2% accuracy loss
- ✅ Better weight distribution for INT8
- ✅ More robust to quantization noise

### Training Process
1. Insert fake quantization modules
2. Train with simulated INT8 operations
3. Model weights adapt to quantization constraints
4. Extract weights → ONNX → Apply real INT8

## 🐛 Troubleshooting

### Issue: "File not found: train.tsv"
**Solution:** Place TSV files in parent directory (`bt-test/`), not in `int8_eager/`

### Issue: "onnxruntime.quantization not found"
**Solution:** `pip install onnxruntime` (includes quantization tools)

### Issue: "Sastrawi import error"
**Solution:** `pip install Sastrawi`

### Issue: "Training takes too long"
**Solution:** Reduce `num_train_epochs` to 2 or use smaller dataset for testing

### Issue: "Accuracy drops significantly (>3%)"
**Solution:** 
- Ensure QAT training completed successfully
- Check if you're using the QAT model (not PTQ)
- Try increasing training epochs to 4-5

## 🚀 Production Deployment

### Using Python/ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load INT8 model
session = ort.InferenceSession(
    "models/indobert-qat-int8-smsa/model_qat_int8.onnx",
    providers=['CPUExecutionProvider']  # INT8 optimized for CPU
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

# Map predictions to labels
id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
predicted_labels = [id2label[p] for p in predictions]
```

### Benefits in Production
- ✅ 4x smaller deployment (saves bandwidth, storage)
- ✅ 2-4x faster inference (higher throughput)
- ✅ 4x less memory (more concurrent requests)
- ✅ Lower latency (better user experience)
- ✅ Energy efficient (lower server costs)

### Deployment Checklist
- [ ] Model tested on production-like data
- [ ] Accuracy meets requirements (< 2% loss)
- [ ] Inference speed benchmarked
- [ ] Memory usage profiled
- [ ] Error handling implemented
- [ ] Monitoring setup

## 📚 Additional Notes

### Text Preprocessing
This pipeline includes Indonesian text preprocessing:
- Lowercasing
- Special character removal
- Stopword removal (using Sastrawi)
- Whitespace normalization

**Why preprocess?**
- Reduces vocabulary size
- Improves model generalization
- Removes noise from text

### Label Mapping
```python
label2id = {
    'positive': 0,
    'neutral': 1, 
    'negative': 2
}
```

### Quantization Details
- **Weights:** INT8, symmetric quantization
- **Activations:** INT8, dynamic quantization  
- **Embeddings:** Not quantized (FP32)
- **Quantization granularity:** Per-tensor

### Hardware Recommendations
- **CPU**: Intel/AMD with AVX512 or VNNI for best INT8 performance
- **RAM**: 4GB minimum (reduced from 8GB due to smaller model)
- **GPU**: Not recommended (INT8 optimized for CPU)

## 🔍 Understanding the Numbers

### Why 4x Reduction?
- FP32: 32 bits per weight
- INT8: 8 bits per weight
- Ratio: 32/8 = 4x reduction ✅

### Why 2-4x Speed Improvement?
- INT8 operations are faster than FP32
- Better cache utilization (smaller model)
- Hardware acceleration (VNNI, AVX512)
- Actual speedup depends on CPU architecture

## ✅ Summary

This INT8 QAT pipeline provides:
1. ✅ Quantization-Aware Training (minimal accuracy loss)
2. ✅ Clean ONNX export preserving QAT benefits
3. ✅ True INT8 deployment (4x smaller, 2-4x faster)
4. ✅ Production-ready with ONNX Runtime
5. ✅ Complete evaluation and deployment guide

**Result:** Production-quality INT8 model optimized through QAT! 🚀

---

## 🆚 Quick Comparison: FP16 vs INT8

| Aspect | FP16 | INT8 |
|--------|------|------|
| **Size reduction** | 2x | 4x ✅ |
| **Speed improvement** | 1.3-1.8x | 2-4x ✅ |
| **Accuracy loss** | ~0.5-1% | ~1-2% |
| **Best hardware** | GPU, ARM | CPU ✅ |
| **Deployment** | Modern devices | Any device ✅ |

**Choose INT8** for maximum compression and CPU deployment!
