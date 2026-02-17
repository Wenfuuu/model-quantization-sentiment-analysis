# Model Quantization for Sentiment Analysis

Research project for model quantization techniques applied to Indonesian sentiment analysis using IndoBERT.

## Project Structure

```
model-quantization-sentiment-analysis/
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── manager.py
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── ptq/
│   │   │   ├── __init__.py
│   │   │   ├── dynamic.py
│   │   │   ├── fp16.py
│   │   │   ├── int4.py
│   │   │   └── engine.py
│   │   └── qat/
│   │       ├── __init__.py
│   │       ├── trainer.py
│   │       └── config.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plotter.py
│   │   └── reports.py
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── explainer.py
│   │   ├── attention.py
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── scripts/
│   ├── __init__.py
│   ├── run_ptq.py
│   ├── run_qat.py
│   └── run_xai.py
├── datasets/
│   ├── test.tsv
│   └── INA_TweetsPPKM_Labeled_Pure.csv
├── finetuned-model/
│   └── indobert-fp32-smsa-3label-finetuned/
├── outputs/
├── requirements.txt
└── README.md
```

## Modules

### PTQ (Post-Training Quantization)
Quantization techniques applied after model training:
- **FP16** (Half-Precision) - `src/quantization/ptq/fp16.py`
- **INT8** (Dynamic Quantization) - `src/quantization/ptq/dynamic.py`
- **INT4** (4-bit Quantization) - `src/quantization/ptq/int4.py`

### QAT (Quantization-Aware Training)
Training with quantization simulation for better accuracy preservation.

### XAI (Explainable AI)
Model interpretability and explanation methods:
- Integrated Gradients
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Occlusion

## Usage

### Run PTQ Experiments
```bash
python scripts/run_ptq.py [experiment_name]
```

Available experiments:
- `original_smsa` - Original IndoBERT + test.tsv (SMSA dataset)
- `finetuned_smsa` - Finetuned IndoBERT + test.tsv (SMSA dataset)
- `original_tweets` - Original IndoBERT + INA_TweetsPPKM (Tweets dataset)
- `finetuned_tweets` - Finetuned IndoBERT + INA_TweetsPPKM (Tweets dataset)

### Experiment Matrix

|                    | test.tsv (SMSA) | INA_TweetsPPKM (Tweets) |
|--------------------|-----------------|-------------------------|
| **Original Model** | original_smsa   | original_tweets         |
| **Finetuned Model**| finetuned_smsa  | finetuned_tweets        |

### Run QAT Training
```bash
python scripts/run_qat.py [experiment_name]
```

### Run XAI Analysis
```bash
python scripts/run_xai.py [model_path] [text]
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- See requirements.txt for full list
