import os
import torch
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}

QUANTIZATION_DTYPE = torch.qint8
TARGET_LAYERS = {torch.nn.Linear}

TRAINING_SEEDS = [42, 123, 456]

LARGE_N_STABILITY_SAMPLES = 300
LARGE_N_STABILITY_MIN_PER_CLASS = 100

MODEL_REGISTRY = {
    "indobert":    "indobenchmark/indobert-base-p2",
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "mbert":       "google-bert/bert-base-multilingual-cased",
}
DEFAULT_MODEL_TAG = "indobert"
MODEL_TAG = os.environ.get("MODEL_TAG", DEFAULT_MODEL_TAG).strip().lower()
if MODEL_TAG not in MODEL_REGISTRY:
    raise ValueError(
        f"MODEL_TAG={MODEL_TAG!r} not in MODEL_REGISTRY "
        f"(keys: {sorted(MODEL_REGISTRY)})"
    )
MODEL_ID = MODEL_REGISTRY[MODEL_TAG]


def _tag_suffix(tag: str = MODEL_TAG) -> str:
    return "" if tag == DEFAULT_MODEL_TAG else f"_{tag}"


def fp32_seed_dir(seed: int, tag: str = MODEL_TAG) -> Path:
    return BASE_DIR / "models" / f"fp32_seed{seed}{_tag_suffix(tag)}"


def fp32_control_seed_dir(seed: int, tag: str = MODEL_TAG) -> Path:
    return BASE_DIR / "models" / f"fp32_control_seed{seed}{_tag_suffix(tag)}"


SEEDED_MODEL_DIRS = {
    seed: fp32_seed_dir(seed) for seed in TRAINING_SEEDS
}

SEEDED_CONTROL_MODEL_DIRS = {
    seed: fp32_control_seed_dir(seed) for seed in TRAINING_SEEDS
}

LEGACY_MODEL_DIR = BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"

HF_DATASET_PATH = BASE_DIR / "datasets" / "hf_smsa"
FP32_MODEL_DIR = LEGACY_MODEL_DIR
PTQ_MODEL_PATH = BASE_DIR / "outputs" / "finetuned-smsa" / "ptq_int8.pth"
QAT_MODEL_PATH = BASE_DIR / "outputs" / "indobert-qat-int8-smsa" / "qat_trained.pt"

_TAG_OUT_SUFFIX = _tag_suffix()

EXPERIMENT_CONFIGS = {
    "original_smsa": {
        "model_id": MODEL_ID,
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / f"original-smsa{_TAG_OUT_SUFFIX}",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "finetuned_smsa": {
        "model_id": str(fp32_seed_dir(42)),
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / f"finetuned-smsa{_TAG_OUT_SUFFIX}",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "fp32_control_smsa": {
        "model_id": str(fp32_control_seed_dir(42)),
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / f"fp32-control-smsa{_TAG_OUT_SUFFIX}",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },

}

QAT_EXPERIMENT_CONFIGS = {
    "qat_eager_smsa": {
        "model_paths": {
            "int8": str(BASE_DIR / "outputs" / "indobert-qat-int8-smsa" / "hf_model"),
            "fp16": str(BASE_DIR / "outputs" / "indobert-qat-fp16-smsa" / "hf_model"),
            "int4": str(BASE_DIR / "outputs" / "indobert-qat-int4-smsa" / "hf_model"),
        },
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / f"qat-eager-smsa{_TAG_OUT_SUFFIX}",
    },
    "fp32_control_smsa": {
        "model_paths": {
            seed: str(SEEDED_CONTROL_MODEL_DIRS[seed]) for seed in TRAINING_SEEDS
        },
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / f"fp32-control-smsa{_TAG_OUT_SUFFIX}",
        "description": (
            "Continued FP32 fine-tune control (no fake-quant) — matches the "
            "QAT extra-training schedule (epochs/lr/batch/seq-len/optimizer/"
            "embedding handling) so the QAT-FP32 stability drop can be split "
            "into 'extra training' vs 'fake-quant gradient reshaping'."
        ),
    },

}

DATASET_PATHS = {
    "smsa": BASE_DIR / "datasets" / "test.tsv",
    "smsa_train": BASE_DIR / "datasets" / "train.tsv",
    "smsa_valid": BASE_DIR / "datasets" / "valid.tsv",
    "tweets": BASE_DIR / "datasets" / "INA_TweetsPPKM_Labeled_Pure.csv",
    "nusax_ind_train": BASE_DIR / "data" / "processed" / "nusax_ind_train.csv",
    "nusax_ind_valid": BASE_DIR / "data" / "processed" / "nusax_ind_valid.csv",
    "nusax_ind_test":  BASE_DIR / "data" / "processed" / "nusax_ind_test.csv",
}

NUSAX_SOURCE_LABEL_REMAP = {
    "positive": "POSITIVE",
    "neutral":  "NEUTRAL",
    "negative": "NEGATIVE",
}
NUSAX_CONTAMINATION_NOTE = (
    "NusaX-senti 'ind' is derived from SmSA; reuse as an out-of-distribution "
    "set is contaminated. Treat NusaX numbers as the in-language ceiling, not "
    "as held-out generalisation."
)
EVAL_DATASETS = ("smsa", "nusax_ind")

DEPLOYMENT_STABILITY_RHO_ACCEPTABLE = 0.90
DEPLOYMENT_AGREEMENT_MIN = 0.97
DEPLOYMENT_F1_DROP_TOLERANCE = 0.01
DEPLOYMENT_ECE_MAX = 0.08
DEPLOYMENT_LATENCY_CRITICAL_MS = 15.0
DEPLOYMENT_SIZE_CRITICAL_MB = 200.0
DEPLOYMENT_FAITHFULNESS_COMP_MIN = 0.05

DEPLOYMENT_VARIANTS = (
    "fp32",
    "ptq_fp16",
    "ptq_int8",
    "ptq_int4",
    "qat_fp32",
    "qat_onnx_fp16",
    "qat_onnx_int8",
    "qat_onnx_int4",
)
DEPLOYMENT_VARIANT_ALIASES = {
    "FP32":          "fp32",
    "PTQ-FP16":      "ptq_fp16",
    "PTQ-INT8":      "ptq_int8",
    "PTQ-INT4":      "ptq_int4",
    "QAT-FP32":      "qat_fp32",
    "QAT-ONNX-FP16": "qat_onnx_fp16",
    "QAT-ONNX-INT8": "qat_onnx_int8",
    "QAT-ONNX-INT4": "qat_onnx_int4",
}
DEPLOYMENT_RECOMMENDATION_DIR = BASE_DIR / "outputs" / "deployment-recommendation"
