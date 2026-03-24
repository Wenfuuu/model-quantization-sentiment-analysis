import torch
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}

QUANTIZATION_DTYPE = torch.qint8
TARGET_LAYERS = {torch.nn.Linear}

TRAINING_SEEDS = [42, 123, 456]

SEEDED_MODEL_DIRS = {
    seed: BASE_DIR / "models" / f"fp32_seed{seed}"
    for seed in TRAINING_SEEDS
}

LEGACY_MODEL_DIR = BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"

HF_DATASET_PATH = BASE_DIR / "datasets" / "hf_smsa"
FP32_MODEL_DIR = LEGACY_MODEL_DIR
PTQ_MODEL_PATH = BASE_DIR / "outputs" / "finetuned-smsa" / "ptq_int8.pth"
QAT_MODEL_PATH = BASE_DIR / "outputs" / "indobert-qat-int8-smsa" / "qat_trained.pt"

EXPERIMENT_CONFIGS = {
    "original_smsa": {
        "model_id": "indobenchmark/indobert-base-p2",
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / "original-smsa",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "finetuned_smsa": {
        "model_id": str(BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"),
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / "finetuned-smsa",
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
        "output_dir": BASE_DIR / "outputs" / "qat-eager-smsa",
    },

}

DATASET_PATHS = {
    "smsa": BASE_DIR / "datasets" / "test.tsv",
    "smsa_train": BASE_DIR / "datasets" / "train.tsv",
    "smsa_valid": BASE_DIR / "datasets" / "valid.tsv",
    "tweets": BASE_DIR / "datasets" / "INA_TweetsPPKM_Labeled_Pure.csv",
}
