import torch
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}

QUANTIZATION_DTYPE = torch.qint8
TARGET_LAYERS = {torch.nn.Linear}

TRAINING_SEEDS = [42, 123, 7]

SEEDED_MODEL_DIRS = {
    seed: BASE_DIR / "finetuned-model" / f"indobert-fp32-smsa-3label-seed{seed}"
    for seed in TRAINING_SEEDS
}

LEGACY_MODEL_DIR = BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"

EXPERIMENT_CONFIGS = {
    "original_smsa": {
        "model_id": "indobenchmark/indobert-base-p2",
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / "original-smsa",
        "num_inference_runs": 1,
        "warmup_runs": 5,
    },
    "finetuned_smsa": {
        "model_id": str(BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"),
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / "finetuned-smsa",
        "num_inference_runs": 1,
        "warmup_runs": 5,
    },
    "original_tweets": {
        "model_id": "indobenchmark/indobert-base-p2",
        "dataset": "tweets",
        "output_dir": BASE_DIR / "outputs" / "original-tweets",
        "num_inference_runs": 1,
        "warmup_runs": 5,
    },
    "finetuned_tweets": {
        "model_id": str(BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"),
        "dataset": "tweets",
        "output_dir": BASE_DIR / "outputs" / "finetuned-tweets",
        "num_inference_runs": 1,
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
    "qat_fake_smsa": {
        "model_paths": {
            "int8": str(BASE_DIR / "outputs" / "indobert-smsa-qat-int8-fake"),
            "fp16": str(BASE_DIR / "outputs" / "indobert-smsa-qat-fp16-fake"),
            "int4": str(BASE_DIR / "outputs" / "indobert-smsa-qat-int4-fake"),
        },
        "dataset": "smsa",
        "output_dir": BASE_DIR / "outputs" / "qat-fake-smsa",
    },
}

DATASET_PATHS = {
    "smsa": BASE_DIR / "datasets" / "test.tsv",
    "smsa_train": BASE_DIR / "datasets" / "train.tsv",
    "smsa_valid": BASE_DIR / "datasets" / "valid.tsv",
    "tweets": BASE_DIR / "datasets" / "INA_TweetsPPKM_Labeled_Pure.csv",
}
