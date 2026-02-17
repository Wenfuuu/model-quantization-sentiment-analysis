import torch
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}

QUANTIZATION_DTYPE = torch.qint8
TARGET_LAYERS = {torch.nn.Linear}

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
    "original_tweets": {
        "model_id": "indobenchmark/indobert-base-p2",
        "dataset": "tweets",
        "output_dir": BASE_DIR / "outputs" / "original-tweets",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "finetuned_tweets": {
        "model_id": str(BASE_DIR / "finetuned-model" / "indobert-fp32-smsa-3label-finetuned"),
        "dataset": "tweets",
        "output_dir": BASE_DIR / "outputs" / "finetuned-tweets",
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
}

DATASET_PATHS = {
    "smsa": BASE_DIR / "datasets" / "test.tsv",
    "tweets": BASE_DIR / "datasets" / "INA_TweetsPPKM_Labeled_Pure.csv",
}
