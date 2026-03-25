from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from src.config import LABELS, DEVICE
from .base import BaseModel


class ModelManager:
    @staticmethod
    def _is_local_model_dir(model_id):
        return Path(model_id).is_dir()

    @staticmethod
    def _infer_base_model_id(model_dir: Path):
        config_path = model_dir / "config.json"
        if config_path.exists():
            config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
            base_model_id = getattr(config, "_name_or_path", None)
            if base_model_id and base_model_id != str(model_dir):
                return base_model_id
        return "indobenchmark/indobert-base-p2"

    @staticmethod
    def _resolve_tokenizer_source(model_id):
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            if model_dir.is_absolute():
                raise FileNotFoundError(
                    f"Local model directory not found: {model_dir}\n"
                    "Run finetuning first, or check that the path in EXPERIMENT_CONFIGS is correct."
                )
            return model_id

        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
        ]
        if any((model_dir / filename).exists() for filename in tokenizer_files):
            return str(model_dir)

        return ModelManager._infer_base_model_id(model_dir)

    @staticmethod
    def _load_model_from_local_dir(model_dir: Path):
        base_model_id = ModelManager._infer_base_model_id(model_dir)
        weight_candidates = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        ]
        has_weights = any(path.exists() for path in weight_candidates)
        has_config = (model_dir / "config.json").exists()

        if has_weights and has_config:
            return AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                num_labels=len(LABELS),
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True,
        )

        state_path = next((path for path in weight_candidates if path.exists()), None)
        if state_path is None:
            raise FileNotFoundError(
                f"No model weights found in local checkpoint directory: {model_dir}"
            )

        if state_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading local .safetensors checkpoints requires the safetensors package."
                ) from exc
            state_dict = load_file(str(state_path))
        else:
            state_dict = torch.load(str(state_path), map_location="cpu")

        model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def load_model(model_id, device=None):
        device = device if device else DEVICE
        tokenizer = AutoTokenizer.from_pretrained(
            ModelManager._resolve_tokenizer_source(model_id)
        )

        if ModelManager._is_local_model_dir(model_id):
            model = ModelManager._load_model_from_local_dir(Path(model_id))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=len(LABELS),
                ignore_mismatched_sizes=True,
            )

        return BaseModel(model, tokenizer, device=device)
