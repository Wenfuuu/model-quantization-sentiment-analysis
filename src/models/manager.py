from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import LABELS
from .base import BaseModel


class ModelManager:
    @staticmethod
    def load_model(model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True
        )
        return BaseModel(model, tokenizer)
