import re
import time
import torch
import numpy as np
from src.config import LABELS, DEVICE

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class BaseModel:
    def __init__(self, model, tokenizer, device=None):
        self.device = device if device else DEVICE
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, text, use_fp16=False):
        text = preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        end_time = time.perf_counter()

        logits = outputs.logits.float()
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class_id = logits.argmax().item()
        confidence = probabilities[predicted_class_id].item()

        return {
            "label": LABELS[predicted_class_id],
            "class_id": predicted_class_id,
            "confidence": confidence,
            "probabilities": {LABELS[i]: prob.item() for i, prob in enumerate(probabilities)},
            "inference_time": end_time - start_time
        }

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params

class OnnxBaseModel:
    def __init__(self, session, tokenizer, hf_model, device):
        self.session = session
        self.tokenizer = tokenizer
        self.model = hf_model
        self.device = device

    def predict(self, text, use_fp16=False):
        text = text.lower().strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        start_time = time.perf_counter()
        logits = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        end_time = time.perf_counter()

        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        pred_idx = int(np.argmax(logits, axis=1)[0])

        return {
            "label": LABELS[pred_idx],
            "class_id": pred_idx,
            "confidence": float(probs[0, pred_idx]),
            "probabilities": {LABELS[i]: float(probs[0, i]) for i in range(len(LABELS))},
            "inference_time": end_time - start_time,
        }
