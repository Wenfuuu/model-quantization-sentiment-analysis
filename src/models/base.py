import time
import torch
from src.config import LABELS


class BaseModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, text, use_fp16=False):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

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
