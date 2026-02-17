import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from pathlib import Path


class LIMEExplainer:
    def __init__(self, base_model, labels, use_fp16=False):
        self.base_model = base_model
        self.labels = labels
        self.use_fp16 = use_fp16
        self.label_names = [labels[i] for i in sorted(labels.keys())]
        self.explainer = LimeTextExplainer(class_names=self.label_names)

    def predict_proba(self, texts):
        probabilities = []
        for text in texts:
            result = self.base_model.predict(text, use_fp16=self.use_fp16)
            probs = [result["probabilities"][self.label_names[i]] for i in range(len(self.label_names))]
            probabilities.append(probs)
        return np.array(probabilities)

    def explain(self, text, num_features=10, num_samples=300):
        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=list(range(len(self.label_names)))
        )
        return explanation

    def explain_and_save(self, text, output_path, num_features=10, num_samples=300):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        explanation = self.explain(text, num_features, num_samples)
        explanation.save_to_file(str(output_path))

        predicted_label_idx = int(np.argmax(explanation.predict_proba))
        feature_weights = explanation.as_list(label=predicted_label_idx)

        return {
            "predicted_label": self.label_names[predicted_label_idx],
            "prediction_probabilities": {
                self.label_names[i]: float(explanation.predict_proba[i])
                for i in range(len(self.label_names))
            },
            "top_features": feature_weights,
            "output_path": str(output_path)
        }
