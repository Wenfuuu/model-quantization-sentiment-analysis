import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class SHAPExplainer:
    def __init__(self, base_model, labels, use_fp16=False):
        self.base_model = base_model
        self.labels = labels
        self.use_fp16 = use_fp16
        self.label_names = [labels[i] for i in sorted(labels.keys())]

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        probabilities = []
        for text in texts:
            result = self.base_model.predict(text, use_fp16=self.use_fp16)
            probs = [result["probabilities"][self.label_names[i]] for i in range(len(self.label_names))]
            probabilities.append(probs)
        return np.array(probabilities)

    def explain(self, text, max_evals=200):
        masker = shap.maskers.Text(r"\s+")
        explainer = shap.Explainer(
            self.predict_proba,
            masker,
            output_names=self.label_names
        )
        shap_values = explainer([text], max_evals=max_evals)
        return shap_values

    def explain_and_save(self, text, output_path, max_evals=200):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap_values = self.explain(text, max_evals=max_evals)

        predicted_class = int(np.argmax(self.predict_proba(text)))

        token_importance = {}
        if hasattr(shap_values[0], 'data') and hasattr(shap_values[0], 'values'):
            data = shap_values[0].data
            values = shap_values[0].values
            for i, token in enumerate(data):
                if isinstance(token, str) and token.strip():
                    token_importance[token] = float(values[i][predicted_class])

        sorted_importance = sorted(token_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

        if sorted_importance:
            tokens = [t[0] for t in sorted_importance]
            weights = [t[1] for t in sorted_importance]
            colors = ['#ff4444' if w < 0 else '#4444ff' for w in weights]

            fig, ax = plt.subplots(figsize=(12, max(4, len(tokens) * 0.4)))
            y_pos = range(len(tokens))
            ax.barh(y_pos, weights, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens)
            ax.invert_yaxis()
            ax.set_xlabel(f"SHAP value (impact on {self.label_names[predicted_class]})")
            ax.set_title(f"SHAP Token Importance - Predicted: {self.label_names[predicted_class]}")
            ax.axvline(x=0, color='black', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()

        return {
            "predicted_label": self.label_names[predicted_class],
            "token_importance": sorted_importance[:10],
            "output_path": str(output_path)
        }
