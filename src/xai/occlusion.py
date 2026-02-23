import numpy as np


class OcclusionExplainer:
    def __init__(self, base_model, labels, use_fp16=False):
        self.base_model = base_model
        self.labels = labels
        self.use_fp16 = use_fp16
        self.label_names = [labels[i] for i in sorted(labels.keys())]

    def explain(self, text, window_size=1):
        baseline_result = self.base_model.predict(text, use_fp16=self.use_fp16)
        predicted_idx = self.label_names.index(baseline_result["label"])
        baseline_conf = baseline_result["probabilities"][self.label_names[predicted_idx]]

        words = text.split()
        token_importance = []

        for i in range(len(words)):
            start = max(0, i)
            end = min(len(words), i + window_size)
            occluded_words = words[:start] + ["[MASK]"] * (end - start) + words[end:]
            occluded_text = " ".join(occluded_words)

            occluded_result = self.base_model.predict(occluded_text, use_fp16=self.use_fp16)
            occluded_conf = occluded_result["probabilities"][self.label_names[predicted_idx]]

            importance = baseline_conf - occluded_conf
            token_importance.append((words[i], float(importance)))

        sorted_importance = sorted(token_importance, key=lambda x: abs(x[1]), reverse=True)

        return {
            "predicted_label": self.label_names[predicted_idx],
            "token_importance": sorted_importance,
            "all_tokens_ordered": token_importance,
        }
