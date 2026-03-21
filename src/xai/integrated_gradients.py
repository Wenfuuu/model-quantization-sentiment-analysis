import warnings
import torch
import numpy as np
from captum.attr import IntegratedGradients

IG_SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16", "qat_fp32", "qat_int8"})


class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer, device=None, precision="fp32"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.precision = precision

    def explain(self, text, target=None, steps=30):
        if self.precision not in IG_SUPPORTED_PRECISIONS:
            raise ValueError(
                f"IG is not supported for precision='{self.precision}'. "
                f"Supported: {sorted(IG_SUPPORTED_PRECISIONS)}. "
                "Dynamic INT8/INT4 quantization breaks autograd; use LIME, "
                "SHAP, or Occlusion for these variants."
            )

        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted = int(torch.argmax(logits, dim=-1).item())

        target_class = predicted if target is None else target

        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)

        def forward_fn(embeds):
            return self.model(inputs_embeds=embeds, attention_mask=attention_mask).logits

        ig = IntegratedGradients(forward_fn)
        try:
            attributions = ig.attribute(
                embeddings,
                target=target_class,
                n_steps=steps,
                baselines=torch.zeros_like(embeddings),
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "not have been used in the graph" not in msg:
                raise
            raise RuntimeError(
                f"IG attribution failed for precision='{self.precision}': "
                "gradient graph was broken by a non-differentiable operator. "
                "This model variant does not support IG. Original error: "
                + msg
            ) from exc

        scores = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()

        if np.allclose(scores, 0.0):
            warnings.warn(
                f"IG returned all-zero attributions for precision='{self.precision}'. "
                "This usually means gradient flow through the model is broken. "
                "Results for this variant should not be used for stability analysis.",
                RuntimeWarning,
                stacklevel=2,
            )

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        return {
            "tokens": tokens,
            "scores": scores,
            "predicted_class": predicted,
        }
