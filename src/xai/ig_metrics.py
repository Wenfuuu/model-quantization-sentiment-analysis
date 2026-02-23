from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from scipy.stats import spearmanr
from torch.nn.functional import softmax, cosine_similarity as torch_cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    k = max(1, k)
    idx_a = set(np.argsort(-a)[:k])
    idx_b = set(np.argsort(-b)[:k])
    return len(idx_a & idx_b) / float(k)

def integrated_gradients_tokens(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    target: int = None,
) -> Tuple[List[str], np.ndarray]:
    encoding = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    target_class = target if target is not None else pred_class

    embeddings = model.bert.embeddings.word_embeddings(input_ids)

    def forward_fn(embeds: torch.Tensor, mask: torch.Tensor):
        outputs = model(inputs_embeds=embeds, attention_mask=mask)
        return outputs.logits

    ig = IntegratedGradients(lambda e: forward_fn(e, attention_mask))

    attributions = ig.attribute(
        embeddings,
        target=target_class,
        n_steps=30,
        baselines=torch.zeros_like(embeddings),
    )

    token_attrs = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    return tokens, token_attrs

def attribution_similarity(
    model_a: AutoModelForSequenceClassification,
    model_b: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    top_k: int = 10,
) -> Dict[str, float]:
    tokens_a, attr_a = integrated_gradients_tokens(model_a, tokenizer, text)
    _, attr_b = integrated_gradients_tokens(model_b, tokenizer, text)

    L = min(len(attr_a), len(attr_b))
    attr_a = attr_a[:L]
    attr_b = attr_b[:L]

    return {
        "cosine": _cosine(attr_a, attr_b),
        "spearman": float(spearmanr(attr_a, attr_b).correlation),
        "topk_overlap": _topk_overlap(attr_a, attr_b, k=top_k),
    }

def auc(curve: Sequence[float]) -> float:
    if not curve:
        return 0.0
    x = np.linspace(0, 1, len(curve))
    return float(np.trapz(curve, x))

@dataclass
class InsertionDeletionResult:
    deletion_curve: List[float]
    insertion_curve: List[float]
    deletion_auc: float
    insertion_auc: float

class InsertionDeletionEvaluator:
    def __init__(self, tokenizer: AutoTokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    def _predict_conf(self, model: AutoModelForSequenceClassification, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)
        return float(probs.max().item())

    def _mask_token(self) -> str:
        return self.tokenizer.mask_token or "[MASK]"

    def _scores(self, model: AutoModelForSequenceClassification, tokens: List[str], order: np.ndarray) -> Tuple[List[float], List[float]]:
        # deletion: progressively mask top-attribution tokens
        deletion_probs = []
        masked_tokens = tokens.copy()
        mask_token = self._mask_token()

        for k in range(len(tokens)):
            for i in order[:k]:
                masked_tokens[i] = mask_token
            text = self.tokenizer.convert_tokens_to_string(masked_tokens)
            deletion_probs.append(self._predict_conf(model, text))

        # insertion: start fully masked, gradually reveal
        insertion_probs = []
        masked_tokens = [mask_token] * len(tokens)
        for k in range(len(tokens)):
            for i in order[:k]:
                masked_tokens[i] = tokens[i]
            text = self.tokenizer.convert_tokens_to_string(masked_tokens)
            insertion_probs.append(self._predict_conf(model, text))

        return deletion_probs, insertion_probs

    def evaluate(
        self,
        model: AutoModelForSequenceClassification,
        tokens: List[str],
        attributions: np.ndarray,
    ) -> InsertionDeletionResult:
        order = np.argsort(-attributions)
        deletion_curve, insertion_curve = self._scores(model, tokens, order)

        return InsertionDeletionResult(
            deletion_curve=deletion_curve,
            insertion_curve=insertion_curve,
            deletion_auc=auc(deletion_curve),
            insertion_auc=auc(insertion_curve),
        )

def layer_cls_similarity(
    model_a: AutoModelForSequenceClassification,
    model_b: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(next(model_a.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out_a = model_a(**inputs, output_hidden_states=True)
    with torch.no_grad():
        out_b = model_b(**inputs, output_hidden_states=True)

    sims = []
    for h_a, h_b in zip(out_a.hidden_states, out_b.hidden_states):
        cls_a = h_a[:, 0, :]
        cls_b = h_b[:, 0, :]
        sims.append(float(torch_cosine(cls_a, cls_b, dim=-1).mean().item()))

    return sims
