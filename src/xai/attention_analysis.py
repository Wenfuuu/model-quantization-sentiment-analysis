from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_LOG_EPS   = 1e-12 
_NORM_EPS  = 1e-8 

@dataclass
class AttentionWeights:
    tokens:          List[str]
    weights:         np.ndarray
    n_layers:        int
    n_heads:         int
    seq_len:         int
    predicted_class: int
    text:            str
    precision:       str = "unknown"

    def head_weights(self, layer: int, head: int) -> np.ndarray:
        """Return A^(layer, head) ∈ ℝ^{T×T}."""
        return self.weights[layer, head]

    def cls_row(self, layer: int, head: int) -> np.ndarray:
        """Return p = A^(layer,head)_{0,:} — the [CLS] attention distribution."""
        return self.weights[layer, head, 0, :]


@dataclass
class RolloutResult:
    tokens:           List[str]
    rollout_matrix:   np.ndarray
    token_importance: np.ndarray
    layer_rollouts:   List[np.ndarray]
    precision:        str = "unknown"
    text:             str = ""

    def to_dict(self) -> dict:
        return {
            "text":             self.text,
            "precision":        self.precision,
            "tokens":           self.tokens,
            "token_importance": self.token_importance.tolist(),
            "layer_cls_rows":   [
                r[0, :].tolist() for r in self.layer_rollouts
            ],
        }


@dataclass
class AttentionEntropyResult:
    entropy:      np.ndarray
    entropy_norm: np.ndarray
    max_entropy:  float        
    seq_len:      int
    tokens:       List[str]
    precision:    str = "unknown"
    text:         str = ""

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "precision":     self.precision,
            "seq_len":       self.seq_len,
            "max_entropy":   self.max_entropy,
            "entropy":       self.entropy.tolist(),
            "entropy_norm":  self.entropy_norm.tolist(),
        }

    def mean_entropy_per_layer(self) -> np.ndarray:
        return self.entropy_norm.mean(axis=1)

    def max_entropy_per_layer(self) -> np.ndarray:
        return self.entropy_norm.max(axis=1)


@dataclass
class AttentionComparisonResult:
    text:                        str
    base_precision:              str
    variant_precision:           str
    tokens:                      List[str]
    rollout_delta:               np.ndarray    # (T,)
    rollout_cosine:              float
    rollout_spearman:            float
    entropy_delta:               np.ndarray    # (n_layers, n_heads)
    mean_entropy_delta_per_layer: np.ndarray   # (n_layers,)
    same_prediction:             bool
    base_class:                  int
    variant_class:               int

    def to_dict(self) -> dict:
        return {
            "text":                          self.text,
            "base_precision":                self.base_precision,
            "variant_precision":             self.variant_precision,
            "tokens":                        self.tokens,
            "rollout_cosine":                self.rollout_cosine,
            "rollout_spearman":              self.rollout_spearman,
            "rollout_delta":                 self.rollout_delta.tolist(),
            "entropy_delta":                 self.entropy_delta.tolist(),
            "mean_entropy_delta_per_layer":  self.mean_entropy_delta_per_layer.tolist(),
            "same_prediction":               self.same_prediction,
            "base_class":                    self.base_class,
            "variant_class":                 self.variant_class,
        }

def extract_attention_weights(
    model:     AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text:      str,
    device:    Optional[torch.device] = None,
    precision: str = "unknown",
) -> AttentionWeights:
    _device = device or next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )

    seq_len = inputs["input_ids"].shape[1]
    if seq_len == 512:
        warnings.warn(
            f"Input was truncated to 512 tokens for text: '{text[:60]}...'. "
            "Attention analysis may miss tail tokens.",
            UserWarning, stacklevel=2,
        )

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            outputs = model(**inputs, output_attentions=True)
        except TypeError as exc:
            raise RuntimeError(
                "Model does not accept output_attentions=True.  "
                "Ensure the model is a HuggingFace AutoModelForSequenceClassification "
                f"from the BERT family.  Original error: {exc}"
            ) from exc

    if outputs.attentions is None:
        raise RuntimeError(
            "outputs.attentions is None even though output_attentions=True was "
            "passed.  This may indicate the model's config has "
            "output_attentions=False hardcoded.  Set model.config.output_attentions=True "
            "before calling this function."
        )

    weights_np = np.stack(
        [attn[0].detach().cpu().float().numpy() for attn in outputs.attentions],
        axis=0,
    )

    n_layers, n_heads = weights_np.shape[:2]

    logits = outputs.logits.float()
    predicted_class = int(logits.argmax(dim=-1).item())

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    return AttentionWeights(
        tokens=tokens,
        weights=weights_np,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        predicted_class=predicted_class,
        text=text,
        precision=precision,
    )

def compute_rollout(
    attn: AttentionWeights,
    keep_layer_rollouts: bool = True,
) -> RolloutResult:
    T            = attn.seq_len
    n_layers     = attn.n_layers
    weights      = attn.weights

    layer_rollouts: List[np.ndarray] = []
    R = np.eye(T, dtype=np.float64)

    for ell in range(n_layers):
        A_bar = weights[ell].mean(axis=0).astype(np.float64)

        A_hat = 0.5 * A_bar + 0.5 * np.eye(T, dtype=np.float64)

        row_sums = A_hat.sum(axis=1, keepdims=True)
        A_hat = A_hat / np.maximum(row_sums, _NORM_EPS)

        R = A_hat @ R

        if keep_layer_rollouts:
            layer_rollouts.append(R.copy())

    token_importance = R[0, :].copy()

    if not keep_layer_rollouts:
        layer_rollouts = []

    return RolloutResult(
        tokens=attn.tokens,
        rollout_matrix=R.astype(np.float32),
        token_importance=token_importance.astype(np.float32),
        layer_rollouts=[r.astype(np.float32) for r in layer_rollouts],
        precision=attn.precision,
        text=attn.text,
    )

def compute_entropy(attn: AttentionWeights) -> AttentionEntropyResult:
    T        = attn.seq_len
    n_layers = attn.n_layers
    n_heads  = attn.n_heads
    weights  = attn.weights

    entropy      = np.zeros((n_layers, n_heads), dtype=np.float64)
    entropy_norm = np.zeros((n_layers, n_heads), dtype=np.float64)

    max_entropy = float(np.log(T)) if T > 1 else 1.0

    for ell in range(n_layers):
        for h in range(n_heads):
            p = weights[ell, h, 0, :].astype(np.float64)

            p = np.clip(p, 0.0, 1.0)

            H = -float(np.sum(p * np.log(p + _LOG_EPS)))

            entropy[ell, h]      = H
            entropy_norm[ell, h] = H / max_entropy if max_entropy > 0 else 0.0

    return AttentionEntropyResult(
        entropy=entropy.astype(np.float32),
        entropy_norm=entropy_norm.astype(np.float32),
        max_entropy=max_entropy,
        seq_len=T,
        tokens=attn.tokens,
        precision=attn.precision,
        text=attn.text,
    )

def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _NORM_EPS or norm_b < _NORM_EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    result = spearmanr(a, b)
    corr = result.statistic if hasattr(result, "statistic") else result.correlation
    return float(corr) if corr is not None else float("nan")


def compare_attention(
    base:    AttentionWeights,
    variant: AttentionWeights,
) -> AttentionComparisonResult:
    if base.seq_len != variant.seq_len:
        raise ValueError(
            f"Sequence lengths differ: base has T={base.seq_len}, "
            f"variant has T={variant.seq_len}. "
        )
    if base.n_layers != variant.n_layers or base.n_heads != variant.n_heads:
        raise ValueError(
            f"Architecture mismatch: base ({base.n_layers}L, {base.n_heads}H) "
            f"vs variant ({variant.n_layers}L, {variant.n_heads}H). "
        )

    rollout_base    = compute_rollout(base,    keep_layer_rollouts=False)
    rollout_variant = compute_rollout(variant, keep_layer_rollouts=False)

    ri_base    = rollout_base.token_importance.astype(np.float64)
    ri_variant = rollout_variant.token_importance.astype(np.float64)

    rollout_delta    = (ri_variant - ri_base).astype(np.float32)
    rollout_cosine   = _cosine_np(ri_base, ri_variant)
    rollout_spearman = _spearman_np(ri_base, ri_variant)

    entropy_base    = compute_entropy(base)
    entropy_variant = compute_entropy(variant)

    entropy_delta = (
        entropy_variant.entropy_norm.astype(np.float64)
        - entropy_base.entropy_norm.astype(np.float64)
    ).astype(np.float32)

    mean_entropy_delta = entropy_delta.mean(axis=1)  # (n_layers,)

    return AttentionComparisonResult(
        text=base.text,
        base_precision=base.precision,
        variant_precision=variant.precision,
        tokens=base.tokens,
        rollout_delta=rollout_delta,
        rollout_cosine=rollout_cosine,
        rollout_spearman=rollout_spearman,
        entropy_delta=entropy_delta,
        mean_entropy_delta_per_layer=mean_entropy_delta,
        same_prediction=(base.predicted_class == variant.predicted_class),
        base_class=base.predicted_class,
        variant_class=variant.predicted_class,
    )

class AttentionAnalyzer:
    def __init__(
        self,
        model:     AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device:    Optional[torch.device] = None,
        precision: str = "unknown",
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or next(model.parameters()).device
        self.precision = precision

    def extract(self, text: str) -> AttentionWeights:
        return extract_attention_weights(
            self.model, self.tokenizer, text,
            device=self.device, precision=self.precision,
        )

    def analyze(self, text: str, keep_layer_rollouts: bool = True) -> dict:
        weights = self.extract(text)
        rollout = compute_rollout(weights, keep_layer_rollouts=keep_layer_rollouts)
        entropy = compute_entropy(weights)
        return {"weights": weights, "rollout": rollout, "entropy": entropy}

def analyze_attention_batch(
    texts:     List[str],
    model:     AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    precision: str = "unknown",
    device:    Optional[torch.device] = None,
    keep_layer_rollouts: bool = False,
    verbose:   bool = True,
) -> List[dict]:
    analyzer = AttentionAnalyzer(model, tokenizer, device=device, precision=precision)
    was_training = model.training
    model.eval()

    results = []
    try:
        for i, text in enumerate(texts):
            result = analyzer.analyze(text, keep_layer_rollouts=keep_layer_rollouts)
            results.append(result)
            if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(texts)):
                print(f"  [attention] {i+1}/{len(texts)} samples processed")
    finally:
        if was_training:
            model.train()

    return results


def compare_attention_batch(
    texts:         List[str],
    base_model:    AutoModelForSequenceClassification,
    variant_model: AutoModelForSequenceClassification,
    tokenizer:     AutoTokenizer,
    base_precision:    str = "fp32",
    variant_precision: str = "int8",
    device:        Optional[torch.device] = None,
    verbose:       bool = True,
) -> List[AttentionComparisonResult]:
    base_dev    = device or next(base_model.parameters()).device
    variant_dev = device or next(variant_model.parameters()).device

    base_model.eval()
    variant_model.eval()

    comparisons: List[AttentionComparisonResult] = []
    for i, text in enumerate(texts):
        attn_base    = extract_attention_weights(
            base_model,    tokenizer, text, device=base_dev,    precision=base_precision
        )
        attn_variant = extract_attention_weights(
            variant_model, tokenizer, text, device=variant_dev, precision=variant_precision
        )
        comparisons.append(compare_attention(attn_base, attn_variant))

        if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(texts)):
            print(f"  [attention_compare] {i+1}/{len(texts)} samples")

    return comparisons

def aggregate_attention_comparisons(
    comparisons: List[AttentionComparisonResult],
) -> dict:
    if not comparisons:
        return {}

    def _stats(vals: List[float]) -> dict:
        n   = len(vals)
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "n":      n,
        }

    rollout_cosines   = [c.rollout_cosine   for c in comparisons]
    rollout_spearmans = [c.rollout_spearman for c in comparisons]
    label_flips       = [not c.same_prediction for c in comparisons]

    entropy_deltas = np.stack(
        [c.mean_entropy_delta_per_layer for c in comparisons], axis=0
    )
    n_layers = entropy_deltas.shape[1]

    entropy_mean = entropy_deltas.mean(axis=0).tolist()
    entropy_std  = (
        entropy_deltas.std(axis=0, ddof=1).tolist()
        if len(comparisons) > 1
        else [0.0] * n_layers
    )

    return {
        "base_precision":    comparisons[0].base_precision,
        "variant_precision": comparisons[0].variant_precision,
        "n_samples":         len(comparisons),
        "rollout_cosine":    _stats(rollout_cosines),
        "rollout_spearman":  _stats(rollout_spearmans),
        "label_flip_rate":   float(np.mean(label_flips)),
        "mean_entropy_delta_per_layer": {
            "mean": entropy_mean,
            "std":  entropy_std,
        },
    }

def save_attention_results(
    results:     List[dict],
    output_path: Path,
    *,
    metadata:    Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialised = []
    for r in results:
        entry = {
            "text":      r["rollout"].text,
            "precision": r["rollout"].precision,
            "rollout":   r["rollout"].to_dict(),
            "entropy":   r["entropy"].to_dict(),
        }
        serialised.append(entry)

    payload = {
        "metadata":  metadata or {},
        "n_samples": len(serialised),
        "results":   serialised,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_attention_comparisons(
    comparisons: List[AttentionComparisonResult],
    output_path: Path,
    *,
    metadata:    Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata":  metadata or {},
        "n_samples": len(comparisons),
        "results":   [c.to_dict() for c in comparisons],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_attention_results(path: Path) -> Tuple[List[dict], dict]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Attention results file not found: {path}\n"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["results"], payload.get("metadata", {})
