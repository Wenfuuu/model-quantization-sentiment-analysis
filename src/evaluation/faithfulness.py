from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_SPECIAL_TOKENS = frozenset({"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"})


def _is_special(token: str) -> bool:
    return token in _SPECIAL_TOKENS

def _mask_subword(
    tokens: List[str],
    mask_indices: Sequence[int],
    mask_token: str,
) -> List[str]:
    masked = list(tokens)
    for idx in mask_indices:
        if not _is_special(tokens[idx]):
            masked[idx] = mask_token
    return masked

def _get_confidence_for_class(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    class_idx: int,
    device: torch.device,
) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.float()
        probs  = softmax(logits, dim=-1)

    n_classes = probs.shape[-1]
    if class_idx >= n_classes:
        raise ValueError(
            f"class_idx={class_idx} exceeds model output size {n_classes}. "
            "The model and the attribution result must correspond to the same "
            "label space."
        )
    return float(probs[0, class_idx].item())

def _select_topk_indices(
    tokens: List[str],
    scores: Sequence[float],
    k: int,
    by_absolute: bool = True,
) -> List[int]:
    content = [
        (i, scores[i])
        for i in range(len(tokens))
        if not _is_special(tokens[i])
    ]

    if len(content) == 0:
        warnings.warn(
            "All tokens are special tokens; no content tokens available for "
            "top-k selection.  Returning empty index list.",
            UserWarning, stacklevel=3,
        )
        return []

    if len(content) < k:
        warnings.warn(
            f"Only {len(content)} content tokens available, but k={k} was "
            "requested.  Using all {len(content)} content tokens.  This "
            "sample's sufficiency/comprehensiveness value at k is not "
            "comparable with samples that have ≥ k content tokens.",
            UserWarning, stacklevel=3,
        )
        k = len(content)

    score_fn = (lambda x: abs(x[1])) if by_absolute else (lambda x: x[1])
    ranked = sorted(content, key=score_fn, reverse=True)

    top_indices = sorted(idx for idx, _ in ranked[:k])  # ascending: positional order
    return top_indices

def _build_sufficiency_text(
    tokens: List[str],
    top_indices: List[int],
    mask_token: str,
    tokenizer: AutoTokenizer,
) -> str:
    selected_set = set(top_indices)
    masked_tokens = _mask_subword(
        tokens,
        mask_indices=[i for i in range(len(tokens))
                      if i not in selected_set and not _is_special(tokens[i])],
        mask_token=mask_token,
    )
    return tokenizer.convert_tokens_to_string(
        [t for t in masked_tokens if not _is_special(t)]
    ).strip()


def _build_comprehensiveness_text(
    tokens: List[str],
    top_indices: List[int],
    mask_token: str,
    tokenizer: AutoTokenizer,
) -> str:
    masked_tokens = _mask_subword(tokens, mask_indices=top_indices, mask_token=mask_token)
    return tokenizer.convert_tokens_to_string(
        [t for t in masked_tokens if not _is_special(t)]
    ).strip()

def _build_sufficiency_text_word(
    words: List[str],
    top_indices: List[int],
    mask_token: str = "[MASK]",
) -> str:
    selected_set = set(top_indices)
    masked_words = [
        w if i in selected_set else mask_token
        for i, w in enumerate(words)
    ]
    return " ".join(masked_words).strip()


def _build_comprehensiveness_text_word(
    words: List[str],
    top_indices: List[int],
    mask_token: str = "[MASK]",
) -> str:
    selected_set = set(top_indices)
    masked_words = [
        mask_token if i in selected_set else w
        for i, w in enumerate(words)
    ]
    return " ".join(masked_words).strip()

@dataclass
class FaithfulnessAtK:
    k:                            int
    original_confidence:          float
    sufficiency_confidence:       float
    comprehensiveness_confidence: float
    sufficiency:                  float
    comprehensiveness:            float
    predicted_class:              int
    n_content_tokens:             int
    top_indices:                  List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "k":                            self.k,
            "original_confidence":          self.original_confidence,
            "sufficiency_confidence":       self.sufficiency_confidence,
            "comprehensiveness_confidence": self.comprehensiveness_confidence,
            "sufficiency":                  self.sufficiency,
            "comprehensiveness":            self.comprehensiveness,
            "predicted_class":              self.predicted_class,
            "n_content_tokens":             self.n_content_tokens,
            "top_indices":                  self.top_indices,
        }


@dataclass
class FaithfulnessResult:
    text:         str
    method:       str
    precision:    str
    per_k:        Dict[int, FaithfulnessAtK] = field(default_factory=dict)
    token_level:  str = "subword"

    def to_dict(self) -> dict:
        return {
            "text":         self.text,
            "method":       self.method,
            "precision":    self.precision,
            "token_level":  self.token_level,
            "per_k":        {str(k): v.to_dict() for k, v in self.per_k.items()},
        }

class FaithfulnessEvaluator:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        k_values: Tuple[int, ...] = (5, 10, 20),
        by_absolute: bool = True,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or next(model.parameters()).device
        self.k_values  = k_values
        self.by_absolute = by_absolute

        self._mask_token = tokenizer.mask_token or "[MASK]"

    def evaluate(
        self,
        text: str,
        tokens: List[str],
        scores: Sequence[float],
        method: str = "unknown",
        precision: str = "unknown",
        token_level: str = "subword",
    ) -> FaithfulnessResult:
        if token_level not in ("subword", "word"):
            raise ValueError(
                f"token_level must be 'subword' or 'word', got {token_level!r}."
            )

        scores_arr = np.array(scores, dtype=float)

        orig_inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        orig_inputs = {k: v.to(self.device) for k, v in orig_inputs.items()}
        with torch.no_grad():
            logits = self.model(**orig_inputs).logits.float()
            probs  = softmax(logits, dim=-1)[0]

        predicted_class   = int(probs.argmax().item())
        orig_confidence   = float(probs[predicted_class].item())

        per_k: Dict[int, FaithfulnessAtK] = {}

        n_content = sum(1 for t in tokens if not _is_special(t))

        for k in self.k_values:
            top_indices = _select_topk_indices(
                tokens, scores_arr, k, by_absolute=self.by_absolute
            )

            if token_level == "subword":
                suff_text  = _build_sufficiency_text(
                    tokens, top_indices, self._mask_token, self.tokenizer
                )
                comp_text  = _build_comprehensiveness_text(
                    tokens, top_indices, self._mask_token, self.tokenizer
                )
            else:
                suff_text  = _build_sufficiency_text_word(
                    tokens, top_indices, self._mask_token
                )
                comp_text  = _build_comprehensiveness_text_word(
                    tokens, top_indices, self._mask_token
                )

            if not suff_text.strip():
                warnings.warn(
                    f"Sufficiency text is empty for k={k} (text='{text[:50]}...'). "
                    "This happens when k ≥ all content tokens.  Assigning "
                    "sufficiency_confidence = NaN for this k.",
                    UserWarning, stacklevel=2,
                )
                suff_conf = float("nan")
            else:
                suff_conf = _get_confidence_for_class(
                    self.model, self.tokenizer, suff_text,
                    predicted_class, self.device,
                )

            if not comp_text.strip():
                warnings.warn(
                    f"Comprehensiveness text is empty for k={k} "
                    f"(text='{text[:50]}...').  Assigning NaN.",
                    UserWarning, stacklevel=2,
                )
                comp_conf = float("nan")
            else:
                comp_conf = _get_confidence_for_class(
                    self.model, self.tokenizer, comp_text,
                    predicted_class, self.device,
                )

            suff_val = float("nan") if np.isnan(suff_conf) else max(0.0, orig_confidence - suff_conf)
            comp_val = float("nan") if np.isnan(comp_conf) else (orig_confidence - comp_conf)

            per_k[k] = FaithfulnessAtK(
                k=k,
                original_confidence=orig_confidence,
                sufficiency_confidence=suff_conf,
                comprehensiveness_confidence=comp_conf,
                sufficiency=suff_val,
                comprehensiveness=comp_val,
                predicted_class=predicted_class,
                n_content_tokens=n_content,
                top_indices=top_indices,
            )

        return FaithfulnessResult(
            text=text,
            method=method,
            precision=precision,
            per_k=per_k,
            token_level=token_level,
        )

def evaluate_faithfulness_batch(
    samples: List[dict],
    attribution_results: List[dict],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    method: str,
    precision: str,
    token_level: str = "subword",
    k_values: Tuple[int, ...] = (5, 10, 20),
    device: Optional[torch.device] = None,
) -> List[FaithfulnessResult]:
    if len(samples) != len(attribution_results):
        raise ValueError(
            f"samples ({len(samples)}) and attribution_results "
            f"({len(attribution_results)}) must have the same length."
        )

    evaluator = FaithfulnessEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        k_values=k_values,
    )

    was_training = model.training
    model.eval()

    results: List[FaithfulnessResult] = []
    try:
        for i, (sample, attr) in enumerate(zip(samples, attribution_results)):
            result = evaluator.evaluate(
                text=sample["text"],
                tokens=attr["tokens"],
                scores=attr["scores"],
                method=method,
                precision=precision,
                token_level=token_level,
            )
            results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"  [faithfulness] {i+1}/{len(samples)} samples processed")
    finally:
        if was_training:
            model.train()

    return results

def aggregate_faithfulness(
    results: List[FaithfulnessResult],
    k_values: Optional[Tuple[int, ...]] = None,
    min_content_tokens: int = 0,
) -> Dict[str, Dict[str, float]]:
    if not results:
        return {}

    if k_values is None:
        k_values = tuple(sorted(results[0].per_k.keys()))

    suff_by_k: Dict[int, List[float]] = {k: [] for k in k_values}
    comp_by_k: Dict[int, List[float]] = {k: [] for k in k_values}

    for r in results:
        for k in k_values:
            if k not in r.per_k:
                continue
            entry = r.per_k[k]

            if min_content_tokens > 0 and entry.n_content_tokens < min_content_tokens:
                continue

            suff_val = entry.sufficiency
            comp_val = entry.comprehensiveness

            if not np.isnan(suff_val):
                suff_by_k[k].append(suff_val)
            if not np.isnan(comp_val):
                comp_by_k[k].append(comp_val)

    def _stats(vals: List[float]) -> Dict[str, float]:
        n = len(vals)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(vals, dtype=float)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1) if n > 1 else 0.0),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "median": float(np.median(arr)),
            "n":      n,
        }

    output: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        output[f"suff_k{k}"] = _stats(suff_by_k[k])
        output[f"comp_k{k}"] = _stats(comp_by_k[k])

    return output

def compare_faithfulness(
    base_results: List[FaithfulnessResult],
    variant_results: List[FaithfulnessResult],
    base_label: str = "fp32",
    variant_label: str = "int8",
    k_values: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Dict[str, float]]:
    if len(base_results) != len(variant_results):
        warnings.warn(
            f"base_results ({len(base_results)}) and variant_results "
            f"({len(variant_results)}) have different lengths. "
            "Only paired samples (by index) will be compared.",
            UserWarning, stacklevel=2,
        )

    n_pairs = min(len(base_results), len(variant_results))

    if k_values is None:
        k_values = tuple(sorted(base_results[0].per_k.keys()))

    delta_suff: Dict[int, List[float]] = {k: [] for k in k_values}
    delta_comp: Dict[int, List[float]] = {k: [] for k in k_values}

    for i in range(n_pairs):
        b = base_results[i]
        v = variant_results[i]
        for k in k_values:
            if k not in b.per_k or k not in v.per_k:
                continue
            bs = b.per_k[k].sufficiency
            vs = v.per_k[k].sufficiency
            bc = b.per_k[k].comprehensiveness
            vc = v.per_k[k].comprehensiveness

            if not (np.isnan(bs) or np.isnan(vs)):
                delta_suff[k].append(vs - bs)
            if not (np.isnan(bc) or np.isnan(vc)):
                delta_comp[k].append(vc - bc)

    def _stats(vals: List[float]) -> Dict[str, float]:
        n = len(vals)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(vals, dtype=float)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1) if n > 1 else 0.0),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "median": float(np.median(arr)),
            "n":      n,
        }

    output: Dict[str, Dict[str, float]] = {
        "base_label":    base_label,
        "variant_label": variant_label,
        "n_pairs":       n_pairs,
    }
    for k in k_values:
        output[f"delta_suff_k{k}"] = _stats(delta_suff[k])
        output[f"delta_comp_k{k}"] = _stats(delta_comp[k])

    return output

import json as _json
from pathlib import Path as _Path


def save_faithfulness_results(
    results: List[FaithfulnessResult],
    output_path: _Path,
    *,
    metadata: Optional[dict] = None,
) -> None:
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata":  metadata or {},
        "n_samples": len(results),
        "results":   [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)


def load_faithfulness_results(path: _Path) -> Tuple[List[dict], dict]:
    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Faithfulness results file not found: {path}\n"
            "Run evaluate_faithfulness_batch() and save_faithfulness_results() first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    return payload["results"], payload.get("metadata", {})
