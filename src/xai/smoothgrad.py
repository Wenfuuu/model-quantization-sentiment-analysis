from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def _get_word_embeddings(model: AutoModelForSequenceClassification):
    if hasattr(model, "bert"):
        return model.bert.embeddings.word_embeddings
    emb = model.get_input_embeddings()
    if emb is None:
        raise AttributeError(
            "Cannot locate word embedding layer.  Model has neither .bert nor "
            "a registered input embedding via get_input_embeddings().  "
            "Update _get_word_embeddings() for this architecture."
        )
    return emb

def compute_sigma(
    embeddings:   torch.Tensor,
    noise_coeff:  float = 0.15,
) -> float:
    E = embeddings.detach().float()
    T_d = float(E.numel())
    frob = float(E.norm(p="fro").item())

    sigma_scale = frob / (T_d ** 0.5)
    return float(noise_coeff * sigma_scale)

@dataclass
class SmoothGradResult:
    tokens:          List[str]
    raw_ig_scores:   np.ndarray          # (T,)
    smoothed_scores: np.ndarray          # (T,)
    score_std:       Optional[np.ndarray]  # (T,) or None
    predicted_class: int
    sigma_used:      float
    n_draws:         int
    n_steps:         int
    precision:       str = "unknown"
    text:            str = ""

    def to_dict(self) -> dict:
        return {
            "text":            self.text,
            "precision":       self.precision,
            "predicted_class": self.predicted_class,
            "sigma_used":      self.sigma_used,
            "n_draws":         self.n_draws,
            "n_steps":         self.n_steps,
            "tokens":          self.tokens,
            "raw_ig_scores":   self.raw_ig_scores.tolist(),
            "smoothed_scores": self.smoothed_scores.tolist(),
            "score_std": (
                self.score_std.tolist() if self.score_std is not None else None
            ),
        }

    def as_attribution_dict(self, use_smoothed: bool = True) -> dict:
        scores = self.smoothed_scores if use_smoothed else self.raw_ig_scores
        return {"tokens": self.tokens, "scores": scores.tolist()}


@dataclass
class SmoothGradComparison:
    text:               str
    base_precision:     str
    variant_precision:  str
    tokens:             List[str]
    smoothed_cosine:    float
    smoothed_spearman:  float
    smoothed_delta:     np.ndarray
    raw_cosine:         float
    raw_spearman:       float
    raw_delta:          np.ndarray
    delta_reduction:    np.ndarray
    variance_ratio:     Optional[np.ndarray]
    mean_variance_ratio: Optional[float]
    same_prediction:    bool
    base_class:         int
    variant_class:      int

    def to_dict(self) -> dict:
        return {
            "text":                self.text,
            "base_precision":      self.base_precision,
            "variant_precision":   self.variant_precision,
            "tokens":              self.tokens,
            "smoothed_cosine":     self.smoothed_cosine,
            "smoothed_spearman":   self.smoothed_spearman,
            "smoothed_delta":      self.smoothed_delta.tolist(),
            "raw_cosine":          self.raw_cosine,
            "raw_spearman":        self.raw_spearman,
            "raw_delta":           self.raw_delta.tolist(),
            "delta_reduction":     self.delta_reduction.tolist(),
            "variance_ratio": (
                self.variance_ratio.tolist()
                if self.variance_ratio is not None else None
            ),
            "mean_variance_ratio": self.mean_variance_ratio,
            "same_prediction":     self.same_prediction,
            "base_class":          self.base_class,
            "variant_class":       self.variant_class,
        }

def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    result = spearmanr(a, b)
    corr = getattr(result, "statistic", None) or getattr(result, "correlation", None)
    return float(corr) if corr is not None else float("nan")

class SmoothGradExplainer:
    def __init__(
        self,
        model:       AutoModelForSequenceClassification,
        tokenizer:   AutoTokenizer,
        device:      Optional[torch.device] = None,
        n_draws:     int = 20,
        n_steps:     int = 20,
        noise_coeff: float = 0.15,
        precision:   str = "unknown",
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.device      = device or next(model.parameters()).device
        self.n_draws     = n_draws
        self.n_steps     = n_steps
        self.noise_coeff = noise_coeff
        self.precision   = precision
        self._emb_layer  = _get_word_embeddings(model)

    def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            embeddings = self._emb_layer(input_ids)  # (1, T, d)

        return input_ids, attention_mask, embeddings

    def _forward_fn(
        self,
        embeds:         torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = embeds.shape[0]
        mask  = attention_mask.expand(batch, -1)  # (batch, T)
        return self.model(inputs_embeds=embeds, attention_mask=mask).logits

    def _raw_ig(
        self,
        embeddings:     torch.Tensor,
        attention_mask: torch.Tensor,
        target_class:   int,
    ) -> np.ndarray:
        ig = IntegratedGradients(
            lambda e: self._forward_fn(e, attention_mask)
        )
        baseline = torch.zeros_like(embeddings)

        attributions = ig.attribute(
            embeddings,
            target=target_class,
            n_steps=self.n_steps,
            baselines=baseline,
        )
        return attributions.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()

    def explain(
        self,
        text:          str,
        target:        Optional[int] = None,
    ) -> SmoothGradResult:
        input_ids, attention_mask, embeddings = self._encode(text)

        with torch.no_grad():
            logits       = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted    = int(logits.argmax(dim=-1).item())
        target_class = predicted if target is None else target

        sigma = compute_sigma(embeddings, self.noise_coeff)

        raw_scores = self._raw_ig(embeddings, attention_mask, target_class)

        ig  = IntegratedGradients(lambda e: self._forward_fn(e, attention_mask))
        nt  = NoiseTunnel(ig)
        baseline = torch.zeros_like(embeddings)

        smoothed_attr = nt.attribute(
            embeddings,
            nt_type="smoothgrad",
            nt_samples=self.n_draws,
            stdevs=sigma,
            target=target_class,
            n_steps=self.n_steps,
            baselines=baseline,
        )

        smoothed_scores = (
            smoothed_attr.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()
        )
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        return SmoothGradResult(
            tokens=tokens,
            raw_ig_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            score_std=None,
            predicted_class=predicted,
            sigma_used=sigma,
            n_draws=self.n_draws,
            n_steps=self.n_steps,
            precision=self.precision,
            text=text,
        )

    def explain_with_variance(
        self,
        text:   str,
        target: Optional[int] = None,
    ) -> SmoothGradResult:
        input_ids, attention_mask, embeddings = self._encode(text)

        with torch.no_grad():
            logits    = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted = int(logits.argmax(dim=-1).item())
        target_class = predicted if target is None else target

        sigma = compute_sigma(embeddings, self.noise_coeff)

        raw_scores = self._raw_ig(embeddings, attention_mask, target_class)

        ig      = IntegratedGradients(lambda e: self._forward_fn(e, attention_mask))
        baseline = torch.zeros_like(embeddings)

        draw_scores: List[np.ndarray] = []
        rng = torch.Generator(device=self.device)

        for _ in range(self.n_draws):
            eps = torch.empty_like(embeddings)
            eps.normal_(mean=0.0, std=sigma, generator=rng)

            perturbed = (embeddings + eps).detach().requires_grad_(False)
            draw_attr = ig.attribute(
                perturbed,
                target=target_class,
                n_steps=self.n_steps,
                baselines=baseline,
            )

            draw_score = (
                draw_attr.sum(dim=-1).squeeze(0).detach().cpu().float().numpy()
            )
            draw_scores.append(draw_score)

        draw_matrix    = np.stack(draw_scores, axis=0)
        smoothed_scores = draw_matrix.mean(axis=0)
        score_std       = draw_matrix.std(axis=0, ddof=1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        return SmoothGradResult(
            tokens=tokens,
            raw_ig_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            score_std=score_std,
            predicted_class=predicted,
            sigma_used=sigma,
            n_draws=self.n_draws,
            n_steps=self.n_steps,
            precision=self.precision,
            text=text,
        )

def compare_smoothgrad(
    base:    SmoothGradResult,
    variant: SmoothGradResult,
) -> SmoothGradComparison:
    T_base    = len(base.tokens)
    T_variant = len(variant.tokens)
    if T_base != T_variant:
        raise ValueError(
            f"Token sequence lengths differ: base T={T_base}, variant T={T_variant}. "
        )

    raw_b = base.raw_ig_scores.astype(np.float64)
    raw_v = variant.raw_ig_scores.astype(np.float64)
    sm_b  = base.smoothed_scores.astype(np.float64)
    sm_v  = variant.smoothed_scores.astype(np.float64)

    raw_delta      = (raw_v - raw_b).astype(np.float32)     # (T,)
    smoothed_delta = (sm_v  - sm_b ).astype(np.float32)     # (T,)

    delta_reduction = (
        np.abs(raw_delta) - np.abs(smoothed_delta)
    ).astype(np.float32)

    raw_cosine      = _cosine_np(raw_b, raw_v)
    raw_spearman    = _spearman_np(raw_b, raw_v)
    smoothed_cosine = _cosine_np(sm_b,  sm_v)
    smoothed_spearman = _spearman_np(sm_b, sm_v)

    variance_ratio      = None
    mean_variance_ratio = None

    if base.score_std is not None and variant.score_std is not None:
        variance_ratio = (
            variant.score_std.astype(np.float64)
            / (base.score_std.astype(np.float64) + 1e-9)
        ).astype(np.float32)

        content_mask = np.ones(T_base, dtype=bool)
        if T_base >= 2:
            content_mask[0]  = False   # [CLS]
            content_mask[-1] = False   # [SEP]

        content_ratios = variance_ratio[content_mask]
        if content_ratios.size > 0:
            mean_variance_ratio = float(np.mean(content_ratios))

    return SmoothGradComparison(
        text=base.text,
        base_precision=base.precision,
        variant_precision=variant.precision,
        tokens=base.tokens,
        smoothed_cosine=smoothed_cosine,
        smoothed_spearman=smoothed_spearman,
        smoothed_delta=smoothed_delta,
        raw_cosine=raw_cosine,
        raw_spearman=raw_spearman,
        raw_delta=raw_delta,
        delta_reduction=delta_reduction,
        variance_ratio=variance_ratio,
        mean_variance_ratio=mean_variance_ratio,
        same_prediction=(base.predicted_class == variant.predicted_class),
        base_class=base.predicted_class,
        variant_class=variant.predicted_class,
    )

def explain_smoothgrad_batch(
    texts:        List[str],
    model:        AutoModelForSequenceClassification,
    tokenizer:    AutoTokenizer,
    precision:    str = "unknown",
    device:       Optional[torch.device] = None,
    n_draws:      int = 20,
    n_steps:      int = 20,
    noise_coeff:  float = 0.15,
    with_variance: bool = False,
    verbose:      bool = True,
) -> List[SmoothGradResult]:
    explainer = SmoothGradExplainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_draws=n_draws,
        n_steps=n_steps,
        noise_coeff=noise_coeff,
        precision=precision,
    )

    was_training = model.training
    model.eval()
    results: List[SmoothGradResult] = []

    try:
        for i, text in enumerate(texts):
            if with_variance:
                result = explainer.explain_with_variance(text)
            else:
                result = explainer.explain(text)
            results.append(result)

            if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(texts)):
                print(f"  [smoothgrad] {i+1}/{len(texts)} samples processed")
    finally:
        if was_training:
            model.train()

    return results


def compare_smoothgrad_batch(
    texts:             List[str],
    base_model:        AutoModelForSequenceClassification,
    variant_model:     AutoModelForSequenceClassification,
    tokenizer:         AutoTokenizer,
    base_precision:    str = "fp32",
    variant_precision: str = "int8",
    device:            Optional[torch.device] = None,
    n_draws:           int = 20,
    n_steps:           int = 20,
    noise_coeff:       float = 0.15,
    with_variance:     bool = False,
    verbose:           bool = True,
) -> Tuple[List[SmoothGradResult], List[SmoothGradResult], List[SmoothGradComparison]]:
    base_explainer = SmoothGradExplainer(
        base_model, tokenizer,
        device=device, n_draws=n_draws, n_steps=n_steps,
        noise_coeff=noise_coeff, precision=base_precision,
    )
    variant_explainer = SmoothGradExplainer(
        variant_model, tokenizer,
        device=device, n_draws=n_draws, n_steps=n_steps,
        noise_coeff=noise_coeff, precision=variant_precision,
    )

    base_model.eval()
    variant_model.eval()

    base_results:    List[SmoothGradResult]    = []
    variant_results: List[SmoothGradResult]    = []
    comparisons:     List[SmoothGradComparison] = []

    for i, text in enumerate(texts):
        explain_fn = (
            (lambda exp: exp.explain_with_variance(text))
            if with_variance
            else (lambda exp: exp.explain(text))
        )
        b = explain_fn(base_explainer)
        v = explain_fn(variant_explainer)

        base_results.append(b)
        variant_results.append(v)
        comparisons.append(compare_smoothgrad(b, v))

        if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(texts)):
            print(f"  [smoothgrad_compare] {i+1}/{len(texts)} samples")

    return base_results, variant_results, comparisons

def aggregate_smoothgrad_comparisons(
    comparisons: List[SmoothGradComparison],
) -> dict:
    if not comparisons:
        return {}

    def _stats(vals: List[float]) -> dict:
        clean = [v for v in vals if v is not None and not np.isnan(v)]
        if not clean:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(clean, dtype=np.float64)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "n":      len(clean),
        }

    smoothing_benefits = [
        c.smoothed_spearman - c.raw_spearman
        for c in comparisons
        if not (np.isnan(c.smoothed_spearman) or np.isnan(c.raw_spearman))
    ]

    mean_var_ratios = [
        c.mean_variance_ratio
        for c in comparisons
        if c.mean_variance_ratio is not None
    ]

    return {
        "n_samples":          len(comparisons),
        "base_precision":     comparisons[0].base_precision,
        "variant_precision":  comparisons[0].variant_precision,
        "label_flip_rate":    float(np.mean([not c.same_prediction for c in comparisons])),
        "smoothed_cosine":    _stats([c.smoothed_cosine   for c in comparisons]),
        "smoothed_spearman":  _stats([c.smoothed_spearman for c in comparisons]),
        "raw_cosine":         _stats([c.raw_cosine         for c in comparisons]),
        "raw_spearman":       _stats([c.raw_spearman        for c in comparisons]),
        "smoothing_benefit_spearman": _stats(smoothing_benefits),
        "mean_variance_ratio": _stats(mean_var_ratios) if mean_var_ratios else None,
    }

import json as _json
from pathlib import Path as _Path


def save_smoothgrad_results(
    results:     List[SmoothGradResult],
    output_path: _Path,
    *,
    metadata:    Optional[dict] = None,
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


def save_smoothgrad_comparisons(
    comparisons: List[SmoothGradComparison],
    output_path: _Path,
    *,
    metadata:    Optional[dict] = None,
) -> None:
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata":  metadata or {},
        "n_samples": len(comparisons),
        "results":   [c.to_dict() for c in comparisons],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)


def load_smoothgrad_results(path: _Path) -> Tuple[List[dict], dict]:

    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"SmoothGrad results file not found: {path}\n"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    return payload["results"], payload.get("metadata", {})
