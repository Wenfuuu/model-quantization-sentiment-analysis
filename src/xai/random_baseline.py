from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

def _sample_seed(text: str, baseline_seed: int, draw_index: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    text_hash = int.from_bytes(digest[:8], byteorder="little") % (2**31)
    return (text_hash ^ baseline_seed ^ draw_index) % (2**31)

def _random_scores(
    n_tokens: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.normal(loc=0.0, scale=max(sigma, 1e-6), size=n_tokens)

@dataclass
class RandomAttributionResult:
    tokens:     List[str]
    scores:     np.ndarray
    draw_index: int
    seed_used:  int
    sigma_used: float

class RandomAttributionBaseline:
    def __init__(self, baseline_seed: int = 0, n_draws: int = 30):
        if n_draws < 5:
            warnings.warn(
                f"n_draws={n_draws} is too low for reliable floor estimation. "
                "Use at least 10 draws; 30 is recommended.",
                UserWarning,
                stacklevel=2,
            )
        self.baseline_seed = baseline_seed
        self.n_draws = n_draws

    def draw(
        self,
        text: str,
        tokens: List[str],
        real_scores: np.ndarray,
        draw_index: int,
    ) -> RandomAttributionResult:
        if len(tokens) != len(real_scores):
            raise ValueError(
                f"tokens ({len(tokens)}) and real_scores ({len(real_scores)}) "
                "must have the same length.  Pass the tokens and scores from "
                "the same explainer output."
            )

        seed = _sample_seed(text, self.baseline_seed, draw_index)
        rng  = np.random.default_rng(seed)

        sigma = float(np.std(real_scores)) if len(real_scores) > 1 else 1.0

        scores = _random_scores(len(tokens), sigma, rng)

        return RandomAttributionResult(
            tokens=list(tokens),
            scores=scores,
            draw_index=draw_index,
            seed_used=seed,
            sigma_used=sigma,
        )

    def draw_all(
        self,
        text: str,
        tokens: List[str],
        real_scores: np.ndarray,
    ) -> List[RandomAttributionResult]:
        return [
            self.draw(text, tokens, real_scores, i)
            for i in range(self.n_draws)
        ]

    def to_attribution_dict(self, result: RandomAttributionResult) -> dict:
        return {
            "tokens": result.tokens,
            "scores": result.scores.tolist(),
        }

def compute_baseline_floor(
    text: str,
    real_tokens: List[str],
    real_scores: np.ndarray,
    baseline: RandomAttributionBaseline,
    k_values: Tuple[int, ...] = (3, 5, 10),
) -> Dict[str, Dict[str, float]]:
    from src.evaluation.explanation_drift import (
        spearman_rank_correlation,
        top_k_jaccard,
        sign_flip_rate,
        normalized_magnitude_shift,
    )

    draws = baseline.draw_all(text, real_tokens, real_scores)

    cosine_vals    : List[float] = []
    spearman_vals  : List[float] = []
    flip_vals      : List[float] = []
    mag_shift_vals : List[float] = []
    jaccard_vals   : Dict[int, List[float]] = {k: [] for k in k_values}

    for draw in draws:
        rand_tokens = draw.tokens
        rand_scores = draw.scores.tolist()

        a = np.array(real_scores, dtype=float)
        b = np.array(rand_scores, dtype=float)
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]
        cos_val = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )
        cosine_vals.append(cos_val)

        rho, _ = spearman_rank_correlation(
            real_tokens, list(real_scores),
            rand_tokens, rand_scores,
        )
        spearman_vals.append(rho)

        flip = sign_flip_rate(
            real_tokens, list(real_scores),
            rand_tokens, rand_scores,
        )
        flip_vals.append(flip)

        mag = normalized_magnitude_shift(
            real_tokens, list(real_scores),
            rand_tokens, rand_scores,
        )
        mag_shift_vals.append(mag)

        for k in k_values:
            j = top_k_jaccard(
                real_tokens, list(real_scores),
                rand_tokens, rand_scores,
                k=k,
            )
            jaccard_vals[k].append(j)

    def _floor_stats(vals: List[float]) -> Dict[str, float]:
        finite = [v for v in vals if not np.isnan(v)]
        n = len(finite)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n_draws": 0}
        return {
            "mean":    float(np.mean(finite)),
            "std":     float(np.std(finite, ddof=1) if n > 1 else 0.0),
            "min":     float(np.min(finite)),
            "max":     float(np.max(finite)),
            "n_draws": n,
        }

    result = {
        "cosine":                    _floor_stats(cosine_vals),
        "spearman":                  _floor_stats(spearman_vals),
        "sign_flip_rate":            _floor_stats(flip_vals),
        "normalized_magnitude_shift": _floor_stats(mag_shift_vals),
    }
    for k in k_values:
        result[f"jaccard_top{k}"] = _floor_stats(jaccard_vals[k])

    return result

def above_floor_delta(
    real_metrics: Dict[str, float],
    floor_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for metric_name, real_val in real_metrics.items():
        if metric_name in floor_metrics:
            floor_mean = floor_metrics[metric_name]["mean"]
            if np.isnan(real_val) or np.isnan(floor_mean):
                deltas[metric_name] = float("nan")
            else:
                deltas[metric_name] = float(real_val) - floor_mean
    return deltas

def evaluate_baseline_floor_batch(
    samples: List[dict],
    real_attribution_results: List[dict],
    baseline: RandomAttributionBaseline,
    k_values: Tuple[int, ...] = (3, 5, 10),
) -> List[Dict[str, Dict[str, float]]]:
    if len(samples) != len(real_attribution_results):
        raise ValueError(
            f"samples ({len(samples)}) and real_attribution_results "
            f"({len(real_attribution_results)}) must have the same length."
        )

    floors = []
    for i, (sample, attr_result) in enumerate(
        zip(samples, real_attribution_results)
    ):
        text        = sample["text"]
        real_tokens = attr_result["tokens"]
        real_scores = np.array(attr_result["scores"], dtype=float)

        floor = compute_baseline_floor(
            text=text,
            real_tokens=real_tokens,
            real_scores=real_scores,
            baseline=baseline,
            k_values=k_values,
        )
        floors.append(floor)

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [random baseline] {i+1}/{len(samples)} samples processed")

    return floors

import json as _json
from pathlib import Path as _Path

def save_baseline_floors(
    floors: List[Dict[str, Dict[str, float]]],
    output_path: _Path,
    *,
    metadata: Optional[dict] = None,
) -> None:
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata":  metadata or {},
        "n_samples": len(floors),
        "floors":    floors,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)

def load_baseline_floors(
    path: _Path,
) -> Tuple[List[Dict[str, Dict[str, float]]], dict]:
    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline floor file not found: {path}\n"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    return payload["floors"], payload.get("metadata", {})

def run_random_baselines():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [(int(row["sample_id"]), row["text"]) for _, row in df_sub.iterrows()]

    baseline  = RandomAttributionBaseline(baseline_seed=42, n_draws=30)
    n_draws   = baseline.n_draws
    n_total   = len(samples) * n_draws
    n_done    = 0
    n_skipped = 0

    print(f"\n  Random Baselines: {len(samples)} samples x {n_draws} draws = {n_total} files")
    print(f"  sigma source priority: ig_fp32 > occ_fp32 > shap_fp32")
    print(f"  Output: {_OUT_DIR}")

    for sid, text in samples:
        sigma_source = None
        for prefix in ("ig_fp32", "occ_fp32", "shap_fp32"):
            p = _OUT_DIR / f"{prefix}_{sid}.npy"
            if p.exists():
                sigma_source = p
                break

        if sigma_source is None:
            print(f"  [SKIP] sid={sid}: no sigma source (ig_fp32/occ_fp32/shap_fp32)")
            n_skipped += n_draws
            continue

        real_scores = np.load(sigma_source).astype(np.float64)
        words = text.split()
        L = min(len(words), len(real_scores))
        real_scores = real_scores[:L]
        words_trimmed = words[:L]

        for i in range(n_draws):
            out_path = _OUT_DIR / f"random_{sid}_{i}.npy"
            if out_path.exists():
                n_done += 1
                continue
            result = baseline.draw(text, words_trimmed, real_scores, i)
            np.save(out_path, result.scores.astype(np.float32))
            n_done += 1

        if n_done % 500 == 0 or n_done + n_skipped == n_total:
            print(f"  [progress] {n_done}/{n_total}  skipped={n_skipped}")

    print(f"\n  Random baselines complete: {n_done}/{n_total} files saved  skipped={n_skipped}")
    print(f"  File pattern: random_{{sample_id}}_{{run}}.npy")
