import numpy as np
from scipy import stats
from typing import List, Optional, Tuple
import warnings

def _align_attributions(
    tokens_a: List[str],
    scores_a: List[float],
    tokens_b: List[str],
    scores_b: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    vocab = list(dict.fromkeys(tokens_a + tokens_b))
    dict_a = dict(zip(tokens_a, scores_a))
    dict_b = dict(zip(tokens_b, scores_b))
    arr_a = np.array([dict_a.get(t, 0.0) for t in vocab], dtype=float)
    arr_b = np.array([dict_b.get(t, 0.0) for t in vocab], dtype=float)
    return arr_a, arr_b

def spearman_rank_correlation(
    tokens_a: List[str],
    scores_a: List[float],
    tokens_b: List[str],
    scores_b: List[float],
) -> Tuple[float, float]:
    arr_a, arr_b = _align_attributions(tokens_a, scores_a, tokens_b, scores_b)
    if len(arr_a) < 3:
        return float("nan"), float("nan")
    rho, p_val = stats.spearmanr(arr_a, arr_b)
    return float(rho), float(p_val)


def top_k_jaccard(
    tokens_a: List[str],
    scores_a: List[float],
    tokens_b: List[str],
    scores_b: List[float],
    k: int = 5,
    by_absolute: bool = True,
) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")

    score_fn = abs if by_absolute else (lambda x: x)

    top_a = {t for t, s in sorted(zip(tokens_a, scores_a), key=lambda x: score_fn(x[1]), reverse=True)[:k]}
    top_b = {t for t, s in sorted(zip(tokens_b, scores_b), key=lambda x: score_fn(x[1]), reverse=True)[:k]}

    intersection = len(top_a & top_b)
    union = len(top_a | top_b)
    return intersection / union if union > 0 else 0.0


def sign_flip_rate(
    tokens_a: List[str],
    scores_a: List[float],
    tokens_b: List[str],
    scores_b: List[float],
    threshold: float = 1e-6,
) -> float:
    dict_a = dict(zip(tokens_a, scores_a))
    dict_b = dict(zip(tokens_b, scores_b))
    shared = set(dict_a.keys()) & set(dict_b.keys())

    flips = 0
    eligible = 0
    for t in shared:
        sa = dict_a[t]
        sb = dict_b[t]
        if abs(sa) > threshold and abs(sb) > threshold:
            eligible += 1
            if np.sign(sa) != np.sign(sb):
                flips += 1

    return flips / eligible if eligible > 0 else float("nan")

def normalized_magnitude_shift(
    tokens_a: List[str],
    scores_a: List[float],
    tokens_b: List[str],
    scores_b: List[float],
) -> float:
    arr_a, arr_b = _align_attributions(tokens_a, scores_a, tokens_b, scores_b)
    eps = 1e-9
    scale = np.mean(np.abs(arr_a)) + eps
    shift = np.mean(np.abs(arr_b - arr_a))
    return float(shift / scale)

def aggregate_explanation_drift(
    explanations_a: List[dict],
    explanations_b: List[dict],
    predictions_a: Optional[List[str]] = None,
    predictions_b: Optional[List[str]] = None,
    k_values: Tuple[int, ...] = (3, 5, 10),
) -> dict:
    assert len(explanations_a) == len(explanations_b), (
        f"Length mismatch: {len(explanations_a)} vs {len(explanations_b)}"
    )
    n = len(explanations_a)

    rhos, p_vals = [], []
    jaccards = {k: [] for k in k_values}
    flips = []
    mag_shifts = []
    prediction_flipped = []

    for i, (exp_a, exp_b) in enumerate(zip(explanations_a, explanations_b)):
        ta, sa = exp_a["tokens"], exp_a["scores"]
        tb, sb = exp_b["tokens"], exp_b["scores"]

        rho, p = spearman_rank_correlation(ta, sa, tb, sb)
        rhos.append(rho)
        p_vals.append(p)

        for k in k_values:
            j = top_k_jaccard(ta, sa, tb, sb, k=k)
            jaccards[k].append(j)

        flips.append(sign_flip_rate(ta, sa, tb, sb))
        mag_shifts.append(normalized_magnitude_shift(ta, sa, tb, sb))

        if predictions_a is not None and predictions_b is not None:
            prediction_flipped.append(int(predictions_a[i] != predictions_b[i]))

    def _stats(arr):
        a = np.array([x for x in arr if not np.isnan(x)], dtype=float)
        if len(a) == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "n": int(len(a)),
        }

    results = {
        "n_samples": n,
        "spearman_rho": _stats(rhos),
        "sign_flip_rate": _stats(flips),
        "normalized_magnitude_shift": _stats(mag_shifts),
    }

    for k in k_values:
        results[f"jaccard_top{k}"] = _stats(jaccards[k])

    if prediction_flipped:
        n_flipped = sum(prediction_flipped)
        results["prediction_flip"] = {
            "n_flipped": n_flipped,
            "rate": n_flipped / n,
        }
        flip_rhos = [r for r, f in zip(rhos, prediction_flipped) if f == 1 and not np.isnan(r)]
        stable_rhos = [r for r, f in zip(rhos, prediction_flipped) if f == 0 and not np.isnan(r)]
        results["spearman_rho_on_prediction_flip"] = _stats(flip_rhos)
        results["spearman_rho_on_prediction_stable"] = _stats(stable_rhos)

    return results

def lime_to_attribution(lime_result: dict, label_idx: Optional[int] = None) -> dict:
    features = lime_result.get("top_features", [])
    tokens = [f[0] for f in features]
    scores = [f[1] for f in features]
    return {"tokens": tokens, "scores": scores}


def shap_to_attribution(shap_result: dict) -> dict:
    token_imp = shap_result.get("token_importance", [])
    tokens = [t[0] for t in token_imp]
    scores = [t[1] for t in token_imp]
    return {"tokens": tokens, "scores": scores}


def ig_to_attribution(ig_result: dict) -> dict:
    tokens = ig_result.get("tokens", [])
    scores = ig_result.get("scores", [])
    filtered = [(t, s) for t, s in zip(tokens, scores)
                if t not in ("[CLS]", "[SEP]", "[PAD]")]
    if not filtered:
        return {"tokens": [], "scores": []}
    t_out, s_out = zip(*filtered)
    return {"tokens": list(t_out), "scores": list(s_out)}

def wilcoxon_drift_test(
    explanations_a: List[dict],
    explanations_b: List[dict],
) -> dict:
    rhos = []
    for exp_a, exp_b in zip(explanations_a, explanations_b):
        rho, _ = spearman_rank_correlation(
            exp_a["tokens"], exp_a["scores"],
            exp_b["tokens"], exp_b["scores"],
        )
        if not np.isnan(rho):
            rhos.append(rho)

    if len(rhos) < 10:
        warnings.warn("Fewer than 10 valid Spearman values; test result unreliable.")

    diffs = [r - 1.0 for r in rhos]
    stat, p = stats.wilcoxon(diffs, alternative="less")
    z = stats.norm.ppf(p)
    effect_r = abs(z) / np.sqrt(len(rhos))

    return {
        "wilcoxon_statistic": float(stat),
        "p_value": float(p),
        "effect_size_r": float(effect_r),
        "n_valid": len(rhos),
        "median_rho": float(np.median(rhos)),
        "interpretation": (
            "Quantization significantly alters explanation rankings (p < 0.05)"
            if p < 0.05 else
            "No significant ranking difference detected (p >= 0.05)"
        ),
    }
