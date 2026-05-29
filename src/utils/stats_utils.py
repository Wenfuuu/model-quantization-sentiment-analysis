from __future__ import annotations

from typing import List

import numpy as np
import scipy.stats as _sp_stats
from statsmodels.stats.multitest import multipletests as _multipletests

def bootstrap_spearman(
    a,
    b,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
    a_arr, b_arr = a_arr[mask], b_arr[mask]

    if len(a_arr) < 3:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rho = float(_sp_stats.spearmanr(a_arr, b_arr).statistic)

    rng = np.random.default_rng(random_state)
    n = len(a_arr)
    boot_rhos: List[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            r = float(_sp_stats.spearmanr(a_arr[idx], b_arr[idx]).statistic)
        except Exception:
            r = float("nan")
        if not np.isnan(r):
            boot_rhos.append(r)

    if len(boot_rhos) < 10:
        return {"rho": round(rho, 6), "ci_low": float("nan"), "ci_high": float("nan")}

    alpha = 1.0 - confidence_level
    ci_low  = float(np.percentile(boot_rhos, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))
    return {
        "rho":     round(rho,     6),
        "ci_low":  round(ci_low,  6),
        "ci_high": round(ci_high, 6),
    }

def wilcoxon_test(rho_vec_a, rho_vec_b) -> dict:
    a = np.asarray(rho_vec_a, dtype=float)
    b = np.asarray(rho_vec_b, dtype=float)
    diffs = a - b
    mask  = ~np.isnan(diffs)
    diffs = diffs[mask]
    nonzero = diffs[diffs != 0.0]

    if len(nonzero) < 10:
        return {"stat": float("nan"), "p_value": float("nan")}

    try:
        stat, p = _sp_stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        return {"stat": float("nan"), "p_value": float("nan")}

    return {"stat": float(stat), "p_value": float(p)}

def bonferroni_correct(p_values: List[float]) -> List[float]:
    arr = np.asarray(p_values, dtype=float)
    out = np.full(len(arr), float("nan"))
    valid_idx = np.where(~np.isnan(arr))[0]

    if len(valid_idx) == 0:
        return out.tolist()

    _, corrected, _, _ = _multipletests(
        arr[valid_idx], alpha=0.05, method="bonferroni"
    )
    for i, ci in zip(valid_idx, corrected):
        out[i] = float(ci)

    return out.tolist()

def cohens_d(a, b) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]

    if len(a_arr) < 2 or len(b_arr) < 2:
        return float("nan")

    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))
    var_a     = float(np.var(a_arr, ddof=1))
    var_b     = float(np.var(b_arr, ddof=1))
    pooled_sd = float(np.sqrt(
        ((len(a_arr) - 1) * var_a + (len(b_arr) - 1) * var_b)
        / (len(a_arr) + len(b_arr) - 2)
    ))

    if pooled_sd == 0.0:
        return float("nan")

    return round(mean_diff / pooled_sd, 6)

def rank_biserial_one_sample(values, mu: float = 1.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    diffs = arr - float(mu)
    nonzero = diffs[diffs != 0.0]
    n = len(nonzero)

    if n < 1:
        return float("nan")

    ranks = _sp_stats.rankdata(np.abs(nonzero))
    w_pos = float(np.sum(ranks[nonzero > 0]))
    w_neg = float(np.sum(ranks[nonzero < 0]))
    total = n * (n + 1) / 2.0

    if total == 0.0:
        return float("nan")

    return round((w_pos - w_neg) / total, 6)
