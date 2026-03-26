import numpy as np
from pathlib import Path
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

def bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    a = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if len(a) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(a, size=len(a), replace=True).mean()
                           for _ in range(n_boot)])
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return lo, hi

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

def run_stability_analysis():
    import pandas as pd
    from scipy.stats import wilcoxon as scipy_wilcoxon
    from scipy import stats as scipy_stats

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [(int(row["sample_id"]), row["text"]) for _, row in df_sub.iterrows()]

    VARIANTS_7 = ["ptq_fp16", "ptq_int8", "ptq_int4", "qat_fp32",
                   "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4"]
    METHODS = ["lime", "occ", "shap"]

    def _load_pair(method, vname, sid, text):
        fp32_path = _OUT_DIR / f"{method}_fp32_{sid}.npy"
        var_path  = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
        if not fp32_path.exists() or not var_path.exists():
            return None
        words = text.split()
        fp32_s = np.load(fp32_path).astype(np.float64)
        var_s  = np.load(var_path).astype(np.float64)
        L = min(len(words), len(fp32_s), len(var_s))
        return words[:L], fp32_s[:L].tolist(), var_s[:L].tolist()

    per_rows = []
    n_found = 0

    for method in METHODS:
        for vname in VARIANTS_7:
            for sid, text in samples:
                pair = _load_pair(method, vname, sid, text)
                if pair is None:
                    continue
                words_l, fp32_l, var_l = pair
                rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l, var_l)
                if not np.isnan(rho):
                    j3  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=3)
                    j5  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=5)
                    j10 = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=10)
                    per_rows.append({"method": method, "variant": vname, "sample_id": sid,
                                      "spearman_rho": rho,
                                      "jaccard_k3": j3, "jaccard_k5": j5, "jaccard_k10": j10})
                    n_found += 1
            print(f"  {method:5s} x {vname:20s}: {n_found} samples found")
            n_found = 0

    ig_found = 0
    for sid, text in samples:
        fp32_path = _OUT_DIR / f"ig_fp32_{sid}.npy"
        qat_path  = _OUT_DIR / f"ig_qat_ste_{sid}.npy"
        if not fp32_path.exists() or not qat_path.exists():
            continue
        words = text.split()
        fp32_s = np.load(fp32_path).astype(np.float64)
        var_s  = np.load(qat_path).astype(np.float64)
        L = min(len(words), len(fp32_s), len(var_s))
        words_l = words[:L]
        fp32_l  = fp32_s[:L].tolist()
        var_l   = var_s[:L].tolist()
        rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l, var_l)
        if not np.isnan(rho):
            j3  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=3)
            j5  = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=5)
            j10 = top_k_jaccard(words_l, fp32_l, words_l, var_l, k=10)
            per_rows.append({"method": "ig", "variant": "qat_ste", "sample_id": sid,
                              "spearman_rho": rho,
                              "jaccard_k3": j3, "jaccard_k5": j5, "jaccard_k10": j10})
            ig_found += 1
    print(f"  ig    x qat_ste             : {ig_found} samples")

    if not per_rows:
        print("  [WARN] No .npy pairs found. Run LIME/OCC/SHAP steps first.")
        return

    df_per = pd.DataFrame(per_rows)
    per_path = _RES_DIR / "stability_perSample.csv"
    df_per.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df_per)} rows -> {per_path}")

    n_comparisons    = df_per.groupby(["method", "variant"]).ngroups
    alpha_bonferroni = 0.05 / n_comparisons

    sum_rows = []
    for (method, vname), grp in df_per.groupby(["method", "variant"]):
        rhos = grp["spearman_rho"].dropna().tolist()
        j5s  = grp["jaccard_k5"].dropna().tolist()
        if len(rhos) < 2:
            continue

        rho_lo, rho_hi = bootstrap_mean_ci(rhos)
        j5_lo,  j5_hi  = bootstrap_mean_ci(j5s)

        diffs = [r - 1.0 for r in rhos]
        try:
            w_stat, w_p = scipy_wilcoxon(diffs, alternative="less")
            z         = scipy_stats.norm.ppf(w_p)
            effect_r  = abs(z) / np.sqrt(len(rhos))
        except ValueError:
            w_stat = w_p = effect_r = float("nan")

        sum_rows.append({
            "method": method, "variant": vname, "n": len(rhos),
            "mean_rho":     round(float(np.mean(rhos)), 4),
            "std_rho":      round(float(np.std(rhos)),  4),
            "ci95_lo_rho":  round(rho_lo, 4),
            "ci95_hi_rho":  round(rho_hi, 4),
            "mean_j5":      round(float(np.mean(j5s)), 4),
            "ci95_lo_j5":   round(j5_lo,  4),
            "ci95_hi_j5":   round(j5_hi,  4),
            "wilcoxon_stat":          round(w_stat,    4),
            "wilcoxon_p":             round(w_p,       6),
            "bonferroni_alpha":        round(alpha_bonferroni, 6),
            "significant_bonferroni":  bool(w_p < alpha_bonferroni),
            "effect_size_r":           round(effect_r,  4),
        })

    df_sum = pd.DataFrame(sum_rows)
    sum_path = _RES_DIR / "stability_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary ({n_comparisons} comparisons, "
          f"Bonferroni alpha={alpha_bonferroni:.5f}) -> {sum_path}")

    floor_rows = []
    for method in METHODS:
        for sid, text in samples:
            fp32_path  = _OUT_DIR / f"{method}_fp32_{sid}.npy"
            rand_files = sorted(_OUT_DIR.glob(f"random_{sid}_*.npy"))
            if not fp32_path.exists() or not rand_files:
                continue
            words  = text.split()
            fp32_s = np.load(fp32_path).astype(np.float64)
            L      = min(len(words), len(fp32_s))
            fp32_l = fp32_s[:L].tolist()
            words_l = words[:L]
            rho_vals, j5_vals = [], []
            for rf in rand_files:
                rand_s = np.load(rf).astype(np.float64)
                Lr = min(L, len(rand_s))
                rand_l = rand_s[:Lr].tolist()
                rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l[:Lr], rand_l)
                j5     = top_k_jaccard(words_l, fp32_l, words_l[:Lr], rand_l, k=5)
                rho_vals.append(rho)
                j5_vals.append(j5)
            floor_rows.append({"method": method, "sample_id": sid,
                                "floor_mean_rho": round(float(np.nanmean(rho_vals)), 4),
                                "floor_mean_j5":  round(float(np.nanmean(j5_vals)),  4)})

    df_floor = pd.DataFrame(floor_rows)
    floor_path = _RES_DIR / "stability_random_floor.csv"
    df_floor.to_csv(floor_path, index=False, encoding="utf-8")
    print(f"  Saved random floor -> {floor_path}")

    print(f"\n  {'method':5s}  {'variant':20s}  {'rho':>6s} [95% CI]        "
          f"{'J@5':>5s}  {'r':>5s}  sig?")
    for _, row in df_sum.sort_values(["method", "mean_rho"]).iterrows():
        sig = "*" if row["significant_bonferroni"] else " "
        print(f"  {row['method']:5s}  {row['variant']:20s}  "
              f"{row['mean_rho']:6.3f} [{row['ci95_lo_rho']:.3f},{row['ci95_hi_rho']:.3f}]  "
              f"{row['mean_j5']:5.3f}  {row['effect_size_r']:5.3f}  {sig}")

    compute_power_analysis()

def compute_power_analysis():
    import math
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _RES_DIR = _PROJECT_ROOT / "results"
    per_path = _RES_DIR / "stability_perSample.csv"
    if not per_path.exists():
        print("  [WARN] stability_perSample.csv not found. Run Stability Analysis first.")
        return

    df = pd.read_csv(per_path)
    Z = 2.487
    n = 50
    power_rows = []
    print(f"\n  Power analysis (alpha=0.05, power=0.80, n={n}):")
    print(f"  {'method':5s}  {'observed_sd':>11s}  {'min_detectable_delta_rho':>24s}")
    for method, grp in df.groupby("method"):
        sd = float(grp["spearman_rho"].dropna().std())
        delta = Z * sd / math.sqrt(n)
        power_rows.append({"method": method, "observed_sd": round(sd, 4),
                            "min_detectable_delta_rho": round(delta, 4)})
        print(f"  {method:5s}  {sd:11.4f}  {delta:24.4f}")

    out_path = _RES_DIR / "power_analysis.csv"
    pd.DataFrame(power_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved -> {out_path}")

def compute_cross_method_agreement():
    import pandas as pd
    from itertools import combinations
    from scipy.stats import spearmanr

    _PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [(int(row["sample_id"]), row["text"]) for _, row in df_sub.iterrows()]

    METHODS = ["ig", "gxi", "lime", "occ", "shap"]

    scores: dict = {m: {} for m in METHODS}
    for method in METHODS:
        for sid, text in samples:
            p = _OUT_DIR / f"{method}_fp32_{sid}.npy"
            if p.exists():
                scores[method][sid] = np.load(p).astype(np.float64)

    present = [m for m in METHODS if scores[m]]
    if len(present) < 2:
        print(f"  [WARN] Need at least 2 methods with FP32 .npy files, found: {present}")
        return

    per_rows = []
    for m_a, m_b in combinations(present, 2):
        shared_sids = sorted(set(scores[m_a]) & set(scores[m_b]))
        for sid in shared_sids:
            a = scores[m_a][sid]
            b = scores[m_b][sid]
            L = min(len(a), len(b))
            if L < 3:
                continue
            rho, _ = spearmanr(a[:L], b[:L])
            per_rows.append({"method_a": m_a, "method_b": m_b,
                             "sample_id": sid, "spearman_rho": float(rho)})

    if not per_rows:
        print("  [WARN] No overlapping FP32 files found across any method pair.")
        return

    df_per = pd.DataFrame(per_rows)
    per_path = _RES_DIR / "cross_method_agreement.csv"
    df_per.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df_per)} rows -> {per_path}")

    sum_rows = []
    for (m_a, m_b), grp in df_per.groupby(["method_a", "method_b"]):
        rhos = grp["spearman_rho"].dropna()
        sum_rows.append({"method_a": m_a, "method_b": m_b, "n": len(rhos),
                         "mean_rho": round(float(rhos.mean()), 4),
                         "std_rho":  round(float(rhos.std()),  4)})
    df_sum = pd.DataFrame(sum_rows)
    sum_path = _RES_DIR / "cross_method_agreement_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary -> {sum_path}")

    mean_lookup = {(r["method_a"], r["method_b"]): r["mean_rho"] for _, r in df_sum.iterrows()}
    header = f"  {'':6s}" + "".join(f"  {m:>6s}" for m in present)
    print(f"\n{header}")
    for m_a in present:
        row_str = f"  {m_a:6s}"
        for m_b in present:
            if m_a == m_b:
                row_str += f"  {'1.000':>6s}"
            else:
                val = mean_lookup.get((m_a, m_b), mean_lookup.get((m_b, m_a)))
                row_str += f"  {val:>6.3f}" if val is not None else f"  {'  n/a':>6s}"
        print(row_str)