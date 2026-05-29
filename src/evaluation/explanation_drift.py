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


def _build_stratified_subsample(
    test_csv: Path,
    val_csv: Path,
    n_total: int = 500,
    min_per_class: int = 166,
    seed: int = 42,
) -> List[Tuple[int, str]]:
    import pandas as pd

    rng = np.random.default_rng(seed)
    all_rows: List[Tuple[int, str, int]] = []

    if test_csv.exists():
        df = pd.read_csv(test_csv).dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""].reset_index(drop=True)
        for idx, row in df.iterrows():
            all_rows.append((int(idx), str(row["text"]), int(row["label"])))

    n_test = len(all_rows)
    if val_csv.exists():
        df_val = pd.read_csv(val_csv).dropna(subset=["text", "label"])
        df_val["text"] = df_val["text"].astype(str).str.strip()
        df_val = df_val[df_val["text"] != ""].reset_index(drop=True)
        for idx, row in df_val.iterrows():
            all_rows.append((int(idx + n_test), str(row["text"]), int(row["label"])))

    if not all_rows:
        warnings.warn(f"No data found in {test_csv} or {val_csv}")
        return []

    by_label: dict = {}
    for sid, text, label in all_rows:
        by_label.setdefault(label, []).append((sid, text))

    chosen_sids: set = set()
    chosen: List[Tuple[int, str]] = []
    for label, rows in sorted(by_label.items()):
        n_draw = min(min_per_class, len(rows))
        indices = rng.choice(len(rows), size=n_draw, replace=False)
        for i in indices:
            sid, text = rows[i]
            if sid not in chosen_sids:
                chosen.append((sid, text))
                chosen_sids.add(sid)

    remaining = [(sid, text) for sid, text, _ in all_rows if sid not in chosen_sids]
    if remaining and len(chosen) < n_total:
        perm = rng.permutation(len(remaining))
        for i in perm:
            if len(chosen) >= n_total:
                break
            sid, text = remaining[i]
            if sid not in chosen_sids:
                chosen.append((sid, text))
                chosen_sids.add(sid)

    return chosen[:n_total]

def _scipy_bootstrap_ci(
    values: List[float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    from scipy.stats import bootstrap as _scipy_bootstrap
    a = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if len(a) < 2:
        return float("nan"), float("nan")
    result = _scipy_bootstrap(
        (a,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


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
    import json as _json
    from scipy.stats import wilcoxon as scipy_wilcoxon
    from scipy import stats as scipy_stats
    from statsmodels.stats.multitest import multipletests

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _DATA_DIR  = _PROJECT_ROOT / "data" / "processed"
    _TEST_CSV  = _DATA_DIR / "smsa_test_v2.csv"
    _VAL_CSV   = _DATA_DIR / "smsa_val_v2.csv"
    _OUT_DIR   = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR   = _PROJECT_ROOT / "results"

    samples = _build_stratified_subsample(_TEST_CSV, _VAL_CSV, n_total=500, min_per_class=166, seed=42)
    if not samples:
        print(f"  [ERROR] Could not build stratified subsample from {_TEST_CSV}")
        return
    print(f"  Stratified subsample: {len(samples)} samples")

    VARIANTS_7 = ["ptq_fp16", "ptq_int8", "ptq_int4", "qat_fp32",
                   "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4", "fp32_control"]
    METHODS = ["lime", "occ", "shap"]

    def _load_pair(method, vname, sid, text):
        fp32_path = _OUT_DIR / f"{method}_fp32_{sid}.npy"
        var_path  = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
        if not fp32_path.exists() or not var_path.exists():
            return None
        words  = text.split()
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
        words  = text.split()
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

    sum_rows = []
    for (method, vname), grp in df_per.groupby(["method", "variant"]):
        rhos = grp["spearman_rho"].dropna().tolist()
        j5s  = grp["jaccard_k5"].dropna().tolist()
        if len(rhos) < 2:
            continue

        rho_lo, rho_hi = _scipy_bootstrap_ci(rhos, n_resamples=2000)
        j5_lo,  j5_hi  = _scipy_bootstrap_ci(j5s,  n_resamples=2000)

        diffs = [r - 1.0 for r in rhos]
        try:
            w_stat, w_p = scipy_wilcoxon(diffs, alternative="less")
            z        = scipy_stats.norm.ppf(w_p)
            effect_r = abs(z) / np.sqrt(len(rhos))
        except ValueError:
            w_stat = w_p = effect_r = float("nan")

        sum_rows.append({
            "method": method, "variant": vname, "n": len(rhos),
            "mean_rho":      round(float(np.mean(rhos)), 4),
            "std_rho":       round(float(np.std(rhos)),  4),
            "ci95_lo_rho":   round(rho_lo, 4),
            "ci95_hi_rho":   round(rho_hi, 4),
            "mean_j5":       round(float(np.mean(j5s)), 4),
            "ci95_lo_j5":    round(j5_lo,  4),
            "ci95_hi_j5":    round(j5_hi,  4),
            "wilcoxon_stat": round(w_stat,   4),
            "wilcoxon_p":    round(w_p,      6),
            "effect_size_r": round(effect_r, 4),
        })

    df_sum = pd.DataFrame(sum_rows)
    n_comparisons = len(df_sum)

    valid_ps = df_sum["wilcoxon_p"].fillna(1.0).tolist()
    _, p_bonf, _, _ = multipletests(valid_ps, alpha=0.05, method="bonferroni")
    df_sum["p_bonferroni"]           = [round(float(p), 6) for p in p_bonf]
    df_sum["significant_bonferroni"] = df_sum["p_bonferroni"] < 0.05
    df_sum["bonferroni_alpha"]       = round(0.05 / max(1, n_comparisons), 6)

    sum_path = _RES_DIR / "stability_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary ({n_comparisons} comparisons) -> {sum_path}")

    PTQ_QAT_PAIRS = [
        ("ptq_fp16", "qat_onnx_fp16"),
        ("ptq_int8", "qat_onnx_int8"),
        ("ptq_int4", "qat_onnx_int4"),
    ]
    ptq_qat_rows = []
    for method in METHODS:
        for ptq_v, qat_v in PTQ_QAT_PAIRS:
            ptq_grp = df_per[(df_per["method"] == method) & (df_per["variant"] == ptq_v)]
            qat_grp = df_per[(df_per["method"] == method) & (df_per["variant"] == qat_v)]
            merged  = ptq_grp.merge(qat_grp, on="sample_id", suffixes=("_ptq", "_qat"))
            n_paired = int(len(merged.dropna(subset=["spearman_rho_ptq", "spearman_rho_qat"])))
            if n_paired < 10:
                ptq_qat_rows.append({
                    "method": method, "ptq_variant": ptq_v, "qat_variant": qat_v,
                    "n_paired": n_paired,
                    "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
                    "p_bonferroni": float("nan"), "significant": False,
                })
                continue
            diffs = (merged["spearman_rho_ptq"].values
                     - merged["spearman_rho_qat"].values)
            diffs = diffs[~np.isnan(diffs)]
            try:
                w_stat, w_p = scipy_wilcoxon(diffs)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            ptq_qat_rows.append({
                "method": method, "ptq_variant": ptq_v, "qat_variant": qat_v,
                "n_paired": n_paired,
                "wilcoxon_stat": round(float(w_stat), 4),
                "wilcoxon_p":    round(float(w_p),    6),
                "p_bonferroni": float("nan"), "significant": False,
            })

    pq_ps = [r["wilcoxon_p"] if not np.isnan(r["wilcoxon_p"]) else 1.0
              for r in ptq_qat_rows]
    if pq_ps:
        _, pq_bonf, _, _ = multipletests(pq_ps, alpha=0.05, method="bonferroni")
        for i, row in enumerate(ptq_qat_rows):
            row["p_bonferroni"] = round(float(pq_bonf[i]), 6)
            row["significant"]  = bool(pq_bonf[i] < 0.05)

    df_pq = pd.DataFrame(ptq_qat_rows)
    pq_path = _RES_DIR / "ptq_vs_qat_wilcoxon.csv"
    df_pq.to_csv(pq_path, index=False, encoding="utf-8")
    print(f"  Saved PTQ vs QAT Wilcoxon -> {pq_path}")

    cross_seed_samples = _build_stratified_subsample(
        _TEST_CSV, _VAL_CSV, n_total=150, min_per_class=50, seed=0
    )
    SEED_PAIRS = [(42, 123), (42, 456), (123, 456)]
    cross_seed_rows = []
    for method in METHODS:
        for seed_a, seed_b in SEED_PAIRS:
            rho_vals = []
            for sid, text in cross_seed_samples:
                p_a = _OUT_DIR / f"{method}_fp32_s{seed_a}_{sid}.npy"
                p_b = _OUT_DIR / f"{method}_fp32_s{seed_b}_{sid}.npy"
                if not p_a.exists() or not p_b.exists():
                    continue
                words = text.split()
                s_a = np.load(p_a).astype(np.float64)
                s_b = np.load(p_b).astype(np.float64)
                L = min(len(words), len(s_a), len(s_b))
                if L < 3:
                    continue
                rho, _ = spearman_rank_correlation(
                    words[:L], s_a[:L].tolist(), words[:L], s_b[:L].tolist()
                )
                if not np.isnan(rho):
                    rho_vals.append(rho)
            if rho_vals:
                ci_lo, ci_hi = _scipy_bootstrap_ci(rho_vals, n_resamples=2000)
            else:
                ci_lo = ci_hi = float("nan")
            cross_seed_rows.append({
                "method":      method,
                "seed_a":      seed_a,
                "seed_b":      seed_b,
                "n":           len(rho_vals),
                "mean_rho":    round(float(np.nanmean(rho_vals)), 4) if rho_vals else float("nan"),
                "ci95_lo_rho": round(ci_lo, 4),
                "ci95_hi_rho": round(ci_hi, 4),
            })

    df_cross = pd.DataFrame(cross_seed_rows)
    cross_path = _RES_DIR / "cross_seed_stability.csv"
    df_cross.to_csv(cross_path, index=False, encoding="utf-8")
    print(f"  Saved cross-seed stability ({len(cross_seed_samples)} samples) -> {cross_path}")

    floor_rows = []
    for method in METHODS:
        for sid, text in samples:
            fp32_path  = _OUT_DIR / f"{method}_fp32_{sid}.npy"
            rand_files = sorted(_OUT_DIR.glob(f"random_{sid}_*.npy"))
            if not fp32_path.exists() or not rand_files:
                continue
            words   = text.split()
            fp32_s  = np.load(fp32_path).astype(np.float64)
            L       = min(len(words), len(fp32_s))
            fp32_l  = fp32_s[:L].tolist()
            words_l = words[:L]
            rho_vals, j5_vals = [], []
            for rf in rand_files:
                rand_s = np.load(rf).astype(np.float64)
                Lr     = min(L, len(rand_s))
                rand_l = rand_s[:Lr].tolist()
                rho, _ = spearman_rank_correlation(words_l, fp32_l, words_l[:Lr], rand_l)
                j5     = top_k_jaccard(words_l, fp32_l, words_l[:Lr], rand_l, k=5)
                rho_vals.append(rho)
                j5_vals.append(j5)
            floor_rows.append({
                "method":         method,
                "sample_id":      sid,
                "floor_mean_rho": round(float(np.nanmean(rho_vals)), 4),
                "floor_mean_j5":  round(float(np.nanmean(j5_vals)),  4),
            })

    df_floor = pd.DataFrame(floor_rows)
    floor_path = _RES_DIR / "stability_random_floor.csv"
    df_floor.to_csv(floor_path, index=False, encoding="utf-8")
    print(f"  Saved random floor -> {floor_path}")

    _DISPLAY = {"fp32_control": "FP32-Control"}
    print(f"\n  {'method':5s}  {'variant':22s}  {'rho':>6s} [95% CI]        "
          f"{'J@5':>5s}  {'r':>5s}  sig?")
    for _, row in df_sum.sort_values(["method", "mean_rho"]).iterrows():
        sig   = "*" if row["significant_bonferroni"] else " "
        label = _DISPLAY.get(row["variant"], row["variant"])
        print(f"  {row['method']:5s}  {label:22s}  "
              f"{row['mean_rho']:6.3f} [{row['ci95_lo_rho']:.3f},{row['ci95_hi_rho']:.3f}]  "
              f"{row['mean_j5']:5.3f}  {row['effect_size_r']:5.3f}  {sig}")

    stability_v2: dict = {
        "_meta": {
            "n_samples":             len(samples),
            "n_cross_seed_samples":  len(cross_seed_samples),
            "bootstrap_n_resamples": 2000,
            "bonferroni_method":     "statsmodels.multipletests",
        },
        "stability_per_variant": {},
        "ptq_vs_qat_wilcoxon":   {},
        "cross_seed":            {},
    }

    for _, row in df_sum.iterrows():
        variant = str(row["variant"])
        method  = str(row["method"])
        stability_v2["stability_per_variant"].setdefault(variant, {})[method] = {
            "n":                      int(row["n"]),
            "mean_rho":               float(row["mean_rho"]),
            "std_rho":                float(row["std_rho"]),
            "ci95_lo_rho":            float(row["ci95_lo_rho"]),
            "ci95_hi_rho":            float(row["ci95_hi_rho"]),
            "mean_j5":                float(row["mean_j5"]),
            "ci95_lo_j5":             float(row["ci95_lo_j5"]),
            "ci95_hi_j5":             float(row["ci95_hi_j5"]),
            "wilcoxon_p":             float(row["wilcoxon_p"]),
            "p_bonferroni":           float(row["p_bonferroni"]),
            "significant_bonferroni": bool(row["significant_bonferroni"]),
            "effect_size_r":          float(row["effect_size_r"]),
        }

    for _, row in df_pq.iterrows():
        key    = f"{row['ptq_variant']}_vs_{row['qat_variant']}"
        method = str(row["method"])
        stability_v2["ptq_vs_qat_wilcoxon"].setdefault(key, {})[method] = {
            "n_paired":      int(row["n_paired"]),
            "wilcoxon_stat": float(row["wilcoxon_stat"]),
            "wilcoxon_p":    float(row["wilcoxon_p"]),
            "p_bonferroni":  float(row["p_bonferroni"]),
            "significant":   bool(row["significant"]),
        }

    for _, row in df_cross.iterrows():
        key    = f"seed{int(row['seed_a'])}_vs_seed{int(row['seed_b'])}"
        method = str(row["method"])
        stability_v2["cross_seed"].setdefault(key, {})[method] = {
            "n":           int(row["n"]),
            "mean_rho":    None if np.isnan(row["mean_rho"])    else float(row["mean_rho"]),
            "ci95_lo_rho": None if np.isnan(row["ci95_lo_rho"]) else float(row["ci95_lo_rho"]),
            "ci95_hi_rho": None if np.isnan(row["ci95_hi_rho"]) else float(row["ci95_hi_rho"]),
        }

    json_v2_path = _RES_DIR / "stability_results_v2.json"
    with open(json_v2_path, "w", encoding="utf-8") as _f:
        _json.dump(stability_v2, _f, indent=2)
    print(f"\n  stability_results_v2.json saved -> {json_v2_path}")

    stability_json: dict = {}
    for _, row in df_sum.iterrows():
        variant = str(row["variant"])
        method  = str(row["method"])
        stability_json.setdefault(variant, {})[method] = {
            "n":                      int(row["n"]),
            "mean_rho":               float(row["mean_rho"]),
            "std_rho":                float(row["std_rho"]),
            "ci95_lo_rho":            float(row["ci95_lo_rho"]),
            "ci95_hi_rho":            float(row["ci95_hi_rho"]),
            "mean_j5":                float(row["mean_j5"]),
            "ci95_lo_j5":             float(row["ci95_lo_j5"]),
            "ci95_hi_j5":             float(row["ci95_hi_j5"]),
            "wilcoxon_p":             float(row["wilcoxon_p"]),
            "significant_bonferroni": bool(row["significant_bonferroni"]),
            "effect_size_r":          float(row["effect_size_r"]),
        }

    json_path = _RES_DIR / "stability_results.json"
    with open(json_path, "w", encoding="utf-8") as _f:
        _json.dump(stability_json, _f, indent=2)
    print(f"  Stability results JSON saved -> {json_path}")

    if "fp32_control" in stability_json:
        ctrl = stability_json["fp32_control"]
        mean_rhos = [v["mean_rho"] for v in ctrl.values()]
        print(f"  [FP32-Control] mean rho across methods: "
              f"{float(np.mean(mean_rhos)):.4f}  (written under key 'fp32_control')")
    else:
        print(
            "  [FP32-Control] No attribution files found "
            f"(expected: results/attributions/{{method}}_fp32_control_{{sid}}.npy). "
            "Run the XAI pipeline on fp32_control_seed{{seed}} checkpoints first."
        )

    run_full_stability_stats()
    compute_stability_by_family()
    compute_power_analysis()
    decompose_qat_drift()


def _attribution_filename(method: str, variant: str, sid: int, seed: int) -> str:
    sfx = "" if seed == 42 else f"_s{seed}"
    return f"{method}_{variant}{sfx}_{sid}.npy"


def _generate_per_seed_attributions(
    seed: int,
    samples: List[Tuple[int, str]],
    methods: Tuple[str, ...],
    variants: Tuple[str, ...],
    out_dir: Path,
) -> dict:
    import torch
    from src.config import LABELS, SEEDED_MODEL_DIRS
    from src.models import ModelManager
    from src.models.base import BaseModel, OnnxBaseModel
    from src.quantization.ptq import PTQQuantizer
    from src.xai.lime_explainer import LIMEExplainer
    from src.xai.shap_explainer import SHAPExplainer
    from src.xai.occlusion import OcclusionExplainer

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _MODELS_DIR = _PROJECT_ROOT / "models"

    fp32_dir = SEEDED_MODEL_DIRS.get(seed)
    if fp32_dir is None or not Path(fp32_dir).exists():
        raise FileNotFoundError(
            f"FP32 model for seed {seed} not found at {fp32_dir}. "
            "Run [5] Finetune first."
        )

    print(f"\n  [seed={seed}]  Loading FP32 base: {fp32_dir}")
    fp32_base = ModelManager.load_model(str(fp32_dir))
    fp32_base.model.eval()

    qat_clean_dir = _MODELS_DIR / f"qat_seed{seed}_clean"

    def _load_onnx(precision):
        import onnxruntime as ort
        onnx_dir = _MODELS_DIR / f"qat_onnx_{precision}_seed{seed}"
        onnx_file = onnx_dir / f"model_qat_{precision}.onnx"
        if not onnx_file.exists():
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxBaseModel(session, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        device = torch.device("cpu") if precision == "int8" else fp32_base.device
        return BaseModel(m, fp32_base.tokenizer, device=device)

    variant_builders = {
        "fp32":          (lambda: fp32_base, False),
        "ptq_fp16":      (lambda: _build_ptq("fp16"), True),
        "ptq_int8":      (lambda: _build_ptq("int8"), False),
        "ptq_int4":      (lambda: _build_ptq("int4"), False),
        "qat_fp32":      (lambda: (ModelManager.load_model(str(qat_clean_dir))
                                    if qat_clean_dir.exists() else None), False),
        "qat_onnx_fp16": (lambda: _load_onnx("fp16"), True),
        "qat_onnx_int8": (lambda: _load_onnx("int8"), False),
        "qat_onnx_int4": (lambda: _load_onnx("int4"), False),
    }

    stats = {"saved": 0, "skipped": 0, "missing_model": 0, "errors": 0}

    for vname in ("fp32",) + tuple(variants):
        if vname not in variant_builders:
            print(f"  [seed={seed}]  [SKIP] unknown variant {vname}")
            continue

        all_done = all(
            (out_dir / _attribution_filename(m, vname, sid, seed)).exists()
            for m in methods for sid, _ in samples
        )
        if all_done:
            print(f"  [seed={seed}]  [{vname.upper()}]  all attributions present, skipping load")
            stats["skipped"] += len(methods) * len(samples)
            continue

        loader, use_fp16 = variant_builders[vname]
        print(f"  [seed={seed}]  [{vname.upper()}]  loading model...")
        model = loader()
        if model is None:
            print(f"  [seed={seed}]  [SKIP] {vname}: model not found")
            stats["missing_model"] += len(methods) * len(samples)
            continue

        explainers = {}
        if "lime" in methods:
            explainers["lime"] = LIMEExplainer(model, LABELS, use_fp16=use_fp16, random_state=42)
        if "shap" in methods:
            explainers["shap"] = SHAPExplainer(model, LABELS, use_fp16=use_fp16)
        if "occ" in methods:
            explainers["occ"] = OcclusionExplainer(model, LABELS, use_fp16=use_fp16)

        for method in methods:
            explainer = explainers.get(method)
            if explainer is None:
                continue
            n_method_saved = 0
            n_method_skip  = 0
            for sid, text in samples:
                out_path = out_dir / _attribution_filename(method, vname, sid, seed)
                if out_path.exists():
                    n_method_skip += 1
                    stats["skipped"] += 1
                    continue
                try:
                    if method == "lime":
                        n_words = max(len(text.split()), 10)
                        exp = explainer.explain(text, num_features=n_words, num_samples=1000)
                        pred_idx = int(np.argmax(exp.predict_proba))
                        scores_dict = dict(exp.as_list(label=pred_idx))
                        word_scores = np.array(
                            [scores_dict.get(w, 0.0) for w in text.split()],
                            dtype=np.float32,
                        )
                    elif method == "shap":
                        sv = explainer.explain(text, max_evals=500)
                        pred = int(np.argmax(explainer.predict_proba(text)))
                        sd = {}
                        for j, tok in enumerate(sv[0].data):
                            if isinstance(tok, str) and tok.strip():
                                sd[tok.strip()] = float(sv[0].values[j][pred])
                        word_scores = np.array(
                            [sd.get(w, 0.0) for w in text.split()],
                            dtype=np.float32,
                        )
                    else:  # occ
                        r = explainer.explain(text, window_size=1)
                        word_scores = np.array(
                            [s for _, s in r["all_tokens_ordered"]],
                            dtype=np.float32,
                        )
                    np.save(out_path, word_scores)
                    n_method_saved += 1
                    stats["saved"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    err_log = out_dir / f"large_sample_errors_seed{seed}.log"
                    with open(err_log, "a", encoding="utf-8") as _log:
                        _log.write(f"{method}\t{vname}\t{sid}\t{type(exc).__name__}: {exc}\n")
            print(f"  [seed={seed}]  [{vname.upper()}]  {method:4s}: saved={n_method_saved}  skipped={n_method_skip}")

    return stats


def _compute_per_seed_stability(
    seed: int,
    samples: List[Tuple[int, str]],
    methods: Tuple[str, ...],
    variants: Tuple[str, ...],
    out_dir: Path,
) -> "pd.DataFrame":
    import pandas as pd

    rows: list = []
    for method in methods:
        for vname in variants:
            for sid, text in samples:
                fp32_path = out_dir / _attribution_filename(method, "fp32", sid, seed)
                var_path  = out_dir / _attribution_filename(method, vname,  sid, seed)
                if not fp32_path.exists() or not var_path.exists():
                    continue
                words = text.split()
                fp32_s = np.load(fp32_path).astype(np.float64)
                var_s  = np.load(var_path).astype(np.float64)
                L = min(len(words), len(fp32_s), len(var_s))
                if L < 3:
                    continue
                words_l = words[:L]
                rho, _ = spearman_rank_correlation(words_l, fp32_s[:L].tolist(),
                                                    words_l, var_s[:L].tolist())
                if np.isnan(rho):
                    continue
                j3  = top_k_jaccard(words_l, fp32_s[:L].tolist(), words_l, var_s[:L].tolist(), k=3)
                j5  = top_k_jaccard(words_l, fp32_s[:L].tolist(), words_l, var_s[:L].tolist(), k=5)
                j10 = top_k_jaccard(words_l, fp32_s[:L].tolist(), words_l, var_s[:L].tolist(), k=10)
                rows.append({
                    "seed": seed, "method": method, "variant": vname,
                    "sample_id": sid, "spearman_rho": rho,
                    "jaccard_k3": j3, "jaccard_k5": j5, "jaccard_k10": j10,
                })
    return pd.DataFrame(rows)


def run_large_sample_cross_seed_stability(
    n_total: Optional[int] = None,
    min_per_class: Optional[int] = None,
    methods: Tuple[str, ...] = ("lime", "occ", "shap"),
    variants: Tuple[str, ...] = (
        "ptq_fp16", "ptq_int8", "ptq_int4",
        "qat_fp32", "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4",
    ),
) -> Optional[dict]:
    import json as _json
    import pandas as pd
    from scipy.stats import wilcoxon as scipy_wilcoxon
    from statsmodels.stats.multitest import multipletests
    from src.config import (
        TRAINING_SEEDS,
        LARGE_N_STABILITY_SAMPLES,
        LARGE_N_STABILITY_MIN_PER_CLASS,
    )
    from src.utils.seed_aggregation import (
        aggregate_seed_results,
        save_aggregated_results,
    )
    from src.utils.stats_utils import rank_biserial_one_sample
    from src.visualization.reports import render_large_sample_stability

    if n_total is None:
        n_total = LARGE_N_STABILITY_SAMPLES
    if min_per_class is None:
        min_per_class = LARGE_N_STABILITY_MIN_PER_CLASS

    if len(TRAINING_SEEDS) < 3:
        warnings.warn(
            f"TRAINING_SEEDS has {len(TRAINING_SEEDS)} seeds; cross-seed analysis "
            "requires at least 3. Proceeding anyway.",
            UserWarning, stacklevel=2,
        )

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _DATA_DIR = _PROJECT_ROOT / "data" / "processed"
    _TEST_CSV = _DATA_DIR / "smsa_test_v2.csv"
    _VAL_CSV  = _DATA_DIR / "smsa_val_v2.csv"
    _OUT_DIR  = _PROJECT_ROOT / "results" / "attributions"
    _ARTIFACT_DIR = _PROJECT_ROOT / "outputs" / "multi-seed" / "large-sample-stability"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    samples = _build_stratified_subsample(
        _TEST_CSV, _VAL_CSV,
        n_total=n_total, min_per_class=min_per_class, seed=42,
    )
    if not samples:
        print(f"  [ERROR] Could not build stratified subsample (n_total={n_total})")
        return None

    df_test = pd.read_csv(_TEST_CSV) if _TEST_CSV.exists() else pd.DataFrame()
    df_val  = pd.read_csv(_VAL_CSV)  if _VAL_CSV.exists()  else pd.DataFrame()
    label_lookup = {}
    if not df_test.empty:
        for idx, row in df_test.iterrows():
            label_lookup[int(idx)] = int(row["label"])
        n_test = len(df_test)
        if not df_val.empty:
            for idx, row in df_val.iterrows():
                label_lookup[int(idx) + n_test] = int(row["label"])
    class_counts = {0: 0, 1: 0, 2: 0}
    for sid, _ in samples:
        lab = label_lookup.get(sid)
        if lab is not None:
            class_counts[lab] = class_counts.get(lab, 0) + 1
    print(f"\n  Large-sample cross-seed stability")
    print(f"  =================================")
    print(f"  n_total={len(samples)}  min_per_class={min_per_class}")
    print(f"  class counts: POS={class_counts.get(0,0)}  NEU={class_counts.get(1,0)}  NEG={class_counts.get(2,0)}")
    print(f"  seeds: {list(TRAINING_SEEDS)}")
    print(f"  methods: {list(methods)}")
    print(f"  variants: {list(variants)}")
    print(f"  artifact dir: {_ARTIFACT_DIR}")

    if class_counts:
        cnts = [v for v in class_counts.values() if v > 0]
        if cnts and (max(cnts) - min(cnts)) > 0.10 * sum(cnts):
            warnings.warn(
                f"Class counts deviate >10%: {class_counts}. "
                "Sample is not perfectly balanced.",
                UserWarning, stacklevel=2,
            )

    per_seed_dfs: dict = {}
    for seed in TRAINING_SEEDS:
        print(f"\n  ---- seed={seed} ----")
        try:
            _generate_per_seed_attributions(
                seed, samples, methods, variants, _OUT_DIR,
            )
        except FileNotFoundError as exc:
            print(f"  [SKIP seed={seed}] {exc}")
            continue
        df_seed = _compute_per_seed_stability(
            seed, samples, methods, variants, _OUT_DIR,
        )
        if df_seed.empty:
            print(f"  [WARN seed={seed}] No stability rows computed.")
            continue
        per_seed_csv = _ARTIFACT_DIR / f"stability_perSample_seed{seed}.csv"
        df_seed.to_csv(per_seed_csv, index=False, encoding="utf-8")
        print(f"  [seed={seed}] saved {len(df_seed)} rows -> {per_seed_csv.name}")
        per_seed_dfs[seed] = df_seed

    if not per_seed_dfs:
        print("  [ERROR] No per-seed stability data computed across any seed.")
        return None

    per_seed_summaries: dict = {}
    seed_results_for_agg: list = []

    for seed, df_seed in per_seed_dfs.items():
        seed_sum_rows = []
        scalar_metrics: dict = {"seed": seed}
        for (method, vname), grp in df_seed.groupby(["method", "variant"]):
            rhos = grp["spearman_rho"].dropna().tolist()
            j5s  = grp["jaccard_k5"].dropna().tolist()
            if len(rhos) < 2:
                continue
            rho_lo, rho_hi = _scipy_bootstrap_ci(rhos, n_resamples=2000)
            j5_lo,  j5_hi  = _scipy_bootstrap_ci(j5s,  n_resamples=2000)
            diffs = [r - 1.0 for r in rhos]
            try:
                w_stat, w_p = scipy_wilcoxon(diffs, alternative="less")
            except ValueError:
                w_stat = w_p = float("nan")
            eff_r = rank_biserial_one_sample(rhos, mu=1.0)
            seed_sum_rows.append({
                "seed": seed, "method": method, "variant": vname, "n": len(rhos),
                "mean_rho": round(float(np.mean(rhos)), 4),
                "std_rho":  round(float(np.std(rhos)), 4),
                "ci95_lo_rho": round(rho_lo, 4),
                "ci95_hi_rho": round(rho_hi, 4),
                "mean_j5":     round(float(np.mean(j5s)), 4),
                "ci95_lo_j5":  round(j5_lo, 4),
                "ci95_hi_j5":  round(j5_hi, 4),
                "wilcoxon_stat": round(float(w_stat), 4),
                "wilcoxon_p":    round(float(w_p), 6),
                "effect_r":      eff_r,
            })
            scalar_metrics[f"{method}__{vname}__mean_rho"] = float(np.mean(rhos))
            scalar_metrics[f"{method}__{vname}__mean_j5"]  = float(np.mean(j5s))

        df_sum = pd.DataFrame(seed_sum_rows)
        if not df_sum.empty:
            n_comp = len(df_sum)
            ps = df_sum["wilcoxon_p"].fillna(1.0).tolist()
            _, p_bonf, _, _ = multipletests(ps, alpha=0.05, method="bonferroni")
            df_sum["p_bonferroni"] = [round(float(p), 6) for p in p_bonf]
            df_sum["significant_bonferroni"] = df_sum["p_bonferroni"] < 0.05
            df_sum["bonferroni_alpha"] = round(0.05 / max(1, n_comp), 6)
            sum_csv = _ARTIFACT_DIR / f"stability_summary_seed{seed}.csv"
            df_sum.to_csv(sum_csv, index=False, encoding="utf-8")
            print(f"  [seed={seed}] summary ({n_comp} comparisons) -> {sum_csv.name}")
        per_seed_summaries[seed] = df_sum
        seed_results_for_agg.append(scalar_metrics)

    aggregated = aggregate_seed_results(seed_results_for_agg) if len(seed_results_for_agg) >= 2 else None

    all_per_sample = pd.concat(per_seed_dfs.values(), ignore_index=True)
    pooled_path = _ARTIFACT_DIR / "stability_perSample_all_seeds.csv"
    all_per_sample.to_csv(pooled_path, index=False, encoding="utf-8")
    print(f"\n  pooled per-sample rows -> {pooled_path.name}  (n={len(all_per_sample)})")

    aggregate_rows = []
    bonf_n_comparisons = 0
    for (method, vname), grp in all_per_sample.groupby(["method", "variant"]):
        per_seed_means = []
        for seed, df_seed in per_seed_dfs.items():
            sub = df_seed[(df_seed["method"] == method) & (df_seed["variant"] == vname)]
            rhos_seed = sub["spearman_rho"].dropna().tolist()
            if rhos_seed:
                per_seed_means.append(float(np.mean(rhos_seed)))
        if len(per_seed_means) < 2:
            continue
        bonf_n_comparisons += 1
        rhos_all = grp["spearman_rho"].dropna().tolist()
        j5s_all  = grp["jaccard_k5"].dropna().tolist()
        rho_lo, rho_hi = _scipy_bootstrap_ci(rhos_all, n_resamples=2000)
        j5_lo,  j5_hi  = _scipy_bootstrap_ci(j5s_all,  n_resamples=2000)
        try:
            w_stat, w_p = scipy_wilcoxon([r - 1.0 for r in rhos_all], alternative="less")
        except ValueError:
            w_stat = w_p = float("nan")
        eff_r = rank_biserial_one_sample(rhos_all, mu=1.0)
        aggregate_rows.append({
            "method": method, "variant": vname,
            "n_seeds": len(per_seed_means),
            "n_samples_total": len(rhos_all),
            "mean_rho_across_seeds":  round(float(np.mean(per_seed_means)), 4),
            "std_rho_across_seeds":   round(float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0, 4),
            "pooled_mean_rho":        round(float(np.mean(rhos_all)), 4),
            "pooled_ci95_lo_rho":     round(rho_lo, 4),
            "pooled_ci95_hi_rho":     round(rho_hi, 4),
            "pooled_mean_j5":         round(float(np.mean(j5s_all)), 4),
            "pooled_ci95_lo_j5":      round(j5_lo, 4),
            "pooled_ci95_hi_j5":      round(j5_hi, 4),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_p":    round(float(w_p), 6),
            "effect_r":      eff_r,
        })

    df_agg = pd.DataFrame(aggregate_rows)
    if not df_agg.empty:
        ps = df_agg["wilcoxon_p"].fillna(1.0).tolist()
        _, p_bonf, _, _ = multipletests(ps, alpha=0.05, method="bonferroni")
        df_agg["p_bonferroni"] = [round(float(p), 6) for p in p_bonf]
        df_agg["significant_bonferroni"] = df_agg["p_bonferroni"] < 0.05
        df_agg["bonferroni_alpha"] = round(0.05 / max(1, bonf_n_comparisons), 6)
        agg_csv = _ARTIFACT_DIR / "stability_aggregate_acrossSeeds.csv"
        df_agg.to_csv(agg_csv, index=False, encoding="utf-8")
        print(f"  aggregate across seeds -> {agg_csv.name}")

    payload = {
        "_meta": {
            "n_total_samples_requested": int(n_total),
            "n_total_samples_used":      int(len(samples)),
            "min_per_class":             int(min_per_class),
            "class_counts": {LABELS_NAME(k): v for k, v in class_counts.items()},
            "seeds":   list(TRAINING_SEEDS),
            "methods": list(methods),
            "variants": list(variants),
            "bonferroni_family_size": int(bonf_n_comparisons),
            "bonferroni_alpha":       round(0.05 / max(1, bonf_n_comparisons), 6),
            "bootstrap_n_resamples":  2000,
        },
        "per_seed_summary": {
            int(s): df_sum.to_dict(orient="records")
            for s, df_sum in per_seed_summaries.items()
        },
        "aggregate_across_seeds": df_agg.to_dict(orient="records") if not df_agg.empty else [],
        "seed_aggregated_metrics": aggregated,
    }
    json_path = _ARTIFACT_DIR / "large_sample_stability.json"
    if aggregated is not None:
        save_aggregated_results(aggregated, _ARTIFACT_DIR / "seed_aggregated_metrics.json", exclude_raw=False)
    with open(json_path, "w", encoding="utf-8") as _f:
        _json.dump(payload, _f, indent=2, default=float)
    print(f"  payload -> {json_path.name}")

    if not df_agg.empty:
        render_large_sample_stability(
            json_path=json_path,
            csv_path=_ARTIFACT_DIR / "stability_aggregate_acrossSeeds.csv",
        )

    return payload


def LABELS_NAME(k: int) -> str:
    return {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}.get(int(k), str(k))


def decompose_qat_drift(
    per_sample_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Optional[dict]:
    import json as _json
    import pandas as pd
    from scipy.stats import wilcoxon as _sp_wilcoxon

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _RES_DIR = out_dir if out_dir is not None else (_PROJECT_ROOT / "results")
    per_path = per_sample_csv if per_sample_csv is not None else (_RES_DIR / "stability_perSample.csv")

    if not Path(per_path).exists():
        print(f"  [WARN] {per_path} not found — run Stability Analysis first.")
        return None

    df = pd.read_csv(per_path)
    needed = {"qat_fp32", "fp32_control"}
    have = set(df["variant"].unique().tolist())
    missing = needed - have
    if missing:
        print(
            f"  [WARN] Missing variants {sorted(missing)} in {per_path.name}. "
            "Run LIME/SHAP/Occlusion attributions on the fp32_control checkpoint, "
            "then re-run Stability Analysis."
        )
        return None

    methods = sorted(df["method"].unique().tolist())
    rows: list = []

    for method in methods:
        ctrl = df[(df["method"] == method) & (df["variant"] == "fp32_control")]
        qat  = df[(df["method"] == method) & (df["variant"] == "qat_fp32")]
        merged = ctrl.merge(qat, on="sample_id", suffixes=("_ctrl", "_qat"))
        if merged.empty:
            continue
        merged = merged.dropna(subset=["spearman_rho_ctrl", "spearman_rho_qat"])
        if len(merged) < 2:
            continue

        rho_ctrl = merged["spearman_rho_ctrl"].astype(float).to_numpy()
        rho_qat  = merged["spearman_rho_qat"].astype(float).to_numpy()

        total_drift    = 1.0 - rho_qat
        training_drift = 1.0 - rho_ctrl
        fakequant_residual = rho_ctrl - rho_qat

        total_mean    = float(np.mean(total_drift))
        train_mean    = float(np.mean(training_drift))
        residual_mean = float(np.mean(fakequant_residual))

        denom = total_mean if abs(total_mean) > 1e-12 else float("nan")
        train_share    = float(train_mean    / denom) if not np.isnan(denom) else float("nan")
        residual_share = float(residual_mean / denom) if not np.isnan(denom) else float("nan")

        total_lo,    total_hi    = _scipy_bootstrap_ci(total_drift.tolist(),       n_resamples=2000)
        train_lo,    train_hi    = _scipy_bootstrap_ci(training_drift.tolist(),    n_resamples=2000)
        residual_lo, residual_hi = _scipy_bootstrap_ci(fakequant_residual.tolist(), n_resamples=2000)

        try:
            w_stat, w_p = _sp_wilcoxon(fakequant_residual)
            w_stat = float(w_stat)
            w_p    = float(w_p)
        except ValueError:
            w_stat = float("nan")
            w_p    = float("nan")

        rows.append({
            "method":               method,
            "n_paired":             int(len(merged)),
            "rho_fp32_control":     round(float(np.mean(rho_ctrl)), 4),
            "rho_qat_fp32":         round(float(np.mean(rho_qat)),  4),
            "total_drift":          round(total_mean,    4),
            "total_ci95_lo":        round(total_lo,      4),
            "total_ci95_hi":        round(total_hi,      4),
            "training_component":   round(train_mean,    4),
            "training_ci95_lo":     round(train_lo,      4),
            "training_ci95_hi":     round(train_hi,      4),
            "fakequant_residual":   round(residual_mean, 4),
            "fakequant_ci95_lo":    round(residual_lo,   4),
            "fakequant_ci95_hi":    round(residual_hi,   4),
            "training_share":       round(train_share,    4),
            "fakequant_share":      round(residual_share, 4),
            "residual_wilcoxon_stat": round(w_stat, 4),
            "residual_wilcoxon_p":    round(w_p,    6),
        })

    if not rows:
        print("  [WARN] No paired (fp32_control, qat_fp32) samples found.")
        return None

    df_dec = pd.DataFrame(rows)
    csv_path = _RES_DIR / "qat_drift_decomposition.csv"
    df_dec.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  QAT-FP32 drift decomposition saved -> {csv_path}")

    payload = {
        "_meta": {
            "definition": (
                "total_drift = 1 - rho(FP32 vs QAT-FP32);  "
                "training_component = 1 - rho(FP32 vs FP32-Control);  "
                "fakequant_residual = rho(FP32-Control) - rho(QAT-FP32);  "
                "total = training_component + fakequant_residual (per sample, then averaged)."
            ),
            "source": str(per_path),
        },
        "per_method": {r["method"]: {k: v for k, v in r.items() if k != "method"} for r in rows},
    }
    json_path = _RES_DIR / "qat_drift_decomposition.json"
    with open(json_path, "w", encoding="utf-8") as _f:
        _json.dump(payload, _f, indent=2)
    print(f"  QAT-FP32 drift decomposition JSON saved -> {json_path}")

    print(f"\n  {'method':5s}  {'rho_ctrl':>8s}  {'rho_qat':>8s}  "
          f"{'total':>7s}  {'train':>7s}  {'fq':>7s}  {'train%':>7s}  {'fq%':>7s}")
    for r in rows:
        print(f"  {r['method']:5s}  {r['rho_fp32_control']:8.3f}  {r['rho_qat_fp32']:8.3f}  "
              f"{r['total_drift']:7.3f}  {r['training_component']:7.3f}  "
              f"{r['fakequant_residual']:7.3f}  "
              f"{r['training_share']*100:6.1f}%  {r['fakequant_share']*100:6.1f}%")

    return payload

_VARIANT_DISPLAY = {
    "ptq_fp16":      "PTQ-FP16",
    "ptq_int8":      "PTQ-INT8",
    "ptq_int4":      "PTQ-INT4",
    "qat_fp32":      "QAT-FP32",
    "qat_onnx_fp16": "QAT-FP16",
    "qat_onnx_int8": "QAT-INT8",
    "qat_onnx_int4": "QAT-INT4",
    "fp32_control":  "FP32-Ctrl",
}

_PTQ_QAT_LABEL_PAIRS = [
    ("ptq_fp16",  "qat_onnx_fp16"),
    ("ptq_int8",  "qat_onnx_int8"),
    ("ptq_int4",  "qat_onnx_int4"),
]

FAMILY: dict = {
    "ig":   "gradient",
    "gxi":  "gradient",
    "lime": "perturbation",
    "occ":  "perturbation",
    "shap": "perturbation",
}

_METHOD_DISPLAY: dict = {
    "ig":   "IG",
    "gxi":  "GxI",
    "lime": "LIME",
    "occ":  "Occlusion",
    "shap": "SHAP",
}

QAT_ONNX_VARIANTS: frozenset = frozenset({
    "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4",
})

_VARIANTS_ORDER: tuple = (
    "ptq_fp16", "ptq_int8", "ptq_int4",
    "qat_fp32", "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4",
)

def compute_stability_by_family() -> None:
    import json as _json
    import shutil as _shutil
    import pandas as pd
    from scipy.stats import bootstrap as _scipy_bootstrap
    from scipy.stats import wilcoxon as _sp_wilcoxon
    from src.utils.stats_utils import bonferroni_correct

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _RES_DIR      = _PROJECT_ROOT / "results"

    per_path = _RES_DIR / "stability_perSample.csv"
    if not per_path.exists():
        print("  [WARN] stability_perSample.csv not found – run Stability Analysis first.")
        return

    df = pd.read_csv(per_path)
    df["family"] = df["method"].map(FAMILY)

    _FAMILIES = ("gradient", "perturbation")

    entries: dict       = {}
    raw_ps:  List[float] = []
    pv_keys: List[tuple]  = []

    for family in _FAMILIES:
        family_methods = sorted(m for m, f in FAMILY.items() if f == family)
        method_labels  = [_METHOD_DISPLAY.get(m, m) for m in family_methods]
        df_fam         = df[df["family"] == family]

        for vname in _VARIANTS_ORDER:
            if family == "gradient" and vname in QAT_ONNX_VARIANTS:
                entries[(vname, family)] = {
                    "rho": None, "ci_low": None, "ci_high": None,
                    "label":   "STE_PROXY_INVALID",
                    "methods": method_labels,
                }
                continue

            rhos = (df_fam[df_fam["variant"] == vname]["spearman_rho"]
                    .dropna().tolist())

            if len(rhos) < 2:
                entries[(vname, family)] = {
                    "rho": None, "ci_low": None, "ci_high": None,
                    "n":     len(rhos),
                    "label": "insufficient_data",
                    "methods": method_labels,
                }
                raw_ps.append(float("nan"))
                pv_keys.append((vname, family))
                continue

            ci_res = _scipy_bootstrap(
                (np.array(rhos, dtype=float),),
                np.mean,
                n_resamples=2000,
                confidence_level=0.95,
                method="percentile",
                random_state=42,
            )

            diffs = [r - 1.0 for r in rhos]
            try:
                w_stat, w_p = _sp_wilcoxon(diffs, alternative="less")
            except ValueError:
                w_stat = w_p = float("nan")

            entries[(vname, family)] = {
                "rho":     round(float(np.mean(rhos)), 4),
                "ci_low":  round(float(ci_res.confidence_interval.low),  4),
                "ci_high": round(float(ci_res.confidence_interval.high), 4),
                "n":       len(rhos),
                "wilcoxon_stat":  (round(float(w_stat), 4)
                                   if not np.isnan(w_stat) else None),
                "wilcoxon_p_raw": (round(float(w_p), 6)
                                   if not np.isnan(w_p) else None),
                "methods": method_labels,
            }
            raw_ps.append(float(w_p) if not np.isnan(w_p) else float("nan"))
            pv_keys.append((vname, family))

    bonf_ps = bonferroni_correct(raw_ps)
    for (vname, family), p_b in zip(pv_keys, bonf_ps):
        e = entries.get((vname, family))
        if e is None or e.get("rho") is None:
            continue
        e["wilcoxon_p_bonferroni"] = (round(float(p_b), 6)
                                       if not np.isnan(p_b) else None)
        e["significant"] = (not np.isnan(p_b) and float(p_b) < 0.05)

    family_results: dict = {
        fam: {vn: entries[(vn, fam)] for vn in _VARIANTS_ORDER
              if (vn, fam) in entries}
        for fam in _FAMILIES
    }

    sum_path   = _RES_DIR / "stability_summary.csv"
    agg_table3: dict = {}
    if sum_path.exists():
        for _, row in pd.read_csv(sum_path).iterrows():
            vn = str(row["variant"])
            m  = str(row["method"])
            agg_table3.setdefault(vn, {})[m] = {
                "mean_rho":               float(row["mean_rho"]),
                "p_bonferroni":           float(row["p_bonferroni"]),
                "significant_bonferroni": bool(row["significant_bonferroni"]),
            }

    payload = {
        "gradient_family":    family_results["gradient"],
        "perturbation_family": family_results["perturbation"],
        "aggregate_table3":   agg_table3,
    }
    json_path = _RES_DIR / "stability_by_family.json"
    with open(json_path, "w", encoding="utf-8") as _f:
        _json.dump(payload, _f, indent=2)
    print(f"\n  stability_by_family.json -> {json_path}")

    def _csv_rows(family: str) -> list:
        rows = []
        for vn in _VARIANTS_ORDER:
            e     = family_results[family].get(vn, {})
            vlabel = _VARIANT_DISPLAY.get(vn, vn)
            lbl    = e.get("label", "")
            if lbl in ("STE_PROXY_INVALID", "insufficient_data") or e.get("rho") is None:
                rows.append({
                    "variant": vlabel,
                    "rho": "N/A", "ci_low": "N/A", "ci_high": "N/A",
                    "n": e.get("n", ""),
                    "methods": ",".join(e.get("methods", [])),
                    "significant_bonferroni": "",
                    "note": lbl,
                })
            else:
                rows.append({
                    "variant": vlabel,
                    "rho":     round(e["rho"],     4),
                    "ci_low":  round(e["ci_low"],  4),
                    "ci_high": round(e["ci_high"], 4),
                    "n":       e["n"],
                    "methods": ",".join(e.get("methods", [])),
                    "significant_bonferroni": bool(e.get("significant", False)),
                    "note": "",
                })
        return rows

    for family, fname in (("gradient", "table3a_gradient"),
                          ("perturbation", "table3b_perturbation")):
        csv_path = _RES_DIR / f"{fname}.csv"
        pd.DataFrame(_csv_rows(family)).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  {fname}.csv -> {csv_path}")

    if sum_path.exists():
        agg_path = _RES_DIR / "table3_aggregate_appendix.csv"
        _shutil.copy2(sum_path, agg_path)
        print(f"  table3_aggregate_appendix.csv -> {agg_path}")

    for family in _FAMILIES:
        _sample = next(
            (e for e in family_results[family].values() if e.get("methods")), {}
        )
        methods_str = ", ".join(_sample.get("methods", ["?"]))
        print(f"\n  {family.upper()} family (methods: {methods_str})")
        print(f"  {'variant':22s}  {'rho':>6s}  {'95% CI':^15s}  sig?")
        for vn in _VARIANTS_ORDER:
            e     = family_results[family].get(vn, {})
            vlab  = _VARIANT_DISPLAY.get(vn, vn)
            lbl   = e.get("label", "")
            if lbl == "STE_PROXY_INVALID":
                print(f"  {vlab:22s}  {'N/A':>6s}  {'':15s}  STE")
            elif e.get("rho") is None:
                print(f"  {vlab:22s}  {'N/A':>6s}  {'':15s}  {lbl}")
            else:
                ci_s = f"[{e['ci_low']:.3f},{e['ci_high']:.3f}]"
                sig  = "*" if e.get("significant") else " "
                print(f"  {vlab:22s}  {e['rho']:6.3f}  {ci_s:15s}  {sig}")

def run_full_stability_stats():
    import pandas as pd
    import json as _json
    from scipy.stats import bootstrap as _scipy_bootstrap
    from scipy.stats import wilcoxon as _sp_wilcoxon
    from src.utils.stats_utils import (
        bootstrap_spearman,
        wilcoxon_test,
        bonferroni_correct,
        cohens_d,
    )

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _RES_DIR = _PROJECT_ROOT / "results"
    per_path = _RES_DIR / "stability_perSample.csv"

    if not per_path.exists():
        print("  [WARN] stability_perSample.csv not found – run Stability Analysis first.")
        return

    df = pd.read_csv(per_path)

    VARIANTS = [
        "ptq_fp16", "ptq_int8", "ptq_int4",
        "qat_fp32", "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4",
    ]
    METHODS = ["lime", "occ", "shap"]

    per_variant: dict = {}
    pv_raw_ps: List[float] = []
    pv_keys:   List[tuple]  = []

    for vname in VARIANTS:
        per_variant[vname] = {}
        for method in METHODS:
            grp  = df[(df["variant"] == vname) & (df["method"] == method)]
            rhos = grp["spearman_rho"].dropna().tolist()

            if len(rhos) < 2:
                per_variant[vname][method] = None
                pv_raw_ps.append(float("nan"))
                pv_keys.append((vname, method))
                continue

            ci_res = _scipy_bootstrap(
                (np.array(rhos, dtype=float),),
                np.mean,
                n_resamples=2000,
                confidence_level=0.95,
                method="percentile",
                random_state=42,
            )
            ci_low  = float(ci_res.confidence_interval.low)
            ci_high = float(ci_res.confidence_interval.high)

            diffs = [r - 1.0 for r in rhos]
            try:
                w_stat, w_p = _sp_wilcoxon(diffs, alternative="less")
            except ValueError:
                w_stat = w_p = float("nan")

            per_variant[vname][method] = {
                "n":               len(rhos),
                "mean_rho":        round(float(np.mean(rhos)), 6),
                "ci_low":          round(ci_low,               6),
                "ci_high":         round(ci_high,              6),
                "wilcoxon_stat":   (round(float(w_stat), 4)
                                    if not np.isnan(w_stat) else None),
                "wilcoxon_p_raw":  (round(float(w_p), 6)
                                    if not np.isnan(w_p) else None),
            }
            pv_raw_ps.append(float(w_p) if not np.isnan(w_p) else float("nan"))
            pv_keys.append((vname, method))

    pv_bonf = bonferroni_correct(pv_raw_ps)
    for (vname, method), p_bonf in zip(pv_keys, pv_bonf):
        if per_variant[vname][method] is None:
            continue
        per_variant[vname][method]["wilcoxon_p_bonferroni"] = (
            round(p_bonf, 6) if not np.isnan(p_bonf) else None
        )
        per_variant[vname][method]["significant"] = (
            not np.isnan(p_bonf) and float(p_bonf) < 0.05
        )

    ptq_vs_qat: dict       = {}
    pq_raw_ps:  List[float] = []
    pq_keys:    List[tuple]  = []

    for ptq_v, qat_v in _PTQ_QAT_LABEL_PAIRS:
        pair_key = f"{ptq_v}_vs_{qat_v}"
        ptq_vs_qat[pair_key] = {}
        for method in METHODS:
            ptq_grp  = df[(df["variant"] == ptq_v) & (df["method"] == method)]
            qat_grp  = df[(df["variant"] == qat_v) & (df["method"] == method)]
            merged   = ptq_grp.merge(qat_grp, on="sample_id", suffixes=("_ptq", "_qat"))
            merged   = merged.dropna(subset=["spearman_rho_ptq", "spearman_rho_qat"])
            n_paired = int(len(merged))

            if n_paired < 10:
                ptq_vs_qat[pair_key][method] = None
                pq_raw_ps.append(float("nan"))
                pq_keys.append((pair_key, method))
                continue

            ptq_rhos = merged["spearman_rho_ptq"].tolist()
            qat_rhos = merged["spearman_rho_qat"].tolist()

            bs   = bootstrap_spearman(ptq_rhos, qat_rhos, n_resamples=2000)
            wt   = wilcoxon_test(ptq_rhos, qat_rhos)
            d    = cohens_d(ptq_rhos, qat_rhos)

            ptq_vs_qat[pair_key][method] = {
                "n_paired":               n_paired,
                "wilcoxon_stat":          wt["stat"],
                "wilcoxon_p_raw":         wt["p_value"],
                "cohens_d":               d,
                "spearman_rho_stability": bs["rho"],
                "rho_ci_low":             bs["ci_low"],
                "rho_ci_high":            bs["ci_high"],
            }
            pq_raw_ps.append(
                wt["p_value"] if not np.isnan(wt["p_value"]) else float("nan")
            )
            pq_keys.append((pair_key, method))

    pq_bonf = bonferroni_correct(pq_raw_ps)
    for (pair_key, method), p_bonf in zip(pq_keys, pq_bonf):
        if ptq_vs_qat[pair_key][method] is None:
            continue
        ptq_vs_qat[pair_key][method]["wilcoxon_p_bonferroni"] = (
            round(p_bonf, 6) if not np.isnan(p_bonf) else None
        )
        ptq_vs_qat[pair_key][method]["significant"] = (
            not np.isnan(p_bonf) and float(p_bonf) < 0.05
        )

    full_stats: dict = {
        "_meta": {
            "bootstrap_n_resamples":        2000,
            "bootstrap_method":             "percentile",
            "wilcoxon_per_variant":         "one-sided (H1: rho < 1.0)",
            "wilcoxon_ptq_vs_qat":          "two-sided",
            "bonferroni_scope_per_variant": len(pv_keys),
            "bonferroni_scope_ptq_vs_qat":  len(pq_keys),
        },
        "per_variant": per_variant,
        "ptq_vs_qat":  ptq_vs_qat,
    }

    json_path = _RES_DIR / "stability_full_stats.json"
    with open(json_path, "w", encoding="utf-8") as _f:
        _json.dump(full_stats, _f, indent=2)
    print(f"\n  stability_full_stats.json -> {json_path}")


    hdr = (f"\n  {'variant':22s}  {'method':5s}  {'rho':>6s}  "
           f"{'95% CI':^15s}  {'W':>8s}  {'p_Bonf':>9s}  sig?")
    print(hdr)
    for vname in VARIANTS:
        for method in METHODS:
            row = per_variant[vname].get(method)
            if row is None:
                continue
            p_b   = row.get("wilcoxon_p_bonferroni")
            sig   = "*" if (p_b is not None and not np.isnan(p_b) and p_b < 0.05) else " "
            pb_s  = f"{p_b:.4f}" if (p_b is not None and not np.isnan(p_b)) else "  n/a "
            w_s   = (f"{row['wilcoxon_stat']:.1f}"
                     if row.get("wilcoxon_stat") is not None else " n/a")
            ci_s  = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            print(f"  {vname:22s}  {method:5s}  "
                  f"{row['mean_rho']:6.3f}  {ci_s:15s}  "
                  f"{w_s:>8s}  {pb_s:>9s}  {sig}")

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
    n = 500
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

def probe_attribution_analysis():
    import pandas as pd
    import torch
    from src.models import ModelManager
    from src.models.base import BaseModel, OnnxBaseModel
    from src.quantization.ptq import PTQQuantizer
    from src.config import LABELS
    from src.xai.occlusion import OcclusionExplainer
    import onnxruntime as ort

    _ROOT = Path(__file__).resolve().parent.parent.parent
    _RES_DIR = _ROOT / "results"
    _MODELS_DIR = _ROOT / "models"
    _PRED_CSV = _RES_DIR / "probe_predictions_allseeds.csv"

    if not _PRED_CSV.exists():
        print(f"  [ERROR] {_PRED_CSV} not found — run stress test first.")
        return

    df = pd.read_csv(_PRED_CSV)
    df_fp32 = df[(df["precision"] == "fp32") & (df["seed"] == 42) &
                 (df["predicted"] == df["expected"])].copy()

    selected = (df_fp32.groupby("phenomenon", group_keys=False)
                .apply(lambda g: g.head(5))
                .head(40)
                .reset_index(drop=True))
    print(f"\n  Probe attribution: {len(selected)} probes × 3 variants")

    fp32_base = ModelManager.load_model(str(_MODELS_DIR / "fp32_seed42"))
    fp32_base.model.eval()

    def _build_ptq_int4():
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = ptq.quantize_int4()
        return BaseModel(m, fp32_base.tokenizer, device=torch.device("cpu"))

    def _load_qat_onnx_int4():
        onnx_file = _MODELS_DIR / "qat_onnx_int4_seed42" / "model_qat_int4.onnx"
        if not onnx_file.exists():
            return None
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        return OnnxBaseModel(sess, fp32_base.tokenizer, None, torch.device("cpu"))

    variants = [
        ("fp32_seed42",       fp32_base,          False),
        ("ptq_int4_seed42",   _build_ptq_int4(),  False),
        ("qat_onnx_int4_seed42", _load_qat_onnx_int4(), False),
    ]
    variants = [(n, m, f) for n, m, f in variants if m is not None]

    rows = []
    for _, probe_row in selected.iterrows():
        text = probe_row["text"]
        phenomenon = probe_row["phenomenon"]
        raw_tokens = probe_row.get("phenomenon_tokens", "[]")
        if isinstance(raw_tokens, str):
            import ast
            try:
                phenom_tokens = ast.literal_eval(raw_tokens)
            except Exception:
                phenom_tokens = [t.strip() for t in raw_tokens.strip("[]").split(",") if t.strip()]
        else:
            phenom_tokens = list(raw_tokens)

        for vname, model, use_fp16 in variants:
            explainer = OcclusionExplainer(model, LABELS, use_fp16=use_fp16)
            result = explainer.explain(text, window_size=1)
            ranked = result["token_importance"] 
            rank_map = {word: rank + 1 for rank, (word, _) in enumerate(ranked)}
            score_map = {word: score for word, score in ranked}

            for tok in phenom_tokens:
                rank = rank_map.get(tok)
                score = score_map.get(tok, float("nan"))
                rows.append({
                    "phenomenon": phenomenon,
                    "variant": vname,
                    "token": tok,
                    "text": text,
                    "rank": rank,
                    "score": score,
                    "in_top5": (rank is not None and rank <= 5),
                })

    df_out = pd.DataFrame(rows)
    out_path = _RES_DIR / "probe_attribution_analysis.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Saved {len(df_out)} rows -> {out_path}")

    print(f"\n  {'phenomenon':20s}  {'variant':25s}  top5_frac")
    for (phenom, vname), grp in df_out.groupby(["phenomenon", "variant"]):
        frac = grp["in_top5"].mean()
        print(f"  {phenom:20s}  {vname:25s}  {frac:.2f}")