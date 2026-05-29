import json
from pathlib import Path

import pandas as pd


def render_qat_drift_decomposition(
    json_path=None,
    csv_path=None,
):
    if json_path is None or csv_path is None:
        _project_root = Path(__file__).resolve().parent.parent.parent
        _res_dir = _project_root / "results"
        if json_path is None:
            json_path = _res_dir / "qat_drift_decomposition.json"
        if csv_path is None:
            csv_path = _res_dir / "qat_drift_decomposition.csv"

    json_path = Path(json_path)
    csv_path  = Path(csv_path)

    if not json_path.exists() or not csv_path.exists():
        print(
            f"  [WARN] Decomposition artifacts missing "
            f"(expected {json_path.name} and {csv_path.name}). "
            "Run Stability Analysis first (Option [3] -> Stability Analysis)."
        )
        return None

    with open(json_path, "r", encoding="utf-8") as _f:
        payload = json.load(_f)
    df = pd.read_csv(csv_path)

    print("\n  QAT-FP32 Drift Decomposition (per attribution method)")
    print("  " + "-" * 96)
    print(f"  {payload.get('_meta', {}).get('definition', '')}")
    print("  " + "-" * 96)
    print(
        f"  {'method':5s}  {'n':>4s}  "
        f"{'rho(Ctrl)':>9s}  {'rho(QAT)':>8s}  "
        f"{'Total':>16s}  {'Training':>16s}  {'FakeQuant':>16s}  "
        f"{'Train%':>6s}  {'FQ%':>6s}  {'p(FQ)':>8s}"
    )
    for _, row in df.iterrows():
        total_str = f"{row['total_drift']:.3f} [{row['total_ci95_lo']:.3f},{row['total_ci95_hi']:.3f}]"
        train_str = f"{row['training_component']:.3f} [{row['training_ci95_lo']:.3f},{row['training_ci95_hi']:.3f}]"
        fq_str    = f"{row['fakequant_residual']:.3f} [{row['fakequant_ci95_lo']:.3f},{row['fakequant_ci95_hi']:.3f}]"
        print(
            f"  {row['method']:5s}  {int(row['n_paired']):>4d}  "
            f"{row['rho_fp32_control']:9.3f}  {row['rho_qat_fp32']:8.3f}  "
            f"{total_str:>16s}  {train_str:>16s}  {fq_str:>16s}  "
            f"{row['training_share']*100:6.1f}  {row['fakequant_share']*100:6.1f}  "
            f"{row['residual_wilcoxon_p']:8.4f}"
        )
    print("  " + "-" * 96)
    print("  Total = 1 - rho(FP32, QAT-FP32). Training = 1 - rho(FP32, FP32-Ctrl).")
    print("  FakeQuant = rho(FP32-Ctrl) - rho(QAT-FP32). Shares are means of components / mean total.")
    print(f"  Source: {payload.get('_meta', {}).get('source', csv_path)}")

    return df


def render_large_sample_stability(
    json_path=None,
    csv_path=None,
):
    if json_path is None or csv_path is None:
        _project_root = Path(__file__).resolve().parent.parent.parent
        _res_dir = _project_root / "outputs" / "multi-seed" / "large-sample-stability"
        if json_path is None:
            json_path = _res_dir / "large_sample_stability.json"
        if csv_path is None:
            csv_path = _res_dir / "stability_aggregate_acrossSeeds.csv"

    json_path = Path(json_path)
    csv_path  = Path(csv_path)

    if not csv_path.exists():
        print(
            f"  [WARN] Large-sample stability aggregate not found "
            f"(expected {csv_path.name}). "
            "Run XAI -> Large-sample cross-seed stability first."
        )
        return None

    df = pd.read_csv(csv_path)
    meta = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as _f:
            meta = json.load(_f).get("_meta", {})

    bonf_family = meta.get("bonferroni_family_size", "?")
    bonf_alpha  = meta.get("bonferroni_alpha", float("nan"))

    print("\n  Large-sample cross-seed stability (per attribution method × variant)")
    print("  " + "-" * 118)
    print(
        f"  Bonferroni family size = {bonf_family}, alpha_corrected = {bonf_alpha} "
        f"(family-wise alpha = 0.05). Wilcoxon H1: rho < 1.0 (one-sided)."
    )
    print("  " + "-" * 118)
    print(
        f"  {'method':5s}  {'variant':15s}  {'k':>3s}  {'n':>5s}  "
        f"{'mean±std':>14s}  {'95% CI':>17s}  "
        f"{'J@5':>5s}  {'W':>8s}  {'p_raw':>8s}  {'p_Bonf':>8s}  "
        f"{'effect_r':>9s}  sig?"
    )
    for _, row in df.sort_values(["method", "variant"]).iterrows():
        mean_std = f"{row['mean_rho_across_seeds']:.3f}±{row['std_rho_across_seeds']:.3f}"
        ci_str   = f"[{row['pooled_ci95_lo_rho']:.3f},{row['pooled_ci95_hi_rho']:.3f}]"
        w_val    = row.get("wilcoxon_stat", float("nan"))
        p_raw    = row.get("wilcoxon_p", float("nan"))
        p_bonf   = row.get("p_bonferroni", float("nan"))
        eff_r    = row.get("effect_r", float("nan"))
        sig      = "*" if bool(row.get("significant_bonferroni", False)) else " "
        w_s      = f"{w_val:8.1f}" if pd.notna(w_val) else "     n/a"
        praw_s   = f"{p_raw:8.4f}" if pd.notna(p_raw) else "     n/a"
        pbonf_s  = f"{p_bonf:8.4f}" if pd.notna(p_bonf) else "     n/a"
        effr_s   = f"{eff_r:9.3f}" if pd.notna(eff_r) else "      n/a"
        print(
            f"  {row['method']:5s}  {str(row['variant']):15s}  "
            f"{int(row['n_seeds']):>3d}  {int(row['n_samples_total']):>5d}  "
            f"{mean_std:>14s}  {ci_str:>17s}  "
            f"{row['pooled_mean_j5']:5.3f}  {w_s}  {praw_s}  {pbonf_s}  "
            f"{effr_s}  {sig}"
        )
    print("  " + "-" * 118)
    print(
        "  Columns: k=#seeds, n=pooled #samples, mean±std across per-seed means, "
        "95% CI = bootstrap percentile on pooled rho, J@5 = mean top-5 Jaccard, "
        "W = Wilcoxon signed-rank statistic on (rho-1), p_Bonf = Bonferroni-corrected p, "
        "effect_r = rank-biserial effect size."
    )
    print(f"  Source: {csv_path}")
    return df


def generate_comparison_report(fp32_results, fp16_results, int8_results, int4_results,
                               fp32_size_mb, fp16_size_mb, int8_size_mb, int4_size_mb):
    comparison_data = {
        "Metric": [
            "Model Size (MB)", "Accuracy (%)", "Avg Confidence (%)",
            "Mean Latency (ms)", "Median Latency (ms)", "Std Latency (ms)"
        ],
        "FP32 (Baseline)": [
            f"{fp32_size_mb:.2f}", f"{fp32_results['accuracy']*100:.4f}",
            f"{fp32_results['avg_confidence']*100:.4f}",
            f"{fp32_results['latency_stats']['mean']*1000:.4f}",
            f"{fp32_results['latency_stats']['median']*1000:.4f}",
            f"{fp32_results['latency_stats']['std']*1000:.4f}"
        ],
        "FP16 (Half)": [
            f"{fp16_size_mb:.2f}", f"{fp16_results['accuracy']*100:.4f}",
            f"{fp16_results['avg_confidence']*100:.4f}",
            f"{fp16_results['latency_stats']['mean']*1000:.4f}",
            f"{fp16_results['latency_stats']['median']*1000:.4f}",
            f"{fp16_results['latency_stats']['std']*1000:.4f}"
        ],
        "INT8 (Quantized)": [
            f"{int8_size_mb:.2f}", f"{int8_results['accuracy']*100:.4f}",
            f"{int8_results['avg_confidence']*100:.4f}",
            f"{int8_results['latency_stats']['mean']*1000:.4f}",
            f"{int8_results['latency_stats']['median']*1000:.4f}",
            f"{int8_results['latency_stats']['std']*1000:.4f}"
        ],
        "INT4 (4-bit)": [
            f"{int4_size_mb:.2f}", f"{int4_results['accuracy']*100:.4f}",
            f"{int4_results['avg_confidence']*100:.4f}",
            f"{int4_results['latency_stats']['mean']*1000:.4f}",
            f"{int4_results['latency_stats']['median']*1000:.4f}",
            f"{int4_results['latency_stats']['std']*1000:.4f}"
        ],
        "FP16 vs FP32": [
            f"{(1 - fp16_size_mb/fp32_size_mb)*100:+.4f}%",
            f"{(fp16_results['accuracy'] - fp32_results['accuracy'])*100:+.4f}%",
            f"{(fp16_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.4f}%",
            f"{((fp16_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.4f}%",
            f"{((fp16_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.4f}%",
            "N/A"
        ],
        "INT8 vs FP32": [
            f"{(1 - int8_size_mb/fp32_size_mb)*100:+.4f}%",
            f"{(int8_results['accuracy'] - fp32_results['accuracy'])*100:+.4f}%",
            f"{(int8_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.4f}%",
            f"{((int8_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.4f}%",
            f"{((int8_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.4f}%",
            "N/A"
        ],
        "INT4 vs FP32": [
            f"{(1 - int4_size_mb/fp32_size_mb)*100:+.4f}%",
            f"{(int4_results['accuracy'] - fp32_results['accuracy'])*100:+.4f}%",
            f"{(int4_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.4f}%",
            f"{((int4_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.4f}%",
            f"{((int4_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.4f}%",
            "N/A"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    return df_comparison


def generate_prediction_comparison(fp32_results, fp16_results, int8_results, int4_results):
    prediction_comparison = []
    for i, (fp32_pred, fp16_pred, int8_pred, int4_pred) in enumerate(
        zip(fp32_results["predictions"], fp16_results["predictions"],
            int8_results["predictions"], int4_results["predictions"])
    ):
        fp16_match = "Y" if fp32_pred["predicted"] == fp16_pred["predicted"] else "N"
        int8_match = "Y" if fp32_pred["predicted"] == int8_pred["predicted"] else "N"
        int4_match = "Y" if fp32_pred["predicted"] == int4_pred["predicted"] else "N"
        prediction_comparison.append({
            "#": i+1,
            "Text": fp32_pred["text"][:30] + "...",
            "Expected": fp32_pred["expected"],
            "FP32": fp32_pred["predicted"],
            "FP32%": f"{fp32_pred['confidence']*100:.4f}%",
            "FP16": fp16_pred["predicted"],
            "FP16%": f"{fp16_pred['confidence']*100:.4f}%",
            "FP16=FP32": fp16_match,
            "INT8": int8_pred["predicted"],
            "INT8%": f"{int8_pred['confidence']*100:.4f}%",
            "INT8=FP32": int8_match,
            "INT4": int4_pred["predicted"],
            "INT4%": f"{int4_pred['confidence']*100:.4f}%",
            "INT4=FP32": int4_match
        })

    df_predictions = pd.DataFrame(prediction_comparison)

    fp16_mismatches = sum(1 for p in prediction_comparison if p["FP16=FP32"] == "N")
    int8_mismatches = sum(1 for p in prediction_comparison if p["INT8=FP32"] == "N")
    int4_mismatches = sum(1 for p in prediction_comparison if p["INT4=FP32"] == "N")

    consistency = {
        "fp16": (len(prediction_comparison) - fp16_mismatches) / len(prediction_comparison),
        "int8": (len(prediction_comparison) - int8_mismatches) / len(prediction_comparison),
        "int4": (len(prediction_comparison) - int4_mismatches) / len(prediction_comparison)
    }

    return df_predictions, consistency


def render_deployment_recommendation(
    json_path=None,
    csv_path=None,
    rebuild: bool = True,
):
    """Print the deployment-recommendation decision table.

    By default this calls the synthesizer to regenerate the underlying
    artifacts (deterministic — same inputs, same output bytes) and then
    renders the CSV. Pass rebuild=False to render whatever is already on disk.
    """
    if json_path is None or csv_path is None:
        _project_root = Path(__file__).resolve().parent.parent.parent
        _res_dir = _project_root / "outputs" / "deployment-recommendation"
        if json_path is None:
            json_path = _res_dir / "deployment_recommendation.json"
        if csv_path is None:
            csv_path = _res_dir / "deployment_recommendation.csv"

    json_path = Path(json_path)
    csv_path  = Path(csv_path)

    if rebuild:
        from src.evaluation.deployment_recommendation import (
            build_deployment_recommendation,
        )
        build_deployment_recommendation()

    if not csv_path.exists() or not json_path.exists():
        print(
            f"  [WARN] Deployment recommendation artifacts missing "
            f"(expected {json_path.name} and {csv_path.name}). "
            "Run XAI Diagnostics -> Deployment recommendation with rebuild=True."
        )
        return None

    with open(json_path, "r", encoding="utf-8") as _f:
        payload = json.load(_f)
    df = pd.read_csv(csv_path)

    meta       = payload.get("_meta", {})
    thresholds = meta.get("thresholds", {})
    preferred  = payload.get("preferred_per_constraint", {})

    print("\n  Deployment Recommendation (constraint-keyed)")
    print("  " + "-" * 118)
    print(
        f"  Thresholds: rho>={thresholds.get('stability_rho_acceptable')}, "
        f"agreement>={thresholds.get('agreement_min')}, "
        f"dF1<={thresholds.get('f1_drop_tolerance')}, "
        f"ECE<={thresholds.get('ece_max')}, "
        f"latency<={thresholds.get('latency_critical_ms')}ms, "
        f"size<={thresholds.get('size_critical_mb')}MB, "
        f"comp@5>={thresholds.get('faithfulness_comp_min')}"
    )
    print("  " + "-" * 118)

    # Per-variant metric table
    print(
        f"  {'variant':14s}  {'acc':>6s}  {'F1':>6s}  {'dF1':>7s}  "
        f"{'agr':>6s}  {'ECE':>6s}  {'rho':>6s}  {'comp@5':>7s}  "
        f"{'size_MB':>8s}  {'lat_ms':>7s}  {'recommended_use':<35s}"
    )
    print("  " + "-" * 118)
    for _, row in df.iterrows():
        def _fmt(v, w, d):
            if v is None or (isinstance(v, float) and (v != v)):
                return f"{'-':>{w}s}"
            try:
                return f"{float(v):>{w}.{d}f}"
            except (TypeError, ValueError):
                return f"{'-':>{w}s}"
        print(
            f"  {row['variant']:14s}  "
            f"{_fmt(row.get('accuracy_mean'),    6, 4)}  "
            f"{_fmt(row.get('macro_f1_mean'),    6, 4)}  "
            f"{_fmt(row.get('macro_f1_drop_vs_fp32'), 7, 4)}  "
            f"{_fmt(row.get('agreement_mean'),   6, 4)}  "
            f"{_fmt(row.get('ece_mean'),         6, 4)}  "
            f"{_fmt(row.get('rho_mean'),         6, 4)}  "
            f"{_fmt(row.get('mean_comp_k5'),     7, 4)}  "
            f"{_fmt(row.get('model_size_mb'),    8, 2)}  "
            f"{_fmt(row.get('latency_ms_mean'),  7, 2)}  "
            f"{str(row.get('recommended_use') or '-'):<35s}"
        )
    print("  " + "-" * 118)

    # Per-constraint preferred + rationale lines
    print("\n  Preferred per constraint (top qualified variant):")
    metrics = payload.get("metrics_per_variant", {})
    recs    = payload.get("recommendations", {})
    for constraint in meta.get("constraints", []):
        pref = preferred.get(constraint)
        if pref:
            row = metrics.get(pref, {})
            rationale = next(
                (r["rationale"] for r in recs.get(constraint, [])
                 if r["variant"] == pref),
                "",
            )
            print(f"    {constraint:28s} -> {pref:14s} ({rationale})")
        else:
            # Show the closest disqualified candidate so the user sees why
            # nothing qualified.
            ranked = recs.get(constraint, [])
            closest = next((r for r in ranked if r["score"] is not None), None)
            if closest:
                fail_str = "; ".join(closest["disqualifications"])
                print(
                    f"    {constraint:28s} -> NONE QUALIFIED "
                    f"(closest: {closest['variant']} — {fail_str})"
                )
            else:
                print(f"    {constraint:28s} -> NONE QUALIFIED (no data)")

    print("  " + "-" * 118)
    print(
        "  Columns: acc/F1 = mean across seeds, dF1 = F1 drop vs FP32, "
        "agr = FP32-prediction agreement, rho = mean Spearman across XAI "
        "methods, comp@5 = mean comprehensiveness at top-5."
    )
    print(f"  Source: {csv_path}")
    sources = meta.get("source_artifacts", {})
    if sources:
        print("  Backing artifacts:")
        for tag, path in sources.items():
            print(f"    {tag:14s} {path}")
    return df
