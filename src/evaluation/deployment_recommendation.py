"""Cross-variant deployment-recommendation synthesizer.

Reads the classification, stability, and faithfulness summaries that the
existing pipeline already produces and emits a constraint-keyed
recommendation table (interpretability-critical, latency-critical,
size-critical) plus a machine-readable rationale per variant.

The module is strictly artifact-driven: every value in the output traces
back to a key in `results/` (or `outputs/multi-seed/`) — nothing is
recomputed from raw predictions/attributions here. Decision thresholds are
named constants in `src.config`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    BASE_DIR,
    DEPLOYMENT_AGREEMENT_MIN,
    DEPLOYMENT_ECE_MAX,
    DEPLOYMENT_F1_DROP_TOLERANCE,
    DEPLOYMENT_FAITHFULNESS_COMP_MIN,
    DEPLOYMENT_LATENCY_CRITICAL_MS,
    DEPLOYMENT_RECOMMENDATION_DIR,
    DEPLOYMENT_SIZE_CRITICAL_MB,
    DEPLOYMENT_STABILITY_RHO_ACCEPTABLE,
    DEPLOYMENT_VARIANT_ALIASES,
    DEPLOYMENT_VARIANTS,
)

_RESULTS_DIR = BASE_DIR / "results"
_MULTISEED_DIR = BASE_DIR / "outputs" / "multi-seed"

_CLASSIFICATION_CSV = _RESULTS_DIR / "classification_summary_multiseed.csv"
_ECE_SUMMARY_CSV    = _RESULTS_DIR / "ece_summary.csv"
_STABILITY_JSON     = _RESULTS_DIR / "stability_results.json"
_FAITHFULNESS_CSV   = _RESULTS_DIR / "faithfulness_summary.csv"
_PTQ_AGG_JSON       = _MULTISEED_DIR / "aggregated_ptq_results.json"
_QAT_AGG_JSON       = _MULTISEED_DIR / "aggregated_qat_results.json"

_FAITHFULNESS_K = 5  # comprehensiveness floor is defined at top-k=5


# ---------------------------------------------------------------------------
# Loaders — each one reads ONE existing summary and returns a per-variant dict.
# Missing artifacts produce empty dicts (not exceptions) so the synthesizer
# can render a partial table when the user has only run some phases.
# ---------------------------------------------------------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def _canon(variant: str) -> str:
    return DEPLOYMENT_VARIANT_ALIASES.get(variant, variant).strip().lower()


def _load_classification() -> Dict[str, Dict[str, Optional[float]]]:
    if not _CLASSIFICATION_CSV.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(_CLASSIFICATION_CSV)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for raw_variant, grp in df.groupby("variant"):
        key = _canon(str(raw_variant))
        accs = grp["smsa_acc"].dropna() if "smsa_acc" in grp.columns else []
        f1s  = grp["smsa_f1"].dropna()  if "smsa_f1"  in grp.columns else []
        agrs = (grp["agreement_rate"].dropna()
                if "agreement_rate" in grp.columns else [])
        eces = grp["ece"].dropna() if "ece" in grp.columns else []
        out[key] = {
            "accuracy_mean":  _safe_float(accs.mean())  if len(accs) else None,
            "accuracy_std":   _safe_float(accs.std(ddof=1))  if len(accs) > 1 else None,
            "macro_f1_mean":  _safe_float(f1s.mean())   if len(f1s)  else None,
            "macro_f1_std":   _safe_float(f1s.std(ddof=1))   if len(f1s)  > 1 else None,
            "agreement_mean": _safe_float(agrs.mean())  if len(agrs) else None,
            "ece_mean":       _safe_float(eces.mean())  if len(eces) else None,
            "n_seeds":        int(len(grp)),
        }
    return out


def _load_ece_summary() -> Dict[str, Optional[float]]:
    if not _ECE_SUMMARY_CSV.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(_ECE_SUMMARY_CSV)
    return {
        _canon(str(r["variant"])): _safe_float(r["mean_ece"])
        for _, r in df.iterrows()
    }


def _load_stability() -> Dict[str, Dict[str, Optional[float]]]:
    if not _STABILITY_JSON.exists():
        return {}
    with open(_STABILITY_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for variant, methods in raw.items():
        key = _canon(str(variant))
        if not isinstance(methods, dict) or not methods:
            continue
        rhos = [_safe_float(m.get("mean_rho")) for m in methods.values()]
        rhos = [r for r in rhos if r is not None]
        j5s  = [_safe_float(m.get("mean_j5"))  for m in methods.values()]
        j5s  = [j for j in j5s  if j is not None]
        sig_flags = [bool(m.get("significant_bonferroni"))
                     for m in methods.values()]
        out[key] = {
            "rho_mean_across_methods":
                _safe_float(sum(rhos) / len(rhos)) if rhos else None,
            "rho_min_across_methods":
                _safe_float(min(rhos)) if rhos else None,
            "j5_mean_across_methods":
                _safe_float(sum(j5s) / len(j5s)) if j5s else None,
            "n_methods": len(methods),
            "n_methods_significant_drop": int(sum(sig_flags)),
        }
    return out


def _load_faithfulness() -> Dict[str, Dict[str, Optional[float]]]:
    if not _FAITHFULNESS_CSV.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(_FAITHFULNESS_CSV)
    df_k = df[df["k"] == _FAITHFULNESS_K] if "k" in df.columns else df
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for raw_variant, grp in df_k.groupby("variant"):
        key = _canon(str(raw_variant))
        out[key] = {
            "mean_comp_k5":
                _safe_float(grp["mean_comp"].mean())
                if "mean_comp" in grp.columns else None,
            "mean_suff_k5":
                _safe_float(grp["mean_suff"].mean())
                if "mean_suff" in grp.columns else None,
        }
    return out


def _load_size_latency() -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if _PTQ_AGG_JSON.exists():
        with open(_PTQ_AGG_JSON, "r", encoding="utf-8") as f:
            ptq = json.load(f)
        for variant, payload in ptq.items():
            if not isinstance(payload, dict):
                continue
            key = _canon(variant)
            if key not in DEPLOYMENT_VARIANTS:
                continue
            size_mb = _safe_float(payload.get("model_size_mb"))
            latency_ms = _extract_latency_ms_from_per_seed(
                ptq.get("per_seed", []), variant
            )
            out[key] = {
                "model_size_mb": size_mb,
                "latency_ms_mean": latency_ms,
            }
    if _QAT_AGG_JSON.exists():
        with open(_QAT_AGG_JSON, "r", encoding="utf-8") as f:
            qat = json.load(f)
        per_seed = qat.get("per_seed", [])
        for variant in ("qat_fp32", "qat_onnx_fp16",
                        "qat_onnx_int8", "qat_onnx_int4"):
            size_mb = _extract_size_from_per_seed(per_seed, variant)
            latency_ms = _extract_latency_ms_from_per_seed(per_seed, variant)
            if size_mb is None and latency_ms is None:
                continue
            out.setdefault(variant, {})
            out[variant].setdefault("model_size_mb", size_mb)
            out[variant].setdefault("latency_ms_mean", latency_ms)
    return out


def _extract_latency_ms_from_per_seed(
    per_seed: List[dict], variant: str,
) -> Optional[float]:
    vals: List[float] = []
    for entry in per_seed or []:
        if not isinstance(entry, dict):
            continue
        v = entry.get(variant) if variant in entry else entry
        if not isinstance(v, dict):
            continue
        lat = v.get("latency_stats") or v.get("latency") or {}
        if isinstance(lat, dict):
            mean_ms = lat.get("mean_ms")
            if mean_ms is None and "mean" in lat:
                mean_ms = float(lat["mean"]) * 1000.0
            f = _safe_float(mean_ms)
            if f is not None:
                vals.append(f)
    if not vals:
        return None
    return _safe_float(sum(vals) / len(vals))


def _extract_size_from_per_seed(
    per_seed: List[dict], variant: str,
) -> Optional[float]:
    vals: List[float] = []
    for entry in per_seed or []:
        if not isinstance(entry, dict):
            continue
        v = entry.get(variant) if variant in entry else entry
        if not isinstance(v, dict):
            continue
        f = _safe_float(v.get("model_size_mb"))
        if f is not None:
            vals.append(f)
    if not vals:
        return None
    return _safe_float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Per-variant metric assembly
# ---------------------------------------------------------------------------

def _assemble_metrics() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Returns (per_variant_metrics, provenance) where provenance lists the
    source artifact(s) each variant's numbers came from."""
    classification = _load_classification()
    ece_summary    = _load_ece_summary()
    stability      = _load_stability()
    faithfulness   = _load_faithfulness()
    size_latency   = _load_size_latency()

    metrics: Dict[str, Dict[str, Any]] = {}
    provenance: Dict[str, List[str]] = {}

    fp32_f1 = (classification.get("fp32", {}) or {}).get("macro_f1_mean")

    for variant in DEPLOYMENT_VARIANTS:
        row: Dict[str, Any] = {"variant": variant}
        sources: List[str] = []

        cls = classification.get(variant, {})
        if cls:
            row["accuracy_mean"]  = cls.get("accuracy_mean")
            row["macro_f1_mean"]  = cls.get("macro_f1_mean")
            row["agreement_mean"] = cls.get("agreement_mean")
            row["n_seeds"]        = cls.get("n_seeds")
            sources.append(str(_CLASSIFICATION_CSV.relative_to(BASE_DIR)))

        ece_v = ece_summary.get(variant)
        if ece_v is None:
            ece_v = cls.get("ece_mean") if cls else None
        if ece_v is not None:
            row["ece_mean"] = ece_v
            if _ECE_SUMMARY_CSV.exists():
                sources.append(str(_ECE_SUMMARY_CSV.relative_to(BASE_DIR)))

        stab = stability.get(variant, {})
        if stab:
            row["rho_mean"] = stab.get("rho_mean_across_methods")
            row["rho_min"]  = stab.get("rho_min_across_methods")
            row["n_methods_significant_drop"] = stab.get(
                "n_methods_significant_drop")
            sources.append(str(_STABILITY_JSON.relative_to(BASE_DIR)))

        fai = faithfulness.get(variant, {})
        if fai:
            row["mean_comp_k5"] = fai.get("mean_comp_k5")
            row["mean_suff_k5"] = fai.get("mean_suff_k5")
            sources.append(str(_FAITHFULNESS_CSV.relative_to(BASE_DIR)))

        sl = size_latency.get(variant, {})
        if sl:
            if sl.get("model_size_mb") is not None:
                row["model_size_mb"] = sl["model_size_mb"]
            if sl.get("latency_ms_mean") is not None:
                row["latency_ms_mean"] = sl["latency_ms_mean"]
            if variant.startswith("ptq") or variant == "fp32":
                if _PTQ_AGG_JSON.exists():
                    sources.append(str(_PTQ_AGG_JSON.relative_to(BASE_DIR)))
            if variant.startswith("qat"):
                if _QAT_AGG_JSON.exists():
                    sources.append(str(_QAT_AGG_JSON.relative_to(BASE_DIR)))

        # Derived helper: f1 drop vs FP32 (still artifact-derived — no
        # recomputation; FP32 F1 came from the same CSV).
        if fp32_f1 is not None and row.get("macro_f1_mean") is not None:
            row["macro_f1_drop_vs_fp32"] = round(
                fp32_f1 - row["macro_f1_mean"], 6
            )

        # Size ratio (vs FP32) — purely derived from already-loaded sizes.
        fp32_size = (size_latency.get("fp32") or {}).get("model_size_mb")
        if fp32_size and row.get("model_size_mb"):
            row["size_ratio_vs_fp32"] = round(
                row["model_size_mb"] / fp32_size, 4
            )

        # Latency speedup (vs FP32).
        fp32_lat = (size_latency.get("fp32") or {}).get("latency_ms_mean")
        if fp32_lat and row.get("latency_ms_mean"):
            row["latency_speedup_vs_fp32"] = round(
                fp32_lat / row["latency_ms_mean"], 4
            )

        metrics[variant] = row
        provenance[variant] = sorted(set(sources))

    return metrics, provenance


# ---------------------------------------------------------------------------
# Constraint-keyed scoring
# ---------------------------------------------------------------------------

_CONSTRAINTS = (
    "interpretability_critical",
    "latency_critical",
    "size_critical",
)


def _passes_baseline(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Generic baseline filter applied to every constraint: accuracy must
    not regress beyond the configured tolerance, agreement must be at or
    above the floor, and ECE must be below the ceiling. Returns
    (passes, reasons_for_failure)."""
    fails: List[str] = []
    drop = row.get("macro_f1_drop_vs_fp32")
    if drop is not None and drop > DEPLOYMENT_F1_DROP_TOLERANCE:
        fails.append(
            f"F1 drop {drop:.4f} > tolerance {DEPLOYMENT_F1_DROP_TOLERANCE:.4f}"
        )
    agr = row.get("agreement_mean")
    if agr is not None and agr < DEPLOYMENT_AGREEMENT_MIN:
        fails.append(
            f"agreement {agr:.4f} < min {DEPLOYMENT_AGREEMENT_MIN:.4f}"
        )
    ece = row.get("ece_mean")
    if ece is not None and ece > DEPLOYMENT_ECE_MAX:
        fails.append(
            f"ECE {ece:.4f} > max {DEPLOYMENT_ECE_MAX:.4f}"
        )
    return (len(fails) == 0, fails)


def _score_constraint(
    constraint: str, row: Dict[str, Any]
) -> Tuple[Optional[float], List[str], List[str]]:
    """Returns (score, qualifications, disqualifications). Higher score is
    better within a constraint lane; None means insufficient data."""
    passes, baseline_fails = _passes_baseline(row)
    quals: List[str] = []
    disquals: List[str] = list(baseline_fails)

    if constraint == "interpretability_critical":
        rho = row.get("rho_mean")
        comp = row.get("mean_comp_k5")
        if rho is None:
            return None, quals, disquals + ["stability rho unavailable"]
        if rho < DEPLOYMENT_STABILITY_RHO_ACCEPTABLE:
            disquals.append(
                f"rho {rho:.4f} < acceptable "
                f"{DEPLOYMENT_STABILITY_RHO_ACCEPTABLE:.2f}"
            )
        else:
            quals.append(f"rho {rho:.4f} >= {DEPLOYMENT_STABILITY_RHO_ACCEPTABLE:.2f}")
        if comp is not None:
            if comp < DEPLOYMENT_FAITHFULNESS_COMP_MIN:
                disquals.append(
                    f"comp@5 {comp:.4f} < min "
                    f"{DEPLOYMENT_FAITHFULNESS_COMP_MIN:.4f}"
                )
            else:
                quals.append(
                    f"comp@5 {comp:.4f} >= {DEPLOYMENT_FAITHFULNESS_COMP_MIN:.4f}"
                )
        score = rho  # rank by mean rho; ties broken by F1 drop below
        if not passes or disquals != baseline_fails:
            # baseline OR constraint-specific dq — still report a score so
            # the user can see how close it got, but mark as disqualified.
            pass
        # Tie-break: subtract any F1 regression so a slightly-worse-F1
        # variant ranks below an otherwise-equal one.
        drop = row.get("macro_f1_drop_vs_fp32") or 0.0
        score = score - max(0.0, drop)
        return score, quals, disquals

    if constraint == "latency_critical":
        lat = row.get("latency_ms_mean")
        if lat is None:
            return None, quals, disquals + ["latency unavailable"]
        if lat > DEPLOYMENT_LATENCY_CRITICAL_MS:
            disquals.append(
                f"latency {lat:.2f} ms > budget "
                f"{DEPLOYMENT_LATENCY_CRITICAL_MS:.2f} ms"
            )
        else:
            quals.append(
                f"latency {lat:.2f} ms <= {DEPLOYMENT_LATENCY_CRITICAL_MS:.2f} ms"
            )
        score = -lat  # lower latency is better
        return score, quals, disquals

    if constraint == "size_critical":
        size = row.get("model_size_mb")
        if size is None:
            return None, quals, disquals + ["model size unavailable"]
        if size > DEPLOYMENT_SIZE_CRITICAL_MB:
            disquals.append(
                f"size {size:.2f} MB > budget "
                f"{DEPLOYMENT_SIZE_CRITICAL_MB:.2f} MB"
            )
        else:
            quals.append(
                f"size {size:.2f} MB <= {DEPLOYMENT_SIZE_CRITICAL_MB:.2f} MB"
            )
        score = -size  # smaller is better
        return score, quals, disquals

    raise ValueError(f"unknown constraint: {constraint}")


def _rationale(row: Dict[str, Any]) -> str:
    """Compact machine-readable rationale derived from row values + thresholds."""
    parts: List[str] = []
    if row.get("rho_mean") is not None:
        parts.append(f"rho={row['rho_mean']:.3f}")
    if row.get("latency_speedup_vs_fp32") is not None:
        parts.append(f"{row['latency_speedup_vs_fp32']:.2f}x latency")
    elif row.get("latency_ms_mean") is not None:
        parts.append(f"latency={row['latency_ms_mean']:.2f}ms")
    if row.get("size_ratio_vs_fp32") is not None:
        parts.append(f"{row['size_ratio_vs_fp32'] * 100:.0f}% size")
    elif row.get("model_size_mb") is not None:
        parts.append(f"size={row['model_size_mb']:.0f}MB")
    if row.get("macro_f1_drop_vs_fp32") is not None:
        parts.append(f"dF1={row['macro_f1_drop_vs_fp32']:+.4f}")
    if row.get("ece_mean") is not None:
        parts.append(f"ECE={row['ece_mean']:.3f}")
    return ", ".join(parts) if parts else "no metrics available"


def _rank_constraint(
    constraint: str, metrics: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for variant in DEPLOYMENT_VARIANTS:
        row = metrics.get(variant, {})
        score, quals, disquals = _score_constraint(constraint, row)
        ranked.append({
            "variant": variant,
            "score": score,
            "qualified": (score is not None and not disquals),
            "qualifications": quals,
            "disqualifications": disquals,
            "rationale": _rationale(row),
        })
    # Qualified entries first (by score desc), then disqualified-with-score,
    # then unavailable. Stable on variant order for reproducibility.
    def _sort_key(r: Dict[str, Any]) -> Tuple[int, float, int]:
        if r["score"] is None:
            return (2, 0.0, DEPLOYMENT_VARIANTS.index(r["variant"]))
        bucket = 0 if r["qualified"] else 1
        return (bucket, -float(r["score"]),
                DEPLOYMENT_VARIANTS.index(r["variant"]))
    ranked.sort(key=_sort_key)
    return ranked


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_deployment_recommendation() -> Dict[str, Any]:
    """Synthesize the recommendation payload from existing summaries.

    Returns the payload dict; also writes:
      outputs/deployment-recommendation/deployment_recommendation.json
      outputs/deployment-recommendation/deployment_recommendation.csv
    """
    metrics, provenance = _assemble_metrics()

    recommendations = {
        c: _rank_constraint(c, metrics) for c in _CONSTRAINTS
    }

    preferred = {}
    for c, ranked in recommendations.items():
        qualified = [r for r in ranked if r["qualified"]]
        preferred[c] = qualified[0]["variant"] if qualified else None

    payload: Dict[str, Any] = {
        "_meta": {
            "source_artifacts": {
                "classification": str(
                    _CLASSIFICATION_CSV.relative_to(BASE_DIR)),
                "ece_summary": str(_ECE_SUMMARY_CSV.relative_to(BASE_DIR)),
                "stability":   str(_STABILITY_JSON.relative_to(BASE_DIR)),
                "faithfulness": str(_FAITHFULNESS_CSV.relative_to(BASE_DIR)),
                "ptq_aggregate": str(_PTQ_AGG_JSON.relative_to(BASE_DIR)),
                "qat_aggregate": str(_QAT_AGG_JSON.relative_to(BASE_DIR)),
            },
            "thresholds": {
                "stability_rho_acceptable":
                    DEPLOYMENT_STABILITY_RHO_ACCEPTABLE,
                "agreement_min":         DEPLOYMENT_AGREEMENT_MIN,
                "f1_drop_tolerance":     DEPLOYMENT_F1_DROP_TOLERANCE,
                "ece_max":               DEPLOYMENT_ECE_MAX,
                "latency_critical_ms":   DEPLOYMENT_LATENCY_CRITICAL_MS,
                "size_critical_mb":      DEPLOYMENT_SIZE_CRITICAL_MB,
                "faithfulness_comp_min": DEPLOYMENT_FAITHFULNESS_COMP_MIN,
            },
            "constraints": list(_CONSTRAINTS),
            "faithfulness_k": _FAITHFULNESS_K,
        },
        "metrics_per_variant": metrics,
        "provenance_per_variant": provenance,
        "recommendations": recommendations,
        "preferred_per_constraint": preferred,
    }

    DEPLOYMENT_RECOMMENDATION_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEPLOYMENT_RECOMMENDATION_DIR / "deployment_recommendation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, sort_keys=True)

    _write_csv(metrics, recommendations, preferred)

    print(f"  Deployment recommendation JSON saved -> {json_path}")
    return payload


def _write_csv(
    metrics: Dict[str, Dict[str, Any]],
    recommendations: Dict[str, List[Dict[str, Any]]],
    preferred: Dict[str, Optional[str]],
) -> None:
    import pandas as pd

    # Build a per-variant row: metric columns + one recommended_use column
    # listing the constraint(s) for which this variant is the preferred pick.
    rows: List[Dict[str, Any]] = []
    for variant in DEPLOYMENT_VARIANTS:
        row = dict(metrics.get(variant, {"variant": variant}))
        rec_uses = sorted([c for c, v in preferred.items() if v == variant])
        row["recommended_use"] = ";".join(rec_uses) if rec_uses else ""
        # Pull per-constraint qualification flags so the table is self-contained.
        for c in _CONSTRAINTS:
            ranked = recommendations.get(c, [])
            entry = next((r for r in ranked if r["variant"] == variant), None)
            row[f"{c}_qualified"] = bool(entry and entry["qualified"])
            row[f"{c}_disqualifications"] = (
                "; ".join(entry["disqualifications"]) if entry else ""
            )
        row["rationale"] = _rationale(metrics.get(variant, {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Stable column ordering so re-runs are byte-identical.
    leading = [
        "variant", "accuracy_mean", "macro_f1_mean", "macro_f1_drop_vs_fp32",
        "agreement_mean", "ece_mean", "rho_mean", "rho_min",
        "mean_comp_k5", "mean_suff_k5",
        "model_size_mb", "size_ratio_vs_fp32",
        "latency_ms_mean", "latency_speedup_vs_fp32",
        "n_seeds", "n_methods_significant_drop",
        "interpretability_critical_qualified",
        "latency_critical_qualified",
        "size_critical_qualified",
        "interpretability_critical_disqualifications",
        "latency_critical_disqualifications",
        "size_critical_disqualifications",
        "recommended_use", "rationale",
    ]
    cols = [c for c in leading if c in df.columns] + [
        c for c in df.columns if c not in leading
    ]
    df = df[cols]
    csv_path = DEPLOYMENT_RECOMMENDATION_DIR / "deployment_recommendation.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Deployment recommendation CSV  saved -> {csv_path}")
