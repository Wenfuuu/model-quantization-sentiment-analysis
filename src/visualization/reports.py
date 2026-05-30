"""Tabular report renderers.

Every multi-variant result table goes through a single render engine driven
by a declarative spec. Adding a new table is one TABLE_SPECS entry + one
thin wrapper. No bespoke per-table formatters; no hardcoded numeric cells —
every value comes from the CSV/JSON the upstream analysis already writes
to outputs/ or results/.

Registered tables (canonical artifact stems):
    qat_drift_decomposition  -> results/qat_drift_decomposition.{json,csv}
    large_sample_stability   -> outputs/multi-seed/large-sample-stability/
                                large_sample_stability.json,
                                stability_aggregate_acrossSeeds.csv
    stability_by_family      -> results/stability_by_family.json,
                                results/table3a_gradient.csv,
                                results/table3b_perturbation.csv
                                (also exposed as render_generalization — the
                                 cross-method-family generalization view)
    deployment_recommendation -> outputs/deployment-recommendation/
                                 deployment_recommendation.{json,csv}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DASH = "-"


# ---------------------------------------------------------------------------
# Cell formatting primitives
# ---------------------------------------------------------------------------

def _fmt_cell(value: Any, fmt: str, width: int, align: str = "right") -> str:
    """Format a single cell to a fixed-width field.

    fmt codes: "s" (string), "d" (int), ".Nf" (float), ".Nf%" (float * 100,
    no literal %). Sentinel strings (e.g. "N/A") render right-padded as-is
    even when fmt is numeric. None / NaN render as "-".
    """
    pad = ">" if align == "right" else "<"
    if value is None or (isinstance(value, float) and value != value):
        return f"{_DASH:{pad}{width}s}"
    if isinstance(value, str) and fmt != "s":
        return f"{value:{pad}{width}s}"
    try:
        if fmt == "s":
            return f"{str(value):{pad}{width}s}"
        if fmt == "d":
            return f"{int(value):{pad}{width}d}"
        if fmt.endswith("f%"):
            return f"{float(value) * 100:{pad}{width}{fmt[:-1]}}"
        return f"{float(value):{pad}{width}{fmt}}"
    except (TypeError, ValueError):
        return f"{_DASH:{pad}{width}s}"


def _ci_cell(center_key: str, lo_key: str, hi_key: str,
             width: int, decimals: int):
    def _r(row: Dict[str, Any]) -> str:
        try:
            c  = float(row[center_key])
            lo = float(row[lo_key])
            hi = float(row[hi_key])
        except (KeyError, TypeError, ValueError):
            return f"{_DASH:>{width}s}"
        if any(v != v for v in (c, lo, hi)):
            return f"{_DASH:>{width}s}"
        return f"{c:.{decimals}f} [{lo:.{decimals}f},{hi:.{decimals}f}]".rjust(width)
    return _r


def _ci_only_cell(lo_key: str, hi_key: str, width: int, decimals: int,
                  align: str = "left"):
    pad = ">" if align == "right" else "<"
    def _r(row: Dict[str, Any]) -> str:
        try:
            lo = float(row[lo_key]); hi = float(row[hi_key])
        except (KeyError, TypeError, ValueError):
            return f"{'':{pad}{width}s}"
        if lo != lo or hi != hi:
            return f"{'':{pad}{width}s}"
        return f"[{lo:.{decimals}f},{hi:.{decimals}f}]".__format__(f"{pad}{width}s")
    return _r


def _mean_std_cell(mean_key: str, std_key: str, width: int, decimals: int):
    def _r(row: Dict[str, Any]) -> str:
        try:
            m = float(row[mean_key]); s = float(row[std_key])
        except (KeyError, TypeError, ValueError):
            return f"{_DASH:>{width}s}"
        if m != m or s != s:
            return f"{_DASH:>{width}s}"
        return f"{m:.{decimals}f}±{s:.{decimals}f}".rjust(width)
    return _r


def _sig_cell(key: str, width: int):
    def _r(row: Dict[str, Any]) -> str:
        return ("*" if bool(row.get(key)) else " ").rjust(width)
    return _r


def _sig_or_note_cell(sig_key: str, note_key: str, width: int):
    """Sig column for the stability-by-family table: shows '*' when significant,
    'STE' for STE_PROXY_INVALID, the note label for other invalid cases,
    ' ' otherwise."""
    def _r(row: Dict[str, Any]) -> str:
        note = str(row.get(note_key, "") or "")
        if note == "STE_PROXY_INVALID":
            return "STE".ljust(width)
        if note:
            return note.ljust(width)
        return ("*" if bool(row.get(sig_key)) else " ").ljust(width)
    return _r


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _Col:
    label: str
    width: int
    key: Optional[str] = None
    fmt: str = "s"
    align: str = "right"
    render: Optional[Callable[[Dict[str, Any]], str]] = None

    def header_cell(self) -> str:
        pad = ">" if self.align == "right" else "<"
        return f"{self.label:{pad}{self.width}s}"

    def render_cell(self, row: Dict[str, Any]) -> str:
        if self.render is not None:
            return self.render(row)
        return _fmt_cell(row.get(self.key), self.fmt, self.width, self.align)


@dataclass
class _RenderSpec:
    title: str
    default_paths: Dict[str, Sequence[str]]
    missing_msg: str
    columns: List[_Col]
    rule_width: int = 100
    sort_keys: Sequence[str] = field(default_factory=tuple)
    header_fn: Optional[Callable[[Dict[str, Any], pd.DataFrame], List[str]]] = None
    extra_sections: List[Callable[[Dict[str, Any], pd.DataFrame], List[str]]] = field(default_factory=list)
    footer_fn: Optional[Callable[[Dict[str, Any], pd.DataFrame, Path], List[str]]] = None


def _resolve_default(spec: _RenderSpec, kind: str) -> Path:
    return _PROJECT_ROOT.joinpath(*spec.default_paths[kind])


# ---------------------------------------------------------------------------
# Render engine
# ---------------------------------------------------------------------------

def _render_from_spec(
    spec: _RenderSpec,
    json_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    json_path = Path(json_path) if json_path else _resolve_default(spec, "json")
    csv_path  = Path(csv_path)  if csv_path  else _resolve_default(spec, "csv")

    if not csv_path.exists():
        print(f"  [WARN] {csv_path.name} not found. {spec.missing_msg}")
        return None

    df = pd.read_csv(csv_path)
    payload: Dict[str, Any] = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as _f:
            payload = json.load(_f)

    rule = "  " + "-" * spec.rule_width
    print(f"\n  {spec.title}")
    print(rule)

    if spec.header_fn:
        for line in spec.header_fn(payload, df):
            print(line if line == "" else f"  {line}")
        print(rule)

    print("  " + "  ".join(c.header_cell() for c in spec.columns))

    rows = df.sort_values(list(spec.sort_keys)) if spec.sort_keys else df
    for _, row in rows.iterrows():
        row_dict = row.to_dict()
        print("  " + "  ".join(c.render_cell(row_dict) for c in spec.columns))
    print(rule)

    for section_fn in spec.extra_sections:
        for line in section_fn(payload, df):
            print(line if line == "" else f"  {line}")
        print(rule)

    if spec.footer_fn:
        for line in spec.footer_fn(payload, df, csv_path):
            print(line if line == "" else f"  {line}")

    return df


# ---------------------------------------------------------------------------
# Registered specs
# ---------------------------------------------------------------------------

def _qat_drift_footer(payload: Dict[str, Any], _df: pd.DataFrame, csv_path: Path) -> List[str]:
    meta = payload.get("_meta", {})
    boot = meta.get("bootstrap_n_resamples")
    boot_line = (f"Bootstrap: percentile CIs from n_resamples={boot}."
                 if boot is not None else None)
    src = meta.get("source", str(csv_path))
    lines = [
        "Total = 1 - rho(FP32, QAT-FP32). Training = 1 - rho(FP32, FP32-Ctrl).",
        "FakeQuant = rho(FP32-Ctrl) - rho(QAT-FP32). Shares are means of components / mean total.",
    ]
    if boot_line:
        lines.append(boot_line)
    lines.append(f"Source: {src}")
    return lines


_QAT_DRIFT_DECOMPOSITION = _RenderSpec(
    title="QAT-FP32 Drift Decomposition (per attribution method)",
    default_paths={
        "json": ("results", "qat_drift_decomposition.json"),
        "csv":  ("results", "qat_drift_decomposition.csv"),
    },
    missing_msg="Run Stability Analysis first (Option [3] -> Stability Analysis).",
    rule_width=96,
    header_fn=lambda payload, _df: [
        payload.get("_meta", {}).get("definition", ""),
    ],
    columns=[
        _Col("method",     5, key="method",           fmt="s",   align="left"),
        _Col("n",          4, key="n_paired",         fmt="d"),
        _Col("rho(Ctrl)",  9, key="rho_fp32_control", fmt=".3f"),
        _Col("rho(QAT)",   8, key="rho_qat_fp32",     fmt=".3f"),
        _Col("Total",     16, render=_ci_cell("total_drift",        "total_ci95_lo",     "total_ci95_hi",     16, 3)),
        _Col("Training",  16, render=_ci_cell("training_component", "training_ci95_lo",  "training_ci95_hi",  16, 3)),
        _Col("FakeQuant", 16, render=_ci_cell("fakequant_residual", "fakequant_ci95_lo", "fakequant_ci95_hi", 16, 3)),
        _Col("Train%",     6, key="training_share",         fmt=".1f%"),
        _Col("FQ%",        6, key="fakequant_share",        fmt=".1f%"),
        _Col("p(FQ)",      8, key="residual_wilcoxon_p",    fmt=".4f"),
    ],
    footer_fn=_qat_drift_footer,
)


def _large_n_header(payload: Dict[str, Any], _df: pd.DataFrame) -> List[str]:
    meta = payload.get("_meta", {})
    return [
        (f"Bonferroni family size = {meta.get('bonferroni_family_size', '?')}, "
         f"alpha_corrected = {meta.get('bonferroni_alpha', float('nan'))} "
         f"(family-wise alpha = 0.05). Wilcoxon H1: rho < 1.0 (one-sided)."),
    ]


def _large_n_footer(_payload: Dict[str, Any], _df: pd.DataFrame, csv_path: Path) -> List[str]:
    return [
        ("Columns: k=#seeds, n=pooled #samples, mean±std across per-seed means, "
         "95% CI = bootstrap percentile on pooled rho, J@5 = mean top-5 Jaccard, "
         "W = Wilcoxon signed-rank statistic on (rho-1), p_Bonf = Bonferroni-corrected p, "
         "effect_r = rank-biserial effect size."),
        f"Source: {csv_path}",
    ]


_LARGE_SAMPLE_STABILITY = _RenderSpec(
    title="Large-sample cross-seed stability (per attribution method × variant)",
    default_paths={
        "json": ("outputs", "multi-seed", "large-sample-stability", "large_sample_stability.json"),
        "csv":  ("outputs", "multi-seed", "large-sample-stability", "stability_aggregate_acrossSeeds.csv"),
    },
    missing_msg="Run XAI -> Large-sample cross-seed stability first.",
    rule_width=118,
    sort_keys=("method", "variant"),
    header_fn=_large_n_header,
    columns=[
        _Col("method",    5, key="method",           fmt="s",   align="left"),
        _Col("variant",  15, key="variant",          fmt="s",   align="left"),
        _Col("k",         3, key="n_seeds",          fmt="d"),
        _Col("n",         5, key="n_samples_total",  fmt="d"),
        _Col("mean±std", 14, render=_mean_std_cell("mean_rho_across_seeds",
                                                   "std_rho_across_seeds", 14, 3)),
        _Col("95% CI",   17, render=_ci_only_cell("pooled_ci95_lo_rho",
                                                  "pooled_ci95_hi_rho", 17, 3)),
        _Col("J@5",       5, key="pooled_mean_j5",       fmt=".3f"),
        _Col("W",         8, key="wilcoxon_stat",        fmt=".1f"),
        _Col("p_raw",     8, key="wilcoxon_p",           fmt=".4f"),
        _Col("p_Bonf",    8, key="p_bonferroni",         fmt=".4f"),
        _Col("effect_r",  9, key="effect_r",             fmt=".3f"),
        _Col("sig?",      4, render=_sig_cell("significant_bonferroni", 4)),
    ],
    footer_fn=_large_n_footer,
)


def _family_header(family_label: str) -> Callable:
    def _h(_payload: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        methods = ""
        if "methods" in df.columns and not df.empty:
            methods = str(df["methods"].iloc[0])
        return [f"{family_label} family (methods: {methods or '?'})"]
    return _h


_STABILITY_BY_FAMILY_COLUMNS = [
    _Col("variant", 22, key="variant", fmt="s", align="left"),
    _Col("rho",      6, key="rho",     fmt=".3f"),
    _Col("95% CI",  15, render=_ci_only_cell("ci_low", "ci_high", 15, 3, align="left"), align="left"),
    _Col("sig?",     4, render=_sig_or_note_cell("significant_bonferroni", "note", 4), align="left"),
]


_STABILITY_BY_FAMILY_GRADIENT = _RenderSpec(
    title="Stability by method family — generalization across attribution methods",
    default_paths={
        "json": ("results", "stability_by_family.json"),
        "csv":  ("results", "table3a_gradient.csv"),
    },
    missing_msg="Run XAI Diagnostics -> Stability Analysis first (writes stability_by_family.json).",
    rule_width=60,
    header_fn=_family_header("GRADIENT"),
    columns=_STABILITY_BY_FAMILY_COLUMNS,
    footer_fn=lambda _p, _d, csv_path: [f"Source: {csv_path}"],
)


_STABILITY_BY_FAMILY_PERTURBATION = _RenderSpec(
    title="Stability by method family — generalization across attribution methods",
    default_paths={
        "json": ("results", "stability_by_family.json"),
        "csv":  ("results", "table3b_perturbation.csv"),
    },
    missing_msg="Run XAI Diagnostics -> Stability Analysis first (writes stability_by_family.json).",
    rule_width=60,
    header_fn=_family_header("PERTURBATION"),
    columns=_STABILITY_BY_FAMILY_COLUMNS,
    footer_fn=lambda _p, _d, csv_path: [f"Source: {csv_path}"],
)


def _deployment_header(payload: Dict[str, Any], _df: pd.DataFrame) -> List[str]:
    thresholds = payload.get("_meta", {}).get("thresholds", {})
    return [
        (f"Thresholds: rho>={thresholds.get('stability_rho_acceptable')}, "
         f"agreement>={thresholds.get('agreement_min')}, "
         f"dF1<={thresholds.get('f1_drop_tolerance')}, "
         f"ECE<={thresholds.get('ece_max')}, "
         f"latency<={thresholds.get('latency_critical_ms')}ms, "
         f"size<={thresholds.get('size_critical_mb')}MB, "
         f"comp@5>={thresholds.get('faithfulness_comp_min')}"),
    ]


def _deployment_preferred_section(payload: Dict[str, Any], _df: pd.DataFrame) -> List[str]:
    meta       = payload.get("_meta", {})
    preferred  = payload.get("preferred_per_constraint", {})
    metrics    = payload.get("metrics_per_variant", {})
    recs       = payload.get("recommendations", {})
    lines: List[str] = ["", "Preferred per constraint (top qualified variant):"]
    for constraint in meta.get("constraints", []):
        pref = preferred.get(constraint)
        if pref:
            _ = metrics.get(pref, {})
            rationale = next(
                (r["rationale"] for r in recs.get(constraint, [])
                 if r["variant"] == pref),
                "",
            )
            lines.append(f"  {constraint:28s} -> {pref:14s} ({rationale})")
        else:
            ranked = recs.get(constraint, [])
            closest = next((r for r in ranked if r.get("score") is not None), None)
            if closest:
                fail_str = "; ".join(closest.get("disqualifications", []))
                lines.append(
                    f"  {constraint:28s} -> NONE QUALIFIED "
                    f"(closest: {closest['variant']} — {fail_str})"
                )
            else:
                lines.append(f"  {constraint:28s} -> NONE QUALIFIED (no data)")
    return lines


def _deployment_footer(payload: Dict[str, Any], _df: pd.DataFrame, csv_path: Path) -> List[str]:
    lines = [
        ("Columns: acc/F1 = mean across seeds, dF1 = F1 drop vs FP32, "
         "agr = FP32-prediction agreement, rho = mean Spearman across XAI "
         "methods, comp@5 = mean comprehensiveness at top-5."),
        f"Source: {csv_path}",
    ]
    sources = payload.get("_meta", {}).get("source_artifacts", {})
    if sources:
        lines.append("Backing artifacts:")
        for tag, path in sources.items():
            lines.append(f"  {tag:14s} {path}")
    return lines


_DEPLOYMENT_RECOMMENDATION = _RenderSpec(
    title="Deployment Recommendation (constraint-keyed)",
    default_paths={
        "json": ("outputs", "deployment-recommendation", "deployment_recommendation.json"),
        "csv":  ("outputs", "deployment-recommendation", "deployment_recommendation.csv"),
    },
    missing_msg="Run XAI Diagnostics -> Deployment recommendation with rebuild=True.",
    rule_width=118,
    header_fn=_deployment_header,
    columns=[
        _Col("variant",         14, key="variant",                  fmt="s",    align="left"),
        _Col("acc",              6, key="accuracy_mean",            fmt=".4f"),
        _Col("F1",               6, key="macro_f1_mean",            fmt=".4f"),
        _Col("dF1",              7, key="macro_f1_drop_vs_fp32",    fmt=".4f"),
        _Col("agr",              6, key="agreement_mean",           fmt=".4f"),
        _Col("ECE",              6, key="ece_mean",                 fmt=".4f"),
        _Col("rho",              6, key="rho_mean",                 fmt=".4f"),
        _Col("comp@5",           7, key="mean_comp_k5",             fmt=".4f"),
        _Col("size_MB",          8, key="model_size_mb",            fmt=".2f"),
        _Col("lat_ms",           7, key="latency_ms_mean",          fmt=".2f"),
        _Col("recommended_use", 35, key="recommended_use",          fmt="s",    align="left"),
    ],
    extra_sections=[_deployment_preferred_section],
    footer_fn=_deployment_footer,
)


TABLE_SPECS: Dict[str, _RenderSpec] = {
    "qat_drift_decomposition":          _QAT_DRIFT_DECOMPOSITION,
    "large_sample_stability":           _LARGE_SAMPLE_STABILITY,
    "stability_by_family_gradient":     _STABILITY_BY_FAMILY_GRADIENT,
    "stability_by_family_perturbation": _STABILITY_BY_FAMILY_PERTURBATION,
    "deployment_recommendation":        _DEPLOYMENT_RECOMMENDATION,
}


# ---------------------------------------------------------------------------
# Thin wrappers — preserve the public API used elsewhere in the codebase
# ---------------------------------------------------------------------------

def render_qat_drift_decomposition(json_path=None, csv_path=None):
    return _render_from_spec(TABLE_SPECS["qat_drift_decomposition"],
                             json_path=json_path, csv_path=csv_path)


def render_large_sample_stability(json_path=None, csv_path=None):
    return _render_from_spec(TABLE_SPECS["large_sample_stability"],
                             json_path=json_path, csv_path=csv_path)


def render_stability_by_family(
    json_path=None,
    csv_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Render the cross-method-family stability table (the generalization view).

    Two sub-tables — gradient and perturbation families — read from the
    table3a/table3b CSVs that `compute_stability_by_family()` already writes.
    Pass csv_paths={"gradient": Path(...), "perturbation": Path(...)} to
    override defaults.
    """
    csv_paths = csv_paths or {}
    return {
        "gradient": _render_from_spec(
            TABLE_SPECS["stability_by_family_gradient"],
            json_path=json_path,
            csv_path=csv_paths.get("gradient"),
        ),
        "perturbation": _render_from_spec(
            TABLE_SPECS["stability_by_family_perturbation"],
            json_path=json_path,
            csv_path=csv_paths.get("perturbation"),
        ),
    }


# Alias — matches the conversational name "generalization" for the same view.
render_generalization = render_stability_by_family


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
    if rebuild:
        from src.evaluation.deployment_recommendation import (
            build_deployment_recommendation,
        )
        build_deployment_recommendation()
    return _render_from_spec(TABLE_SPECS["deployment_recommendation"],
                             json_path=json_path, csv_path=csv_path)


def render_table(name: str, json_path=None, csv_path=None, **kwargs):
    """Dispatcher — render any registered table by name.

    Names: 'qat_drift_decomposition', 'large_sample_stability',
    'stability_by_family' (alias 'generalization'),
    'deployment_recommendation'.
    """
    if name == "qat_drift_decomposition":
        return render_qat_drift_decomposition(json_path=json_path, csv_path=csv_path)
    if name == "large_sample_stability":
        return render_large_sample_stability(json_path=json_path, csv_path=csv_path)
    if name in ("stability_by_family", "generalization"):
        return render_stability_by_family(json_path=json_path,
                                          csv_paths=kwargs.get("csv_paths"))
    if name == "deployment_recommendation":
        return render_deployment_recommendation(
            json_path=json_path, csv_path=csv_path,
            rebuild=kwargs.get("rebuild", True),
        )
    raise ValueError(
        f"Unknown table '{name}'. Registered: {sorted(TABLE_SPECS.keys())}"
    )


# ---------------------------------------------------------------------------
# Legacy helpers — kept verbatim for back-compat with scripts/run_ptq.py
# ---------------------------------------------------------------------------

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
