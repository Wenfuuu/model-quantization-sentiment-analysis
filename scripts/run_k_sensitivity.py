from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import fp32_seed_dir, _tag_suffix
from src.evaluation.explanation_drift import top_k_jaccard

_ROOT       = Path(__file__).parent.parent
_OUT_DIR    = _ROOT / "results" / "attributions"
_RES_DIR    = _ROOT / "results"
_MODELS_DIR = _ROOT / "models"
_SUBSAMPLE  = _ROOT / "data" / "explainability_subsample_v2.csv"
_FP32_DIR   = fp32_seed_dir(42)

_VARIANTS = [
    "ptq_fp16", "ptq_int8", "ptq_int4",
    "qat_fp32", "qat_onnx_fp16", "qat_onnx_int8", "qat_onnx_int4",
]
_METHODS = ["lime", "occ", "shap"]
_DISPLAY = {
    "ptq_fp16":      "PTQ-FP16",
    "ptq_int8":      "PTQ-INT8",
    "ptq_int4":      "PTQ-INT4",
    "qat_fp32":      "QAT-FP32",
    "qat_onnx_fp16": "QAT-FP16",
    "qat_onnx_int8": "QAT-INT8",
    "qat_onnx_int4": "QAT-INT4",
}

def _load_subsample() -> List[Tuple[int, str]]:
    import pandas as pd
    if not _SUBSAMPLE.exists():
        print(f"  [ERROR] Subsample not found: {_SUBSAMPLE}")
        return []
    df = pd.read_csv(_SUBSAMPLE)
    return [(int(row.sample_id), str(row.text)) for row in df.itertuples()]


def _expected_chance_jaccard(k: int, mean_len: float) -> float:
    n = mean_len
    if n <= 0 or k >= 2 * n:
        return float("nan")
    return round(k / (2 * n - k), 4)

def step2_coverage_diagnostic(
    k_values: List[int],
    samples: List[Tuple[int, str]],
) -> dict:
    print("\n[STEP 2] Token coverage diagnostic")

    seq_lens = [len(text.split()) for _, text in samples]
    mean_len = float(np.mean(seq_lens)) if seq_lens else float("nan")

    coverage_at_k: Dict[str, float] = {}
    for k in k_values:
        cov = (k / mean_len) if (mean_len > 0 and not np.isnan(mean_len)) else float("nan")
        coverage_at_k[str(k)] = round(cov, 4)

    k5_cov = coverage_at_k.get("5")
    k5_pct = round(float(k5_cov) * 100, 1) if k5_cov is not None else None
    k5_ok  = bool(float(k5_cov) < 0.30) if k5_cov is not None else None

    diag = {
        "mean_seq_len":    round(mean_len, 2),
        "n_samples":       len(seq_lens),
        "coverage_at_k":   coverage_at_k,
        "k5_coverage_pct": k5_pct,
        "k5_acceptable":   k5_ok,
    }

    out = _RES_DIR / "k_coverage_diagnostic.json"
    out.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"  mean seq len = {mean_len:.1f} tokens over {len(seq_lens)} samples")
    for k in k_values:
        cov_pct = float(coverage_at_k[str(k)]) * 100
        flag    = "  [WARNING: trivially high — > 30 %]" if cov_pct > 30 else ""
        print(f"    k={k:2d}: coverage = {cov_pct:.1f} %{flag}")
    verdict = "ACCEPTABLE" if k5_ok else "WARNING (> 30 %)"
    print(f"  k=5 verdict: {verdict}")
    print(f"  -> {out}")
    return diag


def step3_jaccard_sweep(
    k_values: List[int],
    samples: List[Tuple[int, str]],
) -> Dict[str, Dict[int, float]]:
    print("\n[STEP 3a] Jaccard sweep (from .npy attribution files)")

    totals: Dict[str, Dict[int, List[float]]] = {
        v: {k: [] for k in k_values} for v in _VARIANTS
    }

    n_pairs_found = 0
    for method in _METHODS:
        for vname in _VARIANTS:
            for sid, text in samples:
                fp32_path = _OUT_DIR / f"{method}_fp32_{sid}.npy"
                var_path  = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
                if not fp32_path.exists() or not var_path.exists():
                    continue
                words  = text.split()
                fp32_s = np.load(fp32_path).astype(np.float64)
                var_s  = np.load(var_path).astype(np.float64)
                L = min(len(words), len(fp32_s), len(var_s))
                if L < 1:
                    continue
                wl = words[:L]
                fl = fp32_s[:L].tolist()
                vl = var_s[:L].tolist()
                for k in k_values:
                    j = top_k_jaccard(wl, fl, wl, vl, k=k)
                    totals[vname][k].append(j)
                n_pairs_found += 1
        print(f"  {method}: done")

    results: Dict[str, Dict[int, float]] = {}
    for vname in _VARIANTS:
        results[vname] = {}
        for k in k_values:
            vals = totals[vname][k]
            results[vname][k] = float(np.mean(vals)) if vals else float("nan")

    print(f"  Total (method×variant×sample) triples processed: {n_pairs_found}")
    print(f"\n  {'variant':22s}", " ".join(f"  k={k}" for k in k_values))
    for vname in _VARIANTS:
        row = results[vname]
        vals_str = "  ".join(
            f"{row[k]:.3f}" if not np.isnan(row[k]) else "  n/a"
            for k in k_values
        )
        print(f"  {_DISPLAY.get(vname, vname):22s}  {vals_str}")

    return results

def step3_eraser_sweep(
    k_values: List[int],
    samples: List[Tuple[int, str]],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    print("\n[STEP 3b] ERASER sweep (comp + suff; requires model inference)")

    import torch
    from src.models import ModelManager
    from src.models.base import BaseModel, OnnxBaseModel
    from src.quantization.ptq import PTQQuantizer
    from src.evaluation.faithfulness import FaithfulnessEvaluator, _OnnxTorchAdapter

    _QAT_CLEAN = _MODELS_DIR / f"qat_seed42_clean{_tag_suffix()}"

    if not _FP32_DIR.exists():
        print(f"  [ERROR] FP32 model not found at {_FP32_DIR} — skipping ERASER sweep")
        return {}

    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()
    print(f"  FP32 base loaded from {_FP32_DIR}")

    def _load_onnx(precision: str):
        import onnxruntime as ort
        onnx_file = _MODELS_DIR / f"qat_onnx_{precision}_seed42{_tag_suffix()}" / f"model_qat_{precision}.onnx"
        if not onnx_file.exists():
            return None
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxBaseModel(sess, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision: str):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        return BaseModel(m, fp32_base.tokenizer, device=fp32_base.device)

    _LOADERS = {
        "ptq_fp16":      lambda: _build_ptq("fp16"),
        "ptq_int8":      lambda: _build_ptq("int8"),
        "ptq_int4":      lambda: _build_ptq("int4"),
        "qat_fp32":      lambda: (ModelManager.load_model(str(_QAT_CLEAN))
                                  if _QAT_CLEAN.exists() else None),
        "qat_onnx_fp16": lambda: _load_onnx("fp16"),
        "qat_onnx_int8": lambda: _load_onnx("int8"),
        "qat_onnx_int4": lambda: _load_onnx("int4"),
    }

    k_tuple   = tuple(k_values)
    eraser_out: Dict[str, Dict[int, Dict[str, float]]] = {}

    for vname in _VARIANTS:
        model = _LOADERS[vname]()
        if model is None:
            print(f"  [SKIP] {_DISPLAY.get(vname, vname)} — checkpoint not found")
            continue

        if isinstance(model, OnnxBaseModel):
            infer_model = _OnnxTorchAdapter(model)
            device      = torch.device("cpu")
        else:
            infer_model = model.model
            device      = model.device

        evaluator = FaithfulnessEvaluator(
            infer_model, fp32_base.tokenizer, device=device, k_values=k_tuple,
        )

        comp_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        suff_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        n_done = 0

        for method in _METHODS:
            for sid, text in samples:
                npy = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
                if not npy.exists():
                    continue
                words  = text.split()
                scores = np.load(npy).astype(np.float64)
                L      = min(len(words), len(scores))
                if L < 1:
                    continue
                try:
                    res = evaluator.evaluate(
                        text, words[:L], scores[:L].tolist(),
                        method=method, precision=vname, token_level="word",
                    )
                    for k in k_values:
                        if k not in res.per_k:
                            continue
                        fk = res.per_k[k]
                        if not np.isnan(fk.comprehensiveness):
                            comp_acc[k].append(fk.comprehensiveness)
                        if not np.isnan(fk.sufficiency):
                            suff_acc[k].append(fk.sufficiency)
                    n_done += 1
                except Exception:
                    pass

        label = _DISPLAY.get(vname, vname)
        print(f"  {label:12s}: {n_done} evaluations")
        eraser_out[vname] = {
            k: {
                "comp": float(np.mean(comp_acc[k])) if comp_acc[k] else float("nan"),
                "suff": float(np.mean(suff_acc[k])) if suff_acc[k] else float("nan"),
            }
            for k in k_values
        }

    return eraser_out

def step4_plot(
    k_values: List[int],
    jaccard_results: Dict[str, Dict[int, float]],
    mean_len: float,
) -> None:
    print("\n[STEP 4] Generating plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = _RES_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors

    for i, vname in enumerate(_VARIANTS):
        data   = jaccard_results.get(vname, {})
        ks     = [k for k in k_values if not np.isnan(data.get(k, float("nan")))]
        vals   = [data[k] for k in ks]
        if not ks:
            continue
        label = _DISPLAY.get(vname, vname)
        ax.plot(ks, vals, marker="o", linewidth=1.8,
                color=colors[i % len(colors)], label=label)

    if not np.isnan(mean_len) and mean_len > 0:
        chance_ks   = [k for k in k_values if k < 2 * mean_len]
        chance_vals = [k / (2 * mean_len - k) for k in chance_ks]
        ax.plot(
            chance_ks, chance_vals,
            linestyle="--", color="gray", linewidth=1.2,
            label=f"chance (n≈{mean_len:.0f})",
        )

    ax.set_xlabel("k  (number of top-attributed tokens)", fontsize=11)
    ax.set_ylabel("Mean Jaccard similarity", fontsize=11)
    ax.set_title(
        "Top-$k$ Jaccard sensitivity to $k$\n"
        "(FP32 vs. quantized variant; averaged across methods and samples)",
        fontsize=11,
    )
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = fig_dir / f"k_sensitivity_plot.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {p}")

    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="k-sensitivity sweep for top-k Jaccard and ERASER faithfulness metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="3,5,10,15",
        metavar="INT,INT,...",
        help="Comma-separated k values to sweep",
    )
    parser.add_argument(
        "--skip_eraser",
        action="store_true",
        help="Skip ERASER comp/suff metrics (only compute Jaccard, no model inference needed)",
    )
    args = parser.parse_args()

    k_values: List[int] = sorted(
        {int(k.strip()) for k in args.k_values.split(",") if k.strip()}
    )
    print(f"\n  k-sensitivity sweep  |  k = {k_values}")
    print(f"  ERASER metrics: {'disabled (--skip_eraser)' if args.skip_eraser else 'enabled'}")

    samples = _load_subsample()
    if not samples:
        sys.exit(1)
    print(f"  Subsample: {len(samples)} samples ({_SUBSAMPLE.name})")

    diag     = step2_coverage_diagnostic(k_values, samples)
    mean_len = float(diag["mean_seq_len"])

    jaccard_results = step3_jaccard_sweep(k_values, samples)

    eraser_results: Dict[str, Dict[int, Dict[str, float]]] = {}
    if not args.skip_eraser:
        try:
            eraser_results = step3_eraser_sweep(k_values, samples)
        except Exception as exc:
            print(f"  [WARN] ERASER sweep failed: {exc}")
            print("  Re-run with --skip_eraser to produce Jaccard-only output.")

    sensitivity: dict = {
        "k_values":              k_values,
        "chance_jaccard_at_k":   {
            str(k): _expected_chance_jaccard(k, mean_len) for k in k_values
        },
        "variants": {},
    }

    for vname in _VARIANTS:
        label  = _DISPLAY.get(vname, vname)
        jac    = jaccard_results.get(vname, {})
        era    = eraser_results.get(vname, {})
        per_k: dict = {}
        for k in k_values:
            entry: dict = {
                "jaccard": round(float(jac.get(k, float("nan"))), 4),
            }
            if era and k in era:
                entry["comp"] = round(float(era[k].get("comp", float("nan"))), 4)
                entry["suff"] = round(float(era[k].get("suff", float("nan"))), 4)
            per_k[str(k)] = entry
        sensitivity["variants"][label] = per_k

    json_path = _RES_DIR / "k_sensitivity.json"
    json_path.write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    print(f"\n  k_sensitivity.json -> {json_path}")

    step4_plot(k_values, jaccard_results, mean_len)

    k5_ok = diag.get("k5_acceptable")
    if k5_ok is True:
        print("  k=5 coverage < 30 % → k=5 is an ACCEPTABLE choice for this corpus.")
    elif k5_ok is False:
        print("  k=5 coverage >= 30 % → CONSIDER raising k or using a relative threshold.")
    else:
        print("  k=5 coverage: could not evaluate (no subsample data).")
    print("  Done.\n")


if __name__ == "__main__":
    main()
