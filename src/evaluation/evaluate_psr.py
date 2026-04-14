from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binom as _binom

CANONICAL_8: List[str] = [
    "verbal_neg",
    "nominal_neg",
    "double_neg",
    "scope_neg",
    "scalar_intens",
    "reduplication",
    "contrastive",
    "epistemic_hedge",
]

_DISPLAY_NAMES: Dict[str, str] = {
    "fp32":     "FP32",
    "ptq_fp16": "PTQ-FP16",
    "ptq_int8": "PTQ-INT8",
    "ptq_int4": "PTQ-INT4",
    "qat_fp16": "QAT-FP16",
    "qat_int8": "QAT-INT8",
    "qat_int4": "QAT-INT4",
    "qat_ste":  "QAT-STE",
}

def _get_remap():
    _datasets_dir = Path(__file__).resolve().parent.parent.parent / "datasets"
    if str(_datasets_dir) not in sys.path:
        sys.path.insert(0, str(_datasets_dir))
    from linguistic_probes import remap_to_8_categories  # type: ignore
    return remap_to_8_categories


def verify_phenomenon_tags(probes: list) -> None:
    remap = _get_remap()
    missing = []
    for p in probes:
        remapped = remap(p)
        if remapped not in CANONICAL_8:
            missing.append({
                "text":        p.text[:80],
                "phenomenon":  getattr(p, "phenomenon", "<missing>"),
                "remapped_to": remapped,
            })

    if missing:
        print(f"\n[PSR WARN] {len(missing)} probe(s) have unknown/missing phenomenon tags:")
        for entry in missing:
            print(f"  text='{entry['text']}' | raw='{entry['phenomenon']}' | mapped='{entry['remapped_to']}'")
        raise ValueError(
            f"{len(missing)} probe(s) could not be remapped to CANONICAL_8. See warnings above."
        )

    print(f"  [OK] All {len(probes)} probes have valid phenomenon tags (8 categories).")

def compute_psr_matrix(
    models: Dict[str, object],
    probes: list,
    use_fp16_map: Optional[Dict[str, bool]] = None,
) -> Dict[Tuple[str, str], Dict]:
    remap = _get_remap()
    if use_fp16_map is None:
        use_fp16_map = {}

    raw_preds: Dict[str, list] = {}
    for variant, model in models.items():
        use_fp16 = use_fp16_map.get(variant, False)
        preds = []
        print(f"  [{variant}] running inference on {len(probes)} probes...")
        for probe in probes:
            result   = model.predict(probe.text, use_fp16=use_fp16)
            predicted = result["label"]
            phenom_8  = remap(probe)
            preds.append({
                "predicted": predicted,
                "expected":  probe.expected_label,
                "correct":   int(predicted == probe.expected_label),
                "phenom_8":  phenom_8,
            })
        raw_preds[variant] = preds

    matrix: Dict[Tuple[str, str], Dict] = {}
    for variant, preds in raw_preds.items():
        accum: Dict[str, Dict] = {ph: {"n_pairs": 0, "n_changed": 0} for ph in CANONICAL_8}
        for pred in preds:
            ph = pred["phenom_8"]
            if ph in accum:
                accum[ph]["n_pairs"]   += 1
                accum[ph]["n_changed"] += pred["correct"]
        for phenom, counts in accum.items():
            n, nc = counts["n_pairs"], counts["n_changed"]
            matrix[(variant, phenom)] = {
                "psr":       round(nc / n, 4) if n > 0 else float("nan"),
                "n_pairs":   n,
                "n_changed": nc,
            }

    matrix["_raw_preds"] = raw_preds
    return matrix

def run_mcnemar_tests(
    matrix: Dict,
    probes: list,
    phenomena: List[str],
    non_fp32_variants: List[str],
    fp32_key: str = "fp32",
) -> Tuple[Dict, List[Dict]]:
    from statsmodels.stats.multitest import multipletests

    raw_preds: Dict[str, list] = matrix.pop("_raw_preds", {})

    def _by_phenom(preds: list) -> Dict[str, list]:
        d: Dict[str, list] = {ph: [] for ph in phenomena}
        for pred in preds:
            ph = pred["phenom_8"]
            if ph in d:
                d[ph].append(pred["correct"])
        return d

    fp32_by_phenom = _by_phenom(raw_preds.get(fp32_key, []))

    mcnemar_rows: list = []
    low_power_cells: list = []

    for variant in non_fp32_variants:
        var_by_phenom = _by_phenom(raw_preds.get(variant, []))

        for phenom in phenomena:
            fp_c = np.array(fp32_by_phenom[phenom], dtype=int)
            vr_c = np.array(var_by_phenom[phenom], dtype=int)
            n = min(len(fp_c), len(vr_c))

            if n < 4:
                mcnemar_rows.append({
                    "variant": variant, "phenomenon": phenom,
                    "p_mcnemar": float("nan"), "p_bonferroni": float("nan"),
                    "significant": False, "n_discordant": 0, "low_power": True,
                    "n00": 0, "n01": 0, "n10": 0, "n11": 0,
                })
                low_power_cells.append({"variant": variant, "phenomenon": phenom, "n_discordant": 0})
                continue

            fp_c, vr_c = fp_c[:n], vr_c[:n]
            n00 = int(((fp_c == 0) & (vr_c == 0)).sum())
            n01 = int(((fp_c == 0) & (vr_c == 1)).sum())
            n10 = int(((fp_c == 1) & (vr_c == 0)).sum())
            n11 = int(((fp_c == 1) & (vr_c == 1)).sum())
            n_disc = n01 + n10

            p_mc = float(_binom.sf(n10 - 1, n_disc, 0.5)) if n_disc > 0 else 1.0
            low_power = n_disc < 10

            mcnemar_rows.append({
                "variant": variant, "phenomenon": phenom,
                "p_mcnemar": round(p_mc, 6), "p_bonferroni": float("nan"),
                "significant": False, "n_discordant": n_disc, "low_power": low_power,
                "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            })
            if low_power:
                low_power_cells.append({
                    "variant": variant, "phenomenon": phenom, "n_discordant": n_disc,
                })

    raw_ps = [
        r["p_mcnemar"] if not (isinstance(r["p_mcnemar"], float) and np.isnan(r["p_mcnemar"])) else 1.0
        for r in mcnemar_rows
    ]
    if raw_ps:
        _, p_bonf_arr, _, _ = multipletests(raw_ps, alpha=0.05, method="bonferroni")
        for i, row in enumerate(mcnemar_rows):
            row["p_bonferroni"] = round(float(p_bonf_arr[i]), 6)
            row["significant"]  = bool(p_bonf_arr[i] < 0.05)

    for row in mcnemar_rows:
        v, ph = row["variant"], row["phenomenon"]
        cell = matrix.get((v, ph), {})
        cell.update({
            "p_mcnemar":    row["p_mcnemar"],
            "p_bonferroni": row["p_bonferroni"],
            "significant":  row["significant"],
            "n_discordant": row["n_discordant"],
            "low_power":    row["low_power"],
        })
        matrix[(v, ph)] = cell

    return matrix, low_power_cells

def _write_psr_json(
    matrix: Dict,
    phenomena: List[str],
    variants: List[str],
    results_dir: Path,
    n_probes_total: int,
) -> None:
    display_matrix: Dict[str, Dict] = {}
    for variant in variants:
        dname = _DISPLAY_NAMES.get(variant, variant)
        display_matrix[dname] = {}
        for phenom in phenomena:
            cell = matrix.get((variant, phenom), {})
            display_matrix[dname][phenom] = {k: v for k, v in cell.items()}

    non_fp32 = [v for v in variants if v != "fp32"]
    all_changed = sum(matrix.get((v, ph), {}).get("n_changed", 0) for v in non_fp32 for ph in phenomena)
    all_pairs   = sum(matrix.get((v, ph), {}).get("n_pairs",   0) for v in non_fp32 for ph in phenomena)
    agg_psr = round(all_changed / all_pairs, 4) if all_pairs > 0 else float("nan")

    out = {
        "matrix":        display_matrix,
        "aggregate_psr": agg_psr,
        "n_probes_total": n_probes_total,
    }
    path = results_dir / "psr_per_phenomenon.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")

def _write_psr_csv(
    matrix: Dict,
    phenomena: List[str],
    variants: List[str],
    results_dir: Path,
) -> None:
    path = results_dir / "psr_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + phenomena)
        for variant in variants:
            dname = _DISPLAY_NAMES.get(variant, variant)
            row = [dname]
            for phenom in phenomena:
                cell = matrix.get((variant, phenom), {})
                psr  = cell.get("psr", float("nan"))
                sig  = cell.get("significant", False)
                if isinstance(psr, float) and np.isnan(psr):
                    row.append("n/a")
                else:
                    row.append(f"{psr:.3f}{'*' if sig else ''}")
            writer.writerow(row)
    print(f"  Saved: {path}")


def _write_low_power_warnings(low_power_cells: List[Dict], results_dir: Path) -> None:
    path = results_dir / "psr_low_power_warnings.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# PSR Low-Power Cells (n_discordant < 10)\n")
        f.write("# These cells have too few discordant pairs for reliable McNemar testing.\n")
        f.write("# Interpret significance flags for these cells with caution.\n\n")
        if not low_power_cells:
            f.write("(none)\n")
        else:
            for cell in low_power_cells:
                f.write(
                    f"variant={cell['variant']!r:14s}  phenomenon={cell['phenomenon']!r:18s}"
                    f"  n_discordant={cell['n_discordant']}\n"
                )
    print(f"  Saved: {path} ({len(low_power_cells)} low-power cell(s))")


def _print_summary_table(
    matrix: Dict, phenomena: List[str], variants: List[str]
) -> None:
    col_w  = max(len(p) for p in phenomena) + 2
    var_w  = max(len(_DISPLAY_NAMES.get(v, v)) for v in variants) + 2
    header = f"  {'variant':{var_w}}" + "".join(f" {p:>{col_w}}" for p in phenomena)
    print(f"\n  PSR per phenomenon (* = Bonferroni-significant):")
    print(header)
    for variant in variants:
        dname   = _DISPLAY_NAMES.get(variant, variant)
        row_str = f"  {dname:{var_w}}"
        for ph in phenomena:
            cell = matrix.get((variant, ph), {})
            psr  = cell.get("psr", float("nan"))
            sig  = "*" if cell.get("significant", False) else " "
            if isinstance(psr, float) and np.isnan(psr):
                val = "  n/a "
            else:
                val = f"{psr:.3f}{sig}"
            row_str += f" {val:>{col_w}}"
        print(row_str)

def run_psr_evaluation(
    models: Dict[str, object],
    probes: list,
    results_dir: Path,
    use_fp16_map: Optional[Dict[str, bool]] = None,
    fp32_key: str = "fp32",
) -> Dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PSR] STEP 1 — Verifying phenomenon tags...")
    verify_phenomenon_tags(probes)

    print("\n[PSR] STEP 2 — Running inference + computing PSR matrix...")
    matrix = compute_psr_matrix(models, probes, use_fp16_map=use_fp16_map)

    phenomena  = CANONICAL_8
    variants   = list(models.keys())
    non_fp32   = [v for v in variants if v != fp32_key]

    print(f"\n[PSR] STEP 3 — McNemar tests ({len(non_fp32)} variants × {len(phenomena)} phenomena)"
          f" + Bonferroni correction ({len(non_fp32) * len(phenomena)} comparisons)...")
    matrix, low_power_cells = run_mcnemar_tests(
        matrix, probes, phenomena, non_fp32, fp32_key=fp32_key
    )

    print("\n[PSR] STEP 4 — Writing output files...")
    _write_psr_json(matrix, phenomena, variants, results_dir, n_probes_total=len(probes))
    _write_psr_csv(matrix, phenomena, variants, results_dir)
    _write_low_power_warnings(low_power_cells, results_dir)

    _print_summary_table(matrix, phenomena, variants)

    all_changed = sum(matrix.get((v, ph), {}).get("n_changed", 0) for v in non_fp32 for ph in phenomena)
    all_pairs   = sum(matrix.get((v, ph), {}).get("n_pairs",   0) for v in non_fp32 for ph in phenomena)
    agg_psr = round(all_changed / all_pairs, 4) if all_pairs > 0 else float("nan")
    print(f"\n  Aggregate PSR across all variants/phenomena: {agg_psr:.4f}")

    return {
        "matrix":          matrix,
        "low_power_cells": low_power_cells,
        "aggregate_psr":   agg_psr,
    }
