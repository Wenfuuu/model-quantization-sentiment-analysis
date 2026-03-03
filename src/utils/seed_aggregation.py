from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ResultDict = Dict[str, Any]

def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    arr_finite = arr[np.isfinite(arr)]
    n_finite = len(arr_finite)

    if n_finite == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "min":  float("nan"), "max": float("nan"),
            "median": float("nan"), "n": 0, "values": list(values),
        }

    return {
        "mean":   float(np.mean(arr_finite)),
        "std":    float(np.std(arr_finite, ddof=1) if n_finite > 1 else 0.0),
        "min":    float(np.min(arr_finite)),
        "max":    float(np.max(arr_finite)),
        "median": float(np.median(arr_finite)),
        "n":      n_finite,
        "values": [float(v) for v in values],
    }


def _collect_scalar_keys(results: List[ResultDict]) -> List[str]:
    if not results:
        return []
    candidate_keys = set(
        k for k, v in results[0].items() if _is_scalar(v)
    )
    for r in results[1:]:
        candidate_keys &= {k for k, v in r.items() if _is_scalar(v)}
    return sorted(candidate_keys)



def aggregate_seed_results(results: List[ResultDict]) -> Dict[str, Any]:
    if len(results) < 2:
        warnings.warn(
            f"aggregate_seed_results received only {len(results)} result(s). "
            "At least 3 seeds are required for valid variance estimation. "
            "Results will be computed but statistical tests will be unreliable.",
            UserWarning,
            stacklevel=2,
        )
    seeds = []
    for i, r in enumerate(results):
        if "seed" not in r:
            raise ValueError(
                f"Result at index {i} is missing the 'seed' key. "
                "Every fine-tuning result dict must record its seed for "
                "cross-seed aggregation to be traceable."
            )
        seeds.append(int(r["seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"Duplicate seeds detected: {seeds}. "
            "Each seed must correspond to an independent training run. "
            "Do not aggregate the same seed twice."
        )

    hp_warnings: List[str] = []
    hp_consistent = True

    if all("hyperparameters" in r for r in results):
        ref_hp = results[0]["hyperparameters"]
        for i, r in enumerate(results[1:], start=1):
            for k, ref_v in ref_hp.items():
                other_v = r["hyperparameters"].get(k)
                if other_v != ref_v:
                    msg = (
                        f"Hyperparameter mismatch at seed index {i}: "
                        f"'{k}' = {ref_v!r} (seed 0) vs {other_v!r} "
                        f"(seed {seeds[i]}). "
                        "This violates the identical-HP assumption."
                    )
                    hp_warnings.append(msg)
                    hp_consistent = False

    for w in hp_warnings:
        warnings.warn(w, UserWarning, stacklevel=2)

    scalar_keys = _collect_scalar_keys(results)

    metrics: Dict[str, Dict[str, float]] = {}
    for key in scalar_keys:
        values = [r[key] for r in results]
        metrics[key] = _stats(values)

    return {
        "seeds":   seeds,
        "n_seeds": len(seeds),
        "metrics": metrics,
        "hyperparameter_consistent": hp_consistent,
        "hyperparameter_warnings": hp_warnings,
        "raw_results": results,
    }


def aggregate_paired_differences(
    base_results: List[ResultDict],
    variant_results: List[ResultDict],
    base_label: str = "FP32",
    variant_label: str = "quantized",
) -> Dict[str, Any]:
    if len(base_results) != len(variant_results):
        warnings.warn(
            f"base_results ({len(base_results)}) and variant_results "
            f"({len(variant_results)}) have different lengths. "
            "Only matched seeds will be used for paired analysis.",
            UserWarning,
            stacklevel=2,
        )

    def _seed_of(r: ResultDict, idx: int) -> int:
        if "seed" not in r:
            raise ValueError(
                f"Result at index {idx} is missing 'seed'. "
                "Cannot perform paired analysis without seed traceability."
            )
        return int(r["seed"])

    base_by_seed = {_seed_of(r, i): r for i, r in enumerate(base_results)}
    var_by_seed  = {_seed_of(r, i): r for i, r in enumerate(variant_results)}
    common_seeds = sorted(set(base_by_seed) & set(var_by_seed))

    missing_in_base    = sorted(set(var_by_seed)  - set(base_by_seed))
    missing_in_variant = sorted(set(base_by_seed) - set(var_by_seed))
    if missing_in_base:
        warnings.warn(
            f"Seeds {missing_in_base} are in variant but missing from base. "
            "Excluded from paired analysis.",
            UserWarning, stacklevel=2,
        )
    if missing_in_variant:
        warnings.warn(
            f"Seeds {missing_in_variant} are in base but missing from variant. "
            "Excluded from paired analysis.",
            UserWarning, stacklevel=2,
        )

    if len(common_seeds) < 2:
        raise ValueError(
            f"Only {len(common_seeds)} common seed(s) found between base and "
            "variant.  At least 2 paired observations are required for any "
            "difference estimate; at least 3 for valid variance estimation."
        )

    paired_base = [base_by_seed[s] for s in common_seeds]
    paired_var  = [var_by_seed[s]  for s in common_seeds]

    all_results = paired_base + paired_var
    scalar_keys = _collect_scalar_keys(all_results)

    diff_by_key: Dict[str, List[float]] = {k: [] for k in scalar_keys}
    for b, v in zip(paired_base, paired_var):
        for k in scalar_keys:
            diff_by_key[k].append(float(v[k]) - float(b[k]))

    differences = {k: _stats(diffs) for k, diffs in diff_by_key.items()}

    direction_consistency = {
        k: float(np.mean([d > 0 for d in diffs]))
        for k, diffs in diff_by_key.items()
    }

    return {
        "base_label":     base_label,
        "variant_label":  variant_label,
        "seeds":          common_seeds,
        "n_pairs":        len(common_seeds),
        "base_metrics":   aggregate_seed_results(paired_base)["metrics"],
        "variant_metrics": aggregate_seed_results(paired_var)["metrics"],
        "differences":    differences,
        "direction_consistency": direction_consistency,
    }

def load_seed_results(
    seed_dirs: Sequence[Path],
    filename: str = "finetune_results.json",
) -> List[ResultDict]:
    results = []
    for d in seed_dirs:
        path = Path(d) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Result file not found: {path}\n"
                f"Expected one '{filename}' per seed directory. "
            )
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def save_aggregated_results(
    aggregated: Dict[str, Any],
    output_path: Path,
    *,
    exclude_raw: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = dict(aggregated)
    if exclude_raw and "raw_results" in data:
        data = {k: v for k, v in data.items() if k != "raw_results"}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
