from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.seed_aggregation import (
    aggregate_seed_results,
    load_seed_results,
    save_aggregated_results,
)

DEFAULT_SEEDS = [42, 123, 7]

_CKPT_TEMPLATE = "indobert-fp32-smsa-3label-seed{seed}"

_AGG_OUTPUT_DIR  = _PROJECT_ROOT / "outputs" / "multi-seed"
_AGG_OUTPUT_FILE = _AGG_OUTPUT_DIR / "aggregated_finetune_results.json"

_FINETUNE_SCRIPT = Path(__file__).parent / "finetune_smsa_fp32.py"

def seed_checkpoint_dir(seed: int) -> Path:
    return _PROJECT_ROOT / "finetuned-model" / _CKPT_TEMPLATE.format(seed=seed)


def run_single_seed(
    seed: int,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    skip_if_exists: bool = True,
) -> Path:
    ckpt_dir = seed_checkpoint_dir(seed)
    result_file = ckpt_dir / "finetune_results.json"

    if skip_if_exists and result_file.exists():
        print(f"  [skip] Seed {seed}: results already exist at {ckpt_dir}")
        return ckpt_dir

    print(f"\n{'='*60}")
    print(f"  Running fine-tuning: seed={seed}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(_FINETUNE_SCRIPT),
        "--seed",       str(seed),
        "--epochs",     str(epochs),
        "--lr",         str(lr),
        "--batch-size", str(batch_size),
        "--save-dir",   str(ckpt_dir),
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        warnings.warn(
            f"Fine-tuning for seed={seed} exited with code "
            f"{result.returncode}.  Check output above.  This seed will be "
            "excluded from aggregation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ckpt_dir

    return ckpt_dir


def write_aggregated_summary(
    seeds: list[int],
    agg: dict,
    output_path: Path,
) -> None:
    enriched = dict(agg)
    enriched["provenance"] = {
        "script":         str(_FINETUNE_SCRIPT.relative_to(_PROJECT_ROOT)),
        "aggregated_by":  str(Path(__file__).relative_to(_PROJECT_ROOT)),
        "ckpt_dirs":      {s: str(seed_checkpoint_dir(s)) for s in seeds},
    }

    save_aggregated_results(enriched, output_path, exclude_raw=False)
    print(f"\n  Aggregated results saved → {output_path}")


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 60)
    print("  MULTI-SEED FINE-TUNING SUMMARY")
    print(f"  Seeds: {agg['seeds']}  |  N = {agg['n_seeds']}")
    print("=" * 60)

    if not agg.get("hyperparameter_consistent", True):
        print("\nHYPERPARAMETER INCONSISTENCY DETECTED:")
        for w in agg.get("hyperparameter_warnings", []):
            print(f"     {w}")

    print("\n  Metric                     mean ± std (across seeds)")
    print("  " + "-" * 50)

    priority_keys = ["test_macro_f1", "test_accuracy", "best_val_macro_f1"]
    metrics = agg.get("metrics", {})

    for key in priority_keys:
        if key in metrics:
            m = metrics[key]
            per_seed_str = "  ".join(f"seed={s}: {v:.4f}"
                                     for s, v in zip(agg["seeds"], m["values"]))
            print(f"  {key:<26} {m['mean']:.4f} ± {m['std']:.4f}")
            print(f"     ({per_seed_str})")

    print()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run IndoBERT fine-tuning for multiple seeds and aggregate "
                    "results.  All seeds use identical hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="List of integer seeds.  At least 3 are recommended for valid "
             "variance estimation.",
    )
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument(
        "--no-skip", action="store_true", default=False,
        help="Re-train even if a checkpoint already exists for a seed.  "
             "Use this only when you want to overwrite existing results.",
    )
    p.add_argument(
        "--agg-output", type=str, default=str(_AGG_OUTPUT_FILE),
        help="Path for the aggregated summary JSON.",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    seeds = args.seeds

    if len(seeds) < 3:
        warnings.warn(
            f"Only {len(seeds)} seed(s) requested.  Publication-grade variance "
            "estimation requires at least 3 independent training runs.  "
            "Proceeding, but statistical tests on these results will be "
            "underpowered.",
            UserWarning,
        )

    print(f"\nHyperparameters (identical across all seeds):")
    print(f"  epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}")
    print(f"\nSeeds to run: {seeds}\n")

    successful_seeds = []
    for seed in seeds:
        ckpt_dir = run_single_seed(
            seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            skip_if_exists=not args.no_skip,
        )
        result_file = ckpt_dir / "finetune_results.json"
        if result_file.exists():
            successful_seeds.append(seed)
        else:
            print(f"  [warn] No result file for seed={seed}; skipping aggregation.")

    if len(successful_seeds) < 2:
        print(
            f"\n[ERROR] Only {len(successful_seeds)} seed(s) completed "
            "successfully.  Cannot aggregate."
        )
        sys.exit(1)

    seed_dirs = [seed_checkpoint_dir(s) for s in successful_seeds]

    print(f"\nLoading results for seeds: {successful_seeds}")
    per_seed_results = load_seed_results(seed_dirs, filename="finetune_results.json")

    print("Aggregating...")
    agg = aggregate_seed_results(per_seed_results)

    print_summary(agg)

    write_aggregated_summary(successful_seeds, agg, Path(args.agg_output))

    for s in successful_seeds:
        print(f"    seed={s:4d} → {seed_checkpoint_dir(s)}")
    print()


if __name__ == "__main__":
    main(parse_args())
