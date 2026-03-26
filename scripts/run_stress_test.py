import json
import sys
import torch
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENT_CONFIGS, LABELS, DEVICE, TRAINING_SEEDS, SEEDED_MODEL_DIRS
from scipy.stats import binom as _binom
from src.data import load_smsa_dataset, load_tweets_dataset
from src.models import ModelManager
from src.quantization.ptq import PTQQuantizer
from src.models.base import BaseModel
from src.evaluation import ModelEvaluator
from src.evaluation.stress_test import (
    run_edge_case_test,
    run_noise_robustness_test,
    run_calibration_stress_test,
)
from src.utils import print_section

sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))
from linguistic_probes import (
    get_all_probes,
    get_probe_samples,
    probe_accuracy_by_phenomenon,
)

warnings.filterwarnings("ignore")

ALL_TESTS = ["edge_case", "noise_robustness", "calibration", "linguistic_probes"]
TEST_LABELS = {
    "edge_case": "Linguistic Edge Cases",
    "noise_robustness": "Input Noise Robustness",
    "calibration": "Calibration Under Stress (ECE)",
    "linguistic_probes": "Linguistic Probe Accuracy (PSR)",
}


def _build_models(base_model):
    print_section("PREPARING QUANTIZED MODELS")
    models = {}
    use_fp16_map = {}

    print("  Building FP32 (baseline)...")
    models["fp32"] = base_model
    use_fp16_map["fp32"] = False

    print("  Building FP16...")
    ptq = PTQQuantizer(base_model.model)
    model_fp16, t = ptq.quantize_fp16()
    models["fp16"] = BaseModel(model_fp16, base_model.tokenizer, device=torch.device("cpu"))
    use_fp16_map["fp16"] = True
    print(f"    FP16 quantized in {t:.2f}s")

    print("  Building INT8...")
    model_int8, t = ptq.quantize_int8()
    models["int8"] = BaseModel(model_int8, base_model.tokenizer, device=torch.device("cpu"))
    use_fp16_map["int8"] = False
    print(f"    INT8 quantized in {t:.2f}s")

    print("  Building INT4...")
    model_int4, t = ptq.quantize_int4()
    models["int4"] = BaseModel(model_int4, base_model.tokenizer, device=torch.device("cpu"))
    use_fp16_map["int4"] = False
    print(f"    INT4 quantized in {t:.2f}s")

    print("  All models ready.\n")
    return models, use_fp16_map


def _run_quick_evaluation(models, test_samples, use_fp16_map):
    print_section("QUICK EVALUATION (collecting predictions)")
    all_predictions = {}

    for precision, model in models.items():
        use_fp16 = use_fp16_map.get(precision, False)
        preds = []
        correct = 0

        for sample in test_samples:
            pred = model.predict(sample["text"], use_fp16=use_fp16)
            preds.append({
                "text": sample["text"],
                "expected": sample["expected"],
                "predicted": pred["label"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
            })
            if pred["label"] == sample["expected"]:
                correct += 1

        acc = correct / len(test_samples)
        all_predictions[precision] = preds
        print(f"  {precision.upper()}: {acc*100:.2f}% accuracy ({correct}/{len(test_samples)})")

    return all_predictions


def run_stress_test_experiment(version_key, tests=None):
    if tests is None:
        tests = ALL_TESTS

    config = EXPERIMENT_CONFIGS[version_key]
    output_dir = Path(config["output_dir"]) / "stress_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section(f"STRESS TEST: {version_key}")
    print(f"Model: {config['model_id']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Tests: {', '.join(tests)}")
    print(f"Output: {output_dir}")

    print(f"\nLoading model: {config['model_id']}")
    base_model = ModelManager.load_model(config["model_id"], device=torch.device("cpu"))
    total_params, _ = base_model.count_parameters()
    print(f"Parameters: {total_params:,}")

    models, use_fp16_map = _build_models(base_model)

    if config["dataset"] == "smsa":
        test_samples = load_smsa_dataset()
    else:
        test_samples = load_tweets_dataset()
    print(f"Test samples: {len(test_samples)}")

    all_predictions = None
    if "calibration" in tests:
        all_predictions = _run_quick_evaluation(models, test_samples, use_fp16_map)

    all_results = {}

    if "edge_case" in tests:
        print_section("STRESS TEST 1: LINGUISTIC EDGE CASES")
        result = run_edge_case_test(models, output_dir, use_fp16_map)
        all_results["edge_case"] = result

        for p in models:
            overall = result["summary"][p]["_overall"]
            print(f"  {p.upper()}: {overall['accuracy']*100:.1f}% ({overall['correct']}/{overall['total']})")

    if "noise_robustness" in tests:
        print_section("STRESS TEST 2: INPUT NOISE ROBUSTNESS")
        result = run_noise_robustness_test(
            models, test_samples, output_dir,
            max_samples=50, use_fp16_map=use_fp16_map,
        )
        all_results["noise_robustness"] = result

        for noise_type in result:
            print(f"\n  {noise_type.replace('_', ' ').title()}:")
            for p in models:
                levels = result[noise_type][p]
                baseline = levels[0.0]["accuracy"] * 100
                worst = min(v["accuracy"] for v in levels.values()) * 100
                print(f"    {p.upper()}: {baseline:.1f}% → {worst:.1f}% (Δ{worst - baseline:+.1f}%)")

    if "calibration" in tests:
        print_section("STRESS TEST 3: CALIBRATION (ECE)")
        result = run_calibration_stress_test(all_predictions, output_dir)
        all_results["calibration"] = result

        for p in models:
            r = result[p]
            print(f"  {p.upper()}: ECE={r['ece']:.4f}  MCE={r['mce']:.4f}"
                  f"  overconf={r['overconfidence_rate']:.4f}"
                  f"  gap={r['confidence_accuracy_gap']:+.4f}")

    if "linguistic_probes" in tests:
        print_section("STRESS TEST 4: LINGUISTIC PROBES (PSR)")
        probes = get_all_probes()
        probe_samples = get_probe_samples(include_minimal_pairs=True)
        primary_samples = get_probe_samples(include_minimal_pairs=False)

        _all_seed_results = {}
        _rows_pred, _rows_psr = [], []

        for _seed in TRAINING_SEEDS:
            _seed_base = ModelManager.load_model(str(SEEDED_MODEL_DIRS[_seed]), device=torch.device("cpu"))
            _seed_models, _seed_fp16_map = _build_models(_seed_base)

            probe_results = {}
            for precision, model in _seed_models.items():
                use_fp16 = _seed_fp16_map.get(precision, False)
                preds_primary = []
                preds_pairs = []

                for i, sample in enumerate(primary_samples):
                    pred = model.predict(sample["text"], use_fp16=use_fp16)
                    preds_primary.append({
                        "predicted": pred["label"],
                        "expected": sample["expected"],
                        "confidence": pred["confidence"],
                        "phenomenon": sample["meta"]["phenomenon"],
                        "phenomenon_tokens": sample["meta"]["phenomenon_tokens"],
                        "text": sample["text"],
                    })

                direction_aware = []
                for probe, pred in zip(probes, preds_primary):
                    if probe.minimal_pair is None:
                        direction_aware.append(None)
                        continue
                    pair_pred = model.predict(probe.minimal_pair, use_fp16=use_fp16)
                    original_label = pred["predicted"]
                    pair_label = pair_pred["label"]
                    flipped = original_label != pair_label
                    correct_direction = pair_label == (probe.expected_direction or probe.expected_label)
                    direction_aware.append({
                        "original_pred": original_label,
                        "pair_pred": pair_label,
                        "flipped": flipped,
                        "correct_direction": correct_direction,
                        "minimal_pair_text": probe.minimal_pair,
                    })

                by_phenomenon = probe_accuracy_by_phenomenon(preds_primary, probes)

                dir_psr = {}
                for phenom in by_phenomenon:
                    entries = [
                        d for d, p in zip(direction_aware, probes)
                        if p.phenomenon == phenom and d is not None
                    ]
                    if entries:
                        n_flipped_correct = sum(1 for e in entries if e["correct_direction"])
                        dir_psr[phenom] = {
                            "n_with_pair": len(entries),
                            "n_correct_direction": n_flipped_correct,
                            "direction_psr": n_flipped_correct / len(entries),
                        }

                _all_dir = [d for d in direction_aware if d is not None]
                _n_correct_dir = sum(1 for d in _all_dir if d["correct_direction"])
                _n_probes = len(probes)
                _g_psr = _n_correct_dir / _n_probes if _n_probes else float("nan")
                _ci = _binom.interval(0.95, _n_probes, max(0.0, min(1.0, _g_psr)))
                _ci_low, _ci_high = _ci[0] / _n_probes, _ci[1] / _n_probes

                probe_results[precision] = {
                    "by_phenomenon": by_phenomenon,
                    "direction_aware_psr": dir_psr,
                    "raw_predictions": preds_primary,
                    "global_direction_psr": _g_psr,
                    "ci_low": _ci_low,
                    "ci_high": _ci_high,
                }
                _rows_psr.append({"seed": _seed, "precision": precision,
                                   "direction_psr": _g_psr, "ci_low": _ci_low, "ci_high": _ci_high})
                for _pr in preds_primary:
                    _rows_pred.append({**_pr, "seed": _seed, "precision": precision})

            phenomena = list(probe_results[list(_seed_models.keys())[0]]["by_phenomenon"].keys())
            print(f"\n  [seed={_seed}]")
            for phenom in phenomena:
                print(f"\n  [{phenom}]")
                for p in _seed_models:
                    ph = probe_results[p]["by_phenomenon"].get(phenom, {})
                    acc = ph.get("accuracy", float("nan"))
                    total = ph.get("total", 0)
                    dir_ph = probe_results[p]["direction_aware_psr"].get(phenom, {})
                    dir_psr_val = dir_ph.get("direction_psr", float("nan"))
                    print(f"    {p.upper()}: acc={acc:.2f} ({ph.get('correct',0)}/{total})  "
                          f"dir-PSR={dir_psr_val:.2f}")

            probe_out = output_dir / f"linguistic_probe_results_seed{_seed}.json"
            with open(probe_out, "w", encoding="utf-8") as f:
                json.dump(probe_results, f, ensure_ascii=False, indent=2)
            print(f"\n  Saved: {probe_out}")
            _all_seed_results[_seed] = probe_results

        import pandas as pd
        _res_dir = Path(__file__).resolve().parent.parent / "results"
        _res_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_rows_pred).to_csv(_res_dir / "probe_predictions_allseeds.csv", index=False)
        pd.DataFrame(_rows_psr).to_csv(_res_dir / "probe_psr_perseed.csv", index=False)
        _df_psr = pd.DataFrame(_rows_psr)
        _summary = _df_psr.groupby("precision")["direction_psr"].agg(mean="mean", std="std").reset_index()
        _ci_means = _df_psr.groupby("precision")[["ci_low", "ci_high"]].mean().reset_index()
        _summary.merge(_ci_means, on="precision").to_csv(_res_dir / "probe_psr_summary.csv", index=False)
        print(f"\n  Saved probe CSVs to {_res_dir}")
        all_results["linguistic_probes"] = _all_seed_results

    print_section("STRESS TEST COMPLETED")
    print(f"All results saved to: {output_dir}")

    return all_results

def interactive_menu():
    print("\n" + "=" * 60)
    print("  STRESS TEST (ROBUSTNESS ANALYSIS)")
    print("=" * 60)

    print("\n  Select Model:")
    print("  [1] Original IndoBERT (indobenchmark/indobert-base-p2)")
    print("  [2] Finetuned IndoBERT (indobert-fp32-smsa-3label)")
    print("  [3] Both")

    model_choice = input("\n  Enter choice (1/2/3): ").strip()

    print("\n  Select Dataset:")
    print("  [1] SMSA (test.tsv)")
    print("  [2] Tweets (INA_TweetsPPKM)")
    print("  [3] Both")

    dataset_choice = input("\n  Enter choice (1/2/3): ").strip()

    print("\n  Select Tests to Run:")
    for i, test_name in enumerate(ALL_TESTS, 1):
        print(f"  [{i}] {TEST_LABELS[test_name]}")
    print(f"  [{len(ALL_TESTS) + 1}] All Tests")

    test_choice = input(f"\n  Enter choice (comma-separated, e.g. 1,2 or {len(ALL_TESTS) + 1} for all): ").strip()

    models_list = []
    if model_choice == "1":
        models_list = ["original"]
    elif model_choice == "2":
        models_list = ["finetuned"]
    else:
        models_list = ["original", "finetuned"]

    datasets_list = []
    if dataset_choice == "1":
        datasets_list = ["smsa"]
    elif dataset_choice == "2":
        datasets_list = ["tweets"]
    else:
        datasets_list = ["smsa", "tweets"]

    if test_choice == str(len(ALL_TESTS) + 1) or test_choice.upper() == "A":
        selected_tests = ALL_TESTS[:]
    else:
        indices = [int(x.strip()) - 1 for x in test_choice.split(",")]
        selected_tests = [ALL_TESTS[i] for i in indices if 0 <= i < len(ALL_TESTS)]
        if not selected_tests:
            selected_tests = ALL_TESTS[:]

    selected_experiments = []
    for m in models_list:
        for d in datasets_list:
            selected_experiments.append(f"{m}_{d}")

    return selected_experiments, selected_tests


def run_stress_test():
    selected_experiments, selected_tests = interactive_menu()

    total = len(selected_experiments)
    print("\n" + "=" * 80)
    print(f"STARTING STRESS TESTS - {total} EXPERIMENT(S)")
    print(f"Tests: {', '.join(TEST_LABELS[t] for t in selected_tests)}")
    print("=" * 80)
    for i, key in enumerate(selected_experiments, 1):
        print(f"  [{i}/{total}] {key}")
    print("=" * 80 + "\n")

    for idx, version_key in enumerate(selected_experiments, 1):
        print("\n" + "#" * 80)
        print(f"# STRESS TEST [{idx}/{total}]: {version_key.upper()}")
        print("#" * 80 + "\n")
        run_stress_test_experiment(version_key, tests=selected_tests)
        print(f"\n  Completed [{idx}/{total}]: {version_key}")


if __name__ == "__main__":
    run_stress_test()
