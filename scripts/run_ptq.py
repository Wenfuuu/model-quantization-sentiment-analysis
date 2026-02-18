import sys
import torch
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENT_CONFIGS, LABELS
from src.data import load_smsa_dataset, load_tweets_dataset
from src.models import ModelManager
from src.quantization.ptq import PTQQuantizer
from src.evaluation import ModelEvaluator
from src.visualization import QuantizationPlotter, generate_comparison_report, generate_prediction_comparison
from src.quantization.utils import save_quantized_model, get_model_size
from src.utils import print_section
from src.models.base import BaseModel
from src.evaluation.metrics import statistical_test, confidence_comparison
from src.config import DEVICE

warnings.filterwarnings('ignore')


def run_ptq_experiment(version_key, num_runs_override=None):
    config = EXPERIMENT_CONFIGS[version_key]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_runs_override is not None:
        num_runs = num_runs_override
        warmup = 0 if num_runs_override == 0 else config["warmup_runs"]
    else:
        num_runs = config["num_inference_runs"]
        warmup = config["warmup_runs"]
    
    if num_runs == 0:
        num_runs = 1
        warmup = 0
    
    print_section(f"EXPERIMENT: {version_key}")
    print(f"Model: {config['model_id']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Output: {output_dir}")
    
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nLoading model: {config['model_id']}")
    base_model = ModelManager.load_model(config['model_id'])
    
    total_params, trainable_params = base_model.count_parameters()
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    if config["dataset"] == "smsa":
        test_samples = load_smsa_dataset()
    else:
        test_samples = load_tweets_dataset()
    
    print(f"\nPrepared {len(test_samples)} test samples")
    
    print_section("BASELINE EVALUATION (FP32 Model)")
    
    evaluator = ModelEvaluator(base_model)
    fp32_results = evaluator.evaluate(test_samples, num_runs=num_runs, warmup=warmup)
    
    print(f"\nAccuracy: {fp32_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence: {fp32_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency: {fp32_results['latency_stats']['mean']*1000:.2f} ms")
    
    fp32_path = output_dir / "model_fp32.pth"
    save_quantized_model(base_model.model, fp32_path)
    fp32_size_mb = get_model_size(fp32_path)
    print(f"\nFP32 model saved: {fp32_path} ({fp32_size_mb:.2f} MB)")
    
    print_section("HALF-PRECISION CONVERSION (FP16)")
    
    ptq_quantizer = PTQQuantizer(base_model.model)
    model_fp16, fp16_time = ptq_quantizer.quantize_fp16()
    print(f"FP16 Quantization completed in {fp16_time:.2f} seconds")
    
    fp16_path = output_dir / "model_fp16.pth"
    save_quantized_model(model_fp16, fp16_path)
    fp16_size_mb = get_model_size(fp16_path)
    print(f"FP16 model saved: {fp16_path} ({fp16_size_mb:.2f} MB)")
    print(f"Size Reduction: {(1 - fp16_size_mb/fp32_size_mb)*100:.2f}%")
    
    base_model_fp16 = BaseModel(model_fp16, base_model.tokenizer)
    evaluator_fp16 = ModelEvaluator(base_model_fp16)
    fp16_results = evaluator_fp16.evaluate(test_samples, num_runs=num_runs, warmup=warmup, use_fp16=True)
    
    print(f"\nAccuracy: {fp16_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence: {fp16_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency: {fp16_results['latency_stats']['mean']*1000:.2f} ms")
    
    print_section("DYNAMIC QUANTIZATION (INT8)")
    
    model_int8, int8_time = ptq_quantizer.quantize_int8()
    print(f"INT8 Quantization completed in {int8_time:.2f} seconds")
    
    int8_path = output_dir / "model_int8.pth"
    save_quantized_model(model_int8, int8_path)
    int8_size_mb = get_model_size(int8_path)
    print(f"INT8 model saved: {int8_path} ({int8_size_mb:.2f} MB)")
    print(f"Size Reduction: {(1 - int8_size_mb/fp32_size_mb)*100:.2f}%")
    
    base_model_int8 = BaseModel(model_int8, base_model.tokenizer, device=torch.device("cpu"))
    evaluator_int8 = ModelEvaluator(base_model_int8)
    int8_results = evaluator_int8.evaluate(test_samples, num_runs=num_runs, warmup=warmup)
    
    print(f"\nAccuracy: {int8_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence: {int8_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency: {int8_results['latency_stats']['mean']*1000:.2f} ms")
    
    print_section("INT4 QUANTIZATION")
    
    model_int4, int4_time = ptq_quantizer.quantize_int4()
    print(f"INT4 Quantization completed in {int4_time:.2f} seconds")
    
    int4_path = output_dir / "model_int4.pth"
    save_quantized_model(model_int4, int4_path)
    int4_size_mb = get_model_size(int4_path)
    print(f"INT4 model saved: {int4_path} ({int4_size_mb:.2f} MB)")
    print(f"Size Reduction: {(1 - int4_size_mb/fp32_size_mb)*100:.2f}%")
    
    base_model_int4 = BaseModel(model_int4, base_model.tokenizer)
    evaluator_int4 = ModelEvaluator(base_model_int4)
    int4_results = evaluator_int4.evaluate(test_samples, num_runs=num_runs, warmup=warmup)
    
    print(f"\nAccuracy: {int4_results['accuracy']*100:.2f}%")
    print(f"Avg Confidence: {int4_results['avg_confidence']*100:.2f}%")
    print(f"Mean Latency: {int4_results['latency_stats']['mean']*1000:.2f} ms")
    
    print_section("QUANTIZATION COMPARISON SUMMARY")
    
    df_comparison = generate_comparison_report(
        fp32_results, fp16_results, int8_results, int4_results,
        fp32_size_mb, fp16_size_mb, int8_size_mb, int4_size_mb
    )
    print(df_comparison.to_string(index=False))
    
    print_section("DETAILED PREDICTION COMPARISON")
    
    df_predictions, consistency = generate_prediction_comparison(
        fp32_results, fp16_results, int8_results, int4_results
    )
    print(df_predictions.to_string(index=False))
    
    print(f"\nFP16 vs FP32 Consistency: {consistency['fp16']*100:.1f}%")
    print(f"INT8 vs FP32 Consistency: {consistency['int8']*100:.1f}%")
    print(f"INT4 vs FP32 Consistency: {consistency['int4']*100:.1f}%")
    
    print_section("PREDICTION DIVERGENCES")
    
    divergences = []
    precisions_list = ["fp32", "fp16", "int8", "int4"]
    all_preds = {
        "fp32": fp32_results["predictions"],
        "fp16": fp16_results["predictions"],
        "int8": int8_results["predictions"],
        "int4": int4_results["predictions"]
    }
    
    for i in range(len(test_samples)):
        preds_by_precision = {}
        for p in precisions_list:
            preds_by_precision[p] = {
                "label": all_preds[p][i]["predicted"],
                "confidence": float(all_preds[p][i]["confidence"])
            }
        
        labels_set = set(preds_by_precision[p]["label"] for p in precisions_list)
        if len(labels_set) > 1:
            divergences.append({
                "sample_idx": i,
                "text": test_samples[i]["text"],
                "expected": test_samples[i]["expected"],
                "predictions": preds_by_precision
            })
    
    divergence_data = {
        "experiment": version_key,
        "total_samples": len(test_samples),
        "num_divergences": len(divergences),
        "divergences": divergences
    }
    
    divergence_path = output_dir / "prediction_divergences.json"
    with open(divergence_path, "w", encoding="utf-8") as f:
        json.dump(divergence_data, f, ensure_ascii=False, indent=2)
    
    print(f"Found {len(divergences)} divergent samples (out of {len(test_samples)} total)")
    print(f"Saved to: {divergence_path}")
    
    if divergences:
        for d in divergences[:10]:
            preds_str = "  ".join(f"{p.upper()}={d['predictions'][p]['label']}({d['predictions'][p]['confidence']*100:.1f}%)" for p in precisions_list)
            print(f"\n  Sample #{d['sample_idx']+1}: Expected={d['expected']}")
            print(f"    {preds_str}")
        if len(divergences) > 10:
            print(f"\n  ... and {len(divergences) - 10} more divergent samples")
    else:
        print("  All models agree on all predictions!")
    
    print_section("STATISTICAL ANALYSIS")
    
    int8_stats = statistical_test(fp32_results["latencies"], int8_results["latencies"])
    print(f"\nINT8 vs FP32 Latency (t-test)")
    print(f"  t-statistic: {int8_stats['t_statistic']:.4f}")
    print(f"  p-value: {int8_stats['p_value']:.6f}")
    print(f"  Cohen's d: {int8_stats['cohens_d']:.4f}")
    print(f"  Significant: {'Yes' if int8_stats['significant'] else 'No'}")
    
    int4_stats = statistical_test(fp32_results["latencies"], int4_results["latencies"])
    print(f"\nINT4 vs FP32 Latency (t-test)")
    print(f"  t-statistic: {int4_stats['t_statistic']:.4f}")
    print(f"  p-value: {int4_stats['p_value']:.6f}")
    print(f"  Cohen's d: {int4_stats['cohens_d']:.4f}")
    print(f"  Significant: {'Yes' if int4_stats['significant'] else 'No'}")
    
    conf_comp_int8 = confidence_comparison(fp32_results, int8_results)
    conf_comp_int4 = confidence_comparison(fp32_results, int4_results)
    
    print(f"\nConfidence Score Comparison")
    print(f"  INT8 vs FP32 p-value: {conf_comp_int8['p_value']:.6f}")
    print(f"  INT4 vs FP32 p-value: {conf_comp_int4['p_value']:.6f}")
    
    plotter = QuantizationPlotter(output_dir)
    chart_path = plotter.create_comprehensive_plot(
        {
            "fp32": fp32_results,
            "fp16": fp16_results,
            "int8": int8_results,
            "int4": int4_results
        },
        test_samples,
        version_key
    )
    print(f"\nVisualization saved to: {chart_path}")
    
    return {
        "version": version_key,
        "fp32_results": fp32_results,
        "fp16_results": fp16_results,
        "int8_results": int8_results,
        "int4_results": int4_results,
        "fp32_size_mb": fp32_size_mb,
        "fp16_size_mb": fp16_size_mb,
        "int8_size_mb": int8_size_mb,
        "int4_size_mb": int4_size_mb,
    }


def interactive_menu():
    print("\n" + "=" * 60)
    print("  PTQ QUANTIZATION EXPERIMENT RUNNER")
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

    num_runs_str = input("\n  Number of inference runs per sample (default 20, 0 = skip latency benchmark): ").strip()
    num_runs_input = int(num_runs_str) if num_runs_str else None

    models = []
    if model_choice == "1":
        models = ["original"]
    elif model_choice == "2":
        models = ["finetuned"]
    else:
        models = ["original", "finetuned"]

    datasets = []
    if dataset_choice == "1":
        datasets = ["smsa"]
    elif dataset_choice == "2":
        datasets = ["tweets"]
    else:
        datasets = ["smsa", "tweets"]

    selected = []
    for m in models:
        for d in datasets:
            selected.append(f"{m}_{d}")

    return selected, num_runs_input


if __name__ == "__main__":
    num_runs_override = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            selected = list(EXPERIMENT_CONFIGS.keys())
        else:
            selected = sys.argv[1:]
            for key in selected:
                if key not in EXPERIMENT_CONFIGS:
                    print(f"Unknown version: {key}")
                    print(f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
                    sys.exit(1)
    else:
        selected, num_runs_override = interactive_menu()

    print("\n" + "=" * 80)
    print(f"STARTING PTQ EXPERIMENTS - {len(selected)} VERSION(S) TO RUN")
    print("=" * 80)
    for i, key in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {key}")
    print("=" * 80 + "\n")

    all_results = {}
    for idx, version_key in enumerate(selected, 1):
        print("\n" + "#" * 80)
        print(f"# RUNNING EXPERIMENT [{idx}/{len(selected)}]: {version_key.upper()}")
        print("#" * 80 + "\n")
        result = run_ptq_experiment(version_key, num_runs_override=num_runs_override)
        all_results[version_key] = result
        print(f"\n  Completed [{idx}/{len(selected)}]: {version_key}")

    print_section("ALL EXPERIMENTS COMPLETED")
    for key, res in all_results.items():
        print(f"\n{key}:")
        print(f"  Accuracy: FP32={res['fp32_results']['accuracy']*100:.2f}% | FP16={res['fp16_results']['accuracy']*100:.2f}% | INT8={res['int8_results']['accuracy']*100:.2f}% | INT4={res['int4_results']['accuracy']*100:.2f}%")
        print(f"  Size: FP32={res['fp32_size_mb']:.1f}MB | FP16={res['fp16_size_mb']:.1f}MB | INT8={res['int8_size_mb']:.1f}MB | INT4={res['int4_size_mb']:.1f}MB")
