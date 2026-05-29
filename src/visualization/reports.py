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
