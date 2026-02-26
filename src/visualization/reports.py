import pandas as pd


def generate_comparison_report(fp32_results, fp16_results, int8_results, int4_results, 
                               fp32_size_mb, fp16_size_mb, int8_size_mb, int4_size_mb):
    comparison_data = {
        "Metric": [
            "Model Size (MB)", "Accuracy (%)", "Avg Confidence (%)",
            "Mean Latency (ms)", "Median Latency (ms)", "Std Latency (ms)"
        ],
        "FP32 (Baseline)": [
            f"{fp32_size_mb:.2f}", f"{fp32_results['accuracy']*100:.2f}",
            f"{fp32_results['avg_confidence']*100:.2f}",
            f"{fp32_results['latency_stats']['mean']*1000:.2f}",
            f"{fp32_results['latency_stats']['median']*1000:.2f}",
            f"{fp32_results['latency_stats']['std']*1000:.2f}"
        ],
        "FP16 (Half)": [
            f"{fp16_size_mb:.2f}", f"{fp16_results['accuracy']*100:.2f}",
            f"{fp16_results['avg_confidence']*100:.2f}",
            f"{fp16_results['latency_stats']['mean']*1000:.2f}",
            f"{fp16_results['latency_stats']['median']*1000:.2f}",
            f"{fp16_results['latency_stats']['std']*1000:.2f}"
        ],
        "INT8 (Quantized)": [
            f"{int8_size_mb:.2f}", f"{int8_results['accuracy']*100:.2f}",
            f"{int8_results['avg_confidence']*100:.2f}",
            f"{int8_results['latency_stats']['mean']*1000:.2f}",
            f"{int8_results['latency_stats']['median']*1000:.2f}",
            f"{int8_results['latency_stats']['std']*1000:.2f}"
        ],
        "INT4 (4-bit)": [
            f"{int4_size_mb:.2f}", f"{int4_results['accuracy']*100:.2f}",
            f"{int4_results['avg_confidence']*100:.2f}",
            f"{int4_results['latency_stats']['mean']*1000:.2f}",
            f"{int4_results['latency_stats']['median']*1000:.2f}",
            f"{int4_results['latency_stats']['std']*1000:.2f}"
        ],
        "FP16 vs FP32": [
            f"{(1 - fp16_size_mb/fp32_size_mb)*100:+.2f}%",
            f"{(fp16_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%",
            f"{(fp16_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.2f}%",
            f"{((fp16_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.2f}%",
            f"{((fp16_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.2f}%",
            "N/A"
        ],
        "INT8 vs FP32": [
            f"{(1 - int8_size_mb/fp32_size_mb)*100:+.2f}%",
            f"{(int8_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%",
            f"{(int8_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.2f}%",
            f"{((int8_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.2f}%",
            f"{((int8_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.2f}%",
            "N/A"
        ],
        "INT4 vs FP32": [
            f"{(1 - int4_size_mb/fp32_size_mb)*100:+.2f}%",
            f"{(int4_results['accuracy'] - fp32_results['accuracy'])*100:+.2f}%",
            f"{(int4_results['avg_confidence'] - fp32_results['avg_confidence'])*100:+.2f}%",
            f"{((int4_results['latency_stats']['mean'] - fp32_results['latency_stats']['mean'])/fp32_results['latency_stats']['mean'])*100:+.2f}%",
            f"{((int4_results['latency_stats']['median'] - fp32_results['latency_stats']['median'])/fp32_results['latency_stats']['median'])*100:+.2f}%",
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
            "FP32%": f"{fp32_pred['confidence']*100:.1f}%",
            "FP16": fp16_pred["predicted"],
            "FP16%": f"{fp16_pred['confidence']*100:.1f}%",
            "FP16=FP32": fp16_match,
            "INT8": int8_pred["predicted"],
            "INT8%": f"{int8_pred['confidence']*100:.1f}%",
            "INT8=FP32": int8_match,
            "INT4": int4_pred["predicted"],
            "INT4%": f"{int4_pred['confidence']*100:.1f}%",
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
