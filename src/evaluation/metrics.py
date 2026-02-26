import numpy as np
from scipy import stats


def compute_metrics(results):
    return {
        "accuracy": results["accuracy"],
        "avg_confidence": results["avg_confidence"],
        "latency": results["latency_stats"]
    }


def compare_predictions(fp32_results, quantized_results):
    mismatches = 0
    for fp32_pred, quant_pred in zip(fp32_results["predictions"], quantized_results["predictions"]):
        if fp32_pred["predicted"] != quant_pred["predicted"]:
            mismatches += 1
    
    consistency = 1 - (mismatches / len(fp32_results["predictions"]))
    return consistency, mismatches


def statistical_test(fp32_latencies, quantized_latencies):
    t_stat, p_value = stats.ttest_ind(fp32_latencies, quantized_latencies)
    
    pooled_std = np.sqrt(((len(fp32_latencies)-1)*np.std(fp32_latencies)**2 +
                          (len(quantized_latencies)-1)*np.std(quantized_latencies)**2) /
                         (len(fp32_latencies) + len(quantized_latencies) - 2))
    cohens_d = (np.mean(fp32_latencies) - np.mean(quantized_latencies)) / pooled_std
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05
    }


def confidence_comparison(fp32_results, quantized_results):
    fp32_confidences = [p["confidence"] for p in fp32_results["predictions"]]
    quant_confidences = [p["confidence"] for p in quantized_results["predictions"]]
    
    t_stat, p_value = stats.ttest_rel(fp32_confidences, quant_confidences)
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "fp32_mean": np.mean(fp32_confidences),
        "quantized_mean": np.mean(quant_confidences)
    }
