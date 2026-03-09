import os
import numpy as np
from tqdm import tqdm


def get_memory_mb():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_model_param_memory_mb(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)


class ModelEvaluator:
    def __init__(self, base_model):
        self.base_model = base_model

    def evaluate(self, samples, num_runs=20, warmup=5, use_fp16=False):
        results = {"predictions": [], "latencies": [], "accuracy": 0, "avg_confidence": 0}
        correct = 0
        total_confidence = 0
        mem_before = get_memory_mb()

        total_inferences = len(samples) * (warmup + num_runs + 1)
        print(f"Total samples: {len(samples)}")
        print(f"Total inference operations: {total_inferences:,}\n")

        pbar = tqdm(samples, desc="Evaluating samples", unit="sample")

        for sample in pbar:
            for _ in range(warmup):
                _ = self.base_model.predict(sample["text"], use_fp16=use_fp16)

            sample_latencies = []
            for _ in range(num_runs):
                pred = self.base_model.predict(sample["text"], use_fp16=use_fp16)
                sample_latencies.append(pred["inference_time"])

            final_pred = self.base_model.predict(sample["text"], use_fp16=use_fp16)

            results["predictions"].append({
                "text": sample["text"],
                "expected": sample["expected"],
                "predicted": final_pred["label"],
                "confidence": final_pred["confidence"],
                "probabilities": final_pred["probabilities"],
                "avg_latency": np.mean(sample_latencies),
                "std_latency": np.std(sample_latencies)
            })

            results["latencies"].extend(sample_latencies)

            if final_pred["label"] == sample["expected"]:
                correct += 1
            total_confidence += final_pred["confidence"]

            current_accuracy = correct / len(results["predictions"])
            pbar.set_postfix({"Accuracy": f"{current_accuracy:.2%}", "Correct": f"{correct}/{len(results['predictions'])}"})

        mem_after = get_memory_mb()
        results["memory_usage_mb"] = mem_after - mem_before
        results["accuracy"] = correct / len(samples)
        results["avg_confidence"] = total_confidence / len(samples)
        results["latency_stats"] = {
            "mean": np.mean(results["latencies"]),
            "std": np.std(results["latencies"]),
            "min": np.min(results["latencies"]),
            "max": np.max(results["latencies"]),
            "median": np.median(results["latencies"])
        }

        return results
