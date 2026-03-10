import random
import numpy as np
import torch

DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] Global seed set to {seed}")

def print_section(title, width=80, char="="):
    print("\n" + char * width)
    print(title)
    print(char * width)

def format_metrics(results):
    return {
        "accuracy": f"{results['accuracy']*100:.2f}%",
        "confidence": f"{results['avg_confidence']*100:.2f}%",
        "latency": f"{results['latency_stats']['mean']*1000:.2f} ms"
    }
