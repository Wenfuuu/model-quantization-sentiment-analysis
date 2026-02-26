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
