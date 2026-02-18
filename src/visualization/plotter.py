import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class QuantizationPlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_model_size_comparison(self, sizes, labels):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
        bars = ax.bar(labels, sizes, color=colors[:len(sizes)], edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Model Size (MB)', fontweight='bold')
        ax.set_title('Model Size Comparison', fontweight='bold')
        
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size:.1f} MB', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_latency_distribution(self, latency_data, labels):
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(latency_data, labels=labels, patch_artist=True)
        
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors[:len(latency_data)]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Inference Latency (ms)', fontweight='bold')
        ax.set_title('Latency Distribution Comparison', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_confidence_comparison(self, predictions_list, labels, test_samples):
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(test_samples))
        width = 0.8 / len(predictions_list)
        
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
        for i, (predictions, label) in enumerate(zip(predictions_list, labels)):
            confidences = [p['confidence']*100 for p in predictions]
            offset = (i - len(predictions_list)/2 + 0.5) * width
            ax.bar(x + offset, confidences, width, label=label, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Confidence (%)', fontweight='bold')
        ax.set_title('Per-Sample Confidence Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in x])
        ax.legend()
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        return fig

    def plot_performance_metrics(self, metrics_list, labels):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metric_names = ['Accuracy', 'Avg Confidence', 'Speed Gain']
        x = np.arange(len(metric_names))
        width = 0.8 / len(metrics_list)
        
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
        for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
            offset = (i - len(metrics_list)/2 + 0.5) * width
            ax.bar(x + offset, metrics, width, label=label, color=colors[i], alpha=0.8)
        
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Overall Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig

    def create_comprehensive_plot(self, all_results, test_samples, version_key):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'PTQ Analysis: {version_key}', fontsize=14, fontweight='bold')

        precision_labels = ['FP32', 'FP16', 'INT8', 'INT4']
        precision_keys = ['fp32', 'fp16', 'int8', 'int4']
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

        ax = axes[0, 0]
        accuracies = [all_results[k]['accuracy'] * 100 for k in precision_keys]
        bars = ax.bar(precision_labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_ylim(0, max(accuracies) * 1.15)

        ax = axes[0, 1]
        latency_data = []
        for k in precision_keys:
            latencies_ms = [l * 1000 for l in all_results[k]['latencies']]
            latency_data.append(latencies_ms)
        bp = ax.boxplot(latency_data, labels=precision_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Inference Latency (ms)', fontweight='bold')
        ax.set_title('Latency Distribution', fontweight='bold')

        ax = axes[1, 0]
        confidences = [all_results[k]['avg_confidence'] * 100 for k in precision_keys]
        bars = ax.bar(precision_labels, confidences, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, confidences):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Confidence (%)', fontweight='bold')
        ax.set_title('Average Confidence', fontweight='bold')
        ax.set_ylim(0, max(confidences) * 1.15)

        ax = axes[1, 1]
        mean_latencies = [all_results[k]['latency_stats']['mean'] * 1000 for k in precision_keys]
        bars = ax.bar(precision_labels, mean_latencies, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, mean_latencies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Mean Latency (ms)', fontweight='bold')
        ax.set_title('Mean Inference Latency', fontweight='bold')

        plt.tight_layout()
        chart_path = self.output_dir / f'quantization_analysis_{version_key}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path
