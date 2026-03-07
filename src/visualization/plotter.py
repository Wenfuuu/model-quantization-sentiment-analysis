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
            confidences = [p['confidence'] * 100 for p in predictions]
            offset = (i - len(predictions_list)/2 + 0.5) * width
            ax.bar(x + offset, confidences, width, label=label, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Confidence', fontweight='bold')
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
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Overall Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig

    def create_comprehensive_plot(self, all_results, test_samples, version_key, model_sizes=None, memory_usages=None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'PTQ Analysis: {version_key}', fontsize=14, fontweight='bold')

        precision_labels = ['FP32', 'FP16', 'INT8', 'INT4']
        precision_keys = ['fp32', 'fp16', 'int8', 'int4']
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

        ax = axes[0, 0]
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

        ax = axes[0, 1]
        accuracies = [all_results[k]['accuracy'] * 100 for k in precision_keys]
        bars = ax.bar(precision_labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_ylim(0, max(accuracies) * 1.15)

        ax = axes[1, 0]
        if model_sizes is not None:
            sizes = [model_sizes.get(k, 0) for k in precision_keys]
        else:
            sizes = [0] * len(precision_keys)
        bars = ax.bar(precision_labels, sizes, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Model Size (MB)', fontweight='bold')
        ax.set_title('Model Size Comparison', fontweight='bold')

        ax = axes[1, 1]
        if memory_usages is not None:
            mem_vals = [memory_usages.get(k, 0) for k in precision_keys]
        else:
            mem_vals = [0] * len(precision_keys)
        bars = ax.bar(precision_labels, mem_vals, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, mem_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
        ax.set_title('Memory Usage', fontweight='bold')

        plt.tight_layout()
        chart_path = self.output_dir / f'quantization_analysis_{version_key}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path

    def create_qat_comparison_plot(self, all_results, method_name, model_sizes=None, memory_usages=None):
        quant_labels = []
        quant_keys = []
        for k in ['int8', 'fp16', 'int4']:
            if k in all_results:
                quant_keys.append(k)
                quant_labels.append(k.upper())

        if len(quant_keys) < 2:
            return None

        colors_map = {'int8': '#2ecc71', 'fp16': '#9b59b6', 'int4': '#e74c3c'}
        colors = [colors_map[k] for k in quant_keys]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'QAT Analysis: {method_name.upper()} Method', fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        has_latencies = any('latencies' in all_results[k] for k in quant_keys)
        if has_latencies:
            latency_data = []
            latency_labels = []
            latency_colors = []
            for i, k in enumerate(quant_keys):
                if 'latencies' in all_results[k]:
                    latencies_ms = [l * 1000 for l in all_results[k]['latencies']]
                    latency_data.append(latencies_ms)
                    latency_labels.append(quant_labels[i])
                    latency_colors.append(colors[i])
            bp = ax.boxplot(latency_data, labels=latency_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], latency_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel('Inference Latency (ms)', fontweight='bold')
            ax.set_title('Latency Distribution', fontweight='bold')
        else:
            latency_stats = []
            for k in quant_keys:
                stats = all_results[k].get('latency_stats', {})
                latency_stats.append(stats.get('mean', 0) * 1000)
            bars = ax.bar(quant_labels, latency_stats, color=colors, edgecolor='black', linewidth=1.2)
            for bar, val in zip(bars, latency_stats):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax.set_ylabel('Mean Latency (ms)', fontweight='bold')
            ax.set_title('Latency Distribution', fontweight='bold')

        ax = axes[0, 1]
        accuracies = []
        for k in quant_keys:
            metrics = all_results[k]['overall_metrics']
            acc = metrics.get('accuracy', metrics.get('eval_accuracy', 0))
            accuracies.append(acc * 100)
        bars = ax.bar(quant_labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_ylim(0, max(accuracies) * 1.15)

        ax = axes[1, 0]
        if model_sizes is not None:
            sizes = [model_sizes.get(k, 0) for k in quant_keys]
        else:
            sizes = [0] * len(quant_keys)
        bars = ax.bar(quant_labels, sizes, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Model Size (MB)', fontweight='bold')
        ax.set_title('Model Size Comparison', fontweight='bold')

        ax = axes[1, 1]
        if memory_usages is not None:
            mem_vals = [memory_usages.get(k, 0) for k in quant_keys]
        else:
            mem_vals = [0] * len(quant_keys)
        bars = ax.bar(quant_labels, mem_vals, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, mem_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
        ax.set_title('Memory Usage', fontweight='bold')

        plt.tight_layout()
        chart_path = self.output_dir / f'qat_comparison_{method_name}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path
