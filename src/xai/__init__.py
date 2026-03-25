from .integrated_gradients import IntegratedGradientsExplainer
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .occlusion import OcclusionExplainer
from .ig_metrics import (
	attribution_similarity,
	integrated_gradients_tokens,
	gradient_times_input_tokens,
	InsertionDeletionEvaluator,
	InsertionDeletionResult,
	layer_cls_similarity,
)
# Random attribution baseline — required for null-floor measurement.
# Every similarity and drift metric must be compared against this floor
# before any claim of "attribution preservation" can be made.
from .random_baseline import (
	RandomAttributionBaseline,
	RandomAttributionResult,
	compute_baseline_floor,
	above_floor_delta,
	evaluate_baseline_floor_batch,
	save_baseline_floors,
	load_baseline_floors,
)
# Word-to-subword alignment layer.
# MUST be applied before any cross-method (SHAP/LIME vs IG) comparison.
# Without alignment, _align_attributions() in explanation_drift.py produces
# zero-fill artifacts that invalidate all Spearman/Jaccard/sign-flip metrics.
from .alignment import (
	UNALIGNED,
	WordSubwordAlignment,
	build_alignment,
	build_alignment_batch,
	project_word_to_subword,
	project_subword_to_word,
	sparse_word_to_subword,
	align_for_comparison,
	align_attribution_batch,
	fragmentation_report,
)
# SmoothGrad: noise-averaged IG for gradient instability vs structural drift.
# Separates quantization-induced gradient noise from genuine attribution drift.
# Reference: Smilkov et al. (2017) "SmoothGrad: removing noise by adding noise".
from .smoothgrad import (
	SmoothGradResult,
	SmoothGradComparison,
	SmoothGradExplainer,
	compute_sigma,
	compare_smoothgrad,
	explain_smoothgrad_batch,
	compare_smoothgrad_batch,
	aggregate_smoothgrad_comparisons,
	save_smoothgrad_results,
	save_smoothgrad_comparisons,
	load_smoothgrad_results,
)
# Attention-based diagnostics: raw weights, rollout (Abnar & Zuidema 2020),
# per-head entropy, and cross-precision comparison.
from .attention_analysis import (
	AttentionWeights,
	RolloutResult,
	AttentionEntropyResult,
	AttentionComparisonResult,
	AttentionAnalyzer,
	extract_attention_weights,
	compute_rollout,
	compute_entropy,
	compare_attention,
	analyze_attention_batch,
	compare_attention_batch,
	aggregate_attention_comparisons,
	save_attention_results,
	save_attention_comparisons,
	load_attention_results,
)

__all__ = [
	"IntegratedGradientsExplainer",
	"LIMEExplainer",
	"SHAPExplainer",
	"OcclusionExplainer",
	"attribution_similarity",
	"integrated_gradients_tokens",
	"gradient_times_input_tokens",
	"InsertionDeletionEvaluator",
	"InsertionDeletionResult",
	"layer_cls_similarity",
	# Random baseline
	"RandomAttributionBaseline",
	"RandomAttributionResult",
	"compute_baseline_floor",
	"above_floor_delta",
	"evaluate_baseline_floor_batch",
	"save_baseline_floors",
	"load_baseline_floors",
	# Word-to-subword alignment
	"UNALIGNED",
	"WordSubwordAlignment",
	"build_alignment",
	"build_alignment_batch",
	"project_word_to_subword",
	"project_subword_to_word",
	"sparse_word_to_subword",
	"align_for_comparison",
	"align_attribution_batch",
	"fragmentation_report",
	# SmoothGrad
	"SmoothGradResult",
	"SmoothGradComparison",
	"SmoothGradExplainer",
	"compute_sigma",
	"compare_smoothgrad",
	"explain_smoothgrad_batch",
	"compare_smoothgrad_batch",
	"aggregate_smoothgrad_comparisons",
	"save_smoothgrad_results",
	"save_smoothgrad_comparisons",
	"load_smoothgrad_results",
	# Attention analysis
	"AttentionWeights",
	"RolloutResult",
	"AttentionEntropyResult",
	"AttentionComparisonResult",
	"AttentionAnalyzer",
	"extract_attention_weights",
	"compute_rollout",
	"compute_entropy",
	"compare_attention",
	"analyze_attention_batch",
	"compare_attention_batch",
	"aggregate_attention_comparisons",
	"save_attention_results",
	"save_attention_comparisons",
	"load_attention_results",
]
