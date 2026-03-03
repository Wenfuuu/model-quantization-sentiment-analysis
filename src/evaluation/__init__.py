from .evaluator import ModelEvaluator
from .metrics import compute_metrics, compare_predictions
from .explanation_drift import (
    aggregate_explanation_drift,
    wilcoxon_drift_test,
    spearman_rank_correlation,
    top_k_jaccard,
    sign_flip_rate,
    normalized_magnitude_shift,
    lime_to_attribution,
    shap_to_attribution,
    ig_to_attribution,
)
from .calibration import (
    expected_calibration_error,
    brier_score,
    extract_calibration_inputs,
    plot_reliability_diagram,
    plot_calibration_comparison,
    compare_calibration,
)
from .per_class_analysis import (
    per_class_report,
    per_class_delta,
    confusion_matrix_arrays,
    plot_confusion_matrix_comparison,
    plot_per_class_f1_comparison,
    mcnemar_test,
    mcnemar_per_class,
)
from .faithfulness import (
    FaithfulnessAtK,
    FaithfulnessResult,
    FaithfulnessEvaluator,
    evaluate_faithfulness_batch,
    aggregate_faithfulness,
    compare_faithfulness,
    save_faithfulness_results,
    load_faithfulness_results,
)
from .stress_test import (
    run_edge_case_test,
    run_noise_robustness_test,
    run_calibration_stress_test,
)

__all__ = [
    "ModelEvaluator",
    "compute_metrics",
    "compare_predictions",
    # Explanation drift
    "aggregate_explanation_drift",
    "wilcoxon_drift_test",
    "spearman_rank_correlation",
    "top_k_jaccard",
    "sign_flip_rate",
    "normalized_magnitude_shift",
    "lime_to_attribution",
    "shap_to_attribution",
    "ig_to_attribution",
    # Calibration
    "expected_calibration_error",
    "brier_score",
    "extract_calibration_inputs",
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "compare_calibration",
    # Per-class
    "per_class_report",
    "per_class_delta",
    "confusion_matrix_arrays",
    "plot_confusion_matrix_comparison",
    "plot_per_class_f1_comparison",
    "mcnemar_test",
    "mcnemar_per_class",
    # Faithfulness (ERASER: sufficiency + comprehensiveness)
    "FaithfulnessAtK",
    "FaithfulnessResult",
    "FaithfulnessEvaluator",
    "evaluate_faithfulness_batch",
    "aggregate_faithfulness",
    "compare_faithfulness",
    "save_faithfulness_results",
    "load_faithfulness_results",
    # Stress test
    "run_edge_case_test",
    "run_noise_robustness_test",
    "run_calibration_stress_test",
]
