import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from pathlib import Path
from src.models import ModelManager
from src.models.base import BaseModel, OnnxBaseModel
from src.quantization.ptq import PTQQuantizer
from src.config import LABELS


class LIMEExplainer:
    def __init__(self, base_model, labels, use_fp16=False, random_state=None):
        self.base_model = base_model
        self.labels = labels
        self.use_fp16 = use_fp16
        self.label_names = [labels[i] for i in sorted(labels.keys())]
        self.explainer = LimeTextExplainer(
            class_names=self.label_names,
            random_state=random_state,
        )

    def predict_proba(self, texts):
        probabilities = []
        for text in texts:
            result = self.base_model.predict(text, use_fp16=self.use_fp16)
            probs = [result["probabilities"][self.label_names[i]] for i in range(len(self.label_names))]
            probabilities.append(probs)
        return np.array(probabilities)

    def explain(self, text, num_features=10, num_samples=300):
        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=list(range(len(self.label_names)))
        )
        return explanation

    def explain_and_save(self, text, output_path, num_features=10, num_samples=300):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        explanation = self.explain(text, num_features, num_samples)
        explanation.save_to_file(str(output_path))

        predicted_label_idx = int(np.argmax(explanation.predict_proba))
        feature_weights = explanation.as_list(label=predicted_label_idx)

        return {
            "predicted_label": self.label_names[predicted_label_idx],
            "prediction_probabilities": {
                self.label_names[i]: float(explanation.predict_proba[i])
                for i in range(len(self.label_names))
            },
            "top_features": feature_weights,
            "output_path": str(output_path)
        }

def run_lime_attribution():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _LOG_PATH = _OUT_DIR / "lime_errors.log"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        print("  Run: python scripts/prepare_datasets.py")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"],
         "expected": LABELS[int(row["true_label"])]}
        for _, row in df_sub.iterrows()
    ]
    print(f"\n  Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")

    if not _FP32_DIR.exists():
        print(f"  [ERROR] FP32 model not found: {_FP32_DIR}")
        return
    print(f"\n  Loading FP32 base: {_FP32_DIR}")
    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()

    def _load_onnx(variant):
        import onnxruntime as ort
        onnx_dir = _MODELS_DIR / f"qat_onnx_{variant}_seed42"
        onnx_file = onnx_dir / f"model_qat_{variant}.onnx"
        if not onnx_file.exists():
            print(f"  [SKIP] ONNX file not found: {onnx_file}")
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxBaseModel(session, fp32_base.tokenizer, None, torch.device("cpu"))

    def _build_ptq(precision):
        ptq = PTQQuantizer(fp32_base.model)
        m, _ = getattr(ptq, f"quantize_{precision}")()
        return BaseModel(m, fp32_base.tokenizer, device=fp32_base.device)

    VARIANTS = [
        ("fp32",          lambda: fp32_base,           False),
        ("ptq_fp16",      lambda: _build_ptq("fp16"),  True),
        ("ptq_int8",      lambda: _build_ptq("int8"),  False),
        ("ptq_int4",      lambda: _build_ptq("int4"),  False),
        ("qat_fp32",      lambda: (ModelManager.load_model(str(_QAT_CLEAN_DIR))
                                   if _QAT_CLEAN_DIR.exists() else None), False),
        ("qat_onnx_fp16", lambda: _load_onnx("fp16"), True),
        ("qat_onnx_int8", lambda: _load_onnx("int8"), False),
        ("qat_onnx_int4", lambda: _load_onnx("int4"), False),
    ]

    n_variants = len(VARIANTS)
    n_samples  = len(samples)
    total_expected = n_variants * n_samples
    global_run = 0
    global_done = 0

    print(f"\n  Running LIME: {n_variants} variants x {n_samples} samples "
          f"x 1000 perturbations ~ {total_expected} total calls")
    print(f"  Output: {_OUT_DIR}")
    print(f"  Error log: {_LOG_PATH}")

    for vname, load_fn, use_fp16 in VARIANTS:
        print(f"\n{'-'*60}")
        print(f"  [{vname.upper()}]  Loading model...")

        model = load_fn()
        if model is None:
            print(f"  [SKIP] {vname} - model or ONNX file not found.")
            global_run += n_samples
            continue

        lime_explainer = LIMEExplainer(model, LABELS, use_fp16=use_fp16, random_state=42)

        for sample in samples:
            global_run += 1
            sid  = sample["sample_id"]
            text = sample["text"]
            out_path = _OUT_DIR / f"lime_{vname}_{sid}.npy"

            if out_path.exists():
                global_done += 1
                if global_run % 50 == 0:
                    print(f"  [progress] {global_run}/{total_expected} "
                          f"({global_done} saved, {global_run - global_done} skipped/error)")
                continue

            try:
                n_words = len(text.split())
                exp = lime_explainer.explain(
                    text,
                    num_features=max(n_words, 10),
                    num_samples=1000,
                )
                pred_idx = int(np.argmax(exp.predict_proba))
                scores_dict = dict(exp.as_list(label=pred_idx))
                word_scores = np.array(
                    [scores_dict.get(w, 0.0) for w in text.split()],
                    dtype=np.float32,
                )
                np.save(out_path, word_scores)
                global_done += 1
            except Exception as exc:
                with open(_LOG_PATH, "a", encoding="utf-8") as _log:
                    _log.write(f"{vname}\t{sid}\t{type(exc).__name__}: {exc}\n")

            if global_run % 50 == 0:
                pct = 100 * global_run / total_expected
                print(f"  [progress] {global_run}/{total_expected} ({pct:.1f}%)  "
                      f"{global_done} saved  variant={vname}  sample_id={sid}")

    print(f"\n  LIME attribution complete.")
    print(f"  {global_done}/{total_expected} files saved to {_OUT_DIR}")
    print(f"  File pattern: lime_{{variant}}_{{sample_id}}.npy")
