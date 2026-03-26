import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.models import ModelManager
from src.models.base import BaseModel, OnnxBaseModel
from src.quantization.ptq import PTQQuantizer
from src.config import LABELS

class SHAPExplainer:
    def __init__(self, base_model, labels, use_fp16=False):
        self.base_model = base_model
        self.labels = labels
        self.use_fp16 = use_fp16
        self.label_names = [labels[i] for i in sorted(labels.keys())]

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        probabilities = []
        for text in texts:
            result = self.base_model.predict(text, use_fp16=self.use_fp16)
            probs = [result["probabilities"][self.label_names[i]] for i in range(len(self.label_names))]
            probabilities.append(probs)
        return np.array(probabilities)

    def explain(self, text, max_evals=200):
        masker = shap.maskers.Text(r"\s+")
        explainer = shap.Explainer(
            self.predict_proba,
            masker,
            output_names=self.label_names
        )
        shap_values = explainer([text], max_evals=max_evals)
        return shap_values

    def explain_and_save(self, text, output_path, max_evals=200):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap_values = self.explain(text, max_evals=max_evals)

        predicted_class = int(np.argmax(self.predict_proba(text)))

        token_importance = {}
        if hasattr(shap_values[0], 'data') and hasattr(shap_values[0], 'values'):
            data = shap_values[0].data
            values = shap_values[0].values
            for i, token in enumerate(data):
                if isinstance(token, str) and token.strip():
                    token_importance[token] = float(values[i][predicted_class])

        sorted_importance = sorted(token_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

        if sorted_importance:
            tokens = [t[0] for t in sorted_importance]
            weights = [t[1] for t in sorted_importance]
            colors = ['#ff4444' if w < 0 else '#4444ff' for w in weights]

            fig, ax = plt.subplots(figsize=(12, max(4, len(tokens) * 0.4)))
            y_pos = range(len(tokens))
            ax.barh(y_pos, weights, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens)
            ax.invert_yaxis()
            ax.set_xlabel(f"SHAP value (impact on {self.label_names[predicted_class]})")
            ax.set_title(f"SHAP Token Importance - Predicted: {self.label_names[predicted_class]}")
            ax.axvline(x=0, color='black', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()

        return {
            "predicted_label": self.label_names[predicted_class],
            "token_importance": sorted_importance[:10],
            "output_path": str(output_path)
        }

def run_shap_attribution():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _LOG_PATH = _OUT_DIR / "shap_errors.log"
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

    print(f"\n  Running SHAP: {n_variants} variants x {n_samples} samples "
          f"x 500 max_evals ~ {total_expected} total calls")
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

        shap_explainer = SHAPExplainer(model, LABELS, use_fp16=use_fp16)

        for sample in samples:
            global_run += 1
            sid  = sample["sample_id"]
            text = sample["text"]
            out_path = _OUT_DIR / f"shap_{vname}_{sid}.npy"

            if out_path.exists():
                global_done += 1
                if global_run % 50 == 0:
                    print(f"  [progress] {global_run}/{total_expected} "
                          f"({global_done} saved, {global_run - global_done} skipped/error)")
                continue

            try:
                shap_values = shap_explainer.explain(text, max_evals=500)
                predicted_class = int(np.argmax(shap_explainer.predict_proba(text)))
                scores_dict = {}
                sv = shap_values
                for j, token in enumerate(sv[0].data):
                    if isinstance(token, str) and token.strip():
                        scores_dict[token.strip()] = float(sv[0].values[j][predicted_class])
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

    print(f"\n  SHAP attribution complete.")
    print(f"  {global_done}/{total_expected} files saved to {_OUT_DIR}")
    print(f"  File pattern: shap_{{variant}}_{{sample_id}}.npy")
