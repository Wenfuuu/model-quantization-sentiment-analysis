import numpy as np
import torch
from pathlib import Path
from src.models import ModelManager
from src.models.base import BaseModel, OnnxBaseModel
from src.quantization.ptq import PTQQuantizer
from src.config import LABELS

class OcclusionExplainer:
    def __init__(self, base_model, labels, use_fp16=False):
        self.base_model = base_model
        self.labels = labels
        self.use_fp16 = use_fp16
        self.label_names = [labels[i] for i in sorted(labels.keys())]

    def explain(self, text, window_size=1):
        baseline_result = self.base_model.predict(text, use_fp16=self.use_fp16)
        predicted_idx = self.label_names.index(baseline_result["label"])
        baseline_conf = baseline_result["probabilities"][self.label_names[predicted_idx]]

        words = text.split()
        token_importance = []

        for i in range(len(words)):
            start = max(0, i)
            end = min(len(words), i + window_size)
            occluded_words = words[:start] + ["[MASK]"] * (end - start) + words[end:]
            occluded_text = " ".join(occluded_words)

            occluded_result = self.base_model.predict(occluded_text, use_fp16=self.use_fp16)
            occluded_conf = occluded_result["probabilities"][self.label_names[predicted_idx]]

            importance = baseline_conf - occluded_conf
            token_importance.append((words[i], float(importance)))

        sorted_importance = sorted(token_importance, key=lambda x: abs(x[1]), reverse=True)

        return {
            "predicted_label": self.label_names[predicted_idx],
            "token_importance": sorted_importance,
            "all_tokens_ordered": token_importance,
        }

def run_occlusion_attribution():
    import pandas as pd

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _LOG_PATH = _OUT_DIR / "occ_errors.log"
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

    print(f"\n  Running Occlusion: {n_variants} variants x {n_samples} samples "
          f"(window_size=1, [MASK] substitution)")
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

        occlusion_explainer = OcclusionExplainer(model, LABELS, use_fp16=use_fp16)

        for sample in samples:
            global_run += 1
            sid  = sample["sample_id"]
            text = sample["text"]
            out_path = _OUT_DIR / f"occ_{vname}_{sid}.npy"

            if out_path.exists():
                global_done += 1
                if global_run % 50 == 0:
                    print(f"  [progress] {global_run}/{total_expected} "
                          f"({global_done} saved, {global_run - global_done} skipped/error)")
                continue

            try:
                occ_result = occlusion_explainer.explain(text, window_size=1)
                word_scores = np.array(
                    [score for _, score in occ_result["all_tokens_ordered"]],
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

    print(f"\n  Occlusion attribution complete.")
    print(f"  {global_done}/{total_expected} files saved to {_OUT_DIR}")
    print(f"  File pattern: occ_{{variant}}_{{sample_id}}.npy")

def run_cross_seed_verification():
    import pandas as pd
    import onnxruntime as ort
    from scipy.stats import spearmanr

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _MODELS_DIR    = _PROJECT_ROOT / "models"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"]}
        for _, row in df_sub.iterrows()
    ][:20]
    print(f"\n  Cross-seed verification: {len(samples)} samples, seeds [42, 123, 456]")

    all_rows = []

    for seed in [42, 123, 456]:
        print(f"\n  [{seed}]  loading models...")

        fp32_model = ModelManager.load_model(str(_MODELS_DIR / f"fp32_seed{seed}"))
        fp32_model.model.eval()

        ptq_q = PTQQuantizer(fp32_model.model)
        ptq_m, _ = ptq_q.quantize_int4()
        ptq_model = BaseModel(ptq_m, fp32_model.tokenizer, device=fp32_model.device)

        onnx_file = _MODELS_DIR / f"qat_onnx_int4_seed{seed}" / "model_qat_int4.onnx"
        if not onnx_file.exists():
            print(f"  [SKIP] {onnx_file} not found")
            qat_model = None
        else:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            opts.log_severity_level = 3
            session = ort.InferenceSession(
                str(onnx_file), opts, providers=["CPUExecutionProvider"]
            )
            qat_model = OnnxBaseModel(session, fp32_model.tokenizer, None, torch.device("cpu"))

        fp32_occ = OcclusionExplainer(fp32_model, LABELS, use_fp16=False)
        ptq_occ  = OcclusionExplainer(ptq_model,  LABELS, use_fp16=False)
        qat_occ  = OcclusionExplainer(qat_model,  LABELS, use_fp16=False) if qat_model else None

        for sample in samples:
            sid  = sample["sample_id"]
            text = sample["text"]

            sfx = "" if seed == 42 else f"_s{seed}"
            fp32_path = _OUT_DIR / f"occ_fp32{sfx}_{sid}.npy"
            ptq_path  = _OUT_DIR / f"occ_ptq_int4{sfx}_{sid}.npy"
            qat_path  = _OUT_DIR / f"occ_qat_onnx_int4{sfx}_{sid}.npy"

            if not fp32_path.exists():
                r = fp32_occ.explain(text, window_size=1)
                np.save(fp32_path, np.array([s for _, s in r["all_tokens_ordered"]], dtype=np.float32))
            if not ptq_path.exists():
                r = ptq_occ.explain(text, window_size=1)
                np.save(ptq_path,  np.array([s for _, s in r["all_tokens_ordered"]], dtype=np.float32))
            if qat_occ and not qat_path.exists():
                r = qat_occ.explain(text, window_size=1)
                np.save(qat_path,  np.array([s for _, s in r["all_tokens_ordered"]], dtype=np.float32))

            if not fp32_path.exists():
                print(f"  [SKIP] sid={sid} seed={seed}: fp32 missing")
                continue

            fp32_s = np.load(fp32_path).astype(np.float64)

            if ptq_path.exists():
                ptq_s = np.load(ptq_path).astype(np.float64)
                L = min(len(fp32_s), len(ptq_s))
                rho, _ = spearmanr(fp32_s[:L], ptq_s[:L])
                all_rows.append({"seed": seed, "sample_id": sid,
                                  "comparison": "fp32_vs_ptq_int4", "spearman_rho": float(rho)})

            if qat_path.exists():
                qat_s = np.load(qat_path).astype(np.float64)
                L = min(len(fp32_s), len(qat_s))
                rho, _ = spearmanr(fp32_s[:L], qat_s[:L])
                all_rows.append({"seed": seed, "sample_id": sid,
                                  "comparison": "fp32_vs_qat_onnx_int4", "spearman_rho": float(rho)})

        n_seed = len([r for r in all_rows if r["seed"] == seed])
        print(f"  seed={seed}: {n_seed} correlation rows computed")

    if not all_rows:
        print("  [WARN] No rows computed.")
        return

    df = pd.DataFrame(all_rows)
    per_path = _RES_DIR / "cross_seed_verification.csv"
    df.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df)} rows -> {per_path}")

    seed42_means = {
        cmp: df[(df["seed"] == 42) & (df["comparison"] == cmp)]["spearman_rho"].mean()
        for cmp in df["comparison"].unique()
    }
    summary_rows = []
    for (cmp, seed), grp in df.groupby(["comparison", "seed"]):
        mean_rho = grp["spearman_rho"].mean()
        std_rho  = grp["spearman_rho"].std()
        ref      = seed42_means.get(cmp, float("nan"))
        delta    = abs(mean_rho - ref)
        verdict  = "CONSISTENT" if delta <= 0.10 else "VARIABLE"
        summary_rows.append({"comparison": cmp, "seed": seed,
                              "mean_rho": round(mean_rho, 4), "std_rho": round(std_rho, 4),
                              "delta_vs_seed42": round(delta, 4), "verdict": verdict})

    df_sum = pd.DataFrame(summary_rows)
    sum_path = _RES_DIR / "cross_seed_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary -> {sum_path}\n")
    for _, row in df_sum.iterrows():
        print(f"  {row['comparison']:30s}  seed={row['seed']}  "
              f"mean_rho={row['mean_rho']:.3f}  delta={row['delta_vs_seed42']:.3f}  -> {row['verdict']}")