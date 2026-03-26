from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_SPECIAL_TOKENS = frozenset({"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"})


def _is_special(token: str) -> bool:
    return token in _SPECIAL_TOKENS

def _mask_subword(
    tokens: List[str],
    mask_indices: Sequence[int],
    mask_token: str,
) -> List[str]:
    masked = list(tokens)
    for idx in mask_indices:
        if not _is_special(tokens[idx]):
            masked[idx] = mask_token
    return masked

def _get_confidence_for_class(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    class_idx: int,
    device: torch.device,
) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.float()
        probs  = softmax(logits, dim=-1)

    n_classes = probs.shape[-1]
    if class_idx >= n_classes:
        raise ValueError(
            f"class_idx={class_idx} exceeds model output size {n_classes}. "
            "The model and the attribution result must correspond to the same "
            "label space."
        )
    return float(probs[0, class_idx].item())

def _select_topk_indices(
    tokens: List[str],
    scores: Sequence[float],
    k: int,
    by_absolute: bool = True,
) -> List[int]:
    content = [
        (i, scores[i])
        for i in range(len(tokens))
        if not _is_special(tokens[i])
    ]

    if len(content) == 0:
        warnings.warn(
            "All tokens are special tokens; no content tokens available for "
            "top-k selection.  Returning empty index list.",
            UserWarning, stacklevel=3,
        )
        return []

    if len(content) < k:
        warnings.warn(
            f"Only {len(content)} content tokens available, but k={k} was "
            "requested.  Using all {len(content)} content tokens.  This "
            "sample's sufficiency/comprehensiveness value at k is not "
            "comparable with samples that have ≥ k content tokens.",
            UserWarning, stacklevel=3,
        )
        k = len(content)

    score_fn = (lambda x: abs(x[1])) if by_absolute else (lambda x: x[1])
    ranked = sorted(content, key=score_fn, reverse=True)

    top_indices = sorted(idx for idx, _ in ranked[:k])  # ascending: positional order
    return top_indices

def _build_sufficiency_text(
    tokens: List[str],
    top_indices: List[int],
    mask_token: str,
    tokenizer: AutoTokenizer,
) -> str:
    selected_set = set(top_indices)
    masked_tokens = _mask_subword(
        tokens,
        mask_indices=[i for i in range(len(tokens))
                      if i not in selected_set and not _is_special(tokens[i])],
        mask_token=mask_token,
    )
    return tokenizer.convert_tokens_to_string(
        [t for t in masked_tokens if not _is_special(t)]
    ).strip()


def _build_comprehensiveness_text(
    tokens: List[str],
    top_indices: List[int],
    mask_token: str,
    tokenizer: AutoTokenizer,
) -> str:
    masked_tokens = _mask_subword(tokens, mask_indices=top_indices, mask_token=mask_token)
    return tokenizer.convert_tokens_to_string(
        [t for t in masked_tokens if not _is_special(t)]
    ).strip()

def _build_sufficiency_text_word(
    words: List[str],
    top_indices: List[int],
    mask_token: str = "[MASK]",
) -> str:
    selected_set = set(top_indices)
    masked_words = [
        w if i in selected_set else mask_token
        for i, w in enumerate(words)
    ]
    return " ".join(masked_words).strip()


def _build_comprehensiveness_text_word(
    words: List[str],
    top_indices: List[int],
    mask_token: str = "[MASK]",
) -> str:
    selected_set = set(top_indices)
    masked_words = [
        mask_token if i in selected_set else w
        for i, w in enumerate(words)
    ]
    return " ".join(masked_words).strip()

@dataclass
class FaithfulnessAtK:
    k:                            int
    original_confidence:          float
    sufficiency_confidence:       float
    comprehensiveness_confidence: float
    sufficiency:                  float
    comprehensiveness:            float
    predicted_class:              int
    n_content_tokens:             int
    top_indices:                  List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "k":                            self.k,
            "original_confidence":          self.original_confidence,
            "sufficiency_confidence":       self.sufficiency_confidence,
            "comprehensiveness_confidence": self.comprehensiveness_confidence,
            "sufficiency":                  self.sufficiency,
            "comprehensiveness":            self.comprehensiveness,
            "predicted_class":              self.predicted_class,
            "n_content_tokens":             self.n_content_tokens,
            "top_indices":                  self.top_indices,
        }


@dataclass
class FaithfulnessResult:
    text:         str
    method:       str
    precision:    str
    per_k:        Dict[int, FaithfulnessAtK] = field(default_factory=dict)
    token_level:  str = "subword"

    def to_dict(self) -> dict:
        return {
            "text":         self.text,
            "method":       self.method,
            "precision":    self.precision,
            "token_level":  self.token_level,
            "per_k":        {str(k): v.to_dict() for k, v in self.per_k.items()},
        }

class FaithfulnessEvaluator:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        k_values: Tuple[int, ...] = (5, 10, 20),
        by_absolute: bool = True,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or next(model.parameters()).device
        self.k_values  = k_values
        self.by_absolute = by_absolute

        self._mask_token = tokenizer.mask_token or "[MASK]"

    def evaluate(
        self,
        text: str,
        tokens: List[str],
        scores: Sequence[float],
        method: str = "unknown",
        precision: str = "unknown",
        token_level: str = "subword",
    ) -> FaithfulnessResult:
        if token_level not in ("subword", "word"):
            raise ValueError(
                f"token_level must be 'subword' or 'word', got {token_level!r}."
            )

        scores_arr = np.array(scores, dtype=float)

        orig_inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        orig_inputs = {k: v.to(self.device) for k, v in orig_inputs.items()}
        with torch.no_grad():
            logits = self.model(**orig_inputs).logits.float()
            probs  = softmax(logits, dim=-1)[0]

        predicted_class   = int(probs.argmax().item())
        orig_confidence   = float(probs[predicted_class].item())

        per_k: Dict[int, FaithfulnessAtK] = {}

        n_content = sum(1 for t in tokens if not _is_special(t))

        for k in self.k_values:
            top_indices = _select_topk_indices(
                tokens, scores_arr, k, by_absolute=self.by_absolute
            )

            if token_level == "subword":
                suff_text  = _build_sufficiency_text(
                    tokens, top_indices, self._mask_token, self.tokenizer
                )
                comp_text  = _build_comprehensiveness_text(
                    tokens, top_indices, self._mask_token, self.tokenizer
                )
            else:
                suff_text  = _build_sufficiency_text_word(
                    tokens, top_indices, self._mask_token
                )
                comp_text  = _build_comprehensiveness_text_word(
                    tokens, top_indices, self._mask_token
                )

            if not suff_text.strip():
                warnings.warn(
                    f"Sufficiency text is empty for k={k} (text='{text[:50]}...'). "
                    "This happens when k ≥ all content tokens.  Assigning "
                    "sufficiency_confidence = NaN for this k.",
                    UserWarning, stacklevel=2,
                )
                suff_conf = float("nan")
            else:
                suff_conf = _get_confidence_for_class(
                    self.model, self.tokenizer, suff_text,
                    predicted_class, self.device,
                )

            if not comp_text.strip():
                warnings.warn(
                    f"Comprehensiveness text is empty for k={k} "
                    f"(text='{text[:50]}...').  Assigning NaN.",
                    UserWarning, stacklevel=2,
                )
                comp_conf = float("nan")
            else:
                comp_conf = _get_confidence_for_class(
                    self.model, self.tokenizer, comp_text,
                    predicted_class, self.device,
                )

            suff_val = suff_conf
            comp_val = float("nan") if np.isnan(comp_conf) else (orig_confidence - comp_conf)

            per_k[k] = FaithfulnessAtK(
                k=k,
                original_confidence=orig_confidence,
                sufficiency_confidence=suff_conf,
                comprehensiveness_confidence=comp_conf,
                sufficiency=suff_val,
                comprehensiveness=comp_val,
                predicted_class=predicted_class,
                n_content_tokens=n_content,
                top_indices=top_indices,
            )

        return FaithfulnessResult(
            text=text,
            method=method,
            precision=precision,
            per_k=per_k,
            token_level=token_level,
        )

def evaluate_faithfulness_batch(
    samples: List[dict],
    attribution_results: List[dict],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    method: str,
    precision: str,
    token_level: str = "subword",
    k_values: Tuple[int, ...] = (5, 10, 20),
    device: Optional[torch.device] = None,
) -> List[FaithfulnessResult]:
    if len(samples) != len(attribution_results):
        raise ValueError(
            f"samples ({len(samples)}) and attribution_results "
            f"({len(attribution_results)}) must have the same length."
        )

    evaluator = FaithfulnessEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        k_values=k_values,
    )

    was_training = model.training
    model.eval()

    results: List[FaithfulnessResult] = []
    try:
        for i, (sample, attr) in enumerate(zip(samples, attribution_results)):
            result = evaluator.evaluate(
                text=sample["text"],
                tokens=attr["tokens"],
                scores=attr["scores"],
                method=method,
                precision=precision,
                token_level=token_level,
            )
            results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"  [faithfulness] {i+1}/{len(samples)} samples processed")
    finally:
        if was_training:
            model.train()

    return results

def aggregate_faithfulness(
    results: List[FaithfulnessResult],
    k_values: Optional[Tuple[int, ...]] = None,
    min_content_tokens: int = 0,
) -> Dict[str, Dict[str, float]]:
    if not results:
        return {}

    if k_values is None:
        k_values = tuple(sorted(results[0].per_k.keys()))

    suff_by_k: Dict[int, List[float]] = {k: [] for k in k_values}
    comp_by_k: Dict[int, List[float]] = {k: [] for k in k_values}

    for r in results:
        for k in k_values:
            if k not in r.per_k:
                continue
            entry = r.per_k[k]

            if min_content_tokens > 0 and entry.n_content_tokens < min_content_tokens:
                continue

            suff_val = entry.sufficiency
            comp_val = entry.comprehensiveness

            if not np.isnan(suff_val):
                suff_by_k[k].append(suff_val)
            if not np.isnan(comp_val):
                comp_by_k[k].append(comp_val)

    def _stats(vals: List[float]) -> Dict[str, float]:
        n = len(vals)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(vals, dtype=float)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1) if n > 1 else 0.0),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "median": float(np.median(arr)),
            "n":      n,
        }

    output: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        output[f"suff_k{k}"] = _stats(suff_by_k[k])
        output[f"comp_k{k}"] = _stats(comp_by_k[k])

    return output

def compare_faithfulness(
    base_results: List[FaithfulnessResult],
    variant_results: List[FaithfulnessResult],
    base_label: str = "fp32",
    variant_label: str = "int8",
    k_values: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Dict[str, float]]:
    if len(base_results) != len(variant_results):
        warnings.warn(
            f"base_results ({len(base_results)}) and variant_results "
            f"({len(variant_results)}) have different lengths. "
            "Only paired samples (by index) will be compared.",
            UserWarning, stacklevel=2,
        )

    n_pairs = min(len(base_results), len(variant_results))

    if k_values is None:
        k_values = tuple(sorted(base_results[0].per_k.keys()))

    delta_suff: Dict[int, List[float]] = {k: [] for k in k_values}
    delta_comp: Dict[int, List[float]] = {k: [] for k in k_values}

    for i in range(n_pairs):
        b = base_results[i]
        v = variant_results[i]
        for k in k_values:
            if k not in b.per_k or k not in v.per_k:
                continue
            bs = b.per_k[k].sufficiency
            vs = v.per_k[k].sufficiency
            bc = b.per_k[k].comprehensiveness
            vc = v.per_k[k].comprehensiveness

            if not (np.isnan(bs) or np.isnan(vs)):
                delta_suff[k].append(vs - bs)
            if not (np.isnan(bc) or np.isnan(vc)):
                delta_comp[k].append(vc - bc)

    def _stats(vals: List[float]) -> Dict[str, float]:
        n = len(vals)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(vals, dtype=float)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1) if n > 1 else 0.0),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "median": float(np.median(arr)),
            "n":      n,
        }

    output: Dict[str, Dict[str, float]] = {
        "base_label":    base_label,
        "variant_label": variant_label,
        "n_pairs":       n_pairs,
    }
    for k in k_values:
        output[f"delta_suff_k{k}"] = _stats(delta_suff[k])
        output[f"delta_comp_k{k}"] = _stats(delta_comp[k])

    return output

import json as _json
from pathlib import Path as _Path


def save_faithfulness_results(
    results: List[FaithfulnessResult],
    output_path: _Path,
    *,
    metadata: Optional[dict] = None,
) -> None:
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata":  metadata or {},
        "n_samples": len(results),
        "results":   [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)


def load_faithfulness_results(path: _Path) -> Tuple[List[dict], dict]:
    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Faithfulness results file not found: {path}\n"
            "Run evaluate_faithfulness_batch() and save_faithfulness_results() first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    return payload["results"], payload.get("metadata", {})

class _OnnxTorchAdapter:
    def __init__(self, onnx_model):
        self._session = onnx_model.session
        self.training = False

    def parameters(self):
        import torch as _torch
        yield _torch.zeros(1, device="cpu")

    def eval(self):  self.training = False; return self
    def train(self): self.training = True;  return self

    def __call__(self, **kwargs):
        import torch as _torch
        ids  = kwargs["input_ids"].cpu().numpy().astype("int64")
        mask = kwargs["attention_mask"].cpu().numpy().astype("int64")
        logits_np = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        class _R: logits = None
        r = _R()
        r.logits = _torch.tensor(logits_np, dtype=_torch.float32)
        return r

def run_faithfulness_evaluation():
    import pandas as pd
    from src.models import ModelManager
    from src.models.base import BaseModel, OnnxBaseModel
    from src.quantization.ptq import PTQQuantizer
    from src.config import LABELS

    _PROJECT_ROOT  = _Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _OUT_DIR       = _PROJECT_ROOT / "results" / "attributions"
    _RES_DIR       = _PROJECT_ROOT / "results"
    _FP32_DIR      = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_CLEAN_DIR = _PROJECT_ROOT / "models" / "qat_seed42_clean"
    _MODELS_DIR    = _PROJECT_ROOT / "models"

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample not found: {_SUBSAMPLE_CSV}")
        return
    df_sub  = pd.read_csv(_SUBSAMPLE_CSV)
    samples = list(df_sub.itertuples(index=False))

    print(f"\n  Loading FP32 base: {_FP32_DIR}")
    fp32_base = ModelManager.load_model(str(_FP32_DIR))
    fp32_base.model.eval()

    def _load_onnx(variant):
        import onnxruntime as ort
        onnx_dir  = _MODELS_DIR / f"qat_onnx_{variant}_seed42"
        onnx_file = onnx_dir / f"model_qat_{variant}.onnx"
        if not onnx_file.exists():
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
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

    METHODS = ["lime", "occ", "shap"]
    per_rows = []

    for method in METHODS:
        for vname, load_fn, _ in VARIANTS:
            model = load_fn()
            if model is None:
                print(f"  [SKIP] {method} x {vname}")
                continue

            if isinstance(model, OnnxBaseModel):
                infer_model = _OnnxTorchAdapter(model)
                device = torch.device("cpu")
            else:
                infer_model = model.model
                device = model.device

            evaluator = FaithfulnessEvaluator(
                infer_model, fp32_base.tokenizer, device=device, k_values=(5,)
            )

            n_found = 0
            for sample in samples:
                sid  = int(sample.sample_id)
                text = sample.text
                npy_path = _OUT_DIR / f"{method}_{vname}_{sid}.npy"
                if not npy_path.exists():
                    continue
                words  = text.split()
                scores = np.load(npy_path).astype(np.float64)
                L = min(len(words), len(scores))
                if L < 1:
                    continue
                try:
                    result = evaluator.evaluate(
                        text, words[:L], scores[:L].tolist(),
                        method=method, precision=vname, token_level="word",
                    )
                    fk = result.per_k[5]
                    per_rows.append({
                        "method": method, "variant": vname, "sample_id": sid,
                        "orig_conf": round(fk.original_confidence, 4),
                        "suff":      fk.sufficiency       if not np.isnan(fk.sufficiency)       else float("nan"),
                        "comp":      fk.comprehensiveness if not np.isnan(fk.comprehensiveness) else float("nan"),
                        "n_content": fk.n_content_tokens,
                        "k":         fk.k,
                    })
                    n_found += 1
                except Exception as exc:
                    print(f"  [ERR] {method} {vname} sid={sid}: {exc}")

            print(f"  {method:5s} x {vname:20s}: {n_found} samples")

    if not per_rows:
        print("  [WARN] No .npy files found. Run LIME/OCC/SHAP steps first.")
        return

    df_per = pd.DataFrame(per_rows)
    per_path = _RES_DIR / "faithfulness_perSample.csv"
    df_per.to_csv(per_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(df_per)} rows -> {per_path}")

    sum_rows = []
    for (method, vname), grp in df_per.groupby(["method", "variant"]):
        sum_rows.append({
            "method": method, "variant": vname, "n": len(grp),
            "mean_suff": round(float(grp["suff"].dropna().mean()), 4),
            "std_suff":  round(float(grp["suff"].dropna().std()),  4),
            "mean_comp": round(float(grp["comp"].dropna().mean()), 4),
            "std_comp":  round(float(grp["comp"].dropna().std()),  4),
        })
    df_sum = pd.DataFrame(sum_rows)
    sum_path = _RES_DIR / "faithfulness_summary.csv"
    df_sum.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"  Saved summary -> {sum_path}")

    print(f"\n  {'method':5s}  {'variant':20s}  {'mean_suff':>9s}  {'mean_comp':>9s}")
    for _, row in df_sum.sort_values(["method", "variant"]).iterrows():
        print(f"  {row['method']:5s}  {row['variant']:20s}  "
              f"{row['mean_suff']:9.4f}  {row['mean_comp']:9.4f}")
