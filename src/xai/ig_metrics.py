from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import warnings
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import spearmanr
from torch.nn.functional import softmax, cosine_similarity as torch_cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from src.xai.alignment import build_alignment, project_subword_to_word

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    k = max(1, k)
    idx_a = set(np.argsort(-a)[:k])
    idx_b = set(np.argsort(-b)[:k])
    return len(idx_a & idx_b) / float(k)

def integrated_gradients_tokens(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    target: int = None,
) -> Tuple[List[str], np.ndarray]:
    encoding = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    target_class = target if target is not None else pred_class

    emb_layer = model.bert.embeddings.word_embeddings

    def forward_fn(input_ids_: torch.Tensor):
        outputs = model(input_ids=input_ids_, attention_mask=attention_mask)
        return outputs.logits

    lig = LayerIntegratedGradients(forward_fn, emb_layer)

    try:
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            target=target_class,
            n_steps=30,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "not have been used in the graph" not in msg:
            raise
        warnings.warn(
            "Returning zero attributions.",
            RuntimeWarning,
        )
        attributions = torch.zeros_like(model.bert.embeddings.word_embeddings(input_ids))

    token_attrs = attributions.mean(dim=-1).squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    return tokens, token_attrs

def gradient_times_input_tokens(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    target: int = None,
) -> Tuple[List[str], np.ndarray]:
    encoding = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    target_class = target if target is not None else pred_class

    embeddings = model.bert.embeddings.word_embeddings(input_ids).detach().requires_grad_(True)

    logits = model(inputs_embeds=embeddings, attention_mask=attention_mask).logits

    model.zero_grad()
    logits[0, target_class].backward()

    gxi = (embeddings.grad * embeddings).sum(dim=-1).squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    return tokens, gxi

def attribution_similarity(
    model_a: AutoModelForSequenceClassification,
    model_b: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    top_k: int = 10,
) -> Dict[str, float]:
    tokens_a, attr_a = integrated_gradients_tokens(model_a, tokenizer, text)
    _, attr_b = integrated_gradients_tokens(model_b, tokenizer, text)

    L = min(len(attr_a), len(attr_b))
    attr_a = attr_a[:L]
    attr_b = attr_b[:L]

    return {
        "cosine": _cosine(attr_a, attr_b),
        "spearman": float(spearmanr(attr_a, attr_b).correlation),
        "topk_overlap": _topk_overlap(attr_a, attr_b, k=top_k),
    }

def auc(curve: Sequence[float]) -> float:
    if not curve:
        return 0.0
    x = np.linspace(0, 1, len(curve))
    return float(np.trapz(curve, x))

@dataclass
class InsertionDeletionResult:
    deletion_curve: List[float]
    insertion_curve: List[float]
    deletion_auc: float
    insertion_auc: float

class InsertionDeletionEvaluator:
    def __init__(self, tokenizer: AutoTokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    def _predict_conf(self, model: AutoModelForSequenceClassification, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)
        return float(probs.max().item())

    def _mask_token(self) -> str:
        return self.tokenizer.mask_token or "[MASK]"

    def _scores(self, model: AutoModelForSequenceClassification, tokens: List[str], order: np.ndarray) -> Tuple[List[float], List[float]]:
        deletion_probs = []
        masked_tokens = tokens.copy()
        mask_token = self._mask_token()

        for k in range(len(tokens)):
            for i in order[:k]:
                masked_tokens[i] = mask_token
            text = self.tokenizer.convert_tokens_to_string(masked_tokens)
            deletion_probs.append(self._predict_conf(model, text))

        insertion_probs = []
        masked_tokens = [mask_token] * len(tokens)
        for k in range(len(tokens)):
            for i in order[:k]:
                masked_tokens[i] = tokens[i]
            text = self.tokenizer.convert_tokens_to_string(masked_tokens)
            insertion_probs.append(self._predict_conf(model, text))

        return deletion_probs, insertion_probs

    def evaluate(
        self,
        model: AutoModelForSequenceClassification,
        tokens: List[str],
        attributions: np.ndarray,
    ) -> InsertionDeletionResult:
        order = np.argsort(-attributions)
        deletion_curve, insertion_curve = self._scores(model, tokens, order)

        return InsertionDeletionResult(
            deletion_curve=deletion_curve,
            insertion_curve=insertion_curve,
            deletion_auc=auc(deletion_curve),
            insertion_auc=auc(insertion_curve),
        )

def layer_cls_similarity(
    model_a: AutoModelForSequenceClassification,
    model_b: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(next(model_a.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out_a = model_a(**inputs, output_hidden_states=True)
    with torch.no_grad():
        out_b = model_b(**inputs, output_hidden_states=True)

    sims = []
    for h_a, h_b in zip(out_a.hidden_states, out_b.hidden_states):
        cls_a = h_a[:, 0, :]
        cls_b = h_b[:, 0, :]
        sims.append(float(torch_cosine(cls_a, cls_b, dim=-1).mean().item()))

    return sims

def run_ste_ig_analysis():
    import pandas as pd

    _INT2LABEL = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _SUBSAMPLE_CSV = _PROJECT_ROOT / "data" / "explainability_subsample_v2.csv"
    _FP32_DIR = _PROJECT_ROOT / "models" / "fp32_seed42"
    _QAT_DIR = _PROJECT_ROOT / "models" / "qat_seed42_with_observers"
    _OUT_DIR = _PROJECT_ROOT / "results" / "attributions"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _SUBSAMPLE_CSV.exists():
        print(f"  [ERROR] Subsample CSV not found: {_SUBSAMPLE_CSV}")
        print("  Run: python scripts/prepare_datasets.py  (requires models/fp32_seed42/predictions.csv)")
        return
    df_sub = pd.read_csv(_SUBSAMPLE_CSV)
    samples = [
        {"sample_id": int(row["sample_id"]), "text": row["text"],
         "expected": _INT2LABEL[int(row["true_label"])]}
        for _, row in df_sub.iterrows()
    ]
    print(f"\n  Loaded {len(samples)} samples from {_SUBSAMPLE_CSV.name}")

    from src.models import ModelManager as _MM
    print(f"\n  Loading FP32 model: {_FP32_DIR}")
    fp32_model = _MM.load_model(str(_FP32_DIR))
    fp32_model.model.eval()

    if not _QAT_DIR.exists():
        print(f"  [ERROR] QAT model directory not found: {_QAT_DIR}")
        print("  Run QAT training first (multiseed_qat saves qat_seed42_with_observers/).")
        return
    print(f"\n  Loading QAT model: {_QAT_DIR}")
    qat_model = _MM.load_model(str(_QAT_DIR))
    qat_model.model.eval()

    print("\n  Verifying fake quantizers in QAT model:")
    observer_names = [
        name for name, mod in qat_model.model.named_modules()
        if "FakeQuantize" in type(mod).__name__ or "Observer" in type(mod).__name__
    ]
    if observer_names:
        print(f"  Found {len(observer_names)} observer/fake-quantizer module(s):")
        for n in observer_names[:10]:
            print(f"    {n}")
        if len(observer_names) > 10:
            print(f"    ... and {len(observer_names) - 10} more")
    else:
        print("  [WARN] No FakeQuantize/Observer modules found. Gradients may not flow via STE.")

    tokenizer = fp32_model.tokenizer

    ig_fp32 = IntegratedGradientsExplainer(
        fp32_model.model, tokenizer, device=fp32_model.device, precision="fp32"
    )
    ig_qat = IntegratedGradientsExplainer(
        qat_model.model, tokenizer, device=qat_model.device, precision="qat_ste"
    )

    metadata_rows = []
    example_count = 0

    print(f"\n  Running IG + GxI on {len(samples)} samples (n_steps=30, MEAN aggregation)...\n")

    for idx, sample in enumerate(samples):
        sid = sample["sample_id"]
        text = sample["text"]
        expected = sample["expected"]

        alignment = build_alignment(text, tokenizer)

        res_fp32 = ig_fp32.explain(text, steps=30)
        words_fp32, ig_word_scores_fp32 = project_subword_to_word(
            res_fp32["tokens"], res_fp32["scores"], alignment, strategy="mean"
        )

        res_qat = ig_qat.explain(text, steps=30)
        _, ig_word_scores_qat = project_subword_to_word(
            res_qat["tokens"], res_qat["scores"], alignment, strategy="mean"
        )

        np.save(_OUT_DIR / f"ig_fp32_{sid}.npy", ig_word_scores_fp32)
        np.save(_OUT_DIR / f"ig_qat_ste_{sid}.npy", ig_word_scores_qat)

        gxi_tokens_fp32, gxi_sw_fp32 = gradient_times_input_tokens(
            fp32_model.model, tokenizer, text, target=res_fp32["predicted_class"]
        )
        _, gxi_word_scores_fp32 = project_subword_to_word(
            gxi_tokens_fp32, gxi_sw_fp32, alignment, strategy="mean"
        )

        gxi_tokens_qat, gxi_sw_qat = gradient_times_input_tokens(
            qat_model.model, tokenizer, text, target=res_qat["predicted_class"]
        )
        _, gxi_word_scores_qat = project_subword_to_word(
            gxi_tokens_qat, gxi_sw_qat, alignment, strategy="mean"
        )

        np.save(_OUT_DIR / f"gxi_fp32_{sid}.npy", gxi_word_scores_fp32)
        np.save(_OUT_DIR / f"gxi_qat_ste_{sid}.npy", gxi_word_scores_qat)

        metadata_rows.append({
            "sample_id": sid,
            "text": text,
            "expected": expected,
            "predicted_fp32": _INT2LABEL.get(res_fp32["predicted_class"], str(res_fp32["predicted_class"])),
            "predicted_qat_ste": _INT2LABEL.get(res_qat["predicted_class"], str(res_qat["predicted_class"])),
            "n_words": len(words_fp32),
            "ig_npy_fp32": f"ig_fp32_{sid}.npy",
            "ig_npy_qat_ste": f"ig_qat_ste_{sid}.npy",
            "gxi_npy_fp32": f"gxi_fp32_{sid}.npy",
            "gxi_npy_qat_ste": f"gxi_qat_ste_{sid}.npy",
        })

        if example_count < 3:
            example_count += 1
            print(f"  -- Example {example_count}: [{expected}] \"{text[:80]}\"")
            top5 = lambda ws, ss: sorted(zip(ws, ss), key=lambda x: -abs(x[1]))[:5]
            fmt = lambda pairs: "  ".join(f"{w}({s:+.3f})" for w, s in pairs)
            print(f"    IG  FP32  top-5: {fmt(top5(words_fp32, ig_word_scores_fp32))}")
            print(f"    IG  QAT   top-5: {fmt(top5(words_fp32, ig_word_scores_qat))}")
            print(f"    GxI FP32  top-5: {fmt(top5(words_fp32, gxi_word_scores_fp32))}")
            print(f"    GxI QAT   top-5: {fmt(top5(words_fp32, gxi_word_scores_qat))}")
            print()

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(samples)}] done")

    meta_path = _OUT_DIR / "ig_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False, encoding="utf-8")
    print(f"\n  Saved metadata -> {meta_path}")
    print(f"  Saved {len(metadata_rows)} x 4 .npy files in {_OUT_DIR}")
    print("  File pattern: ig_fp32_N.npy  ig_qat_ste_N.npy  gxi_fp32_N.npy  gxi_qat_ste_N.npy")
