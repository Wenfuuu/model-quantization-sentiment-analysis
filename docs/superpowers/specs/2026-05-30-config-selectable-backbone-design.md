# Config-selectable backbone for pathway-vs-bitwidth study

**Status:** Draft for review
**Date:** 2026-05-30
**Author:** brainstorming session with user

## 1. Goal

Enable swapping the pretrained backbone (currently hardcoded `indobenchmark/indobert-base-p2`) via a single environment variable so the QAT-vs-PTQ-vs-XAI study can be re-run on a second Indonesian-capable architecture (e.g. `indobenchmark/indobert-lite-base-p2`, `xlm-roberta-base`, `bert-base-multilingual-cased`). IndoBERT remains the default; existing artifacts on disk are untouched.

### Non-goals

- Per-backbone hyperparameter overrides (`max_length`, `batch_size`). Use CLI args until a backbone forces a change.
- Touching `scripts/finetune_smsa_fp32.py` (deprecated Sastrawi variant — kept hardcoded to IndoBERT per `CLAUDE.md`).
- Migrating any existing IndoBERT artifacts on disk.
- A CLI `--model` flag — env var only.
- A new top-level menu number — switching backbones happens before launching `main.py`, not inside it.

## 2. Selection mechanism

`src/config.py` gains a model registry, default tag, and env-var-driven active tag:

```python
import os

MODEL_REGISTRY = {
    "indobert": {
        "hf_id": "indobenchmark/indobert-base-p2",
        "display_name": "IndoBERT base p2",
    },
    "indobert-lite": {
        "hf_id": "indobenchmark/indobert-lite-base-p2",
        "display_name": "IndoBERT-lite base p2",
    },
    "xlmr": {
        "hf_id": "xlm-roberta-base",
        "display_name": "XLM-RoBERTa base",
    },
    "mbert": {
        "hf_id": "bert-base-multilingual-cased",
        "display_name": "mBERT base (cased)",
    },
}

DEFAULT_MODEL_TAG = "indobert"
MODEL_TAG = os.environ.get("MQSA_MODEL", DEFAULT_MODEL_TAG)

if MODEL_TAG not in MODEL_REGISTRY:
    raise ValueError(
        f"MQSA_MODEL={MODEL_TAG!r} not in registry "
        f"({sorted(MODEL_REGISTRY)}). Edit src/config.py to add it."
    )

BASE_MODEL_HF_ID = MODEL_REGISTRY[MODEL_TAG]["hf_id"]
BASE_MODEL_DISPLAY_NAME = MODEL_REGISTRY[MODEL_TAG]["display_name"]
```

Switch backbones with `MQSA_MODEL=xlmr python main.py`. Default invocation is unchanged. Adding a new backbone is a one-line entry in `MODEL_REGISTRY`.

## 3. Path namespacing

Two helpers in `src/config.py`, applied uniformly to checkpoints and outputs:

```python
def tagged_model_dir(name: str) -> Path:
    """models/<name>          (default tag)
       models/{tag}_<name>    (non-default)"""
    if MODEL_TAG == DEFAULT_MODEL_TAG:
        return BASE_DIR / "models" / name
    return BASE_DIR / "models" / f"{MODEL_TAG}_{name}"

def tagged_output_dir(*parts: str) -> Path:
    """outputs/<parts...>            (default tag)
       outputs/{tag}/<parts...>      (non-default)"""
    if MODEL_TAG == DEFAULT_MODEL_TAG:
        return BASE_DIR / "outputs" / Path(*parts)
    return BASE_DIR / "outputs" / MODEL_TAG / Path(*parts)
```

Behavior:

| Tag | Checkpoint example | Output example |
|---|---|---|
| `indobert` (default) | `models/fp32_seed42/` | `outputs/finetuned-smsa/` |
| `xlmr` | `models/xlmr_fp32_seed42/` | `outputs/xlmr/finetuned-smsa/` |

Existing default-tag paths are preserved byte-for-byte. Non-default tags produce siblings that never collide with IndoBERT artifacts.

Cosmetic note: subdir names that historically contained the substring "indobert" (e.g. `indobert-qat-int8-smsa/`) are kept as-is — the parent `outputs/{tag}/` disambiguates ownership. The QAT/PTQ report headers (which use `BASE_MODEL_DISPLAY_NAME`) make the active model unambiguous in human-facing rendering.

## 4. Config dict rewiring

The four existing dicts (`SEEDED_MODEL_DIRS`, `SEEDED_CONTROL_MODEL_DIRS`, `EXPERIMENT_CONFIGS`, `QAT_EXPERIMENT_CONFIGS`) are rewritten to use the helpers at module load:

```python
SEEDED_MODEL_DIRS = {
    seed: tagged_model_dir(f"fp32_seed{seed}") for seed in TRAINING_SEEDS
}
SEEDED_CONTROL_MODEL_DIRS = {
    seed: tagged_model_dir(f"fp32_control_seed{seed}") for seed in TRAINING_SEEDS
}

EXPERIMENT_CONFIGS = {
    "original_smsa": {
        "model_id": BASE_MODEL_HF_ID,
        "dataset": "smsa",
        "output_dir": tagged_output_dir("original-smsa"),
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "finetuned_smsa": {
        "model_id": str(tagged_model_dir("fp32_seed42")),
        "dataset": "smsa",
        "output_dir": tagged_output_dir("finetuned-smsa"),
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
    "fp32_control_smsa": {
        "model_id": str(tagged_model_dir("fp32_control_seed42")),
        "dataset": "smsa",
        "output_dir": tagged_output_dir("fp32-control-smsa"),
        "num_inference_runs": 20,
        "warmup_runs": 5,
    },
}

QAT_EXPERIMENT_CONFIGS = {
    "qat_eager_smsa": {
        "model_paths": {
            "int8": str(tagged_output_dir("indobert-qat-int8-smsa") / "hf_model"),
            "fp16": str(tagged_output_dir("indobert-qat-fp16-smsa") / "hf_model"),
            "int4": str(tagged_output_dir("indobert-qat-int4-smsa") / "hf_model"),
        },
        "dataset": "smsa",
        "output_dir": tagged_output_dir("qat-eager-smsa"),
    },
    "fp32_control_smsa": {
        "model_paths": {
            seed: str(SEEDED_CONTROL_MODEL_DIRS[seed]) for seed in TRAINING_SEEDS
        },
        "dataset": "smsa",
        "output_dir": tagged_output_dir("fp32-control-smsa"),
        "description": (
            "Continued FP32 fine-tune control (no fake-quant) — matches the "
            "QAT extra-training schedule (epochs/lr/batch/seq-len/optimizer/"
            "embedding handling) so the QAT-FP32 stability drop can be split "
            "into 'extra training' vs 'fake-quant gradient reshaping'."
        ),
    },
}
```

`LEGACY_MODEL_DIR` / `FP32_MODEL_DIR` remain un-namespaced — they refer to a single legacy single-seed checkpoint that pre-dates multi-seed work and is read only by `scripts/evaluate_models.py`. No need to touch.

`PTQ_MODEL_PATH` and `QAT_MODEL_PATH` (module-level legacy paths) are migrated to the helper:

```python
PTQ_MODEL_PATH = tagged_output_dir("finetuned-smsa", "ptq_int8.pth")
QAT_MODEL_PATH = tagged_output_dir("indobert-qat-int8-smsa", "qat_trained.pt")
```

## 5. Finetune script

`scripts/finetune_smsa_fp32_no_sw.py` changes:

- Remove local `MODEL_ID = "indobenchmark/indobert-base-p2"` constant.
- Import from config:
  ```python
  from src.config import (
      BASE_MODEL_HF_ID,
      BASE_MODEL_DISPLAY_NAME,
      DEVICE,
      tagged_model_dir,
  )
  ```
- All `from_pretrained(MODEL_ID)` calls → `from_pretrained(BASE_MODEL_HF_ID)`.
- Banner log line `print(f"  Model:   {MODEL_ID}")` → `print(f"  Model:   {BASE_MODEL_DISPLAY_NAME} ({BASE_MODEL_HF_ID})")`.
- Results JSON's `"model_id"` field stores `BASE_MODEL_HF_ID` (so the saved record names the actual backbone, not the registry tag).
- `main()`'s save-dir branches become tag-aware:
  - `SAVE_BASE / f"fp32_seed{seed}"` → `tagged_model_dir(f"fp32_seed{seed}")`
  - `SAVE_BASE / f"fp32_control_seed{seed}"` → `tagged_model_dir(f"fp32_control_seed{seed}")`
  - The sidecar `SAVE_BASE / f"fp32_control_seed{seed}.pt"` becomes `tagged_model_dir(f"fp32_control_seed{seed}").parent / f"{tagged_model_dir(f'fp32_control_seed{seed}').name}.pt"` — i.e. the prefix flows into the sidecar filename, never colliding across tags.
- `--save-dir` CLI override semantics are unchanged (it's an explicit user override).

`scripts/finetune_smsa_fp32.py` (deprecated Sastrawi variant) is **not modified** — per CLAUDE.md it is the deprecated baseline.

## 6. Multi-seed orchestrator

`scripts/finetune_multi_seed.py` changes:

- Replace local `_CKPT_TEMPLATE = "fp32_seed{seed}"` with importing `tagged_model_dir` from `src.config`.
- `seed_checkpoint_dir(seed, ckpt_suffix)` becomes a thin wrapper over `tagged_model_dir(f"fp32_seed{seed}")`. The `ckpt_suffix` argument is preserved as a no-op (matches current behavior — the existing function accepts the arg but never uses it; preserved for API compatibility with callers that pass it positionally).
- `_AGG_OUTPUT_DIR` and `_AGG_OUTPUT_FILE` route through `tagged_output_dir("multi-seed")` and `tagged_output_dir("multi-seed", "aggregated_finetune_results.json")` so per-tag multi-seed summaries don't collide.

## 7. ModelManager fallback

`src/models/manager.py:22` — `_infer_base_model_id`'s hardcoded fallback `"indobenchmark/indobert-base-p2"` becomes `BASE_MODEL_HF_ID` (imported from `src.config`). This matters for QAT checkpoints whose `config.json` lacks `_name_or_path`: the fallback now follows the active backbone instead of always returning IndoBERT.

Rationale: a non-IndoBERT QAT checkpoint with a missing `_name_or_path` would silently load as IndoBERT under the current code — silent corruption. Tying the fallback to `BASE_MODEL_HF_ID` makes the failure mode explicit (it loads the *active* backbone, which is at least the user's stated intent at process startup).

## 8. run_*.py orchestrators

Audit each of `scripts/run_ptq.py`, `scripts/run_qat.py`, `scripts/run_xai.py`, `scripts/run_stress_test.py` for:

1. Hardcoded `"indobenchmark/indobert-base-p2"` strings used in `from_pretrained(...)` calls → replace with `BASE_MODEL_HF_ID` from `src.config`.
2. Hardcoded paths reconstructing `outputs/finetuned-smsa/...` or `outputs/indobert-qat-*/...` independently of `EXPERIMENT_CONFIGS` → route through `tagged_output_dir(...)`.
3. Display strings referencing "IndoBERT" in runtime log lines → replace with `BASE_MODEL_DISPLAY_NAME`. Static menu labels (cosmetic strings in `interactive_menu()`) may be left alone since they're shown before the model is loaded, but log lines emitted after model load should reflect the active backbone.

The implementation plan will enumerate the exact file:line touch points after a `grep` pass. Expected scope: small (a handful of sites per script).

## 9. Documentation

`CLAUDE.md` gains a new short subsection under "Conventions and gotchas":

> **Selectable backbone.** The active pretrained backbone is controlled by `MODEL_REGISTRY` + `MODEL_TAG` in `src/config.py`. Default tag is `indobert` → `indobenchmark/indobert-base-p2`. Override with `MQSA_MODEL=<tag> python main.py` (e.g. `MQSA_MODEL=xlmr`). Non-default tags prefix checkpoints (`models/{tag}_fp32_seed{N}/`) and parent-dir-namespace outputs (`outputs/{tag}/...`), so artifacts never collide. The default tag preserves all existing un-prefixed paths verbatim. To add a backbone, append one entry to `MODEL_REGISTRY` — no other code changes needed.

## 10. Validation plan (no test suite — manual)

1. **Smoke test: default tag is unchanged.**
   `python -c "from src.config import EXPERIMENT_CONFIGS, SEEDED_MODEL_DIRS; print(EXPERIMENT_CONFIGS['finetuned_smsa']['output_dir']); print(SEEDED_MODEL_DIRS[42])"`
   Expected: `.../outputs/finetuned-smsa` and `.../models/fp32_seed42` — no tag prefix.

2. **Smoke test: non-default tag.**
   `MQSA_MODEL=xlmr python -c "..."` — same expression.
   Expected: `.../outputs/xlmr/finetuned-smsa` and `.../models/xlmr_fp32_seed42`.

3. **Invalid tag is rejected.**
   `MQSA_MODEL=bogus python -c "import src.config"` → `ValueError` mentioning the registry.

4. **1-epoch finetune on a non-default backbone.**
   `MQSA_MODEL=xlmr python scripts/finetune_multi_seed.py --seeds 42 --epochs 1 --lr 2e-5 --batch-size 16`
   Verify checkpoint lands at `models/xlmr_fp32_seed42/` and is loadable via `ModelManager.load_model(str(...))`.

5. **`BaseModel.predict` shape check.**
   On the loaded XLM-R checkpoint, run one inference and confirm `{label, confidence, prob, latency}` matches the IndoBERT contract.

6. **PTQ-INT8 round-trip.**
   `MQSA_MODEL=xlmr python main.py` → `[1] PTQ` → INT8 path. Confirm eager INT8 CPU quantization succeeds on the new architecture (every candidate backbone uses `nn.Linear` for the affected layers) and writes to `outputs/xlmr/finetuned-smsa/ptq_int8.pth` (or analogous per the existing PTQ output convention).

7. **Small-N stability pass.**
   `MQSA_MODEL=xlmr` → `[3] XAI` → run a small attribution + stability check. Confirm output lands under `outputs/xlmr/...` with no path collision against `outputs/finetuned-smsa/...`.

8. **Tree diff.**
   After step 7, `ls outputs/` should show `outputs/xlmr/` as the only new entry. Existing IndoBERT directories are untouched.

## 11. Risk register

| Risk | Mitigation |
|---|---|
| Future code adds `models/{tag}_...` f-strings directly, drifting from the helper | Centralize in `tagged_model_dir` / `tagged_output_dir`; mention the rule in CLAUDE.md |
| The "default tag has no prefix" special case is forgotten by future-me | One-line CLAUDE.md note + the helpers are the only correct way to construct paths |
| A non-IndoBERT QAT checkpoint with no `_name_or_path` loads as the wrong backbone | `_infer_base_model_id` fallback follows `BASE_MODEL_HF_ID`, so it tracks the active session intent |
| An XLM-R or mBERT model has tokenizer or `Linear`-layer differences that break eager INT8 PTQ | Validated explicitly in step 6 of the validation plan; if it breaks, the failure is local to that backbone, not to the framework |
| Researcher accidentally runs `MQSA_MODEL=xlmr` and writes to default IndoBERT artifact dirs | Cannot happen — non-default tag forces namespacing at the helper level; cross-contamination requires editing `tagged_model_dir` itself |

## 12. Out of scope (explicit YAGNI)

- Per-backbone `max_length` / `batch_size` overrides
- Touching `scripts/finetune_smsa_fp32.py`
- Migrating existing artifacts on disk
- CLI `--model` flag
- New menu number
- Cross-backbone comparison reporting (a downstream concern, not a config concern)
