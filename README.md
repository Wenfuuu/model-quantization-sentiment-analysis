# Quantization Effects on Explainability in Indonesian Transformer-Based Sentiment Analysis

Research codebase for studying what happens to a transformer's **explanations** when the model is
compressed. Most quantization papers report accuracy, latency and model size. This one asks a
different question: after you quantize an Indonesian sentiment classifier, does it still give the
same *reasons* for its predictions?

Accepted at **ICCSCI 2025**.

**Authors:** Helena Aurelia Sanjaya, Marvel Collin, Bertrand Geraldo Tjahyadi, Prof. Derwin Suhartono.

---

## Table of contents

- [What this repository does](#what-this-repository-does)
- [Experimental design](#experimental-design)
- [Model variants under test](#model-variants-under-test)
- [Metrics](#metrics)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Full experimental pipeline](#full-experimental-pipeline)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Outputs](#outputs)
- [Known findings and caveats](#known-findings-and-caveats)
- [Citation](#citation)

---

## What this repository does

The pipeline takes IndoBERT (`indobenchmark/indobert-base-p2`), fine-tunes it on the SmSA
three-class sentiment task from IndoNLU, then produces eight deployment variants across
post-training quantization (PTQ) and quantization-aware training (QAT). Every variant is measured on
four axes:

1. **Task performance.** Accuracy, macro F1, per-class F1, latency, on-disk size.
2. **Calibration.** Expected calibration error (ECE) over ten confidence bins.
3. **Explanation stability.** Do LIME, SHAP, Occlusion and Integrated Gradients still rank the same
   tokens as important after quantization? Measured with Spearman rank correlation, top-k Jaccard,
   sign-flip rate and normalised magnitude shift, all bootstrapped and Bonferroni corrected.
4. **Explanation faithfulness.** Sufficiency and comprehensiveness at k, so that a "stable"
   explanation is not just stably wrong.

On top of that sit robustness stress tests (character noise, word repetition, linguistic edge
cases), a curated Indonesian linguistic probe suite (negation scope, reduplication, hedging,
contrastive markers), statistical significance testing, and a deployment recommender that ranks the
eight variants under configurable constraints.

Everything runs across three training seeds (42, 123, 456) so that quantization effects can be
separated from seed noise. That separation is the point: a QAT run changes both the training
schedule *and* the gradient path, so the repository also trains an FP32 continued fine-tuning
control with a matched schedule and no fake quantization, which lets the observed drift be split
into "extra training" versus "fake-quant gradient reshaping".

---

## Experimental design

```
                      SmSA train/valid/test  (IndoNLU, 3 classes)
                                   |
                    fine-tune IndoBERT FP32, seeds 42/123/456
                                   |
              +--------------------+---------------------+
              |                                          |
        PTQ (no retraining)                   QAT (retrain with fake quant)
              |                                          |
     FP16 / INT8 / INT4                    QAT-FP32  -->  ONNX export
                                                          |
                                             ONNX FP16 / INT8 / INT4
              |                                          |
              +--------------------+---------------------+
                                   |
             evaluate: accuracy, F1, latency, size, ECE
             explain:  LIME, SHAP, Occlusion, IG, SmoothGrad, attention
             compare:  Spearman, Jaccard@k, sign flips, faithfulness
             stress:   noise, edge cases, 29 linguistic probes
             decide:   deployment recommendation under constraints
```

**Fine-tuning defaults:** 3 epochs, learning rate 2e-5, batch size 16, max sequence length 128,
AdamW with weight decay 0.01, linear warmup schedule, best checkpoint selected on validation F1.

**Preprocessing note.** Two fine-tuning scripts exist on purpose.
`scripts/finetune_smsa_fp32.py` applies Sastrawi stopword removal; `finetune_smsa_fp32_no_sw.py`
does not, keeping only lowercasing and whitespace normalisation. The no-stopword variant is the one
wired into the multi-seed pipeline, because stripping Indonesian stopwords deletes exactly the
negation and hedging tokens that the explainability analysis needs to observe.

---

## Model variants under test

| Variant | How it is produced | Notes |
| --- | --- | --- |
| `fp32` | Standard fine-tune | Baseline for every comparison |
| `ptq_fp16` | `.half()` cast of the FP32 weights | Size halves, no retraining |
| `ptq_int8` | `torch.quantization.quantize_dynamic` over `nn.Linear` | The usual production default |
| `ptq_int4` | Custom symmetric per-tensor INT4, `nn.Linear` replaced by `INT4Linear` | Weights stored as INT8-backed 4-bit codes, dequantised on the fly, so this measures the *accuracy* effect of 4-bit weights rather than real 4-bit kernels |
| `qat_fp32` | Fake-quant training, then converted back | Isolates the training-time effect |
| `qat_onnx_fp16` | QAT then ONNX export then FP16 conversion | Type-mismatch prone, see caveats |
| `qat_onnx_int8` | QAT then ONNX export then dynamic INT8 | The only path that produced real compression *and* real speedup |
| `qat_onnx_int4` | QAT then ONNX export then 4-bit weight quantization | Most aggressive setting |

A ninth configuration, `fp32_control`, is the continued FP32 fine-tune with the QAT schedule and no
fake quantization. It is not a deployment candidate; it exists so QAT effects can be attributed
correctly.

Three backbones are registered and swappable through `MODEL_TAG`: `indobert` (default),
`xlm-roberta` (`FacebookAI/xlm-roberta-base`) and `mbert`
(`google-bert/bert-base-multilingual-cased`).

---

## Metrics

### Task and deployment

| Metric | Where | Meaning |
| --- | --- | --- |
| Accuracy, macro F1, per-class F1 | `src/evaluation/metrics.py`, `per_class_analysis.py` | Standard classification quality |
| Latency (mean, std) | `src/evaluation/evaluator.py` | 20 timed inference runs per sample after 5 warmup runs |
| Model size (MB) | `src/quantization/utils.py` | On-disk parameter footprint |
| ECE (10 bins) | `src/evaluation/calibration.py` | Confidence versus correctness gap |

### Explanation stability

| Metric | Function | Reads as |
| --- | --- | --- |
| Spearman rho | `spearman_rank_correlation` | Rank agreement between FP32 and quantized attributions |
| Top-k Jaccard | `top_k_jaccard` | Overlap of the k most important tokens |
| Sign-flip rate | `sign_flip_rate` | Fraction of tokens whose attribution changed direction |
| Normalised magnitude shift | `normalized_magnitude_shift` | How much attribution mass moved |

All four are bootstrapped (`bootstrap_mean_ci`, `_scipy_bootstrap_ci`) and tested with Wilcoxon
signed-rank plus Bonferroni correction (`src/utils/stats_utils.py`), with Cohen's d and rank-biserial
effect sizes reported alongside p values. `compute_power_analysis` reports whether a given cell had
enough samples to make its non-significant result meaningful.

### Explanation faithfulness

`src/evaluation/faithfulness.py` implements sufficiency and comprehensiveness at k. Sufficiency
keeps only the top-k attributed tokens and asks whether the prediction survives. Comprehensiveness
removes them and asks whether the prediction collapses. Both operate at word level with subword
alignment handled by `src/xai/alignment.py`, and both are compared against a random attribution
floor (`src/xai/random_baseline.py`, 30 draws) so that a score is only reported as real if it clears
what random token selection would achieve.

### Robustness

- **Input noise:** character-level perturbation and word repetition at increasing rates
  (`src/stress/perturbation.py`).
- **Linguistic edge cases:** malformed, empty, very long and code-mixed inputs.
- **Calibration under stress:** ECE recomputed on perturbed inputs.
- **Probe Success Rate (PSR):** 29 hand-written Indonesian minimal-pair probes across eight
  phenomena, with McNemar tests between variants and explicit low-power warnings for undersized
  cells (`src/evaluation/evaluate_psr.py`).

The eight probe phenomena: verbal negation, nominal negation, double negation, scope-sensitive
negation, scalar intensifiers, morphological reduplication, contrastive discourse markers and
epistemic hedging.

---

## Repository layout

```
main.py                        Interactive entry point, six top-level menus
run.sh                         Pull latest main, then launch main.py
requirements.txt

scripts/                       Runnable experiment drivers
  prepare_datasets.py          SmSA reprocessing, CASA/HoASA cross-domain sets,
                               NusaX-senti prep, explainability subsample
  finetune_smsa_fp32.py        Fine-tune with Sastrawi stopword removal
  finetune_smsa_fp32_no_sw.py  Fine-tune without stopword removal (the one used)
  finetune_multi_seed.py       Runs the above across seeds, aggregates results
  run_ptq.py                   PTQ experiments, single model or multi-seed
  run_qat.py                   QAT eager, multi-seed QAT, ONNX pipeline, FP32 control
  run_xai.py                   14 explainability sub-experiments
  run_qat_xai.py               XAI on QAT checkpoints specifically
  run_stress_test.py           Robustness suite and linguistic probes
  run_attribution_comparison.py  Side-by-side attribution plots across variants
  run_k_sensitivity.py         Sensitivity of Jaccard results to the choice of k
  evaluate_models.py           Quick FP32 / PTQ / QAT comparison
  evaluate_on_dataset.py       Evaluate seeded checkpoints on a named dataset

src/
  config.py                    Single source of truth: paths, seeds, registries,
                               deployment thresholds
  data/loader.py               Dataset loading and eval-set switching
  models/                      Backbone wrapper and model manager
  quantization/
    ptq/                       fp16, int8, int4 quantizers, engine, multiseed driver
    qat/                       eager trainer, fake-quant trainer, ONNX export and
                               INT8/FP16/INT4 conversion, QAT config
    utils.py                   Save/load and size measurement
  training/fp32_trainer.py     Baseline trainer
  xai/
    lime_explainer.py          LIME
    shap_explainer.py          SHAP
    integrated_gradients.py    IG (Captum)
    occlusion.py               Occlusion, window size 1
    smoothgrad.py              SmoothGrad with sigma estimation
    attention_analysis.py      Attention rollout, entropy, cross-variant comparison
    ig_metrics.py              Insertion/deletion AUC, layer CLS similarity
    alignment.py               Word to subword projection, fragmentation reporting
    random_baseline.py         Random attribution floor
  evaluation/
    evaluator.py               Accuracy and latency harness
    metrics.py                 Metrics and significance tests
    calibration.py             ECE
    explanation_drift.py       The core stability analysis, 1.8k lines
    faithfulness.py            Sufficiency and comprehensiveness
    stress_test.py             Edge cases, noise, calibration under stress
    evaluate_psr.py            Probe success rate and McNemar
    per_class_analysis.py      Per-class breakdowns
    deployment_recommendation.py  Constraint-based ranking of the eight variants
  stress/perturbation.py       Perturbation primitives
  utils/                       Seeding, logging, seed aggregation, stats helpers
  visualization/               Plots and generated reports

datasets/
  train.tsv valid.tsv test.tsv SmSA splits (11,000 / 1,260 / 1,000 rows)
  linguistic_probes.py         29 probe pairs across 8 phenomena
  create_combined_dataset.py
  INA_TweetsPPKM_Labeled_Pure.csv  Indonesian PPKM tweets, secondary eval set
  alvin_hanafie/  deniyulian/  smsa/  tiktok_shop/   EDA notebooks and raw sources

research-recap/                Dated supervision notes and progress records
```

Generated artefacts (`models/`, `outputs/`, `results/`, `data/`, `*.onnx`, `*.pt`) are gitignored.
Only code, source datasets and research notes are tracked.

---

## Installation

Python 3.9 or newer, and a CUDA-capable GPU if you intend to fine-tune.

```bash
git clone https://github.com/Wenfuuu/model-quantization-sentiment-analysis.git
cd model-quantization-sentiment-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use `.venv\Scripts\activate`.

Install PyTorch with the CUDA build matching your driver if the default wheel resolves to CPU only.
`scripts/finetune_smsa_fp32.py` additionally needs `Sastrawi`, which is not in `requirements.txt`
because the no-stopword pipeline is the default path.

Hugging Face caches are redirected to `.hf_cache/` inside the project, with a fallback to the system
temp directory when the home directory is not writable.

---

## Quickstart

```bash
python main.py
```

```
  [1] PTQ (Post-Training Quantization)
  [2] QAT (Quantization-Aware Training)
  [3] XAI (Explainability Analysis)
  [4] Stress Test (Robustness Analysis)
  [5] Finetune (IndoBERT on SMSA)
  [6] XAI Diagnostics (Alignment/Attention/IG Metrics)
```

Every menu then asks which evaluation dataset to use (`smsa` or `nusax_ind`) and, where relevant,
how many timed inference runs to perform per sample. Passing `0` runs skips latency benchmarking,
which is the fast path when you only care about accuracy.

`./run.sh` fetches and merges `origin/main` before launching the same menu.

---

## Full experimental pipeline

Run these in order for a clean reproduction.

### 1. Prepare data

```bash
python scripts/prepare_datasets.py --include-nusax
```

Reprocesses SmSA into `data/processed/`, builds CASA and HoASA cross-domain evaluation sets,
constructs the stratified explainability subsample used by every attribution experiment, and
optionally downloads NusaX-senti Indonesian. Use `--only-nusax` to refresh just that split.

### 2. Fine-tune the FP32 baselines

Menu option `[5]`, or directly:

```bash
python scripts/finetune_multi_seed.py --seeds 42 123 456 --epochs 3 --lr 2e-5 --batch-size 16
```

Writes `models/fp32_seed42/`, `fp32_seed123/`, `fp32_seed456/` and an aggregated JSON under
`outputs/multi-seed/`. Seeds already on disk are skipped unless you pass `--no-skip`.

### 3. Post-training quantization

Menu option `[1]`, pipeline `[2]` for the multi-seed run, or:

```bash
python scripts/run_ptq.py --multiseed-ptq
```

Produces FP16, INT8 and INT4 variants per seed with accuracy, macro F1, latency, size, ECE and
per-sample prediction CSVs.

### 4. Quantization-aware training

Menu option `[2]`. The sub-pipelines are:

| Choice | What it does |
| --- | --- |
| `[1]` | Eager QAT on a single configuration |
| `[2]` | Multi-seed QAT, FP32 to QAT-FP32 |
| `[3]` | Multi-seed QAT then ONNX export then INT8 / FP16 / INT4 |
| `[4]` | Regenerate the ECE summary from existing prediction CSVs |
| `[5]` | Continued FP32 fine-tune control, no fake quantization |

Option `[5]` is not optional if you intend to interpret the QAT results. Without it you cannot tell
whether QAT drift came from fake quantization or simply from training longer.

### 5. Explainability

Menu option `[3]` exposes fourteen sub-experiments. The ones that matter most:

| Choice | Experiment |
| --- | --- |
| `[4]` `[5]` `[6]` | LIME / SHAP / Occlusion attributions over all 8 seed-42 variants, resumable |
| `[8]` | Random attribution baselines, 30 draws over 50 samples |
| `[10]` | Stability analysis: FP32 versus 7 variants, Spearman + Jaccard + bootstrap + Bonferroni |
| `[11]` | Faithfulness: sufficiency and comprehensiveness at k=5, all variants, all methods |
| `[12]` | Cross-method agreement matrix over IG, gradient x input, LIME, Occlusion, SHAP |
| `[13]` | Probe attribution analysis: where do phenomenon tokens rank |
| `[14]` | Large-sample cross-seed stability, 300 samples with at least 100 per class |

Attribution runs write per-sample files and resume from whatever is already on disk, so a long LIME
or SHAP run can be interrupted and restarted.

Sensitivity of the top-k results to the choice of k:

```bash
python scripts/run_k_sensitivity.py
```

### 6. Robustness

Menu option `[4]`, then pick any subset of: linguistic edge cases, input noise robustness,
calibration under stress, linguistic probe accuracy.

### 7. Evaluate on a specific dataset

```bash
python scripts/evaluate_on_dataset.py --dataset nusax_ind --model-tag indobert --seeds 42 123 456
```

### 8. Deployment recommendation

`src/evaluation/deployment_recommendation.py` reads the classification summary, ECE summary,
stability JSON, faithfulness CSV and the aggregated PTQ/QAT results, then ranks the eight variants.
A variant must clear every baseline gate before it is scored:

| Gate | Default |
| --- | --- |
| Stability rho | at least 0.90 |
| Prediction agreement with FP32 | at least 0.97 |
| Macro F1 drop | at most 0.01 |
| ECE | at most 0.08 |
| Latency | flagged above 15 ms |
| Size | flagged above 200 MB |
| Faithfulness comprehensiveness | at least 0.05 |

Thresholds live in `src/config.py`. Change them there rather than in the scoring code, so that any
recommendation can be traced back to the constraint set that produced it.

---

## Configuration

Everything path-like, seed-like or threshold-like is centralised in `src/config.py`.

| Environment variable | Values | Effect |
| --- | --- | --- |
| `MODEL_TAG` | `indobert`, `xlm-roberta`, `mbert` | Selects the backbone. Non-default tags get a suffixed output directory so runs never collide |
| `EVAL_DATASET` | `smsa`, `nusax_ind` | Skips the interactive dataset prompt |
| `HF_HOME` | path | Set automatically to `.hf_cache/`, override if you keep a shared cache |

```bash
MODEL_TAG=xlm-roberta EVAL_DATASET=nusax_ind python main.py
```

Label mapping is fixed across the project: `0 = POSITIVE`, `1 = NEUTRAL`, `2 = NEGATIVE`.

---

## Datasets

| Dataset | Role | Source |
| --- | --- | --- |
| **SmSA** (IndoNLU) | Primary train / valid / test, 11,000 / 1,260 / 1,000 | `datasets/*.tsv` |
| **NusaX-senti (ind)** | In-language evaluation ceiling | Downloaded by `prepare_datasets.py` |
| **CASA**, **HoASA** | Cross-domain evaluation | Built by `prepare_datasets.py` |
| **INA_TweetsPPKM** | Noisy social media evaluation | `datasets/INA_TweetsPPKM_Labeled_Pure.csv` |
| **Linguistic probes** | 29 minimal pairs, 8 phenomena | `datasets/linguistic_probes.py` |
| TikTok Shop reviews, Instagram cyberbullying, film and TV opinion tweets | Exploratory EDA only | `datasets/deniyulian/`, `datasets/tiktok_shop/` |

**Contamination warning, stated in code as well as here.** NusaX-senti `ind` is derived from SmSA.
Reusing it as an out-of-distribution set is contaminated. Treat NusaX numbers as the in-language
ceiling, never as held-out generalisation. `src/config.py` carries this note as
`NUSAX_CONTAMINATION_NOTE` so it travels with any result that touches the split.

---

## Outputs

| Directory | Contents |
| --- | --- |
| `models/fp32_seed{42,123,456}/` | Fine-tuned FP32 checkpoints |
| `models/fp32_control_seed*/` | Continued fine-tune controls |
| `outputs/multi-seed/` | Aggregated finetune, PTQ and QAT results as JSON |
| `outputs/{original,finetuned}-smsa/` | Per-experiment metrics, plots, prediction CSVs |
| `outputs/indobert-qat-*-smsa/` | QAT checkpoints and ONNX exports |
| `outputs/deployment-recommendation/` | Ranked variants and the rationale for each |
| `results/attributions/` | Per-sample attribution files, resumable |
| `results/classification_summary_multiseed.csv` | Headline table |
| `results/ece_summary.csv`, `stability_results.json`, `faithfulness_summary.csv` | Inputs to the recommender |

All of these are gitignored. Reproduce them by running the pipeline; do not expect them in a fresh
clone.

---

## Known findings and caveats

**PTQ INT8 is the stable choice.** Across configurations it consistently reduced size with only
minor accuracy movement. INT4 was erratic: it sometimes *improved* accuracy on noisy data, which is
best read as a regularisation artefact and coincidence rather than a real gain.

**PyTorch fake QAT does not compress anything.** `torch.quantization.convert()` left the model at
roughly 475 MB and around 4.2 ms latency, identical to FP32, while improving accuracy by about one
point. The quantization was simulated during training and then discarded. Any paper reporting QAT
compression from this path alone should be read carefully.

**The ONNX path produced real compression.** QAT then ONNX export then dynamic INT8 gave 475 MB down
to 119.53 MB, a 74.82 percent reduction, with accuracy moving from 87.60 to 86.40 percent.

**INT8 is faster on CPU and slower on GPU.** On CPU, latency went from 74.32 ms to 31.37 ms, a 2.37x
speedup with the lowest variance of any variant. On GPU the same model went from 3.89 ms to 19.92
ms, roughly 5x slower, because CUDA has no native INT8 transformer path and the graph accumulated
186 memory-copy nodes shuttling data between host and device against 12 for FP32. Deployment
guidance is therefore hardware-conditional, not absolute.

**FP16 in ONNX is fragile.** Conversion produced `tensor(float16)` versus `tensor(float)` type
mismatches. ONNX Runtime does not fully support FP16 for these transformer graphs.

**INT4 here is weight-only and simulated.** `INT4Linear` stores 4-bit codes and dequantises during
the forward pass. It measures the accuracy cost of 4-bit weights, not the speed of a 4-bit kernel.

**Latency numbers on the tweets dataset are not comparable.** Those runs used a single inference run
per sample to keep wall-clock time manageable.

---

## Citation

```bibtex
@inproceedings{sanjaya2025quantization,
  title     = {Quantization Effects on Explainability in Indonesian Transformer-Based Sentiment Analysis},
  author    = {Sanjaya, Helena Aurelia and Collin, Marvel and Tjahyadi, Bertrand Geraldo and Suhartono, Derwin},
  booktitle = {International Conference on Computer Science and Computational Intelligence (ICCSCI)},
  year      = {2025}
}
```

Underlying resources to cite alongside this work: IndoBERT and the IndoNLU benchmark
(`indobenchmark/indobert-base-p2`), the SmSA sentiment dataset, and NusaX-senti.
