from __future__ import annotations

import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data" / "processed"
_SMSA_DIR     = _PROJECT_ROOT / "datasets"

_DATA_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

PHENOMENON_TOKENS = {
    "negation":    ["tidak", "bukan", "tak", "belum", "jangan"],
    "intensifier": ["sangat", "sekali", "banget", "amat"],
    "contrastive": ["tapi", "namun", "walaupun", "meskipun"],
    "hedging":     ["mungkin", "sepertinya", "kayaknya"],
}

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def print_divider(title: str = "") -> None:
    line = "=" * 70
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def label_distribution(series: pd.Series) -> dict:
    counts = series.value_counts().to_dict()
    total  = len(series)
    return {k: {"count": v, "pct": f"{100*v/total:.1f}%"} for k, v in sorted(counts.items())}

def task_a() -> dict[str, pd.DataFrame]:
    print_divider("TASK A — Reprocess SmSA")

    splits = {
        "train": _SMSA_DIR / "train.tsv",
        "val":   _SMSA_DIR / "valid.tsv",
        "test":  _SMSA_DIR / "test.tsv",
    }

    dfs: dict[str, pd.DataFrame] = {}

    for split, path in splits.items():
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
        df = df.dropna(subset=["text", "label"])
        df["text"]  = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["text"] != ""]
        df = df[df["label"].isin(LABEL2ID)]

        df["text"]  = df["text"].apply(preprocess_text)
        df["label"] = df["label"].map(LABEL2ID)

        out_path = _DATA_DIR / f"smsa_{split}_v2.csv"
        df[["text", "label"]].to_csv(out_path, index=False, encoding="utf-8")
        dfs[split] = df

        dist = label_distribution(df["label"].map(ID2LABEL))
        print(f"\n  [{split.upper()}]  rows={len(df):,}  →  {out_path.name}")
        for cls, info in dist.items():
            print(f"    {cls:10s}: {info['count']:5d}  ({info['pct']})")

    train_texts = " ".join(dfs["train"]["text"].tolist())
    tokens_in_train = train_texts.split()
    token_counter = Counter(tokens_in_train)

    print("\n  Phenomenon token counts in SmSA TRAINING data:")
    for phenom, tokens in PHENOMENON_TOKENS.items():
        counts = {t: token_counter.get(t, 0) for t in tokens}
        total  = sum(counts.values())
        print(f"    {phenom:12s}: total={total:5d}  →  " +
              "  ".join(f"{t}={c}" for t, c in counts.items()))

    reduplications = [t for t in tokens_in_train if re.match(r"^\w+-\w+$", t)]
    print(f"    {'reduplication':12s}: {len(reduplications):5d} hyphenated tokens  "
          f"(top 5: {', '.join(t for t, _ in Counter(reduplications).most_common(5))})")

    return dfs

def majority_vote(
    label_list: list[str],
    valid_labels: set[str],
    exclude_labels: set[str],
    min_valid: int = 1,
) -> tuple[str | None, dict]:
    counts: dict[str, int] = {lbl: 0 for lbl in valid_labels}
    for lbl in label_list:
        if lbl in exclude_labels:
            continue
        if lbl in valid_labels:
            counts[lbl] += 1

    n_valid = sum(counts.values())
    if n_valid < min_valid:
        return None, counts

    winner = max(counts, key=lambda k: counts[k])
    if counts[winner] / n_valid > 0.5:
        return winner, counts
    return None, counts 

def _normalize_label(val, int_to_str: dict | None = None) -> str:
    if isinstance(val, (int, float)):
        val = int(val)
        if int_to_str and val in int_to_str:
            return int_to_str[val]
        return {0: "negative", 1: "neutral", 2: "positive"}.get(val, "unknown")
    return str(val).lower().strip()

def task_b() -> pd.DataFrame | None:
    print_divider("TASK B — CASA Cross-Domain Evaluation Set")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] `datasets` library not installed. Run: pip install datasets")
        return None

    print("  Loading indonlp/indonlu (config='casa') ...")
    try:
        ds = load_dataset("indonlp/indonlu", "casa", trust_remote_code=True)
    except Exception as exc:
        print(f"  [ERROR] Could not load CASA dataset: {exc}")
        return None

    split_name = "test" if "test" in ds else list(ds.keys())[0]
    raw = ds[split_name].to_pandas()
    print(f"  Split used : '{split_name}'  |  raw rows : {len(raw):,}")
    print(f"  Columns    : {list(raw.columns)}")

    text_col = next((c for c in ["sentence", "text", "review"] if c in raw.columns), None)
    if text_col is None:
        print(f"  [ERROR] No text column found. Available: {list(raw.columns)}")
        return None

    aspect_cols = [c for c in raw.columns if c != text_col]
    print(f"  Text column: '{text_col}'  |  Aspect columns ({len(aspect_cols)}): {aspect_cols}")

    sample_vals = set()
    for c in aspect_cols:
        sample_vals.update(raw[c].dropna().unique().tolist())
    print(f"  Unique label values: {sorted(sample_vals, key=str)}")

    int_to_str: dict | None = None
    if all(isinstance(v, (int, float)) for v in sample_vals if not pd.isna(v)):
        try:
            feature = ds[split_name].features[aspect_cols[0]]
            if hasattr(feature, "names"):
                int_to_str = {i: n.lower() for i, n in enumerate(feature.names)}
                print(f"  ClassLabel mapping: {int_to_str}")
        except Exception:
            pass

    valid_labels  = {"positive", "negative", "neutral"}
    exclude_labels: set[str] = set()

    rows = []
    n_excluded = 0

    for _, row in raw.iterrows():
        text = preprocess_text(str(row[text_col]))
        if not text:
            n_excluded += 1
            continue

        labels = [_normalize_label(row[c], int_to_str) for c in aspect_cols]
        winner, counts = majority_vote(
            labels,
            valid_labels=valid_labels,
            exclude_labels=exclude_labels,
            min_valid=1,
        )
        if winner is None:
            n_excluded += 1
            continue

        rows.append({
            "text":                  text,
            "label":                 LABEL2ID[winner],
            "num_aspects_positive":  counts.get("positive", 0),
            "num_aspects_negative":  counts.get("negative", 0),
            "num_aspects_neutral":   counts.get("neutral",  0),
            "derivation_method":     "majority_vote_6_aspects",
        })

    df = pd.DataFrame(rows)
    out_path = _DATA_DIR / "casa_test_v2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n  Original rows  : {len(raw):,}")
    print(f"  Excluded       : {n_excluded:,}  (no strict majority or empty text)")
    print(f"  Retained       : {len(df):,}  →  {out_path.name}")

    dist = label_distribution(df["label"].map(ID2LABEL))
    print("\n  Derived label distribution:")
    for cls, info in dist.items():
        print(f"    {cls:10s}: {info['count']:5d}  ({info['pct']})")

    print("\n  5 example rows:")
    for _, r in df.head(5).iterrows():
        print(f"    [{ID2LABEL[r['label']].upper():8s}] "
              f"pos={r['num_aspects_positive']} neg={r['num_aspects_negative']} "
              f"neu={r['num_aspects_neutral']}  |  "
              f"{textwrap.shorten(r['text'], width=70)}")

    return df

def task_c() -> pd.DataFrame | None:
    print_divider("TASK C — HoASA Cross-Domain Evaluation Set")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] `datasets` library not installed.")
        return None

    print("  Loading indonlp/indonlu (config='hoasa') ...")
    try:
        ds = load_dataset("indonlp/indonlu", "hoasa", trust_remote_code=True)
    except Exception as exc:
        print(f"  [ERROR] Could not load HoASA via HuggingFace: {exc}")
        print("  Attempting local fallback: datasets/hoasa_absa-airy/ ...")
        local_path = _PROJECT_ROOT / "datasets" / "hoasa_absa-airy"
        if not local_path.exists():
            print(f"  [ERROR] Local path not found: {local_path}")
            return None
        try:
            ds = load_dataset(str(local_path), trust_remote_code=True)
        except Exception as exc2:
            print(f"  [ERROR] Local load also failed: {exc2}")
            return None

    split_name = "test" if "test" in ds else list(ds.keys())[0]
    raw = ds[split_name].to_pandas()
    print(f"  Split used : '{split_name}'  |  raw rows : {len(raw):,}")
    print(f"  Columns    : {list(raw.columns)}")

    text_col = next((c for c in ["sentence", "text", "review"] if c in raw.columns), None)
    if text_col is None:
        print(f"  [ERROR] No text column found. Available: {list(raw.columns)}")
        return None

    aspect_cols = [c for c in raw.columns if c != text_col]
    print(f"  Text column: '{text_col}'  |  Aspect columns ({len(aspect_cols)}): {aspect_cols}")

    sample_vals = set()
    for c in aspect_cols:
        sample_vals.update(raw[c].dropna().unique().tolist())
    print(f"  Unique label values: {sorted(sample_vals, key=str)}")

    int_to_str: dict | None = None
    if all(isinstance(v, (int, float)) for v in sample_vals if not pd.isna(v)):
        try:
            feature = ds[split_name].features[aspect_cols[0]]
            if hasattr(feature, "names"):
                int_to_str = {i: n.lower() for i, n in enumerate(feature.names)}
                print(f"  ClassLabel mapping: {int_to_str}")
        except Exception:
            pass

    valid_labels   = {"positive", "negative", "neutral"}
    exclude_labels = {"positive-negative", "pos-neg"}

    rows = []
    n_excluded_no_majority  = 0
    n_excluded_min_valid    = 0
    n_excluded_empty        = 0

    for _, row in raw.iterrows():
        text = preprocess_text(str(row[text_col]))
        if not text:
            n_excluded_empty += 1
            continue

        raw_labels = [_normalize_label(row[c], int_to_str) for c in aspect_cols]
        effective_exclude = exclude_labels | {
            lbl for lbl in raw_labels if lbl not in valid_labels and lbl != "unknown"
        }

        winner, counts = majority_vote(
            raw_labels,
            valid_labels=valid_labels,
            exclude_labels=effective_exclude,
            min_valid=3,
        )

        n_valid = sum(counts.values())
        if n_valid < 3:
            n_excluded_min_valid += 1
            continue
        if winner is None:
            n_excluded_no_majority += 1
            continue

        rows.append({
            "text":                  text,
            "label":                 LABEL2ID[winner],
            "num_aspects_positive":  counts.get("positive", 0),
            "num_aspects_negative":  counts.get("negative", 0),
            "num_aspects_neutral":   counts.get("neutral",  0),
            "derivation_method":     "majority_vote_hoasa_excl_posneg",
        })

    df = pd.DataFrame(rows)
    out_path = _DATA_DIR / "hoasa_test_v2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n  Original rows         : {len(raw):,}")
    print(f"  Excluded (empty text) : {n_excluded_empty:,}")
    print(f"  Excluded (<3 valid aspects after excl.) : {n_excluded_min_valid:,}")
    print(f"  Excluded (no strict majority)           : {n_excluded_no_majority:,}")
    print(f"  Retained              : {len(df):,}  →  {out_path.name}")

    dist = label_distribution(df["label"].map(ID2LABEL))
    print("\n  Derived label distribution:")
    for cls, info in dist.items():
        print(f"    {cls:10s}: {info['count']:5d}  ({info['pct']})")

    print("\n  5 example rows:")
    for _, r in df.head(5).iterrows():
        print(f"    [{ID2LABEL[r['label']].upper():8s}] "
              f"pos={r['num_aspects_positive']} neg={r['num_aspects_negative']} "
              f"neu={r['num_aspects_neutral']}  |  "
              f"{textwrap.shorten(r['text'], width=70)}")

    return df

def task_d(smsa_test: pd.DataFrame, casa: pd.DataFrame | None, hoasa: pd.DataFrame | None) -> None:
    print_divider("TASK D — Cross-Dataset Validation")

    datasets: dict[str, pd.DataFrame] = {"SmSA-test": smsa_test}
    if casa  is not None and len(casa)  > 0: datasets["CASA"]  = casa
    if hoasa is not None and len(hoasa) > 0: datasets["HoASA"] = hoasa

    print("\n  [1] Sentence length (word count) distributions:")
    header = f"  {'Metric':12s}" + "".join(f"  {name:>12s}" for name in datasets)
    print(header)
    print("  " + "-" * (len(header) - 2))

    lengths: dict[str, pd.Series] = {}
    for name, df in datasets.items():
        lengths[name] = df["text"].apply(lambda t: len(str(t).split()))

    for metric, fn in [("min",    lambda s: s.min()),
                        ("p25",    lambda s: s.quantile(0.25)),
                        ("median", lambda s: s.median()),
                        ("mean",   lambda s: s.mean()),
                        ("p75",    lambda s: s.quantile(0.75)),
                        ("max",    lambda s: s.max())]:
        row = f"  {metric:12s}"
        for name, s in lengths.items():
            row += f"  {fn(s):>12.1f}"
        print(row)

    print("\n  [2] Vocabulary overlap with SmSA training data:")

    smsa_train_path = _DATA_DIR / "smsa_train_v2.csv"
    if not smsa_train_path.exists():
        print("  [SKIP] smsa_train_v2.csv not found — run Task A first.")
    else:
        train_df    = pd.read_csv(smsa_train_path)
        train_vocab = set(" ".join(train_df["text"].tolist()).split())
        print(f"  SmSA training vocabulary size: {len(train_vocab):,} unique tokens")

        for name, df in datasets.items():
            if name == "SmSA-test":
                continue
            eval_vocab = set(" ".join(df["text"].tolist()).split())
            overlap    = eval_vocab & train_vocab
            pct        = 100 * len(overlap) / len(eval_vocab) if eval_vocab else 0
            print(f"  {name:12s}: {len(eval_vocab):6,} unique tokens  →  "
                  f"{len(overlap):,} ({pct:.1f}%) appear in SmSA train")

    print("\n  [3] Class balance flags (< 10% threshold):")
    flagged_any = False
    for name, df in datasets.items():
        if name == "SmSA-test":
            continue
        dist = df["label"].value_counts(normalize=True)
        for cls_id in range(3):
            pct = dist.get(cls_id, 0.0) * 100
            if pct < 10.0:
                print(f"  ⚠  {name}: class '{ID2LABEL[cls_id]}' = {pct:.1f}%  (< 10%)")
                flagged_any = True
    if not flagged_any:
        print("  All cross-domain classes are ≥ 10%. No flags.")

def main() -> None:
    print_divider("Dataset Preparation Pipeline")
    print(f"  Output directory: {_DATA_DIR}")

    smsa_dfs = task_a()

    casa  = task_b()
    hoasa = task_c()

    task_d(
        smsa_test=smsa_dfs["test"],
        casa=casa,
        hoasa=hoasa,
    )

    print_divider("DONE")
    print(f"  Outputs written to: {_DATA_DIR}")


if __name__ == "__main__":
    main()
