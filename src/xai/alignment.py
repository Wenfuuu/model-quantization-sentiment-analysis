from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoTokenizer

UNALIGNED = -1
_WHITESPACE_PATTERN = re.compile(r"\S+")

@dataclass
class WordSubwordAlignment:
    text:            str
    words:           List[str]
    subwords:        List[str]
    word_spans:      List[Tuple[int, int]]
    subword_spans:   List[Tuple[int, int]]
    word_to_subword: List[List[int]]
    subword_to_word: List[int]
    n_words:         int
    n_subwords:      int
    fragmentation_per_word: List[int] = field(default_factory=list)

    def average_fragmentation(self) -> float:
        if not self.fragmentation_per_word:
            return 0.0
        return float(np.mean(self.fragmentation_per_word))

    def words_with_multiple_subwords(self) -> List[Tuple[int, str, List[str]]]:
        result = []
        for j, (word, sw_indices) in enumerate(zip(self.words, self.word_to_subword)):
            if len(sw_indices) > 1:
                sw_strs = [self.subwords[i] for i in sw_indices]
                result.append((j, word, sw_strs))
        return result

    def fragmentation_summary(self) -> dict:
        n_multi = sum(1 for f in self.fragmentation_per_word if f > 1)
        return {
            "n_words":          self.n_words,
            "n_subwords":       self.n_subwords,
            "avg_fragments":    self.average_fragmentation(),
            "max_fragments":    max(self.fragmentation_per_word) if self.fragmentation_per_word else 0,
            "n_words_multi":    n_multi,
            "frac_words_multi": n_multi / self.n_words if self.n_words > 0 else 0.0,
        }

def build_alignment(
    text:      str,
    tokenizer: AutoTokenizer,
) -> WordSubwordAlignment:
    word_matches = list(_WHITESPACE_PATTERN.finditer(text))
    words      = [m.group() for m in word_matches]
    word_spans = [(m.start(), m.end()) for m in word_matches]
    W = len(words)

    if W == 0:
        return WordSubwordAlignment(
            text=text, words=[], subwords=[],
            word_spans=[], subword_spans=[],
            word_to_subword=[], subword_to_word=[],
            n_words=0, n_subwords=0, fragmentation_per_word=[],
        )

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )

    offset_mapping: Optional[List[Tuple[int, int]]] = encoding.get("offset_mapping")
    all_tokens: List[str] = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    if offset_mapping is None:
        warnings.warn(
            "Tokenizer does not support offset_mapping (slow tokenizer "
            "detected).  Falling back to ##-propagation alignment.  "
            "Results may be less accurate for punctuation-attached words.",
            UserWarning, stacklevel=2,
        )
        return _build_alignment_slow(text, tokenizer, words, word_spans)

    content_subwords: List[str]               = []
    content_spans:    List[Tuple[int, int]]   = []

    for tok, span in zip(all_tokens, offset_mapping):
        cs, ce = span
        if ce > cs:
            content_subwords.append(tok)
            content_spans.append((cs, ce))

    S = len(content_subwords)

    subword_to_word: List[int]       = [UNALIGNED] * S
    word_to_subword: List[List[int]] = [[] for _ in range(W)]

    for i, (s_start, s_end) in enumerate(content_spans):
        for j, (w_start, w_end) in enumerate(word_spans):
            if s_start >= w_start and s_end <= w_end:
                subword_to_word[i] = j
                word_to_subword[j].append(i)
                break
        if subword_to_word[i] == UNALIGNED:
            warnings.warn(
                f"Content subword '{content_subwords[i]}' at span "
                f"({s_start}, {s_end}) was not contained in any word span.  ",
                UserWarning, stacklevel=2,
            )

    fragmentation = [len(word_to_subword[j]) for j in range(W)]

    for j, f in enumerate(fragmentation):
        if f == 0:
            warnings.warn(
                f"Word '{words[j]}' at span {word_spans[j]} has no aligned ",
                UserWarning, stacklevel=2,
            )

    return WordSubwordAlignment(
        text=text,
        words=words,
        subwords=content_subwords,
        word_spans=word_spans,
        subword_spans=content_spans,
        word_to_subword=word_to_subword,
        subword_to_word=subword_to_word,
        n_words=W,
        n_subwords=S,
        fragmentation_per_word=fragmentation,
    )

def _build_alignment_slow(
    text:       str,
    tokenizer:  AutoTokenizer,
    words:      List[str],
    word_spans: List[Tuple[int, int]],
) -> WordSubwordAlignment:
    encoding    = tokenizer(text, add_special_tokens=True)
    all_tokens  = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    content     = [t for t in all_tokens
                   if t not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")]

    W = len(words)
    S = len(content)

    word_to_subword: List[List[int]] = [[] for _ in range(W)]
    subword_to_word: List[int]       = [UNALIGNED] * S

    current_word = 0
    for i, tok in enumerate(content):
        if tok.startswith("##"):
            subword_to_word[i] = current_word
            word_to_subword[current_word].append(i)
        else:
            if i > 0:
                current_word = min(current_word + 1, W - 1)
            subword_to_word[i] = current_word
            word_to_subword[current_word].append(i)

    fragmentation = [len(word_to_subword[j]) for j in range(W)]
    content_spans = [(0, 0)] * S

    return WordSubwordAlignment(
        text=text,
        words=words,
        subwords=content,
        word_spans=word_spans,
        subword_spans=content_spans,
        word_to_subword=word_to_subword,
        subword_to_word=subword_to_word,
        n_words=W,
        n_subwords=S,
        fragmentation_per_word=fragmentation,
    )

def project_word_to_subword(
    words:        Sequence[str],
    word_scores:  Sequence[float],
    alignment:    WordSubwordAlignment,
    strategy:     str = "equal",
) -> Tuple[List[str], np.ndarray]:
    if strategy not in ("equal", "duplicate"):
        raise ValueError(
            f"strategy must be 'equal' or 'duplicate', got {strategy!r}. "
            "'equal' is recommended for IG comparison."
        )

    if len(words) != len(word_scores):
        raise ValueError(
            f"words ({len(words)}) and word_scores ({len(word_scores)}) "
            "must have the same length."
        )

    subword_scores = np.zeros(alignment.n_subwords, dtype=np.float64)

    word_score_dict: Dict[str, float] = dict(zip(words, word_scores))

    for j, (word, sw_indices) in enumerate(zip(alignment.words, alignment.word_to_subword)):
        if word not in word_score_dict:
            continue
        s_j  = word_score_dict[word]
        N_j  = len(sw_indices)
        if N_j == 0:
            continue

        if strategy == "equal":
            unit = s_j / N_j
        else:
            unit = s_j

        for i in sw_indices:
            subword_scores[i] = unit

    return alignment.subwords, subword_scores.astype(np.float32)


def sparse_word_to_subword(
    word_score_dict: Dict[str, float],
    alignment:       WordSubwordAlignment,
    strategy:        str = "equal",
) -> Tuple[List[str], np.ndarray]:
    full_words  = alignment.words
    full_scores = [word_score_dict.get(w, 0.0) for w in full_words]
    return project_word_to_subword(full_words, full_scores, alignment, strategy)

def project_subword_to_word(
    subwords:       Sequence[str],
    subword_scores: Sequence[float],
    alignment:      WordSubwordAlignment,
    strategy:       str = "sum",
) -> Tuple[List[str], np.ndarray]:
    if strategy not in ("sum", "mean", "max"):
        raise ValueError(
            f"strategy must be 'sum', 'mean', or 'max', got {strategy!r}. "
            "'sum' is recommended for IG-to-word aggregation."
        )

    if len(subwords) != len(subword_scores):
        raise ValueError(
            f"subwords ({len(subwords)}) and subword_scores ({len(subword_scores)}) "
            "must have the same length."
        )

    if len(subword_scores) != alignment.n_subwords:
        warnings.warn(
            f"subword_scores length ({len(subword_scores)}) does not match "
            f"alignment.n_subwords ({alignment.n_subwords}).  This may happen "
            "when truncation occurs.",
            UserWarning, stacklevel=2,
        )

    scores_arr  = np.array(subword_scores, dtype=np.float64)
    word_scores = np.zeros(alignment.n_words, dtype=np.float64)

    for j, sw_indices in enumerate(alignment.word_to_subword):
        if not sw_indices:
            continue

        valid = [i for i in sw_indices if i < len(scores_arr)]
        if not valid:
            continue

        chunk = scores_arr[valid]

        if strategy == "sum":
            word_scores[j] = float(chunk.sum())

        elif strategy == "mean":
            word_scores[j] = float(chunk.mean())

        elif strategy == "max":
            # absolute attribution represents the word.
            idx_max = int(np.argmax(np.abs(chunk)))
            word_scores[j] = float(chunk[idx_max])

    return alignment.words, word_scores.astype(np.float32)

def align_for_comparison(
    word_attribution:    dict,
    subword_attribution: dict,
    alignment:           WordSubwordAlignment,
    direction:           str = "subword_to_word",
    word_strategy:       str = "equal",
    subword_strategy:    str = "sum",
) -> Tuple[dict, dict]:
    if direction not in ("subword_to_word", "word_to_subword"):
        raise ValueError(
            f"direction must be 'subword_to_word' or 'word_to_subword', "
            f"got {direction!r}."
        )

    w_tokens  = word_attribution["tokens"]
    w_scores  = word_attribution["scores"]
    sw_tokens = subword_attribution["tokens"]
    sw_scores = subword_attribution["scores"]

    if direction == "subword_to_word":
        projected_words, projected_scores = project_subword_to_word(
            sw_tokens, sw_scores, alignment, strategy=subword_strategy
        )
        aligned_subword = {
            "tokens": list(projected_words),
            "scores": projected_scores.tolist(),
        }
        aligned_word = word_attribution
        return aligned_word, aligned_subword

    else:
        word_score_dict = dict(zip(w_tokens, w_scores))
        projected_sws, projected_scores = sparse_word_to_subword(
            word_score_dict, alignment, strategy=word_strategy
        )
        aligned_word = {
            "tokens": list(projected_sws),
            "scores": projected_scores.tolist(),
        }
        aligned_subword = subword_attribution
        return aligned_word, aligned_subword

def build_alignment_batch(
    texts:         List[str],
    tokenizer:     AutoTokenizer,
    verbose:       bool = False,
) -> List[WordSubwordAlignment]:
    alignments = []
    for i, text in enumerate(texts):
        alignments.append(build_alignment(text, tokenizer))
        if verbose and ((i + 1) % 50 == 0 or (i + 1) == len(texts)):
            print(f"  [alignment] {i+1}/{len(texts)} texts aligned")
    return alignments


def align_attribution_batch(
    word_attributions:    List[dict],
    subword_attributions: List[dict],
    alignments:           List[WordSubwordAlignment],
    direction:            str = "subword_to_word",
    word_strategy:        str = "equal",
    subword_strategy:     str = "sum",
) -> Tuple[List[dict], List[dict]]:
    if not (len(word_attributions) == len(subword_attributions) == len(alignments)):
        raise ValueError(
            f"All three lists must have the same length.  Got "
            f"word={len(word_attributions)}, subword={len(subword_attributions)}, "
            f"alignments={len(alignments)}."
        )

    aligned_words    = []
    aligned_subwords = []

    for wa, sa, al in zip(word_attributions, subword_attributions, alignments):
        aw, asw = align_for_comparison(
            wa, sa, al,
            direction=direction,
            word_strategy=word_strategy,
            subword_strategy=subword_strategy,
        )
        aligned_words.append(aw)
        aligned_subwords.append(asw)

    return aligned_words, aligned_subwords

def fragmentation_report(alignments: List[WordSubwordAlignment]) -> dict:
    all_frags = []
    per_sample_frac = []

    for al in alignments:
        all_frags.extend(al.fragmentation_per_word)
        summary = al.fragmentation_summary()
        per_sample_frac.append(summary["frac_words_multi"])

    if not all_frags:
        return {}

    arr = np.array(all_frags, dtype=float)
    return {
        "n_samples":               len(alignments),
        "n_total_words":           int(len(arr)),
        "avg_fragments":           float(np.mean(arr)),
        "std_fragments":           float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "max_fragments":           int(np.max(arr)),
        "frac_words_multi":        float(np.mean(arr > 1)),
        "avg_frac_multi_per_sample": float(np.mean(per_sample_frac)),
    }
