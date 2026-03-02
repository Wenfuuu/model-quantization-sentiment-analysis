"""
perturbation.py
----------------
Text perturbation generators for the quantization stress test.

Perturbations simulate real-world noise that Indonesian social-media text
frequently contains: typos, OCR artifacts, character swaps, and emphatic
word repetition.
"""

import random
import re


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePerturbation:
    def apply(self, text: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Character-level noise
# ---------------------------------------------------------------------------

class CharacterNoisePerturbation(BasePerturbation):
    """
    Simulates character-level noise found in social-media text:
      - swap two adjacent characters
      - delete a random character
      - insert a random duplicate character

    Parameters
    ----------
    severity : float
        Fraction of *words* that will be perturbed (0 = no noise, 1 = all words).
    seed : int | None
        Random seed for reproducibility.
    """

    INDONESIAN_COMMON_CHARS = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, severity: float = 0.2, seed: int | None = 42):
        self.severity = severity
        self.rng = random.Random(seed)

    def _perturb_word(self, word: str) -> str:
        if len(word) < 2:
            return word
        ops = ["swap", "delete", "insert"]
        op = self.rng.choice(ops)

        chars = list(word)
        idx = self.rng.randint(0, len(chars) - 2)

        if op == "swap":
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "delete":
            del chars[idx]
        elif op == "insert":
            chars.insert(idx, self.rng.choice(self.INDONESIAN_COMMON_CHARS))

        return "".join(chars)

    def apply(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if self.rng.random() < self.severity and re.match(r"[a-zA-Z]{2,}", word):
                result.append(self._perturb_word(word))
            else:
                result.append(word)
        return " ".join(result)


# ---------------------------------------------------------------------------
# Word repetition
# ---------------------------------------------------------------------------

class WordRepetitionPerturbation(BasePerturbation):
    """
    Repeats randomly selected words to simulate emphatic / spammy text.

    Parameters
    ----------
    repeat_range : tuple[int, int]
        Min and max number of repetitions for a selected word.
    target_words : list[str] | None
        If provided, only these words will be repeated (useful for sentiment
        intensifiers like "sangat", "banget", "sekali").
    repeat_fraction : float
        Fraction of eligible words to repeat.
    seed : int | None
    """

    INDONESIAN_INTENSIFIERS = [
        "sangat", "banget", "sekali", "benar-benar", "sungguh",
        "amat", "terlalu", "cukup", "lumayan", "agak",
    ]

    def __init__(
        self,
        repeat_range: tuple = (2, 4),
        target_words: list | None = None,
        repeat_fraction: float = 0.3,
        seed: int | None = 42,
    ):
        self.repeat_range = repeat_range
        self.target_words = target_words or self.INDONESIAN_INTENSIFIERS
        self.repeat_fraction = repeat_fraction
        self.rng = random.Random(seed)

    def apply(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            if word.lower() in self.target_words and self.rng.random() < self.repeat_fraction:
                n = self.rng.randint(*self.repeat_range)
                result.extend([word] * n)
            else:
                result.append(word)
        return " ".join(result)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PerturbationPipeline(BasePerturbation):
    """
    Chains multiple perturbations sequentially.

    Example
    -------
    >>> pipeline = PerturbationPipeline([
    ...     CharacterNoisePerturbation(severity=0.15),
    ...     WordRepetitionPerturbation(repeat_range=(2, 3)),
    ... ])
    >>> noisy_text = pipeline.apply("produknya bagus sekali")
    """

    def __init__(self, perturbations: list[BasePerturbation]):
        self.perturbations = perturbations

    def apply(self, text: str) -> str:
        for p in self.perturbations:
            text = p.apply(text)
        return text
