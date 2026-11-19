"""Near-duplicate detection with character 5-gram Jaccard similarity (docs/SPEC.md 13.2).

Why characters, not words: it catches copies that differ by punctuation, spacing or a typo,
works the same for Hindi and English, and needs no model or network.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

NEAR_DUPLICATE = 0.6


def char_ngrams(text: str, n: int = 5) -> frozenset[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) <= n:
        return frozenset({t}) if t else frozenset()
    return frozenset(t[i : i + n] for i in range(len(t) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def similarity(x: str, y: str) -> float:
    return jaccard(char_ngrams(x), char_ngrams(y))


def near_duplicate_pairs(
    texts: Sequence[str], others: Sequence[str] | None = None, threshold: float = NEAR_DUPLICATE
) -> list[tuple[int, int, float]]:
    """(i, j, score) for pairs at or above `threshold`.

    With `others`, compares every text against every other-text (cross-set leakage).
    Without it, compares texts within the one list (i < j).
    """
    grams = [char_ngrams(t) for t in texts]
    if others is None:
        return [
            (i, j, s)
            for i in range(len(grams))
            for j in range(i + 1, len(grams))
            if (s := jaccard(grams[i], grams[j])) >= threshold
        ]
    other_grams = [char_ngrams(t) for t in others]
    return [
        (i, j, s)
        for i, g in enumerate(grams)
        for j, h in enumerate(other_grams)
        if (s := jaccard(g, h)) >= threshold
    ]
