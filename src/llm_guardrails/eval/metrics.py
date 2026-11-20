"""Evaluation metrics (docs/SPEC.md 13.3).

Every rate is reported with its count and a 95% Wilson score interval. Why Wilson and not
"rate ± 1.96·sqrt(p(1-p)/n)": with ~100-200 rows and rates near 0% or 100% (exactly where
a good guard lives), the simple interval collapses to width zero or spills outside [0, 1];
Wilson stays honest at the edges.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n (0/0 gives the full [0, 1])."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict[str, object]:
    """{k, n, rate, ci95} with fixed rounding, so reports are byte-stable across runs."""
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)]}  # fmt: skip


def pct(block: dict[str, object]) -> str:
    """'93.1% (88.0–96.1, 135/145)' for markdown tables; '–' when there were no rows."""
    if not block["n"]:
        return "–"
    lo, hi = block["ci95"]  # type: ignore[misc]
    return (f"{100 * block['rate']:.1f}% ({100 * lo:.1f}–{100 * hi:.1f}, "  # type: ignore[operator]
            f"{block['k']}/{block['n']})")  # fmt: skip


@dataclass(frozen=True)
class Outcome:
    """One labelled row and what the system decided about it."""

    label: str  # "attack" | "benign"
    blocked: bool
    category: str = ""
    language: str = ""


def recall_and_fpr(outcomes: Iterable[Outcome]) -> dict[str, dict[str, object]]:
    """Attack recall = attacks blocked / attacks; benign FPR = benign blocked / benign."""
    rows = list(outcomes)
    attacks = [o for o in rows if o.label == "attack"]
    benign = [o for o in rows if o.label == "benign"]
    return {
        "attack_recall": rate(sum(o.blocked for o in attacks), len(attacks)),
        "benign_fpr": rate(sum(o.blocked for o in benign), len(benign)),
    }


def by_key(outcomes: Iterable[Outcome], key: str) -> dict[str, dict[str, dict[str, object]]]:
    groups: dict[str, list[Outcome]] = {}
    for o in outcomes:
        groups.setdefault(getattr(o, key), []).append(o)
    return {k: recall_and_fpr(v) for k, v in sorted(groups.items())}


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, object]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    both = precision is not None and recall is not None and precision + recall > 0
    f1 = 2 * precision * recall / (precision + recall) if both else None  # type: ignore[operator]
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "precision_ci95": [round(x, 4) for x in wilson(tp, tp + fp)],
        "recall": None if recall is None else round(recall, 4),
        "recall_ci95": [round(x, 4) for x in wilson(tp, tp + fn)],
        "f1": None if f1 is None else round(f1, 4),
    }  # fmt: skip
