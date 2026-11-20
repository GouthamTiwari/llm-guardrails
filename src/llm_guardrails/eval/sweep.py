"""Dev-only threshold sweep for the injection guard (docs/SPEC.md 7.4, decision 4).

Objective, fixed before looking at the test split: maximise dev attack recall subject to
benign false-positive rate <= 2.0% AND hard-benign false-positive rate <= 10%; ties go to
fewer classifier calls, then lower benign FPR, then the more conservative thresholds.

Every dev row is scored once (rules, per-window embedding similarities, classifier), so
the ~69,000 combinations below cost nothing beyond those calls. The winner is written to
reports/operating_point.json and becomes InjectionConfig's defaults (a test keeps the two
equal). The test split is never loaded here: `guardrails sweep` refuses --split test.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from llm_guardrails.eval.datasets import DATA_DIR, FILES
from llm_guardrails.eval.metrics import rate
from llm_guardrails.eval.reports import dataset_sha256, write_json

MAX_BENIGN_FPR = 0.02
MAX_HARD_BENIGN_FPR = 0.10
GRID = {
    "t_block": np.round(np.arange(0.40, 0.9001, 0.01), 2),
    "t_allow": np.round(np.arange(0.20, 0.6001, 0.01), 2),
    "t_llm": np.round(np.arange(0.30, 0.8001, 0.05), 2),
    "benign_lambda": (0.0, 0.5, 1.0),
}


async def load_dev_scores(*, live: bool, settings: Any):
    from llm_guardrails.eval.run_injection import score_split

    scores, _ = await score_split("dev", live=live, settings=settings, op=None)
    return scores


def sweep(scores: list) -> dict[str, Any]:
    """Exhaustive grid search; returns the chosen point and how many points were feasible."""
    if any(s.llm is None for s in scores):
        raise ValueError("every dev row needs a classifier score before the sweep")
    attack = np.array([s.label == "attack" for s in scores])
    benign = ~attack
    hard = np.array([s.category == "benign_hard" for s in scores])
    rules = np.array([s.rules for s in scores])
    llm = np.array([1.0 if s.llm_error else s.llm for s in scores])
    n_attack, n_benign, n_hard = attack.sum(), benign.sum(), hard.sum()

    best_key, best, feasible = None, None, 0
    for lam in GRID["benign_lambda"]:
        embed = np.array([s.embed(lam) for s in scores])
        combined = np.maximum(rules, embed)
        for tb in GRID["t_block"]:
            sure = (rules >= 1.0) | (embed >= tb)
            for ta in GRID["t_allow"]:
                if ta > tb:
                    continue
                grey = ~sure & (combined >= ta)
                calls = int(grey.sum())
                for tl in GRID["t_llm"]:
                    blocked = sure | (grey & (llm >= tl))
                    fpr = blocked[benign].sum() / n_benign
                    hard_fpr = blocked[hard].sum() / n_hard if n_hard else 0.0
                    if fpr > MAX_BENIGN_FPR or hard_fpr > MAX_HARD_BENIGN_FPR:
                        continue
                    feasible += 1
                    recall = blocked[attack].sum() / n_attack
                    # Larger is better on every element (fixed tie-break order).
                    key = (round(recall, 6), -calls, -round(fpr, 6), tb, ta, tl, -lam)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = {"t_block": float(tb), "t_allow": float(ta), "t_llm": float(tl),
                                "benign_lambda": float(lam),
                                "dev": {"attack_recall": rate(int(blocked[attack].sum()),
                                                              int(n_attack)),
                                        "benign_fpr": rate(int(blocked[benign].sum()),
                                                           int(n_benign)),
                                        "hard_benign_fpr": rate(int(blocked[hard].sum()),
                                                                int(n_hard)),
                                        "classifier_calls": calls}}  # fmt: skip
    if best is None:
        raise RuntimeError("no threshold combination meets the false-positive constraints")
    return {**best, "feasible_points": feasible}


def write_operating_point(result: dict[str, Any], reports_dir: Path) -> dict[str, Any]:
    point = {
        "source": "dev sweep",
        "date": date.today().isoformat(),
        "t_block": result["t_block"],
        "t_allow": result["t_allow"],
        "t_llm": result["t_llm"],
        "benign_lambda": result["benign_lambda"],
        "objective": "max dev attack recall s.t. benign FPR <= 2.0% and hard-benign FPR <= "
        "10%; ties: fewer classifier calls, lower benign FPR, higher thresholds",
        "grid": {k: [float(v[0]), float(v[-1]), len(v)] for k, v in GRID.items()},
        "feasible_points": result["feasible_points"],
        "dev": result["dev"],
        "dataset_sha256": dataset_sha256(
            [
                DATA_DIR / FILES["injection"],
                DATA_DIR / "attack_corpus.jsonl",
                DATA_DIR / "benign_anchors.jsonl",
            ]
        ),  # fmt: skip
    }
    write_json("operating_point.json", point, reports_dir)
    return point


async def run(*, live: bool, settings: Any, reports_dir: Path) -> dict[str, Any]:
    scores = await load_dev_scores(live=live, settings=settings)
    return write_operating_point(sweep(scores), reports_dir)


def load(reports_dir: Path) -> dict[str, Any]:
    return json.loads((reports_dir / "operating_point.json").read_text(encoding="utf-8"))
