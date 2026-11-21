"""Human spot-check of the datasets (docs/SPEC.md 13.1).

1. `guardrails dataset review-sample --suite all --frac 0.10 --seed 7` writes
   `data/review/review_sample.csv`: id, suite, text, label, and an empty `verdict` column.
2. A reviewer fills `verdict` with `ok`, `edit:<new label>` or `drop`.
3. `guardrails dataset apply-review` applies it to the datasets and records the counts in
   `docs/DATASET_REVIEW.md`.

What `edit:` changes per suite: injection -> the category (and so the label); grounding ->
the label; pii -> only `edit:none` (the row has no PII; other entity fixes are edited by
hand in the JSONL); schema -> the input kind. A dropped row stays in the file, marked
`human-dropped`, and every loader skips it, so the decision stays visible in git.
"""

from __future__ import annotations

import csv
import io
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from llm_guardrails.eval.datasets import (
    ATTACK_CATEGORIES,
    BENIGN_CATEGORIES,
    DATA_DIR,
    FILES,
    GROUNDING_LABELS,
    REPO_ROOT,
    SCHEMA_KINDS,
    SUITES,
    load_suite,
    row_text,
    write_jsonl,
)

SAMPLE_PATH = DATA_DIR / "review" / "review_sample.csv"
REVIEW_DOC = REPO_ROOT / "docs" / "DATASET_REVIEW.md"
COLUMNS = ["id", "suite", "split", "text", "label", "verdict"]


def label_of(suite: str, row: dict) -> str:
    if suite == "injection":
        return row["category"]
    if suite == "pii":
        return "; ".join(f"{e['type']}={e['value']}" for e in row["entities"]) or "none"
    if suite == "grounding":
        return row["label"]
    return row["notes"]


def sample_rows(suites: list[str], frac: float, seed: int, data_dir: Path = DATA_DIR):
    rng = random.Random(seed)
    out = []
    for suite in suites:
        rows = load_suite(suite, data_dir)
        k = max(1, round(len(rows) * frac)) if rows else 0
        for row in sorted(rng.sample(rows, k), key=lambda r: r["id"]):
            text = row_text(suite, row)
            if suite == "grounding":
                sources = " | ".join(f"[{s['id']}] {s['text']}" for s in row["sources"])
                text = f"Q: {row['question']}\nA: {row['answer']}\nSOURCES: {sources}"
            out.append({"id": row["id"], "suite": suite, "split": row.get("split") or "",
                        "text": text, "label": label_of(suite, row), "verdict": ""})  # fmt: skip
    return out


def write_sample(rows: list[dict], path: Path = SAMPLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    # utf-8-sig: Excel opens Hindi text correctly only when the file starts with a BOM.
    path.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")


@dataclass
class ReviewOutcome:
    counts: Counter = field(default_factory=Counter)  # ok / edited / dropped / blank
    errors: list[str] = field(default_factory=list)


def _apply_edit(suite: str, row: dict, new: str) -> str | None:
    """Apply `edit:<new>`; returns an error message, or None when applied."""
    if suite == "injection":
        if new not in ATTACK_CATEGORIES + BENIGN_CATEGORIES:
            return f"unknown injection category {new!r}"
        row["category"] = new
        row["label"] = "benign" if new in BENIGN_CATEGORIES else "attack"
    elif suite == "grounding":
        if new not in GROUNDING_LABELS:
            return f"unknown grounding label {new!r}"
        row["label"] = new
    elif suite == "pii":
        if new != "none":
            return "pii rows accept only edit:none (fix other entity changes in the JSONL)"
        row["entities"] = []
    else:
        if new not in SCHEMA_KINDS:
            return f"unknown schema input kind {new!r}"
        row["notes"] = new
    return None


def apply_review(csv_path: Path = SAMPLE_PATH, data_dir: Path = DATA_DIR) -> ReviewOutcome:
    outcome = ReviewOutcome()
    text = csv_path.read_text(encoding="utf-8-sig")
    decisions: dict[str, dict[str, str]] = {s: {} for s in SUITES}
    for rec in csv.DictReader(io.StringIO(text)):
        verdict = (rec.get("verdict") or "").strip()
        if rec.get("suite") not in decisions:
            outcome.errors.append(f"{rec.get('id')}: unknown suite")
            continue
        if not verdict:
            outcome.counts["blank"] += 1
            continue
        decisions[rec["suite"]][rec["id"]] = verdict

    for suite, by_id in decisions.items():
        if not by_id:
            continue
        rows = load_suite(suite, data_dir, include_dropped=True)
        index = {r["id"]: r for r in rows}
        for rid, verdict in by_id.items():
            row = index.get(rid)
            if row is None:
                outcome.errors.append(f"{rid}: not in {FILES[suite]}")
                continue
            if verdict == "ok":
                row["review"] = "human-ok"
                outcome.counts["ok"] += 1
            elif verdict == "drop":
                row["review"] = "human-dropped"
                outcome.counts["dropped"] += 1
            elif verdict.startswith("edit:"):
                error = _apply_edit(suite, row, verdict[5:].strip())
                if error:
                    outcome.errors.append(f"{rid}: {error}")
                    continue
                row["review"] = "human-edited"
                outcome.counts["edited"] += 1
            else:
                outcome.errors.append(f"{rid}: verdict must be ok, drop or edit:<label>")
        write_jsonl(data_dir / FILES[suite], rows)
    return outcome


def review_doc(status: str, sampled: int, outcome: ReviewOutcome | None = None) -> str:
    lines = [
        "# Dataset review",
        "",
        f"Status: **{status}** (updated {date.today().isoformat()}).",
        "",
        "All rows were written with Amazon Nova from hand-written seeds (PII values are synthetic "
        "and filled in by code) and checked by deterministic "
        "validators (`guardrails dataset validate`, `guardrails dataset leakcheck`). A random "
        f"10% sample ({sampled} rows, seed 7) is in `data/review/review_sample.csv` for a human "
        "spot-check.",
        "",
        "How to review: open the CSV, fill the `verdict` column with `ok`, `drop` or "
        "`edit:<new label>`, save, then run `guardrails dataset apply-review`.",
        "",
    ]
    if outcome is None:
        lines += ["Until the spot-check is done, the README says: \"machine-checked, human "
                  "spot-check pending\"."]  # fmt: skip
    else:
        c = outcome.counts
        lines += [
            "| Verdict | Rows |",
            "|---|---|",
            f"| ok | {c['ok']} |",
            f"| edited | {c['edited']} |",
            f"| dropped | {c['dropped']} |",
            f"| not yet reviewed (blank) | {c['blank']} |",
        ]
        if outcome.errors:
            lines += ["", f"{len(outcome.errors)} row(s) could not be applied; see the CLI output."]
    return "\n".join(lines) + "\n"
