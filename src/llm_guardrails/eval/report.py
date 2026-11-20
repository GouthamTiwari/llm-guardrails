"""reports/summary.md and the README results block (docs/SPEC.md 13.7).

The numbers rule: every number in the README comes from a committed file in reports/.
This module builds the summary ONLY from those JSON files (never from memory or a live
run), `guardrails report readme` copies it between the README markers, and
tests/test_reports.py fails CI if the README block, summary.md and the JSON files
disagree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_guardrails.eval.datasets import REPO_ROOT
from llm_guardrails.eval.metrics import pct
from llm_guardrails.eval.reports import REPORTS_DIR, write_text

README = REPO_ROOT / "README.md"
REVIEW_DOC = REPO_ROOT / "docs" / "DATASET_REVIEW.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(reports_dir: Path, name: str) -> dict[str, Any] | None:
    path = reports_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _review_status(path: Path = REVIEW_DOC) -> str:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    m = re.search(r"Status: \*\*([^*]+)\*\*", text)
    status = m.group(1) if m else "pending"
    if status == "pending":
        return "assistant-reviewed, human spot-check pending"
    return f"human spot-check {status} (see docs/DATASET_REVIEW.md)"


def summary_markdown(reports_dir: Path = REPORTS_DIR, review_doc: Path = REVIEW_DOC) -> str:
    inj = _load(reports_dir, "injection.json")
    pii = _load(reports_dir, "pii.json")
    sch = _load(reports_dir, "schema.json")
    grd = _load(reports_dir, "grounding.json")
    e2e = _load(reports_dir, "e2e.json")
    base = _load(reports_dir, "baseline_bedrock_guardrails.json")
    rows = ["| Check | Result (test split, 95% interval, counts) | Report |", "|---|---|---|"]
    if inj:
        t = inj["test"]["layers"]["all"]
        rows += [
            f"| Prompt injection: attacks blocked | {pct(t['attack_recall'])} | "
            "[injection](reports/injection.md) |",
            f"| Prompt injection: normal messages blocked | {pct(t['benign_fpr'])} | "
            "[injection](reports/injection.md) |",
            f"| Prompt injection: tricky-but-harmless messages blocked | "
            f"{pct(inj['test']['hard_benign_fpr'])} | [injection](reports/injection.md) |",
        ]
        ext = inj.get("external")
        if ext:
            e = ext["layers"]["all"]
            rows.append(
                f"| Prompt injection, out of domain (deepset test split): attacks "
                f"blocked / normal blocked | {pct(e['attack_recall'])} / "
                f"{pct(e['benign_fpr'])} | [injection](reports/injection.md) |"
            )
    if pii:
        m = pii["test"]["shipped"]["micro"]
        ab = pii["test"]["ablation"]
        rows += [
            f"| PII detection (8 Indian ID types): precision / recall / F1 | "
            f"{100 * m['precision']:.1f}% / {100 * m['recall']:.1f}% / {100 * m['f1']:.1f}% "
            f"(TP {m['tp']}, FP {m['fp']}, FN {m['fn']}) | [pii](reports/pii.md) |",
            f"| PII look-alikes flagged: regex only → with checksums | "
            f"{ab['regex_only']['rows_without_pii_with_a_detection']['k']}/"
            f"{ab['regex_only']['rows_without_pii_with_a_detection']['n']} → "
            f"{ab['regex_validation_context']['rows_without_pii_with_a_detection']['k']}/"
            f"{ab['regex_validation_context']['rows_without_pii_with_a_detection']['n']} | "
            "[pii](reports/pii.md) |",
        ]
    if sch:
        s = sch["styles"]
        rows += [
            f"| Structured output valid on first try: plain prompt → schema in prompt | "
            f"{pct(s['plain']['first_try_validity'])} → {pct(s['schema']['first_try_validity'])} "
            "| [schema](reports/schema.md) |",
            f"| Structured output valid after error-fed retries | "
            f"{pct(s['schema_retry']['final_validity'])} | [schema](reports/schema.md) |",
        ]
    if grd:
        g = grd["splits"]["test"]
        rows += [
            f"| Grounding: bad answers rejected (deterministic → + LLM judge) | "
            f"{pct(g['deterministic']['reject_recall'])} → "
            f"{pct(g['with_entailment']['reject_recall'])} | [grounding](reports/grounding.md) |",
            f"| Grounding: good answers passed (deterministic → + LLM judge) | "
            f"{pct(g['deterministic']['good_answers_passed'])} → "
            f"{pct(g['with_entailment']['good_answers_passed'])} | "
            "[grounding](reports/grounding.md) |",
        ]
    if e2e:
        rows += [
            f"| End to end ({e2e['requests']} requests): decision matched the label | "
            f"{pct(e2e['decision_matches_label'])} | [e2e](reports/e2e.md) |",
            f"| End to end, compromised model: leaks caught at the output | "
            f"{pct(e2e['compromised_model']['caught_at_output'])} | [e2e](reports/e2e.md) |",
            f"| End to end: raw PII that reached the model | "
            f"{pct(e2e['pii']['raw_pii_reached_model'])} | [e2e](reports/e2e.md) |",
        ]
        ca = e2e.get("context_ablation")
        if ca:
            rows.append(f"| Poisoned retrieved documents neutralised: document classifier off "
                        f"(default) → on | {pct(ca['classifier_off']['poisoned_neutralised'])} → "
                        f"{pct(ca['classifier_on']['poisoned_neutralised'])} | "
                        "[e2e](reports/e2e.md) |")  # fmt: skip
        live = e2e.get("live_latency")
        if live:
            ms = live["end_to_end_ms"]
            rows.append(f"| Latency, live sample of {live['requests']} requests: p50 / p95 end "
                        f"to end | {ms['p50']:.0f} ms / {ms['p95']:.0f} ms | "
                        "[e2e](reports/e2e.md) |")  # fmt: skip
    if base:
        rows.append(
            f"| Amazon Bedrock Guardrails baseline | {base.get('summary', 'see report')} "
            "| [baseline](reports/baseline_bedrock_guardrails.md) |"
        )
    spend = sum(r.get("spend_usd", 0.0) for r in (inj, pii, sch, grd, e2e, base) if r)
    lines = [
        *rows,
        "",
        f"All data is synthetic or licence-clean; labels are {_review_status(review_doc)}. "
        "Thresholds were tuned on the dev split only and the test split was evaluated once. "
        f"Model calls recorded in these reports cost USD {spend:.4f}; `guardrails eval "
        "--suite all` replays them offline for free.",
    ]
    return "\n".join(lines) + "\n"


def write_summary(reports_dir: Path = REPORTS_DIR) -> str:
    text = summary_markdown(reports_dir)
    write_text("summary.md", text, reports_dir)
    return text


def readme_block(summary: str) -> str:
    return f"{START}\n{summary.rstrip()}\n{END}"


def update_readme(summary: str, readme: Path = README) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise ValueError("README has no results markers")
    new = pattern.sub(lambda _: readme_block(summary), text)
    if new != text:
        readme.write_text(new, encoding="utf-8", newline="\n")
    return new != text


def current_block(readme: Path = README) -> str:
    text = readme.read_text(encoding="utf-8")
    m = re.search(re.escape(START) + r".*?" + re.escape(END), text, re.S)
    return m.group(0) if m else ""
