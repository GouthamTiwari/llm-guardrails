"""Grounding evaluation (docs/SPEC.md 11.6, 13.3-13.4).

Positive class = "should be rejected": every gold label except `supported` and
`correct_abstention`. Reject precision = correctly rejected / rejected; reject recall =
correctly rejected / should-reject.

1. `min_overlap` is swept on dev only (0.30-0.80, step 0.05). Objective, fixed before any
   result was seen (SPEC decision 14): the best dev F1 of "should reject"; ties go to the
   value closest to the spec default 0.5, then the lower one. The choice is written to
   reports/grounding_operating_point.json and becomes the policy default (tested).
2. At that value, dev and test are scored by the deterministic checks alone and with the
   entailment judge added (`optional`: a judge failure falls back to the deterministic
   result and is counted). Each answer's outcome is "pass" or the reason it was rejected,
   so the confusion matrix shows WHICH check caught each kind of bad answer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from llm_guardrails.eval.datasets import DATA_DIR, FILES, GROUNDING_LABELS, load_suite
from llm_guardrails.eval.metrics import pct, rate
from llm_guardrails.eval.reports import meta, write_json, write_text
from llm_guardrails.guards.grounding import (
    GroundingResult,
    check_grounding,
    check_grounding_async,
)
from llm_guardrails.types import Action, SourceDoc

RECORDINGS = DATA_DIR / "recordings" / "grounding_eval.jsonl"
OPERATING_POINT = "grounding_operating_point.json"
SHOULD_PASS = frozenset({"supported", "correct_abstention"})
GRID = [round(float(x), 2) for x in np.arange(0.30, 0.8001, 0.05)]
SPEC_DEFAULT = 0.5


def _sources(row: dict) -> list[SourceDoc]:
    return [SourceDoc(**s) for s in row["sources"]]


def outcome(result: GroundingResult) -> str:
    """'pass', or the short name of the first reason (number_mismatch, entailment_...)."""
    if result.action is not Action.BLOCK:
        return "pass"
    return result.reasons[0].split(".", 1)[1] if result.reasons else "blocked"


def reject_metrics(rows: list[dict], rejected: list[bool]) -> dict[str, Any]:
    should = [r["label"] not in SHOULD_PASS for r in rows]
    tp = sum(s and x for s, x in zip(should, rejected, strict=True))
    fp = sum(not s and x for s, x in zip(should, rejected, strict=True))
    fn = sum(s and not x for s, x in zip(should, rejected, strict=True))
    p = tp / (tp + fp) if tp + fp else 0.0
    rr = tp / (tp + fn) if tp + fn else 0.0
    return {
        "reject_precision": rate(tp, tp + fp),
        "reject_recall": rate(tp, tp + fn),
        "f1": round(2 * p * rr / (p + rr), 4) if p + rr else 0.0,
        "good_answers_passed": rate(sum(not s and not x for s, x in
                                        zip(should, rejected, strict=True)),
                                    sum(not s for s in should)),
    }  # fmt: skip


def confusion(rows: list[dict], outcomes: list[str]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = {label: Counter() for label in GROUNDING_LABELS}
    for row, out in zip(rows, outcomes, strict=True):
        table[row["label"]][out] += 1
    return {label: dict(sorted(c.items())) for label, c in table.items() if c}


def deterministic(rows: list[dict], min_overlap: float) -> list[GroundingResult]:
    return [check_grounding(r["answer"], _sources(r), min_overlap=min_overlap) for r in rows]


def sweep(dev: list[dict]) -> dict[str, Any]:
    scored = []
    for m in GRID:
        rejected = [res.action is Action.BLOCK for res in deterministic(dev, m)]
        scored.append((m, reject_metrics(dev, rejected)))
    best_m, best = max(scored, key=lambda x: (x[1]["f1"], -abs(x[0] - SPEC_DEFAULT), -x[0]))
    return {"min_overlap": best_m, "dev": best,
            "grid": {str(m): s["f1"] for m, s in scored}}  # fmt: skip


def write_operating_point(result: dict[str, Any], reports_dir: Path) -> dict[str, Any]:
    point = {
        "source": "dev sweep",
        "min_overlap": result["min_overlap"],
        "objective": "max dev F1 of should-reject; ties: closest to 0.5, then lower",
        "dev_f1_by_min_overlap": result["grid"],
        "dev": result["dev"],
    }
    write_json(OPERATING_POINT, point, reports_dir)
    return point


def run_sweep(reports_dir: Path) -> dict[str, Any]:
    dev = [r for r in load_suite("grounding") if r["split"] == "dev"]
    return write_operating_point(sweep(dev), reports_dir)


async def with_entailment(rows: list[dict], min_overlap: float, llm: Any) -> list[GroundingResult]:
    out = []
    for r in rows:  # one answer at a time, one sentence at a time: no throttling
        out.append(await check_grounding_async(
            r["answer"], _sources(r), min_overlap=min_overlap, entailment="optional",
            llm=llm, timeout_s=120.0, concurrency=1,
        ))  # fmt: skip
    return out


def score(rows: list[dict], results: list[GroundingResult]) -> dict[str, Any]:
    outs = [outcome(r) for r in results]
    block = reject_metrics(rows, [o != "pass" for o in outs])
    block["confusion"] = confusion(rows, outs)
    return block


async def run(*, live: bool, settings: Any, reports_dir: Path) -> dict[str, Any]:
    from llm_guardrails.eval.run_injection import recorder

    point = json.loads((reports_dir / OPERATING_POINT).read_text(encoding="utf-8"))
    m = point["min_overlap"]
    llm = recorder(RECORDINGS, live=live, settings=settings)
    rows = load_suite("grounding")
    report: dict[str, Any] = {"min_overlap": m, "splits": {}}
    spend = 0.0
    for split in ("test", "dev"):
        part = [r for r in rows if r["split"] == split]
        det = deterministic(part, m)
        ent = await with_entailment(part, m, llm)
        n = len(part)
        judged = [r for r in ent if r.judge_calls]
        spend += sum(r.cost_usd for r in ent)
        report["splits"][split] = {
            "answers": n,
            "deterministic": score(part, det),
            "with_entailment": score(part, ent) | {
                "judge_calls": sum(r.judge_calls for r in ent),
                "answers_judged": len(judged),
                "judge_unavailable_fell_back": sum(r.entailment == "unavailable" for r in ent),
                "extra_usd_per_answer": round(sum(r.cost_usd for r in ent) / n, 8),
                "extra_latency_ms_per_judged_answer": (
                    round(sum(r.judge_latency_ms for r in judged) / len(judged), 1)
                    if judged else None),
            },
        }  # fmt: skip
    llm.ensure_complete()
    full = meta(paths=[DATA_DIR / FILES["grounding"]], mode="live" if live else "replay",
                spend_usd=spend, operating_point={"min_overlap": m}) | report  # fmt: skip
    write_json("grounding.json", full, reports_dir)
    write_text("grounding.md", grounding_markdown(full), reports_dir)
    return full


def grounding_markdown(r: dict[str, Any]) -> str:
    t = r["splits"]["test"]
    lines = [
        "# Grounding: does the answer say only what its cited sources say?",
        "",
        f"Mode: {r['mode']}. min_overlap {r['min_overlap']} (dev sweep). Positive class = "
        "answers that should be rejected (uncited, fabricated citation, unsupported, "
        "contradicted, overgeneralised, partially supported).",
        "",
        f"## Test split ({t['answers']} answers)",
        "",
        "| Checks | Reject precision | Reject recall | F1 | Good answers passed |",
        "|---|---|---|---|---|",
    ]
    for key, name in (("deterministic", "Deterministic checks"),
                      ("with_entailment", "Deterministic + LLM entailment judge")):  # fmt: skip
        m = t[key]
        lines.append(f"| {name} | {pct(m['reject_precision'])} | {pct(m['reject_recall'])} | "
                     f"{m['f1']} | {pct(m['good_answers_passed'])} |")  # fmt: skip
    e = t["with_entailment"]
    lines += [
        "",
        f"The judge ran on {e['answers_judged']} answers ({e['judge_calls']} calls), adding USD "
        f"{e['extra_usd_per_answer']:.6f} per answer on average and "
        f"{e['extra_latency_ms_per_judged_answer']} ms per judged answer; "
        f"{e['judge_unavailable_fell_back']} fell back to the deterministic result.",
        "",
    ]
    for key, name in (("deterministic", "deterministic"), ("with_entailment", "+ entailment")):
        conf = t[key]["confusion"]
        cols = sorted({c for row in conf.values() for c in row})
        lines += [
            f"### Confusion matrix, test, {name} (gold label x outcome)",
            "",
            "| Gold label | " + " | ".join(cols) + " |",
            "|---" * (len(cols) + 1) + "|",
        ]
        for label, row in conf.items():
            lines.append(f"| {label} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
        lines.append("")
    d = r["splits"]["dev"]
    lines.append(
        f"Dev split (min_overlap was chosen on it): deterministic F1 "
        f"{d['deterministic']['f1']}, with entailment F1 {d['with_entailment']['f1']}."
    )
    return "\n".join(lines) + "\n"
