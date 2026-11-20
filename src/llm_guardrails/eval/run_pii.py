"""PII evaluation (docs/SPEC.md 13.3): per-type precision/recall/F1, look-alike false
positives, and the three-step ablation. No model calls, so it always runs offline.

Detection runs on the normaliser's clean text, exactly as the PII guard does (so Hindi
digits are folded first). A prediction is a true positive when the same (type, normalised
value) is in the row's gold entities; matching is a multiset, so a value written twice
must be found twice.

Ablation (each step on top of the previous one):
1. regex only      - patterns, no checksum or format validation, no context words;
2. + validation    - Verhoeff, Luhn + card prefix, GSTIN check character, PAN holder letter;
3. + context       - negative words such as "order", "tracking", "invoice" suppress a match
                     unless a positive word ("aadhaar", "card", ...) is near (the shipped
                     configuration).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from llm_guardrails.detectors.entities import detect_entities, get_detector
from llm_guardrails.eval.datasets import DATA_DIR, FILES, load_suite
from llm_guardrails.eval.metrics import precision_recall_f1, rate
from llm_guardrails.eval.reports import meta, write_json, write_text
from llm_guardrails.eval.synth import ascii_digits
from llm_guardrails.guards.normalize import normalize
from llm_guardrails.policy_config import load_policy

STEP_LABELS = {
    "regex_only": "regex only",
    "regex_validation": "regex + validation",
    "regex_validation_context": "regex + validation + context (shipped)",
}
STEPS = {
    "regex_only": {"validate": False, "use_context": False},
    "regex_validation": {"validate": True, "use_context": False},
    "regex_validation_context": {"validate": True, "use_context": True},  # shipped
}


def gold_pairs(row: dict) -> Counter:
    return Counter((e["type"], get_detector(e["type"]).normalize(ascii_digits(e["value"])))
                   for e in row["entities"])  # fmt: skip


def predicted_pairs(row: dict, *, validate: bool, use_context: bool,
                    min_confidence: float) -> Counter:  # fmt: skip
    result = detect_entities(normalize(row["text"]).clean, min_confidence=min_confidence,
                             validate=validate, use_context=use_context)  # fmt: skip
    return Counter((m.type, m.value) for m in result.matches)


def evaluate(rows: list[dict], *, validate: bool, use_context: bool,
             min_confidence: float) -> dict[str, Any]:  # fmt: skip
    tp, fp, fn = Counter(), Counter(), Counter()
    lookalike_fp_rows = 0
    lookalike_rows = 0
    for row in rows:
        gold = gold_pairs(row)
        pred = predicted_pairs(row, validate=validate, use_context=use_context,
                               min_confidence=min_confidence)  # fmt: skip
        hit = gold & pred
        for (etype, _), n in hit.items():
            tp[etype] += n
        for (etype, _), n in (pred - gold).items():
            fp[etype] += n
        for (etype, _), n in (gold - pred).items():
            fn[etype] += n
        if not row["entities"]:
            lookalike_rows += 1
            lookalike_fp_rows += bool(pred)
    types = sorted(set(tp) | set(fp) | set(fn))
    return {
        "per_type": {t: precision_recall_f1(tp[t], fp[t], fn[t]) for t in types},
        "micro": precision_recall_f1(sum(tp.values()), sum(fp.values()), sum(fn.values())),
        "rows_without_pii_with_a_detection": rate(lookalike_fp_rows, lookalike_rows),
    }


def run(*, reports_dir: Path) -> dict[str, Any]:
    min_conf = load_policy("default").pii.min_confidence
    rows = load_suite("pii")
    report: dict[str, Any] = {"min_confidence": min_conf}
    for split in ("test", "dev"):
        part = [r for r in rows if r["split"] == split]
        steps = {name: evaluate(part, min_confidence=min_conf, **flags)
                 for name, flags in STEPS.items()}  # fmt: skip
        report[split] = {
            "rows": len(part),
            "rows_without_pii": sum(not r["entities"] for r in part),
            "shipped": steps["regex_validation_context"],
            "ablation": {
                name: {"micro": s["micro"],
                       "rows_without_pii_with_a_detection":
                           s["rows_without_pii_with_a_detection"]}
                for name, s in steps.items()
            },
        }  # fmt: skip
    full = meta(paths=[DATA_DIR / FILES["pii"]], mode="offline", spend_usd=0.0,
                operating_point=None) | report  # fmt: skip
    write_json("pii.json", full, reports_dir)
    write_text("pii.md", pii_markdown(full), reports_dir)
    return full


def _num(x: Any) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def pii_markdown(r: dict[str, Any]) -> str:
    t = r["test"]
    lines = [
        "# PII detection",
        "",
        f"Test split: {t['rows']} rows, {t['rows_without_pii']} of them with no PII "
        "(look-alikes: order and tracking numbers, mistyped Aadhaar, declined-payment "
        f"references, promo codes...). Minimum confidence {r['min_confidence']}. No model "
        "calls: this report always runs offline.",
        "",
        "## Per type (shipped: regex + validation + context)",
        "",
        "| Type | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for etype, m in t["shipped"]["per_type"].items():
        lines.append(f"| {etype} | {_num(m['precision'])} | {_num(m['recall'])} | "
                     f"{_num(m['f1'])} | {m['tp']} | {m['fp']} | {m['fn']} |")  # fmt: skip
    m = t["shipped"]["micro"]
    lines += [
        f"| **micro** | {_num(m['precision'])} | {_num(m['recall'])} | {_num(m['f1'])} | "
        f"{m['tp']} | {m['fp']} | {m['fn']} |",
        "",
        "## Ablation (test split): what each step removes",
        "",
        "| Configuration | Precision | Recall | False positives | Rows without PII that got "
        "a detection |",
        "|---|---|---|---|---|",
    ]  # fmt: skip
    for name, s in t["ablation"].items():
        mm, la = s["micro"], s["rows_without_pii_with_a_detection"]
        lines.append(f"| {STEP_LABELS[name]} | {_num(mm['precision'])} | "
                     f"{_num(mm['recall'])} | {mm['fp']} | {la['k']}/{la['n']} |")  # fmt: skip
    d = r["dev"]["shipped"]["micro"]
    lines += ["", f"Dev split (shipped, for reference): precision {_num(d['precision'])}, "
              f"recall {_num(d['recall'])}, F1 {_num(d['f1'])}.", ""]  # fmt: skip
    return "\n".join(lines)
