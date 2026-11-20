"""Structured-output experiment (docs/SPEC.md 10.4, 13.3-13.4): does a schema in the prompt,
and then error-fed retries, turn free text into JSON the application can trust?

Three styles on every ticket input (all 105 rows: this experiment tunes nothing, so it
does not need a held-out split):
- plain        - "summarise as JSON with title, severity, tags and due_days", one call;
- schema       - the JSON Schema and rules in the system prompt, one call;
- schema_retry - the same prompt plus the retry loop (up to 2 retries, each told exactly
                 what was wrong).

`schema` and the first attempt of `schema_retry` send the identical request, so the retry
style replays that recorded call and only its retries are new.
"""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from llm_guardrails.eval.datasets import DATA_DIR, FILES, load_suite
from llm_guardrails.eval.metrics import pct, rate
from llm_guardrails.eval.reports import meta, write_json, write_text
from llm_guardrails.guards.schema import generate_validated, ticket_prompt
from llm_guardrails.types import SchemaValidationFailed

RECORDINGS = DATA_DIR / "recordings" / "schema_eval.jsonl"
PURPOSE = "schema_eval"
STYLES = {"plain": 0, "schema": 0, "schema_retry": 2}  # style -> max retries
REFUSAL_KINDS = {"empty", "hostile", "off_topic"}


async def run_one(llm: Any, style: str, text: str) -> dict[str, Any]:
    system, user = ticket_prompt("plain" if style == "plain" else "schema", text)
    try:
        result = await generate_validated(llm, schema_name="TicketSummary", system=system,
                                          user=user, purpose=PURPOSE,
                                          max_retries=STYLES[style])  # fmt: skip
        attempts, value, valid = result.attempts, result.value, True
    except SchemaValidationFailed as e:
        attempts, value, valid = e.trace, None, False
    return {
        "valid": valid,
        "valid_first": bool(attempts[0].valid),
        "attempts": len(attempts),
        "first_error_kinds": sorted(set(attempts[0].kinds)),
        "tokens": [a.tokens_in + a.tokens_out for a in attempts],
        "cost": [a.cost_usd for a in attempts],
        "latency_ms": [a.latency_ms for a in attempts],
        "kind": getattr(value, "kind", None),
    }


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def summarise(rows: list[dict], results: list[dict[str, Any]], style: str) -> dict[str, Any]:
    n = len(results)
    extra_tokens = [sum(r["tokens"][1:]) for r in results]
    extra_cost = [sum(r["cost"][1:]) for r in results]
    added_latency = [sum(r["latency_ms"][1:]) for r in results]
    retried = [x for x, r in zip(added_latency, results, strict=True) if r["attempts"] > 1]
    should_refuse = [(row, r) for row, r in zip(rows, results, strict=True)
                     if row["notes"] in REFUSAL_KINDS and r["valid"]]  # fmt: skip
    should_ticket = [(row, r) for row, r in zip(rows, results, strict=True)
                     if row["notes"] not in REFUSAL_KINDS and r["valid"]]  # fmt: skip
    return {
        "inputs": n,
        "first_try_validity": rate(sum(r["valid_first"] for r in results), n),
        "final_validity": rate(sum(r["valid"] for r in results), n),
        "mean_retries": round(sum(r["attempts"] - 1 for r in results) / n, 4),
        "extra_tokens_per_input": round(sum(extra_tokens) / n, 2),
        "extra_usd_per_input": round(sum(extra_cost) / n, 8),
        "usd_per_input": round(sum(sum(r["cost"]) for r in results) / n, 8),
        "added_latency_ms_when_retried": {"p50": _p(retried, 50), "p95": _p(retried, 95),
                                          "inputs_retried": len(retried)},  # fmt: skip
        "first_attempt_errors": dict(sorted(Counter(k for r in results
                                                    for k in r["first_error_kinds"]).items())),
        "refused_when_it_should": rate(sum(r["kind"] == "refusal" for _, r in should_refuse),
                                       len(should_refuse)),
        "ticket_when_it_should": rate(sum(r["kind"] == "ticket" for _, r in should_ticket),
                                      len(should_ticket)),
    }  # fmt: skip


async def run(*, live: bool, settings: Any, reports_dir: Path) -> dict[str, Any]:
    from llm_guardrails.eval.run_injection import recorder

    llm = recorder(RECORDINGS, live=live, settings=settings)
    rows = load_suite("schema")
    report: dict[str, Any] = {"styles": {}}
    spend = 0.0
    for style in STYLES:
        results = [await run_one(llm, style, r["input"]) for r in rows]
        report["styles"][style] = summarise(rows, results, style)
        # Count each recorded call once: schema_retry's first attempt IS the schema call.
        spend += sum(sum(r["cost"][1:] if style == "schema_retry" else r["cost"])
                     for r in results)  # fmt: skip
    llm.ensure_complete()
    full = meta(paths=[DATA_DIR / FILES["schema"]], mode="live" if live else "replay",
                spend_usd=spend, operating_point=None) | report  # fmt: skip
    write_json("schema.json", full, reports_dir)
    write_text("schema.md", schema_markdown(full), reports_dir)
    return full


def schema_markdown(r: dict[str, Any]) -> str:
    s = r["styles"]
    lines = [
        "# Structured output: plain vs schema vs schema + retries",
        "",
        f"Mode: {r['mode']}. {s['plain']['inputs']} support-ticket inputs (normal, "
        "multiple-issue, number-heavy, Hindi/Hinglish, hostile, empty, very long, off-topic), "
        "validated strictly against `TicketSummary | Refusal` (no type coercion, no extra keys).",
        "",
        "| Style | Valid on first try | Valid after retries | Mean retries | USD per input |",
        "|---|---|---|---|---|",
    ]
    names = {"plain": "Plain prompt", "schema": "Schema in prompt",
             "schema_retry": "Schema + error-fed retries (shipped)"}  # fmt: skip
    for key, m in s.items():
        lines.append(f"| {names[key]} | {pct(m['first_try_validity'])} | "
                     f"{pct(m['final_validity'])} | {m['mean_retries']} | "
                     f"{m['usd_per_input']:.6f} |")  # fmt: skip
    sr = s["schema_retry"]
    lat = sr["added_latency_ms_when_retried"]
    lines += [
        "",
        f"Retries cost {sr['extra_tokens_per_input']} extra tokens and USD "
        f"{sr['extra_usd_per_input']:.6f} per input on average; {lat['inputs_retried']} "
        f"input(s) needed a retry (added latency p50 {lat['p50']} ms, p95 {lat['p95']} ms).",
        "",
        "Refusal behaviour (schema + retries, valid outputs only): refused empty, hostile or "
        f"off-topic input {pct(sr['refused_when_it_should'])}; produced a ticket for real "
        f"tickets {pct(sr['ticket_when_it_should'])}.",
        "",
        "## First-attempt errors by type",
        "",
    ]  # fmt: skip
    kinds = sorted({k for m in s.values() for k in m["first_attempt_errors"]})
    lines.append("| Style | " + " | ".join(kinds) + " |")
    lines.append("|---" * (len(kinds) + 1) + "|")
    for key, m in s.items():
        lines.append(
            f"| {names[key]} | "
            + " | ".join(str(m["first_attempt_errors"].get(k, 0)) for k in kinds)
            + " |"
        )
    return "\n".join(lines) + "\n"
