"""Prometheus metrics (docs/SPEC.md 14.2), served by the API at /metrics.

Everything is derived from the PipelineReport after a request finishes, so guards stay free
of global state and the numbers always agree with the audit log. Recording never raises:
observability must never block a user (spec 4.3, "metrics/log export: open").
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Histogram

from llm_guardrails.types import Action, PipelineReport, Reason

logger = logging.getLogger("llm_guardrails.metrics")

LATENCY_BUCKETS = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)  # fmt: skip

# Label values are fixed vocabularies (guard names, actions, reason codes, entity types),
# never user text: a label holding free text would also explode the number of series.
REQUESTS = Counter("guardrails_requests_total", "Requests by route and decision.",
                   ["route", "decision"])  # fmt: skip
GUARD_ACTIONS = Counter("guardrails_guard_actions_total", "Verdicts by guard and action.",
                        ["guard", "action"])  # fmt: skip
BLOCKS = Counter("guardrails_blocks_total", "Blocking verdicts by guard and reason code.",
                 ["guard", "reason"])  # fmt: skip
REDACTIONS = Counter("guardrails_redactions_total", "Redacted PII entities by type.",
                     ["entity"])  # fmt: skip
SCHEMA_RETRIES = Counter("guardrails_schema_retries_total", "Extra model calls to fix JSON.")
GROUNDING_REJECTS = Counter("guardrails_grounding_rejects_total",
                            "Grounding failures (rejected or stripped) by reason.",
                            ["reason"])  # fmt: skip
GUARD_ERRORS = Counter("guardrails_guard_errors_total", "Guard errors and timeouts.",
                       ["guard", "kind"])  # fmt: skip
CANARY_LEAKS = Counter("guardrails_canary_leaks_total", "Answers that contained the canary.")
LLM_COST = Counter("guardrails_llm_cost_usd_total", "Model spend in USD by purpose.",
                   ["purpose"])  # fmt: skip
GUARD_LATENCY = Histogram("guardrails_guard_latency_seconds", "Guard latency.", ["guard"],
                          buckets=LATENCY_BUCKETS)  # fmt: skip


def _redacted_entities(v) -> dict[str, int]:
    """Entity type -> count for a verdict that redacted PII (input PII or output leak)."""
    types: set[str] = set()
    for r in v.reasons:
        output_redaction = r.startswith("leak.pii.") and v.action is Action.REDACT
        if r.startswith("pii.redact.") or output_redaction:
            types.add(r.rsplit(".", 1)[-1])
    counts = {e.type: e.count for e in v.entities}
    return {t: counts.get(t, 1) for t in types}


def record(report: PipelineReport, route: str) -> None:
    try:
        _record(report, route)
    except Exception as exc:  # noqa: BLE001 - metrics fail open, by design
        logger.warning("metrics recording failed with %s", type(exc).__name__)


def _record(report: PipelineReport, route: str) -> None:
    REQUESTS.labels(route=route, decision=report.decision).inc()
    for v in report.verdicts:
        GUARD_ACTIONS.labels(guard=v.guard, action=v.action.value).inc()
        GUARD_LATENCY.labels(guard=v.guard).observe(v.latency_ms / 1000)
        if v.action is Action.BLOCK:
            for reason in v.reasons or ["none"]:
                BLOCKS.labels(guard=v.guard, reason=reason).inc()
        for entity, n in _redacted_entities(v).items():
            REDACTIONS.labels(entity=entity).inc(n)
        if v.guard == "model" and v.scores.get("retries"):
            SCHEMA_RETRIES.inc(v.scores["retries"])
        if v.guard == "grounding" and v.action in (Action.BLOCK, Action.REDACT):
            for reason in v.reasons:
                GROUNDING_REJECTS.labels(reason=reason).inc()
        if v.error_kind is not None:
            GUARD_ERRORS.labels(guard=v.guard, kind=v.error_kind).inc()
        if v.guard == "leak" and Reason.LEAK_CANARY in v.reasons:
            CANARY_LEAKS.inc()
        if v.cost_usd > 0:
            LLM_COST.labels(purpose=v.guard).inc(v.cost_usd)
