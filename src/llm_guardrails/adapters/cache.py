"""May this exchange go into a cache? (docs/SPEC.md 16)

A cached answer is served to OTHER users later, so the rule is strict: only an exchange
that touched no personal data, was not changed or blocked by any guard, and whose answer
passed grounding may be cached. Anything else is answered fresh every time.
"""

from __future__ import annotations

from typing import Any

from llm_guardrails.types import Action, PipelineReport

NOT_CLEAN = (Action.REDACT, Action.RETRY, Action.BLOCK)


def _report(result: Any) -> PipelineReport | None:
    return getattr(result, "report", None)


def should_cache(input_result: Any, output_result: Any) -> bool:
    """False if either side was blocked, any guard redacted, retried or blocked, any PII
    entity was detected (even one the policy allowed through), or grounding did not pass.
    No grounding verdict at all counts as "did not pass": an unverified answer is never
    cached."""
    reports = [_report(input_result), _report(output_result)]
    if any(r is None or r.decision != "allow" for r in reports):
        return False
    verdicts = [v for r in reports for v in r.verdicts]  # type: ignore[union-attr]
    if any(v.action in NOT_CLEAN for v in verdicts):
        return False
    if any(v.entities for v in verdicts):
        return False
    grounding = [v for v in verdicts if v.guard == "grounding"]
    return bool(grounding) and all(v.action is Action.ALLOW for v in grounding)
