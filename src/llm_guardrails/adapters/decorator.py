"""`@guarded`: put the guardrails around an existing RAG answer function (docs/SPEC.md 16).

    @guarded(policy="banking_tax")
    def answer(question: str, sources: list[SourceDoc]) -> str:
        ...  # your retrieval-augmented call, citing sources as [S1]

    result = answer("What is the minimum balance?", sources)   # -> GuardedCall

The wrapped function:
1. runs the input stage (normalise, policy, PII, injection, documents);
   a block stops here and the function is never called;
2. calls YOUR function with the sanitised question (PII replaced by placeholders) and the
   cleaned, PII-redacted documents;
3. runs the output stage (schema or grounding, leak scan, restore) on its answer.

If your function raises, the result is a block (guard "upstream"), never an unchecked
answer. Async functions get an async wrapper; sync functions a sync one (which must not be
called from inside a running event loop - use an async function there).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any

from llm_guardrails import __version__
from llm_guardrails.pipeline import (
    BLOCKED_MESSAGE,
    GuardedAnswer,
    InputResult,
    OutputResult,
    Pipeline,
)
from llm_guardrails.types import Action, PipelineReport, Reason, SourceDoc, Verdict

logger = logging.getLogger("llm_guardrails.adapters")


@dataclass
class GuardedCall(GuardedAnswer):
    """GuardedAnswer plus both stage results (e.g. for `should_cache`)."""

    input_result: InputResult | None = None
    output_result: OutputResult | None = None


def _pipeline(policy: str, grounding: bool, schema: str | None) -> Pipeline:
    from llm_guardrails.runtime import Pipelines
    from llm_guardrails.settings import load_settings

    pipes = Pipelines(load_settings())
    p = pipes.policy(policy).model_copy(deep=True)
    p.grounding.enabled = grounding
    return Pipeline(p, pipes.llm, pipes.embedder, schema_name=schema)


def _merge(inp: InputResult, out: OutputResult | None, extra: list[Verdict],
           policy_name: str) -> PipelineReport:  # fmt: skip
    verdicts = list(inp.report.verdicts) + (list(out.report.verdicts) if out else []) + extra
    blocked = inp.decision == "block" or (out is not None and out.decision == "block") or any(
        v.action is Action.BLOCK for v in extra)  # fmt: skip
    stage = "input" if inp.decision == "block" else ("output" if blocked else "complete")
    return PipelineReport(
        request_id=inp.request_id, decision="block" if blocked else "allow", stage=stage,
        verdicts=verdicts,
        latency_ms=inp.report.latency_ms + (out.report.latency_ms if out else 0.0),
        cost_usd=sum(v.cost_usd for v in verdicts),
        tokens_in=sum(v.tokens_in for v in verdicts),
        tokens_out=sum(v.tokens_out for v in verdicts),
        policy_name=policy_name, version=__version__,
    )  # fmt: skip


async def _guarded_call(pipe: Pipeline, fn: Any, question: str,
                        sources: list[SourceDoc] | None) -> GuardedCall:  # fmt: skip
    inp = await pipe.check_input(question, list(sources or []))
    if inp.decision == "block":
        return GuardedCall(inp.request_id, "block", None, None, inp.user_message,
                           _merge(inp, None, [], pipe.policy.name), inp, None)  # fmt: skip
    started = time.perf_counter()
    try:
        answer = fn(inp.sanitized_text, inp.docs)
        if inspect.isawaitable(answer):
            answer = await answer
        if not isinstance(answer, str):
            raise TypeError("the guarded function must return a string")
    except Exception as exc:  # noqa: BLE001 - an upstream failure must never pass unchecked
        logger.warning("guarded function failed with %s", type(exc).__name__)
        failed = Verdict(guard="upstream", action=Action.BLOCK, reasons=[Reason.GUARD_ERROR],
                         error_kind="error",
                         latency_ms=(time.perf_counter() - started) * 1000)  # fmt: skip
        return GuardedCall(inp.request_id, "block", None, None, BLOCKED_MESSAGE,
                           _merge(inp, None, [failed], pipe.policy.name), inp, None)  # fmt: skip
    out = await pipe.check_output(answer, sources=inp.docs, schema_name=pipe.schema_name,
                                  pii_map=inp.pii_map)  # fmt: skip
    report = _merge(inp, out, [], pipe.policy.name)
    allowed = report.decision == "allow"
    return GuardedCall(inp.request_id, report.decision, out.output if allowed else None,
                       out.data if allowed else None, None if allowed else BLOCKED_MESSAGE,
                       report, inp, out)  # fmt: skip


def guarded(policy: str = "banking_tax", grounding: bool = True, schema: str | None = None,
            *, pipeline: Pipeline | None = None):  # fmt: skip
    """Decorate `fn(question, sources) -> str` (sync or async) with the full guardrails.

    `pipeline` lets tests or apps pass their own Pipeline (model, embedder, policy);
    otherwise one is built from the environment (GUARDRAILS_LLM_MODE) on first use."""

    def decorate(fn):
        state: dict[str, Pipeline] = {}

        def pipe() -> Pipeline:
            if "p" not in state:
                state["p"] = pipeline or _pipeline(policy, grounding, schema)
            return state["p"]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(question: str, sources: list[SourceDoc] | None = None):
                return await _guarded_call(pipe(), fn, question, sources)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(question: str, sources: list[SourceDoc] | None = None):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_guarded_call(pipe(), fn, question, sources))
            raise RuntimeError(
                "a sync @guarded function was called inside a running event "
                "loop; make the function async instead"
            )

        return sync_wrapper

    return decorate
