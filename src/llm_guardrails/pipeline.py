"""The pipeline: every guard in order around one model call (docs/SPEC.md 14).

    input:  normalize -> policy -> pii (+ documents) -> injection -> context
    model:  one Converse call, or the schema retry loop when a schema is set
    output: schema -> grounding -> leak -> restore

Composition is where fail-open bugs usually hide, so the rules here are blunt:
- every step runs through `run_guard` with a timeout and the policy's fail mode, and every
  security step is fail-closed: an exception or timeout is a BLOCK, never an allow;
- the first BLOCK stops the request, and a request blocked on input never calls the model;
- a RETRY verdict (schema guard on someone else's output) counts as a block;
- the whole request has its own deadline (`timeouts_s.request`, 90 s by default);
- one audit line and one metrics update per request, built from allow-listed fields only,
  and neither may ever block or crash a request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from pydantic import BaseModel

from llm_guardrails import __version__, metrics
from llm_guardrails.detectors.entities import detect_entities
from llm_guardrails.guards import run_guard
from llm_guardrails.guards.context import ContextGuard
from llm_guardrails.guards.grounding import ABSTENTION, GroundingGuard
from llm_guardrails.guards.injection import InjectionGuard
from llm_guardrails.guards.leak import LeakGuard, new_canary, with_canary
from llm_guardrails.guards.normalize import NormalizeGuard, normalize
from llm_guardrails.guards.pii import (
    PiiGuard,
    Redactor,
    escape_placeholders,
    restore,
)
from llm_guardrails.guards.policy import PolicyGuard
from llm_guardrails.guards.schema import (
    SchemaGuard,
    generate_validated,
    schema_verdict,
    ticket_prompt,
    validate_output,
)
from llm_guardrails.llm import Message
from llm_guardrails.logging_utils import audit_log, summarize_entities
from llm_guardrails.policy_config import Policy
from llm_guardrails.schemas import json_schema
from llm_guardrails.types import (
    Action,
    GuardContext,
    PipelineReport,
    Reason,
    SchemaValidationFailed,
    SourceDoc,
    Verdict,
)

logger = logging.getLogger("llm_guardrails.pipeline")

ROUTE = "guarded-chat"
BLOCKED_MESSAGE = "Sorry, I can't help with that request."
INPUT_GUARDS = ("normalize", "policy", "pii", "injection", "context")
CITATION_INSTRUCTIONS = (
    "Cite the id of the document after every factual sentence, like [S1] or [S1, S2]. "
    f'If the documents do not contain the answer, reply exactly: "{ABSTENTION}"'
)


def default_system_prompt() -> str:
    folder = resources.files("llm_guardrails.prompts")
    return folder.joinpath("system_default.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- results


@dataclass
class InputResult:
    request_id: str
    decision: str  # "allow" | "block"
    sanitized_text: str | None = field(repr=False)  # redacted question; None when blocked
    wrapped_context: str = field(repr=False)  # preamble + nonce-wrapped clean documents
    docs: list[SourceDoc] = field(repr=False)  # clean, PII-redacted documents
    pii_map: dict[str, str] = field(repr=False)
    user_message: str | None
    report: PipelineReport


@dataclass
class OutputResult:
    request_id: str
    decision: str
    output: str | None = field(repr=False)  # final text (restored per policy); None if blocked
    data: BaseModel | None
    report: PipelineReport


@dataclass
class GuardedAnswer:
    request_id: str
    decision: str
    answer: str | None = field(repr=False)
    data: BaseModel | None
    user_message: str | None
    report: PipelineReport


# ----------------------------------------------------------------------------- helpers


class _Step:
    """Wraps an internal pipeline step (document redaction, the model call) as a guard, so
    it gets the same timeout and fail-closed handling as every real guard."""

    security = True

    def __init__(self, name: str, fn: Callable[[GuardContext], Awaitable[Verdict]]) -> None:
        self.name = name
        self._fn = fn

    async def check(self, ctx: GuardContext) -> Verdict:
        return await self._fn(ctx)


@dataclass
class _Run:
    """Mutable state of one request, so a request-level timeout can still report what ran."""

    request_id: str
    verdicts: list[Verdict] = field(default_factory=list)
    stage: str = "input"
    start: float = field(default_factory=time.perf_counter)

    def blocked(self) -> bool:
        return any(v.action in (Action.BLOCK, Action.RETRY) for v in self.verdicts)

    def report(self, policy: Policy) -> PipelineReport:
        blocked = self.blocked()
        return PipelineReport(
            request_id=self.request_id,
            decision="block" if blocked else "allow",
            stage=self.stage if blocked else "complete",  # type: ignore[arg-type]
            verdicts=self.verdicts,
            latency_ms=(time.perf_counter() - self.start) * 1000,
            cost_usd=sum(v.cost_usd for v in self.verdicts),
            tokens_in=sum(v.tokens_in for v in self.verdicts),
            tokens_out=sum(v.tokens_out for v in self.verdicts),
            policy_name=policy.name,
            version=__version__,
        )

    def user_message(self) -> str | None:
        if not self.blocked():
            return None
        for v in self.verdicts:
            message = v.detail.get("user_message")
            if v.action is Action.BLOCK and isinstance(message, str):
                return message  # fixed, value-free text (e.g. the PII guard's card message)
        return BLOCKED_MESSAGE


def _redact_documents(
    docs: list[SourceDoc], policy: Policy, redactor: Redactor
) -> tuple[list[SourceDoc], list[tuple[str, str]]]:
    """PII in retrieved documents is replaced by placeholders numbered together with the
    question's (SPEC decision 10). A type whose input action is `block` is redacted here
    too: the customer didn't type it, so blocking their request would not help them, but
    the model still must not see it. Placeholder look-alikes inside a document are escaped
    first, or a document could make `restore` put the customer's own data somewhere else."""
    out: list[SourceDoc] = []
    found: list[tuple[str, str]] = []
    for doc in docs:
        text = escape_placeholders(doc.text)
        result = detect_entities(
            text, min_confidence=policy.pii.min_confidence, types=policy.pii.entities.keys()
        )
        pieces: list[str] = []
        cursor = 0
        for m in result.matches:
            if policy.pii.entities[m.type].input == "allow":
                continue
            pieces += [text[cursor : m.start], redactor.placeholder_for(m)]
            cursor = m.end
            found.append((m.type, m.value))
        pieces.append(text[cursor:])
        out.append(SourceDoc(id=doc.id, text="".join(pieces)))
    return out, found


def schema_system_prompt(schema_name: str) -> str:
    if schema_name == "TicketSummary":
        return ticket_prompt("schema", "")[0]
    schema = json.dumps(json_schema(schema_name), indent=1)
    return (
        "Reply with exactly one JSON object and nothing else: no prose, no code fences. "
        f"It must match this JSON Schema:\n{schema}"
    )


# ----------------------------------------------------------------------------- pipeline


class Pipeline:
    def __init__(
        self,
        policy: Policy,
        llm: Any,
        embedder: Any = None,
        system_prompt: str | None = None,
        schema_name: str | None = None,
        *,
        guards: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.llm = llm
        self.embedder = embedder
        self.system_prompt = system_prompt if system_prompt is not None else default_system_prompt()
        self.schema_name = schema_name
        # Replaceable per name, which is how the fail-closed tests inject broken guards.
        self.guards: dict[str, Any] = {
            "normalize": NormalizeGuard(),
            "policy": PolicyGuard(),
            "pii": PiiGuard(),
            "injection": InjectionGuard(),
            "context": ContextGuard(),
            "schema": SchemaGuard(),
            "grounding": GroundingGuard(),
            "leak": LeakGuard(),
        } | (guards or {})

    # ------------------------------------------------------------------ plumbing

    def _ctx(self, request_id: str, text: str = "") -> GuardContext:
        return GuardContext(
            request_id=request_id, text=text, policy=self.policy, llm=self.llm,
            embedder=self.embedder,
        )  # fmt: skip

    async def _guard(self, run: _Run, name: str, ctx: GuardContext) -> Verdict:
        v = await run_guard(
            self.guards[name], ctx, timeout_s=self.policy.timeout_for(name),
            fail_mode=self.policy.fail_mode_for(name),
        )  # fmt: skip
        run.verdicts.append(v)
        return v

    async def _step(self, run: _Run, step: _Step, ctx: GuardContext, timeout_s: float) -> Verdict:
        v = await run_guard(step, ctx, timeout_s=timeout_s, fail_mode="closed")
        run.verdicts.append(v)
        return v

    def _finish(self, run: _Run, route: str) -> PipelineReport:
        report = run.report(self.policy)
        try:
            audit_log(report, route)
        except Exception as exc:  # noqa: BLE001 - logging fails open: never block a user
            logger.warning("audit log failed with %s", type(exc).__name__)
        metrics.record(report, route)
        return report

    # ------------------------------------------------------------------ input stage

    async def _input_stage(self, run: _Run, ctx: GuardContext) -> tuple[bool, str, list[SourceDoc]]:
        """Returns (ok, wrapped_context, clean_docs). Mutates ctx as the text is cleaned."""
        run.stage = "input"
        v = await self._guard(run, "normalize", ctx)
        if v.action is Action.BLOCK:
            return False, "", []
        ctx.scan = v.detail.get("scan")
        ctx.text = v.modified_text if v.modified_text is not None else ctx.text

        if (await self._guard(run, "policy", ctx)).action is Action.BLOCK:
            return False, "", []

        v = await self._guard(run, "pii", ctx)
        if v.action is Action.BLOCK:
            return False, "", []
        if v.modified_text is not None:
            ctx.text = v.modified_text
        ctx.pii_map = dict(v.detail.get("pii_map", {}))

        async def redact_docs_and_rescan(c: GuardContext) -> Verdict:
            redactor = Redactor.from_map(c.pii_map)
            docs, found = await asyncio.to_thread(
                _redact_documents, c.context_docs, self.policy, redactor
            )
            c.context_docs = docs
            c.pii_map = dict(redactor.pii_map)
            # Rebuild the scan copy from the REDACTED question: the injection layers (and
            # their embedding and classifier calls) must never see raw PII (spec 3).
            c.scan = await asyncio.to_thread(normalize, c.text)
            return Verdict(
                guard="pii_docs",
                action=Action.REDACT if found else Action.ALLOW,
                scores={"docs": float(len(docs)), "entities": float(len(found))},
                entities=summarize_entities(found),
            )

        step = _Step("pii_docs", redact_docs_and_rescan)
        budget = self.policy.timeout_for("pii") * (1 + len(ctx.context_docs))
        if (await self._step(run, step, ctx, budget)).action is Action.BLOCK:
            return False, "", []

        if (await self._guard(run, "injection", ctx)).action is Action.BLOCK:
            return False, "", []

        v = await self._guard(run, "context", ctx)
        if v.action is Action.BLOCK:
            return False, "", []
        docs = list(v.detail.get("docs", []))
        return True, v.modified_text or "", docs

    # ------------------------------------------------------------------ model call

    def _prompts(self, question: str, wrapped: str, schema_name: str | None) -> tuple[str, str]:
        if schema_name:
            system = schema_system_prompt(schema_name)
            safe = question.replace("<<<", "‹‹‹").replace(">>>", "›››")
            user = f"Input:\n<<<INPUT>>>\n{safe}\n<<<END INPUT>>>"
        else:
            system = self.system_prompt.rstrip()
            if self.policy.grounding.enabled:
                system += "\n\n" + CITATION_INSTRUCTIONS
            user = f"Question: {question}"
        if wrapped:
            user = f"{wrapped}\n\n{user}"
        return system, user

    async def _model_stage(self, run: _Run, ctx: GuardContext, wrapped: str) -> bool:
        schema_name = ctx.schema_name
        system, user = self._prompts(ctx.text, wrapped, schema_name)
        ctx.canary = new_canary()
        full_system = with_canary(system, ctx.canary)
        # The leak guard compares the answer with the prompt, minus phrases the model is
        # TOLD to say: a correct abstention must not count as a system-prompt leak.
        ctx.system_prompt = full_system.replace(ABSTENTION, " ")
        max_tokens = self.policy.limits.max_output_tokens

        async def call(c: GuardContext) -> Verdict:
            if schema_name:
                try:
                    result = await generate_validated(
                        self.llm, schema_name=schema_name, system=full_system, user=user,
                        purpose="answer", max_retries=self.policy.schema_.max_retries,
                        max_tokens=min(512, max_tokens), max_tokens_cap=max_tokens,
                    )  # fmt: skip
                except SchemaValidationFailed as e:
                    return schema_verdict(e).model_copy(update={"guard": "model"})
                c.output = result.value.model_dump_json()
                return schema_verdict(result).model_copy(update={"guard": "model", "detail": {}})
            r = await self.llm.converse(
                system=full_system, messages=[Message("user", user)], temperature=0.0,
                max_tokens=max_tokens, purpose="answer",
            )  # fmt: skip
            c.output = r.text
            return Verdict(
                guard="model", action=Action.ALLOW, cost_usd=r.cost_usd,
                tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                scores={"truncated": float(r.stop_reason == "max_tokens")},
            )  # fmt: skip

        key = "schema" if schema_name else "model"  # the schema budget covers every retry
        v = await self._step(run, _Step("model", call), ctx, self.policy.timeout_for(key))
        return v.action is not Action.BLOCK and ctx.output is not None

    # ------------------------------------------------------------------ output stage

    async def _output_stage(self, run: _Run, ctx: GuardContext) -> tuple[str | None, Any]:
        """Returns (final_text, data); final_text is None when blocked."""
        run.stage = "output"
        data = None
        if ctx.schema_name:
            v = await self._guard(run, "schema", ctx)
            if v.action is not Action.ALLOW:  # RETRY included: nobody can retry here
                return None, None
            data = v.detail.get("value")
        elif self.policy.grounding.enabled:
            # Grounding judges prose with citations; a schema output is constrained by its
            # schema instead (SPEC decision 10).
            v = await self._guard(run, "grounding", ctx)
            if v.action is Action.BLOCK:
                return None, None
            if v.modified_text is not None:
                ctx.output = v.modified_text

        v = await self._guard(run, "leak", ctx)
        if v.action is Action.BLOCK:
            return None, None
        if v.modified_text is not None:
            ctx.output = v.modified_text

        final = restore(ctx.output or "", ctx.pii_map, self.policy.pii.restore)
        if ctx.schema_name and final != ctx.output:
            check = validate_output(final, ctx.schema_name)  # redaction/restore changed it
            if not check.ok:
                run.verdicts.append(Verdict(guard="finalize", action=Action.BLOCK,
                                            reasons=[Reason.SCHEMA_INVALID]))  # fmt: skip
                return None, None
            data = check.value
        return final, data

    # ------------------------------------------------------------------ public API

    async def check_input(self, text: str, context: list[SourceDoc] | None = None) -> InputResult:
        run = _Run(uuid.uuid4().hex)
        ctx = self._ctx(run.request_id, text)
        ctx.context_docs = list(context or [])
        try:
            ok, wrapped, docs = await asyncio.wait_for(
                self._input_stage(run, ctx), self.policy.timeout_for("request")
            )
        except TimeoutError:
            run.verdicts.append(_request_timeout())
            ok, wrapped, docs = False, "", []
        report = self._finish(run, "check-input")
        return InputResult(
            request_id=run.request_id, decision=report.decision,
            sanitized_text=ctx.text if ok else None, wrapped_context=wrapped if ok else "",
            docs=docs if ok else [], pii_map=ctx.pii_map if ok else {},
            user_message=run.user_message(), report=report,
        )  # fmt: skip

    async def check_output(
        self,
        output: str,
        *,
        sources: list[SourceDoc] | None = None,
        schema_name: str | None = None,
        canary: str | None = None,
        system_prompt: str | None = None,
        pii_map: dict[str, str] | None = None,
    ) -> OutputResult:
        run = _Run(uuid.uuid4().hex)
        ctx = self._ctx(run.request_id)
        ctx.output = output
        ctx.sources = list(sources or [])
        ctx.schema_name = schema_name
        ctx.canary = canary or ""
        ctx.system_prompt = system_prompt or ""
        ctx.pii_map = dict(pii_map or {})
        try:
            final, data = await asyncio.wait_for(
                self._output_stage(run, ctx), self.policy.timeout_for("request")
            )
        except TimeoutError:
            run.verdicts.append(_request_timeout())
            final, data = None, None
        report = self._finish(run, "check-output")
        allowed = report.decision == "allow"
        return OutputResult(run.request_id, report.decision, final if allowed else None,
                            data if allowed else None, report)  # fmt: skip

    async def run(self, question: str, context: list[SourceDoc] | None = None) -> GuardedAnswer:
        run = _Run(uuid.uuid4().hex)
        ctx = self._ctx(run.request_id, question)
        ctx.context_docs = list(context or [])
        ctx.schema_name = self.schema_name

        async def body() -> tuple[str | None, Any]:
            ok, wrapped, docs = await self._input_stage(run, ctx)
            if not ok:
                return None, None  # blocked on input: the model is never called
            ctx.sources = docs
            run.stage = "output"
            if not await self._model_stage(run, ctx, wrapped):
                return None, None
            return await self._output_stage(run, ctx)

        try:
            answer, data = await asyncio.wait_for(body(), self.policy.timeout_for("request"))
        except TimeoutError:
            run.verdicts.append(_request_timeout())
            answer, data = None, None
        report = self._finish(run, ROUTE)
        allowed = report.decision == "allow"
        return GuardedAnswer(
            request_id=run.request_id, decision=report.decision,
            answer=answer if allowed else None, data=data if allowed else None,
            user_message=run.user_message(), report=report,
        )  # fmt: skip


def _request_timeout() -> Verdict:
    return Verdict(guard="request", action=Action.BLOCK, reasons=[Reason.GUARD_TIMEOUT],
                   error_kind="timeout")  # fmt: skip
