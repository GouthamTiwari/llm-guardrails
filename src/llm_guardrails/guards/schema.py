"""Schema guard: structured output that actually holds (docs/SPEC.md 10).

    ask for JSON -> extract it (fences, stray prose) -> validate with Pydantic
      -> invalid? retry with the exact errors fed back (max N, cooler temperature each time)
      -> still invalid? fail closed with SchemaValidationFailed. Never return unvalidated text.

Two entry points:
- `generate_validated(...)`: when we own the model call (pipeline, judges): retries.
- `SchemaGuard`: when someone hands us an output to check (`/v1/check-output`): one
  validation, and on failure a RETRY verdict carrying the errors so the caller can retry.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from llm_guardrails.llm import LLMClient, Message
from llm_guardrails.schemas import get_adapter, json_schema
from llm_guardrails.types import (
    Action,
    GuardContext,
    Reason,
    SchemaValidationFailed,
    Verdict,
)

MAX_ERRORS_FED_BACK = 5
MAX_ERROR_CHARS = 500
MAX_PREVIOUS_OUTPUT_CHARS = 1000
TEMPERATURES = (0.3, 0.1, 0.0)  # attempt 1, 2, 3+: cooler each time, more deterministic
TOKEN_GROWTH = 1.5  # a truncated answer gets 50% more room next time

ErrorKind = Literal[
    "no_json", "multiple_objects", "truncated", "invalid_json", "duplicate_key",
    "extra_key", "wrong_enum", "wrong_type", "missing_field", "constraint",
]  # fmt: skip

_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)


# ----------------------------------------------------------------------------- extraction


@dataclass(frozen=True)
class Extracted:
    json_text: str | None
    error: str | None = None
    kind: ErrorKind | None = None
    prose_stripped: bool = False


def _top_level_objects(text: str) -> tuple[list[tuple[int, int]], bool]:
    """Spans of complete top-level {...} objects, ignoring braces inside strings.
    The flag says an object was opened but never closed (truncated output)."""
    spans: list[tuple[int, int]] = []
    depth, start, in_str, escaped = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans, depth > 0


def extract_json(text: str) -> Extracted:
    """Find exactly one JSON object in a model reply (spec 10.2)."""
    fence = _FENCE.search(text)
    body = fence.group(1) if fence else text
    spans, unterminated = _top_level_objects(body)
    if not spans:
        if unterminated or (fence is None and "```" in text):
            return Extracted(None, "output ends before the JSON object is closed", "truncated")
        return Extracted(None, "no JSON object found", "no_json")
    if len(spans) > 1:
        # Guessing which object was meant could pick an injected one: refuse instead.
        return Extracted(None, "multiple JSON objects; reply with exactly one", "multiple_objects")
    if unterminated:
        return Extracted(None, "a second JSON object is not closed", "truncated")
    start, end = spans[0]
    stripped = bool(fence) or bool(body[:start].strip() or body[end:].strip())
    return Extracted(body[start:end], prose_stripped=stripped)


class DuplicateKeyError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            # json.loads would silently keep the last value, which can hide an override.
            raise DuplicateKeyError(f"duplicate key {key!r}")
        out[key] = value
    return out


# ----------------------------------------------------------------------------- validation

_ERROR_KIND_BY_TYPE: dict[str, ErrorKind] = {
    "extra_forbidden": "extra_key",
    "literal_error": "wrong_enum",
    "union_tag_invalid": "wrong_enum",
    "missing": "missing_field",
    "union_tag_not_found": "missing_field",
}


def _kind_of(error_type: str) -> ErrorKind:
    if error_type in _ERROR_KIND_BY_TYPE:
        return _ERROR_KIND_BY_TYPE[error_type]
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "wrong_type"
    return "constraint"


@dataclass
class Validation:
    value: BaseModel | None
    errors: list[str] = field(default_factory=list)
    kinds: list[ErrorKind] = field(default_factory=list)
    prose_stripped: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None


def format_errors(errors: Sequence[str]) -> str:
    text = "; ".join(errors[:MAX_ERRORS_FED_BACK])
    return text[:MAX_ERROR_CHARS]


def validate_output(text: str, schema_name: str, stop_reason: str | None = None) -> Validation:
    """Extract and validate. Error messages name fields and rules, never echo input values."""
    extracted = extract_json(text)
    if extracted.json_text is None:
        kind = extracted.kind or "no_json"
        if stop_reason == "max_tokens":
            kind = "truncated"
        return Validation(None, [extracted.error or "invalid output"], [kind])
    try:
        data = json.loads(extracted.json_text, object_pairs_hook=_no_duplicates)
    except DuplicateKeyError as exc:
        return Validation(None, [str(exc)], ["duplicate_key"], extracted.prose_stripped)
    except json.JSONDecodeError as exc:
        kind: ErrorKind = "truncated" if stop_reason == "max_tokens" else "invalid_json"
        return Validation(None, [f"not valid JSON: {exc.msg}"], [kind], extracted.prose_stripped)
    try:
        value = get_adapter(schema_name).validate_python(data)
    except ValidationError as exc:
        errors, kinds = [], []
        for err in exc.errors(include_input=False, include_url=False):
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            errors.append(f"{loc}: {err['msg']}")
            kinds.append(_kind_of(err["type"]))
        return Validation(None, errors, kinds, extracted.prose_stripped)
    return Validation(value, prose_stripped=extracted.prose_stripped)


# ----------------------------------------------------------------------------- retry loop


@dataclass(frozen=True)
class Attempt:
    number: int
    temperature: float
    max_tokens: int
    valid: bool
    kinds: tuple[str, ...]
    stop_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float


@dataclass
class SchemaResult:
    value: BaseModel
    attempts: list[Attempt]
    prose_stripped: bool

    @property
    def retries(self) -> int:
        return len(self.attempts) - 1

    @property
    def tokens_in(self) -> int:
        return sum(a.tokens_in for a in self.attempts)

    @property
    def tokens_out(self) -> int:
        return sum(a.tokens_out for a in self.attempts)

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def latency_ms(self) -> float:
        return sum(a.latency_ms for a in self.attempts)


def retry_message(errors: Sequence[str]) -> str:
    return (
        f"Your JSON was invalid: {format_errors(errors)}. "
        "Reply with only the corrected JSON object."
    )


async def generate_validated(
    llm: LLMClient,
    *,
    schema_name: str,
    system: str,
    user: str,
    purpose: str,
    max_retries: int = 2,
    max_tokens: int = 512,
    max_tokens_cap: int = 1024,
    temperatures: Sequence[float] = TEMPERATURES,
) -> SchemaResult:
    """Call the model until its output validates, at most `max_retries + 1` times.

    Raises SchemaValidationFailed (fail closed) when every attempt is invalid. Model or
    network errors propagate unchanged; the guard runner turns them into a block.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    messages: list[Message] = [Message("user", user)]
    attempts: list[Attempt] = []
    last_errors: list[str] = []
    tokens = max_tokens
    for n in range(1, max_retries + 2):  # bounded: there is no other loop
        temperature = temperatures[min(n - 1, len(temperatures) - 1)]
        start = time.perf_counter()
        result = await llm.converse(
            system=system, messages=messages, temperature=temperature, max_tokens=tokens,
            purpose=purpose,
        )  # fmt: skip
        latency = result.latency_ms or (time.perf_counter() - start) * 1000
        check = validate_output(result.text, schema_name, result.stop_reason)
        attempts.append(
            Attempt(
                number=n, temperature=temperature, max_tokens=tokens, valid=check.ok,
                kinds=tuple(check.kinds), stop_reason=result.stop_reason,
                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                cost_usd=result.cost_usd, latency_ms=latency,
            )
        )  # fmt: skip
        if check.ok:
            assert check.value is not None
            return SchemaResult(check.value, attempts, check.prose_stripped)
        last_errors = check.errors
        if "truncated" in check.kinds:
            tokens = min(max_tokens_cap, int(tokens * TOKEN_GROWTH))
        previous = result.text[:MAX_PREVIOUS_OUTPUT_CHARS] or "(empty response)"
        messages += [Message("assistant", previous), Message("user", retry_message(check.errors))]
    raise SchemaValidationFailed(len(attempts), last_errors, attempts)


def schema_verdict(outcome: SchemaResult | SchemaValidationFailed) -> Verdict:
    """Verdict for a model call the pipeline made through `generate_validated`."""
    if isinstance(outcome, SchemaResult):
        a = outcome.attempts
        return Verdict(
            guard="schema",
            action=Action.ALLOW,
            scores={"attempts": float(len(a)), "retries": float(outcome.retries)},
            cost_usd=outcome.cost_usd,
            tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            detail={"value": outcome.value, "attempts": a},
        )
    trace = outcome.trace
    return Verdict(
        guard="schema",
        action=Action.BLOCK,
        reasons=[Reason.SCHEMA_INVALID_AFTER_RETRIES],
        scores={"attempts": float(outcome.attempts)},
        cost_usd=sum(getattr(t, "cost_usd", 0.0) for t in trace),
        tokens_in=sum(getattr(t, "tokens_in", 0) for t in trace),
        tokens_out=sum(getattr(t, "tokens_out", 0) for t in trace),
        detail={"errors": outcome.errors, "attempts": trace},
    )


# ----------------------------------------------------------------------------- prompts

PromptStyle = Literal["plain", "schema", "schema_retry"]


def _template(name: str) -> str:
    return resources.files("llm_guardrails.prompts").joinpath(name).read_text(encoding="utf-8")


def ticket_prompt(style: PromptStyle, ticket_text: str) -> tuple[str, str]:
    """(system, user) for the TicketSummary experiment (spec 10.4)."""
    if style == "plain":
        system = _template("ticket_plain.txt")
    else:
        schema = json.dumps(json_schema("TicketSummary"), indent=1)
        system = _template("ticket_schema.txt").replace("{schema}", schema)
    # The ticket is data inside fixed markers; marker look-alikes in it are defused.
    safe = ticket_text.replace("<<<", "‹‹‹").replace(">>>", "›››")
    user = f"Support ticket:\n<<<TICKET>>>\n{safe}\n<<<END TICKET>>>"
    return system, user


# ----------------------------------------------------------------------------- guard


class SchemaGuard:
    """Checks an output someone else produced. It can't call the model again, so an invalid
    output gets a RETRY verdict with the errors; the pipeline never treats RETRY as allow."""

    name = "schema"
    security = True

    def __init__(self, schema_name: str | None = None) -> None:
        self.schema_name = schema_name

    def _run(self, ctx: GuardContext) -> Verdict:
        schema_name = ctx.schema_name or self.schema_name
        if schema_name is None:
            return Verdict(guard=self.name, action=Action.ALLOW)
        check = validate_output(ctx.output or "", schema_name)
        if check.ok:
            return Verdict(
                guard=self.name,
                action=Action.ALLOW,
                detail={"value": check.value, "prose_stripped": check.prose_stripped},
            )
        reason = Reason.SCHEMA_TRUNCATED if "truncated" in check.kinds else Reason.SCHEMA_INVALID
        return Verdict(
            guard=self.name,
            action=Action.RETRY,
            reasons=[reason],
            detail={"errors": check.errors, "kinds": check.kinds,
                    "retry_message": retry_message(check.errors)},
        )  # fmt: skip

    async def check(self, ctx: GuardContext) -> Verdict:
        return await asyncio.to_thread(self._run, ctx)
